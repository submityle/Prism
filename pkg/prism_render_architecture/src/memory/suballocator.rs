//! Generic offset suballocator for a single GPU heap region.
//!
//! A backend commits one large device allocation per [`HeapClass`](super::HeapClass)
//! and then carves resource-sized, alignment-respecting ranges out of it on the
//! `CPU` without ever touching the `GPU`. This module owns that carving: it is a
//! deterministic free-list allocator that hands back byte offsets, folds freed
//! ranges back together, and reports the fragmentation statistics the heap
//! contract needs. It holds no `GPU` handles and performs only integer math, so
//! its behaviour is fully reproducible across runs and platforms.
//!
//! The real device memory object is bound by the backend and is *pending the GPU
//! backend*; this layer only decides *where* inside that object each resource
//! lives.

use alloc::collections::BTreeMap;
use core::fmt;

use super::HeapStats;

/// Rounds `value` up to the next multiple of a power-of-two `align`.
///
/// Returns `None` when the rounding would overflow `u64`, so callers surface an
/// allocation failure instead of wrapping. `align` must be a power of two; that
/// precondition is validated by every public entry point before this is called.
fn align_up(value: u64, align: u64) -> Option<u64> {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let mask = align - 1;
    value.checked_add(mask).map(|rounded| rounded & !mask)
}

/// Strategy used to choose which free block satisfies an allocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FitStrategy {
    /// Take the lowest-offset block that fits. Cheapest and keeps allocations
    /// packed toward the start of the heap.
    #[default]
    FirstFit,
    /// Take the block that leaves the least free space behind. Minimizes
    /// external fragmentation at the cost of scanning every free block.
    BestFit,
}

/// Reason an [`SubAllocator::allocate`] request could not be satisfied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubAllocError {
    /// A zero-byte allocation was requested; callers must ask for at least one
    /// byte so every allocation has a distinct offset.
    ZeroSize,
    /// The requested alignment was not a non-zero power of two.
    InvalidAlignment(u64),
    /// No free block could host `requested` bytes at the requested alignment.
    OutOfMemory {
        /// Bytes the caller asked for.
        requested: u64,
        /// Alignment the caller asked for.
        alignment: u64,
        /// Largest contiguous free block currently available.
        largest_free_block: u64,
    },
}

impl fmt::Display for SubAllocError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSize => formatter.write_str("cannot suballocate zero bytes"),
            Self::InvalidAlignment(alignment) => {
                write!(formatter, "alignment {alignment} is not a power of two")
            }
            Self::OutOfMemory {
                requested,
                alignment,
                largest_free_block,
            } => write!(
                formatter,
                "out of heap memory: requested {requested} bytes aligned to {alignment}, \
                 largest free block is {largest_free_block}"
            ),
        }
    }
}

impl std::error::Error for SubAllocError {}

/// Reason an [`SubAllocator::free`] call was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubFreeError {
    /// No live allocation starts at the given offset (double free or bad
    /// offset).
    UnknownOffset(u64),
}

impl fmt::Display for SubFreeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOffset(offset) => {
                write!(formatter, "no live allocation starts at offset {offset}")
            }
        }
    }
}

impl std::error::Error for SubFreeError {}

/// A byte range carved out of the heap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubAllocation {
    /// Aligned start offset within the heap region.
    pub offset: u64,
    /// Length of the allocation in bytes.
    pub size: u64,
}

impl SubAllocation {
    /// One-past-the-end offset of this allocation.
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.offset + self.size
    }
}

/// Deterministic free-list suballocator over a fixed-size heap region.
///
/// Free space is tracked as a set of non-overlapping `[offset, size)` blocks
/// keyed by offset, so neighbouring blocks are found in `O(log n)` and merged on
/// free. Allocated ranges are tracked separately so a stray offset cannot be
/// freed twice.
#[derive(Clone, Debug)]
pub struct SubAllocator {
    capacity: u64,
    strategy: FitStrategy,
    free: BTreeMap<u64, u64>,
    allocated: BTreeMap<u64, u64>,
    used_bytes: u64,
}

impl SubAllocator {
    /// Creates an allocator that owns `[0, capacity)` using first-fit.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        Self::with_strategy(capacity, FitStrategy::FirstFit)
    }

    /// Creates an allocator owning `[0, capacity)` with an explicit strategy.
    #[must_use]
    pub fn with_strategy(capacity: u64, strategy: FitStrategy) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self {
            capacity,
            strategy,
            free,
            allocated: BTreeMap::new(),
            used_bytes: 0,
        }
    }

    /// Total committed capacity of the heap region.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes currently handed out to live allocations.
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Bytes not currently allocated (`capacity - used`).
    #[must_use]
    pub const fn free_bytes(&self) -> u64 {
        self.capacity - self.used_bytes
    }

    /// Fit strategy in effect.
    #[must_use]
    pub const fn strategy(&self) -> FitStrategy {
        self.strategy
    }

    /// Number of live allocations.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.allocated.len()
    }

    /// Number of distinct free blocks; a proxy for external fragmentation.
    #[must_use]
    pub fn free_block_count(&self) -> usize {
        self.free.len()
    }

    /// Largest contiguous free block, i.e. the biggest single allocation that
    /// could currently succeed at alignment one.
    #[must_use]
    pub fn largest_free_block(&self) -> u64 {
        self.free.values().copied().max().unwrap_or(0)
    }

    /// External fragmentation in parts-per-thousand.
    ///
    /// `0` means all free space is one block; `1000` means the free space is
    /// maximally shattered. Computed as `(free - largest_free) / free` so it is
    /// independent of how much of the heap is in use. Uses 128-bit intermediate
    /// math to stay exact for full 64-bit heaps.
    #[must_use]
    pub fn fragmentation_permille(&self) -> u32 {
        let free = self.free_bytes();
        if free == 0 {
            return 0;
        }
        let scattered = free - self.largest_free_block();
        let permille = u128::from(scattered) * 1000 / u128::from(free);
        permille as u32
    }

    /// Snapshot of this region as a [`HeapStats`] contract value.
    #[must_use]
    pub fn heap_stats(&self) -> HeapStats {
        HeapStats {
            committed_bytes: self.capacity,
            used_bytes: self.used_bytes,
            largest_free_block: self.largest_free_block(),
        }
    }

    /// Carves `size` bytes aligned to `alignment` out of the heap.
    ///
    /// Returns the chosen [`SubAllocation`] or a [`SubAllocError`] describing why
    /// the request could not be honoured. Never panics on a full or fragmented
    /// heap.
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Result<SubAllocation, SubAllocError> {
        if size == 0 {
            return Err(SubAllocError::ZeroSize);
        }
        if !alignment.is_power_of_two() {
            return Err(SubAllocError::InvalidAlignment(alignment));
        }

        let choice = self.pick_block(size, alignment);
        let Some((block_offset, block_size, aligned)) = choice else {
            return Err(SubAllocError::OutOfMemory {
                requested: size,
                alignment,
                largest_free_block: self.largest_free_block(),
            });
        };

        // The block is consumed and its untouched head/tail returned to the pool.
        self.free.remove(&block_offset);
        let block_end = block_offset + block_size;
        let alloc_end = aligned + size;
        if aligned > block_offset {
            self.free.insert(block_offset, aligned - block_offset);
        }
        if block_end > alloc_end {
            self.free.insert(alloc_end, block_end - alloc_end);
        }

        self.allocated.insert(aligned, size);
        self.used_bytes += size;
        Ok(SubAllocation {
            offset: aligned,
            size,
        })
    }

    /// Returns a live allocation's bytes to the pool, merging with neighbours.
    pub fn free(&mut self, offset: u64) -> Result<(), SubFreeError> {
        let size = self
            .allocated
            .remove(&offset)
            .ok_or(SubFreeError::UnknownOffset(offset))?;
        self.used_bytes -= size;
        self.insert_free_coalesced(offset, size);
        Ok(())
    }

    /// Drops every allocation and returns the heap to a single free block.
    pub fn reset(&mut self) {
        self.allocated.clear();
        self.free.clear();
        if self.capacity > 0 {
            self.free.insert(0, self.capacity);
        }
        self.used_bytes = 0;
    }

    /// Scans the free set for a block that can host the request, honouring the
    /// configured [`FitStrategy`]. Returns the block key, its size, and the
    /// aligned offset the allocation would take within it.
    fn pick_block(&self, size: u64, alignment: u64) -> Option<(u64, u64, u64)> {
        let mut best: Option<(u64, u64, u64, u64)> = None;
        for (&block_offset, &block_size) in &self.free {
            let Some(aligned) = align_up(block_offset, alignment) else {
                continue;
            };
            let Some(alloc_end) = aligned.checked_add(size) else {
                continue;
            };
            let block_end = block_offset + block_size;
            if alloc_end > block_end {
                continue;
            }
            // Free bytes this block would still hold after the carve.
            let leftover = (aligned - block_offset) + (block_end - alloc_end);
            match self.strategy {
                FitStrategy::FirstFit => {
                    return Some((block_offset, block_size, aligned));
                }
                FitStrategy::BestFit => {
                    let better = match best {
                        Some((_, _, _, best_leftover)) => leftover < best_leftover,
                        None => true,
                    };
                    if better {
                        best = Some((block_offset, block_size, aligned, leftover));
                    }
                }
            }
        }
        best.map(|(offset, block_size, aligned, _)| (offset, block_size, aligned))
    }

    /// Inserts a freed `[offset, size)` range, coalescing an immediately
    /// preceding and/or following free block into one entry.
    fn insert_free_coalesced(&mut self, offset: u64, size: u64) {
        let mut start = offset;
        let mut end = offset + size;

        if let Some((&prev_offset, &prev_size)) = self.free.range(..offset).next_back()
            && prev_offset + prev_size == start
        {
            start = prev_offset;
            self.free.remove(&prev_offset);
        }
        if let Some(&next_size) = self.free.get(&end) {
            self.free.remove(&end);
            end += next_size;
        }

        self.free.insert(start, end - start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_sequentially_with_alignment() {
        let mut allocator = SubAllocator::new(1024);
        let a = allocator.allocate(100, 256).unwrap();
        let b = allocator.allocate(50, 256).unwrap();
        assert_eq!(a.offset, 0);
        // b is padded up to the next 256 boundary after a's 100 bytes.
        assert_eq!(b.offset, 256);
        assert_eq!(allocator.used_bytes(), 150);
    }

    #[test]
    fn rejects_zero_size_and_bad_alignment() {
        let mut allocator = SubAllocator::new(1024);
        assert_eq!(allocator.allocate(0, 16), Err(SubAllocError::ZeroSize));
        assert_eq!(
            allocator.allocate(16, 24),
            Err(SubAllocError::InvalidAlignment(24))
        );
        assert_eq!(
            allocator.allocate(16, 0),
            Err(SubAllocError::InvalidAlignment(0))
        );
    }

    #[test]
    fn full_heap_reports_out_of_memory_without_panicking() {
        let mut allocator = SubAllocator::new(256);
        let _ = allocator.allocate(256, 1).unwrap();
        let err = allocator.allocate(1, 1).unwrap_err();
        assert_eq!(
            err,
            SubAllocError::OutOfMemory {
                requested: 1,
                alignment: 1,
                largest_free_block: 0,
            }
        );
    }

    #[test]
    fn alignment_padding_can_exhaust_a_block() {
        let mut allocator = SubAllocator::new(300);
        // Occupy [0,10); the remaining block starts at 10.
        let _ = allocator.allocate(10, 1).unwrap();
        // Aligning to 512 would jump past capacity, so this must fail cleanly.
        assert!(matches!(
            allocator.allocate(8, 512),
            Err(SubAllocError::OutOfMemory { .. })
        ));
    }

    #[test]
    fn free_coalesces_adjacent_blocks() {
        let mut allocator = SubAllocator::new(300);
        let a = allocator.allocate(100, 1).unwrap();
        let b = allocator.allocate(100, 1).unwrap();
        let c = allocator.allocate(100, 1).unwrap();
        assert_eq!(allocator.free_block_count(), 0);

        allocator.free(a.offset).unwrap();
        allocator.free(c.offset).unwrap();
        // Two disjoint holes on either side of b.
        assert_eq!(allocator.free_block_count(), 2);

        allocator.free(b.offset).unwrap();
        // Freeing the middle merges all three into one full-heap block.
        assert_eq!(allocator.free_block_count(), 1);
        assert_eq!(allocator.largest_free_block(), 300);
        assert_eq!(allocator.used_bytes(), 0);
    }

    #[test]
    fn double_free_is_rejected() {
        let mut allocator = SubAllocator::new(128);
        let a = allocator.allocate(64, 1).unwrap();
        allocator.free(a.offset).unwrap();
        assert_eq!(
            allocator.free(a.offset),
            Err(SubFreeError::UnknownOffset(a.offset))
        );
    }

    #[test]
    fn best_fit_prefers_the_tightest_block() {
        let mut allocator = SubAllocator::with_strategy(1000, FitStrategy::BestFit);
        let a = allocator.allocate(100, 1).unwrap(); // [0,100)
        let b = allocator.allocate(200, 1).unwrap(); // [100,300)
        let c = allocator.allocate(100, 1).unwrap(); // [300,400)
        let _tail = allocator.allocate(600, 1).unwrap(); // [400,1000)

        // Free the 100 and 200 holes; best-fit for 90 must take the 100 hole.
        allocator.free(a.offset).unwrap();
        allocator.free(b.offset).unwrap();
        let _ = c;
        let fit = allocator.allocate(90, 1).unwrap();
        assert_eq!(fit.offset, 0);
        assert_eq!(fit.size, 90);
    }

    #[test]
    fn first_fit_takes_the_lowest_block() {
        let mut allocator = SubAllocator::with_strategy(1000, FitStrategy::FirstFit);
        let a = allocator.allocate(100, 1).unwrap();
        let b = allocator.allocate(200, 1).unwrap();
        allocator.free(a.offset).unwrap();
        allocator.free(b.offset).unwrap();
        // Lowest-offset block wins even though the 200 hole also fits.
        let fit = allocator.allocate(90, 1).unwrap();
        assert_eq!(fit.offset, 0);
    }

    #[test]
    fn fragmentation_permille_tracks_scatter() {
        let mut allocator = SubAllocator::new(400);
        let a = allocator.allocate(100, 1).unwrap();
        let b = allocator.allocate(100, 1).unwrap();
        let c = allocator.allocate(100, 1).unwrap();
        let _d = allocator.allocate(100, 1).unwrap();
        // Free two non-adjacent blocks: 200 free bytes in two 100 blocks.
        allocator.free(a.offset).unwrap();
        allocator.free(c.offset).unwrap();
        let _ = b;
        assert_eq!(allocator.free_bytes(), 200);
        assert_eq!(allocator.largest_free_block(), 100);
        // (200 - 100) / 200 = 500 permille.
        assert_eq!(allocator.fragmentation_permille(), 500);
    }

    #[test]
    fn reset_restores_full_capacity() {
        let mut allocator = SubAllocator::new(512);
        let _ = allocator.allocate(128, 1).unwrap();
        let _ = allocator.allocate(128, 1).unwrap();
        allocator.reset();
        assert_eq!(allocator.used_bytes(), 0);
        assert_eq!(allocator.largest_free_block(), 512);
        assert_eq!(allocator.allocation_count(), 0);
    }

    #[test]
    fn heap_stats_reports_committed_used_and_largest() {
        let mut allocator = SubAllocator::new(1024);
        let _ = allocator.allocate(256, 1).unwrap();
        let stats = allocator.heap_stats();
        assert_eq!(stats.committed_bytes, 1024);
        assert_eq!(stats.used_bytes, 256);
        assert_eq!(stats.largest_free_block, 768);
    }

    #[test]
    fn allocation_is_deterministic_across_identical_sequences() {
        fn run() -> Vec<u64> {
            let mut allocator = SubAllocator::with_strategy(4096, FitStrategy::BestFit);
            let mut offsets = Vec::new();
            let a = allocator.allocate(64, 64).unwrap();
            let b = allocator.allocate(128, 128).unwrap();
            allocator.free(a.offset).unwrap();
            let c = allocator.allocate(32, 32).unwrap();
            let d = allocator.allocate(256, 256).unwrap();
            offsets.push(a.offset);
            offsets.push(b.offset);
            offsets.push(c.offset);
            offsets.push(d.offset);
            offsets
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn zero_capacity_heap_allocates_nothing() {
        let mut allocator = SubAllocator::new(0);
        assert_eq!(allocator.largest_free_block(), 0);
        assert!(matches!(
            allocator.allocate(1, 1),
            Err(SubAllocError::OutOfMemory { .. })
        ));
    }
}
