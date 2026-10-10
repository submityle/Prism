//! GPU memory sub-allocators: carve sub-ranges out of a fixed-size heap.
//!
//! These are *offset* allocators. They manage ranges only and never touch real
//! device memory: an allocation is a `(offset, size)` [`Region`] into some heap
//! the caller owns. Nothing here reads or writes bytes, so there is no
//! `unsafe`, and every operation is fully deterministic.
//!
//! Four strategies are provided, each suited to a different allocation pattern:
//! - [`TlsfAllocator`]: a general-purpose Two-Level Segregated Fit allocator
//!   with near-`O(1)` allocate/free and immediate coalescing.
//! - [`BuddyAllocator`]: a power-of-two buddy allocator with `O(log n)`
//!   split/merge, good for alignment-heavy, power-of-two workloads.
//! - [`LinearAllocator`]: a monotonic bump allocator freed all at once via
//!   [`LinearAllocator::reset`].
//! - [`RingAllocator`]: a circular per-frame upload heap with wraparound and
//!   in-flight protection driven by [`RingAllocator::begin_frame`] /
//!   [`RingAllocator::retire`].
//!
//! Every `allocate` respects a power-of-two alignment, uses checked arithmetic,
//! and returns `None` on exhaustion rather than panicking or overflowing.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// A contiguous sub-range of a heap: a byte `offset` and a byte `size`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Region {
    /// The start of the range, in bytes from the heap base.
    pub offset: u64,
    /// The length of the range, in bytes.
    pub size: u64,
}

impl Region {
    /// The one-past-the-end byte offset of the range, or `None` on overflow.
    #[must_use]
    pub fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.size)
    }
}

/// An opaque handle returned by an allocator's `allocate` and consumed by its
/// `free`. It carries the public [`Region`] plus whatever private bookkeeping
/// the owning allocator needs to release and coalesce the range.
///
/// Hand an [`Allocation`] back only to the same allocator that produced it. It
/// is intentionally neither `Copy` nor `Clone` so the type system discourages
/// double frees.
#[derive(Debug, PartialEq, Eq)]
pub struct Allocation {
    region: Region,
    /// Allocator-private metadata: a block-pool slot index for
    /// [`TlsfAllocator`], a size-class order for [`BuddyAllocator`], and unused
    /// (zero) for the linear and ring allocators.
    handle: usize,
}

impl Allocation {
    /// The allocated [`Region`].
    #[must_use]
    pub fn region(&self) -> Region {
        self.region
    }

    /// The start offset of the allocation, in bytes.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.region.offset
    }

    /// The size of the allocation, in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.region.size
    }
}

/// Rounds `value` up to a multiple of `align`, which must be a power of two.
/// Returns `None` if the rounded value would overflow `u64`.
fn align_up(value: u64, align: u64) -> Option<u64> {
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

/// Normalizes a requested alignment: `0` becomes `1`, and anything that is not
/// a power of two is rejected with `None`.
fn normalize_align(align: u64) -> Option<u64> {
    let align = if align == 0 { 1 } else { align };
    if align.is_power_of_two() {
        Some(align)
    } else {
        None
    }
}

// ============================================================================
// TLSF
// ============================================================================

/// Log2 of the number of second-level lists per first-level class.
const SL_LOG2: u32 = 4;
/// The number of second-level lists per first-level class.
const SL_COUNT: usize = 1 << SL_LOG2;
/// Log2 of the smallest distinguishable block size, i.e. the allocator's
/// internal granularity.
const FL_SHIFT: u32 = SL_LOG2;
/// The smallest block size that is mapped by its most-significant bit.
const SMALL_BLOCK: u64 = 1 << FL_SHIFT;
/// The number of first-level size classes (covers sizes up to `u64::MAX`).
const FL_COUNT: usize = 64 - FL_SHIFT as usize;

/// A single managed block in the [`TlsfAllocator`] pool.
///
/// Blocks form two intrusive doubly linked lists by slot index: a *physical*
/// list ordered by address (used for coalescing) and a *free* list within the
/// block's `(first-level, second-level)` size class.
#[derive(Clone, Copy)]
struct Block {
    offset: u64,
    size: u64,
    free: bool,
    prev_phys: Option<usize>,
    next_phys: Option<usize>,
    prev_free: Option<usize>,
    next_free: Option<usize>,
}

/// A Two-Level Segregated Fit allocator.
///
/// Free blocks are bucketed into size classes selected by the most-significant
/// bit of their size (the first level) and a linear subdivision of the next
/// `SL_LOG2` bits (the second level). First- and second-level bitmaps make
/// "find the smallest class that fits" a couple of bit scans, giving near-`O(1)`
/// [`TlsfAllocator::allocate`] and [`TlsfAllocator::free`]. Freed blocks
/// immediately coalesce with physically adjacent free neighbors.
pub struct TlsfAllocator {
    capacity: u64,
    allocated: u64,
    blocks: Vec<Block>,
    free_slots: Vec<usize>,
    fl_bitmap: u64,
    sl_bitmap: [u16; FL_COUNT],
    heads: [[Option<usize>; SL_COUNT]; FL_COUNT],
}

impl TlsfAllocator {
    /// Creates an allocator managing a single heap of `size` bytes.
    #[must_use]
    pub fn new(size: u64) -> Self {
        let mut this = Self {
            capacity: size,
            allocated: 0,
            blocks: Vec::new(),
            free_slots: Vec::new(),
            fl_bitmap: 0,
            sl_bitmap: [0; FL_COUNT],
            heads: [[None; SL_COUNT]; FL_COUNT],
        };
        if size > 0 {
            let idx = this.alloc_slot(Block {
                offset: 0,
                size,
                free: false,
                prev_phys: None,
                next_phys: None,
                prev_free: None,
                next_free: None,
            });
            this.insert_free(idx);
        }
        this
    }

    /// The total heap size, in bytes.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The number of bytes currently handed out to live allocations.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// A fragmentation metric in `[0, 1]`, defined as
    /// `1 - largest_free_block / total_free`. It is `0` when there is no free
    /// space (or all free space is a single block) and approaches `1` as free
    /// space is scattered across many small blocks.
    #[must_use]
    pub fn fragmentation(&self) -> f32 {
        let total_free = self.capacity - self.allocated;
        if total_free == 0 {
            return 0.0;
        }
        let mut largest = 0u64;
        for block in &self.blocks {
            if block.free && block.size > largest {
                largest = block.size;
            }
        }
        1.0 - (largest as f32 / total_free as f32)
    }

    /// Allocates `size` bytes aligned to `align` (which must be a power of two,
    /// where `0` is treated as `1`). Returns `None` if `align` is invalid, the
    /// request is empty, or the heap cannot satisfy it.
    #[must_use]
    pub fn allocate(&mut self, size: u64, align: u64) -> Option<Allocation> {
        if size == 0 {
            return None;
        }
        let align = normalize_align(align)?;

        // Reserve enough slack that any block in the chosen class can host an
        // aligned sub-range of `size` bytes.
        let search_size = size.checked_add(align - 1)?;
        let (fl, sl) = Self::mapping_search(search_size)?;
        let block_idx = self.find_free_block(fl, sl)?;
        self.remove_free(block_idx);

        let base = self.blocks[block_idx].offset;
        let base_size = self.blocks[block_idx].size;
        let aligned = align_up(base, align)?;
        let front_pad = aligned - base;

        // Split off and return any front padding as a free block.
        let working = if front_pad > 0 {
            let rest = self.split(block_idx, front_pad);
            self.insert_free(block_idx);
            rest
        } else {
            block_idx
        };

        // Split off and return any trailing remainder as a free block.
        let working_size = base_size - front_pad;
        if working_size > size {
            let trailing = self.split(working, size);
            self.insert_free(trailing);
        }

        self.blocks[working].free = false;
        self.allocated += size;
        Some(Allocation {
            region: Region {
                offset: aligned,
                size,
            },
            handle: working,
        })
    }

    /// Releases a previous allocation, coalescing it with any physically
    /// adjacent free blocks so that fully freeing the heap restores a single
    /// capacity-sized free block.
    pub fn free(&mut self, allocation: Allocation) {
        let mut idx = allocation.handle;
        self.allocated -= self.blocks[idx].size;

        // Coalesce with the physical successor if it is free.
        if let Some(next) = self.blocks[idx].next_phys
            && self.blocks[next].free
        {
            self.remove_free(next);
            self.blocks[idx].size += self.blocks[next].size;
            let after = self.blocks[next].next_phys;
            self.blocks[idx].next_phys = after;
            if let Some(a) = after {
                self.blocks[a].prev_phys = Some(idx);
            }
            self.free_slot(next);
        }

        // Coalesce with the physical predecessor if it is free.
        if let Some(prev) = self.blocks[idx].prev_phys
            && self.blocks[prev].free
        {
            self.remove_free(prev);
            self.blocks[prev].size += self.blocks[idx].size;
            let after = self.blocks[idx].next_phys;
            self.blocks[prev].next_phys = after;
            if let Some(a) = after {
                self.blocks[a].prev_phys = Some(prev);
            }
            self.free_slot(idx);
            idx = prev;
        }

        self.insert_free(idx);
    }

    /// Allocates a pool slot for `block`, recycling a freed slot when possible.
    fn alloc_slot(&mut self, block: Block) -> usize {
        if let Some(idx) = self.free_slots.pop() {
            self.blocks[idx] = block;
            idx
        } else {
            self.blocks.push(block);
            self.blocks.len() - 1
        }
    }

    /// Returns a pool slot to the recycling list.
    fn free_slot(&mut self, idx: usize) {
        self.free_slots.push(idx);
    }

    /// Splits block `idx` so it keeps `first_size` bytes and a freshly created
    /// trailing block (returned) covers the remainder. The new block is linked
    /// into the physical list immediately after `idx`.
    fn split(&mut self, idx: usize, first_size: u64) -> usize {
        let offset = self.blocks[idx].offset;
        let total = self.blocks[idx].size;
        let old_next = self.blocks[idx].next_phys;
        let new = self.alloc_slot(Block {
            offset: offset + first_size,
            size: total - first_size,
            free: false,
            prev_phys: Some(idx),
            next_phys: old_next,
            prev_free: None,
            next_free: None,
        });
        if let Some(n) = old_next {
            self.blocks[n].prev_phys = Some(new);
        }
        self.blocks[idx].next_phys = Some(new);
        self.blocks[idx].size = first_size;
        new
    }

    /// Maps a block size to its `(first-level, second-level)` class.
    fn mapping(size: u64) -> (usize, usize) {
        if size < SMALL_BLOCK {
            (0, (size >> (FL_SHIFT - SL_LOG2)) as usize)
        } else {
            let fl = 63 - size.leading_zeros();
            let sl = ((size >> (fl - SL_LOG2)) & (SL_COUNT as u64 - 1)) as usize;
            ((fl - FL_SHIFT) as usize, sl)
        }
    }

    /// Like [`Self::mapping`] but rounds `size` up first, so the resulting class
    /// is guaranteed to only contain blocks large enough to satisfy `size`.
    /// Returns `None` if rounding overflows.
    fn mapping_search(size: u64) -> Option<(usize, usize)> {
        let size = if size >= SMALL_BLOCK {
            let fl = 63 - size.leading_zeros();
            let round = (1u64 << (fl - SL_LOG2)) - 1;
            size.checked_add(round)?
        } else {
            size
        };
        Some(Self::mapping(size))
    }

    /// Scans the bitmaps for the first non-empty free class at or above
    /// `(fl, sl)`, returning the head block of that class if any exists.
    fn find_free_block(&self, fl: usize, sl: usize) -> Option<usize> {
        let sl_map = self.sl_bitmap[fl] & (!0u16 << sl);
        let (fl, sl) = if sl_map != 0 {
            (fl, sl_map.trailing_zeros() as usize)
        } else {
            let fl_map = if fl + 1 >= 64 {
                0
            } else {
                self.fl_bitmap & (!0u64 << (fl + 1))
            };
            if fl_map == 0 {
                return None;
            }
            let fl = fl_map.trailing_zeros() as usize;
            let sl = self.sl_bitmap[fl].trailing_zeros() as usize;
            (fl, sl)
        };
        self.heads[fl][sl]
    }

    /// Marks block `idx` free and links it at the head of its size class,
    /// updating the first- and second-level bitmaps.
    fn insert_free(&mut self, idx: usize) {
        let (fl, sl) = Self::mapping(self.blocks[idx].size);
        let head = self.heads[fl][sl];
        self.blocks[idx].free = true;
        self.blocks[idx].prev_free = None;
        self.blocks[idx].next_free = head;
        if let Some(h) = head {
            self.blocks[h].prev_free = Some(idx);
        }
        self.heads[fl][sl] = Some(idx);
        self.fl_bitmap |= 1u64 << fl;
        self.sl_bitmap[fl] |= 1u16 << sl;
    }

    /// Unlinks block `idx` from its size-class free list, clearing bitmap bits
    /// when a list empties, and marks the block as no longer free.
    fn remove_free(&mut self, idx: usize) {
        let (fl, sl) = Self::mapping(self.blocks[idx].size);
        let prev = self.blocks[idx].prev_free;
        let next = self.blocks[idx].next_free;
        if let Some(p) = prev {
            self.blocks[p].next_free = next;
        } else {
            self.heads[fl][sl] = next;
        }
        if let Some(n) = next {
            self.blocks[n].prev_free = prev;
        }
        self.blocks[idx].prev_free = None;
        self.blocks[idx].next_free = None;
        self.blocks[idx].free = false;
        if self.heads[fl][sl].is_none() {
            self.sl_bitmap[fl] &= !(1u16 << sl);
            if self.sl_bitmap[fl] == 0 {
                self.fl_bitmap &= !(1u64 << fl);
            }
        }
    }
}

// ============================================================================
// Buddy
// ============================================================================

/// A binary buddy allocator over a power-of-two heap.
///
/// The capacity is rounded up to a power of two. Allocations are rounded up to
/// a power of two and served by splitting a larger free block in half
/// repeatedly; freeing merges a block with its address buddy whenever the buddy
/// is also free. Both paths are `O(log n)` in the number of size classes.
pub struct BuddyAllocator {
    capacity: u64,
    allocated: u64,
    /// Free block offsets per order; order `k` holds blocks of size `1 << k`.
    free_lists: Vec<Vec<u64>>,
}

impl BuddyAllocator {
    /// Creates an allocator whose capacity is `size` rounded up to the next
    /// power of two (the leaf granularity is one byte).
    #[must_use]
    pub fn new(size: u64) -> Self {
        if size == 0 {
            return Self {
                capacity: 0,
                allocated: 0,
                free_lists: Vec::new(),
            };
        }
        let capacity = size.next_power_of_two();
        let num_orders = capacity.trailing_zeros() as usize + 1;
        let mut free_lists: Vec<Vec<u64>> = Vec::with_capacity(num_orders);
        for _ in 0..num_orders {
            free_lists.push(Vec::new());
        }
        free_lists[num_orders - 1].push(0);
        Self {
            capacity,
            allocated: 0,
            free_lists,
        }
    }

    /// The power-of-two heap capacity, in bytes.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The number of bytes handed out, counting power-of-two rounding.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Allocates at least `size` bytes, rounded up to a power of two. The
    /// returned region's offset is naturally aligned to its (power-of-two)
    /// size. Returns `None` when the request is empty or cannot be satisfied.
    #[must_use]
    pub fn allocate(&mut self, size: u64) -> Option<Allocation> {
        if self.capacity == 0 || size == 0 || size > self.capacity {
            return None;
        }
        let need = size.next_power_of_two();
        let order = need.trailing_zeros() as usize;
        if order >= self.free_lists.len() {
            return None;
        }

        // Find the smallest order at or above `order` with a free block.
        let mut source = order;
        while source < self.free_lists.len() && self.free_lists[source].is_empty() {
            source += 1;
        }
        if source >= self.free_lists.len() {
            return None;
        }
        let offset = self.free_lists[source].pop()?;

        // Split down, parking the upper buddy at each intermediate order.
        while source > order {
            source -= 1;
            let buddy = offset + (1u64 << source);
            self.free_lists[source].push(buddy);
        }

        self.allocated += 1u64 << order;
        Some(Allocation {
            region: Region {
                offset,
                size: 1u64 << order,
            },
            handle: order,
        })
    }

    /// Releases a previous allocation, merging with the address buddy at each
    /// order while that buddy is free.
    pub fn free(&mut self, allocation: Allocation) {
        let mut order = allocation.handle;
        let mut offset = allocation.region.offset;
        self.allocated -= 1u64 << order;

        while order + 1 < self.free_lists.len() {
            let buddy = offset ^ (1u64 << order);
            if let Some(pos) = self.free_lists[order].iter().position(|&o| o == buddy) {
                self.free_lists[order].swap_remove(pos);
                offset = offset.min(buddy);
                order += 1;
            } else {
                break;
            }
        }
        self.free_lists[order].push(offset);
    }
}

// ============================================================================
// Linear / bump
// ============================================================================

/// A monotonic bump allocator.
///
/// Allocations only move a cursor forward; individual ranges are never freed.
/// Call [`LinearAllocator::reset`] to reclaim the whole heap at once, which is
/// ideal for per-pass scratch memory.
pub struct LinearAllocator {
    capacity: u64,
    head: u64,
}

impl LinearAllocator {
    /// Creates a bump allocator over a heap of `size` bytes.
    #[must_use]
    pub fn new(size: u64) -> Self {
        Self {
            capacity: size,
            head: 0,
        }
    }

    /// The total heap size, in bytes.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The number of bytes consumed so far, including alignment padding.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.head
    }

    /// Allocates `size` bytes aligned to `align` (a power of two, where `0` is
    /// treated as `1`). Returns `None` if `align` is invalid, the request is
    /// empty, or the heap is exhausted.
    #[must_use]
    pub fn allocate(&mut self, size: u64, align: u64) -> Option<Allocation> {
        if size == 0 {
            return None;
        }
        let align = normalize_align(align)?;
        let aligned = align_up(self.head, align)?;
        let end = aligned.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        self.head = end;
        Some(Allocation {
            region: Region {
                offset: aligned,
                size,
            },
            handle: 0,
        })
    }

    /// Resets the cursor to the start, making the whole heap allocatable again.
    pub fn reset(&mut self) {
        self.head = 0;
    }
}

// ============================================================================
// Ring
// ============================================================================

/// A circular per-frame upload heap with in-flight protection.
///
/// Allocations advance a monotonic cursor around a ring of `capacity` bytes. A
/// single allocation never straddles the wrap boundary: if it would, the tail
/// of the ring is skipped and the allocation starts over at offset `0`.
///
/// Frames bound regions of the ring. Open a frame with
/// [`RingAllocator::begin_frame`], and once the GPU has finished consuming a
/// frame's uploads mark it done with [`RingAllocator::retire`]. The allocator
/// never hands out a range still owned by an un-retired frame; instead it
/// returns `None`, so in-flight data is never overwritten.
pub struct RingAllocator {
    capacity: u64,
    /// Monotonic write cursor (absolute, never wraps); offset is `head %
    /// capacity`.
    head: u64,
    /// Absolute cursor of the oldest byte still owned by an un-retired frame.
    tail: u64,
    current_frame: Option<u64>,
    /// Closed-but-not-retired frames as `(frame, end_cursor)`, oldest first.
    inflight: VecDeque<(u64, u64)>,
}

impl RingAllocator {
    /// Creates a ring upload heap of `size` bytes.
    #[must_use]
    pub fn new(size: u64) -> Self {
        Self {
            capacity: size,
            head: 0,
            tail: 0,
            current_frame: None,
            inflight: VecDeque::new(),
        }
    }

    /// The ring capacity, in bytes.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The number of bytes currently in flight (allocated and not yet
    /// reclaimed by [`RingAllocator::retire`]).
    #[must_use]
    pub fn in_flight(&self) -> u64 {
        self.head - self.tail
    }

    /// Opens frame `frame`, closing the previously open frame (if any) so it
    /// becomes eligible for retirement. Allocations are attributed to the
    /// currently open frame.
    pub fn begin_frame(&mut self, frame: u64) {
        if let Some(current) = self.current_frame {
            self.inflight.push_back((current, self.head));
        }
        self.current_frame = Some(frame);
    }

    /// Marks `frame` (and every older still-in-flight frame) as consumed,
    /// reclaiming their ring space. Does nothing if `frame` is unknown or
    /// already retired.
    pub fn retire(&mut self, frame: u64) {
        if !self.inflight.iter().any(|&(f, _)| f == frame) {
            return;
        }
        while let Some(&(f, end)) = self.inflight.front() {
            self.inflight.pop_front();
            self.tail = end;
            if f == frame {
                break;
            }
        }
    }

    /// Allocates `size` bytes aligned to `align` (a power of two, where `0` is
    /// treated as `1`) within the open frame. Returns `None` if no frame is
    /// open, `align` is invalid, the request is empty or larger than the ring,
    /// or satisfying it would overwrite in-flight data.
    #[must_use]
    pub fn allocate(&mut self, size: u64, align: u64) -> Option<Allocation> {
        if size == 0 || self.capacity == 0 || self.current_frame.is_none() {
            return None;
        }
        let align = normalize_align(align)?;
        if size > self.capacity {
            return None;
        }

        let off_in_ring = self.head % self.capacity;
        let aligned = align_up(off_in_ring, align)?;

        let (result_offset, start_cursor) = if aligned + size <= self.capacity {
            (aligned, self.head + (aligned - off_in_ring))
        } else {
            // Skip the tail of the ring and restart at offset 0.
            let next_base = self.head - off_in_ring + self.capacity;
            (0u64, next_base)
        };

        let end_cursor = start_cursor.checked_add(size)?;
        if end_cursor - self.tail > self.capacity {
            return None;
        }

        self.head = end_cursor;
        Some(Allocation {
            region: Region {
                offset: result_offset,
                size,
            },
            handle: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TLSF ----------------------------------------------------------

    #[test]
    fn tlsf_basic_alignment_and_exact_fit() {
        let mut a = TlsfAllocator::new(1024);
        let x = a.allocate(100, 256).unwrap();
        assert_eq!(x.offset() % 256, 0);
        assert_eq!(x.size(), 100);
        assert_eq!(a.allocated(), 100);
        assert_eq!(a.capacity(), 1024);
    }

    #[test]
    fn tlsf_alignment_rejects_non_power_of_two() {
        let mut a = TlsfAllocator::new(1024);
        assert!(a.allocate(16, 3).is_none());
        assert!(a.allocate(16, 0).is_some()); // 0 treated as 1
    }

    #[test]
    fn tlsf_split_produces_disjoint_blocks() {
        let mut a = TlsfAllocator::new(1024);
        let x = a.allocate(200, 1).unwrap();
        let y = a.allocate(200, 1).unwrap();
        let xr = x.region();
        let yr = y.region();
        // Disjoint ranges.
        assert!(xr.end().unwrap() <= yr.offset || yr.end().unwrap() <= xr.offset);
        assert_eq!(a.allocated(), 400);
    }

    #[test]
    fn tlsf_oom_returns_none() {
        let mut a = TlsfAllocator::new(256);
        assert!(a.allocate(300, 1).is_none());
        let _ = a.allocate(200, 1).unwrap();
        assert!(a.allocate(100, 1).is_none());
    }

    #[test]
    fn tlsf_free_coalesces_to_single_block() {
        let mut a = TlsfAllocator::new(1024);
        let x = a.allocate(128, 1).unwrap();
        let y = a.allocate(128, 1).unwrap();
        let z = a.allocate(128, 1).unwrap();
        a.free(y);
        a.free(x);
        a.free(z);
        assert_eq!(a.allocated(), 0);
        // Fully coalesced: a single capacity-sized allocation must succeed.
        let whole = a.allocate(1024, 1).unwrap();
        assert_eq!(whole.offset(), 0);
        assert_eq!(whole.size(), 1024);
    }

    #[test]
    fn tlsf_fragmentation_metric() {
        let mut a = TlsfAllocator::new(1024);
        // No allocations: all free space is one block -> 0.
        assert_eq!(a.fragmentation(), 0.0);

        let x = a.allocate(256, 1).unwrap();
        let y = a.allocate(256, 1).unwrap();
        let z = a.allocate(256, 1).unwrap();
        // Fully allocated except one 256 tail block -> single free block -> 0.
        a.free(x);
        a.free(z);
        // Free space is split (freed x, the live y, freed z, tail), so the
        // largest free block is strictly smaller than total free space.
        let frag = a.fragmentation();
        assert!(frag > 0.0 && frag < 1.0, "frag = {frag}");

        // Freeing everything collapses fragmentation back to 0.
        a.free(y);
        assert_eq!(a.fragmentation(), 0.0);
    }

    #[test]
    fn tlsf_many_small_then_reuse() {
        let mut a = TlsfAllocator::new(4096);
        let mut live = Vec::new();
        for _ in 0..32 {
            live.push(a.allocate(64, 16).unwrap());
        }
        assert_eq!(a.allocated(), 32 * 64);
        for alloc in live.drain(..) {
            a.free(alloc);
        }
        assert_eq!(a.allocated(), 0);
        assert!(a.allocate(4096, 1).is_some());
    }

    // ---- Buddy ---------------------------------------------------------

    #[test]
    fn buddy_rounds_capacity_and_requests() {
        let mut a = BuddyAllocator::new(1000);
        assert_eq!(a.capacity(), 1024);
        let x = a.allocate(100).unwrap();
        assert_eq!(x.size(), 128); // rounded up to power of two
        assert_eq!(x.offset() % 128, 0);
    }

    #[test]
    fn buddy_split_merge_round_trip() {
        let mut a = BuddyAllocator::new(1024);
        let x = a.allocate(256).unwrap();
        let y = a.allocate(256).unwrap();
        assert_ne!(x.offset(), y.offset());
        assert_eq!(a.allocated(), 512);
        a.free(x);
        a.free(y);
        assert_eq!(a.allocated(), 0);
        // After full merge the whole heap is one block again.
        let whole = a.allocate(1024).unwrap();
        assert_eq!(whole.offset(), 0);
        assert_eq!(whole.size(), 1024);
    }

    #[test]
    fn buddy_oom_and_empty() {
        let mut a = BuddyAllocator::new(512);
        assert!(a.allocate(0).is_none());
        assert!(a.allocate(1024).is_none());
        let _ = a.allocate(512).unwrap();
        assert!(a.allocate(1).is_none());

        let mut empty = BuddyAllocator::new(0);
        assert_eq!(empty.capacity(), 0);
        assert!(empty.allocate(1).is_none());
    }

    #[test]
    fn buddy_buddy_addresses_merge() {
        let mut a = BuddyAllocator::new(1024);
        // Four quarter blocks; freeing all must recover the full heap.
        let b0 = a.allocate(256).unwrap();
        let b1 = a.allocate(256).unwrap();
        let b2 = a.allocate(256).unwrap();
        let b3 = a.allocate(256).unwrap();
        for b in [b2, b0, b3, b1] {
            a.free(b);
        }
        assert_eq!(a.allocated(), 0);
        assert!(a.allocate(1024).is_some());
    }

    // ---- Linear --------------------------------------------------------

    #[test]
    fn linear_bump_alignment_and_reset() {
        let mut a = LinearAllocator::new(1024);
        let x = a.allocate(10, 1).unwrap();
        assert_eq!(x.offset(), 0);
        let y = a.allocate(10, 64).unwrap();
        assert_eq!(y.offset() % 64, 0);
        assert!(y.offset() >= x.offset() + x.size());
        let used = a.allocated();
        assert!(used > 0);

        a.reset();
        assert_eq!(a.allocated(), 0);
        let z = a.allocate(10, 1).unwrap();
        assert_eq!(z.offset(), 0); // space reused after reset
    }

    #[test]
    fn linear_oom_returns_none() {
        let mut a = LinearAllocator::new(32);
        assert!(a.allocate(0, 1).is_none());
        assert!(a.allocate(16, 1).is_some());
        assert!(a.allocate(32, 1).is_none());
        assert!(a.allocate(16, 3).is_none()); // bad align
    }

    // ---- Ring ----------------------------------------------------------

    #[test]
    fn ring_requires_open_frame() {
        let mut a = RingAllocator::new(256);
        assert!(a.allocate(16, 1).is_none());
        a.begin_frame(0);
        assert!(a.allocate(16, 1).is_some());
    }

    #[test]
    fn ring_wraparound_restarts_at_zero() {
        let mut a = RingAllocator::new(256);
        a.begin_frame(0);
        let x = a.allocate(200, 1).unwrap();
        assert_eq!(x.offset(), 0);
        // 200 used; 56 left to the end, but a 100-byte request can't fit the
        // tail, so it wraps to offset 0. That overwrites frame 0 which is still
        // open (in flight) -> must be refused.
        assert!(a.allocate(100, 1).is_none());
    }

    #[test]
    fn ring_in_flight_blocks_then_retire_reuses() {
        let mut a = RingAllocator::new(256);
        a.begin_frame(0);
        let _ = a.allocate(200, 1).unwrap();
        a.begin_frame(1); // close frame 0 (still in flight)

        // Frame 1 wants to wrap into frame 0's region -> blocked.
        assert!(a.allocate(100, 1).is_none());

        // Retire frame 0; its bytes are reclaimed and the wrap now succeeds.
        a.retire(0);
        let y = a.allocate(100, 1).unwrap();
        assert_eq!(y.offset(), 0);
    }

    #[test]
    fn ring_retire_unknown_frame_is_noop() {
        let mut a = RingAllocator::new(256);
        a.begin_frame(0);
        let _ = a.allocate(64, 1).unwrap();
        let before = a.in_flight();
        a.retire(99); // not in flight
        assert_eq!(a.in_flight(), before);
    }

    #[test]
    fn ring_rejects_oversized_and_bad_align() {
        let mut a = RingAllocator::new(128);
        a.begin_frame(0);
        assert!(a.allocate(256, 1).is_none()); // larger than ring
        assert!(a.allocate(16, 3).is_none()); // bad align
        assert!(a.allocate(0, 1).is_none()); // empty
    }
}
