//! Linear (bump) allocator for transient, frame-scoped heaps.
//!
//! Resources in the [`Transient`](super::HeapClass::Transient) class live for at
//! most one frame: scratch buffers, staging ranges, per-frame uniforms. They are
//! never individually freed, so a full free-list is wasted overhead. Instead a
//! bump allocator hands out strictly increasing offsets and is reset wholesale
//! at frame boundaries, which is both faster and trivially deterministic.
//!
//! Like the rest of this subsystem the allocator is pure `CPU` bookkeeping over a
//! region the backend commits; binding the actual device memory is *pending the
//! GPU backend*.

use super::HeapStats;

/// Rounds `value` up to the next multiple of a power-of-two `align`.
///
/// Returns `None` on overflow so the caller reports an allocation failure rather
/// than wrapping. `align` is validated to be a power of two before this runs.
fn align_up(value: u64, align: u64) -> Option<u64> {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let mask = align - 1;
    value.checked_add(mask).map(|rounded| rounded & !mask)
}

/// Reason a [`LinearAllocator::allocate`] request failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearError {
    /// A zero-byte allocation was requested.
    ZeroSize,
    /// The requested alignment was not a non-zero power of two.
    InvalidAlignment(u64),
    /// The bump cursor plus the aligned request would exceed capacity.
    OutOfMemory {
        /// Bytes requested.
        requested: u64,
        /// Contiguous bytes still available from the current cursor.
        available: u64,
    },
}

impl core::fmt::Display for LinearError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroSize => formatter.write_str("cannot bump-allocate zero bytes"),
            Self::InvalidAlignment(alignment) => {
                write!(formatter, "alignment {alignment} is not a power of two")
            }
            Self::OutOfMemory {
                requested,
                available,
            } => write!(
                formatter,
                "transient heap exhausted: requested {requested} bytes, {available} available"
            ),
        }
    }
}

impl std::error::Error for LinearError {}

/// A byte range handed out by the bump allocator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearAllocation {
    /// Aligned start offset within the transient region.
    pub offset: u64,
    /// Length of the allocation in bytes.
    pub size: u64,
}

impl LinearAllocation {
    /// One-past-the-end offset of this allocation.
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.offset + self.size
    }
}

/// Monotonic bump allocator reset once per frame.
///
/// Allocation is a single aligned cursor advance, so it never fragments and its
/// offsets are a pure function of the request sequence. [`reset`](Self::reset)
/// reclaims everything for the next frame while preserving the observed
/// high-water mark, which sizing heuristics use to right-size the region.
#[derive(Clone, Debug)]
pub struct LinearAllocator {
    capacity: u64,
    cursor: u64,
    high_water: u64,
    allocations: u32,
}

impl LinearAllocator {
    /// Creates a bump allocator over `[0, capacity)`.
    #[must_use]
    pub const fn new(capacity: u64) -> Self {
        Self {
            capacity,
            cursor: 0,
            high_water: 0,
            allocations: 0,
        }
    }

    /// Total committed capacity of the transient region.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes handed out since the last [`reset`](Self::reset).
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.cursor
    }

    /// Contiguous bytes still available from the current cursor.
    #[must_use]
    pub const fn remaining_bytes(&self) -> u64 {
        self.capacity - self.cursor
    }

    /// Largest `used_bytes` observed across all frames since construction (or
    /// the last [`reset_high_water`](Self::reset_high_water)).
    #[must_use]
    pub const fn high_water_mark(&self) -> u64 {
        self.high_water
    }

    /// Number of allocations served since the last [`reset`](Self::reset).
    #[must_use]
    pub const fn allocation_count(&self) -> u32 {
        self.allocations
    }

    /// Snapshot of this region as a [`HeapStats`] contract value. All free space
    /// is contiguous above the cursor, so `largest_free_block == remaining`.
    #[must_use]
    pub const fn heap_stats(&self) -> HeapStats {
        HeapStats {
            committed_bytes: self.capacity,
            used_bytes: self.cursor,
            largest_free_block: self.capacity - self.cursor,
        }
    }

    /// Bumps the cursor to serve `size` bytes aligned to `alignment`.
    ///
    /// Returns the [`LinearAllocation`] or a [`LinearError`]; never panics when
    /// the region is full or the alignment would overflow.
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Result<LinearAllocation, LinearError> {
        if size == 0 {
            return Err(LinearError::ZeroSize);
        }
        if !alignment.is_power_of_two() {
            return Err(LinearError::InvalidAlignment(alignment));
        }

        let aligned = align_up(self.cursor, alignment)
            .filter(|value| *value <= self.capacity)
            .ok_or(LinearError::OutOfMemory {
                requested: size,
                available: self.remaining_bytes(),
            })?;
        let end = aligned
            .checked_add(size)
            .filter(|value| *value <= self.capacity)
            .ok_or(LinearError::OutOfMemory {
                requested: size,
                available: self.remaining_bytes(),
            })?;

        self.cursor = end;
        if self.cursor > self.high_water {
            self.high_water = self.cursor;
        }
        self.allocations += 1;
        Ok(LinearAllocation {
            offset: aligned,
            size,
        })
    }

    /// Reclaims the whole region for the next frame, keeping the high-water mark.
    pub const fn reset(&mut self) {
        self.cursor = 0;
        self.allocations = 0;
    }

    /// Clears the retained high-water mark back to zero.
    pub const fn reset_high_water(&mut self) {
        self.high_water = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bumps_offsets_with_alignment() {
        let mut allocator = LinearAllocator::new(1024);
        let a = allocator.allocate(10, 1).unwrap();
        let b = allocator.allocate(4, 16).unwrap();
        assert_eq!(a.offset, 0);
        // Cursor was 10, aligned up to 16 for b.
        assert_eq!(b.offset, 16);
        assert_eq!(allocator.used_bytes(), 20);
        assert_eq!(allocator.allocation_count(), 2);
    }

    #[test]
    fn rejects_zero_size_and_bad_alignment() {
        let mut allocator = LinearAllocator::new(256);
        assert_eq!(allocator.allocate(0, 16), Err(LinearError::ZeroSize));
        assert_eq!(
            allocator.allocate(16, 6),
            Err(LinearError::InvalidAlignment(6))
        );
    }

    #[test]
    fn full_region_reports_out_of_memory() {
        let mut allocator = LinearAllocator::new(64);
        let _ = allocator.allocate(64, 1).unwrap();
        let err = allocator.allocate(1, 1).unwrap_err();
        assert_eq!(
            err,
            LinearError::OutOfMemory {
                requested: 1,
                available: 0,
            }
        );
    }

    #[test]
    fn alignment_past_capacity_fails_cleanly() {
        let mut allocator = LinearAllocator::new(100);
        let _ = allocator.allocate(10, 1).unwrap();
        // Aligning the cursor (10) up to 256 exceeds capacity.
        assert!(matches!(
            allocator.allocate(1, 256),
            Err(LinearError::OutOfMemory { .. })
        ));
    }

    #[test]
    fn reset_reclaims_but_keeps_high_water() {
        let mut allocator = LinearAllocator::new(512);
        let _ = allocator.allocate(300, 1).unwrap();
        assert_eq!(allocator.high_water_mark(), 300);
        allocator.reset();
        assert_eq!(allocator.used_bytes(), 0);
        assert_eq!(allocator.allocation_count(), 0);
        // High-water survives the frame reset.
        assert_eq!(allocator.high_water_mark(), 300);

        let _ = allocator.allocate(100, 1).unwrap();
        // Smaller frame does not lower the mark.
        assert_eq!(allocator.high_water_mark(), 300);
        allocator.reset_high_water();
        assert_eq!(allocator.high_water_mark(), 0);
    }

    #[test]
    fn heap_stats_free_block_is_contiguous_remainder() {
        let mut allocator = LinearAllocator::new(256);
        let _ = allocator.allocate(64, 1).unwrap();
        let stats = allocator.heap_stats();
        assert_eq!(stats.committed_bytes, 256);
        assert_eq!(stats.used_bytes, 64);
        assert_eq!(stats.largest_free_block, 192);
    }

    #[test]
    fn allocation_is_deterministic_across_frames() {
        fn frame(allocator: &mut LinearAllocator) -> (u64, u64, u64) {
            let a = allocator.allocate(32, 32).unwrap();
            let b = allocator.allocate(1, 64).unwrap();
            let c = allocator.allocate(200, 128).unwrap();
            (a.offset, b.offset, c.offset)
        }
        let mut allocator = LinearAllocator::new(4096);
        let first = frame(&mut allocator);
        allocator.reset();
        let second = frame(&mut allocator);
        assert_eq!(first, second);
    }
}
