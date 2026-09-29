//! `CPU`-verifiable `GPU`/`VRAM` sub-allocation planning — the device-free
//! reference for how per-frame particle passes carve their transient storage
//! out of a larger `GPU` heap (design §5, §9).
//!
//! A production particle engine never issues one `WebGPU` buffer allocation per
//! pass: it reserves a few large device heaps up front and *sub-allocates* the
//! per-pass scratch (sort keys, prefix-sum scans, indirect-draw args, spawn
//! staging) out of them with a cheap bump/ring cursor. This mirrors the
//! `Frostbite` "transient resource allocator": short-lived resources live in a
//! linear arena that is reset every frame, so the only per-frame cost is a
//! cursor bump and the peak footprint is captured by a high-water mark.
//!
//! This module owns the pure-integer contract for that planner so the byte math
//! is defined and tested exactly once, independent of any device:
//!
//! * [`align_up`] / [`is_power_of_two`] — power-of-two alignment with overflow
//!   saturation, the shared primitive every arena rounds offsets with.
//! * [`BumpArena`] — a monotonic (bump) sub-allocator with a reset that keeps
//!   the high-water mark, matching a per-frame linear arena.
//! * [`RingArena`] — a ring (circular) sub-allocator for producer/consumer
//!   transient data whose lifetime spans a bounded number of frames.
//! * [`fragmentation_upper_bound`] — an upper bound on the wasted alignment
//!   holes between a set of placed regions, the metric a heap packer minimizes.
//! * [`typed_bytes`] and the `std430` stride helpers — convert an element count
//!   into a saturating byte size using the shared [`crate::particle::gpu_layout`]
//!   strides.
//!
//! Everything is `u64` integer arithmetic: no `unsafe`, no transcendental
//! functions, and no floating point. Degenerate inputs (zero capacity, a
//! non-power-of-two alignment, an overflowing count) never panic — they clamp
//! or return an [`AllocError`] instead of wrapping to a small allocation.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{U32_STRIDE, VEC2_STRIDE, VEC4_STRIDE};

/// Returns `true` when `v` is a power of two (and therefore a legal alignment).
///
/// Zero is not a power of two. Exactly one bit set means `v & (v - 1) == 0`.
#[must_use]
pub fn is_power_of_two(v: u64) -> bool {
    v != 0 && (v & (v - 1)) == 0
}

/// Normalizes an alignment request to a legal power of two.
///
/// Zero or any non-power-of-two request is treated as the trivial alignment of
/// `1` (byte-aligned), so the sub-allocators never divide by zero or round with
/// a nonsensical modulus.
fn normalize_align(align: u64) -> u64 {
    if is_power_of_two(align) {
        align
    } else {
        1
    }
}

/// Rounds `offset` up to the next multiple of `align`.
///
/// `align` must be a power of two; a zero, one, or non-power-of-two alignment
/// leaves `offset` unchanged (it is treated as byte alignment). The addition
/// saturates at [`u64::MAX`] instead of wrapping, so a near-maximal `offset`
/// can never round down to a small value.
#[must_use]
pub fn align_up(offset: u64, align: u64) -> u64 {
    if align <= 1 || !is_power_of_two(align) {
        return offset;
    }
    // `offset + align - 1`, then floor-divide and re-multiply. The saturating
    // add is the only step that can overflow; after the divide the multiply is
    // bounded by the pre-divide (already saturated) value.
    offset.saturating_add(align - 1) / align * align
}

/// A single placed sub-allocation: where it starts, how big it is, and the
/// alignment it was rounded to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubAllocRegion {
    /// Byte offset of the region from the start of its owning heap.
    pub offset: u64,
    /// Size of the region in bytes.
    pub size: u64,
    /// The (normalized power-of-two) alignment the offset satisfies.
    pub alignment: u64,
}

impl SubAllocRegion {
    /// The exclusive end offset (`offset + size`), saturating at [`u64::MAX`].
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(self.size)
    }
}

/// Why a sub-allocation request could not be satisfied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocError {
    /// The aligned request does not fit in the remaining (or total) capacity.
    OutOfSpace,
    /// A zero-byte allocation was requested, which is never a valid binding.
    ZeroSize,
}

/// A monotonic (bump) sub-allocator over a fixed-capacity heap.
///
/// Each request aligns the cursor, carves a region, and advances the cursor.
/// There is no per-region free: the whole arena is [`reset`](Self::reset) at a
/// frame boundary. The high-water mark records the largest cursor ever reached
/// so a caller can size the backing `GPU` heap to the observed peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BumpArena {
    capacity: u64,
    cursor: u64,
    high_water: u64,
}

impl BumpArena {
    /// Creates an empty bump arena spanning `capacity` bytes.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            cursor: 0,
            high_water: 0,
        }
    }

    /// Aligns the cursor and carves `size` bytes, advancing the cursor.
    ///
    /// Returns [`AllocError::ZeroSize`] for a zero-byte request and
    /// [`AllocError::OutOfSpace`] when the aligned region would exceed the
    /// capacity. On success the high-water mark is raised to the new cursor.
    ///
    /// # Errors
    ///
    /// See the description above for the two failure modes.
    pub fn allocate(&mut self, size: u64, align: u64) -> Result<SubAllocRegion, AllocError> {
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }
        let start = align_up(self.cursor, align);
        let end = start.saturating_add(size);
        if end > self.capacity {
            return Err(AllocError::OutOfSpace);
        }
        self.cursor = end;
        if end > self.high_water {
            self.high_water = end;
        }
        Ok(SubAllocRegion {
            offset: start,
            size,
            alignment: normalize_align(align),
        })
    }

    /// Rewinds the cursor to zero while preserving the high-water mark.
    ///
    /// This is the per-frame reset: the arena is reusable immediately, but the
    /// recorded peak footprint survives so heap sizing sees the true maximum.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Total capacity of the arena in bytes.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes handed out since the last [`reset`](Self::reset) (the cursor).
    #[must_use]
    pub fn used(&self) -> u64 {
        self.cursor
    }

    /// Bytes still available before the next allocation would fail.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.capacity.saturating_sub(self.cursor)
    }

    /// The largest cursor value ever reached across all frames.
    #[must_use]
    pub fn high_water_mark(&self) -> u64 {
        self.high_water
    }
}

/// A ring (circular) sub-allocator for bounded-lifetime transient data.
///
/// Producers [`allocate`](Self::allocate) at the head; consumers release the
/// oldest bytes with [`free_oldest`](Self::free_oldest) at the tail. When the
/// head advances past the capacity it wraps to the front and the wrap counter
/// increments, so a caller can detect how many times the ring has cycled. A
/// request that would exceed the live byte budget fails rather than clobbering
/// data that has not been freed yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RingArena {
    capacity: u64,
    head: u64,
    tail: u64,
    len: u64,
    wraps: u64,
}

impl RingArena {
    /// Creates an empty ring arena spanning `capacity` bytes.
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            head: 0,
            tail: 0,
            len: 0,
            wraps: 0,
        }
    }

    /// Carves `size` bytes (rounded up to `align`) at the head of the ring.
    ///
    /// Returns [`AllocError::ZeroSize`] for a zero-byte request and
    /// [`AllocError::OutOfSpace`] when the aligned request cannot fit in the
    /// currently free portion of the ring (or exceeds the total capacity). On
    /// success the head advances, wrapping to the front — and bumping the wrap
    /// counter — when it reaches or passes the capacity.
    ///
    /// # Errors
    ///
    /// See the description above for the two failure modes.
    pub fn allocate(&mut self, size: u64, align: u64) -> Result<SubAllocRegion, AllocError> {
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }
        let need = align_up(size, align);
        if need > self.capacity || self.len.saturating_add(need) > self.capacity {
            return Err(AllocError::OutOfSpace);
        }
        let offset = self.head;
        let advanced = self.head.saturating_add(need);
        if advanced >= self.capacity {
            self.head = advanced - self.capacity;
            self.wraps = self.wraps.saturating_add(1);
        } else {
            self.head = advanced;
        }
        self.len = self.len.saturating_add(need);
        Ok(SubAllocRegion {
            offset,
            size: need,
            alignment: normalize_align(align),
        })
    }

    /// Releases the `size` oldest bytes at the tail (clamped to what is live).
    ///
    /// Callers pass the same (already aligned) size a prior
    /// [`allocate`](Self::allocate) returned. The tail advances modulo the
    /// capacity, mirroring the head's wrap.
    pub fn free_oldest(&mut self, size: u64) {
        let freed = size.min(self.len);
        let advanced = self.tail.saturating_add(freed);
        self.tail = if advanced >= self.capacity {
            advanced - self.capacity
        } else {
            advanced
        };
        self.len -= freed;
    }

    /// Total capacity of the ring in bytes.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Live (allocated but not yet freed) bytes currently in the ring.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.len
    }

    /// `true` when the ring holds its full capacity and cannot accept more.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// How many times the head has wrapped past the capacity.
    #[must_use]
    pub fn wrap_count(&self) -> u64 {
        self.wraps
    }
}

/// An upper bound, in bytes, on the alignment holes wasted between `regions`.
///
/// The regions are copied and sorted by offset (via [`Vec`], with no external
/// dependency), then the gaps between each region's end and the next region's
/// start are summed. Overlapping regions contribute no negative gap: the
/// running end is the maximum end seen so far, so the result is a true upper
/// bound on internal fragmentation and never underflows.
#[must_use]
pub fn fragmentation_upper_bound(regions: &[SubAllocRegion]) -> u64 {
    if regions.len() < 2 {
        return 0;
    }
    let mut sorted: Vec<SubAllocRegion> = regions.to_vec();
    sorted.sort_unstable_by_key(|r| r.offset);
    let mut total: u64 = 0;
    let mut prev_end: Option<u64> = None;
    for region in &sorted {
        if let Some(end) = prev_end {
            total = total.saturating_add(region.offset.saturating_sub(end));
        }
        let region_end = region.end();
        prev_end = Some(prev_end.map_or(region_end, |end| end.max(region_end)));
    }
    total
}

/// Byte size of `count` elements of `element_stride` bytes each.
///
/// The multiplication saturates at [`u64::MAX`] rather than wrapping, so a
/// degenerate `count` can never collapse to a small allocation.
#[must_use]
pub fn typed_bytes(count: u64, element_stride: u64) -> u64 {
    count.saturating_mul(element_stride)
}

/// Byte size of `count` scalar `u32`/`f32` `std430` elements.
#[must_use]
pub fn u32_bytes(count: u64) -> u64 {
    typed_bytes(count, u64::try_from(U32_STRIDE).unwrap_or(u64::MAX))
}

/// Byte size of `count` `vec2` `std430` elements.
#[must_use]
pub fn vec2_bytes(count: u64) -> u64 {
    typed_bytes(count, u64::try_from(VEC2_STRIDE).unwrap_or(u64::MAX))
}

/// Byte size of `count` `vec4` `std430` elements (also the padded `vec3` size).
#[must_use]
pub fn vec4_bytes(count: u64) -> u64 {
    typed_bytes(count, u64::try_from(VEC4_STRIDE).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_zero_and_one_are_identity() {
        assert_eq!(align_up(37, 0), 37);
        assert_eq!(align_up(37, 1), 37);
    }

    #[test]
    fn align_up_non_power_of_two_is_identity() {
        // 24 is not a power of two, so it is treated as byte alignment.
        assert_eq!(align_up(37, 24), 37);
        assert_eq!(align_up(37, 3), 37);
    }

    #[test]
    fn align_up_rounds_to_power_of_two() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(255, 256), 256);
    }

    #[test]
    fn align_up_saturates_instead_of_wrapping() {
        // offset + align - 1 would overflow u64; the saturating add clamps and
        // the result never rounds down to a small value.
        let rounded = align_up(u64::MAX - 1, 16);
        assert!(rounded >= u64::MAX - 16);
    }

    #[test]
    fn is_power_of_two_edge_cases() {
        assert!(!is_power_of_two(0));
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(!is_power_of_two(3));
        assert!(is_power_of_two(1 << 40));
        assert!(!is_power_of_two((1 << 40) + 1));
    }

    #[test]
    fn region_end_saturates() {
        let region = SubAllocRegion {
            offset: 64,
            size: 128,
            alignment: 16,
        };
        assert_eq!(region.end(), 192);
        let maxed = SubAllocRegion {
            offset: u64::MAX,
            size: 8,
            alignment: 1,
        };
        assert_eq!(maxed.end(), u64::MAX);
    }

    #[test]
    fn bump_allocates_aligned_regions() {
        let mut arena = BumpArena::new(256);
        let a = arena.allocate(10, 16).expect("first fits");
        assert_eq!(a.offset, 0);
        assert_eq!(a.size, 10);
        assert_eq!(a.alignment, 16);
        let b = arena.allocate(10, 16).expect("second fits");
        // cursor was 10, aligned up to 16.
        assert_eq!(b.offset, 16);
        assert_eq!(arena.used(), 26);
        assert_eq!(arena.remaining(), 256 - 26);
        assert_eq!(arena.capacity(), 256);
    }

    #[test]
    fn bump_out_of_space_is_err() {
        let mut arena = BumpArena::new(32);
        assert!(arena.allocate(24, 1).is_ok());
        assert_eq!(arena.allocate(16, 1), Err(AllocError::OutOfSpace));
    }

    #[test]
    fn bump_zero_size_is_err() {
        let mut arena = BumpArena::new(32);
        assert_eq!(arena.allocate(0, 16), Err(AllocError::ZeroSize));
    }

    #[test]
    fn bump_reset_preserves_high_water() {
        let mut arena = BumpArena::new(256);
        arena.allocate(100, 16).expect("fits");
        arena.allocate(50, 16).expect("fits");
        let peak = arena.high_water_mark();
        assert!(peak >= 150);
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.high_water_mark(), peak);
        // A smaller frame does not lower the recorded peak.
        arena.allocate(10, 16).expect("fits");
        assert_eq!(arena.high_water_mark(), peak);
    }

    #[test]
    fn ring_allocates_and_frees() {
        let mut ring = RingArena::new(64);
        let a = ring.allocate(16, 16).expect("fits");
        assert_eq!(a.offset, 0);
        assert_eq!(ring.used(), 16);
        ring.free_oldest(16);
        assert_eq!(ring.used(), 0);
        assert_eq!(ring.wrap_count(), 0);
    }

    #[test]
    fn ring_wraps_and_counts() {
        let mut ring = RingArena::new(64);
        // Fill, free, and re-allocate so the head crosses the capacity edge.
        ring.allocate(48, 16).expect("fits");
        ring.free_oldest(48);
        let b = ring.allocate(32, 16).expect("fits after free");
        assert_eq!(b.offset, 48);
        // head was 48, +32 = 80 >= 64 -> wraps to 16, wrap_count == 1.
        assert_eq!(ring.wrap_count(), 1);
        assert_eq!(ring.used(), 32);
    }

    #[test]
    fn ring_is_full_and_rejects() {
        let mut ring = RingArena::new(32);
        ring.allocate(32, 1).expect("exact fit");
        assert!(ring.is_full());
        assert_eq!(ring.allocate(1, 1), Err(AllocError::OutOfSpace));
    }

    #[test]
    fn ring_rejects_larger_than_capacity_and_zero() {
        let mut ring = RingArena::new(16);
        assert_eq!(ring.allocate(64, 1), Err(AllocError::OutOfSpace));
        assert_eq!(ring.allocate(0, 1), Err(AllocError::ZeroSize));
    }

    #[test]
    fn fragmentation_empty_and_single_is_zero() {
        assert_eq!(fragmentation_upper_bound(&[]), 0);
        let one = [SubAllocRegion {
            offset: 0,
            size: 32,
            alignment: 16,
        }];
        assert_eq!(fragmentation_upper_bound(&one), 0);
    }

    #[test]
    fn fragmentation_contiguous_has_no_holes() {
        let regions = [
            SubAllocRegion {
                offset: 0,
                size: 16,
                alignment: 16,
            },
            SubAllocRegion {
                offset: 16,
                size: 16,
                alignment: 16,
            },
        ];
        assert_eq!(fragmentation_upper_bound(&regions), 0);
    }

    #[test]
    fn fragmentation_sums_unsorted_holes() {
        // Deliberately unsorted input with two holes: [0,16) .. [32,48) .. [80,96).
        let regions = [
            SubAllocRegion {
                offset: 80,
                size: 16,
                alignment: 16,
            },
            SubAllocRegion {
                offset: 0,
                size: 16,
                alignment: 16,
            },
            SubAllocRegion {
                offset: 32,
                size: 16,
                alignment: 16,
            },
        ];
        // Hole after [0,16) to 32 = 16; hole after [32,48) to 80 = 32; total 48.
        assert_eq!(fragmentation_upper_bound(&regions), 48);
    }

    #[test]
    fn fragmentation_ignores_overlap() {
        let regions = [
            SubAllocRegion {
                offset: 0,
                size: 64,
                alignment: 16,
            },
            SubAllocRegion {
                offset: 16,
                size: 16,
                alignment: 16,
            },
        ];
        // The second region is fully contained; no positive gap is produced.
        assert_eq!(fragmentation_upper_bound(&regions), 0);
    }

    #[test]
    fn typed_bytes_multiplies_and_saturates() {
        assert_eq!(typed_bytes(4, 16), 64);
        assert_eq!(typed_bytes(0, 16), 0);
        assert_eq!(typed_bytes(u64::MAX, 2), u64::MAX);
    }

    #[test]
    fn typed_stride_helpers_use_std430_strides() {
        assert_eq!(u32_bytes(4), 16);
        assert_eq!(vec2_bytes(4), 32);
        assert_eq!(vec4_bytes(4), 64);
    }
}
