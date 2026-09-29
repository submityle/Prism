//! First-fit free-list allocator for the variable-length material parameter
//! word heap (design doc §3.3).
//!
//! The unified material ABI stores each surface as an über-BSDF core plus only
//! the optional lobes it actually uses ([`prism_render_material::
//! SurfaceParameterBlock`]). A plain dielectric costs 12 words; a fully-loaded
//! multi-lobe surface costs up to 36. Rather than pay the worst case for every
//! material with a fixed stride, the scene packs each material into a contiguous
//! run of `u32` words at a per-material `parameter_offset`.
//!
//! Allocations are variable length and a material's length can change when its
//! authored lobe set changes, so we need a real sub-allocator over the word
//! address space:
//!
//! * [`ParameterHeap::alloc`] returns the word offset of a fresh run, reusing a
//!   freed hole (first fit) when one is large enough and otherwise bumping the
//!   high-water mark.
//! * [`ParameterHeap::free`] returns a run to the free list and coalesces it
//!   with any adjacent free spans so long-lived scenes do not fragment into
//!   unusable slivers.
//!
//! The allocator is address-space bookkeeping only: it never owns the GPU
//! buffer. The caller (`MaterialGpuBuffers`) writes the packed words into an
//! `AtomicSparseBufferVec<u32>` at the returned offsets, and mirrors the
//! high-water mark as the buffer's logical length. Keeping the two concerns
//! separate makes the allocator exhaustively unit-testable on the CPU without a
//! device.

use alloc::vec::Vec;

/// A single free run in the parameter word heap, `[offset, offset + len)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FreeSpan {
    offset: u32,
    len: u32,
}

/// First-fit free-list allocator over a `u32` word address space.
///
/// Free spans are kept sorted by offset and always maximally coalesced, so the
/// invariant "no two free spans are adjacent" holds after every operation. This
/// keeps [`Self::free`] O(n) in the (small) number of holes and makes the
/// structure trivial to audit.
#[derive(Clone, Debug, Default)]
pub(crate) struct ParameterHeap {
    /// Sorted, coalesced free spans below `high_water`.
    free: Vec<FreeSpan>,
    /// One past the highest word ever handed out; also the logical word length
    /// the backing buffer must cover.
    high_water: u32,
}

impl ParameterHeap {
    /// An empty heap that has never allocated.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The number of words the backing buffer must be able to address, i.e. one
    /// past the highest allocated word. Trailing free spans are folded back into
    /// the high-water mark by [`Self::free`], so a fully-freed heap reports `0`.
    #[cfg(test)]
    pub(crate) fn word_capacity(&self) -> u32 {
        self.high_water
    }

    /// Total number of free words currently available below the high-water mark
    /// (test/diagnostic aid; not counting the unbounded space above it).
    #[cfg(test)]
    pub(crate) fn free_words_below_high_water(&self) -> u32 {
        self.free.iter().map(|span| span.len).sum()
    }

    /// Number of distinct free holes (fragmentation gauge).
    #[cfg(test)]
    pub(crate) fn hole_count(&self) -> usize {
        self.free.len()
    }

    /// Reserve `len` contiguous words and return their starting offset.
    ///
    /// A `len` of zero reserves nothing and returns the current high-water mark;
    /// real material blocks are always at least the core width so this is only
    /// exercised defensively.
    pub(crate) fn alloc(&mut self, len: u32) -> u32 {
        if len == 0 {
            return self.high_water;
        }
        // First fit: the earliest hole that can hold the run. Exact fits drop the
        // hole entirely; larger holes shrink from their front so the tail stays
        // coalesced with whatever follows.
        for index in 0..self.free.len() {
            let span = self.free[index];
            if span.len >= len {
                let offset = span.offset;
                if span.len == len {
                    self.free.remove(index);
                } else {
                    self.free[index] = FreeSpan {
                        offset: span.offset + len,
                        len: span.len - len,
                    };
                }
                return offset;
            }
        }
        // No hole fit: bump the high-water mark.
        let offset = self.high_water;
        self.high_water += len;
        offset
    }

    /// Return the `len` words at `offset` to the free list, coalescing with any
    /// adjacent free spans. A `len` of zero is a no-op.
    ///
    /// # Panics
    /// Panics if the freed run extends past the high-water mark or overlaps an
    /// existing free span, which would indicate a double free or a bookkeeping
    /// desync in the caller.
    pub(crate) fn free(&mut self, offset: u32, len: u32) {
        if len == 0 {
            return;
        }
        let end = offset
            .checked_add(len)
            .expect("freed parameter run overflows the word address space");
        assert!(
            end <= self.high_water,
            "freed parameter run [{offset}, {end}) extends past high-water {}",
            self.high_water
        );

        // Find the sorted insertion point and assert non-overlap with neighbours.
        let insert = self.free.partition_point(|span| span.offset < offset);
        if let Some(prev) = insert.checked_sub(1).and_then(|i| self.free.get(i)) {
            assert!(
                prev.offset + prev.len <= offset,
                "double free: [{offset}, {end}) overlaps free span [{}, {})",
                prev.offset,
                prev.offset + prev.len
            );
        }
        if let Some(next) = self.free.get(insert) {
            assert!(
                end <= next.offset,
                "double free: [{offset}, {end}) overlaps free span [{}, {})",
                next.offset,
                next.offset + next.len
            );
        }

        self.free.insert(insert, FreeSpan { offset, len });
        self.coalesce_around(insert);
        self.trim_high_water();
    }

    /// Merge the span at `index` with its immediate neighbours when they abut.
    fn coalesce_around(&mut self, index: usize) {
        // Merge with the following span first so `index` stays valid.
        if let Some(next) = self.free.get(index + 1).copied() {
            let span = self.free[index];
            if span.offset + span.len == next.offset {
                self.free[index].len += next.len;
                self.free.remove(index + 1);
            }
        }
        if index > 0 {
            let span = self.free[index];
            let prev = self.free[index - 1];
            if prev.offset + prev.len == span.offset {
                self.free[index - 1].len += span.len;
                self.free.remove(index);
            }
        }
    }

    /// If the last free span reaches the high-water mark, reclaim it so a fully
    /// freed heap collapses back to zero and the backing buffer can shrink.
    fn trim_high_water(&mut self) {
        while let Some(last) = self.free.last().copied() {
            if last.offset + last.len == self.high_water {
                self.high_water = last.offset;
                self.free.pop();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_allocs_bump_the_high_water_mark() {
        let mut heap = ParameterHeap::new();
        assert_eq!(heap.alloc(12), 0);
        assert_eq!(heap.alloc(16), 12);
        assert_eq!(heap.alloc(4), 28);
        assert_eq!(heap.word_capacity(), 32);
        assert_eq!(heap.hole_count(), 0);
    }

    #[test]
    fn zero_length_alloc_reserves_nothing() {
        let mut heap = ParameterHeap::new();
        assert_eq!(heap.alloc(0), 0);
        assert_eq!(heap.word_capacity(), 0);
        assert_eq!(heap.alloc(12), 0);
        assert_eq!(heap.alloc(0), 12);
        assert_eq!(heap.word_capacity(), 12);
    }

    #[test]
    fn freeing_the_tail_collapses_the_high_water_mark() {
        let mut heap = ParameterHeap::new();
        let a = heap.alloc(12);
        let b = heap.alloc(16);
        heap.free(b, 16);
        assert_eq!(heap.word_capacity(), 12);
        heap.free(a, 12);
        assert_eq!(heap.word_capacity(), 0);
        assert_eq!(heap.hole_count(), 0);
    }

    #[test]
    fn interior_hole_is_reused_by_first_fit() {
        let mut heap = ParameterHeap::new();
        let a = heap.alloc(12);
        let _b = heap.alloc(16);
        let c = heap.alloc(12);
        // Free the middle run, leaving an interior 16-word hole.
        heap.free(a + 12, 16);
        assert_eq!(heap.hole_count(), 1);
        assert_eq!(heap.free_words_below_high_water(), 16);
        // A run that fits the hole reuses it rather than growing the heap.
        let capacity_before = heap.word_capacity();
        let reused = heap.alloc(12);
        assert_eq!(reused, a + 12);
        assert_eq!(heap.word_capacity(), capacity_before);
        // The 4-word remainder of the hole survives.
        assert_eq!(heap.free_words_below_high_water(), 4);
        let _ = c;
    }

    #[test]
    fn oversized_request_skips_a_too_small_hole() {
        let mut heap = ParameterHeap::new();
        let a = heap.alloc(4);
        let _b = heap.alloc(12);
        heap.free(a, 4);
        // The 4-word hole cannot hold a 12-word run, so the heap grows.
        let capacity_before = heap.word_capacity();
        let grown = heap.alloc(12);
        assert_eq!(grown, capacity_before);
        assert_eq!(heap.free_words_below_high_water(), 4);
    }

    #[test]
    fn adjacent_frees_coalesce_into_one_hole() {
        let mut heap = ParameterHeap::new();
        let a = heap.alloc(8);
        let b = heap.alloc(8);
        let c = heap.alloc(8);
        let _guard = heap.alloc(4); // keep the tail pinned so holes stay interior
        // Free out of order; the three 8-word runs must merge into one 24 hole.
        heap.free(b, 8);
        heap.free(a, 8);
        heap.free(c, 8);
        assert_eq!(heap.hole_count(), 1);
        assert_eq!(heap.free_words_below_high_water(), 24);
        // The merged hole is contiguous from offset 0 and satisfies a 24 run.
        assert_eq!(heap.alloc(24), 0);
        assert_eq!(heap.hole_count(), 0);
    }

    #[test]
    fn free_merges_with_both_neighbours() {
        let mut heap = ParameterHeap::new();
        let a = heap.alloc(8);
        let b = heap.alloc(8);
        let c = heap.alloc(8);
        let _guard = heap.alloc(4);
        // Free the outer two first, then the middle bridges them into one span.
        heap.free(a, 8);
        heap.free(c, 8);
        assert_eq!(heap.hole_count(), 2);
        heap.free(b, 8);
        assert_eq!(heap.hole_count(), 1);
        assert_eq!(heap.free_words_below_high_water(), 24);
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn overlapping_free_panics() {
        let mut heap = ParameterHeap::new();
        let a = heap.alloc(16);
        heap.free(a, 8);
        heap.free(a + 4, 8);
    }

    #[test]
    #[should_panic(expected = "past high-water")]
    fn freeing_past_the_high_water_mark_panics() {
        let mut heap = ParameterHeap::new();
        let _a = heap.alloc(8);
        heap.free(0, 16);
    }

    #[test]
    fn realloc_growth_relocates_and_frees_the_old_run() {
        // Emulates a material whose lobe set grew: free old run, alloc a larger
        // one. With a pinned tail the old hole is interior and reusable.
        let mut heap = ParameterHeap::new();
        let old = heap.alloc(12);
        let _pin = heap.alloc(4);
        heap.free(old, 12);
        let grown = heap.alloc(16);
        // 12-word hole can't hold 16, so it grows past the pin.
        assert_eq!(grown, heap.word_capacity() - 16);
        assert_eq!(heap.free_words_below_high_water(), 12);
        // A later 12-word material slots straight back into the vacated hole.
        assert_eq!(heap.alloc(12), old);
    }
}
