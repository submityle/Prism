//! Variable-height list metrics backed by a prefix-sum offset table.

use alloc::vec::Vec;
use core::ops::Range;

use crate::viewport::Viewport;

/// Windowing metrics for a list whose items may each have a different main-axis
/// size.
///
/// The per-item sizes are summed once into a prefix-sum offset table so that
/// [`VariableList::offset_of`] is constant time and
/// [`VariableList::index_at`] is logarithmic via binary search. The table holds
/// `item_count + 1` entries: `offsets[i]` is the start of item `i`, and the
/// final entry is the total size.
#[derive(Clone, Debug, PartialEq)]
pub struct VariableList {
    /// Prefix-sum offsets; `offsets.len() == item_count + 1`.
    offsets: Vec<f32>,
}

impl VariableList {
    /// Builds a list from per-item main-axis sizes.
    ///
    /// Sizes are expected to be non-negative so that the offset table is
    /// monotonically non-decreasing, which [`VariableList::index_at`] relies on
    /// for its binary search.
    #[must_use]
    pub fn from_sizes<I>(sizes: I) -> Self
    where
        I: IntoIterator<Item = f32>,
    {
        let iter = sizes.into_iter();
        let (lower, _) = iter.size_hint();
        let mut offsets = Vec::with_capacity(lower + 1);
        let mut acc = 0.0_f32;
        offsets.push(acc);
        for size in iter {
            acc += size;
            offsets.push(acc);
        }
        Self { offsets }
    }

    /// Number of items in the list.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The main-axis size of the item at `index`.
    #[must_use]
    pub fn size_of(&self, index: usize) -> f32 {
        debug_assert!(index < self.item_count(), "index out of bounds");
        self.offsets[index + 1] - self.offsets[index]
    }

    /// The total main-axis extent of the whole list.
    #[must_use]
    pub fn total_size(&self) -> f32 {
        *self.offsets.last().unwrap_or(&0.0)
    }

    /// The main-axis offset of the start of the item at `index`.
    ///
    /// Defined for `index` in `0..=item_count`; `offset_of(item_count)` returns
    /// the total size. Indices past the end are clamped to the total size.
    #[must_use]
    pub fn offset_of(&self, index: usize) -> f32 {
        let clamped = index.min(self.item_count());
        self.offsets[clamped]
    }

    /// The index of the item containing `offset`.
    ///
    /// The lookup is a binary search over the offset table (via
    /// [`slice::partition_point`]). Offsets at or below `0` map to the first
    /// item; offsets at or beyond the total size map to the last item. An empty
    /// list always returns `0`.
    #[must_use]
    pub fn index_at(&self, offset: f32) -> usize {
        let count = self.item_count();
        if count == 0 {
            return 0;
        }
        // Number of table entries whose offset is <= `offset`. For any offset
        // inside item `i` this is `i + 1`, so subtracting one recovers `i`.
        let found = self.offsets.partition_point(|&o| o <= offset);
        found.saturating_sub(1).min(count - 1)
    }

    /// The half-open range of item indices intersecting `viewport`, widened by
    /// the viewport's overscan and clamped to `0..item_count`.
    ///
    /// A viewport with a non-positive [`Viewport::length`] yields an empty
    /// range anchored at the clamped first visible index.
    #[must_use]
    pub fn visible_range(&self, viewport: &Viewport) -> Range<usize> {
        let count = self.item_count();
        if count == 0 {
            return 0..0;
        }

        let first = self.index_at(viewport.start());
        let start = first.saturating_sub(viewport.overscan);

        if viewport.length <= 0.0 {
            return start..start;
        }

        let last = self.index_at(viewport.end());
        let end = (last + 1 + viewport.overscan).min(count);
        let start = start.min(end);
        start..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample() -> VariableList {
        // sizes: 10, 15, 5, 20  => offsets: 0, 10, 25, 30, 50
        VariableList::from_sizes(vec![10.0, 15.0, 5.0, 20.0])
    }

    #[test]
    fn prefix_sum_offsets_and_sizes() {
        let list = sample();
        assert_eq!(list.item_count(), 4);
        assert_eq!(list.total_size(), 50.0);
        assert_eq!(list.offset_of(0), 0.0);
        assert_eq!(list.offset_of(1), 10.0);
        assert_eq!(list.offset_of(2), 25.0);
        assert_eq!(list.offset_of(3), 30.0);
        assert_eq!(list.offset_of(4), 50.0);
        assert_eq!(list.size_of(1), 15.0);
        assert_eq!(list.size_of(3), 20.0);
    }

    #[test]
    fn index_at_binary_search_boundaries() {
        let list = sample();
        // Exactly on a boundary belongs to the item starting there.
        assert_eq!(list.index_at(0.0), 0);
        assert_eq!(list.index_at(10.0), 1);
        assert_eq!(list.index_at(25.0), 2);
        assert_eq!(list.index_at(30.0), 3);
        // Interior offsets.
        assert_eq!(list.index_at(9.999), 0);
        assert_eq!(list.index_at(24.0), 1);
        assert_eq!(list.index_at(29.0), 2);
        // Below the start and at/above the end clamp.
        assert_eq!(list.index_at(-5.0), 0);
        assert_eq!(list.index_at(50.0), 3);
        assert_eq!(list.index_at(1000.0), 3);
    }

    #[test]
    fn index_at_empty_list_is_zero() {
        let list = VariableList::from_sizes(vec![]);
        assert_eq!(list.item_count(), 0);
        assert_eq!(list.total_size(), 0.0);
        assert_eq!(list.index_at(0.0), 0);
        assert_eq!(list.index_at(100.0), 0);
        assert!(list.visible_range(&Viewport::new(0.0, 10.0)).is_empty());
    }

    #[test]
    fn visible_range_with_overscan_and_clamp() {
        let list = sample();
        // Window [10, 28): first = index_at(10)=1, last = index_at(28)=2.
        let vp = Viewport::new(10.0, 18.0);
        assert_eq!(list.visible_range(&vp), 1..3);
        // Add overscan 1 on both sides, clamped.
        let vp = Viewport::new(10.0, 18.0).with_overscan(1);
        assert_eq!(list.visible_range(&vp), 0..4);
        // Overscan beyond the ends still clamps to 0..count.
        let vp = Viewport::new(10.0, 18.0).with_overscan(10);
        assert_eq!(list.visible_range(&vp), 0..4);
    }

    #[test]
    fn offset_of_clamps_past_end() {
        let list = sample();
        assert_eq!(list.offset_of(99), 50.0);
    }
}
