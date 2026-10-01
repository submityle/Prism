//! Fixed-height list metrics.

use core::ops::Range;

use crate::viewport::Viewport;

/// Windowing metrics for a list whose items all share the same main-axis size.
///
/// A `FixedList` carries no element state; it is a cheap value describing the
/// geometry of a uniform list so that a windowed subrange and the surrounding
/// spacer sizes can be derived in constant time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedList {
    /// Number of items in the list.
    pub item_count: usize,
    /// Main-axis size of a single item, in logical pixels.
    pub item_height: f32,
    /// Spacing inserted between adjacent items, in logical pixels.
    pub gap: f32,
}

impl FixedList {
    /// Creates fixed-list metrics.
    #[must_use]
    pub const fn new(item_count: usize, item_height: f32, gap: f32) -> Self {
        Self {
            item_count,
            item_height,
            gap,
        }
    }

    /// The distance from the start of one item to the start of the next: the
    /// item size plus the inter-item gap.
    #[must_use]
    pub fn stride(&self) -> f32 {
        self.item_height + self.gap
    }

    /// The total main-axis extent of the whole list, gaps included.
    ///
    /// An empty list has a total size of `0`. Otherwise the size is
    /// `item_count * item_height + (item_count - 1) * gap`.
    #[must_use]
    pub fn total_size(&self) -> f32 {
        if self.item_count == 0 {
            return 0.0;
        }
        let count = self.item_count as f32;
        count * self.item_height + (count - 1.0) * self.gap
    }

    /// The main-axis offset of the start of the item at `index`.
    ///
    /// This is the gap-inclusive position of the item's leading edge. It is
    /// defined for `index` in `0..=item_count`; `offset_of(item_count)` returns
    /// the position just past the final gap.
    #[must_use]
    pub fn offset_of(&self, index: usize) -> f32 {
        index as f32 * self.stride()
    }

    /// The main-axis offset of the trailing edge of the item at `index`.
    #[must_use]
    pub fn end_of(&self, index: usize) -> f32 {
        self.offset_of(index) + self.item_height
    }

    /// The half-open range of item indices intersecting `viewport`, widened by
    /// the viewport's overscan and clamped to `0..item_count`.
    ///
    /// A viewport with a non-positive [`Viewport::length`] yields an empty
    /// range anchored at the clamped first visible index.
    #[must_use]
    pub fn visible_range(&self, viewport: &Viewport) -> Range<usize> {
        if self.item_count == 0 {
            return 0..0;
        }
        let stride = self.stride();
        if stride <= 0.0 {
            // Degenerate: every item sits at offset zero, so all are "visible".
            return 0..self.item_count;
        }

        let start_px = viewport.start();
        let first = (start_px / stride) as usize;
        let start = first.saturating_sub(viewport.overscan).min(self.item_count);

        if viewport.length <= 0.0 {
            return start..start;
        }

        let last = (viewport.end() / stride) as usize;
        let end = (last + 1 + viewport.overscan).min(self.item_count);
        let start = start.min(end);
        start..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_size_includes_internal_gaps() {
        assert_eq!(FixedList::new(0, 10.0, 2.0).total_size(), 0.0);
        assert_eq!(FixedList::new(1, 10.0, 2.0).total_size(), 10.0);
        // 4 items of 10 with 3 gaps of 2 => 40 + 6.
        assert_eq!(FixedList::new(4, 10.0, 2.0).total_size(), 46.0);
    }

    #[test]
    fn offset_of_is_gap_inclusive() {
        let list = FixedList::new(5, 10.0, 2.0);
        assert_eq!(list.offset_of(0), 0.0);
        assert_eq!(list.offset_of(1), 12.0);
        assert_eq!(list.offset_of(3), 36.0);
        assert_eq!(list.end_of(3), 46.0);
    }

    #[test]
    fn visible_range_clamps_at_start() {
        let list = FixedList::new(100, 20.0, 0.0);
        // Scrolled slightly past the top with overscan larger than available.
        let vp = Viewport::new(10.0, 40.0).with_overscan(5);
        // first = 0, last = 2, end = 2 + 1 + 5 = 8, start clamps to 0.
        assert_eq!(list.visible_range(&vp), 0..8);
    }

    #[test]
    fn visible_range_clamps_at_end() {
        let list = FixedList::new(10, 20.0, 0.0);
        // total = 200; scroll near the bottom.
        let vp = Viewport::new(185.0, 40.0).with_overscan(3);
        // last item index 9, end clamps to 10.
        let r = list.visible_range(&vp);
        assert_eq!(r.end, 10);
        assert!(r.start <= r.end);
    }

    #[test]
    fn visible_range_applies_overscan_both_sides() {
        let list = FixedList::new(100, 20.0, 0.0);
        let vp = Viewport::new(200.0, 80.0).with_overscan(2);
        // first = 10, last = 14; overscan 2 => 8..17.
        assert_eq!(list.visible_range(&vp), 8..17);
    }

    #[test]
    fn empty_and_degenerate_lists() {
        assert_eq!(
            FixedList::new(0, 20.0, 0.0).visible_range(&Viewport::new(0.0, 50.0)),
            0..0
        );
        // Zero stride: everything shares offset zero.
        assert_eq!(
            FixedList::new(3, 0.0, 0.0).visible_range(&Viewport::new(0.0, 50.0)),
            0..3
        );
    }

    #[test]
    fn zero_length_viewport_is_empty() {
        let list = FixedList::new(100, 20.0, 0.0);
        let r = list.visible_range(&Viewport::new(100.0, 0.0));
        assert!(r.is_empty());
        assert_eq!(r.start, 5);
    }
}
