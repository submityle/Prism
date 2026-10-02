//! Mapping viewport visibility onto scheduler lanes.
//!
//! The virtualization layer (`prism_ui_virtual`) already computes which item
//! indices are on screen and which sit in the overscan prefetch band. This
//! module turns that geometry into [`Lane`] assignments so a budgeted list
//! build naturally services visible rows first, warms the overscan band next,
//! and only speculatively touches far-off-screen rows when the frame has slack
//! left — exactly the priority ordering §9.5 calls for.
//!
//! It depends on nothing but [`core::ops::Range`], so it cooperates with the
//! virtualization crate by value (passing it a visible range) without coupling
//! the two crates at the type level.

use core::ops::Range;

use crate::lane::Lane;

/// A viewport partition: the on-screen range plus a symmetric overscan band.
///
/// `visible` is the range of indices currently inside the viewport; `overscan`
/// is how many extra rows on each side form the warm-up band. Indices beyond
/// that band are treated as idle, speculative work.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VisibilityWindow {
    visible: Range<usize>,
    overscan: usize,
}

impl VisibilityWindow {
    /// Builds a window from the `visible` range and an `overscan` row count.
    ///
    /// An inverted `visible` range (start past end) is normalized to an empty
    /// range at `start`, so a degenerate viewport yields no visible lane rather
    /// than a reversed span.
    #[must_use]
    pub fn new(visible: Range<usize>, overscan: usize) -> VisibilityWindow {
        let visible = if visible.start > visible.end {
            visible.start..visible.start
        } else {
            visible
        };
        VisibilityWindow { visible, overscan }
    }

    /// The on-screen index range.
    #[must_use]
    pub fn visible(&self) -> Range<usize> {
        self.visible.clone()
    }

    /// The overscan band width in rows per side.
    #[must_use]
    pub const fn overscan(&self) -> usize {
        self.overscan
    }

    /// The index range covered by the visible range widened by the overscan.
    #[must_use]
    pub fn warm_range(&self) -> Range<usize> {
        let start = self.visible.start.saturating_sub(self.overscan);
        let end = self.visible.end.saturating_add(self.overscan);
        start..end
    }

    /// The lane an item at `index` should build on.
    ///
    /// Visible rows map to [`Lane::Visible`], overscan rows to
    /// [`Lane::Offscreen`], and anything further out to [`Lane::Idle`].
    #[must_use]
    pub fn lane_for(&self, index: usize) -> Lane {
        if self.visible.contains(&index) {
            Lane::Visible
        } else if self.warm_range().contains(&index) {
            Lane::Offscreen
        } else {
            Lane::Idle
        }
    }
}
