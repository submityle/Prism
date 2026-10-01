//! The scroll viewport describing the currently visible main-axis window.

/// A scrollable viewport over a list's main axis.
///
/// All measurements are in logical pixels along the list's main axis (the
/// vertical axis for a conventional top-to-bottom list). [`Viewport::length`]
/// is the extent that is actually on screen, while [`Viewport::scroll_offset`]
/// is the distance from the top of the content to the top of the visible
/// window.
///
/// [`Viewport::overscan`] is the number of extra items to materialize on each
/// side of the strictly visible window. Overscan trades a little extra work for
/// smoother scrolling, since items are already built before they scroll into
/// view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Distance from the start of the content to the start of the visible
    /// window, in logical pixels. Negative values are treated as `0`.
    pub scroll_offset: f32,
    /// The visible main-axis extent, in logical pixels.
    pub length: f32,
    /// Extra items to materialize on each side of the visible window.
    pub overscan: usize,
}

impl Viewport {
    /// Creates a viewport with no overscan.
    #[must_use]
    pub const fn new(scroll_offset: f32, length: f32) -> Self {
        Self {
            scroll_offset,
            length,
            overscan: 0,
        }
    }

    /// Returns a copy of this viewport with the given overscan.
    #[must_use]
    pub const fn with_overscan(mut self, overscan: usize) -> Self {
        self.overscan = overscan;
        self
    }

    /// The clamped start of the visible window (never negative).
    #[must_use]
    pub fn start(&self) -> f32 {
        self.scroll_offset.max(0.0)
    }

    /// The clamped end of the visible window (never negative).
    #[must_use]
    pub fn end(&self) -> f32 {
        (self.scroll_offset + self.length).max(0.0)
    }
}
