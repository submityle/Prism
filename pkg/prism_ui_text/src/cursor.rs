//! Caret, selection and IME-composition models plus glyph hit-testing.
//!
//! Positions are expressed as byte offsets into the source text; the helpers
//! translate between offsets and the horizontal pixel coordinates reported by a
//! [`ShapedRun`]. All arithmetic uses `+ - * /` only.

use crate::segmentation;
use crate::shaper::ShapedRun;
use alloc::vec::Vec;

/// A caret: a byte offset into the text and its horizontal pixel position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Caret {
    /// Byte offset of the caret within the source text.
    pub offset: usize,
    /// Horizontal position of the caret in logical pixels.
    pub x: f32,
}

impl Caret {
    /// Builds a caret from an offset and a horizontal position.
    #[must_use]
    pub fn new(offset: usize, x: f32) -> Self {
        Self { offset, x }
    }
}

/// Returns the caret positions for every cluster boundary in `run`.
///
/// The first caret sits at offset `0`/x `0`; each subsequent caret sits at the
/// cumulative advance after the matching glyph, and a final caret marks the end
/// of the text.
#[must_use]
pub fn caret_positions(run: &ShapedRun) -> Vec<Caret> {
    let mut carets = Vec::new();
    let mut x = 0.0f32;
    for glyph in &run.glyphs {
        carets.push(Caret::new(glyph.cluster, x));
        x += glyph.advance;
    }
    carets.push(Caret::new(run.text_len, x));
    carets
}

/// Returns the horizontal pixel position of the caret at byte `offset`.
///
/// Advances are accumulated up to the first glyph whose cluster is at or after
/// `offset`; offsets beyond the run return the full width.
#[must_use]
pub fn x_for_offset(run: &ShapedRun, offset: usize) -> f32 {
    let mut x = 0.0f32;
    for glyph in &run.glyphs {
        if glyph.cluster >= offset {
            return x;
        }
        x += glyph.advance;
    }
    x
}

/// Returns the byte offset of the cluster nearest to horizontal position `x`.
///
/// A click lands on a cluster when `x` is left of that cluster's horizontal
/// midpoint; otherwise it falls through to the next cluster, and past the final
/// midpoint it resolves to the end of the text.
#[must_use]
pub fn hit_test(run: &ShapedRun, x: f32) -> usize {
    let mut left = 0.0f32;
    for glyph in &run.glyphs {
        let mid = left + glyph.advance / 2.0;
        if x < mid {
            return glyph.cluster;
        }
        left += glyph.advance;
    }
    run.text_len
}

/// A text selection defined by an anchor and a focus byte offset.
///
/// The anchor is the fixed end (where selection began) and the focus is the
/// moving end (where the caret currently is); either may be the smaller offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    /// Fixed end of the selection.
    pub anchor: usize,
    /// Moving end of the selection.
    pub focus: usize,
}

impl Selection {
    /// Builds a selection from an anchor and a focus offset.
    #[must_use]
    pub fn new(anchor: usize, focus: usize) -> Self {
        Self { anchor, focus }
    }

    /// Builds a collapsed selection (empty, caret-only) at `offset`.
    #[must_use]
    pub fn caret(offset: usize) -> Self {
        Self {
            anchor: offset,
            focus: offset,
        }
    }

    /// Returns `true` when the selection is empty (anchor equals focus).
    #[must_use]
    pub fn is_collapsed(&self) -> bool {
        self.anchor == self.focus
    }

    /// Returns the lower of the two offsets.
    #[must_use]
    pub fn start(&self) -> usize {
        self.anchor.min(self.focus)
    }

    /// Returns the higher of the two offsets.
    #[must_use]
    pub fn end(&self) -> usize {
        self.anchor.max(self.focus)
    }

    /// Returns the selected length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end() - self.start()
    }

    /// Returns `true` when the selection covers no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An in-progress IME composition (preedit) region with its own caret.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Composition {
    /// Byte offset of the first byte of the preedit region.
    pub start: usize,
    /// Byte offset one past the final byte of the preedit region.
    pub end: usize,
    /// Caret offset within the preedit region, in bytes from `start`.
    pub caret: usize,
}

impl Composition {
    /// Builds a composition region with a caret offset relative to `start`.
    #[must_use]
    pub fn new(start: usize, end: usize, caret: usize) -> Self {
        Self { start, end, caret }
    }

    /// Returns the length of the preedit region in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` when the preedit region is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Returns `true` when `offset` falls within the preedit region.
    #[must_use]
    pub fn contains(&self, offset: usize) -> bool {
        offset >= self.start && offset < self.end
    }
}

/// Moves `offset` one grapheme cluster to the left within `text`.
#[must_use]
pub fn move_left(text: &str, offset: usize) -> usize {
    segmentation::prev_grapheme_boundary(text, offset)
}

/// Moves `offset` one grapheme cluster to the right within `text`.
#[must_use]
pub fn move_right(text: &str, offset: usize) -> usize {
    segmentation::next_grapheme_boundary(text, offset)
}

/// Returns the offset of the start of the text (always `0`).
#[must_use]
pub fn move_line_start(_text: &str) -> usize {
    0
}

/// Returns the offset of the end of `text`.
#[must_use]
pub fn move_line_end(text: &str) -> usize {
    text.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rich_text::TextStyle;
    use crate::shaper::{MetricShaper, Shaper};

    fn run(text: &str) -> ShapedRun {
        MetricShaper::default().shape(text, &TextStyle::default())
    }

    #[test]
    fn caret_positions_bracket_text() {
        let r = run("abc");
        let carets = caret_positions(&r);
        assert_eq!(carets.len(), 4);
        assert_eq!(carets[0].x, 0.0);
        assert_eq!(carets[0].offset, 0);
        assert_eq!(carets[3].offset, 3);
        assert_eq!(carets[3].x, r.width());
    }

    #[test]
    fn x_for_offset_is_monotonic() {
        let r = run("abc");
        assert_eq!(x_for_offset(&r, 0), 0.0);
        assert!(x_for_offset(&r, 1) < x_for_offset(&r, 2));
        assert_eq!(x_for_offset(&r, 99), r.width());
    }

    #[test]
    fn hit_test_round_trips_offsets() {
        let r = run("abc");
        assert_eq!(hit_test(&r, 0.0), 0);
        assert_eq!(hit_test(&r, r.width()), 3);
        // A point just past the first glyph midpoint lands on cluster 1.
        let first = r.glyphs[0].advance;
        assert_eq!(hit_test(&r, first * 0.75), 1);
    }

    #[test]
    fn selection_ordering() {
        let sel = Selection::new(5, 2);
        assert_eq!(sel.start(), 2);
        assert_eq!(sel.end(), 5);
        assert_eq!(sel.len(), 3);
        assert!(!sel.is_collapsed());
        assert!(!sel.is_empty());
    }

    #[test]
    fn collapsed_selection_is_empty() {
        let sel = Selection::caret(4);
        assert!(sel.is_collapsed());
        assert!(sel.is_empty());
        assert_eq!(sel.len(), 0);
    }

    #[test]
    fn composition_region() {
        let c = Composition::new(2, 6, 1);
        assert_eq!(c.len(), 4);
        assert!(!c.is_empty());
        assert!(c.contains(3));
        assert!(!c.contains(6));
    }

    #[test]
    fn movement_respects_clusters() {
        let text = "e\u{0301}z";
        assert_eq!(move_right(text, 0), 3);
        assert_eq!(move_left(text, 3), 0);
        assert_eq!(move_line_start(text), 0);
        assert_eq!(move_line_end(text), text.len());
    }
}
