//! Nine-slice (stretchable) rect splitting.
//!
//! A nine-slice asset — a panel, button, or frame with fixed corners and edges
//! that stretch — is the standard way to draw a resizable UI element from one
//! small texture (the CSS `border-image`, the classic 9-patch). Given the
//! source region, the corner [`Edges`], and the destination rect,
//! [`nine_slice`] computes the nine source tiles and their nine destination
//! tiles: the four corners keep the inset size, the four edges stretch along one
//! axis, and the center stretches along both.
//!
//! The two tile arrays exactly *tile* their rects — the nine destinations
//! partition the destination with no gaps or overlaps — which the tests assert
//! as an independent property. Insets larger than the rect are clamped so no tile
//! ever gets a negative size.

use prism_ui_layout::{Edges, Point, Rect, Size};

/// Index of the top-left corner tile.
pub const TOP_LEFT: usize = 0;
/// Index of the top edge tile.
pub const TOP: usize = 1;
/// Index of the top-right corner tile.
pub const TOP_RIGHT: usize = 2;
/// Index of the left edge tile.
pub const LEFT: usize = 3;
/// Index of the center tile.
pub const CENTER: usize = 4;
/// Index of the right edge tile.
pub const RIGHT: usize = 5;
/// Index of the bottom-left corner tile.
pub const BOTTOM_LEFT: usize = 6;
/// Index of the bottom edge tile.
pub const BOTTOM: usize = 7;
/// Index of the bottom-right corner tile.
pub const BOTTOM_RIGHT: usize = 8;

/// The nine tiles of a nine-slice draw, in row-major order (see the index
/// constants). `source[i]` and `dest[i]` are the two rects of tile `i`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NineSlice {
    /// The tiles within the source (atlas) region.
    pub source: [Rect; 9],
    /// The tiles within the destination region.
    pub dest: [Rect; 9],
}

/// A finite, non-negative inset, or `0` for anything else.
fn sanitize(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Split `rect` into the nine cells described by `insets`, in row-major order.
///
/// Insets are clamped non-negative and so that opposite insets never exceed the
/// rect (a rect narrower than `left + right` gives the middle a zero width
/// rather than a negative one). A `NaN` or non-finite inset is treated as `0`.
#[must_use]
pub fn split_nine(rect: Rect, insets: Edges<f32>) -> [Rect; 9] {
    let w = rect.size.width.max(0.0);
    let h = rect.size.height.max(0.0);
    let left = sanitize(insets.left).min(w);
    let right = sanitize(insets.right).min(w - left);
    let top = sanitize(insets.top).min(h);
    let bottom = sanitize(insets.bottom).min(h - top);

    let x = [
        rect.left(),
        rect.left() + left,
        rect.right() - right,
        rect.right(),
    ];
    let y = [
        rect.top(),
        rect.top() + top,
        rect.bottom() - bottom,
        rect.bottom(),
    ];

    let mut out = [Rect::new(Point::new(0.0, 0.0), Size::new(0.0, 0.0)); 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] = Rect::new(
                Point::new(x[col], y[row]),
                Size::new(x[col + 1] - x[col], y[row + 1] - y[row]),
            );
        }
    }
    out
}

/// Compute the nine-slice tiles for drawing `source` into `dest` with corner
/// `insets` (a single inset set, applied to both rects — corners keep their size
/// on both). See the index constants for the tile order.
#[must_use]
pub fn nine_slice(source: Rect, insets: Edges<f32>, dest: Rect) -> NineSlice {
    NineSlice {
        source: split_nine(source, insets),
        dest: split_nine(dest, insets),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::new(Point::new(x, y), Size::new(w, h))
    }

    fn area(r: Rect) -> f32 {
        r.size.width.max(0.0) * r.size.height.max(0.0)
    }

    fn insets(v: f32) -> Edges<f32> {
        Edges::new(v, v, v, v)
    }

    #[test]
    fn uniform_source_splits_into_nine_equal_cells() {
        let tiles = split_nine(rect(0.0, 0.0, 90.0, 90.0), insets(30.0));
        for tile in tiles {
            assert_eq!(tile.size.width, 30.0);
            assert_eq!(tile.size.height, 30.0);
        }
        // Row-major placement.
        assert_eq!(tiles[TOP_LEFT].location, Point::new(0.0, 0.0));
        assert_eq!(tiles[CENTER].location, Point::new(30.0, 30.0));
        assert_eq!(tiles[BOTTOM_RIGHT].location, Point::new(60.0, 60.0));
    }

    #[test]
    fn corners_keep_their_size_and_the_center_stretches() {
        let slice = nine_slice(
            rect(0.0, 0.0, 90.0, 90.0),
            insets(30.0),
            rect(0.0, 0.0, 150.0, 150.0),
        );
        // Corners are 30x30 on both sides.
        assert_eq!(slice.dest[TOP_LEFT].size, Size::new(30.0, 30.0));
        assert_eq!(slice.dest[BOTTOM_RIGHT].size, Size::new(30.0, 30.0));
        // The top edge stretches horizontally only.
        assert_eq!(slice.dest[TOP].size, Size::new(90.0, 30.0));
        // The left edge stretches vertically only.
        assert_eq!(slice.dest[LEFT].size, Size::new(30.0, 90.0));
        // The center stretches both ways.
        assert_eq!(slice.dest[CENTER].size, Size::new(90.0, 90.0));
    }

    #[test]
    fn tiles_exactly_partition_each_rect() {
        let source = rect(12.0, 40.0, 128.0, 64.0);
        let dest = rect(-5.0, 3.0, 400.0, 220.0);
        let slice = nine_slice(source, insets(16.0), dest);
        for (tiles, whole) in [(slice.source, source), (slice.dest, dest)] {
            let total: f32 = tiles.iter().map(|t| area(*t)).sum();
            assert!(
                (total - area(whole)).abs() < 1e-3,
                "tiles sum to {total}, rect is {}",
                area(whole)
            );
        }
    }

    #[test]
    fn oversized_insets_clamp_without_negative_sizes() {
        // A 20x20 rect with 30-unit insets: the middle collapses to zero and no
        // tile may have a negative extent.
        let tiles = split_nine(rect(0.0, 0.0, 20.0, 20.0), insets(30.0));
        for tile in tiles {
            assert!(tile.size.width >= 0.0 && tile.size.height >= 0.0);
        }
        let total: f32 = tiles.iter().map(|t| area(*t)).sum();
        assert!((total - 400.0).abs() < 1e-3);
        // Left takes the whole width; right collapses to zero.
        assert_eq!(tiles[TOP_LEFT].size.width, 20.0);
        assert_eq!(tiles[TOP_RIGHT].size.width, 0.0);
    }

    #[test]
    fn zero_insets_leave_only_the_center() {
        let tiles = split_nine(rect(0.0, 0.0, 10.0, 10.0), insets(0.0));
        assert_eq!(tiles[CENTER], rect(0.0, 0.0, 10.0, 10.0));
        for (i, tile) in tiles.iter().enumerate() {
            if i != CENTER {
                assert_eq!(area(*tile), 0.0, "tile {i} should be empty");
            }
        }
    }

    #[test]
    fn non_finite_insets_are_treated_as_zero() {
        let tiles = split_nine(
            rect(0.0, 0.0, 10.0, 10.0),
            Edges::new(f32::NAN, -4.0, f32::INFINITY, 0.0),
        );
        assert_eq!(tiles[CENTER], rect(0.0, 0.0, 10.0, 10.0));
    }
}
