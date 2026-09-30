//! `Cohen-Sutherland` line-segment clipping against an axis-aligned rectangular
//! window for the particle screen-space and 2D-culling passes (design §12,
//! §13).
//!
//! Several particle stages need to trim a 2D segment to the visible extent of a
//! rectangular region: a trail renderer clips a ribbon spine to the scissor
//! rect, a 2D broadphase drops the part of a swept edge that leaves a tile, and
//! an authoring overlay crops guide lines to a viewport. This module owns the
//! small, `CPU`-verifiable reference for that operation so a future `GPU` kernel
//! can reproduce it bit for bit.
//!
//! The algorithm is the classic `Cohen-Sutherland` clip. Each endpoint is
//! tagged with a 4-bit *outcode* recording which of the four half-planes
//! (`left`, `right`, `bottom`, `top`) the point lies outside of. The loop then
//! decides in constant work per step: when both outcodes are zero the whole
//! segment is inside and is accepted; when the bitwise `AND` of the two
//! outcodes is non-zero both endpoints share an outside half-plane and the
//! segment is trivially rejected; otherwise one still-outside endpoint is
//! replaced by its intersection with the offending window edge and the loop
//! repeats. Because every step clears at least one outcode bit, the loop
//! terminates after at most four edge clips.
//!
//! # Strict scope
//! This module clips *one 2D segment against one axis-aligned rectangle* and
//! returns the surviving sub-segment. It is deliberately distinct from its
//! neighbours:
//! * [`super::plane_clip`] runs the `Sutherland-Hodgman` convex-polygon clip
//!   against an oriented half-space plane in 3D; that is a different algorithm
//!   over a different object (a polygon against a plane, not a segment against a
//!   box) and this module neither imports nor reconstructs it.
//! * [`super::segment_intersect_2d`] only *classifies* whether two 2D segments
//!   cross and where; it never trims a segment to a region.
//!
//! # No transcendental math
//! Outcodes are pure comparisons and edge intersections use a linear parameter
//! evaluated with `+`, `-`, `*`, `/` only. There is no `sin`, `cos`, `atan`,
//! `exp`, `ln`, `powf`, `sqrt`, or any other transcendental call. `f32`
//! magnitudes are never compared with `==`/`!=`: boundary tolerance goes
//! through [`CMP_EPS`], and ordinary `<`/`>` ordering drives the outcode bits.
//!
//! # Layout
//! [`gpu_storage_bytes`] sizes a `std430` storage buffer holding the clipped
//! endpoints as two `vec2<f32>` slots per segment, matching the shared
//! [`crate::particle::gpu_layout`] stride so a `WebGPU` kernel binds a stable
//! `ABI`.

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// Magnitude below which a coordinate difference against a window edge is
/// treated as zero. Used instead of `==` on `f32`: a point within this band of
/// an edge counts as on the inside of that edge rather than outside it.
pub const CMP_EPS: f32 = 1.0e-6;

/// Outcode value for a point inside every window half-plane (no bits set).
pub const OUTCODE_INSIDE: u8 = 0;

/// Outcode bit set when a point lies to the left of the window (`x < xmin`).
pub const OUTCODE_LEFT: u8 = 0b0001;

/// Outcode bit set when a point lies to the right of the window (`x > xmax`).
pub const OUTCODE_RIGHT: u8 = 0b0010;

/// Outcode bit set when a point lies below the window (`y < ymin`).
pub const OUTCODE_BOTTOM: u8 = 0b0100;

/// Outcode bit set when a point lies above the window (`y > ymax`).
pub const OUTCODE_TOP: u8 = 0b1000;

/// An axis-aligned rectangular clip window `[xmin, xmax] x [ymin, ymax]`.
///
/// The bounds are stored as supplied; [`ClipRect::normalized`] swaps any
/// inverted pair so `min <= max` holds before clipping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipRect {
    /// Lower `x` bound of the window.
    pub xmin: f32,
    /// Lower `y` bound of the window.
    pub ymin: f32,
    /// Upper `x` bound of the window.
    pub xmax: f32,
    /// Upper `y` bound of the window.
    pub ymax: f32,
}

impl ClipRect {
    /// Builds a window from its four bounds without reordering them.
    #[must_use]
    pub const fn new(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Self {
        Self {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }

    /// Returns an equivalent window with `xmin <= xmax` and `ymin <= ymax`.
    ///
    /// Callers that might pass an inverted rectangle can normalize first so the
    /// outcode edges are oriented consistently.
    #[must_use]
    pub fn normalized(&self) -> ClipRect {
        ClipRect {
            xmin: self.xmin.min(self.xmax),
            ymin: self.ymin.min(self.ymax),
            xmax: self.xmin.max(self.xmax),
            ymax: self.ymin.max(self.ymax),
        }
    }

    /// Computes the 4-bit `Cohen-Sutherland` outcode of a point.
    ///
    /// A coordinate within [`CMP_EPS`] of an edge is treated as on the inside of
    /// that edge, so a point exactly on the boundary yields [`OUTCODE_INSIDE`]
    /// for that axis.
    #[must_use]
    pub fn outcode(&self, p: [f32; 2]) -> u8 {
        let mut code = OUTCODE_INSIDE;
        if p[0] < self.xmin - CMP_EPS {
            code |= OUTCODE_LEFT;
        } else if p[0] > self.xmax + CMP_EPS {
            code |= OUTCODE_RIGHT;
        }
        if p[1] < self.ymin - CMP_EPS {
            code |= OUTCODE_BOTTOM;
        } else if p[1] > self.ymax + CMP_EPS {
            code |= OUTCODE_TOP;
        }
        code
    }

    /// Returns `true` when the point lies inside the window (outcode zero).
    #[must_use]
    pub fn contains(&self, p: [f32; 2]) -> bool {
        self.outcode(p) == OUTCODE_INSIDE
    }
}

/// Clips segment `a -> b` to the rectangular window and returns the surviving
/// sub-segment, or `None` when the segment lies wholly outside.
///
/// The returned endpoints keep the original travel direction: the point derived
/// from `a` comes first and the point derived from `b` comes second, so a
/// caller can preserve per-endpoint attributes by side. A segment already
/// inside is returned unchanged; a degenerate zero-length segment survives only
/// when its single point is inside the window.
#[must_use]
pub fn clip_segment(rect: &ClipRect, a: [f32; 2], b: [f32; 2]) -> Option<([f32; 2], [f32; 2])> {
    let mut pa = a;
    let mut pb = b;
    let mut code_a = rect.outcode(pa);
    let mut code_b = rect.outcode(pb);

    loop {
        if (code_a | code_b) == OUTCODE_INSIDE {
            // Both endpoints are inside every half-plane: accept the whole
            // (possibly already-trimmed) segment.
            return Some((pa, pb));
        }
        if (code_a & code_b) != OUTCODE_INSIDE {
            // Both endpoints share an outside half-plane: the segment cannot
            // enter the window, so reject it.
            return None;
        }

        // At least one endpoint is outside and they do not share a region;
        // clip the outside endpoint against one offending edge.
        let out_code = if code_a != OUTCODE_INSIDE {
            code_a
        } else {
            code_b
        };
        let clipped = edge_intersection(rect, pa, pb, out_code);

        if out_code == code_a {
            pa = clipped;
            code_a = rect.outcode(pa);
        } else {
            pb = clipped;
            code_b = rect.outcode(pb);
        }
    }
}

/// Intersects segment `a -> b` with the single window edge named by the highest
/// relevant bit of `out_code`, returning the crossing point.
///
/// The crossing uses a linear parameter along the segment, so the math stays
/// affine (`+ - * /`). Because `out_code` marks an endpoint that is strictly
/// outside the chosen edge while the other endpoint is not, the paired
/// coordinate difference in the denominator is non-zero, so the division is
/// well defined.
fn edge_intersection(rect: &ClipRect, a: [f32; 2], b: [f32; 2], out_code: u8) -> [f32; 2] {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];

    if (out_code & OUTCODE_TOP) != 0 {
        [a[0] + dx * (rect.ymax - a[1]) / dy, rect.ymax]
    } else if (out_code & OUTCODE_BOTTOM) != 0 {
        [a[0] + dx * (rect.ymin - a[1]) / dy, rect.ymin]
    } else if (out_code & OUTCODE_RIGHT) != 0 {
        [rect.xmax, a[1] + dy * (rect.xmax - a[0]) / dx]
    } else {
        // The remaining case is the left edge; a non-inside `out_code` always
        // has at least one of the four bits set.
        [rect.xmin, a[1] + dy * (rect.xmin - a[0]) / dx]
    }
}

/// Total byte size of a `std430` storage buffer holding `count` clipped
/// segments, each stored as two `vec2<f32>` endpoint slots.
///
/// A `WebGPU` storage binding may not be zero-sized, so an empty batch still
/// reserves one element (see [`crate::particle::gpu_layout::storage_bytes`]).
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(VEC2_STRIDE, count.saturating_mul(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: ClipRect = ClipRect::new(0.0, 0.0, 10.0, 10.0);

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    fn points_close(p: [f32; 2], q: [f32; 2]) -> bool {
        approx(p[0], q[0]) && approx(p[1], q[1])
    }

    #[test]
    fn fully_inside_segment_is_returned_unchanged() {
        let (a, b) = clip_segment(&RECT, [2.0, 3.0], [7.0, 8.0]).expect("inside survives");
        assert!(points_close(a, [2.0, 3.0]));
        assert!(points_close(b, [7.0, 8.0]));
    }

    #[test]
    fn fully_outside_left_is_rejected() {
        assert!(clip_segment(&RECT, [-5.0, 2.0], [-1.0, 8.0]).is_none());
    }

    #[test]
    fn fully_outside_right_is_rejected() {
        assert!(clip_segment(&RECT, [11.0, 2.0], [15.0, 8.0]).is_none());
    }

    #[test]
    fn fully_outside_above_is_rejected() {
        assert!(clip_segment(&RECT, [2.0, 11.0], [8.0, 20.0]).is_none());
    }

    #[test]
    fn fully_outside_below_is_rejected() {
        assert!(clip_segment(&RECT, [2.0, -11.0], [8.0, -1.0]).is_none());
    }

    #[test]
    fn crossing_left_edge_clips_entry_point() {
        let (a, b) = clip_segment(&RECT, [-5.0, 5.0], [5.0, 5.0]).expect("enters window");
        assert!(points_close(a, [0.0, 5.0]));
        assert!(points_close(b, [5.0, 5.0]));
    }

    #[test]
    fn crossing_both_horizontal_edges_clips_to_span() {
        let (a, b) = clip_segment(&RECT, [-5.0, 5.0], [15.0, 5.0]).expect("spans window");
        assert!(points_close(a, [0.0, 5.0]));
        assert!(points_close(b, [10.0, 5.0]));
    }

    #[test]
    fn crossing_both_vertical_edges_clips_to_span() {
        let (a, b) = clip_segment(&RECT, [4.0, -5.0], [4.0, 15.0]).expect("spans window");
        assert!(points_close(a, [4.0, 0.0]));
        assert!(points_close(b, [4.0, 10.0]));
    }

    #[test]
    fn one_endpoint_inside_one_outside_top() {
        let (a, b) = clip_segment(&RECT, [5.0, 5.0], [5.0, 15.0]).expect("exits top");
        assert!(points_close(a, [5.0, 5.0]));
        assert!(points_close(b, [5.0, 10.0]));
    }

    #[test]
    fn diagonal_crossing_clips_two_opposite_corners() {
        let (a, b) = clip_segment(&RECT, [-5.0, -5.0], [15.0, 15.0]).expect("crosses diagonally");
        assert!(points_close(a, [0.0, 0.0]));
        assert!(points_close(b, [10.0, 10.0]));
    }

    #[test]
    fn horizontal_interior_line_is_unchanged() {
        let (a, b) = clip_segment(&RECT, [1.0, 6.0], [9.0, 6.0]).expect("interior line");
        assert!(points_close(a, [1.0, 6.0]));
        assert!(points_close(b, [9.0, 6.0]));
    }

    #[test]
    fn vertical_interior_line_is_unchanged() {
        let (a, b) = clip_segment(&RECT, [3.0, 1.0], [3.0, 9.0]).expect("interior line");
        assert!(points_close(a, [3.0, 1.0]));
        assert!(points_close(b, [3.0, 9.0]));
    }

    #[test]
    fn endpoint_exactly_on_boundary_is_accepted() {
        let (a, b) = clip_segment(&RECT, [0.0, 5.0], [6.0, 5.0]).expect("boundary endpoint");
        assert!(points_close(a, [0.0, 5.0]));
        assert!(points_close(b, [6.0, 5.0]));
    }

    #[test]
    fn segment_lying_on_top_edge_is_accepted() {
        let (a, b) = clip_segment(&RECT, [2.0, 10.0], [8.0, 10.0]).expect("edge-coincident");
        assert!(points_close(a, [2.0, 10.0]));
        assert!(points_close(b, [8.0, 10.0]));
    }

    #[test]
    fn degenerate_point_inside_survives() {
        let (a, b) = clip_segment(&RECT, [5.0, 5.0], [5.0, 5.0]).expect("point inside");
        assert!(points_close(a, [5.0, 5.0]));
        assert!(points_close(b, [5.0, 5.0]));
    }

    #[test]
    fn degenerate_point_outside_is_rejected() {
        assert!(clip_segment(&RECT, [20.0, 20.0], [20.0, 20.0]).is_none());
    }

    #[test]
    fn corner_to_corner_crossing_from_bottom_left_to_top_right() {
        // Both endpoints sit outside in opposite corner regions but the line
        // threads the window. It enters through the left edge and leaves
        // through the top edge.
        let (a, b) = clip_segment(&RECT, [-2.0, -1.0], [12.0, 13.0]).expect("threads window");
        assert!(points_close(a, [0.0, 1.0]));
        assert!(points_close(b, [9.0, 10.0]));
        // Both survivors land on the window boundary within its extent.
        assert!(a[0] >= 0.0 - CMP_EPS && a[0] <= 10.0 + CMP_EPS);
        assert!(b[1] >= 0.0 - CMP_EPS && b[1] <= 10.0 + CMP_EPS);
    }

    #[test]
    fn four_outside_corner_endpoints_thread_window() {
        // A line from the outside-left to the outside-upper-right that clips
        // against the left edge, then the top edge, then the right edge.
        let (a, b) = clip_segment(&RECT, [-5.0, 2.0], [15.0, 12.0]).expect("threads window");
        assert!(points_close(a, [0.0, 4.5]));
        assert!(points_close(b, [10.0, 9.5]));
    }

    #[test]
    fn intersection_coordinates_are_exact_on_a_slope() {
        // Line y = x from (-4,-4) to (14,14) clips to the main diagonal of the
        // window with exact integer crossings.
        let (a, b) = clip_segment(&RECT, [-4.0, -4.0], [14.0, 14.0]).expect("diagonal");
        assert!(approx(a[0], 0.0) && approx(a[1], 0.0));
        assert!(approx(b[0], 10.0) && approx(b[1], 10.0));
    }

    #[test]
    fn intersection_on_shallow_slope_is_exact() {
        // From (-10, 2) to (10, 6): slope 0.2. At x = 0, y = 4.
        let (a, b) = clip_segment(&RECT, [-10.0, 2.0], [10.0, 6.0]).expect("shallow slope");
        assert!(approx(a[0], 0.0));
        assert!(approx(a[1], 4.0));
        assert!(points_close(b, [10.0, 6.0]));
    }

    #[test]
    fn swapping_endpoints_yields_the_same_visible_segment() {
        let forward = clip_segment(&RECT, [-5.0, 5.0], [15.0, 5.0]).expect("forward");
        let reverse = clip_segment(&RECT, [15.0, 5.0], [-5.0, 5.0]).expect("reverse");
        // The direction flips, so the reversed clip's first point matches the
        // forward clip's second point and vice versa.
        assert!(points_close(reverse.0, forward.1));
        assert!(points_close(reverse.1, forward.0));
    }

    #[test]
    fn clipping_is_translation_invariant() {
        let a = [-5.0, 5.0];
        let b = [5.0, 5.0];
        let (base_a, base_b) = clip_segment(&RECT, a, b).expect("base clip");

        let shift = [3.0, -2.0];
        let shifted_rect = ClipRect::new(
            RECT.xmin + shift[0],
            RECT.ymin + shift[1],
            RECT.xmax + shift[0],
            RECT.ymax + shift[1],
        );
        let (moved_a, moved_b) = clip_segment(
            &shifted_rect,
            [a[0] + shift[0], a[1] + shift[1]],
            [b[0] + shift[0], b[1] + shift[1]],
        )
        .expect("shifted clip");

        assert!(points_close(
            moved_a,
            [base_a[0] + shift[0], base_a[1] + shift[1]]
        ));
        assert!(points_close(
            moved_b,
            [base_b[0] + shift[0], base_b[1] + shift[1]]
        ));
    }

    #[test]
    fn outcode_inside_point_is_zero() {
        assert_eq!(RECT.outcode([5.0, 5.0]), OUTCODE_INSIDE);
    }

    #[test]
    fn outcode_left_only() {
        assert_eq!(RECT.outcode([-1.0, 5.0]), OUTCODE_LEFT);
    }

    #[test]
    fn outcode_right_only() {
        assert_eq!(RECT.outcode([11.0, 5.0]), OUTCODE_RIGHT);
    }

    #[test]
    fn outcode_bottom_only() {
        assert_eq!(RECT.outcode([5.0, -1.0]), OUTCODE_BOTTOM);
    }

    #[test]
    fn outcode_top_only() {
        assert_eq!(RECT.outcode([5.0, 11.0]), OUTCODE_TOP);
    }

    #[test]
    fn outcode_combines_corner_bits() {
        assert_eq!(RECT.outcode([-1.0, -1.0]), OUTCODE_LEFT | OUTCODE_BOTTOM);
        assert_eq!(RECT.outcode([11.0, 11.0]), OUTCODE_RIGHT | OUTCODE_TOP);
    }

    #[test]
    fn shared_left_region_is_rejected_without_clipping() {
        // Both endpoints strictly left of the window: bitwise AND is non-zero.
        assert!(clip_segment(&RECT, [-3.0, 1.0], [-1.0, 9.0]).is_none());
    }

    #[test]
    fn point_within_epsilon_of_edge_counts_as_inside() {
        let p = [10.0 + CMP_EPS * 0.5, 5.0];
        assert_eq!(RECT.outcode(p), OUTCODE_INSIDE);
        assert!(RECT.contains(p));
    }

    #[test]
    fn normalized_swaps_inverted_bounds() {
        let inverted = ClipRect::new(10.0, 10.0, 0.0, 0.0);
        let fixed = inverted.normalized();
        assert!(approx(fixed.xmin, 0.0) && approx(fixed.ymin, 0.0));
        assert!(approx(fixed.xmax, 10.0) && approx(fixed.ymax, 10.0));
    }

    #[test]
    fn gpu_storage_bytes_counts_two_endpoints_per_segment() {
        assert_eq!(gpu_storage_bytes(4), storage_bytes(VEC2_STRIDE, 8));
        assert_eq!(gpu_storage_bytes(4), VEC2_STRIDE * 8);
    }

    #[test]
    fn gpu_storage_bytes_empty_is_one_element() {
        assert_eq!(gpu_storage_bytes(0), VEC2_STRIDE);
    }
}
