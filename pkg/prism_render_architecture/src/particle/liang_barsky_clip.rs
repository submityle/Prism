//! `Liang-Barsky` parametric line-segment clipping against an axis-aligned
//! rectangle for the particle screen-space and 2D-culling passes (design §12,
//! §13).
//!
//! Several particle stages must trim a 2D segment to the visible extent of a
//! rectangular region and, unlike a pure accept/reject, also need the *scalar
//! parameter* at which the segment enters and leaves the window: a ribbon
//! renderer interpolates per-vertex attributes (width, color, `UV`) at the clip
//! points, and a trail resampler wants the fractional position `t` along the
//! original span rather than only the clipped endpoints. This module owns the
//! small, `CPU`-verifiable reference for that operation so a future `GPU` kernel
//! can reproduce it bit for bit.
//!
//! The algorithm is the classic `Liang-Barsky` clip. The segment is written in
//! parametric form `P(t) = a + t * (b - a)` for `t` in `[0, 1]`. Each of the
//! four window edges (`left`, `right`, `bottom`, `top`) contributes a pair
//! `(p_k, q_k)` such that the point stays inside that edge exactly when
//! `t * p_k <= q_k`. The loop tracks the entry parameter `t_enter` (initialized
//! to `0`) and the exit parameter `t_exit` (initialized to `1`):
//! * when `p_k < 0` the segment crosses the edge going inward, so
//!   `t_enter = max(t_enter, q_k / p_k)`;
//! * when `p_k > 0` the segment crosses going outward, so
//!   `t_exit = min(t_exit, q_k / p_k)`;
//! * when `p_k` is (within tolerance) zero the segment is parallel to the edge
//!   and lies wholly outside it whenever `q_k < 0`, which is an immediate
//!   reject.
//!
//! When `t_enter > t_exit` the surviving interval is empty and the segment is
//! rejected; otherwise `P(t_enter)` and `P(t_exit)` are the clipped endpoints.
//! Because the four edges are folded into `max`/`min` updates, the whole clip
//! is branch-light and allocation-free.
//!
//! # Degenerate inputs
//! * A **zero-length segment** (a point) has a zero direction, so every
//!   `p_k` is zero. It survives with `t_enter = 0`, `t_exit = 1` only when the
//!   point lies inside all four slabs (all `q_k >= 0`), in which case the single
//!   point is returned for both clip endpoints; otherwise the result is `None`.
//! * A **degenerate rectangle** with `min > max` on an axis is normalized by
//!   swapping that axis so `min <= max` holds before clipping (see
//!   [`ClipRect::normalized`]); this module clips against the normalized window
//!   rather than rejecting inverted input.
//!
//! # Strict scope
//! This module clips *one 2D segment against one axis-aligned rectangle* using
//! the parametric `p`/`q` formulation, and it is deliberately distinct from its
//! neighbours:
//! * [`super::cohen_sutherland_clip`] solves the same segment-versus-rectangle
//!   problem but with the *outcode* bit-code method; it iteratively replaces an
//!   outside endpoint by an edge intersection and never exposes the scalar
//!   parameters. Prefer this module when a caller needs the `t` values (for
//!   attribute interpolation) and that one when only the trimmed endpoints
//!   matter.
//! * [`super::plane_clip`] runs a convex-polygon clip against an oriented 3D
//!   half-space plane; that is a different algorithm over a different object (a
//!   polygon against a plane, not a segment against a box).
//! * [`super::segment_intersect_2d`] only *classifies* whether two 2D segments
//!   cross and where; it never trims a segment to a region.
//!
//! # No transcendental math
//! Edge tests and parameter updates use `+`, `-`, `*`, `/`, `abs`, `max`, and
//! `min` only. There is no `sin`, `cos`, `atan`, `exp`, `ln`, `powf`, `sqrt`,
//! or any other transcendental call. `f32` magnitudes are never compared with
//! `==`/`!=`: the parallel-edge test goes through [`CLIP_EPS`] on the absolute
//! value of `p_k`, and ordinary `<`/`>` ordering drives the interval updates so
//! no `NaN` is ever produced.
//!
//! # Layout
//! [`gpu_storage_bytes`] sizes a `std430` storage buffer holding the clipped
//! endpoints as two `vec2<f32>` slots per segment, matching the shared
//! [`crate::particle::gpu_layout`] stride so a `WebGPU` kernel binds a stable
//! `ABI`.

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// Magnitude below which the projected direction `p_k` against an edge is
/// treated as zero (segment parallel to that edge). Used instead of `==` on
/// `f32`: a direction component within this band counts as parallel so the
/// division `q_k / p_k` is never evaluated on a near-zero denominator.
pub const CLIP_EPS: f32 = 1.0e-6;

/// An axis-aligned rectangular clip window `[min.x, max.x] x [min.y, max.y]`.
///
/// The bounds are stored as supplied; [`ClipRect::normalized`] swaps any
/// inverted axis so `min <= max` holds before clipping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipRect {
    /// Lower corner of the window (`[x, y]`).
    pub min: [f32; 2],
    /// Upper corner of the window (`[x, y]`).
    pub max: [f32; 2],
}

impl ClipRect {
    /// Builds a window from its lower and upper corners without reordering.
    #[must_use]
    pub const fn new(min: [f32; 2], max: [f32; 2]) -> Self {
        Self { min, max }
    }

    /// Returns an equivalent window with `min.x <= max.x` and `min.y <= max.y`.
    ///
    /// Callers that might pass an inverted rectangle can normalize first so the
    /// four edge slabs are oriented consistently.
    #[must_use]
    pub fn normalized(&self) -> ClipRect {
        ClipRect {
            min: [self.min[0].min(self.max[0]), self.min[1].min(self.max[1])],
            max: [self.min[0].max(self.max[0]), self.min[1].max(self.max[1])],
        }
    }

    /// Returns `true` when the point lies inside the (normalized) window,
    /// treating a point within [`CLIP_EPS`] of an edge as inside it.
    #[must_use]
    pub fn contains(&self, p: [f32; 2]) -> bool {
        let r = self.normalized();
        p[0] >= r.min[0] - CLIP_EPS
            && p[0] <= r.max[0] + CLIP_EPS
            && p[1] >= r.min[1] - CLIP_EPS
            && p[1] <= r.max[1] + CLIP_EPS
    }
}

/// A 2D line segment from [`Segment2::a`] to [`Segment2::b`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment2 {
    /// Start point of the segment (parameter `t = 0`).
    pub a: [f32; 2],
    /// End point of the segment (parameter `t = 1`).
    pub b: [f32; 2],
}

impl Segment2 {
    /// Builds a segment from its two endpoints.
    #[must_use]
    pub const fn new(a: [f32; 2], b: [f32; 2]) -> Self {
        Self { a, b }
    }

    /// Evaluates the segment at parameter `t`, i.e. `a + t * (b - a)`.
    #[must_use]
    pub fn point_at(&self, t: f32) -> [f32; 2] {
        [
            self.a[0] + t * (self.b[0] - self.a[0]),
            self.a[1] + t * (self.b[1] - self.a[1]),
        ]
    }
}

/// The surviving sub-segment of a `Liang-Barsky` clip.
///
/// The endpoints keep the original travel direction: [`ClipResult::p0`] is the
/// point at the entry parameter [`ClipResult::t0`] (derived from the segment's
/// `a`) and [`ClipResult::p1`] is the point at the exit parameter
/// [`ClipResult::t1`] (derived from `b`). The invariant
/// `0 <= t0 <= t1 <= 1` always holds for an accepted clip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipResult {
    /// Clipped start point, equal to `seg.point_at(t0)`.
    pub p0: [f32; 2],
    /// Clipped end point, equal to `seg.point_at(t1)`.
    pub p1: [f32; 2],
    /// Entry parameter along the original segment, in `[0, 1]`.
    pub t0: f32,
    /// Exit parameter along the original segment, in `[0, 1]`.
    pub t1: f32,
}

/// Clips `seg` to the rectangular window and returns the surviving
/// sub-segment together with its entry/exit parameters, or `None` when the
/// segment lies wholly outside.
///
/// The rectangle is normalized internally, so an inverted `rect` is clipped
/// against its swapped-corner equivalent rather than rejected. A degenerate
/// zero-length segment survives only when its single point is inside the
/// window, in which case both returned endpoints are that point with
/// `t0 = 0`, `t1 = 1`.
#[must_use]
pub fn clip(rect: ClipRect, seg: Segment2) -> Option<ClipResult> {
    let r = rect.normalized();
    let dx = seg.b[0] - seg.a[0];
    let dy = seg.b[1] - seg.a[1];

    // Per-edge `(p_k, q_k)` in the order left, right, bottom, top. The point
    // stays inside edge `k` exactly when `t * p_k <= q_k`.
    let p = [-dx, dx, -dy, dy];
    let q = [
        seg.a[0] - r.min[0],
        r.max[0] - seg.a[0],
        seg.a[1] - r.min[1],
        r.max[1] - seg.a[1],
    ];

    let mut t_enter = 0.0f32;
    let mut t_exit = 1.0f32;

    for (&pk, &qk) in p.iter().zip(q.iter()) {
        if pk.abs() <= CLIP_EPS {
            // Parallel to this edge: reject only when the start point already
            // sits outside the edge's slab.
            if qk < 0.0 {
                return None;
            }
        } else {
            let t = qk / pk;
            if pk < 0.0 {
                // Crossing inward: tighten the entry parameter.
                t_enter = t_enter.max(t);
            } else {
                // Crossing outward: tighten the exit parameter.
                t_exit = t_exit.min(t);
            }
        }
    }

    if t_enter > t_exit {
        return None;
    }

    Some(ClipResult {
        p0: seg.point_at(t_enter),
        p1: seg.point_at(t_exit),
        t0: t_enter,
        t1: t_exit,
    })
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

    const RECT: ClipRect = ClipRect::new([0.0, 0.0], [10.0, 10.0]);

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-4
    }

    fn points_close(p: [f32; 2], q: [f32; 2]) -> bool {
        approx(p[0], q[0]) && approx(p[1], q[1])
    }

    #[test]
    fn fully_inside_segment_is_returned_unchanged() {
        let res = clip(RECT, Segment2::new([2.0, 3.0], [7.0, 8.0])).expect("inside survives");
        assert!(points_close(res.p0, [2.0, 3.0]));
        assert!(points_close(res.p1, [7.0, 8.0]));
        assert!(approx(res.t0, 0.0));
        assert!(approx(res.t1, 1.0));
    }

    #[test]
    fn fully_outside_left_is_rejected() {
        assert!(clip(RECT, Segment2::new([-5.0, 2.0], [-1.0, 8.0])).is_none());
    }

    #[test]
    fn fully_outside_right_is_rejected() {
        assert!(clip(RECT, Segment2::new([11.0, 2.0], [15.0, 8.0])).is_none());
    }

    #[test]
    fn fully_outside_above_is_rejected() {
        assert!(clip(RECT, Segment2::new([2.0, 11.0], [8.0, 20.0])).is_none());
    }

    #[test]
    fn fully_outside_below_is_rejected() {
        assert!(clip(RECT, Segment2::new([2.0, -11.0], [8.0, -1.0])).is_none());
    }

    #[test]
    fn crossing_left_edge_clips_entry_point() {
        let res = clip(RECT, Segment2::new([-5.0, 5.0], [5.0, 5.0])).expect("enters window");
        assert!(points_close(res.p0, [0.0, 5.0]));
        assert!(points_close(res.p1, [5.0, 5.0]));
        assert!(approx(res.t0, 0.5));
        assert!(approx(res.t1, 1.0));
    }

    #[test]
    fn crossing_right_edge_clips_exit_point() {
        let res = clip(RECT, Segment2::new([5.0, 5.0], [15.0, 5.0])).expect("exits window");
        assert!(points_close(res.p0, [5.0, 5.0]));
        assert!(points_close(res.p1, [10.0, 5.0]));
        assert!(approx(res.t0, 0.0));
        assert!(approx(res.t1, 0.5));
    }

    #[test]
    fn horizontal_line_spanning_both_vertical_edges() {
        let res = clip(RECT, Segment2::new([-5.0, 5.0], [15.0, 5.0])).expect("spans window");
        assert!(points_close(res.p0, [0.0, 5.0]));
        assert!(points_close(res.p1, [10.0, 5.0]));
        // Exact parameters: a.x = -5, dx = 20 -> t0 = 5/20, t1 = 15/20.
        assert!(approx(res.t0, 0.25));
        assert!(approx(res.t1, 0.75));
    }

    #[test]
    fn vertical_line_spanning_both_horizontal_edges() {
        let res = clip(RECT, Segment2::new([4.0, -5.0], [4.0, 15.0])).expect("spans window");
        assert!(points_close(res.p0, [4.0, 0.0]));
        assert!(points_close(res.p1, [4.0, 10.0]));
        assert!(approx(res.t0, 0.25));
        assert!(approx(res.t1, 0.75));
    }

    #[test]
    fn horizontal_line_fully_inside_is_unchanged() {
        let res = clip(RECT, Segment2::new([2.0, 6.0], [8.0, 6.0])).expect("inside");
        assert!(points_close(res.p0, [2.0, 6.0]));
        assert!(points_close(res.p1, [8.0, 6.0]));
        assert!(approx(res.t0, 0.0) && approx(res.t1, 1.0));
    }

    #[test]
    fn vertical_line_fully_inside_is_unchanged() {
        let res = clip(RECT, Segment2::new([3.0, 1.0], [3.0, 9.0])).expect("inside");
        assert!(points_close(res.p0, [3.0, 1.0]));
        assert!(points_close(res.p1, [3.0, 9.0]));
    }

    #[test]
    fn diagonal_through_window_hits_opposite_corners() {
        let res = clip(RECT, Segment2::new([-4.0, -4.0], [14.0, 14.0])).expect("diagonal");
        assert!(points_close(res.p0, [0.0, 0.0]));
        assert!(points_close(res.p1, [10.0, 10.0]));
        // a = -4, d = 18 -> t0 = 4/18, t1 = 14/18.
        assert!(approx(res.t0, 4.0 / 18.0));
        assert!(approx(res.t1, 14.0 / 18.0));
    }

    #[test]
    fn shallow_slope_intersection_is_exact() {
        // From (-10, 2) to (10, 6): slope 0.2. At x = 0, y = 4.
        let res = clip(RECT, Segment2::new([-10.0, 2.0], [10.0, 6.0])).expect("shallow slope");
        assert!(points_close(res.p0, [0.0, 4.0]));
        assert!(points_close(res.p1, [10.0, 6.0]));
        assert!(approx(res.t0, 0.5));
        assert!(approx(res.t1, 1.0));
    }

    #[test]
    fn endpoint_on_left_boundary_counts_as_inside() {
        let res = clip(RECT, Segment2::new([0.0, 5.0], [5.0, 5.0])).expect("boundary start");
        assert!(points_close(res.p0, [0.0, 5.0]));
        assert!(points_close(res.p1, [5.0, 5.0]));
        assert!(approx(res.t0, 0.0));
    }

    #[test]
    fn endpoint_on_corner_counts_as_inside() {
        let res = clip(RECT, Segment2::new([10.0, 10.0], [5.0, 5.0])).expect("corner start");
        assert!(points_close(res.p0, [10.0, 10.0]));
        assert!(points_close(res.p1, [5.0, 5.0]));
    }

    #[test]
    fn segment_grazing_a_corner_returns_single_point() {
        // The line touches the corner (0, 0) and is otherwise outside; the
        // surviving interval collapses to a single parameter.
        let res = clip(RECT, Segment2::new([-2.0, 2.0], [2.0, -2.0])).expect("grazes corner");
        assert!(points_close(res.p0, [0.0, 0.0]));
        assert!(points_close(res.p1, [0.0, 0.0]));
        assert!(approx(res.t0, res.t1));
    }

    #[test]
    fn horizontal_parallel_line_above_is_rejected() {
        // dy = 0 so the segment is parallel to the top/bottom edges and sits
        // above the window: the parallel-outside branch rejects it.
        assert!(clip(RECT, Segment2::new([-5.0, 15.0], [15.0, 15.0])).is_none());
    }

    #[test]
    fn horizontal_parallel_line_below_is_rejected() {
        assert!(clip(RECT, Segment2::new([-5.0, -1.0], [15.0, -1.0])).is_none());
    }

    #[test]
    fn vertical_parallel_line_right_is_rejected() {
        assert!(clip(RECT, Segment2::new([15.0, -5.0], [15.0, 15.0])).is_none());
    }

    #[test]
    fn vertical_parallel_line_left_is_rejected() {
        assert!(clip(RECT, Segment2::new([-1.0, -5.0], [-1.0, 15.0])).is_none());
    }

    #[test]
    fn horizontal_parallel_line_inside_survives() {
        // Parallel to top/bottom but within the vertical slab: it must clip on
        // the vertical edges and survive.
        let res = clip(RECT, Segment2::new([-3.0, 5.0], [13.0, 5.0])).expect("parallel inside");
        assert!(points_close(res.p0, [0.0, 5.0]));
        assert!(points_close(res.p1, [10.0, 5.0]));
    }

    #[test]
    fn degenerate_point_inside_returns_that_point() {
        let res = clip(RECT, Segment2::new([5.0, 5.0], [5.0, 5.0])).expect("point inside");
        assert!(points_close(res.p0, [5.0, 5.0]));
        assert!(points_close(res.p1, [5.0, 5.0]));
        assert!(approx(res.t0, 0.0));
        assert!(approx(res.t1, 1.0));
    }

    #[test]
    fn degenerate_point_outside_is_rejected() {
        assert!(clip(RECT, Segment2::new([-5.0, -5.0], [-5.0, -5.0])).is_none());
    }

    #[test]
    fn degenerate_point_on_boundary_survives() {
        let res = clip(RECT, Segment2::new([0.0, 0.0], [0.0, 0.0])).expect("point on corner");
        assert!(points_close(res.p0, [0.0, 0.0]));
        assert!(points_close(res.p1, [0.0, 0.0]));
    }

    #[test]
    fn swapping_endpoints_swaps_the_visible_endpoints() {
        let forward = clip(RECT, Segment2::new([-5.0, 5.0], [15.0, 5.0])).expect("forward");
        let reverse = clip(RECT, Segment2::new([15.0, 5.0], [-5.0, 5.0])).expect("reverse");
        // The direction flips, so the reversed clip's first point matches the
        // forward clip's second point and vice versa.
        assert!(points_close(reverse.p0, forward.p1));
        assert!(points_close(reverse.p1, forward.p0));
    }

    #[test]
    fn known_exact_parameters_on_a_diagonal_span() {
        // From (-2, -2) to (6, 6) with slope 1; enters at (0,0), exits at (6,6).
        let res = clip(RECT, Segment2::new([-2.0, -2.0], [6.0, 6.0])).expect("diagonal span");
        assert!(points_close(res.p0, [0.0, 0.0]));
        assert!(points_close(res.p1, [6.0, 6.0]));
        // a = -2, d = 8 -> t0 = 2/8 = 0.25, exit stays at t1 = 1.
        assert!(approx(res.t0, 0.25));
        assert!(approx(res.t1, 1.0));
    }

    #[test]
    fn parameter_interval_matches_point_at() {
        let seg = Segment2::new([-3.0, -1.0], [12.0, 13.0]);
        let res = clip(RECT, seg).expect("threads window");
        // The reported endpoints must equal the segment evaluated at t0/t1.
        assert!(points_close(res.p0, seg.point_at(res.t0)));
        assert!(points_close(res.p1, seg.point_at(res.t1)));
    }

    #[test]
    fn accepted_parameters_stay_ordered_within_unit_interval() {
        let seg = Segment2::new([-5.0, 2.0], [15.0, 12.0]);
        let res = clip(RECT, seg).expect("threads window");
        assert!(res.t0 >= 0.0 - CLIP_EPS);
        assert!(res.t1 <= 1.0 + CLIP_EPS);
        assert!(res.t0 <= res.t1);
    }

    #[test]
    fn inverted_rectangle_is_normalized_before_clipping() {
        let inverted = ClipRect::new([10.0, 10.0], [0.0, 0.0]);
        let res = clip(inverted, Segment2::new([-5.0, 5.0], [15.0, 5.0])).expect("normalized clip");
        assert!(points_close(res.p0, [0.0, 5.0]));
        assert!(points_close(res.p1, [10.0, 5.0]));
    }

    #[test]
    fn normalized_swaps_inverted_bounds() {
        let inverted = ClipRect::new([10.0, 8.0], [0.0, 2.0]);
        let fixed = inverted.normalized();
        assert!(approx(fixed.min[0], 0.0) && approx(fixed.min[1], 2.0));
        assert!(approx(fixed.max[0], 10.0) && approx(fixed.max[1], 8.0));
    }

    #[test]
    fn contains_reports_inside_and_outside() {
        assert!(RECT.contains([5.0, 5.0]));
        assert!(RECT.contains([0.0, 0.0]));
        assert!(!RECT.contains([-1.0, 5.0]));
        assert!(!RECT.contains([5.0, 11.0]));
    }

    #[test]
    fn point_within_epsilon_of_edge_counts_as_inside() {
        let p = [10.0 + CLIP_EPS * 0.5, 5.0];
        assert!(RECT.contains(p));
    }

    #[test]
    fn entering_bottom_left_and_exiting_top_right() {
        // A line from below-left to above-right that clips on two edges.
        let res = clip(RECT, Segment2::new([-2.0, -1.0], [12.0, 13.0])).expect("threads window");
        assert!(res.t0 > 0.0);
        assert!(res.t1 < 1.0);
        assert!(res.p0[0] >= -CLIP_EPS && res.p0[1] >= -CLIP_EPS);
        assert!(res.p1[0] <= 10.0 + CLIP_EPS && res.p1[1] <= 10.0 + CLIP_EPS);
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
