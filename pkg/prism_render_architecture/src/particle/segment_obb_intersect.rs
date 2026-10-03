//! Finite line segment vs *oriented bounding box* (`OBB`) intersection by the
//! slab-clip method, for the particle subsystem's collision-probe, trail-clip,
//! and analytic-primitive contracts (design §8.2, §10, §14).
//!
//! An **`OBB`** is an [`Aabb`](crate::particle::sort_cull)-shaped box that has
//! been *rotated* into world space: a `center`, three mutually orthogonal
//! **unit** axes `axes[0..3]`, and a non-negative half-extent `half[i]` along
//! each axis. Because an orthonormal basis is its own inverse, the signed
//! coordinate of a world point `p` in the box frame along axis `i` is the dot
//! product `(p - center) . axes[i]`. A **finite segment** runs from `p0` to
//! `p1`; a point on it is `p0 + t * (p1 - p0)` for `t` in `[0, 1]`. Projecting
//! the segment onto each box axis turns the query into three independent 1D
//! slab clips: for each axis the parameter interval where the projection lies
//! inside `[-half[i], half[i]]` is intersected into a running
//! `[t_enter, t_exit]` span, itself seeded to the segment's own `[0, 1]` range
//! so the result is always a *normalized sub-interval* of the segment.
//!
//! [`segment_obb_intersect`] returns the clipped `[t_enter, t_exit]` span
//! (`t_enter <= t_exit`, both in `[0, 1]`) or `None` on a miss;
//! [`segment_obb_overlaps`] is the unsigned boolean predicate; and
//! [`point_in_obb`] is the standalone containment test used by both.
//!
//! # Strict scope — how this differs from its siblings
//! This module clips a **finite** segment against an `OBB` and reports a
//! bounded `[0, 1]` parameter sub-interval. It is deliberately disjoint from
//! the related solvers and neither imports nor reconstructs their primitives:
//!
//! * [`crate::particle::ray_obb`] intersects an *infinite / half-infinite* ray
//!   (`t >= 0`, no upper bound) with an `OBB` and reports a surface hit with a
//!   face normal. This module has a hard *upper* bound at `t = 1` (the segment
//!   end) as well as the lower bound at `t = 0`, and returns the whole clipped
//!   chord rather than a single surface hit.
//! * [`crate::particle::obb_obb_sat_3d`] runs the *separating-axis theorem*
//!   between **two** `OBB`s to decide box-vs-box overlap; here one primitive is
//!   a 1D segment, not a second box, and the answer is a parameter interval,
//!   not a separation flag.
//! * [`crate::particle::sweep_aabb`] sweeps an *axis-aligned* box along a
//!   velocity (a moving-`AABB` continuous-collision query); this module clips a
//!   static segment against a possibly *rotated* box, so its slabs are the
//!   box's own axes recovered by dot products, not the world planes.
//! * [`crate::particle::segment_closest_point_3d`] finds the nearest points
//!   between two segments (a distance query); this module answers a
//!   segment-vs-solid *intersection* question.
//!
//! An `OBB` whose `axes` are the world basis is exactly an `AABB`, so the
//! axis-aligned degenerate case agrees with a plain world-plane slab clip.
//!
//! # Degenerate cases handled explicitly (never a `NaN`)
//! * **Segment parallel to a slab** — when the direction's projection onto an
//!   axis is smaller than [`EPS`] in magnitude, that axis contributes no finite
//!   `1 / f` division: the segment misses immediately if its start projects
//!   outside the `[-half, half]` slab, and otherwise the slab imposes no
//!   constraint and is skipped.
//! * **Zero-length segment** — `p0 == p1` makes every projection parallel, so
//!   the query reduces to [`point_in_obb`] on `p0`, returning the full `[0, 1]`
//!   span when the point is inside and `None` otherwise.
//! * **Endpoint exactly on a face** — falls out of the clip as a degenerate
//!   `t_enter == t_exit` touch (compared against [`EPS`], never with `==`).
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled as
//! free functions in this file, every `f32` division guards its denominator
//! against [`EPS`], no transcendental function is ever called, and no exact
//! `==` / `!=` is ever written on a production `f32` (magnitudes are compared
//! against an explicit epsilon), so this `CPU` reference stays reproducible
//! against a future `GPU` kernel that packs the same box.

/// Magnitude floor used to guard the parallel-slab `1 / f` division and to
/// classify a projected coordinate as inside or outside a slab, so no exact
/// `==` / `!=` is ever written on a production `f32`.
pub const EPS: f32 = 1.0e-6;

/// Component-wise sum `a + b` of two 3-vectors (a free function so no operator
/// trait is implemented on a bare `[f32; 3]`).
#[must_use]
pub fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise difference `a - b` of two 3-vectors.
#[must_use]
pub fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales the 3-vector `a` by the scalar `s`.
#[must_use]
pub fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product `a . b` of two 3-vectors.
#[must_use]
pub fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The clipped parameter span of a segment against an `OBB`: the segment enters
/// the box at `t_enter` and leaves at `t_exit`, both normalized to the segment's
/// own `[0, 1]` parameterization with `t_enter <= t_exit`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentObbHit {
    /// Parameter at which the segment enters the box (in `[0, 1]`).
    pub t_enter: f32,
    /// Parameter at which the segment leaves the box (in `[0, 1]`).
    pub t_exit: f32,
}

/// Tests whether `point` lies inside (or on the surface of) the `OBB` described
/// by `center`, orthonormal `axes`, and non-negative `half` extents.
///
/// The point is inside exactly when the magnitude of its projection onto each
/// axis is within that axis's half-extent (with an [`EPS`] slack so a point
/// exactly on a face counts as inside).
#[must_use]
pub fn point_in_obb(
    point: [f32; 3],
    center: [f32; 3],
    axes: [[f32; 3]; 3],
    half: [f32; 3],
) -> bool {
    let m = v_sub(point, center);
    for (axis, &h) in axes.iter().zip(half.iter()) {
        if v_dot(m, *axis).abs() > h + EPS {
            return false;
        }
    }
    true
}

/// Clips the finite segment `p0 -> p1` against the `OBB` (`center`, orthonormal
/// `axes`, non-negative `half`) and returns the surviving `[t_enter, t_exit]`
/// parameter span, or `None` when the segment never meets the box.
///
/// The span is always a sub-interval of `[0, 1]` with `t_enter <= t_exit`. The
/// solver seeds the running span to the segment's own `[0, 1]` range and
/// tightens it with each axis slab, so both endpoints stay clamped to the
/// finite segment. A direction component whose projection is below [`EPS`] in
/// magnitude is treated as parallel to that slab: the segment misses if its
/// start projects outside the slab and is otherwise unconstrained on that axis.
#[must_use]
pub fn segment_obb_intersect(
    p0: [f32; 3],
    p1: [f32; 3],
    center: [f32; 3],
    axes: [[f32; 3]; 3],
    half: [f32; 3],
) -> Option<SegmentObbHit> {
    let dir = v_sub(p1, p0);
    let m = v_sub(p0, center);
    let mut t_enter = 0.0_f32;
    let mut t_exit = 1.0_f32;
    for (axis, &h) in axes.iter().zip(half.iter()) {
        let e = v_dot(m, *axis);
        let f = v_dot(dir, *axis);
        if f.abs() < EPS {
            // Parallel to this slab: a start projected outside it can never enter.
            if e.abs() > h + EPS {
                return None;
            }
        } else {
            let inv = 1.0 / f;
            let t1 = (-h - e) * inv;
            let t2 = (h - e) * inv;
            let (t_near, t_far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            t_enter = t_enter.max(t_near);
            t_exit = t_exit.min(t_far);
            if t_enter > t_exit {
                return None;
            }
        }
    }
    Some(SegmentObbHit { t_enter, t_exit })
}

/// Convenience predicate: `true` when the finite segment `p0 -> p1` meets the
/// `OBB` at all, discarding the parameter span.
#[must_use]
pub fn segment_obb_overlaps(
    p0: [f32; 3],
    p1: [f32; 3],
    center: [f32; 3],
    axes: [[f32; 3]; 3],
    half: [f32; 3],
) -> bool {
    segment_obb_intersect(p0, p1, center, axes, half).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The world basis, i.e. an `OBB` that is really an `AABB`.
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1.0e-4
    }

    fn point_at(p0: [f32; 3], p1: [f32; 3], t: f32) -> [f32; 3] {
        v_add(p0, v_scale(v_sub(p1, p0), t))
    }

    /// A tiny deterministic linear-congruential generator for the fuzz check.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 33) as u32
        }

        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    #[test]
    fn axis_aligned_pierce_through_center() {
        let hit = segment_obb_intersect(
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("segment pierces the box");
        assert!(approx(hit.t_enter, 0.25));
        assert!(approx(hit.t_exit, 0.75));
    }

    #[test]
    fn axis_aligned_pierce_along_y() {
        let hit = segment_obb_intersect(
            [0.0, -4.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("segment pierces the box along y");
        assert!(approx(hit.t_enter, 0.375));
        assert!(approx(hit.t_exit, 0.625));
    }

    #[test]
    fn segment_fully_inside_spans_whole_unit_interval() {
        let hit = segment_obb_intersect(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [2.0, 2.0, 2.0],
        )
        .expect("segment lies inside the box");
        assert!(approx(hit.t_enter, 0.0));
        assert!(approx(hit.t_exit, 1.0));
    }

    #[test]
    fn segment_fully_outside_misses() {
        let hit = segment_obb_intersect(
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        );
        assert!(hit.is_none());
    }

    #[test]
    fn specific_t_interval_correctness() {
        let hit = segment_obb_intersect(
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("hit expected");
        assert!(approx(hit.t_enter, 0.25));
        assert!(approx(hit.t_exit, 0.75));
    }

    #[test]
    fn diagonal_segment_through_corners() {
        let hit = segment_obb_intersect(
            [-2.0, -2.0, -2.0],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("diagonal pierces the box");
        assert!(approx(hit.t_enter, 0.25));
        assert!(approx(hit.t_exit, 0.75));
    }

    #[test]
    fn graze_along_face_plane() {
        // Segment travels in y at exactly x = 1 (the +x face plane).
        let hit = segment_obb_intersect(
            [1.0, -2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("grazing the face still counts as a touch");
        assert!(approx(hit.t_enter, 0.25));
        assert!(approx(hit.t_exit, 0.75));
    }

    #[test]
    fn parallel_to_face_but_outside_misses() {
        let hit = segment_obb_intersect(
            [2.0, -5.0, 0.0],
            [2.0, 5.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        );
        assert!(hit.is_none());
    }

    #[test]
    fn parallel_to_face_and_inside_hits() {
        let hit = segment_obb_intersect(
            [0.5, -5.0, 0.0],
            [0.5, 5.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("parallel but inside the slab");
        assert!(approx(hit.t_enter, 0.4));
        assert!(approx(hit.t_exit, 0.6));
    }

    #[test]
    fn endpoint_exactly_on_face() {
        let hit = segment_obb_intersect(
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("start lies on the +x face");
        assert!(approx(hit.t_enter, 0.0));
        assert!(approx(hit.t_exit, 0.0));
    }

    #[test]
    fn zero_length_segment_inside() {
        let hit = segment_obb_intersect(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("degenerate point inside the box");
        assert!(approx(hit.t_enter, 0.0));
        assert!(approx(hit.t_exit, 1.0));
    }

    #[test]
    fn zero_length_segment_outside() {
        let hit = segment_obb_intersect(
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        );
        assert!(hit.is_none());
    }

    #[test]
    fn segment_starting_inside_clamps_enter_to_zero() {
        let hit = segment_obb_intersect(
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("start is inside");
        assert!(approx(hit.t_enter, 0.0));
        assert!(approx(hit.t_exit, 0.2));
    }

    #[test]
    fn segment_ending_inside_clamps_exit_to_one() {
        let hit = segment_obb_intersect(
            [-5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("end is inside");
        assert!(approx(hit.t_enter, 0.8));
        assert!(approx(hit.t_exit, 1.0));
    }

    #[test]
    fn interval_is_clamped_to_unit_range() {
        let hit = segment_obb_intersect(
            [-10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("long segment crosses the box");
        assert!((0.0..=1.0).contains(&hit.t_enter));
        assert!((0.0..=1.0).contains(&hit.t_exit));
        assert!(hit.t_enter <= hit.t_exit);
        assert!(approx(hit.t_enter, 0.45));
        assert!(approx(hit.t_exit, 0.55));
    }

    #[test]
    fn offset_center_shifts_interval() {
        let hit = segment_obb_intersect(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("hit near the far end");
        assert!(approx(hit.t_enter, 0.4));
        assert!(approx(hit.t_exit, 0.6));
    }

    #[test]
    fn rotated_box_is_hit() {
        let s = 0.5_f32.sqrt();
        let axes = [[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        let hit = segment_obb_intersect(
            [-3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            axes,
            [1.0, 1.0, 1.0],
        )
        .expect("world-x segment crosses the rotated box");
        assert!(hit.t_enter <= hit.t_exit);
        let mid = point_at(
            [-3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            (hit.t_enter + hit.t_exit) * 0.5,
        );
        assert!(point_in_obb(mid, [0.0, 0.0, 0.0], axes, [1.0, 1.0, 1.0]));
    }

    #[test]
    fn rotated_box_is_missed() {
        let s = 0.5_f32.sqrt();
        let axes = [[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        let hit = segment_obb_intersect(
            [-3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            axes,
            [1.0, 1.0, 1.0],
        );
        assert!(hit.is_none());
    }

    #[test]
    fn rotated_box_diagonal_pierce_interval() {
        // Along the rotated first axis the box reaches +/- 1 in local coords.
        let s = 0.5_f32.sqrt();
        let axes = [[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        // A world-space segment along [s, s, 0] direction from -2*axis0 to +2*axis0.
        let start = v_scale([s, s, 0.0], -2.0);
        let end = v_scale([s, s, 0.0], 2.0);
        let hit = segment_obb_intersect(start, end, [0.0, 0.0, 0.0], axes, [1.0, 1.0, 1.0])
            .expect("segment along the box's own axis");
        assert!(approx(hit.t_enter, 0.25));
        assert!(approx(hit.t_exit, 0.75));
    }

    #[test]
    fn swapping_endpoints_mirrors_the_interval() {
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let a = [-3.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let forward = segment_obb_intersect(a, b, center, IDENTITY, half).expect("forward hit");
        let backward = segment_obb_intersect(b, a, center, IDENTITY, half).expect("backward hit");
        assert!(approx(backward.t_enter, 1.0 - forward.t_exit));
        assert!(approx(backward.t_exit, 1.0 - forward.t_enter));
    }

    #[test]
    fn overlaps_true_when_crossing() {
        assert!(segment_obb_overlaps(
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        ));
    }

    #[test]
    fn overlaps_false_when_apart() {
        assert!(!segment_obb_overlaps(
            [-5.0, 5.0, 0.0],
            [-4.0, 6.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        ));
    }

    #[test]
    fn point_in_obb_true_at_center() {
        assert!(point_in_obb(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        ));
    }

    #[test]
    fn point_in_obb_false_outside() {
        assert!(!point_in_obb(
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        ));
    }

    #[test]
    fn point_in_obb_on_face_counts_as_inside() {
        assert!(point_in_obb(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        ));
    }

    #[test]
    fn point_in_rotated_obb() {
        let s = 0.5_f32.sqrt();
        let axes = [[s, s, 0.0], [-s, s, 0.0], [0.0, 0.0, 1.0]];
        assert!(point_in_obb(
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            axes,
            [1.0, 1.0, 1.0]
        ));
        assert!(!point_in_obb(
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            axes,
            [1.0, 1.0, 1.0]
        ));
    }

    #[test]
    fn non_unit_half_extents_scale_interval() {
        let hit = segment_obb_intersect(
            [-4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [2.0, 1.0, 1.0],
        )
        .expect("wider box on x");
        assert!(approx(hit.t_enter, 0.25));
        assert!(approx(hit.t_exit, 0.75));
    }

    #[test]
    fn barely_touching_corner_edge() {
        // A segment that clips just the +x/+y corner region.
        let hit = segment_obb_intersect(
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        );
        // The line x + y = 3 stays outside the box (max x + y on the box is 2).
        assert!(hit.is_none());
    }

    #[test]
    fn touching_corner_point_is_a_degenerate_hit() {
        // Segment endpoint sits exactly at the (1,1,1) corner.
        let hit = segment_obb_intersect(
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("corner touch");
        assert!(approx(hit.t_enter, 0.0));
        assert!(approx(hit.t_exit, 0.0));
    }

    #[test]
    fn enter_before_exit_invariant_holds() {
        let hit = segment_obb_intersect(
            [-3.0, -0.5, 0.2],
            [3.0, 0.5, -0.2],
            [0.0, 0.0, 0.0],
            IDENTITY,
            [1.0, 1.0, 1.0],
        )
        .expect("skewed crossing");
        assert!(hit.t_enter <= hit.t_exit);
        assert!((0.0..=1.0).contains(&hit.t_enter));
        assert!((0.0..=1.0).contains(&hit.t_exit));
    }

    #[test]
    fn lcg_random_matches_naive_sampling() {
        let mut rng = Lcg::new(0x1234_5678_9abc_def0);
        let steps = 512_u32;
        for _ in 0..200 {
            let center = [
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
            ];
            let half = [
                rng.range(0.5, 2.0),
                rng.range(0.5, 2.0),
                rng.range(0.5, 2.0),
            ];
            let p0 = [
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
            ];
            let p1 = [
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
                rng.range(-6.0, 6.0),
            ];
            let hit = segment_obb_intersect(p0, p1, center, IDENTITY, half);

            // Every densely sampled interior point must lie within the analytic span.
            for k in 0..=steps {
                let t = k as f32 / steps as f32;
                let pt = point_at(p0, p1, t);
                #[expect(
                    clippy::collapsible_if,
                    reason = "two guards read clearer kept nested in this densely sampled interior-point check"
                )]
                if point_in_obb(pt, center, IDENTITY, half) {
                    if let Some(h) = hit {
                        assert!(t >= h.t_enter - 1.0e-2 && t <= h.t_exit + 1.0e-2);
                    }
                }
            }

            // When there is an analytic hit, its midpoint must be inside the box.
            if let Some(h) = hit {
                let mid = point_at(p0, p1, (h.t_enter + h.t_exit) * 0.5);
                assert!(point_in_obb(mid, center, IDENTITY, half));
            }
        }
    }

    #[test]
    fn aabb_agreement_with_direct_slab_clip() {
        // Cross-check the identity-axis path against a hand-written world-plane clip.
        let p0 = [-3.0, 0.5, 0.0];
        let p1 = [3.0, 0.5, 0.0];
        let center = [0.0, 0.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        let hit = segment_obb_intersect(p0, p1, center, IDENTITY, half).expect("hit");
        // Direct: only x varies; enter at x = -1, exit at x = 1.
        assert!(approx(hit.t_enter, (-1.0 - p0[0]) / (p1[0] - p0[0])));
        assert!(approx(hit.t_exit, (1.0 - p0[0]) / (p1[0] - p0[0])));
    }
}
