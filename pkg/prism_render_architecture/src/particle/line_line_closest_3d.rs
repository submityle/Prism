//! Closest points between two infinite 3D lines (and the ray-vs-ray variant).
//!
//! This module is the *infinite-line* half of the subsystem's proximity math,
//! and it is deliberately disjoint from its closest sibling:
//!
//! * [`crate::particle::segment_closest_point_3d`] solves the **finite
//!   segment** problem. Its parameters live in the unit square `(s, t) ∈
//!   [0, 1]²`; every step clamps back into that box, so the answer is always a
//!   point that physically lies *on* the two segments. That is what a
//!   capsule-vs-capsule or trail-vs-trail proximity query wants.
//! * **This module** solves the **infinite line** problem. The parameters are
//!   unconstrained reals `(s, t) ∈ ℝ²`: the returned points lie on the two
//!   *supporting lines*, which may sit far outside any authored segment. This
//!   is the primitive an edit-time snapping gizmo, an axis/rail alignment tool,
//!   or a screen-ray-vs-world-ray picking helper reaches for, where clamping to
//!   an endpoint would be wrong. As a bridge to bounded geometry it also
//!   exposes [`ray_ray_closest`], which restricts both parameters to the
//!   non-negative half-line `s, t ≥ 0`.
//!
//! The line-vs-line solver is the classic closed form. Line A is written
//! `A(s) = p1 + s · d1` and line B is `B(t) = p2 + t · d2` (the directions need
//! not be unit length). Writing `r = p1 − p2` and
//! `a = d1·d1`, `b = d1·d2`, `c = d2·d2`, `d = d1·r`, `e = d2·r`, the squared
//! distance `‖A(s) − B(t)‖²` is minimized where its gradient vanishes, giving
//! the 2×2 system whose determinant is `denom = a·c − b²`. When `denom` is near
//! zero the directions are parallel and the system is rank-deficient, so the
//! code fixes `s = 0` and solves `t = e / c` (guarding the divide), which still
//! reports the true line-to-line gap. Otherwise
//! `s = (b·e − c·d) / denom` and `t = (a·e − b·d) / denom`.
//!
//! The ray variant follows Ericson's *Real-Time Collision Detection* (§5.1.9):
//! solve the unconstrained system, clamp each parameter to `≥ 0`, and when a
//! clamp actually moved a parameter, re-project onto the *other* ray and clamp
//! again so the pair stays mutually closest under the half-line constraints.
//!
//! Everything is a zero-dependency contract: the vector math is a handful of
//! free functions over `[f32; 3]`, and every step uses only `+ - * /`,
//! `f32::sqrt`, `f32::abs`, `f32::min`, `f32::max`, and `f32::clamp`. No
//! transcendental function and no `==` / `!=` on a production `f32` ever
//! appears, so this `CPU` reference stays bit-compatible with a future `GPU`
//! kernel and never manufactures a `NaN` from an unguarded divide.

/// Epsilon guarding every divide and magnitude test, so no exact `==` / `!=`
/// on a production `f32` is ever needed.
///
/// A determinant `denom` is treated as parallel when `denom.abs()` falls at or
/// below `EPS` times the product `a·c` of the two squared direction lengths
/// (a scale-relative test), and any squared length at or below `EPS` is treated
/// as a degenerate zero-length direction.
pub const EPS: f32 = 1.0e-6;

/// Component-wise sum `a + b`.
#[must_use]
pub fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise difference `a − b`.
#[must_use]
pub fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean dot product `a · b`.
#[must_use]
pub fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-handed cross product `a × b`.
#[must_use]
pub fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Uniform scale `a · s`.
#[must_use]
pub fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean length `√(v · v)`.
#[must_use]
pub fn v_length(v: [f32; 3]) -> f32 {
    v_dot(v, v).sqrt()
}

/// The result of a closest-point query between two lines (or two rays).
///
/// `s` is the parameter along the first line `A(s) = p1 + s · d1` and `t` the
/// parameter along the second line `B(t) = p2 + t · d2`. For
/// [`line_line_closest`] the parameters are unconstrained reals; for
/// [`ray_ray_closest`] they are clamped to the non-negative half-line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosestLines {
    /// Parameter along the first line/ray.
    pub s: f32,
    /// Parameter along the second line/ray.
    pub t: f32,
    /// The closest point on the first line/ray, `p1 + s · d1`.
    pub point_on_a: [f32; 3],
    /// The closest point on the second line/ray, `p2 + t · d2`.
    pub point_on_b: [f32; 3],
    /// The Euclidean distance between the two closest points.
    pub distance: f32,
}

/// Closest points between the two **infinite lines** `A(s) = p1 + s · d1` and
/// `B(t) = p2 + t · d2`.
///
/// The directions `d1`, `d2` need not be normalized. Parameters `s` and `t`
/// range over all of `ℝ` (no clamping): the reported points lie on the
/// supporting lines, not on any bounded segment.
///
/// # Degeneracies
///
/// * **Parallel lines.** When `denom = a·c − b²` is near zero (tested as
///   `denom.abs() ≤ EPS · a · c`, a scale-relative parallel test), the two
///   directions are (anti-)parallel and the system is rank-deficient. The code
///   fixes `s = 0` and, when `c > EPS`, solves `t = e / c` — the foot of the
///   perpendicular from `p1` onto line B — so `distance` is the true gap
///   between the parallel lines. Coincident lines fall out of this same branch
///   with `distance ≈ 0`.
/// * **Zero-length directions.** If a direction's squared length is at or below
///   [`EPS`] the corresponding parameter is forced to `0`, collapsing that line
///   to its base point; the query degrades to a point-to-line (or point-to-
///   point) distance without dividing by zero.
#[must_use]
pub fn line_line_closest(p1: [f32; 3], d1: [f32; 3], p2: [f32; 3], d2: [f32; 3]) -> ClosestLines {
    let r = v_sub(p1, p2);
    let a = v_dot(d1, d1);
    let b = v_dot(d1, d2);
    let c = v_dot(d2, d2);
    let d = v_dot(d1, r);
    let e = v_dot(d2, r);

    let denom = a * c - b * b;

    let (s, t);
    // Scale-relative parallel test: `a · c` is the largest magnitude the
    // determinant could reach (`b² ≤ a·c` by Cauchy–Schwarz), so comparing
    // against `EPS · a · c` is invariant to how the directions are scaled.
    if denom.abs() <= EPS * a * c {
        // Parallel (or one direction is degenerate). Pin `s = 0` on line A and
        // drop a perpendicular onto line B. When line B is itself degenerate
        // (`c ≤ EPS`) there is no direction to project along, so `t = 0` too
        // and the query becomes point-to-point.
        s = 0.0;
        t = if c > EPS { e / c } else { 0.0 };
    } else {
        // Full-rank system: unique mutually-closest pair.
        s = (b * e - c * d) / denom;
        t = (a * e - b * d) / denom;
    }

    let point_on_a = v_add(p1, v_scale(d1, s));
    let point_on_b = v_add(p2, v_scale(d2, t));
    let distance = v_length(v_sub(point_on_a, point_on_b));

    ClosestLines {
        s,
        t,
        point_on_a,
        point_on_b,
        distance,
    }
}

/// Closest points between the two **rays** `A(s) = o1 + s · d1` and
/// `B(t) = o2 + t · d2`, restricted to the non-negative half-lines `s, t ≥ 0`.
///
/// The directions `d1`, `d2` need not be normalized. This follows the ray-vs-
/// ray recipe from Ericson's *Real-Time Collision Detection* (§5.1.9): solve
/// the unconstrained line system, clamp each parameter to `≥ 0`, and whenever a
/// clamp moved a parameter, re-project onto the *other* ray and clamp that too,
/// keeping the pair mutually closest under the constraints. When both origins
/// are the nearest approach (each ray points away from the other) the result is
/// `s = t = 0`, i.e. the two origins.
///
/// # Degeneracies
///
/// Zero-length directions are handled exactly as in [`line_line_closest`]: a
/// direction whose squared length is at or below [`EPS`] pins its parameter to
/// `0`, so a degenerate ray collapses to its origin without any unguarded
/// divide.
#[must_use]
pub fn ray_ray_closest(o1: [f32; 3], d1: [f32; 3], o2: [f32; 3], d2: [f32; 3]) -> ClosestLines {
    let r = v_sub(o1, o2);
    let a = v_dot(d1, d1);
    let b = v_dot(d1, d2);
    let c = v_dot(d2, d2);
    let d = v_dot(d1, r);
    let e = v_dot(d2, r);

    let denom = a * c - b * b;

    // First solve for `s` on ray A. When the lines are parallel the numerator
    // `b·e − c·d` is unreliable, so fall back to `s = 0`; otherwise take the
    // unconstrained line solution and clamp it onto the half-line `s ≥ 0`.
    let mut s = if denom.abs() > EPS * a * c {
        ((b * e - c * d) / denom).max(0.0)
    } else {
        0.0
    };

    // Re-project the current `s` onto ray B: `t = (b·s + e) / c` is the foot of
    // the perpendicular from `A(s)` onto line B. Guard the divide when B is
    // degenerate.
    let mut t = if c > EPS { (b * s + e) / c } else { 0.0 };

    // If that projection left the half-line `t ≥ 0`, clamp `t` and re-project
    // back onto ray A, then clamp `s` a final time. `t = (b·s − d) / a` is the
    // foot of the perpendicular from `B(t)` onto line A.
    if t < 0.0 {
        t = 0.0;
        s = if a > EPS { ((-d) / a).max(0.0) } else { 0.0 };
    }

    let point_on_a = v_add(o1, v_scale(d1, s));
    let point_on_b = v_add(o2, v_scale(d2, t));
    let distance = v_length(v_sub(point_on_a, point_on_b));

    ClosestLines {
        s,
        t,
        point_on_a,
        point_on_b,
        distance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for `f32` geometry assertions.
    const TOL: f32 = 1.0e-4;

    /// Asserts two scalars agree within [`TOL`] without an exact `f32` compare.
    fn approx(lhs: f32, rhs: f32) {
        assert!(
            (lhs - rhs).abs() <= TOL,
            "expected {lhs} ≈ {rhs} (|Δ| = {})",
            (lhs - rhs).abs()
        );
    }

    /// Asserts two points agree lane-by-lane within [`TOL`].
    fn approx_vec(lhs: [f32; 3], rhs: [f32; 3]) {
        approx(lhs[0], rhs[0]);
        approx(lhs[1], rhs[1]);
        approx(lhs[2], rhs[2]);
    }

    #[test]
    fn orthogonal_skew_lines_distance_and_points() {
        // Line A along +x through the origin; line B along +y raised to z = 2.
        // The mutual perpendicular is the z axis, gap = 2, feet at the origin
        // of each line's crossing over z.
        let r = line_line_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        );
        approx(r.distance, 2.0);
        approx_vec(r.point_on_a, [0.0, 0.0, 0.0]);
        approx_vec(r.point_on_b, [0.0, 0.0, 2.0]);
        approx(r.s, 0.0);
        approx(r.t, 0.0);
    }

    #[test]
    fn skew_lines_offset_feet() {
        // A along x through (0,0,0); B along y through (3,0,5). Nearest foot on
        // A is x = 3, on B is y = 0, gap = 5.
        let r = line_line_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 5.0],
            [0.0, 1.0, 0.0],
        );
        approx(r.distance, 5.0);
        approx_vec(r.point_on_a, [3.0, 0.0, 0.0]);
        approx_vec(r.point_on_b, [3.0, 0.0, 5.0]);
    }

    #[test]
    fn intersecting_lines_zero_distance() {
        // A along x, B along y, both through the origin: they intersect.
        let r = line_line_closest(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        approx(r.distance, 0.0);
        approx_vec(r.point_on_a, [0.0, 0.0, 0.0]);
        approx_vec(r.point_on_b, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn intersecting_lines_off_origin() {
        // Two lines crossing at (2, 3, 4).
        let cross = [2.0, 3.0, 4.0];
        let d1 = [1.0, 1.0, 0.0];
        let d2 = [0.0, 1.0, 1.0];
        let p1 = v_add(cross, v_scale(d1, -2.0));
        let p2 = v_add(cross, v_scale(d2, 3.0));
        let r = line_line_closest(p1, d1, p2, d2);
        approx(r.distance, 0.0);
        approx_vec(r.point_on_a, cross);
        approx_vec(r.point_on_b, cross);
    }

    #[test]
    fn parallel_lines_gap_and_s_branch() {
        // Both lines along x; B raised by (0, 4, 3): gap = 5. Parallel branch
        // pins s = 0, so point_on_a is exactly p1.
        let r = line_line_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 4.0, 3.0],
            [2.0, 0.0, 0.0],
        );
        approx(r.s, 0.0);
        approx(r.distance, 5.0);
        approx_vec(r.point_on_a, [0.0, 0.0, 0.0]);
        // Foot of perpendicular from origin onto B shares A's x = 0.
        approx(r.point_on_b[0], 0.0);
        approx(r.point_on_b[1], 4.0);
        approx(r.point_on_b[2], 3.0);
    }

    #[test]
    fn antiparallel_lines_gap() {
        // Opposite directions are still parallel; gap must be the line spacing.
        let r = line_line_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 7.0],
            [-3.0, 0.0, 0.0],
        );
        approx(r.s, 0.0);
        approx(r.distance, 7.0);
    }

    #[test]
    fn coincident_lines_zero_distance() {
        // Same supporting line, different base points and scaled direction.
        let r = line_line_closest(
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [9.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        );
        approx(r.distance, 0.0);
    }

    #[test]
    fn non_unit_direction_preserves_geometry() {
        // Scaling a direction must not move the geometric closest point, only
        // rescale the parameter that reaches it.
        let base = line_line_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
        );
        let scaled = line_line_closest(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.25, 0.0],
        );
        approx_vec(base.point_on_a, scaled.point_on_a);
        approx_vec(base.point_on_b, scaled.point_on_b);
        approx(base.distance, scaled.distance);
    }

    #[test]
    fn non_unit_direction_rescales_parameter() {
        // Foot on A is at x = 3 with a unit direction (s = 3); doubling the
        // direction halves s to 1.5 while the point stays put.
        let unit = line_line_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 5.0],
            [0.0, 1.0, 0.0],
        );
        let doubled = line_line_closest(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 5.0],
            [0.0, 1.0, 0.0],
        );
        approx(unit.s, 3.0);
        approx(doubled.s, 1.5);
        approx_vec(unit.point_on_a, doubled.point_on_a);
    }

    #[test]
    fn symmetry_swap_distance_and_points() {
        let p1 = [0.0, 0.0, 0.0];
        let d1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 0.0, 2.0];
        let d2 = [0.0, 1.0, 0.0];
        let ab = line_line_closest(p1, d1, p2, d2);
        let ba = line_line_closest(p2, d2, p1, d1);
        approx(ab.distance, ba.distance);
        // Swapping the arguments swaps the roles of the two feet.
        approx_vec(ab.point_on_a, ba.point_on_b);
        approx_vec(ab.point_on_b, ba.point_on_a);
    }

    #[test]
    fn symmetry_swap_skew_general() {
        let p1 = [1.0, 2.0, 3.0];
        let d1 = [1.0, 1.0, 0.0];
        let p2 = [-2.0, 0.0, 4.0];
        let d2 = [0.0, 1.0, 2.0];
        let ab = line_line_closest(p1, d1, p2, d2);
        let ba = line_line_closest(p2, d2, p1, d1);
        approx(ab.distance, ba.distance);
        approx_vec(ab.point_on_a, ba.point_on_b);
        approx_vec(ab.point_on_b, ba.point_on_a);
    }

    #[test]
    fn feet_are_mutually_perpendicular() {
        // For non-parallel lines the connecting segment is perpendicular to
        // both directions.
        let d1 = [1.0, 2.0, 0.0];
        let d2 = [0.0, 1.0, 3.0];
        let r = line_line_closest([0.0, 0.0, 0.0], d1, [4.0, 1.0, -2.0], d2);
        let diff = v_sub(r.point_on_a, r.point_on_b);
        approx(v_dot(diff, d1), 0.0);
        approx(v_dot(diff, d2), 0.0);
    }

    #[test]
    fn point_reconstruction_matches_parameters() {
        // point_on_a and point_on_b must equal p + param·d exactly.
        let p1 = [1.0, -2.0, 3.0];
        let d1 = [2.0, 1.0, -1.0];
        let p2 = [0.0, 4.0, 1.0];
        let d2 = [1.0, 0.0, 2.0];
        let r = line_line_closest(p1, d1, p2, d2);
        approx_vec(r.point_on_a, v_add(p1, v_scale(d1, r.s)));
        approx_vec(r.point_on_b, v_add(p2, v_scale(d2, r.t)));
    }

    #[test]
    fn degenerate_first_direction_collapses_to_point() {
        // A zero-length "line" A is just the point p1; distance is p1 to line B.
        let r = line_line_closest(
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        approx(r.s, 0.0);
        approx_vec(r.point_on_a, [0.0, 5.0, 0.0]);
        // Nearest point on the x axis to (0,5,0) is the origin; gap = 5.
        approx_vec(r.point_on_b, [0.0, 0.0, 0.0]);
        approx(r.distance, 5.0);
    }

    #[test]
    fn degenerate_both_directions_point_to_point() {
        let r = line_line_closest(
            [1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [4.0, 6.0, 2.0],
            [0.0, 0.0, 0.0],
        );
        approx(r.distance, 5.0);
        approx(r.s, 0.0);
        approx(r.t, 0.0);
    }

    #[test]
    fn v_length_matches_pythagoras() {
        approx(v_length([3.0, 4.0, 0.0]), 5.0);
        approx(v_length([2.0, 3.0, 6.0]), 7.0);
    }

    #[test]
    fn v_cross_is_perpendicular() {
        let a = [1.0, 2.0, 3.0];
        let b = [-2.0, 0.0, 5.0];
        let c = v_cross(a, b);
        approx(v_dot(c, a), 0.0);
        approx(v_dot(c, b), 0.0);
    }

    #[test]
    fn free_functions_basic_algebra() {
        approx_vec(v_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [5.0, 7.0, 9.0]);
        approx_vec(v_sub([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]), [3.0, 3.0, 3.0]);
        approx_vec(v_scale([1.0, -2.0, 3.0], 2.0), [2.0, -4.0, 6.0]);
        approx(v_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn ray_ray_facing_intersection() {
        // Two rays that meet ahead of both origins.
        let r = ray_ray_closest(
            [-2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        approx(r.distance, 0.0);
        approx_vec(r.point_on_a, [0.0, 0.0, 0.0]);
        assert!(r.s >= 0.0);
        assert!(r.t >= 0.0);
    }

    #[test]
    fn ray_ray_diverging_meets_at_origins() {
        // Both rays point away from their crossing: nearest approach is the two
        // origins, so both parameters clamp to 0.
        let r = ray_ray_closest(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        approx(r.s, 0.0);
        approx(r.t, 0.0);
        approx_vec(r.point_on_a, [1.0, 0.0, 0.0]);
        approx_vec(r.point_on_b, [0.0, 1.0, 0.0]);
        approx(r.distance, (2.0f32).sqrt());
    }

    #[test]
    fn ray_ray_one_param_clamped() {
        // A along +x from origin; B along +y but starting behind at y = -3 from
        // x = 5. As lines they cross at (5, 0); that is s = 5 (valid) and
        // t = 3 (valid), so both stay positive — verify the crossing.
        let r = ray_ray_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, -3.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        approx(r.distance, 0.0);
        approx(r.s, 5.0);
        approx(r.t, 3.0);
    }

    #[test]
    fn ray_ray_clamp_forces_origin() {
        // A along +x from origin; B along +y starting at (5, 2). The line
        // crossing needs t = -2 (behind B's origin), so t clamps to 0 and the
        // nearest point on A becomes the foot of B's origin, x = 5.
        let r = ray_ray_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        approx(r.t, 0.0);
        approx(r.s, 5.0);
        approx_vec(r.point_on_a, [5.0, 0.0, 0.0]);
        approx_vec(r.point_on_b, [5.0, 2.0, 0.0]);
        approx(r.distance, 2.0);
    }

    #[test]
    fn ray_ray_grazing_parallel() {
        // Parallel rays offset in z. As lines the perpendicular from B's
        // origin lands at s = 2 on A and stays on both half-lines, so the
        // distance is the z spacing (4) with s = 2, t = 0.
        let r = ray_ray_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 4.0],
            [1.0, 0.0, 0.0],
        );
        approx(r.s, 2.0);
        approx(r.t, 0.0);
        approx(r.distance, 4.0);
    }

    #[test]
    fn ray_ray_both_behind_clamps_both() {
        // Rays pointing away from each other in x from separated origins.
        let r = ray_ray_closest(
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [4.0, 3.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        approx(r.s, 0.0);
        approx(r.t, 0.0);
        approx(r.distance, 5.0);
    }

    #[test]
    fn ray_ray_degenerate_direction() {
        // A degenerate ray A collapses to its origin; distance is origin to B.
        let r = ray_ray_closest(
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        approx(r.s, 0.0);
        approx_vec(r.point_on_a, [0.0, 3.0, 0.0]);
        approx(r.distance, 3.0);
    }

    // --- Deterministic LCG fuzz --------------------------------------------

    /// Minimal 32-bit linear-congruential generator (Numerical Recipes
    /// constants) — stateless-hash-free but fully deterministic for tests.
    struct Lcg {
        state: u32,
    }

    impl Lcg {
        fn new(seed: u32) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            self.state
        }

        /// A float in the half-open range `[-5, 5)`.
        fn next_coord(&mut self) -> f32 {
            let unit = (self.next_u32() >> 8) as f32 / (1_u32 << 24) as f32;
            unit * 10.0 - 5.0
        }

        fn next_vec(&mut self) -> [f32; 3] {
            [self.next_coord(), self.next_coord(), self.next_coord()]
        }
    }

    #[test]
    fn fuzz_feet_perpendicular_for_non_parallel_lines() {
        let mut rng = Lcg::new(0x1234_5678);
        let mut checked = 0_u32;
        for _ in 0..400 {
            let p1 = rng.next_vec();
            let d1 = rng.next_vec();
            let p2 = rng.next_vec();
            let d2 = rng.next_vec();

            let a = v_dot(d1, d1);
            let c = v_dot(d2, d2);
            let b = v_dot(d1, d2);
            let denom = a * c - b * b;

            // Skip degenerate directions and near-parallel pairs, where the
            // perpendicularity property does not uniquely hold.
            if a <= EPS || c <= EPS || denom.abs() <= 1.0e-3 * a * c {
                continue;
            }

            let r = line_line_closest(p1, d1, p2, d2);
            let diff = v_sub(r.point_on_a, r.point_on_b);

            // Normalize the dot residual by the direction magnitude so the
            // tolerance is scale-aware across random inputs.
            let res1 = v_dot(diff, d1).abs() / v_length(d1).max(EPS);
            let res2 = v_dot(diff, d2).abs() / v_length(d2).max(EPS);
            assert!(res1 <= 1.0e-2, "diff·d1 residual too large: {res1}");
            assert!(res2 <= 1.0e-2, "diff·d2 residual too large: {res2}");
            checked += 1;
        }
        assert!(checked >= 200, "too few non-parallel samples: {checked}");
    }

    #[test]
    fn fuzz_distance_is_minimal() {
        // The reported distance must not exceed the gap at nearby parameter
        // perturbations — a numerical minimality check.
        let mut rng = Lcg::new(0x0BAD_F00D);
        let mut checked = 0_u32;
        for _ in 0..400 {
            let p1 = rng.next_vec();
            let d1 = rng.next_vec();
            let p2 = rng.next_vec();
            let d2 = rng.next_vec();

            let a = v_dot(d1, d1);
            let c = v_dot(d2, d2);
            if a <= EPS || c <= EPS {
                continue;
            }

            let r = line_line_closest(p1, d1, p2, d2);
            let base = r.distance;

            for &ds in &[-0.05_f32, 0.05] {
                for &dt in &[-0.05_f32, 0.05] {
                    let pa = v_add(p1, v_scale(d1, r.s + ds));
                    let pb = v_add(p2, v_scale(d2, r.t + dt));
                    let probe = v_length(v_sub(pa, pb));
                    assert!(
                        probe >= base - 1.0e-3,
                        "found closer probe {probe} < {base}"
                    );
                }
            }
            checked += 1;
        }
        assert!(checked >= 200, "too few samples: {checked}");
    }

    #[test]
    fn fuzz_symmetry() {
        let mut rng = Lcg::new(0xDEAD_BEEF);
        for _ in 0..250 {
            let p1 = rng.next_vec();
            let d1 = rng.next_vec();
            let p2 = rng.next_vec();
            let d2 = rng.next_vec();
            let ab = line_line_closest(p1, d1, p2, d2);
            let ba = line_line_closest(p2, d2, p1, d1);
            approx(ab.distance, ba.distance);
        }
    }
}
