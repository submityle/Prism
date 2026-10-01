//! Capsule-vs-capsule closest-distance, penetration, and contact query for the
//! particle subsystem's collision proximity math (design §10, §14).
//!
//! A *capsule* here is a swept sphere: a line segment (the *core axis*) from
//! `a0` to `a1` inflated by a radius `r`. This module answers the collision
//! question a broad phase and the response solver actually need — given two
//! capsules, are their *surfaces* touching, how deeply do they overlap, in what
//! direction should they be pushed apart, and where on each surface does the
//! contact sit?
//!
//! It is deliberately disjoint from its two nearest siblings:
//!
//! * [`crate::particle::segment_closest_point_3d`] answers the pure geometry
//!   question about two *infinitely thin* line segments: the parameters
//!   `(s, t)` of the mutually closest points and the squared distance between
//!   them. It has no notion of a radius, a penetration depth, a contact normal,
//!   or a surface point. This module *re-derives* that segment-vs-segment core
//!   locally (it does not `use` the sibling) and then lifts the result into the
//!   capsule world by subtracting the radius sum and projecting onto each
//!   surface. The two modules share an algorithm, not a purpose.
//! * [`crate::particle::capsule_sdf`] evaluates the analytic signed distance of
//!   a *single* capsule primitive from a *query point* (an `SDF`), for
//!   raymarching and metaball authoring. It never relates two capsules to each
//!   other. This module is a pairwise collision query, not a field sampler.
//!
//! Everything is a zero-dependency contract: the vector math is a set of
//! hand-rolled free functions in this file (`v_add`, `v_sub`, `v_dot`,
//! `v_cross`, `v_scale`, `v_length`) rather than operator-trait implementations,
//! and every step uses only `+ - * /`, `f32::sqrt`, `f32::abs`, `f32::min`,
//! `f32::max`, and `f32::clamp`. No transcendental function is ever called, and
//! no exact `==` / `!=` on a production `f32` ever appears — every parallel,
//! degenerate, and divide-by-zero case is guarded by [`EPS`] so no `NaN` can
//! escape. That keeps this `CPU` reference in agreement, bit for bit, with a
//! future `GPU` (`WESL`) kernel that packs the same capsules.

/// Epsilon that guards every division and every degeneracy test so the module
/// never writes an exact `==` / `!=` on a production `f32` and never emits a
/// `NaN`.
///
/// A segment direction whose squared length is at or below this value is a
/// point; a linear-system determinant at or below it is parallel; an axis
/// distance at or below it is treated as a coincident-axis degeneracy.
pub const EPS: f32 = 1.0e-6;

/// Component-wise vector sum `a + b`.
#[must_use]
pub fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise vector difference `a - b`.
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
///
/// Used by callers and tests that need the mutual perpendicular of two skew
/// axis directions; the closest-point solver itself never needs it.
#[must_use]
pub fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Uniform scale `a * s`.
#[must_use]
pub fn v_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean length `√(a · a)`.
#[must_use]
pub fn v_length(a: [f32; 3]) -> f32 {
    v_dot(a, a).sqrt()
}

/// The result of a capsule-vs-capsule closest query.
///
/// * `distance` is the signed gap between the two *surfaces*: `axis_dist -
///   (ra + rb)`. It is positive when the capsules are apart, zero when their
///   surfaces just touch, and negative when they overlap (the magnitude is the
///   overlap depth).
/// * `penetration` is `max(0, ra + rb - axis_dist)`, i.e. the non-negative
///   overlap depth (`0` whenever the capsules are separated or just touching).
/// * `normal` is the unit vector pointing *from capsule A toward capsule B*
///   along the core-axis closest-point connector. When the two axes are
///   effectively coincident (`axis_dist` at or below [`EPS`]) the direction is
///   undefined, so a stable default of `[0, 1, 0]` is returned instead.
/// * `point_a` is the closest point on capsule A's *surface* (its axis closest
///   point advanced by `ra` along `normal`), and `point_b` is the closest point
///   on capsule B's surface (its axis closest point retreated by `rb` along
///   `normal`).
/// * `intersecting` is `true` when `axis_dist <= ra + rb + EPS`, i.e. the
///   surfaces touch or overlap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleHit {
    /// Signed surface gap `axis_dist - (ra + rb)`; negative when overlapping.
    pub distance: f32,
    /// Non-negative overlap depth `max(0, ra + rb - axis_dist)`.
    pub penetration: f32,
    /// Unit contact normal from A to B (stable default when axes coincide).
    pub normal: [f32; 3],
    /// Closest point on capsule A's surface.
    pub point_a: [f32; 3],
    /// Closest point on capsule B's surface.
    pub point_b: [f32; 3],
    /// Whether the two capsule surfaces touch or overlap.
    pub intersecting: bool,
}

/// The stable fallback normal returned when the two core axes are effectively
/// coincident and no contact direction can be recovered from the geometry.
///
/// Any fixed unit vector is a valid choice for a fully coincident pair; `[0, 1,
/// 0]` is picked so the output stays deterministic and never contains a `NaN`.
const DEFAULT_NORMAL: [f32; 3] = [0.0, 1.0, 0.0];

/// Computes the mutually closest points of two 3D segments (the capsule core
/// axes), returning the clamped parameters `s ∈ [0, 1]` and `t ∈ [0, 1]` and
/// the closest points `pa = a0 + s·(a1 - a0)` and `pb = b0 + t·(b1 - b0)`.
///
/// This is the closed-form solver from Christer Ericson's *Real-Time Collision
/// Detection* (§5.1.9, `ClosestPtSegmentSegment`), re-derived here so the module
/// stays a self-contained zero-dependency contract. Writing `d1 = a1 - a0`,
/// `d2 = b1 - b0`, and `r = a0 - b0`, the coefficients are `a = d1·d1`,
/// `e = d2·d2`, `f = d2·r`, `b = d1·d2`, and `c = d1·r`; the line-line optimum
/// solves a 2×2 system whose determinant is `denom = a·e - b²`. Four regimes
/// are handled, all guarded by [`EPS`] so no division ever produces a `NaN`:
///
/// * **Both degenerate** — both segments are points; the answer is `s = t = 0`.
/// * **First degenerate** — segment A is a point; project it onto segment B.
/// * **Second degenerate** — segment B is a point; project it onto segment A.
/// * **General / parallel** — solve for `s` (clamped); when `denom` is at or
///   below [`EPS`] the axes are parallel, so `s` is pinned to `0`. Recover `t`,
///   and if it leaves `[0, 1]` clamp it and re-derive `s`, so both parameters
///   land inside `[0, 1]²`.
#[must_use]
pub fn segment_segment_closest(
    a0: [f32; 3],
    a1: [f32; 3],
    b0: [f32; 3],
    b1: [f32; 3],
) -> (f32, f32, [f32; 3], [f32; 3]) {
    let d1 = v_sub(a1, a0);
    let d2 = v_sub(b1, b0);
    let r = v_sub(a0, b0);

    let a = v_dot(d1, d1); // squared length of segment A
    let e = v_dot(d2, d2); // squared length of segment B
    let f = v_dot(d2, r);

    let first_degenerate = a <= EPS;
    let second_degenerate = e <= EPS;

    let (s, t) = if first_degenerate && second_degenerate {
        // Both segments collapse to points: nothing to project.
        (0.0, 0.0)
    } else if first_degenerate {
        // Segment A is a point; clamp its projection onto segment B.
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = v_dot(d1, r);
        if second_degenerate {
            // Segment B is a point; clamp its projection onto segment A.
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = v_dot(d1, d2);
            let denom = a * e - b * b;

            // Non-parallel: solve for the line-line optimum along A. Parallel
            // (denom at or below EPS): pin s and let the t recovery pick offset.
            let s_line = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Recover t for this s: t = (b·s + f) / e.
            let t_line = (b * s_line + f) / e;

            if t_line < 0.0 {
                // Clamp t to 0 and re-derive s = -c / a.
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if t_line > 1.0 {
                // Clamp t to 1 and re-derive s = (b - c) / a.
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s_line, t_line)
            }
        }
    };

    let pa = v_add(a0, v_scale(d1, s));
    let pb = v_add(b0, v_scale(d2, t));
    (s, t, pa, pb)
}

/// Computes the closest-distance / penetration / contact query between two
/// capsules: capsule A is segment `(a0, a1)` with radius `ra`, capsule B is
/// segment `(b0, b1)` with radius `rb`.
///
/// The core axes are handed to [`segment_segment_closest`] to get the axis
/// closest points and the axis distance `axis_dist`. From there:
///
/// * `distance = axis_dist - (ra + rb)` — the signed surface gap.
/// * `intersecting = axis_dist <= ra + rb + EPS`.
/// * `penetration = max(0, ra + rb - axis_dist)`.
/// * `normal` is the axis connector `pb - pa` normalized; when `axis_dist` is at
///   or below [`EPS`] the connector is degenerate, so the stable
///   [`DEFAULT_NORMAL`] `[0, 1, 0]` is used so no `NaN` escapes.
/// * `point_a = pa + normal·ra` and `point_b = pb - normal·rb` are the two
///   surface closest points.
///
/// The query is symmetric: swapping the two capsules leaves `distance` and
/// `penetration` unchanged, reverses `normal`, and swaps `point_a` / `point_b`.
#[must_use]
pub fn capsule_capsule_closest(
    a0: [f32; 3],
    a1: [f32; 3],
    ra: f32,
    b0: [f32; 3],
    b1: [f32; 3],
    rb: f32,
) -> CapsuleHit {
    let (_s, _t, pa, pb) = segment_segment_closest(a0, a1, b0, b1);

    let delta = v_sub(pb, pa); // from A's axis point toward B's axis point
    let axis_dist = v_length(delta);
    let radius_sum = ra + rb;

    let distance = axis_dist - radius_sum;
    let penetration = (radius_sum - axis_dist).max(0.0);
    let intersecting = axis_dist <= radius_sum + EPS;

    // Normalize the connector; fall back to a stable default when the axes are
    // effectively coincident (axis_dist at or below EPS) to avoid a NaN.
    let normal = if axis_dist > EPS {
        v_scale(delta, 1.0 / axis_dist)
    } else {
        DEFAULT_NORMAL
    };

    let point_a = v_add(pa, v_scale(normal, ra));
    let point_b = v_sub(pb, v_scale(normal, rb));

    CapsuleHit {
        distance,
        penetration,
        normal,
        point_a,
        point_b,
        intersecting,
    }
}

/// Computes the closest query between a capsule `(a0, a1)` with radius `ra` and
/// a sphere centered at `center` with radius `rs`.
///
/// A sphere is exactly a capsule whose core axis has collapsed to a point, so
/// this simply forwards to [`capsule_capsule_closest`] with `b0 == b1 ==
/// center`. Every field of the returned [`CapsuleHit`] carries the same meaning
/// as in the general case, with `point_b` sitting on the sphere's surface.
#[must_use]
pub fn capsule_sphere_closest(
    a0: [f32; 3],
    a1: [f32; 3],
    ra: f32,
    center: [f32; 3],
    rs: f32,
) -> CapsuleHit {
    capsule_capsule_closest(a0, a1, ra, center, center, rs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparisons that are not bit-exact.
    const TOL: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < TOL, "expected {b}, got {a}");
    }

    fn approx_vec(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    fn in_unit(v: f32) {
        assert!((0.0..=1.0).contains(&v), "parameter {v} out of [0, 1]");
    }

    // --- Vector math ------------------------------------------------------

    #[test]
    fn v_add_sums_components() {
        approx_vec(v_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn v_sub_differences_components() {
        approx_vec(v_sub([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]), [3.0, 3.0, 3.0]);
    }

    #[test]
    fn v_dot_matches_definition() {
        approx(v_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn v_cross_is_right_handed() {
        approx_vec(v_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn v_scale_multiplies_each_lane() {
        approx_vec(v_scale([1.0, -2.0, 3.0], 2.0), [2.0, -4.0, 6.0]);
    }

    #[test]
    fn v_length_is_euclidean() {
        approx(v_length([3.0, 4.0, 0.0]), 5.0);
    }

    // --- Capsule queries --------------------------------------------------

    #[test]
    fn parallel_separated_capsules_have_analytic_distance() {
        // Two x-aligned capsules 3 apart in y, each radius 0.5.
        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            0.5,
        );
        approx(hit.distance, 2.0); // 3 - (0.5 + 0.5)
        assert!(!hit.intersecting);
        approx(hit.penetration, 0.0);
        approx_vec(hit.normal, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn crossing_capsules_overlap() {
        // A along x, B along y offset by 0.5 in z; axes pass within 0.5.
        let hit = capsule_capsule_closest(
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.0, -2.0, 0.5],
            [0.0, 2.0, 0.5],
            0.5,
        );
        assert!(hit.intersecting);
        assert!(hit.penetration > 0.0);
        assert!(hit.distance < 0.0);
        approx(hit.distance, -0.5); // 0.5 - 1.0
        approx(hit.penetration, 0.5);
        approx_vec(hit.normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn endpoint_caps_are_closest() {
        // Collinear x capsules with a gap; closest points fall on the inner
        // endpoints (s = 1, t = 0).
        let (s, t, pa, pb) = segment_segment_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        );
        approx(s, 1.0);
        approx(t, 0.0);
        approx_vec(pa, [1.0, 0.0, 0.0]);
        approx_vec(pb, [5.0, 0.0, 0.0]);

        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            0.5,
        );
        approx(hit.distance, 3.0); // 4 - 1
        approx_vec(hit.normal, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn collinear_end_to_end_just_touch() {
        // Axis gap 1 exactly equals ra + rb: surfaces touch, no penetration.
        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            0.5,
        );
        assert!(hit.intersecting);
        approx(hit.distance, 0.0);
        approx(hit.penetration, 0.0);
    }

    #[test]
    fn collinear_overlapping_axes_use_default_normal() {
        // Overlapping x ranges drive axis_dist to 0, so the default normal is
        // returned and the penetration is the full radius sum.
        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            1.0,
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            1.0,
        );
        assert!(hit.intersecting);
        approx(hit.distance, -2.0);
        approx(hit.penetration, 2.0);
        approx_vec(hit.normal, DEFAULT_NORMAL);
    }

    #[test]
    fn capsule_degenerate_matches_sphere_helper() {
        // Capsule B collapsed to a point must equal the sphere helper.
        let center = [0.0, 4.0, 0.0];
        let general =
            capsule_capsule_closest([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5, center, center, 1.0);
        let helper = capsule_sphere_closest([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.5, center, 1.0);
        assert_eq!(general, helper);
        approx(helper.distance, 2.5); // 4 - (0.5 + 1.0)
        approx_vec(helper.normal, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn two_degenerate_capsules_are_sphere_sphere() {
        // Both axes are points: reduces to center distance minus radius sum.
        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            2.0,
        );
        approx(hit.distance, 2.0); // 5 - 3
        approx_vec(hit.normal, [0.0, 0.0, 1.0]);
        assert!(!hit.intersecting);
    }

    #[test]
    fn perpendicular_skew_axes() {
        // A along x at z = 0, B along y at z = 2; closest axis points differ
        // only in z, so axis_dist = 2.
        let hit = capsule_capsule_closest(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.3,
            [0.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
            0.3,
        );
        approx(hit.distance, 1.4); // 2 - 0.6
        approx_vec(hit.normal, [0.0, 0.0, 1.0]);
        assert!(!hit.intersecting);
    }

    #[test]
    fn contact_boundary_axis_dist_equals_radius_sum() {
        // Parallel capsules exactly ra + rb apart: touching, ~zero penetration.
        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.5,
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            0.5,
        );
        assert!(hit.intersecting);
        approx(hit.distance, 0.0);
        approx(hit.penetration, 0.0);
        approx_vec(hit.normal, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn swapping_capsules_is_symmetric() {
        let a0 = [-2.0, 0.0, 0.0];
        let a1 = [2.0, 0.0, 0.0];
        let ra = 0.5;
        let b0 = [0.0, -2.0, 0.5];
        let b1 = [0.0, 2.0, 0.5];
        let rb = 0.7;

        let ab = capsule_capsule_closest(a0, a1, ra, b0, b1, rb);
        let ba = capsule_capsule_closest(b0, b1, rb, a0, a1, ra);

        approx(ab.distance, ba.distance);
        approx(ab.penetration, ba.penetration);
        assert_eq!(ab.intersecting, ba.intersecting);
        // Normal reverses.
        approx_vec(ab.normal, v_scale(ba.normal, -1.0));
        // Surface points swap.
        approx_vec(ab.point_a, ba.point_b);
        approx_vec(ab.point_b, ba.point_a);
    }

    #[test]
    fn normal_is_unit_length_when_defined() {
        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            0.5,
        );
        approx(v_length(hit.normal), 1.0);
    }

    #[test]
    fn default_normal_is_unit_length() {
        approx(v_length(DEFAULT_NORMAL), 1.0);
    }

    #[test]
    fn capsule_sphere_closest_basic() {
        // Sphere sitting off the middle of a capsule's flank.
        let hit =
            capsule_sphere_closest([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], 0.5, [0.0, 3.0, 0.0], 1.0);
        approx(hit.distance, 1.5); // 3 - (0.5 + 1.0)
        approx_vec(hit.normal, [0.0, 1.0, 0.0]);
        // Surface point on the sphere is its center pulled back by rs along n.
        approx_vec(hit.point_b, [0.0, 2.0, 0.0]);
        // Surface point on the capsule is its axis point pushed out by ra.
        approx_vec(hit.point_a, [0.0, 0.5, 0.0]);
    }

    #[test]
    fn surface_points_are_radius_offsets_of_axis_points() {
        let a0 = [0.0, 0.0, 0.0];
        let a1 = [2.0, 0.0, 0.0];
        let ra = 0.5;
        let b0 = [0.0, 4.0, 0.0];
        let b1 = [2.0, 4.0, 0.0];
        let rb = 1.0;
        let (_s, _t, pa, pb) = segment_segment_closest(a0, a1, b0, b1);
        let hit = capsule_capsule_closest(a0, a1, ra, b0, b1, rb);
        // point_a is pa advanced by ra toward B; point_b is pb retreated by rb.
        approx_vec(hit.point_a, v_add(pa, v_scale(hit.normal, ra)));
        approx_vec(hit.point_b, v_sub(pb, v_scale(hit.normal, rb)));
    }

    #[test]
    fn penetration_is_never_negative() {
        // Separated pair still reports zero (not negative) penetration.
        let hit = capsule_capsule_closest(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.25,
            [0.0, 10.0, 0.0],
            [1.0, 10.0, 0.0],
            0.25,
        );
        assert!(hit.penetration >= 0.0);
        approx(hit.penetration, 0.0);
    }

    #[test]
    fn parameters_stay_in_unit_interval() {
        let (s, t, _pa, _pb) = segment_segment_closest(
            [-3.0, -1.0, 2.0],
            [4.0, 5.0, -1.0],
            [0.0, 7.0, 3.0],
            [6.0, -2.0, 8.0],
        );
        in_unit(s);
        in_unit(t);
    }

    // --- segment_segment_closest core ------------------------------------

    #[test]
    fn ss_orthogonal_skew_connector_is_perpendicular() {
        // A along x at z = 0, B along y at z = 3.
        let a0 = [-1.0, 0.0, 0.0];
        let a1 = [1.0, 0.0, 0.0];
        let b0 = [0.0, -1.0, 3.0];
        let b1 = [0.0, 1.0, 3.0];
        let (_s, _t, pa, pb) = segment_segment_closest(a0, a1, b0, b1);
        let connector = v_sub(pb, pa);
        // Perpendicular to both non-degenerate directions.
        approx(v_dot(connector, v_sub(a1, a0)), 0.0);
        approx(v_dot(connector, v_sub(b1, b0)), 0.0);
        approx(v_length(connector), 3.0);
    }

    #[test]
    fn ss_parallel_segments() {
        let (_s, _t, pa, pb) = segment_segment_closest(
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
        );
        // Parallel: connector is the perpendicular offset of length 2.
        approx(v_length(v_sub(pb, pa)), 2.0);
    }

    #[test]
    fn ss_intersecting_segments_meet() {
        // Two segments crossing at the origin.
        let (_s, _t, pa, pb) = segment_segment_closest(
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        );
        approx(v_length(v_sub(pb, pa)), 0.0);
        approx_vec(pa, [0.0, 0.0, 0.0]);
        approx_vec(pb, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn ss_both_degenerate_points() {
        let (s, t, pa, pb) = segment_segment_closest(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 6.0, 3.0],
            [4.0, 6.0, 3.0],
        );
        approx(s, 0.0);
        approx(t, 0.0);
        approx_vec(pa, [1.0, 2.0, 3.0]);
        approx_vec(pb, [4.0, 6.0, 3.0]);
        approx(v_length(v_sub(pb, pa)), 5.0);
    }

    #[test]
    fn ss_first_degenerate_point_projects_onto_second() {
        // A is a point above the middle of segment B.
        let (s, t, pa, pb) = segment_segment_closest(
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        );
        approx(s, 0.0);
        approx(t, 0.5);
        approx_vec(pa, [1.0, 2.0, 0.0]);
        approx_vec(pb, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn ss_second_degenerate_point_projects_onto_first() {
        let (s, t, pa, pb) = segment_segment_closest(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
        );
        approx(s, 0.5);
        approx(t, 0.0);
        approx_vec(pa, [1.0, 0.0, 0.0]);
        approx_vec(pb, [1.0, 3.0, 0.0]);
    }

    #[test]
    fn ss_connector_perpendicular_to_nondegenerate_dirs() {
        // A general skew configuration; the connector must be perpendicular to
        // each segment direction whenever the closest points are interior.
        let a0 = [0.0, 0.0, 0.0];
        let a1 = [3.0, 0.0, 0.0];
        let b0 = [1.0, 2.0, 1.0];
        let b1 = [1.0, 2.0, 4.0];
        let (s, t, pa, pb) = segment_segment_closest(a0, a1, b0, b1);
        let connector = v_sub(pb, pa);
        // Interior closest points here, so orthogonality holds.
        if (0.0..=1.0).contains(&s) && s > EPS && s < 1.0 - EPS {
            approx(v_dot(connector, v_sub(a1, a0)), 0.0);
        }
        if (0.0..=1.0).contains(&t) && t > EPS && t < 1.0 - EPS {
            approx(v_dot(connector, v_sub(b1, b0)), 0.0);
        }
    }

    // --- Randomized consistency ------------------------------------------

    #[test]
    fn random_axis_distance_matches_reconstruction() {
        // Deterministic LCG (Numerical Recipes constants) drives 240 cases.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Map the top bits into [-5, 5).
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            unit * 10.0 - 5.0
        };

        let mut count = 0u32;
        while count < 240 {
            let a0 = [next(), next(), next()];
            let a1 = [next(), next(), next()];
            let b0 = [next(), next(), next()];
            let b1 = [next(), next(), next()];
            let ra = (next().abs() * 0.2) + 0.05;
            let rb = (next().abs() * 0.2) + 0.05;

            let (_s, _t, pa, pb) = segment_segment_closest(a0, a1, b0, b1);
            let axis_dist_direct = v_length(v_sub(pb, pa));

            let hit = capsule_capsule_closest(a0, a1, ra, b0, b1, rb);
            let axis_dist_from_hit = hit.distance + ra + rb;

            approx(axis_dist_direct, axis_dist_from_hit);
            assert!(hit.penetration >= 0.0);
            assert!(!axis_dist_direct.is_nan());
            assert!(!hit.distance.is_nan());
            assert!(v_length(hit.normal) > 0.5); // always a real unit vector

            count += 1;
        }
    }
}
