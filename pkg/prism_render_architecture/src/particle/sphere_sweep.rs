//! Finite-volume moving-sphere continuous-collision detection (`CCD`) for the
//! particle broad phase (design §10, §13).
//!
//! This module answers one narrow, purely geometric question: given a sphere of
//! a fixed radius that moves at a constant velocity across the unit timestep
//! `t ∈ [0, 1]`, at what fraction of the step does it *first* touch another
//! moving sphere, an infinite plane, or a stationary point, and with what
//! contact normal? The result is expressed with [`SweepHit`]; the `toi`
//! (time-of-impact) field is that step fraction.
//!
//! Unlike a ray cast, the swept primitive here has a genuine radius, so it is a
//! *finite-volume* sweep rather than an infinitely thin line query. The two
//! moving spheres are reduced to a ray-versus-stationary-sphere problem by
//! subtracting the relative velocity and summing the radii, and the quadratic
//! `|d0 + t·dv|² = r²` is solved with the closed-form quadratic formula.
//!
//! # Relationship to the sibling modules (strict boundary)
//!
//! * [`crate::particle::ray_sphere`] is the *analytic ray-versus-sphere* test:
//!   an infinitely thin line, not a swept finite radius, and it has no unit-step
//!   `toi` parameterisation. This module always sweeps a real radius over
//!   `t ∈ [0, 1]`.
//! * [`crate::particle::sweep_aabb`] sweeps *axis-aligned boxes* with the slab
//!   method and never calls `sqrt`. This module sweeps *spheres* and *does* use
//!   `sqrt` to solve the quadratic.
//! * [`crate::particle::collision`] owns the *collision response* (push-out,
//!   restitution, friction). This module never integrates velocity or resolves
//!   a contact; it only reports *when* and *where* a first touch happens.
//!
//! # Determinism
//!
//! Everything is a zero-dependency contract with hand-rolled vector math built
//! from the free functions [`v_add`], [`v_sub`], [`v_dot`], [`v_cross`], and
//! [`v_scale`]. The only floating-point primitives used are `+ - * /`,
//! [`f32::sqrt`], [`f32::abs`], [`f32::min`], [`f32::max`], and
//! [`f32::clamp`]. No transcendental function is ever called, so the `CPU`
//! reference here agrees with a future `GPU` kernel evaluating the same query.
//! Exact `==` / `!=` on an `f32` is never written: every discriminant,
//! parallel, and degeneracy test is guarded against [`EPS`], so a genuinely
//! degenerate input yields a stable fallback rather than a `NaN`.

/// Absolute tolerance used to guard divisions, classify the quadratic
/// discriminant, detect a near-parallel plane sweep, and reject a near-zero
/// direction without ever writing an exact `==` / `!=` on an `f32`.
pub const EPS: f32 = 1.0e-6;

/// The outcome of a swept-sphere query.
///
/// A [`SweepHit::Hit`] carries the `toi` (time-of-impact) as a fraction of the
/// unit timestep, the outward contact `normal` (a unit vector), and the world
/// contact `point` on the shared surface at the moment of first touch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SweepHit {
    /// The swept sphere never touches the target over `t ∈ [0, 1]`.
    Miss,
    /// The swept sphere first touches the target at `toi`.
    Hit {
        /// Time-of-impact as a fraction of the unit step, in `[0, 1]`.
        toi: f32,
        /// Outward unit contact normal at the moment of first touch.
        normal: [f32; 3],
        /// World contact point on the shared surface at the moment of touch.
        point: [f32; 3],
    },
}

/// Component-wise sum `a + b`.
#[must_use]
pub fn v_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise difference `a - b`.
#[must_use]
pub fn v_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot (inner) product `a · b`.
#[must_use]
pub fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a × b` (right-handed).
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

/// Euclidean length `‖v‖`, computed with [`f32::sqrt`].
#[must_use]
pub fn v_length(v: [f32; 3]) -> f32 {
    v_dot(v, v).sqrt()
}

/// Returns `v` normalised to unit length, or `fallback` when `v` is shorter
/// than [`EPS`] (so a near-zero vector never produces a `NaN`).
#[must_use]
pub fn v_normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = v_length(v);
    if len < EPS {
        fallback
    } else {
        v_scale(v, 1.0 / len)
    }
}

/// Swept continuous-collision test between two spheres, each moving at a
/// constant velocity over `t ∈ [0, 1]`.
///
/// The pair is reduced to a moving point against a stationary sphere of radius
/// `r = radius_a + radius_b`: with relative start `d0 = center_a - center_b`
/// and relative velocity `dv = vel_a - vel_b`, the first touch is the smallest
/// root of `|d0 + t·dv|² = r²` in `[0, 1]`. An initial overlap (start distance
/// below `r`) is reported as a `toi` of `0` with a separation normal; a
/// degenerate coincident overlap uses a stable default normal. The hit normal
/// is the unit vector along the two centres at the contact instant, pointing
/// from `center_b` toward `center_a`.
#[must_use]
pub fn sweep_sphere_vs_sphere(
    center_a: [f32; 3],
    radius_a: f32,
    vel_a: [f32; 3],
    center_b: [f32; 3],
    radius_b: f32,
    vel_b: [f32; 3],
) -> SweepHit {
    let d0 = v_sub(center_a, center_b);
    let dv = v_sub(vel_a, vel_b);
    let r = radius_a + radius_b;
    let r_sq = r * r;

    let d0_sq = v_dot(d0, d0);
    let c = d0_sq - r_sq;

    // Already overlapping (or exactly touching): report an immediate contact.
    if c <= EPS {
        let normal = v_normalize_or(d0, [1.0, 0.0, 0.0]);
        let point = v_sub(center_a, v_scale(normal, radius_a));
        return SweepHit::Hit {
            toi: 0.0,
            normal,
            point,
        };
    }

    let a = v_dot(dv, dv);
    // No relative motion while separated: the pair can never meet.
    if a < EPS {
        return SweepHit::Miss;
    }

    let b = 2.0 * v_dot(d0, dv);
    let disc = b * b - 4.0 * a * c;
    if disc < -EPS {
        return SweepHit::Miss;
    }

    // Smaller root is the entry (first-touch) time; `a > 0` here.
    let sqrt_disc = disc.max(0.0).sqrt();
    let toi = (-b - sqrt_disc) / (2.0 * a);
    if !(0.0..=1.0).contains(&toi) {
        return SweepHit::Miss;
    }

    let d_t = v_add(d0, v_scale(dv, toi));
    let normal = v_normalize_or(d_t, [1.0, 0.0, 0.0]);
    let center_a_t = v_add(center_a, v_scale(vel_a, toi));
    let point = v_sub(center_a_t, v_scale(normal, radius_a));
    SweepHit::Hit { toi, normal, point }
}

/// Swept continuous-collision test between a moving sphere and an infinite
/// plane.
///
/// The plane is `dot(plane_normal_unit, x) = plane_d` with a unit normal. The
/// signed distance of the centre is `s0 = dot(n, center) - plane_d` and its
/// rate of change along the motion is `sn = dot(n, vel)`. The sphere touches
/// when the centre's signed distance reaches `±radius` on the approaching side.
/// A near-parallel sweep (`|sn|` below [`EPS`]) that is not already in contact
/// misses; an initial contact (`|s0| ≤ radius`) is a `toi` of `0`; a receding
/// sweep misses. The hit normal is the plane normal oriented toward the sphere,
/// and the contact point is the centre at impact projected onto the plane.
#[must_use]
pub fn sweep_sphere_vs_plane(
    center: [f32; 3],
    radius: f32,
    vel: [f32; 3],
    plane_normal_unit: [f32; 3],
    plane_d: f32,
) -> SweepHit {
    let n = plane_normal_unit;
    let s0 = v_dot(n, center) - plane_d;
    let sn = v_dot(n, vel);

    // Orient the contact normal toward the side the centre currently lies on.
    let side = if s0 >= 0.0 { 1.0 } else { -1.0 };
    let normal = v_scale(n, side);

    // Already touching or penetrating: immediate contact.
    if s0.abs() <= radius + EPS {
        let point = v_sub(center, v_scale(normal, radius));
        return SweepHit::Hit {
            toi: 0.0,
            normal,
            point,
        };
    }

    // Near-parallel sweep with the centre farther than the radius never meets.
    if sn.abs() < EPS {
        return SweepHit::Miss;
    }

    // First touch happens when the signed distance reaches `±radius`.
    let target = side * radius;
    let toi = (target - s0) / sn;
    if !(0.0..=1.0).contains(&toi) {
        return SweepHit::Miss;
    }

    let center_t = v_add(center, v_scale(vel, toi));
    let point = v_sub(center_t, v_scale(normal, radius));
    SweepHit::Hit { toi, normal, point }
}

/// Swept continuous-collision test between a moving sphere and a stationary
/// point.
///
/// This is the degenerate case of [`sweep_sphere_vs_sphere`] where the target
/// is a zero-radius stationary sphere, so it forwards to that solver directly
/// and inherits its `toi`, normal, and contact-point conventions.
#[must_use]
pub fn sweep_sphere_vs_point(
    center: [f32; 3],
    radius: f32,
    vel: [f32; 3],
    point: [f32; 3],
) -> SweepHit {
    sweep_sphere_vs_sphere(center, radius, vel, point, 0.0, [0.0, 0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` assertions in these tests.
    const T: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < T
    }

    fn approx_vec(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    fn unwrap_hit(hit: SweepHit) -> (f32, [f32; 3], [f32; 3]) {
        match hit {
            SweepHit::Hit { toi, normal, point } => (toi, normal, point),
            SweepHit::Miss => panic!("expected a hit but got a miss"),
        }
    }

    // ----- free-function vector math -----

    #[test]
    fn v_add_sub_roundtrip() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.5, -1.0, 4.0];
        assert!(approx_vec(v_add(a, b), [1.5, 1.0, 7.0]));
        assert!(approx_vec(v_sub(a, b), [0.5, 3.0, -1.0]));
    }

    #[test]
    fn v_dot_and_scale() {
        assert!(approx(v_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0));
        assert!(approx_vec(v_scale([1.0, -2.0, 3.0], 2.0), [2.0, -4.0, 6.0]));
    }

    #[test]
    fn v_cross_right_handed() {
        assert!(approx_vec(
            v_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            [0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn v_length_of_345() {
        assert!(approx(v_length([3.0, 4.0, 0.0]), 5.0));
    }

    #[test]
    fn v_normalize_or_uses_fallback_for_zero() {
        assert!(approx_vec(
            v_normalize_or([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn v_normalize_or_scales_to_unit() {
        assert!(approx_vec(
            v_normalize_or([0.0, 10.0, 0.0], [1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0]
        ));
    }

    // ----- sphere versus sphere -----

    #[test]
    fn sphere_head_on_collision() {
        let hit = sweep_sphere_vs_sphere(
            [-5.0, 0.0, 0.0],
            1.0,
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            1.0,
            [-10.0, 0.0, 0.0],
        );
        let (toi, normal, point) = unwrap_hit(hit);
        assert!(approx(toi, 0.4));
        assert!(approx_vec(normal, [-1.0, 0.0, 0.0]));
        // A at t=0.4 is at x = -5 + 4 = -1; contact point at x = -1 - (-1) = 0.
        assert!(approx_vec(point, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn sphere_grazing_miss() {
        let hit = sweep_sphere_vs_sphere(
            [-5.0, 3.0, 0.0],
            1.0,
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            1.0,
            [-10.0, 0.0, 0.0],
        );
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn sphere_receding_miss() {
        let hit = sweep_sphere_vs_sphere(
            [-5.0, 0.0, 0.0],
            1.0,
            [-10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            1.0,
            [10.0, 0.0, 0.0],
        );
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn sphere_initial_overlap_toi_zero() {
        let hit = sweep_sphere_vs_sphere(
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
        );
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 0.0));
        // Separation normal points from B toward A (negative x here).
        assert!(approx_vec(normal, [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn sphere_coincident_overlap_default_normal() {
        let hit = sweep_sphere_vs_sphere(
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
        );
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 0.0));
        assert!(approx_vec(normal, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn sphere_relative_different_speeds() {
        let hit = sweep_sphere_vs_sphere(
            [-10.0, 0.0, 0.0],
            1.0,
            [20.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            [5.0, 0.0, 0.0],
        );
        let (toi, _normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 8.0 / 15.0));
    }

    #[test]
    fn sphere_tangent_double_root() {
        // Vertical offset equal to the radius sum gives a single grazing root.
        let hit = sweep_sphere_vs_sphere(
            [-5.0, 2.0, 0.0],
            1.0,
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            1.0,
            [-10.0, 0.0, 0.0],
        );
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 0.5));
        assert!(approx_vec(normal, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn sphere_boundary_toi_one() {
        let hit = sweep_sphere_vs_sphere(
            [-5.0, 0.0, 0.0],
            1.0,
            [8.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            1.0,
            [-0.0, 0.0, 0.0],
        );
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 1.0));
        assert!(approx_vec(normal, [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn sphere_just_past_toi_one_miss() {
        let hit = sweep_sphere_vs_sphere(
            [-5.0, 0.0, 0.0],
            1.0,
            [7.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
        );
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn sphere_zero_relative_velocity_separated_miss() {
        let hit = sweep_sphere_vs_sphere(
            [0.0, 0.0, 0.0],
            1.0,
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            1.0,
            [1.0, 0.0, 0.0],
        );
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn sphere_zero_relative_velocity_touching_toi_zero() {
        let hit = sweep_sphere_vs_sphere(
            [0.0, 0.0, 0.0],
            1.0,
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            1.0,
            [3.0, 0.0, 0.0],
        );
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 0.0));
        assert!(approx_vec(normal, [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn sphere_boundary_toi_zero_start() {
        // Only sphere A moves; contact reached partway. Confirms t=0 sampling.
        let hit = sweep_sphere_vs_sphere(
            [-3.0, 0.0, 0.0],
            1.0,
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
        );
        let (toi, _normal, _point) = unwrap_hit(hit);
        // Contact when -3 + 4t = -2 => t = 0.25.
        assert!(approx(toi, 0.25));
    }

    #[test]
    fn sphere_contact_point_on_both_surfaces() {
        let hit = sweep_sphere_vs_sphere(
            [-5.0, 0.0, 0.0],
            1.0,
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            2.0,
            [-10.0, 0.0, 0.0],
        );
        let (toi, normal, point) = unwrap_hit(hit);
        // A center at toi and B center at toi; point should be radius_a from A.
        let a_t = v_add([-5.0, 0.0, 0.0], v_scale([10.0, 0.0, 0.0], toi));
        let b_t = v_add([5.0, 0.0, 0.0], v_scale([-10.0, 0.0, 0.0], toi));
        assert!(approx(v_length(v_sub(point, a_t)), 1.0));
        assert!(approx(v_length(v_sub(point, b_t)), 2.0));
        assert!(approx_vec(normal, [-1.0, 0.0, 0.0]));
    }

    // ----- sphere versus plane -----

    #[test]
    fn plane_front_approach() {
        let hit = sweep_sphere_vs_plane(
            [0.0, 5.0, 0.0],
            1.0,
            [0.0, -10.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        );
        let (toi, normal, point) = unwrap_hit(hit);
        assert!(approx(toi, 0.4));
        assert!(approx_vec(normal, [0.0, 1.0, 0.0]));
        assert!(approx_vec(point, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn plane_parallel_miss() {
        let hit =
            sweep_sphere_vs_plane([0.0, 5.0, 0.0], 1.0, [10.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn plane_already_contacting_toi_zero() {
        let hit = sweep_sphere_vs_plane(
            [0.0, 0.5, 0.0],
            1.0,
            [0.0, -10.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        );
        let (toi, normal, point) = unwrap_hit(hit);
        assert!(approx(toi, 0.0));
        assert!(approx_vec(normal, [0.0, 1.0, 0.0]));
        assert!(approx_vec(point, [0.0, -0.5, 0.0]));
    }

    #[test]
    fn plane_receding_miss() {
        let hit =
            sweep_sphere_vs_plane([0.0, 5.0, 0.0], 1.0, [0.0, 10.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn plane_grazing_at_radius_toi_zero() {
        // Centre exactly one radius away and moving parallel: an immediate touch.
        let hit =
            sweep_sphere_vs_plane([0.0, 1.0, 0.0], 1.0, [10.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0);
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 0.0));
        assert!(approx_vec(normal, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn plane_approach_from_negative_side() {
        let hit = sweep_sphere_vs_plane(
            [0.0, -5.0, 0.0],
            1.0,
            [0.0, 10.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        );
        let (toi, normal, point) = unwrap_hit(hit);
        assert!(approx(toi, 0.4));
        assert!(approx_vec(normal, [0.0, -1.0, 0.0]));
        assert!(approx_vec(point, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn plane_offset_normal_and_d() {
        let hit = sweep_sphere_vs_plane(
            [10.0, 0.0, 0.0],
            1.0,
            [-10.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            2.0,
        );
        let (toi, normal, point) = unwrap_hit(hit);
        assert!(approx(toi, 0.7));
        assert!(approx_vec(normal, [1.0, 0.0, 0.0]));
        assert!(approx_vec(point, [2.0, 0.0, 0.0]));
    }

    #[test]
    fn plane_boundary_toi_one() {
        let hit = sweep_sphere_vs_plane(
            [0.0, 11.0, 0.0],
            1.0,
            [0.0, -10.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        );
        let (toi, _normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 1.0));
    }

    #[test]
    fn plane_just_past_toi_one_miss() {
        let hit = sweep_sphere_vs_plane(
            [0.0, 12.0, 0.0],
            1.0,
            [0.0, -10.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        );
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn plane_contact_point_lies_on_plane() {
        let hit = sweep_sphere_vs_plane(
            [0.0, 8.0, 0.0],
            1.5,
            [0.0, -12.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        );
        let (_toi, _normal, point) = unwrap_hit(hit);
        // Point on the plane y = 0 satisfies dot(n, point) - d = 0.
        assert!(approx(v_dot([0.0, 1.0, 0.0], point) - 0.0, 0.0));
    }

    // ----- sphere versus point -----

    #[test]
    fn point_matches_sphere_zero_radius_target() {
        let center = [-5.0, 0.0, 0.0];
        let radius = 1.0;
        let vel = [10.0, 0.0, 0.0];
        let p = [0.0, 0.0, 0.0];
        let via_point = sweep_sphere_vs_point(center, radius, vel, p);
        let via_sphere = sweep_sphere_vs_sphere(center, radius, vel, p, 0.0, [0.0, 0.0, 0.0]);
        assert_eq!(via_point, via_sphere);
    }

    #[test]
    fn point_exact_toi() {
        let hit = sweep_sphere_vs_point([-5.0, 0.0, 0.0], 1.0, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 0.4));
        assert!(approx_vec(normal, [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn point_receding_miss() {
        let hit = sweep_sphere_vs_point([-5.0, 0.0, 0.0], 1.0, [-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert_eq!(hit, SweepHit::Miss);
    }

    #[test]
    fn point_inside_radius_toi_zero() {
        let hit = sweep_sphere_vs_point([0.0, 0.0, 0.0], 2.0, [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
        let (toi, _normal, _point) = unwrap_hit(hit);
        assert!(approx(toi, 0.0));
    }

    #[test]
    fn sphere_diagonal_motion_hit() {
        // A closes on B diagonally; verify a valid in-range toi and unit normal.
        let hit = sweep_sphere_vs_sphere(
            [-4.0, -4.0, 0.0],
            1.0,
            [8.0, 8.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
        );
        let (toi, normal, _point) = unwrap_hit(hit);
        assert!((0.0..=1.0).contains(&toi));
        assert!(approx(v_length(normal), 1.0));
    }
}
