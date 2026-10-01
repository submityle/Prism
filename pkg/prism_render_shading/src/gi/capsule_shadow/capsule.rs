//! Capsule-proxy soft shadows and AO via closest-point reduction to a sphere.
//!
//! A *capsule* is a line segment `[a, b]` swollen by a radius — the standard
//! bounding proxy for a character limb or torso in real-time self-shadowing
//! (Unreal-style capsule shadows).  Rather than integrate over the whole swept
//! volume, this module reduces the capsule to the single [`Sphere`] that best
//! represents it for a given query, then reuses the exact sphere references in
//! [`crate::gi::capsule_shadow::sphere`]:
//!
//! * For a **soft shadow** the relevant slice of the capsule is the one nearest
//!   the shadow ray `receiver -> light`.  [`closest_point_on_segment_to_line`]
//!   finds that slice centre on the capsule axis, and the capsule radius is the
//!   sphere radius ([`capsule_shadow_sphere`]).
//! * For **ambient occlusion** the relevant slice is the one nearest the
//!   receiver, found by the point-to-segment projection
//!   [`closest_point_on_segment`] ([`capsule_ao_sphere`]).
//!
//! The reduction is exact at the endpoints and axis, and degrades gracefully:
//! a capsule whose endpoints coincide (`a == b`) is exactly a sphere, and all
//! closest-point queries fall back to that shared point.
//!
//! # Conventions
//! * Positions are right-handed `Vec3` in a shared world space; distances are
//!   world units.
//! * Soft-shadow and AO results are *occlusion* fractions in `[0, 1]` (`0`
//!   lit/open, `1` shadowed/occluded), matching the sphere module.
//! * The segment parameter `t` is clamped to `[0, 1]`, so every closest point
//!   lies on the finite segment, never on its infinite extension.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no allocation, no global state.  Degenerate inputs (`a == b`, a zero
//!   shadow-ray direction, a non-positive radius) are handled explicitly and
//!   never produce `NaN` or infinities.

use crate::gi::capsule_shadow::sphere::{
    sphere_ambient_occlusion, sphere_soft_shadow, DiskLight, Sphere,
};
use bevy_math::Vec3;

/// Squared length below which a segment or direction is treated as degenerate.
const EPS2: f32 = 1.0e-12;

/// A capsule occluder: a core segment `[a, b]` inflated by `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// First endpoint of the core segment.
    pub a: Vec3,
    /// Second endpoint of the core segment.
    pub b: Vec3,
    /// Capsule radius.  Non-positive radii occlude nothing.
    pub radius: f32,
}

impl Capsule {
    /// Builds a capsule, clamping the radius to be non-negative.
    #[inline]
    pub fn new(a: Vec3, b: Vec3, radius: f32) -> Self {
        Self {
            a,
            b,
            radius: radius.max(0.0),
        }
    }

    /// `true` when the two endpoints coincide, so the capsule is a sphere.
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        (self.b - self.a).length_squared() <= EPS2
    }
}

/// Closest point to `p` on the finite segment `[a, b]`.
///
/// Projects `p` onto the segment's supporting line and clamps the parameter to
/// `[0, 1]`, so the result is an endpoint when the projection falls outside the
/// segment.  A degenerate segment (`a == b`) returns `a`.
#[inline]
pub fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= EPS2 {
        return a;
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    a + ab * t
}

/// Closest point on the finite segment `[a, b]` to the infinite line through
/// `origin` with direction `dir`.
///
/// Solves the two-parameter least-squares system for the closest pair between
/// the segment's line and the query line, then clamps the segment parameter to
/// `[0, 1]`.  Falls back to [`closest_point_on_segment`] when the query line is
/// degenerate (`dir ~ 0`), and to `a` when the segment itself is degenerate.
/// Parallel lines resolve to the segment start.
#[inline]
pub fn closest_point_on_segment_to_line(a: Vec3, b: Vec3, origin: Vec3, dir: Vec3) -> Vec3 {
    let u = b - a;
    let uu = u.length_squared();
    if uu <= EPS2 {
        return a;
    }
    let vv = dir.length_squared();
    if vv <= EPS2 {
        // No usable ray direction: fall back to nearest point to the origin.
        return closest_point_on_segment(origin, a, b);
    }
    let r = a - origin;
    let uv = u.dot(dir);
    let ur = u.dot(r);
    let vr = dir.dot(r);
    let denom = uu * vv - uv * uv;
    let s = if denom <= EPS2 {
        // Segment and query line are (near) parallel: pin to the start.
        0.0
    } else {
        ((uv * vr - vv * ur) / denom).clamp(0.0, 1.0)
    };
    let s = s.clamp(0.0, 1.0);
    a + u * s
}

/// Effective sphere a capsule reduces to for the shadow ray `receiver ->
/// light`: centred on the capsule-axis point nearest that ray, with the
/// capsule radius.
#[inline]
pub fn capsule_shadow_sphere(capsule: Capsule, receiver: Vec3, light_position: Vec3) -> Sphere {
    let center = closest_point_on_segment_to_line(
        capsule.a,
        capsule.b,
        receiver,
        light_position - receiver,
    );
    Sphere::new(center, capsule.radius)
}

/// Effective sphere a capsule reduces to for ambient occlusion at `receiver`:
/// centred on the capsule-axis point nearest the receiver, with the capsule
/// radius.
#[inline]
pub fn capsule_ao_sphere(capsule: Capsule, receiver: Vec3) -> Sphere {
    let center = closest_point_on_segment(receiver, capsule.a, capsule.b);
    Sphere::new(center, capsule.radius)
}

/// Soft-shadow occlusion fraction a single capsule casts onto `receiver` from a
/// disk light, in `[0, 1]` (`0` lit, `1` shadowed).
///
/// Reduces the capsule to its [`capsule_shadow_sphere`] and defers to the exact
/// [`sphere_soft_shadow`] model.
#[inline]
pub fn capsule_soft_shadow(receiver: Vec3, light: DiskLight, capsule: Capsule) -> f32 {
    if capsule.radius <= 0.0 {
        return 0.0;
    }
    let sphere = capsule_shadow_sphere(capsule, receiver, light.position);
    sphere_soft_shadow(receiver, light, sphere)
}

/// Ambient-occlusion fraction a single capsule contributes at `receiver` for a
/// surface normal, in `[0, 1]` (`0` open, `1` occluded).
///
/// Reduces the capsule to its [`capsule_ao_sphere`] and defers to the
/// closed-form [`sphere_ambient_occlusion`].
#[inline]
pub fn capsule_ambient_occlusion(receiver: Vec3, normal: Vec3, capsule: Capsule) -> f32 {
    if capsule.radius <= 0.0 {
        return 0.0;
    }
    let sphere = capsule_ao_sphere(capsule, receiver);
    sphere_ambient_occlusion(receiver, normal, sphere)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < TOL
    }

    #[test]
    fn closest_point_hits_first_endpoint() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        // Query far to the -x side projects before `a`.
        let p = Vec3::new(-5.0, 2.0, 0.0);
        assert!(close(closest_point_on_segment(p, a, b), a));
    }

    #[test]
    fn closest_point_hits_second_endpoint() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let p = Vec3::new(5.0, -3.0, 0.0);
        assert!(close(closest_point_on_segment(p, a, b), b));
    }

    #[test]
    fn closest_point_hits_midpoint() {
        let a = Vec3::new(-2.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        // Directly above the centre projects to the midpoint.
        let p = Vec3::new(0.0, 3.0, 0.0);
        assert!(close(closest_point_on_segment(p, a, b), Vec3::ZERO));
    }

    #[test]
    fn closest_point_for_off_axis_query() {
        let a = Vec3::ZERO;
        let b = Vec3::new(4.0, 0.0, 0.0);
        // Projection of (1, 5, 0) lands at x = 1 on the segment.
        let got = closest_point_on_segment(Vec3::new(1.0, 5.0, 0.0), a, b);
        assert!(close(got, Vec3::new(1.0, 0.0, 0.0)), "got {got:?}");
    }

    #[test]
    fn closest_point_degenerate_segment_returns_a() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        assert!(close(closest_point_on_segment(Vec3::new(9.0, 9.0, 9.0), a, a), a));
    }

    #[test]
    fn closest_point_to_line_recovers_axis_crossing() {
        // Segment along x through the origin; query ray along y offset at x = 1.
        let a = Vec3::new(-3.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 0.0, 0.0);
        let origin = Vec3::new(1.0, -5.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        let got = closest_point_on_segment_to_line(a, b, origin, dir);
        assert!(close(got, Vec3::new(1.0, 0.0, 0.0)), "got {got:?}");
    }

    #[test]
    fn closest_point_to_line_clamps_to_endpoint() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        // Ray crossing the x axis at x = 10 clamps to endpoint `b`.
        let origin = Vec3::new(10.0, -5.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        let got = closest_point_on_segment_to_line(a, b, origin, dir);
        assert!(close(got, b), "got {got:?}");
    }

    #[test]
    fn closest_point_to_line_degenerate_dir_falls_back() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let got = closest_point_on_segment_to_line(a, b, Vec3::new(0.0, 4.0, 0.0), Vec3::ZERO);
        assert!(close(got, Vec3::ZERO), "got {got:?}");
    }

    #[test]
    fn degenerate_capsule_matches_sphere_shadow() {
        // a == b: the capsule reduction must equal a sphere at that point.
        let p = Vec3::new(0.0, 0.0, 3.0);
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 12.0), 0.5);
        let capsule = Capsule::new(p, p, 0.8);
        let sphere = Sphere::new(p, 0.8);
        let via_capsule = capsule_soft_shadow(Vec3::ZERO, light, capsule);
        let via_sphere = sphere_soft_shadow(Vec3::ZERO, light, sphere);
        assert!((via_capsule - via_sphere).abs() < TOL, "{via_capsule} {via_sphere}");
    }

    #[test]
    fn degenerate_capsule_matches_sphere_ao() {
        let p = Vec3::new(0.0, 2.5, 0.0);
        let capsule = Capsule::new(p, p, 1.0);
        let sphere = Sphere::new(p, 1.0);
        let via_capsule = capsule_ambient_occlusion(Vec3::ZERO, Vec3::Y, capsule);
        let via_sphere = sphere_ambient_occlusion(Vec3::ZERO, Vec3::Y, sphere);
        assert!((via_capsule - via_sphere).abs() < TOL, "{via_capsule} {via_sphere}");
    }

    #[test]
    fn capsule_shadow_reduces_to_nearest_axis_slice() {
        // A long vertical capsule in front of the receiver; the shadow sphere
        // should sit near the shadow ray height, giving a real occlusion.
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 15.0), 0.5);
        let capsule = Capsule::new(
            Vec3::new(0.0, -5.0, 3.0),
            Vec3::new(0.0, 5.0, 3.0),
            0.7,
        );
        let occ = capsule_soft_shadow(Vec3::ZERO, light, capsule);
        assert!(occ > 0.0 && occ <= 1.0, "occ = {occ}");
    }

    #[test]
    fn capsule_shadow_in_unit_range() {
        let light = DiskLight::new(Vec3::new(1.0, 1.0, 10.0), 0.6);
        let capsule = Capsule::new(
            Vec3::new(-1.0, -2.0, 3.0),
            Vec3::new(1.0, 2.0, 4.0),
            0.5,
        );
        for z in [1.0f32, 2.0, 5.0] {
            let occ = capsule_soft_shadow(Vec3::new(0.0, 0.0, z), light, capsule);
            assert!((0.0..=1.0).contains(&occ) && occ.is_finite(), "occ = {occ}");
        }
    }

    #[test]
    fn capsule_ao_in_unit_range() {
        let capsule = Capsule::new(
            Vec3::new(-2.0, 2.0, 0.0),
            Vec3::new(2.0, 3.0, 0.0),
            0.9,
        );
        let ao = capsule_ambient_occlusion(Vec3::ZERO, Vec3::Y, capsule);
        assert!((0.0..=1.0).contains(&ao) && ao.is_finite(), "ao = {ao}");
    }

    #[test]
    fn zero_radius_capsule_casts_nothing() {
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 10.0), 0.5);
        let capsule = Capsule::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 1.0, 2.0), 0.0);
        assert_eq!(capsule_soft_shadow(Vec3::ZERO, light, capsule), 0.0);
        assert_eq!(capsule_ambient_occlusion(Vec3::ZERO, Vec3::Y, capsule), 0.0);
    }
}
