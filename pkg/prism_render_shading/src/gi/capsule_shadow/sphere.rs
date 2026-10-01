//! Analytic sphere-proxy soft shadows and closed-form sphere ambient occlusion.
//!
//! A *sphere proxy* is the simplest analytic occluder: a centre and a radius.
//! This module turns such a proxy into two deterministic, GPU-free golden
//! references the WESL/Metal twin must reproduce bit-for-shape:
//!
//! * [`sphere_soft_shadow`] computes the fraction of a disk area light that a
//!   sphere hides from a receiver point.  The light and the occluder are each
//!   projected to an angular disk on the unit sphere around the receiver, and
//!   the shadow fraction is the *circle–circle overlap* of those two angular
//!   disks normalised by the light disk.  This is the classic "soft shadow as
//!   angular coverage" model used by Unreal-style capsule shadows: a point
//!   light gives a hard shadow, and a wider light smoothly feathers the
//!   penumbra.
//! * [`sphere_ambient_occlusion`] evaluates the closed-form sphere AO of
//!   Íñigo Quílez, `AO = (1 - sqrt(1 - (r/d)^2)) * cos(theta)`, where `r` is the
//!   sphere radius, `d` the receiver-to-centre distance, and `theta` the angle
//!   between the surface normal and the direction to the sphere centre.  The
//!   first factor is the fraction of a hemisphere the sphere's solid-angle cap
//!   subtends; the cosine is the Lambertian foreshortening toward the centre.
//!
//! # Conventions
//! * All positions are right-handed `Vec3` in a shared world space; distances
//!   are world units.
//! * A soft-shadow result is an *occlusion* fraction in `[0, 1]`: `0` is fully
//!   lit, `1` is fully shadowed.  An AO result is an *occlusion* fraction in
//!   `[0, 1]`: `0` is fully open, `1` is fully occluded.
//! * The occluder only shadows when it lies between the receiver and the light
//!   (toward the light and nearer than the far edge of the light); otherwise
//!   the shadow fraction is `0`.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no allocation, no global state.  Degenerate inputs (zero light distance,
//!   receiver inside the sphere, a zero normal, a non-positive radius) are
//!   clamped to defined, finite results and never produce `NaN` or infinities.

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Distance below which a length is treated as zero (world units).
const EPS: f32 = 1.0e-6;

/// A circular area light approximated as a disk facing the receiver.
///
/// Only the centre [`position`](Self::position) and the disk
/// [`radius`](Self::radius) matter for the angular-coverage shadow model: the
/// disk is assumed to face the receiver, which is the standard simplification
/// for soft contact shadows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskLight {
    /// World-space centre of the light disk.
    pub position: Vec3,
    /// World-space radius of the light disk.  `0` yields a hard (point) shadow.
    pub radius: f32,
}

impl DiskLight {
    /// Builds a disk light, clamping the radius to be non-negative.
    #[inline]
    pub fn new(position: Vec3, radius: f32) -> Self {
        Self {
            position,
            radius: radius.max(0.0),
        }
    }
}

/// A sphere occluder proxy: a world-space centre and a radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// World-space centre of the sphere.
    pub center: Vec3,
    /// World-space radius.  Non-positive radii occlude nothing.
    pub radius: f32,
}

impl Sphere {
    /// Builds a sphere, clamping the radius to be non-negative.
    #[inline]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
        }
    }
}

/// Fraction of disk `A` (radius `r_a`) covered by disk `B` (radius `r_b`) whose
/// centres are separated by `sep`, all measured in the same angular units.
///
/// This is the analytic area of intersection of two circles divided by the
/// area of circle `A`, so the result is the fraction of the *light* disk that
/// the occluder disk hides.  Returns a value in `[0, 1]`.
///
/// Degenerate handling:
/// * `r_a <= EPS` (a point light): hard shadow — `1` if the occluder disk
///   contains the point (`sep <= r_b`), else `0`.
/// * `r_b <= EPS` (an occluder subtending nothing): `0`.
/// * Disjoint disks (`sep >= r_a + r_b`): `0`.
/// * One disk fully inside the other (`sep <= |r_a - r_b|`): the smaller disk
///   area over the light disk area.
#[inline]
pub fn disk_overlap_fraction(r_a: f32, r_b: f32, sep: f32) -> f32 {
    if r_a <= EPS {
        // Point light: binary coverage.
        return if sep <= r_b { 1.0 } else { 0.0 };
    }
    if r_b <= EPS {
        return 0.0;
    }
    let d = sep.max(0.0);
    if d >= r_a + r_b {
        return 0.0;
    }
    let light_area = PI * r_a * r_a;
    if d <= (r_a - r_b).abs() {
        // One disk lies entirely within the other.
        let min_r = r_a.min(r_b);
        return (PI * min_r * min_r / light_area).clamp(0.0, 1.0);
    }
    // Partial lens-shaped overlap of two circles.
    let d2 = d * d;
    let r1 = r_a;
    let r2 = r_b;
    let cos1 = ((d2 + r1 * r1 - r2 * r2) / (2.0 * d * r1)).clamp(-1.0, 1.0);
    let cos2 = ((d2 + r2 * r2 - r1 * r1) / (2.0 * d * r2)).clamp(-1.0, 1.0);
    let a1 = ops::acos(cos1);
    let a2 = ops::acos(cos2);
    // Heron-like term for the shared chord; clamped to avoid negative round-off.
    let tri_arg = (-d + r2 + r1) * (d + r2 - r1) * (d - r2 + r1) * (d + r2 + r1);
    let tri = 0.5 * tri_arg.max(0.0).sqrt();
    let overlap = r1 * r1 * a1 + r2 * r2 * a2 - tri;
    (overlap / light_area).clamp(0.0, 1.0)
}

/// Soft-shadow occlusion fraction cast by a sphere onto `receiver` from a disk
/// light.
///
/// The light and the sphere are each projected to an angular disk seen from
/// the receiver: the light to a cap of half-angle `asin(light_r / dist_light)`
/// and the sphere to a cap of half-angle `asin(sphere_r / dist_sphere)`, with
/// angular separation `acos(dir_light · dir_sphere)`.  The returned occlusion
/// is the circle–circle overlap of those caps normalised by the light cap, so
/// `0` is fully lit and `1` is fully shadowed.
///
/// The sphere only shadows when it is toward the light and in front of the far
/// edge of the light; otherwise the result is `0`.  A receiver inside the
/// sphere is fully shadowed (`1`).
pub fn sphere_soft_shadow(receiver: Vec3, light: DiskLight, sphere: Sphere) -> f32 {
    let to_light = light.position - receiver;
    let dist_l = to_light.length();
    if dist_l <= EPS {
        // Receiver sits on the light: nothing can occlude it.
        return 0.0;
    }
    if sphere.radius <= 0.0 {
        return 0.0;
    }
    let to_occ = sphere.center - receiver;
    let dist_o = to_occ.length();
    if dist_o <= EPS {
        // Receiver at the sphere centre: fully enclosed.
        return 1.0;
    }
    let dir_l = to_light / dist_l;
    let dir_o = to_occ / dist_o;
    let cos_sep = dir_l.dot(dir_o).clamp(-1.0, 1.0);
    if cos_sep <= 0.0 {
        // Occluder points away from the light direction: no shadow.
        return 0.0;
    }
    if dist_o - sphere.radius >= dist_l {
        // The whole sphere lies at or beyond the light: it cannot occlude it.
        return 0.0;
    }
    if sphere.radius >= dist_o {
        // Receiver is inside the sphere.
        return 1.0;
    }
    let light_ang = ops::asin((light.radius / dist_l).clamp(0.0, 1.0));
    let occ_ang = ops::asin((sphere.radius / dist_o).clamp(0.0, 1.0));
    let sep = ops::acos(cos_sep);
    disk_overlap_fraction(light_ang, occ_ang, sep)
}

/// Closed-form sphere ambient occlusion after Íñigo Quílez.
///
/// Returns `AO = (1 - sqrt(1 - (r/d)^2)) * max(cos(theta), 0)` in `[0, 1]`,
/// where `r` is the sphere radius, `d` the receiver-to-centre distance, and
/// `theta` the angle between `normal` and the direction to the sphere centre.
/// The first factor is the hemispherical solid-angle fraction of the sphere's
/// cap; the cosine foreshortens it toward the surface normal.
///
/// Degenerate handling: a zero normal, a non-positive radius, or a sphere on
/// the far side of the normal returns `0`; a receiver at or inside the sphere
/// returns the cosine-weighted full occlusion (`<= 1`).
pub fn sphere_ambient_occlusion(receiver: Vec3, normal: Vec3, sphere: Sphere) -> f32 {
    if sphere.radius <= 0.0 {
        return 0.0;
    }
    let n = normal.normalize_or_zero();
    if n == Vec3::ZERO {
        return 0.0;
    }
    let to_c = sphere.center - receiver;
    let d = to_c.length();
    if d <= EPS {
        // At the centre: the sphere fills the hemisphere toward +n.
        return 1.0;
    }
    let dir = to_c / d;
    let cos_t = dir.dot(n).max(0.0);
    if cos_t <= 0.0 {
        return 0.0;
    }
    if sphere.radius >= d {
        // Receiver inside the sphere: fully occluded, foreshortened.
        return cos_t.clamp(0.0, 1.0);
    }
    let ratio = (sphere.radius / d).clamp(0.0, 1.0);
    let solid = 1.0 - (1.0 - ratio * ratio).max(0.0).sqrt();
    (solid * cos_t).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    #[test]
    fn overlap_is_full_when_occluder_covers_point_light() {
        // Point light (r_a = 0) fully covered when inside occluder disk.
        assert_eq!(disk_overlap_fraction(0.0, 0.3, 0.1), 1.0);
        assert_eq!(disk_overlap_fraction(0.0, 0.3, 0.4), 0.0);
    }

    #[test]
    fn overlap_is_zero_when_disjoint() {
        assert_eq!(disk_overlap_fraction(0.2, 0.2, 1.0), 0.0);
    }

    #[test]
    fn overlap_is_full_when_occluder_contains_light() {
        // Large occluder, concentric: covers the entire light disk.
        let f = disk_overlap_fraction(0.1, 0.5, 0.0);
        assert!((f - 1.0).abs() < TOL, "f = {f}");
    }

    #[test]
    fn overlap_partial_is_between_zero_and_one() {
        let f = disk_overlap_fraction(0.3, 0.3, 0.3);
        assert!(f > 0.0 && f < 1.0, "f = {f}");
    }

    #[test]
    fn overlap_symmetric_identical_disks_half_separation() {
        // Two identical disks separated by exactly one radius overlap by the
        // well-known lens fraction ~0.391 of a single disk.
        let f = disk_overlap_fraction(1.0, 1.0, 1.0);
        assert!((f - 0.391).abs() < 0.01, "f = {f}");
    }

    #[test]
    fn shadow_full_when_sphere_aligned_and_large() {
        // Sphere centred on the light direction, angularly larger than the
        // light: full shadow.
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 10.0), 0.5);
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, 2.0), 1.0);
        let occ = sphere_soft_shadow(Vec3::ZERO, light, sphere);
        assert!((occ - 1.0).abs() < TOL, "occ = {occ}");
    }

    #[test]
    fn shadow_weakens_with_distance() {
        // Fixed large light, small sphere receding along the same direction:
        // the occluder subtends less and less of the light, so the on-axis
        // coverage fraction decreases monotonically with distance.
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 30.0), 2.0);
        let near = sphere_soft_shadow(
            Vec3::ZERO,
            light,
            Sphere::new(Vec3::new(0.0, 0.0, 5.0), 0.3),
        );
        let mid = sphere_soft_shadow(
            Vec3::ZERO,
            light,
            Sphere::new(Vec3::new(0.0, 0.0, 10.0), 0.3),
        );
        let far = sphere_soft_shadow(
            Vec3::ZERO,
            light,
            Sphere::new(Vec3::new(0.0, 0.0, 15.0), 0.3),
        );
        assert!(near >= mid && mid >= far, "{near} {mid} {far}");
        assert!(near > far, "{near} {far}");
    }

    #[test]
    fn shadow_zero_when_occluder_behind_receiver() {
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 10.0), 0.5);
        // Sphere on the opposite side of the light direction.
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, -3.0), 1.0);
        assert_eq!(sphere_soft_shadow(Vec3::ZERO, light, sphere), 0.0);
    }

    #[test]
    fn shadow_zero_when_occluder_beyond_light() {
        let light = DiskLight::new(Vec3::new(0.0, 0.0, 5.0), 0.5);
        let sphere = Sphere::new(Vec3::new(0.0, 0.0, 9.0), 0.5);
        assert_eq!(sphere_soft_shadow(Vec3::ZERO, light, sphere), 0.0);
    }

    #[test]
    fn shadow_always_in_unit_range() {
        let light = DiskLight::new(Vec3::new(1.0, 2.0, 8.0), 0.7);
        for z in [1.0f32, 2.0, 3.0, 4.0, 6.0] {
            for x in [-2.0f32, 0.0, 1.5] {
                let s =
                    sphere_soft_shadow(Vec3::ZERO, light, Sphere::new(Vec3::new(x, 0.0, z), 0.6));
                assert!((0.0..=1.0).contains(&s) && s.is_finite(), "s = {s}");
            }
        }
    }

    #[test]
    fn ao_is_non_negative_and_bounded() {
        let n = Vec3::Y;
        for d in [1.0f32, 2.0, 4.0, 8.0] {
            let ao = sphere_ambient_occlusion(
                Vec3::ZERO,
                n,
                Sphere::new(Vec3::new(0.0, d, 0.0), 0.5),
            );
            assert!((0.0..=1.0).contains(&ao) && ao.is_finite(), "ao = {ao}");
        }
    }

    #[test]
    fn ao_decreases_with_distance() {
        let n = Vec3::Y;
        let near =
            sphere_ambient_occlusion(Vec3::ZERO, n, Sphere::new(Vec3::new(0.0, 1.5, 0.0), 1.0));
        let far =
            sphere_ambient_occlusion(Vec3::ZERO, n, Sphere::new(Vec3::new(0.0, 6.0, 0.0), 1.0));
        assert!(near > far, "near {near} far {far}");
    }

    #[test]
    fn ao_zero_when_sphere_behind_normal() {
        // Sphere below a +Y facing surface contributes no occlusion.
        let ao = sphere_ambient_occlusion(
            Vec3::ZERO,
            Vec3::Y,
            Sphere::new(Vec3::new(0.0, -3.0, 0.0), 1.0),
        );
        assert_eq!(ao, 0.0);
    }

    #[test]
    fn ao_matches_quilez_closed_form() {
        // Sphere straight along the normal: cos(theta) = 1, so AO equals the
        // solid-angle fraction 1 - sqrt(1 - (r/d)^2).
        let r = 1.0f32;
        let d = 3.0f32;
        let expected = 1.0 - (1.0 - (r / d) * (r / d)).sqrt();
        let ao = sphere_ambient_occlusion(
            Vec3::ZERO,
            Vec3::Y,
            Sphere::new(Vec3::new(0.0, d, 0.0), r),
        );
        assert!((ao - expected).abs() < TOL, "ao {ao} expected {expected}");
    }

    #[test]
    fn ao_degenerates_safely() {
        // Zero normal, zero radius, and receiver-at-centre never NaN.
        assert_eq!(
            sphere_ambient_occlusion(Vec3::ZERO, Vec3::ZERO, Sphere::new(Vec3::Y, 1.0)),
            0.0
        );
        assert_eq!(
            sphere_ambient_occlusion(Vec3::ZERO, Vec3::Y, Sphere::new(Vec3::Y, 0.0)),
            0.0
        );
        let at_center =
            sphere_ambient_occlusion(Vec3::ZERO, Vec3::Y, Sphere::new(Vec3::ZERO, 1.0));
        assert!(at_center.is_finite() && (0.0..=1.0).contains(&at_center));
    }
}
