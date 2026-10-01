//! Non-polygonal area lights: disk, sphere, and line / tube — CPU golden.
//!
//! Complements [`super::polygon`] with the shapes that are awkward to express
//! as explicit polygons.  Disks are integrated with the LTC machinery through a
//! regular polygonal approximation (plus a closed-form coaxial result for
//! validation); spheres and tubes use Karis' 2013 representative-point /
//! representative-area approximations so a single highlight stays plausible
//! across roughness.
//!
//! # Conventions
//! * All inputs are **positions relative to the shaded point** in a local
//!   shading frame whose surface normal is `+Z`.  Returned radiances are
//!   per-unit-radiance contributions: diffuse is `E / L` (an irradiance, so a
//!   fully visible overhead emitter approaches `π`), specular is the LTC form
//!   factor scaled by the lobe amplitude.
//! * Every result is bundled in a [`ShapeContribution`]; both channels are
//!   non-negative and finite by construction.
//! * Transcendental math goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent `f32::sqrt`.  All routines are deterministic pure functions and
//!   clamp against degeneracy so no result is ever `NaN`.
//!
//! # References
//! * Heitz, Dupuy, Hill, Neubelt 2016, *Real-Time Polygonal-Light Shading with
//!   Linearly Transformed Cosines* — disk / polygon LTC integration.
//! * Karis 2013, *Real Shading in Unreal Engine 4* (SIGGRAPH course) — sphere
//!   and tube light representative-point approximations.

use alloc::vec::Vec;
use bevy_math::{Vec3, ops};
use core::f32::consts::PI;

use super::ltc_lut::LtcCoeffs;
use super::polygon::{diffuse_polygon_irradiance, ltc_evaluate};

/// Minimum segment / vector length treated as non-degenerate.
pub const MIN_LEN: f32 = 1.0e-6;

/// Default segment count used when tessellating a disk into a polygon.
pub const DEFAULT_DISK_SEGMENTS: u32 = 32;

/// Diffuse and specular per-unit-radiance contributions of an area light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeContribution {
    /// Diffuse irradiance per unit radiance `E / L` (non-negative, finite).
    pub diffuse: f32,
    /// Specular LTC response per unit radiance (non-negative, finite).
    pub specular: f32,
}

impl ShapeContribution {
    /// A fully dark contribution.
    pub const ZERO: Self = Self {
        diffuse: 0.0,
        specular: 0.0,
    };

    /// Returns the contribution with both channels clamped non-negative and
    /// finite, falling back to zero for non-finite inputs.
    #[inline]
    pub fn sanitized(self) -> Self {
        let d = if self.diffuse.is_finite() { self.diffuse.max(0.0) } else { 0.0 };
        let s = if self.specular.is_finite() { self.specular.max(0.0) } else { 0.0 };
        Self { diffuse: d, specular: s }
    }
}

/// Tessellates a disk area light into a polygon of relative positions.
///
/// The disk is centred at `center` with radius `radius` and lies in the plane
/// spanned by the orthonormal axes `axis_u`, `axis_v`.  `segments` is floored at
/// `3`.  Vertices are returned in counter-clockwise order.
pub fn disk_polygon(
    center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    radius: f32,
    segments: u32,
) -> Vec<Vec3> {
    let segments = segments.max(3);
    let r = radius.max(0.0);
    let u = axis_u.normalize_or_zero();
    let v = axis_v.normalize_or_zero();
    let mut out = Vec::with_capacity(segments as usize);
    let inv = 1.0 / segments as f32;
    for i in 0..segments {
        let phi = 2.0 * PI * (i as f32) * inv;
        let (sp, cp) = ops::sin_cos(phi);
        out.push(center + (u * cp + v * sp) * r);
    }
    out
}

/// Evaluates a disk area light's diffuse and specular contributions.
///
/// The disk is tessellated via [`disk_polygon`] and integrated with the shared
/// polygon routines: diffuse through the clamped-cosine form factor, specular
/// through [`ltc_evaluate`] with the supplied inverse transform.
pub fn disk_contribution(
    center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    radius: f32,
    segments: u32,
    coeffs: &LtcCoeffs,
) -> ShapeContribution {
    let poly = disk_polygon(center, axis_u, axis_v, radius, segments);
    let diffuse = diffuse_polygon_irradiance(&poly);
    let specular = ltc_evaluate(&poly, coeffs);
    ShapeContribution { diffuse, specular }.sanitized()
}

/// Closed-form diffuse irradiance `E / L` from a differential surface element to
/// a coaxial, parallel, front-facing disk.
///
/// For a disk of radius `R` whose axis passes through the shaded point at
/// distance `d`, the configuration factor is `R² / (R² + d²)`, so
/// `E / L = π · R² / (R² + d²)`.  Used as an analytic reference for the
/// tessellated path; `radius` and `distance` are clamped non-negative.
#[inline]
pub fn disk_diffuse_axial(radius: f32, distance: f32) -> f32 {
    let r = radius.max(0.0);
    let d = distance.max(0.0);
    let r2 = r * r;
    let denom = (r2 + d * d).max(1.0e-20);
    let f = r2 / denom;
    PI * f.clamp(0.0, 1.0)
}

/// Solid angle subtended by a sphere of radius `radius` at distance `dist`.
///
/// Returns `2π(1 − cosα)` with `sinα = radius / dist`; the shaded point being
/// inside the sphere (`radius ≥ dist`) saturates at the full hemisphere `2π`.
#[inline]
pub fn sphere_solid_angle(dist: f32, radius: f32) -> f32 {
    let d = dist.max(MIN_LEN);
    let r = radius.max(0.0);
    if r >= d {
        return 2.0 * PI;
    }
    let sin2 = (r * r) / (d * d);
    let cos_alpha = (1.0 - sin2).max(0.0).sqrt();
    (2.0 * PI * (1.0 - cos_alpha)).max(0.0)
}

/// Diffuse irradiance `E / L` of a uniform-radiance sphere light.
///
/// Uses the above-horizon approximation `E / L = π · sin²α · max(cosθ, 0)` with
/// `sinα = R / d` and `cosθ` the cosine between the surface normal `+Z` and the
/// direction to the sphere centre.  A shaded point inside the sphere clamps to
/// the overhead-emitter limit `π`.
pub fn sphere_diffuse_irradiance(center: Vec3, radius: f32) -> f32 {
    let d = center.length();
    if d < MIN_LEN {
        return PI;
    }
    let r = radius.max(0.0);
    if r >= d {
        return PI;
    }
    let sin2 = (r * r) / (d * d);
    let cos_theta = (center.z / d).max(0.0);
    let e = PI * sin2 * cos_theta;
    if e.is_finite() { e.clamp(0.0, PI) } else { 0.0 }
}

/// Karis representative point on a sphere for the reflection ray.
///
/// Finds the point on the sphere surface closest to the reflection ray through
/// the origin along unit direction `reflect_dir`, which best represents the
/// specular highlight.  `reflect_dir` is normalised internally; a degenerate
/// direction falls back to the sphere centre.
pub fn sphere_representative_point(center: Vec3, radius: f32, reflect_dir: Vec3) -> Vec3 {
    let r = reflect_dir.normalize_or_zero();
    if r.length_squared() < 0.5 {
        return center;
    }
    let radius = radius.max(0.0);
    let center_to_ray = r * center.dot(r) - center;
    let len = center_to_ray.length();
    if len < MIN_LEN {
        return center;
    }
    let t = (radius / len).clamp(0.0, 1.0);
    center + center_to_ray * t
}

/// Evaluates a sphere light's diffuse and specular contributions.
///
/// Diffuse uses [`sphere_diffuse_irradiance`]; specular approximates the sphere
/// as a view-facing disk (billboard) of the same radius at its centre and runs
/// it through the LTC disk integration.
pub fn sphere_contribution(
    center: Vec3,
    radius: f32,
    segments: u32,
    coeffs: &LtcCoeffs,
) -> ShapeContribution {
    let diffuse = sphere_diffuse_irradiance(center, radius);
    let d = center.length();
    if d < MIN_LEN {
        return ShapeContribution { diffuse, specular: 0.0 }.sanitized();
    }
    // Billboard the sphere: a disk facing the shaded point at the centre.
    let n = center / d;
    let (u, v) = orthonormal_basis(n);
    let specular = disk_contribution(center, u, v, radius, segments, coeffs).specular;
    ShapeContribution { diffuse, specular }.sanitized()
}

/// Point on the segment `p0 → p1` closest to the reflection ray through the
/// origin along unit direction `reflect_dir` (Karis' representative point for
/// line / tube specular).
///
/// `reflect_dir` is normalised internally; the parameter is clamped to the
/// segment so the returned point always lies on `[p0, p1]`.
pub fn closest_point_on_segment_to_ray(p0: Vec3, p1: Vec3, reflect_dir: Vec3) -> Vec3 {
    let r = reflect_dir.normalize_or_zero();
    if r.length_squared() < 0.5 {
        return (p0 + p1) * 0.5;
    }
    let ld = p1 - p0;
    let ld_len2 = ld.length_squared();
    if ld_len2 < MIN_LEN * MIN_LEN {
        return p0;
    }
    let r_dot_ld = r.dot(ld);
    let denom = ld_len2 - r_dot_ld * r_dot_ld;
    if denom.abs() < 1.0e-12 {
        // Segment parallel to the ray: clamp to the nearer endpoint.
        return if p0.length_squared() <= p1.length_squared() { p0 } else { p1 };
    }
    let t = ((r.dot(p0) * r_dot_ld) - p0.dot(ld)) / denom;
    let t = t.clamp(0.0, 1.0);
    p0 + ld * t
}

/// Builds the thin rectangle (capsule hull) of a tube light of the given
/// `radius`, widened perpendicular to the viewing plane.
fn tube_quad(p0: Vec3, p1: Vec3, radius: f32) -> [Vec3; 4] {
    let mid = (p0 + p1) * 0.5;
    let axis = (p1 - p0).normalize_or_zero();
    let axis = if axis.length_squared() < 0.5 { Vec3::X } else { axis };
    // Offset perpendicular to the tube axis and to the view direction (toward
    // the midpoint), so the strip faces the shaded point.
    let mut off = axis.cross(mid).normalize_or_zero();
    if off.length_squared() < 0.5 {
        off = axis.cross(Vec3::Z).normalize_or_zero();
    }
    if off.length_squared() < 0.5 {
        off = Vec3::Y;
    }
    let w = off * radius.max(0.0);
    [p0 - w, p1 - w, p1 + w, p0 + w]
}

/// Diffuse irradiance `E / L` of a line / tube light.
///
/// Evaluated as the clamped-cosine form factor of the tube's thin rectangular
/// hull (the Karis representative-area limit of a capsule), so a long thin
/// emitter yields the correct elongated falloff.  `radius` is clamped
/// non-negative; a zero-radius wire uses a hairline strip to stay finite.
pub fn i_diffuse_line(p0: Vec3, p1: Vec3, radius: f32) -> f32 {
    let r = radius.max(1.0e-4);
    let quad = tube_quad(p0, p1, r);
    diffuse_polygon_irradiance(&quad)
}

/// Specular LTC response of a line / tube light.
///
/// Uses Karis' representative point on the segment for the reflection ray to
/// build a view-facing disk of the tube radius, then integrates it through the
/// LTC disk path.  Returns the amplitude-scaled form factor.
pub fn i_specular_line(
    p0: Vec3,
    p1: Vec3,
    radius: f32,
    reflect_dir: Vec3,
    segments: u32,
    coeffs: &LtcCoeffs,
) -> f32 {
    let rep = closest_point_on_segment_to_ray(p0, p1, reflect_dir);
    let d = rep.length();
    if d < MIN_LEN {
        return 0.0;
    }
    let n = rep / d;
    let (u, v) = orthonormal_basis(n);
    disk_contribution(rep, u, v, radius.max(1.0e-4), segments, coeffs).specular
}

/// Evaluates a line / tube light's diffuse and specular contributions.
pub fn tube_contribution(
    p0: Vec3,
    p1: Vec3,
    radius: f32,
    reflect_dir: Vec3,
    segments: u32,
    coeffs: &LtcCoeffs,
) -> ShapeContribution {
    let diffuse = i_diffuse_line(p0, p1, radius);
    let specular = i_specular_line(p0, p1, radius, reflect_dir, segments, coeffs);
    ShapeContribution { diffuse, specular }.sanitized()
}

/// Builds an orthonormal tangent basis `(u, v)` for a unit vector `n`.
///
/// Uses a branch-light Duff et al. construction; falls back gracefully for a
/// degenerate `n`.
#[inline]
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let n = n.normalize_or_zero();
    if n.length_squared() < 0.5 {
        return (Vec3::X, Vec3::Y);
    }
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let u = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let v = Vec3::new(b, sign + n.y * n.y * a, -n.y);
    (u.normalize_or_zero(), v.normalize_or_zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::area_light::ltc_lut::fit_ltc_default;

    #[test]
    fn disk_axial_approaches_pi_for_huge_disk() {
        // A disk much larger than its distance fills the hemisphere → E ≈ π.
        let e = disk_diffuse_axial(10_000.0, 0.01);
        assert!((e - PI).abs() < 1e-2, "e={e}");
        // A tiny far disk contributes almost nothing.
        let small = disk_diffuse_axial(0.01, 100.0);
        assert!(small >= 0.0 && small < 1e-3, "small={small}");
    }

    #[test]
    fn tessellated_disk_matches_axial_reference() {
        // A moderately large coaxial facing disk: tessellated form factor should
        // track the analytic one.
        let radius = 1.0f32;
        let dist = 1.0f32;
        let center = Vec3::new(0.0, 0.0, dist);
        let poly = disk_polygon(center, Vec3::X, Vec3::Y, radius, 128);
        let numeric = diffuse_polygon_irradiance(&poly);
        let analytic = disk_diffuse_axial(radius, dist);
        assert!((numeric - analytic).abs() < 2e-2, "num={numeric} ana={analytic}");
    }

    #[test]
    fn disk_contribution_is_finite_nonnegative() {
        let coeffs = fit_ltc_default(0.6, 0.5);
        let c = disk_contribution(
            Vec3::new(0.2, 0.0, 1.0),
            Vec3::X,
            Vec3::Y,
            0.5,
            DEFAULT_DISK_SEGMENTS,
            &coeffs,
        );
        assert!(c.diffuse.is_finite() && c.diffuse >= 0.0);
        assert!(c.specular.is_finite() && c.specular >= 0.0);
    }

    #[test]
    fn sphere_solid_angle_bounds() {
        // Far small sphere: tiny solid angle. Enclosing sphere: full hemisphere.
        let far = sphere_solid_angle(100.0, 1.0);
        assert!(far > 0.0 && far < 0.01, "far={far}");
        let inside = sphere_solid_angle(0.5, 1.0);
        assert!((inside - 2.0 * PI).abs() < 1e-6, "inside={inside}");
    }

    #[test]
    fn sphere_diffuse_falls_off_with_distance() {
        let near = sphere_diffuse_irradiance(Vec3::new(0.0, 0.0, 2.0), 1.0);
        let far = sphere_diffuse_irradiance(Vec3::new(0.0, 0.0, 8.0), 1.0);
        assert!(near > far, "near={near} far={far}");
        assert!(far >= 0.0 && near.is_finite());
        // Below the horizon the contribution vanishes.
        let below = sphere_diffuse_irradiance(Vec3::new(0.0, 0.0, -2.0), 0.5);
        assert_eq!(below, 0.0);
    }

    #[test]
    fn sphere_representative_point_on_surface() {
        let center = Vec3::new(0.0, 0.0, 5.0);
        let radius = 1.0;
        let rep = sphere_representative_point(center, radius, Vec3::Z);
        // The representative point lies within one radius of the centre.
        assert!((rep - center).length() <= radius + 1e-5, "rep={rep:?}");
        // Degenerate reflect direction falls back to the centre.
        assert_eq!(sphere_representative_point(center, radius, Vec3::ZERO), center);
    }

    #[test]
    fn sphere_contribution_finite() {
        let coeffs = fit_ltc_default(0.7, 0.4);
        let c = sphere_contribution(Vec3::new(1.0, 0.0, 3.0), 0.8, 24, &coeffs);
        assert!(c.diffuse.is_finite() && c.diffuse >= 0.0);
        assert!(c.specular.is_finite() && c.specular >= 0.0);
    }

    #[test]
    fn closest_point_stays_on_segment() {
        let p0 = Vec3::new(-2.0, 0.5, 3.0);
        let p1 = Vec3::new(2.0, 0.5, 3.0);
        let rep = closest_point_on_segment_to_ray(p0, p1, Vec3::new(0.0, 0.0, 1.0));
        // Parameterise and confirm t ∈ [0, 1].
        let ld = p1 - p0;
        let t = (rep - p0).dot(ld) / ld.length_squared();
        assert!((0.0..=1.0).contains(&t), "t={t}");
    }

    #[test]
    fn line_diffuse_is_finite_nonnegative() {
        let p0 = Vec3::new(-3.0, 0.0, 2.0);
        let p1 = Vec3::new(3.0, 0.0, 2.0);
        let e = i_diffuse_line(p0, p1, 0.1);
        assert!(e.is_finite() && e >= 0.0, "e={e}");
        // Zero-radius wire still returns a finite value.
        let wire = i_diffuse_line(p0, p1, 0.0);
        assert!(wire.is_finite() && wire >= 0.0, "wire={wire}");
    }

    #[test]
    fn longer_tube_gathers_more_than_short() {
        let short = i_diffuse_line(Vec3::new(-0.2, 0.0, 2.0), Vec3::new(0.2, 0.0, 2.0), 0.1);
        let long = i_diffuse_line(Vec3::new(-3.0, 0.0, 2.0), Vec3::new(3.0, 0.0, 2.0), 0.1);
        assert!(long >= short, "long={long} short={short}");
    }

    #[test]
    fn tube_contribution_finite() {
        let coeffs = fit_ltc_default(0.6, 0.5);
        let c = tube_contribution(
            Vec3::new(-2.0, 0.0, 2.0),
            Vec3::new(2.0, 0.0, 2.0),
            0.15,
            Vec3::new(0.0, 0.0, 1.0),
            24,
            &coeffs,
        );
        assert!(c.diffuse.is_finite() && c.diffuse >= 0.0);
        assert!(c.specular.is_finite() && c.specular >= 0.0);
    }

    #[test]
    fn orthonormal_basis_is_orthonormal() {
        for n in [
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(0.3, 0.4, 0.866).normalize(),
            Vec3::new(-0.5, 0.5, 0.707).normalize(),
        ] {
            let (u, v) = orthonormal_basis(n);
            assert!((u.length() - 1.0).abs() < 1e-4, "u not unit: {u:?}");
            assert!((v.length() - 1.0).abs() < 1e-4, "v not unit: {v:?}");
            assert!(u.dot(n).abs() < 1e-4, "u·n={}", u.dot(n));
            assert!(v.dot(n).abs() < 1e-4, "v·n={}", v.dot(n));
            assert!(u.dot(v).abs() < 1e-4, "u·v={}", u.dot(v));
        }
    }

    #[test]
    fn contribution_sanitize_clears_nan() {
        let c = ShapeContribution {
            diffuse: f32::NAN,
            specular: -1.0,
        }
        .sanitized();
        assert_eq!(c, ShapeContribution::ZERO);
    }
}
