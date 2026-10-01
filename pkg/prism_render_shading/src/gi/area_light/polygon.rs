//! Polygonal area-light integration via Linearly Transformed Cosines — CPU
//! golden reference.
//!
//! Implements the Heitz et al. 2016 polygon pipeline: horizon clipping of a
//! light polygon to the upper hemisphere, the closed-form edge integral of a
//! clamped-cosine distribution, and the full LTC evaluation that transforms the
//! polygon by `M⁻¹` before integrating.  The same machinery, with the identity
//! transform, yields the diffuse (clamped-cosine) irradiance of an area light.
//!
//! # Conventions
//! * All vertices are **positions relative to the shaded point** expressed in a
//!   local shading frame whose surface normal is `+Z`.  The integration treats
//!   each vertex as a direction (its normalisation onto the unit sphere); the
//!   polygon is assumed planar and consistently wound.
//! * [`polygon_form_factor`] returns the *normalised* clamped-cosine integral
//!   `F ∈ [0, 1]` (`1` for a polygon covering the whole upper hemisphere).  The
//!   diffuse irradiance of a Lambertian emitter of radiance `L` is then
//!   `E = π · L · F`; [`diffuse_polygon_irradiance`] returns `E / L = π · F`.
//! * The LTC specular response is `amplitude · F(M⁻¹ · polygon)`; see
//!   [`ltc_evaluate`].
//! * Transcendental math goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent `f32::sqrt`.  Every routine is a deterministic pure function and
//!   clamps against degeneracy so no result is ever `NaN`.
//!
//! # References
//! * Heitz, Dupuy, Hill, Neubelt 2016, *Real-Time Polygonal-Light Shading with
//!   Linearly Transformed Cosines* (SIGGRAPH).
//! * Baum, Rushmeier, Winget 1989, *Improving Radiosity Solutions through the
//!   Use of Analytically Determined Form-Factors* — the edge form factor.

use alloc::vec::Vec;
use bevy_math::{Vec3, ops};
use core::f32::consts::PI;

use super::ltc_lut::LtcCoeffs;

/// Smallest `sin` of the angle between two edge directions treated as
/// non-parallel.  Parallel directions span zero solid angle and contribute `0`.
pub const MIN_SIN: f32 = 1.0e-7;

/// Vertices with squared length below this are treated as coincident with the
/// shaded point and dropped before integration.
pub const MIN_LEN_SQ: f32 = 1.0e-18;

/// Vector edge integral of a clamped-cosine distribution for one polygon edge.
///
/// Returns `(acos(v1·v2) / |v1×v2|) · (v1×v2)`, the contribution of the edge
/// from unit direction `v1` to unit direction `v2` to the vector form factor.
/// Inputs are assumed unit length; near-parallel directions return the zero
/// vector.
#[inline]
pub fn integrate_edge_vec(v1: Vec3, v2: Vec3) -> Vec3 {
    let cos_theta = v1.dot(v2).clamp(-1.0, 1.0);
    let theta = ops::acos(cos_theta);
    let cross = v1.cross(v2);
    let sin_theta = cross.length();
    if sin_theta < MIN_SIN {
        return Vec3::ZERO;
    }
    let scale = theta / sin_theta;
    let result = cross * scale;
    if result.is_finite() { result } else { Vec3::ZERO }
}

/// Scalar `z`-component of [`integrate_edge_vec`], i.e. the edge's contribution
/// to the form factor of a surface whose normal is `+Z`.
#[inline]
pub fn integrate_edge(v1: Vec3, v2: Vec3) -> f32 {
    integrate_edge_vec(v1, v2).z
}

/// Clips a polygon (list of relative positions) to the upper hemisphere
/// `z ≥ 0` using Sutherland–Hodgman clipping against the `z = 0` plane.
///
/// Edges crossing the horizon are split at their `z = 0` intersection so the
/// returned ring stays on or above the horizon.  Degenerate inputs (fewer than
/// three vertices, or an entirely below-horizon polygon) return an empty list.
pub fn clip_horizon(points: &[Vec3]) -> Vec<Vec3> {
    let n = points.len();
    let mut out: Vec<Vec3> = Vec::new();
    if n < 3 {
        return out;
    }
    for i in 0..n {
        let cur = points[i];
        let prev = points[(i + n - 1) % n];
        let cur_in = cur.z >= 0.0;
        let prev_in = prev.z >= 0.0;
        if cur_in {
            if !prev_in {
                // Entering: add the intersection with `z = 0` first.
                if let Some(p) = intersect_horizon(prev, cur) {
                    out.push(p);
                }
            }
            out.push(cur);
        } else if prev_in {
            // Leaving: add only the intersection with `z = 0`.
            if let Some(p) = intersect_horizon(prev, cur) {
                out.push(p);
            }
        }
    }
    out
}

/// Intersection of the segment `a → b` with the `z = 0` plane, if the segment
/// straddles it with a finite parameter.
#[inline]
fn intersect_horizon(a: Vec3, b: Vec3) -> Option<Vec3> {
    let dz = a.z - b.z;
    if dz.abs() < 1.0e-12 {
        return None;
    }
    let t = a.z / dz;
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    let p = a + (b - a) * t;
    if p.is_finite() { Some(p) } else { None }
}

/// Normalised clamped-cosine form factor `F ∈ [0, 1]` of a polygon.
///
/// The polygon is horizon-clipped, each surviving vertex is projected onto the
/// unit sphere, and the edge integrals are summed and divided by `2π`.  The
/// result is clamped to `[0, 1]`; a degenerate polygon yields `0`.
pub fn polygon_form_factor(points: &[Vec3]) -> f32 {
    let clipped = clip_horizon(points);
    let m = clipped.len();
    if m < 3 {
        return 0.0;
    }
    // Project surviving vertices onto the unit sphere, dropping vertices that
    // coincide with the shaded point.
    let mut dirs: Vec<Vec3> = Vec::with_capacity(m);
    for &p in &clipped {
        if p.length_squared() <= MIN_LEN_SQ {
            continue;
        }
        let d = p.normalize_or_zero();
        if d.length_squared() > 0.5 {
            dirs.push(d);
        }
    }
    let k = dirs.len();
    if k < 3 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for i in 0..k {
        let v1 = dirs[i];
        let v2 = dirs[(i + 1) % k];
        sum += integrate_edge(v1, v2);
    }
    let f = sum / (2.0 * PI);
    if f.is_finite() { f.clamp(0.0, 1.0) } else { 0.0 }
}

/// Diffuse (Lambertian) irradiance per unit radiance `E / L = π · F` of an
/// area-light polygon.
///
/// For a polygon covering the full upper hemisphere this approaches `π`, the
/// classical result for a fully visible overhead emitter.
#[inline]
pub fn diffuse_polygon_irradiance(points: &[Vec3]) -> f32 {
    PI * polygon_form_factor(points)
}

/// Evaluates the LTC specular response of a polygon under the inverse transform
/// `coeffs`.
///
/// Each vertex is transformed by `M⁻¹` (via [`LtcCoeffs::apply`]), after which
/// the clamped-cosine [`polygon_form_factor`] is evaluated and scaled by the
/// lobe amplitude.  The returned value is the specular contribution per unit
/// radiance and is non-negative and finite.
pub fn ltc_evaluate(points: &[Vec3], coeffs: &LtcCoeffs) -> f32 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut transformed: Vec<Vec3> = Vec::with_capacity(n);
    for &p in points {
        let t = coeffs.apply(p);
        transformed.push(if t.is_finite() { t } else { Vec3::ZERO });
    }
    let f = polygon_form_factor(&transformed);
    let r = coeffs.amplitude.clamp(0.0, 1.0) * f;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

/// Convenience diffuse form factor for a quad given its four corner positions
/// in winding order.
#[inline]
pub fn quad_form_factor(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> f32 {
    polygon_form_factor(&[p0, p1, p2, p3])
}

/// Convenience LTC specular evaluation for a quad given its four corner
/// positions in winding order.
#[inline]
pub fn quad_ltc_evaluate(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, coeffs: &LtcCoeffs) -> f32 {
    ltc_evaluate(&[p0, p1, p2, p3], coeffs)
}

/// Builds the four corner positions of a rectangle area light.
///
/// The rectangle is centred at `center` with half-extents `half_u`, `half_v`
/// along the orthogonal in-plane axes `axis_u`, `axis_v`.  Corners are returned
/// in counter-clockwise winding as seen from the front face.
#[inline]
pub fn rectangle_points(
    center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    half_u: f32,
    half_v: f32,
) -> [Vec3; 4] {
    let u = axis_u * half_u;
    let v = axis_v * half_v;
    [
        center - u - v,
        center + u - v,
        center + u + v,
        center - u + v,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::area_light::ltc_lut::fit_ltc_default;

    /// A large rectangle directly overhead that nearly fills the upper
    /// hemisphere.
    fn big_overhead() -> [Vec3; 4] {
        rectangle_points(
            Vec3::new(0.0, 0.0, 0.02),
            Vec3::X,
            Vec3::Y,
            2000.0,
            2000.0,
        )
    }

    #[test]
    fn edge_integral_is_antisymmetric() {
        let a = Vec3::new(0.3, 0.1, 0.95).normalize();
        let b = Vec3::new(-0.2, 0.4, 0.89).normalize();
        let f = integrate_edge(a, b);
        let r = integrate_edge(b, a);
        assert!((f + r).abs() < 1e-6, "f={f} r={r}");
    }

    #[test]
    fn parallel_edge_contributes_zero() {
        let a = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(integrate_edge_vec(a, a), Vec3::ZERO);
    }

    #[test]
    fn full_hemisphere_irradiance_approaches_pi() {
        // A large facing rectangle recovers the overhead-emitter result E ≈ πL.
        let e = diffuse_polygon_irradiance(&big_overhead());
        assert!((e - PI).abs() < 1e-2, "irradiance={e}, expected ~{PI}");
    }

    #[test]
    fn form_factor_is_bounded() {
        let f = polygon_form_factor(&big_overhead());
        assert!((0.0..=1.0).contains(&f), "f={f}");
        assert!(f > 0.98, "nearly-full hemisphere f={f}");
    }

    #[test]
    fn clipping_preserves_upper_hemisphere() {
        // A quad straddling the horizon: two vertices below, two above.
        let pts = [
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
        ];
        let clipped = clip_horizon(&pts);
        assert!(clipped.len() >= 3, "clip produced {} verts", clipped.len());
        for v in &clipped {
            assert!(v.z >= -1e-6, "vertex below horizon: {v:?}");
        }
    }

    #[test]
    fn fully_below_horizon_is_zero() {
        let pts = [
            Vec3::new(-1.0, -1.0, -0.5),
            Vec3::new(1.0, -1.0, -0.5),
            Vec3::new(0.0, 1.0, -0.5),
        ];
        assert!(clip_horizon(&pts).is_empty());
        assert_eq!(polygon_form_factor(&pts), 0.0);
        assert_eq!(diffuse_polygon_irradiance(&pts), 0.0);
    }

    #[test]
    fn degenerate_polygon_is_zero() {
        assert_eq!(polygon_form_factor(&[Vec3::Z, Vec3::Z]), 0.0);
        assert_eq!(polygon_form_factor(&[]), 0.0);
        assert_eq!(ltc_evaluate(&[Vec3::Z, Vec3::X], &LtcCoeffs::IDENTITY), 0.0);
    }

    #[test]
    fn ltc_identity_matches_diffuse_form_factor() {
        // With the identity transform the LTC evaluation is the clamped-cosine
        // form factor (amplitude 1).
        let quad = rectangle_points(
            Vec3::new(0.4, 0.0, 1.0),
            Vec3::X,
            Vec3::Y,
            0.5,
            0.5,
        );
        let ltc = ltc_evaluate(&quad, &LtcCoeffs::IDENTITY);
        let ff = polygon_form_factor(&quad);
        assert!((ltc - ff).abs() < 1e-6, "ltc={ltc} ff={ff}");
    }

    #[test]
    fn ltc_evaluate_is_finite_nonnegative() {
        let coeffs = fit_ltc_default(0.5, 0.4);
        let quad = rectangle_points(
            Vec3::new(0.3, 0.1, 0.8),
            Vec3::X,
            Vec3::Y,
            0.4,
            0.6,
        );
        let r = ltc_evaluate(&quad, &coeffs);
        assert!(r.is_finite() && r >= 0.0, "r={r}");
    }

    #[test]
    fn smaller_light_has_smaller_form_factor() {
        let big = rectangle_points(Vec3::new(0.0, 0.0, 1.0), Vec3::X, Vec3::Y, 1.0, 1.0);
        let small = rectangle_points(Vec3::new(0.0, 0.0, 1.0), Vec3::X, Vec3::Y, 0.2, 0.2);
        let fb = polygon_form_factor(&big);
        let fs = polygon_form_factor(&small);
        assert!(fb > fs, "big={fb} small={fs}");
    }

    #[test]
    fn winding_sign_is_handled() {
        // Reversed winding flips the raw sum sign; the form factor clamps to 0
        // for a back-facing polygon rather than returning negative energy.
        let quad = rectangle_points(Vec3::new(0.0, 0.0, 1.0), Vec3::X, Vec3::Y, 0.5, 0.5);
        let rev = [quad[3], quad[2], quad[1], quad[0]];
        let f = polygon_form_factor(&rev);
        assert!(f >= 0.0 && f.is_finite(), "f={f}");
    }
}
