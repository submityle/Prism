//! Anisotropic GGX normal-distribution and Smith masking–shadowing — CPU golden.
//!
//! This module is the backend-neutral numerical reference for the *anisotropic*
//! GGX / Trowbridge–Reitz microfacet lobe used for brushed metal, hair-like
//! anisotropic highlights, and directional roughness.  Unlike the isotropic
//! lobe in [`crate::gi::spec_gi::ggx_lobe`], the microfacet width differs along
//! the surface tangent and bitangent, so the whole model is expressed directly
//! in a *world-space* orthonormal tangent frame `(t, b, n)` supplied by the
//! caller (see [`crate::gi::anisotropy::remap`]).
//!
//! It implements:
//!
//! * the anisotropic GGX normal-distribution function
//!   `D(h) = 1 / (π · α_t · α_b · ((h·t/α_t)² + (h·b/α_b)² + (h·n)²)²)`, and
//! * the height-correlated Smith masking–shadowing term `G2`, built from the
//!   anisotropic Smith `Lambda` and the single-direction `G1`.
//!
//! # Conventions
//! * All directions are *unit* world-space vectors.  The tangent frame
//!   `(tangent, bitangent, normal)` is assumed orthonormal and right-handed;
//!   the caller builds it with
//!   [`crate::gi::anisotropy::remap::orthonormal_tangent_frame`].  A direction's
//!   cosine with the normal is `w·n`; only the upper hemisphere (`w·n > 0`)
//!   carries energy and back-facing configurations return `0`.
//! * `α_t` / `α_b` are the GGX widths along the tangent / bitangent.  Both are
//!   clamped to [`MIN_ALPHA`] so a mirror (`roughness → 0`) stays a finite,
//!   extremely sharp lobe instead of a Dirac delta that would divide by zero.
//! * Setting `α_t == α_b` collapses the model to the isotropic GGX lobe, which
//!   is covered by [`ndf_isotropic_matches`](self) in tests.
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals or
//!   `unsafe`) and defends against degeneracy: denominators are floored,
//!   square-root arguments are clamped non-negative, and every result is finite
//!   and non-negative so the reference can never inject `NaN`/`inf` energy.
//! * `f32` arithmetic throughout mirrors the WESL/GPU twin bit-for-bit.
//!   Transcendental functions go through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent `f32::sqrt`.
//!
//! # References
//! * Walter et al. 2007, *Microfacet Models for Refraction through Rough
//!   Surfaces* — the GGX `D` and Smith `G`.
//! * Heitz 2014, *Understanding the Masking-Shadowing Function in Microfacet-
//!   Based BRDFs* — the height-correlated Smith `G2` and anisotropic `Lambda`.
//! * Burley 2012, *Physically-Based Shading at Disney* — the anisotropic
//!   parameterisation used by [`crate::gi::anisotropy::remap`].

use bevy_math::Vec3;
use core::f32::consts::PI;

/// Minimum GGX `alpha`.  A perfectly smooth mirror would make `alpha = 0`, which
/// turns `D` into a Dirac delta and divides by zero in the normalisation;
/// clamping keeps the lobe a finite, very sharp distribution instead.
pub const MIN_ALPHA: f32 = 1.0e-3;

/// Smallest cosine treated as "above the horizon" when forming `tan²θ`.  Keeps
/// grazing directions from producing an unbounded `Lambda`.
const MIN_COS: f32 = 1.0e-6;

/// Floor applied to squared denominators so divisions stay finite.
const DENOM_FLOOR: f32 = 1.0e-20;

/// Projects a world direction `w` onto the orthonormal tangent frame, returning
/// `(w·tangent, w·bitangent, w·normal)`.
///
/// The frame is assumed orthonormal; this is a change of basis, not a general
/// projection.  Pulled out so `D`, `Lambda`, and the sampler agree on the exact
/// same arithmetic.
#[inline]
pub fn to_tangent_space(w: Vec3, tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Vec3 {
    Vec3::new(w.dot(tangent), w.dot(bitangent), w.dot(normal))
}

/// Anisotropic GGX normal-distribution function `D(h)` for a *world-space* unit
/// half vector `h` and the orthonormal tangent frame `(tangent, bitangent,
/// normal)`.
///
/// Evaluates
/// `D = 1 / (π · α_t · α_b · ((h·t/α_t)² + (h·b/α_b)² + (h·n)²)²)`.
///
/// `D` is normalised so `∫ D(h) (n·h) dω = 1` over the hemisphere (verified by
/// numerical integration in the module tests).  Returns `0` for a back-facing
/// half vector (`h·n <= 0`) and clamps both widths to [`MIN_ALPHA`].
#[inline]
pub fn ndf_anisotropic(
    h: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let hl = to_tangent_space(h, tangent, bitangent, normal);
    ndf_anisotropic_local(hl, alpha_t, alpha_b)
}

/// Anisotropic GGX `D` for a half vector already expressed in the tangent frame
/// (`x` = tangent, `y` = bitangent, `z` = normal).
///
/// Shared by [`ndf_anisotropic`] and the VNDF sampler so both paths use
/// identical arithmetic.  Returns `0` for `h.z <= 0`.
#[inline]
pub fn ndf_anisotropic_local(h: Vec3, alpha_t: f32, alpha_b: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let at = alpha_t.max(MIN_ALPHA);
    let ab = alpha_b.max(MIN_ALPHA);
    let xt = h.x / at;
    let yb = h.y / ab;
    let s = xt * xt + yb * yb + h.z * h.z;
    let denom = PI * at * ab * (s * s).max(DENOM_FLOOR);
    let d = 1.0 / denom;
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

/// Smith `Lambda` for an anisotropic GGX surface and a *world-space* direction
/// `w` in the orthonormal tangent frame.
///
/// `Lambda(w) = (-1 + sqrt(1 + (α_t²(w·t)² + α_b²(w·b)²) / (w·n)²)) / 2`.
/// Returns `0` for a grazing/degenerate direction so `G1 = 1 / (1 + Lambda)`
/// stays well defined.
#[inline]
pub fn smith_lambda(
    w: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let wl = to_tangent_space(w, tangent, bitangent, normal);
    smith_lambda_local(wl, alpha_t, alpha_b)
}

/// Anisotropic Smith `Lambda` for a direction already in the tangent frame.
#[inline]
pub fn smith_lambda_local(w: Vec3, alpha_t: f32, alpha_b: f32) -> f32 {
    let cz = w.z.abs().clamp(MIN_COS, 1.0);
    let at = alpha_t.max(MIN_ALPHA);
    let ab = alpha_b.max(MIN_ALPHA);
    let num = (at * w.x) * (at * w.x) + (ab * w.y) * (ab * w.y);
    let tan2 = num / (cz * cz);
    let lambda = 0.5 * (-1.0 + (1.0 + tan2).max(0.0).sqrt());
    if lambda.is_finite() { lambda.max(0.0) } else { 0.0 }
}

/// Smith single-direction masking term `G1 = 1 / (1 + Lambda(w))` (world-space).
///
/// Clamped to `[0, 1]`; a back-facing direction (`w·n <= 0`) returns `0`.
#[inline]
pub fn smith_g1(
    w: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    if w.dot(normal) <= 0.0 {
        return 0.0;
    }
    let g = 1.0 / (1.0 + smith_lambda(w, tangent, bitangent, normal, alpha_t, alpha_b));
    g.clamp(0.0, 1.0)
}

/// `G1` for a direction already in the tangent frame.
#[inline]
pub fn smith_g1_local(w: Vec3, alpha_t: f32, alpha_b: f32) -> f32 {
    if w.z <= 0.0 {
        return 0.0;
    }
    let g = 1.0 / (1.0 + smith_lambda_local(w, alpha_t, alpha_b));
    g.clamp(0.0, 1.0)
}

/// Height-correlated Smith masking–shadowing `G2` for the anisotropic model.
///
/// `G2 = 1 / (1 + Lambda(wo) + Lambda(wi))` (Heitz 2014).  `wo`/`wi` are
/// world-space unit directions; returns `0` if either is below the surface.
#[inline]
pub fn smith_g2(
    wo: Vec3,
    wi: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    normal: Vec3,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    if wo.dot(normal) <= 0.0 || wi.dot(normal) <= 0.0 {
        return 0.0;
    }
    let lo = smith_lambda(wo, tangent, bitangent, normal, alpha_t, alpha_b);
    let li = smith_lambda(wi, tangent, bitangent, normal, alpha_t, alpha_b);
    let g = 1.0 / (1.0 + lo + li);
    g.clamp(0.0, 1.0)
}

/// Height-correlated Smith `G2` for tangent-frame directions.
#[inline]
pub fn smith_g2_local(wo: Vec3, wi: Vec3, alpha_t: f32, alpha_b: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let g = 1.0
        / (1.0 + smith_lambda_local(wo, alpha_t, alpha_b) + smith_lambda_local(wi, alpha_t, alpha_b));
    g.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;
    use core::f32::consts::TAU;

    /// Canonical identity tangent frame: tangent `+X`, bitangent `+Y`, normal
    /// `+Z`.  Lets the tests feed tangent-space vectors directly to the
    /// world-space entry points.
    fn identity_frame() -> (Vec3, Vec3, Vec3) {
        (Vec3::X, Vec3::Y, Vec3::Z)
    }

    #[test]
    fn ndf_is_nonnegative_and_finite_everywhere() {
        let (t, b, n) = identity_frame();
        for &(at, ab) in &[(0.1f32, 0.4f32), (0.5, 0.5), (0.9, 0.2), (MIN_ALPHA, 1.0)] {
            for i in 0..24u32 {
                for j in 0..24u32 {
                    let theta = (i as f32 + 0.5) / 24.0 * (PI / 2.0);
                    let phi = (j as f32 + 0.5) / 24.0 * TAU;
                    let (st, ct) = ops::sin_cos(theta);
                    let (sp, cp) = ops::sin_cos(phi);
                    let h = Vec3::new(st * cp, st * sp, ct);
                    let d = ndf_anisotropic(h, t, b, n, at, ab);
                    assert!(d.is_finite() && d >= 0.0, "D={d} at ({at},{ab})");
                }
            }
        }
        // A back-facing half vector carries no energy.
        assert_eq!(ndf_anisotropic(Vec3::new(0.0, 0.0, -1.0), t, b, n, 0.3, 0.5), 0.0);
    }

    #[test]
    fn ndf_integrates_to_one_over_hemisphere() {
        // ∫ D(h) (n·h) dω = 1.  Integrate in (theta, phi) with the midpoint
        // rule; anisotropy means we must sweep phi too.
        let (t, b, n) = identity_frame();
        for &(at, ab) in &[(0.3f32, 0.3f32), (0.25, 0.6), (0.5, 0.15)] {
            let n_theta = 400usize;
            let n_phi = 400usize;
            let d_theta = (PI / 2.0) / n_theta as f32;
            let d_phi = TAU / n_phi as f32;
            let mut sum = 0.0f64;
            for i in 0..n_theta {
                let theta = (i as f32 + 0.5) * d_theta;
                let (sin_t, cos_t) = ops::sin_cos(theta);
                for k in 0..n_phi {
                    let phi = (k as f32 + 0.5) * d_phi;
                    let (sp, cp) = ops::sin_cos(phi);
                    let h = Vec3::new(sin_t * cp, sin_t * sp, cos_t);
                    let d = ndf_anisotropic(h, t, b, n, at, ab);
                    // Integrand D * (n·h) * sinθ, with n·h = cos_t.
                    sum += (d * cos_t * sin_t) as f64;
                }
            }
            let integral = sum * (d_theta as f64) * (d_phi as f64);
            assert!(
                (integral - 1.0).abs() < 0.02,
                "at={at} ab={ab} integral={integral}"
            );
        }
    }

    #[test]
    fn ndf_isotropic_matches() {
        // With alpha_t == alpha_b the anisotropic D must equal the isotropic
        // GGX closed form 1/(π α² (( (n·h)² (α²-1) + 1 ))²)·α⁴ … simplified
        // using the local form with x²+y² = sin²θ.
        let alpha = 0.37f32;
        let a2 = alpha * alpha;
        for n_dot_h in [0.15f32, 0.5, 0.85, 1.0] {
            let sin_t = (1.0 - n_dot_h * n_dot_h).max(0.0).sqrt();
            let h = Vec3::new(sin_t, 0.0, n_dot_h);
            let got = ndf_anisotropic_local(h, alpha, alpha);
            let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
            let want = a2 / (PI * denom * denom);
            assert!((got - want).abs() < 1e-3, "got={got} want={want}");
        }
    }

    #[test]
    fn ndf_direction_follows_narrow_axis() {
        // The lobe is wider along the smaller-alpha axis is false; GGX width ∝
        // alpha, so the *larger* alpha axis spreads energy and the peak at a
        // fixed off-normal angle is higher along the narrower axis.  Compare a
        // half vector tilted along tangent vs bitangent.
        let (at, ab) = (0.15f32, 0.6f32);
        let n_dot_h = 0.9f32;
        let s = (1.0 - n_dot_h * n_dot_h).max(0.0).sqrt();
        let along_t = ndf_anisotropic_local(Vec3::new(s, 0.0, n_dot_h), at, ab);
        let along_b = ndf_anisotropic_local(Vec3::new(0.0, s, n_dot_h), at, ab);
        // Tilting along the narrow (tangent) axis drops D faster than along the
        // wide (bitangent) axis.
        assert!(along_b > along_t, "along_b={along_b} along_t={along_t}");
    }

    #[test]
    fn smith_terms_are_bounded_and_symmetric_in_isotropy() {
        let (t, b, n) = identity_frame();
        for alpha in [0.1f32, 0.4, 0.9] {
            for &c in &[0.1f32, 0.4, 0.8, 1.0] {
                let s = (1.0 - c * c).max(0.0).sqrt();
                let w = Vec3::new(s, 0.0, c);
                let g1 = smith_g1(w, t, b, n, alpha, alpha);
                assert!((0.0..=1.0).contains(&g1), "g1={g1}");
            }
            let wo = Vec3::new(0.3, 0.2, 0.9).normalize();
            let wi = Vec3::new(-0.1, 0.4, 0.9).normalize();
            let g2 = smith_g2(wo, wi, t, b, n, alpha, alpha);
            assert!((0.0..=1.0).contains(&g2), "g2={g2}");
        }
        // Below-horizon directions kill the shadowing term.
        let (t, b, n) = identity_frame();
        assert_eq!(smith_g2(Vec3::new(0.0, 0.0, -0.5), Vec3::Z, t, b, n, 0.4, 0.4), 0.0);
    }

    #[test]
    fn smith_lambda_matches_isotropic_closed_form() {
        // Isotropic Lambda: (-1 + sqrt(1 + α² tan²θ)) / 2.
        let alpha = 0.5f32;
        for c in [0.2f32, 0.5, 0.9] {
            let s = (1.0 - c * c).max(0.0).sqrt();
            let w = Vec3::new(s, 0.0, c);
            let got = smith_lambda_local(w, alpha, alpha);
            let tan2 = (1.0 - c * c) / (c * c);
            let want = 0.5 * (-1.0 + (1.0 + alpha * alpha * tan2).sqrt());
            assert!((got - want).abs() < 1e-5, "got={got} want={want}");
        }
    }
}
