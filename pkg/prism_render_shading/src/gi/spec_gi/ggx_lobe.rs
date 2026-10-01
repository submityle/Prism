//! GGX / Trowbridge–Reitz specular microfacet lobe — CPU golden reference.
//!
//! This module is the backend-neutral numerical reference for the glossy
//! specular BRDF used across the engine's reflection and specular-GI passes.
//! It implements the full isotropic *and* anisotropic GGX microfacet model:
//!
//! * the normal-distribution function `D` (Trowbridge–Reitz / GGX),
//! * the Smith height-correlated masking–shadowing term `G2` (and the single
//!   directional `G1`), expressed through the per-direction `Lambda` function,
//! * the Schlick Fresnel approximation `F`,
//! * importance sampling of the *visible* normal distribution (VNDF) following
//!   Heitz 2018, together with the matching solid-angle pdf of the reflected
//!   direction, and
//! * the roughness → `alpha` mappings (isotropic and anisotropic).
//!
//! # Conventions
//! * All directions are expressed in a *local shading frame* where the surface
//!   normal is `+Z`.  A direction's `z` component is therefore its cosine with
//!   the normal; the caller rotates world directions into this frame (see
//!   [`crate::gi::sample::mapping::world_from_local`]) before calling in.
//! * `wo` is the outgoing / view direction and `wi` the incoming / light
//!   direction, both pointing *away* from the surface.  Only the upper
//!   hemisphere (`z > 0`) carries reflected energy; back-facing configurations
//!   return `0`.
//! * Perceptual `roughness ∈ [0, 1]` maps to the GGX width `alpha = roughness^2`
//!   (Burley/Disney).  `alpha` is clamped to [`MIN_ALPHA`] so a mirror
//!   (`roughness → 0`) stays a finite, extremely sharp lobe instead of a Dirac
//!   delta that would divide by zero.
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals
//!   or `unsafe`) and defends against degeneracy: denominators are floored,
//!   square-root arguments are clamped non-negative, and every result is finite
//!   and non-negative so the reference can never inject `NaN` energy.
//! * `f32` arithmetic throughout mirrors the WESL/GPU twin bit-for-bit.
//!   Transcendental functions go through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent `f32::sqrt`.
//!
//! # References
//! * Walter et al. 2007, *Microfacet Models for Refraction through Rough
//!   Surfaces* — the GGX `D` and Smith `G`.
//! * Heitz 2014, *Understanding the Masking-Shadowing Function* — the
//!   height-correlated Smith `G2`.
//! * Heitz 2018, *Sampling the GGX Distribution of Visible Normals* (JCGT) —
//!   the VNDF sampling routine and its pdf.

use bevy_math::{Vec3, ops};
use core::f32::consts::{FRAC_1_PI, PI, TAU};

/// Minimum GGX `alpha`.  A perfectly smooth mirror would make `alpha = 0`, which
/// turns `D` into a Dirac delta and divides by zero in the pdf; clamping keeps
/// the lobe a finite, very sharp distribution instead.
pub const MIN_ALPHA: f32 = 1.0e-3;

/// Maps a perceptual `roughness` to the isotropic GGX width `alpha`.
///
/// Uses the Disney/Burley convention `alpha = roughness^2`, with `roughness`
/// clamped to `[0, 1]` and the result floored at [`MIN_ALPHA`].
#[inline]
pub fn roughness_to_alpha(roughness: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    (r * r).max(MIN_ALPHA)
}

/// Maps a perceptual `roughness` and signed `anisotropy ∈ [-1, 1]` to the
/// anisotropic GGX widths `(alpha_x, alpha_y)` along the tangent/bitangent.
///
/// Follows the Disney parameterisation: `aspect = sqrt(1 - 0.9 * anisotropy)`,
/// `alpha_x = alpha / aspect`, `alpha_y = alpha * aspect`, so positive
/// `anisotropy` stretches the lobe along the tangent.  Both outputs are floored
/// at [`MIN_ALPHA`].
#[inline]
pub fn roughness_to_alpha_anisotropic(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let alpha = roughness_to_alpha(roughness);
    let aniso = anisotropy.clamp(-1.0, 1.0);
    let aspect = (1.0 - 0.9 * aniso).max(1.0e-4).sqrt();
    let ax = (alpha / aspect).max(MIN_ALPHA);
    let ay = (alpha * aspect).max(MIN_ALPHA);
    (ax, ay)
}

/// Isotropic GGX / Trowbridge–Reitz normal-distribution function `D(n·h)`.
///
/// `D` is normalised so `∫ D(h) (n·h) dω = 1` over the hemisphere.  Returns `0`
/// for a back-facing half vector (`n_dot_h <= 0`) and clamps `alpha` to
/// [`MIN_ALPHA`].
#[inline]
pub fn ndf_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    if n_dot_h <= 0.0 {
        return 0.0;
    }
    let a = alpha.max(MIN_ALPHA);
    let a2 = a * a;
    let cos2 = n_dot_h * n_dot_h;
    let denom = cos2 * (a2 - 1.0) + 1.0;
    let d = a2 * FRAC_1_PI / (denom * denom).max(1.0e-20);
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

/// Anisotropic GGX normal-distribution function for a local half vector `h`.
///
/// `h` is given in the shading frame (`+Z` = normal); `h.z` is the cosine with
/// the normal.  Returns `0` for a back-facing half vector and clamps both widths
/// to [`MIN_ALPHA`].
#[inline]
pub fn ndf_ggx_anisotropic(h: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let ax = alpha_x.max(MIN_ALPHA);
    let ay = alpha_y.max(MIN_ALPHA);
    let t = h.x / ax;
    let b = h.y / ay;
    let s = t * t + b * b + h.z * h.z;
    let d = 1.0 / (PI * ax * ay * (s * s).max(1.0e-20));
    if d.is_finite() { d.max(0.0) } else { 0.0 }
}

/// Smith `Lambda` for an isotropic GGX surface and a direction with cosine
/// `cos_theta` (`= n·w`).
///
/// `Lambda(w) = (-1 + sqrt(1 + alpha^2 tan^2θ)) / 2`.  Returns `0` for a
/// grazing/degenerate direction so `G1 = 1 / (1 + Lambda)` stays well defined.
#[inline]
pub fn smith_lambda(cos_theta: f32, alpha: f32) -> f32 {
    let c = cos_theta.abs().clamp(1.0e-6, 1.0);
    let a = alpha.max(MIN_ALPHA);
    let cos2 = c * c;
    let tan2 = (1.0 - cos2) / cos2;
    let lambda = 0.5 * (-1.0 + (1.0 + a * a * tan2).max(0.0).sqrt());
    if lambda.is_finite() { lambda.max(0.0) } else { 0.0 }
}

/// Smith `Lambda` for an anisotropic GGX surface and a local direction `w`.
#[inline]
pub fn smith_lambda_anisotropic(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    let cz = w.z.abs().clamp(1.0e-6, 1.0);
    let ax = alpha_x.max(MIN_ALPHA);
    let ay = alpha_y.max(MIN_ALPHA);
    let num = (ax * w.x) * (ax * w.x) + (ay * w.y) * (ay * w.y);
    let tan2 = num / (cz * cz);
    let lambda = 0.5 * (-1.0 + (1.0 + tan2).max(0.0).sqrt());
    if lambda.is_finite() { lambda.max(0.0) } else { 0.0 }
}

/// Smith single-direction masking term `G1 = 1 / (1 + Lambda)` (isotropic).
#[inline]
pub fn smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    1.0 / (1.0 + smith_lambda(cos_theta, alpha))
}

/// Height-correlated Smith masking–shadowing `G2` for the isotropic model.
///
/// `G2 = 1 / (1 + Lambda(wo) + Lambda(wi))` (Heitz 2014).  Returns `0` if either
/// direction is below the surface.
#[inline]
pub fn smith_g2(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
        return 0.0;
    }
    let g = 1.0 / (1.0 + smith_lambda(n_dot_v, alpha) + smith_lambda(n_dot_l, alpha));
    g.clamp(0.0, 1.0)
}

/// Height-correlated Smith `G2` for the anisotropic model, from local
/// view/light directions.
#[inline]
pub fn smith_g2_anisotropic(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let g = 1.0
        / (1.0
            + smith_lambda_anisotropic(wo, alpha_x, alpha_y)
            + smith_lambda_anisotropic(wi, alpha_x, alpha_y));
    g.clamp(0.0, 1.0)
}

/// Schlick Fresnel reflectance for RGB `f0` at incidence cosine `cos_theta`.
///
/// `F = f0 + (1 - f0) (1 - cosθ)^5`.  `cos_theta` is clamped to `[0, 1]` and the
/// result to `[0, 1]` per channel.
#[inline]
pub fn fresnel_schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let c = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
    let c5 = (c * c) * (c * c) * c;
    let f = f0 + (Vec3::ONE - f0) * c5;
    Vec3::new(f.x.clamp(0.0, 1.0), f.y.clamp(0.0, 1.0), f.z.clamp(0.0, 1.0))
}

/// Scalar Schlick Fresnel for a single reflectance `f0`.
#[inline]
pub fn fresnel_schlick_scalar(f0: f32, cos_theta: f32) -> f32 {
    let c = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
    let c5 = (c * c) * (c * c) * c;
    (f0 + (1.0 - f0) * c5).clamp(0.0, 1.0)
}

/// Mirror-reflects `wo` about the local normal `+Z`: `(-wo.x, -wo.y, wo.z)`.
#[inline]
pub fn reflect_z(wo: Vec3) -> Vec3 {
    Vec3::new(-wo.x, -wo.y, wo.z)
}

/// Samples a GGX *visible* normal (VNDF) for the local view direction `wo`
/// using Heitz 2018's bounded spherical-cap method.
///
/// `(u1, u2) ∈ [0, 1)^2` are canonical uniforms.  The returned half vector `h`
/// is a unit vector in the upper hemisphere distributed according to the
/// visible-normal distribution `D_v(h) = G1(wo) max(0, wo·h) D(h) / wo.z`.  A
/// degenerate (below-horizon) `wo` falls back to the geometric normal `+Z`.
#[inline]
pub fn sample_ggx_vndf(wo: Vec3, alpha_x: f32, alpha_y: f32, u1: f32, u2: f32) -> Vec3 {
    let ax = alpha_x.max(MIN_ALPHA);
    let ay = alpha_y.max(MIN_ALPHA);
    if wo.z <= 0.0 {
        return Vec3::Z;
    }
    // Stretch the view direction into the hemisphere configuration (alpha = 1).
    let vh = Vec3::new(ax * wo.x, ay * wo.y, wo.z).normalize_or_zero();
    let vh = if vh.length_squared() > 0.0 { vh } else { Vec3::Z };

    // Orthonormal basis of the stretched view (Heitz's robust construction).
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 1.0e-12 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = vh.cross(t1);

    // Sample a point on the projected (and reparameterised) disk.
    let r = u1.clamp(0.0, 1.0).sqrt();
    let phi = TAU * u2.clamp(0.0, 1.0);
    let (sin_phi, cos_phi) = ops::sin_cos(phi);
    let p1 = r * cos_phi;
    let mut p2 = r * sin_phi;
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;

    // Reproject onto the hemisphere and unstretch.
    let pz = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    let nh = p1 * t1 + p2 * t2 + pz * vh;
    let h = Vec3::new(ax * nh.x, ay * nh.y, nh.z.max(0.0)).normalize_or_zero();
    if h.length_squared() > 0.0 { h } else { Vec3::Z }
}

/// Solid-angle pdf of the VNDF *half vector* `h` for local view `wo`.
///
/// `D_v(h) = G1(wo) max(0, wo·h) D(h) / wo.z`.  Returns `0` for a degenerate
/// configuration.
#[inline]
pub fn vndf_pdf_h(wo: Vec3, h: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || h.z <= 0.0 {
        return 0.0;
    }
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    let g1 = 1.0 / (1.0 + smith_lambda_anisotropic(wo, alpha_x, alpha_y));
    let d = ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let pdf = g1 * v_dot_h * d / wo.z.max(1.0e-6);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Solid-angle pdf of a reflected direction `wi` produced by VNDF sampling.
///
/// Applies the reflection Jacobian: `pdf(wi) = D_v(h) / (4 (wo·h))` with
/// `h = normalize(wo + wi)`.  Returns `0` for below-horizon directions.
#[inline]
pub fn vndf_pdf_reflect(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = (wo + wi).normalize_or_zero();
    if h.length_squared() == 0.0 {
        return 0.0;
    }
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 1.0e-8 {
        return 0.0;
    }
    let pdf = vndf_pdf_h(wo, h, alpha_x, alpha_y) / (4.0 * v_dot_h);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Evaluates the anisotropic GGX specular BRDF (times nothing — the pure BRDF
/// value, excluding the `n·l` cosine) for local `wo`, `wi` and RGB `f0`.
///
/// `f_spec = D G2 F / (4 (n·wo) (n·wi))`.  Returns [`Vec3::ZERO`] for
/// below-horizon directions.
#[inline]
pub fn ggx_brdf(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32, f0: Vec3) -> Vec3 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return Vec3::ZERO;
    }
    let h = (wo + wi).normalize_or_zero();
    if h.length_squared() == 0.0 {
        return Vec3::ZERO;
    }
    let d = ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let g = smith_g2_anisotropic(wo, wi, alpha_x, alpha_y);
    let f = fresnel_schlick(f0, wo.dot(h).max(0.0));
    let denom = (4.0 * wo.z * wi.z).max(1.0e-6);
    let scale = d * g / denom;
    let v = f * scale;
    if v.is_finite() { v.max(Vec3::ZERO) } else { Vec3::ZERO }
}

/// Scalar GGX specular BRDF (uses a scalar `f0`), convenient for a directional
/// target / lobe weight.  Returns `0` for below-horizon directions.
#[inline]
pub fn ggx_brdf_scalar(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32, f0: f32) -> f32 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    let h = (wo + wi).normalize_or_zero();
    if h.length_squared() == 0.0 {
        return 0.0;
    }
    let d = ndf_ggx_anisotropic(h, alpha_x, alpha_y);
    let g = smith_g2_anisotropic(wo, wi, alpha_x, alpha_y);
    let f = fresnel_schlick_scalar(f0, wo.dot(h).max(0.0));
    let denom = (4.0 * wo.z * wi.z).max(1.0e-6);
    let v = d * g * f / denom;
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roughness_alpha_mapping_is_squared_and_clamped() {
        assert!((roughness_to_alpha(1.0) - 1.0).abs() < 1e-6);
        assert!((roughness_to_alpha(0.5) - 0.25).abs() < 1e-6);
        // A mirror never collapses to zero width.
        assert!(roughness_to_alpha(0.0) >= MIN_ALPHA);
        // Out-of-range input is clamped.
        assert!((roughness_to_alpha(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn anisotropic_alpha_splits_about_isotropic() {
        let (ax, ay) = roughness_to_alpha_anisotropic(0.5, 0.0);
        assert!((ax - ay).abs() < 1e-6, "isotropic when anisotropy=0");
        let (ax, ay) = roughness_to_alpha_anisotropic(0.5, 0.8);
        assert!(ax > ay, "positive anisotropy stretches the tangent axis");
    }

    #[test]
    fn ndf_integrates_to_one_over_hemisphere() {
        // ∫ D(h) (n·h) dω = 1.  Integrate over the hemisphere in (theta, phi).
        for alpha in [0.2f32, 0.5, 0.9] {
            let n_theta = 2000usize;
            let d_theta = (PI / 2.0) / n_theta as f32;
            let mut sum = 0.0f64;
            for i in 0..n_theta {
                let theta = (i as f32 + 0.5) * d_theta;
                let (sin_t, cos_t) = ops::sin_cos(theta);
                // Integrand of ∫ D (n·h) dω: D * cos * sin, times 2*pi for phi.
                sum += (ndf_ggx(cos_t, alpha) * cos_t * sin_t) as f64;
            }
            let integral = sum * (TAU as f64) * d_theta as f64;
            assert!(
                (integral - 1.0).abs() < 0.02,
                "alpha={alpha} integral={integral}"
            );
        }
    }

    #[test]
    fn ndf_isotropic_matches_anisotropic_special_case() {
        let alpha = 0.37f32;
        for n_dot_h in [0.1f32, 0.5, 0.9, 1.0] {
            let sin_t = (1.0 - n_dot_h * n_dot_h).max(0.0).sqrt();
            let h = Vec3::new(sin_t, 0.0, n_dot_h);
            let iso = ndf_ggx(n_dot_h, alpha);
            let aniso = ndf_ggx_anisotropic(h, alpha, alpha);
            assert!((iso - aniso).abs() < 1e-4, "iso={iso} aniso={aniso}");
        }
    }

    #[test]
    fn smith_g_terms_are_bounded() {
        for alpha in [0.1f32, 0.4, 0.9] {
            for c in [0.05f32, 0.3, 0.7, 1.0] {
                let g1 = smith_g1(c, alpha);
                assert!((0.0..=1.0).contains(&g1), "g1={g1}");
            }
            let g2 = smith_g2(0.8, 0.6, alpha);
            assert!((0.0..=1.0).contains(&g2), "g2={g2}");
        }
        // Below-horizon directions kill the term.
        assert_eq!(smith_g2(-0.1, 0.5, 0.4), 0.0);
    }

    #[test]
    fn fresnel_endpoints() {
        let f0 = Vec3::new(0.04, 0.04, 0.04);
        // At normal incidence F == f0.
        let f = fresnel_schlick(f0, 1.0);
        assert!((f - f0).length() < 1e-6);
        // At grazing F -> 1.
        let f = fresnel_schlick(f0, 0.0);
        assert!((f - Vec3::ONE).length() < 1e-6);
    }

    #[test]
    fn vndf_samples_are_unit_upper_hemisphere() {
        let wo = Vec3::new(0.3, 0.1, 0.95).normalize();
        for i in 0..32u32 {
            for j in 0..32u32 {
                let u1 = (i as f32 + 0.5) / 32.0;
                let u2 = (j as f32 + 0.5) / 32.0;
                let h = sample_ggx_vndf(wo, 0.3, 0.5, u1, u2);
                assert!((h.length() - 1.0).abs() < 1e-3, "not unit: {h:?}");
                assert!(h.z >= -1e-5, "below horizon: {h:?}");
            }
        }
    }

    #[test]
    fn vndf_pdf_is_positive_and_finite() {
        let wo = Vec3::new(0.2, 0.0, 0.98).normalize();
        let h = sample_ggx_vndf(wo, 0.4, 0.4, 0.3, 0.6);
        let p = vndf_pdf_h(wo, h, 0.4, 0.4);
        assert!(p.is_finite() && p > 0.0, "pdf={p}");
        // The reflected direction about h has a positive solid-angle pdf.
        let wi = (2.0 * wo.dot(h) * h - wo).normalize();
        let pr = vndf_pdf_reflect(wo, wi, 0.4, 0.4);
        assert!(pr.is_finite() && pr > 0.0, "pdf_reflect={pr}");
    }

    #[test]
    fn vndf_distribution_integrates_to_one() {
        // Monte-Carlo check that D_v integrates to 1 over the hemisphere by
        // averaging D_v / pdf_uniform with uniform-hemisphere samples.
        let wo = Vec3::new(0.25, 0.0, 0.97).normalize();
        let (ax, ay) = (0.5f32, 0.5f32);
        let n = 20_000usize;
        let mut acc = 0.0f64;
        for i in 0..n {
            // Deterministic stratified uniform-hemisphere directions.
            let a = (i as f32 + 0.5) / n as f32;
            let b = ((i * 2654435761u64 as usize) & 0xffff) as f32 / 65536.0;
            let cz = a; // cos(theta) uniform in [0,1)
            let r = (1.0 - cz * cz).max(0.0).sqrt();
            let phi = TAU * b;
            let (sp, cp) = ops::sin_cos(phi);
            let h = Vec3::new(r * cp, r * sp, cz);
            let dv = vndf_pdf_h(wo, h, ax, ay);
            // pdf of uniform hemisphere is 1/(2*pi).
            acc += (dv as f64) * (TAU as f64);
        }
        let mean = acc / n as f64;
        assert!((mean - 1.0).abs() < 0.08, "vndf integral={mean}");
    }

    #[test]
    fn brdf_is_energy_conserving_single_scatter() {
        // The single-scattering GGX BRDF never reflects more than the incident
        // energy: directional albedo <= 1 for f0 = 1 (white, lossless Fresnel).
        let wo = Vec3::new(0.3, 0.0, 0.954).normalize();
        let (ax, ay) = (0.5f32, 0.5f32);
        let n = 40_000usize;
        let mut albedo = 0.0f64;
        for i in 0..n {
            let u1 = (i as f32 + 0.5) / n as f32;
            let u2 = ((i * 48271) % n) as f32 / n as f32;
            let h = sample_ggx_vndf(wo, ax, ay, u1, u2);
            let wi = (2.0 * wo.dot(h) * h - wo).normalize();
            if wi.z <= 0.0 {
                continue;
            }
            let pdf = vndf_pdf_reflect(wo, wi, ax, ay);
            if pdf <= 0.0 {
                continue;
            }
            let f = ggx_brdf_scalar(wo, wi, ax, ay, 1.0);
            // Monte-Carlo estimate of ∫ f cos dω.
            albedo += (f * wi.z / pdf) as f64;
        }
        albedo /= n as f64;
        assert!(albedo <= 1.0 + 1e-2, "albedo={albedo} must not exceed 1");
        assert!(albedo > 0.3, "albedo={albedo} unexpectedly dark");
    }

    #[test]
    fn brdf_is_reciprocal() {
        let a = Vec3::new(0.2, 0.1, 0.974).normalize();
        let b = Vec3::new(-0.3, 0.2, 0.933).normalize();
        let fab = ggx_brdf_scalar(a, b, 0.3, 0.45, 0.5);
        let fba = ggx_brdf_scalar(b, a, 0.3, 0.45, 0.5);
        assert!((fab - fba).abs() < 1e-5, "fab={fab} fba={fba}");
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert_eq!(ndf_ggx(-0.5, 0.3), 0.0);
        assert_eq!(ggx_brdf(Vec3::NEG_Z, Vec3::Z, 0.3, 0.3, Vec3::ONE), Vec3::ZERO);
        let h = sample_ggx_vndf(Vec3::NEG_Z, 0.3, 0.3, 0.5, 0.5);
        assert!(h.is_finite());
        assert_eq!(vndf_pdf_reflect(Vec3::Z, Vec3::NEG_Z, 0.3, 0.3), 0.0);
    }
}
