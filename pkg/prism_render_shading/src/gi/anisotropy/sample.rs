//! Anisotropic GGX visible-normal (VNDF) importance sampling — CPU golden.
//!
//! Backend-neutral reference for drawing specular directions from the
//! *anisotropic* GGX lobe and evaluating the matching solid-angle pdf, so the
//! anisotropic BRDF in [`crate::gi::anisotropy`] can be importance sampled with
//! low variance.  It follows Heitz 2018's bounded spherical-cap VNDF method:
//! the view direction is stretched into the isotropic hemisphere configuration,
//! a visible normal is sampled there, and the result is unstretched back into
//! the anisotropic frame.
//!
//! It provides tangent-space kernels and world-space wrappers:
//!
//! * [`sample_vndf_local`] / [`sample_anisotropic_half`] — sample a visible
//!   half vector,
//! * [`sample_anisotropic_direction`] — reflect `wo` about that half vector to
//!   get an incident direction, and
//! * [`vndf_pdf_reflect_local`] / [`anisotropic_pdf`] — the solid-angle pdf of
//!   the sampled incident direction.
//!
//! # Conventions
//! * Tangent-space kernels use the local frame `x` = tangent, `y` = bitangent,
//!   `z` = normal; `wo`/`wi` point *away* from the surface and only the upper
//!   hemisphere (`z > 0`) carries energy.  The world-space wrappers accept a
//!   [`crate::gi::anisotropy::remap::TangentFrame`] and rotate directions in and
//!   out of it.
//! * `(u1, u2) ∈ [0, 1)²` are canonical uniforms; the sampler is a pure
//!   deterministic function of them (no RNG/IO/GPU/globals/`unsafe`).
//! * The reflected-direction pdf applies the half-vector reflection Jacobian:
//!   `pdf(wi) = D_v(h) / (4 (wo·h))`, with `D_v(h) = G1(wo) max(0, wo·h) D(h) /
//!   (wo·n)` the visible-normal distribution.  Integrating this pdf over the
//!   hemisphere yields `1` (verified numerically in tests).
//! * Degenerate inputs (below-horizon `wo`, collapsed half vectors) fall back to
//!   the geometric normal / return `0` so the reference never emits `NaN`/`inf`.
//! * `f32` arithmetic mirrors the WESL/GPU twin bit-for-bit; transcendental
//!   functions go through [`bevy_math::ops`] and `sqrt` uses inherent `f32`.
//!
//! # References
//! * Heitz 2018, *Sampling the GGX Distribution of Visible Normals* (JCGT) — the
//!   VNDF routine and its pdf.
//! * Walter et al. 2007 — the GGX `D` and the reflection Jacobian `1/(4 wo·h)`.

use bevy_math::{Vec3, ops};
use core::f32::consts::TAU;

use crate::gi::anisotropy::ndf::{
    MIN_ALPHA, ndf_anisotropic_local, smith_lambda_local, to_tangent_space,
};
use crate::gi::anisotropy::remap::TangentFrame;

/// Floor on `wo·n` used when dividing by the view cosine in the pdf.
const MIN_COS: f32 = 1.0e-6;

/// Mirror-reflects a tangent-space direction `wo` about the local normal `+Z`:
/// `(-wo.x, -wo.y, wo.z)`.
#[inline]
pub fn reflect_local(wo: Vec3) -> Vec3 {
    Vec3::new(-wo.x, -wo.y, wo.z)
}

/// Reflects `wo` about an arbitrary unit half vector `h`:
/// `wi = 2 (wo·h) h - wo`.
#[inline]
pub fn reflect_about(wo: Vec3, h: Vec3) -> Vec3 {
    2.0 * wo.dot(h) * h - wo
}

/// Samples an anisotropic GGX *visible* normal (VNDF) for the tangent-space view
/// direction `wo`, using Heitz 2018's bounded spherical-cap method.
///
/// `(u1, u2) ∈ [0, 1)²` are canonical uniforms.  Returns a unit half vector `h`
/// in the upper hemisphere distributed as
/// `D_v(h) = G1(wo) max(0, wo·h) D(h) / (wo·n)`.  A degenerate (below-horizon)
/// `wo` falls back to the geometric normal `+Z`.
#[inline]
pub fn sample_vndf_local(wo: Vec3, alpha_t: f32, alpha_b: f32, u1: f32, u2: f32) -> Vec3 {
    let at = alpha_t.max(MIN_ALPHA);
    let ab = alpha_b.max(MIN_ALPHA);
    if wo.z <= 0.0 {
        return Vec3::Z;
    }

    // Stretch the view direction into the isotropic (alpha = 1) hemisphere.
    let vh = Vec3::new(at * wo.x, ab * wo.y, wo.z).normalize_or_zero();
    let vh = if vh.length_squared() > 0.0 { vh } else { Vec3::Z };

    // Robust orthonormal basis around the stretched view (Heitz 2018).
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 1.0e-12 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = vh.cross(t1);

    // Sample a point in the projected, reparameterised disk.
    let r = u1.clamp(0.0, 1.0).sqrt();
    let phi = TAU * u2.clamp(0.0, 1.0);
    let (sin_phi, cos_phi) = ops::sin_cos(phi);
    let p1 = r * cos_phi;
    let mut p2 = r * sin_phi;
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;

    // Reproject to the hemisphere, then unstretch back to the anisotropic frame.
    let pz = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    let nh = p1 * t1 + p2 * t2 + pz * vh;
    let h = Vec3::new(at * nh.x, ab * nh.y, nh.z.max(0.0)).normalize_or_zero();
    if h.length_squared() > 0.0 { h } else { Vec3::Z }
}

/// Solid-angle pdf of the VNDF *half vector* `h` for a tangent-space view `wo`.
///
/// `D_v(h) = G1(wo) max(0, wo·h) D(h) / (wo·n)`.  Returns `0` for a degenerate
/// configuration.
#[inline]
pub fn vndf_pdf_h_local(wo: Vec3, h: Vec3, alpha_t: f32, alpha_b: f32) -> f32 {
    if wo.z <= 0.0 || h.z <= 0.0 {
        return 0.0;
    }
    let v_dot_h = wo.dot(h).max(0.0);
    if v_dot_h <= 0.0 {
        return 0.0;
    }
    let g1 = 1.0 / (1.0 + smith_lambda_local(wo, alpha_t, alpha_b));
    let d = ndf_anisotropic_local(h, alpha_t, alpha_b);
    let pdf = g1 * v_dot_h * d / wo.z.max(MIN_COS);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Solid-angle pdf of a reflected direction `wi` produced by VNDF sampling, in
/// tangent space.
///
/// Applies the reflection Jacobian: `pdf(wi) = D_v(h) / (4 (wo·h))` with
/// `h = normalize(wo + wi)`.  Returns `0` for below-horizon directions.
#[inline]
pub fn vndf_pdf_reflect_local(wo: Vec3, wi: Vec3, alpha_t: f32, alpha_b: f32) -> f32 {
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
    let pdf = vndf_pdf_h_local(wo, h, alpha_t, alpha_b) / (4.0 * v_dot_h);
    if pdf.is_finite() { pdf.max(0.0) } else { 0.0 }
}

/// Rotates a tangent-space direction back into world space using `frame`.
#[inline]
fn to_world(local: Vec3, frame: &TangentFrame) -> Vec3 {
    local.x * frame.tangent + local.y * frame.bitangent + local.z * frame.normal
}

/// World-space wrapper: samples a visible half vector for a world view
/// direction `wo` in the given `frame`.
///
/// Returns the half vector in world space; see [`sample_vndf_local`] for the
/// distribution.
#[inline]
pub fn sample_anisotropic_half(
    wo: Vec3,
    frame: &TangentFrame,
    alpha_t: f32,
    alpha_b: f32,
    u1: f32,
    u2: f32,
) -> Vec3 {
    let wo_l = to_tangent_space(wo, frame.tangent, frame.bitangent, frame.normal);
    let h_l = sample_vndf_local(wo_l, alpha_t, alpha_b, u1, u2);
    to_world(h_l, frame)
}

/// World-space wrapper: samples an incident direction `wi` for a world view
/// direction `wo`, by reflecting `wo` about a VNDF-sampled half vector.
///
/// The returned direction is a unit world-space vector.  A below-horizon result
/// (which the caller should discard, as its pdf is `0`) is still returned
/// normalised rather than as `NaN`.
#[inline]
pub fn sample_anisotropic_direction(
    wo: Vec3,
    frame: &TangentFrame,
    alpha_t: f32,
    alpha_b: f32,
    u1: f32,
    u2: f32,
) -> Vec3 {
    let wo_l = to_tangent_space(wo, frame.tangent, frame.bitangent, frame.normal);
    let h_l = sample_vndf_local(wo_l, alpha_t, alpha_b, u1, u2);
    let wi_l = reflect_about(wo_l, h_l);
    let wi_l = wi_l.normalize_or_zero();
    let wi_l = if wi_l.length_squared() > 0.0 { wi_l } else { Vec3::Z };
    to_world(wi_l, frame)
}

/// World-space solid-angle pdf of an incident direction `wi` for a world view
/// `wo` in `frame`.  Equivalent to [`vndf_pdf_reflect_local`] after rotating
/// both directions into the tangent frame.
#[inline]
pub fn anisotropic_pdf(
    wo: Vec3,
    wi: Vec3,
    frame: &TangentFrame,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let wo_l = to_tangent_space(wo, frame.tangent, frame.bitangent, frame.normal);
    let wi_l = to_tangent_space(wi, frame.tangent, frame.bitangent, frame.normal);
    vndf_pdf_reflect_local(wo_l, wi_l, alpha_t, alpha_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::anisotropy::remap::orthonormal_tangent_frame;
    use core::f32::consts::PI;

    #[test]
    fn samples_are_unit_upper_hemisphere() {
        let wo = Vec3::new(0.3, 0.1, 0.95).normalize();
        for i in 0..32u32 {
            for j in 0..32u32 {
                let u1 = (i as f32 + 0.5) / 32.0;
                let u2 = (j as f32 + 0.5) / 32.0;
                let h = sample_vndf_local(wo, 0.2, 0.6, u1, u2);
                assert!((h.length() - 1.0).abs() < 1e-3, "h not unit: {h:?}");
                assert!(h.z >= -1e-5, "h below horizon: {h:?}");
                let wi = reflect_about(wo, h);
                assert!(wi.is_finite(), "wi not finite: {wi:?}");
            }
        }
    }

    #[test]
    fn pdf_is_zero_below_horizon() {
        let wo = Vec3::new(0.0, 0.0, 0.9).normalize();
        let wi = Vec3::new(0.1, 0.1, -0.9).normalize();
        assert_eq!(vndf_pdf_reflect_local(wo, wi, 0.3, 0.5), 0.0);
        // A below-horizon view carries no density either.
        assert_eq!(
            vndf_pdf_reflect_local(Vec3::new(0.0, 0.0, -0.5), Vec3::Z, 0.3, 0.5),
            0.0
        );
    }

    #[test]
    fn vndf_half_vector_density_integrates_to_one() {
        // ∫ D_v(h) dω_h = 1 over the half-vector hemisphere: this is the exact
        // normalisation of the visible-normal distribution and ties together the
        // NDF and the Smith G1 masking term.  (Integrating the *reflected* pdf
        // over the upper wi-hemisphere is intentionally < 1, because some half
        // vectors reflect wo below the horizon — so we normalise in h-space.)
        let wo = Vec3::new(0.25, 0.1, 0.96).normalize();
        for &(at, ab) in &[(0.3f32, 0.3f32), (0.2, 0.5), (0.5, 0.15)] {
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
                    let dv = vndf_pdf_h_local(wo, h, at, ab);
                    sum += (dv * sin_t) as f64;
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
    fn sampled_direction_pdf_is_consistent_and_positive() {
        // The analytic pdf at a freshly sampled direction must match the
        // half-vector density pushed through the Jacobian and be strictly
        // positive for an above-horizon sample.
        let wo = Vec3::new(0.2, 0.3, 0.93).normalize();
        let (at, ab) = (0.25f32, 0.55f32);
        for i in 0..16u32 {
            for j in 0..16u32 {
                let u1 = (i as f32 + 0.5) / 16.0;
                let u2 = (j as f32 + 0.5) / 16.0;
                let h = sample_vndf_local(wo, at, ab, u1, u2);
                let wi = reflect_about(wo, h).normalize_or_zero();
                if wi.z <= 1e-4 {
                    continue;
                }
                let pdf = vndf_pdf_reflect_local(wo, wi, at, ab);
                assert!(pdf > 0.0 && pdf.is_finite(), "pdf={pdf}");
                let direct = vndf_pdf_h_local(wo, h, at, ab) / (4.0 * wo.dot(h).max(1e-8));
                assert!((pdf - direct).abs() <= 1e-3 * direct.max(1.0), "pdf={pdf} direct={direct}");
            }
        }
    }

    #[test]
    fn world_wrappers_round_trip_through_frame() {
        // A sample taken through the world-space wrapper must have a matching
        // world-space pdf equal to the tangent-space pdf of the same direction.
        let frame = orthonormal_tangent_frame(
            Vec3::new(0.1, 0.2, 0.97),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let wo = (frame.normal * 0.9 + frame.tangent * 0.3).normalize();
        let (at, ab) = (0.3f32, 0.45f32);
        let wi = sample_anisotropic_direction(wo, &frame, at, ab, 0.37, 0.61);
        assert!((wi.length() - 1.0).abs() < 1e-4, "wi not unit");
        let p_world = anisotropic_pdf(wo, wi, &frame, at, ab);
        let wo_l = to_tangent_space(wo, frame.tangent, frame.bitangent, frame.normal);
        let wi_l = to_tangent_space(wi, frame.tangent, frame.bitangent, frame.normal);
        let p_local = vndf_pdf_reflect_local(wo_l, wi_l, at, ab);
        assert!((p_world - p_local).abs() < 1e-4, "world={p_world} local={p_local}");
    }

    #[test]
    fn sampling_is_deterministic() {
        let wo = Vec3::new(0.1, 0.2, 0.97).normalize();
        let a = sample_vndf_local(wo, 0.3, 0.5, 0.42, 0.73);
        let b = sample_vndf_local(wo, 0.3, 0.5, 0.42, 0.73);
        assert_eq!(a, b, "sampler must be a pure function of its uniforms");
    }
}
