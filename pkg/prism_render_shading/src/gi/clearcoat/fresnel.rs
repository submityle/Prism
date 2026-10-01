//! Clearcoat Fresnel & refraction helpers — Weidlich–Wilkie layering primitives.
//!
//! When light enters a clearcoat it *refracts* at the air→coat interface, so the
//! base layer underneath is lit at a steeper (smaller) angle than the geometric
//! incidence.  This module collects the pure building blocks for that layered
//! optics: Snell refraction of the incidence cosine into the coat, a
//! roughness-aware ("roughened") Schlick Fresnel, the plain clearcoat Schlick
//! Fresnel, and orientation/visibility clamps used to gate the coat lobe.
//!
//! These are deliberately *simplified* Weidlich–Wilkie primitives: they expose
//! the refraction-induced angle change and Fresnel roughening without modelling
//! the full internal multiple-scattering integral, which is the right trade-off
//! for a real-time clearcoat matching the glTF `KHR_materials_clearcoat`
//! appearance while staying analytic and branch-stable.
//!
//! # Conventions
//! * The clearcoat is a dielectric with a fixed [`CLEARCOAT_IOR`] of `1.5`
//!   (relative to air), so entering the coat always refracts *toward* the normal
//!   and never undergoes total internal reflection — [`refracted_cosine`] is
//!   therefore always well defined for a front-facing ray.
//! * All cosines are clamped to `[0, 1]`; square-root arguments are floored at
//!   `0`; every output is verified finite so these helpers can never emit
//!   `NaN`/`inf`.
//! * Pure functions only (no RNG, I/O, GPU, globals or `unsafe`).  Transcendental
//!   needs are limited to `sqrt`, taken through the inherent `f32::sqrt`.
//!
//! # References
//! * Weidlich & Wilkie 2007, *Arbitrarily Layered Micro-Facet Surfaces* — the
//!   refraction-into-coat angle change and layered Fresnel.
//! * Fdez-Agüera 2019, *A Multiple-Scattering Microfacet Model for Real-Time
//!   Image-Based Lighting* — the roughness-aware Fresnel used by [`fresnel_schlick_roughened`].
//! * Khronos `KHR_materials_clearcoat` — the fixed `IOR = 1.5` clearcoat.

use bevy_math::ops;

use crate::gi::spec_gi::ggx_lobe::fresnel_schlick_scalar;

use super::lobe::CLEARCOAT_F0;

/// Relative index of refraction of a clearcoat layer (coat / air).
pub const CLEARCOAT_IOR: f32 = 1.5;

/// Refracts an incidence cosine from air into a medium of relative index `ior`.
///
/// Given `cos_i = cos(θ_i)` (the air-side incidence cosine, clamped to
/// `[0, 1]`), applies Snell's law `sin θ_t = sin θ_i / ior` and returns
/// `cos θ_t`.  Because `ior ≥ 1` for a coat the ray bends *toward* the normal,
/// so `cos θ_t ≥ cos θ_i` and no total internal reflection can occur.  A
/// non-physical `ior ≤ 0` falls back to the un-refracted cosine.
#[inline]
pub fn refracted_cosine(cos_i: f32, ior: f32) -> f32 {
    let ci = cos_i.clamp(0.0, 1.0);
    if !(ior > 0.0) || !ior.is_finite() {
        return ci;
    }
    let sin2_i = (1.0 - ci * ci).max(0.0);
    let sin2_t = sin2_i / (ior * ior);
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    if cos_t.is_finite() {
        cos_t.clamp(0.0, 1.0)
    } else {
        ci
    }
}

/// Refracts an incidence cosine into the clearcoat using the fixed
/// [`CLEARCOAT_IOR`].
///
/// Convenience wrapper over [`refracted_cosine`] for the standard `IOR = 1.5`
/// coat; the returned cosine is the angle at which the *base* layer is actually
/// illuminated beneath the coat.
#[inline]
pub fn weidlich_wilkie_cosine(cos_outside: f32) -> f32 {
    refracted_cosine(cos_outside, CLEARCOAT_IOR)
}

/// Roughness-aware ("roughened") Schlick Fresnel.
///
/// As surfaces roughen, the grazing Fresnel peak is dampened because micro-facet
/// normals spread the effective incidence.  This raises the grazing limit from
/// `1` toward `max(1 - roughness, f0)`, following Fdez-Agüera 2019:
///
/// ```text
/// F = f0 + (max(1 - roughness, f0) - f0) · (1 - cosθ)^5
/// ```
///
/// With `roughness = 0` it collapses exactly to the standard Schlick Fresnel.
/// `cos_theta` is clamped to `[0, 1]`, `roughness` to `[0, 1]`, and the result
/// to `[0, 1]`.
#[inline]
pub fn fresnel_schlick_roughened(f0: f32, cos_theta: f32, roughness: f32) -> f32 {
    let f0 = if f0.is_finite() { f0.clamp(0.0, 1.0) } else { 0.0 };
    let r = roughness.clamp(0.0, 1.0);
    let c = (1.0 - cos_theta.clamp(0.0, 1.0)).max(0.0);
    let c5 = (c * c) * (c * c) * c;
    let f_max = (1.0 - r).max(f0);
    let f = f0 + (f_max - f0) * c5;
    if f.is_finite() { f.clamp(0.0, 1.0) } else { f0 }
}

/// Plain scalar Schlick Fresnel of the clearcoat at the fixed [`CLEARCOAT_F0`].
///
/// `cos_theta` is clamped to `[0, 1]`; equivalent to the lobe's Fresnel and
/// provided here for layering code that works purely in the Fresnel module.
#[inline]
pub fn fresnel_coat(cos_theta: f32) -> f32 {
    fresnel_schlick_scalar(CLEARCOAT_F0, cos_theta.max(0.0))
}

/// Clamps a cosine to a valid visibility weight in `[0, 1]`.
///
/// Non-finite inputs collapse to `0` so a degenerate direction contributes no
/// energy rather than poisoning the result.
#[inline]
pub fn visibility_clamp(cos_theta: f32) -> f32 {
    if cos_theta.is_finite() {
        cos_theta.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Binary orientation gate: `1` when the coat faces both the viewer and the
/// light (both cosines strictly positive), `0` otherwise.
///
/// Used to suppress clearcoat energy on silhouettes and back-lit configurations
/// where the microfacet model is undefined.
#[inline]
pub fn facing_weight(n_dot_v: f32, n_dot_l: f32) -> f32 {
    if n_dot_v > 0.0 && n_dot_l > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Normal-incidence reflectance `F0` of a dielectric interface of relative
/// index `ior`.
///
/// `F0 = ((ior - 1) / (ior + 1))²`.  A non-physical `ior ≤ 0` (or non-finite)
/// falls back to the standard clearcoat [`CLEARCOAT_F0`].  For `ior = 1.5` this
/// returns `0.04`, matching the clearcoat constant.
#[inline]
pub fn f0_from_ior(ior: f32) -> f32 {
    if !(ior > 0.0) || !ior.is_finite() {
        return CLEARCOAT_F0;
    }
    let r = (ior - 1.0) / (ior + 1.0);
    let f0 = r * r;
    if f0.is_finite() { f0.clamp(0.0, 1.0) } else { CLEARCOAT_F0 }
}

/// Schlick Fresnel for an arbitrary dielectric `ior`, deriving `F0` via
/// [`f0_from_ior`] and evaluating at `cos_theta ∈ [0, 1]`.
#[inline]
pub fn fresnel_from_ior(ior: f32, cos_theta: f32) -> f32 {
    fresnel_schlick_scalar(f0_from_ior(ior), cos_theta.max(0.0))
}

/// Beer–Lambert transmittance through the clearcoat along a refracted path.
///
/// Although a clearcoat is nominally clear, authored coats may carry a faint
/// absorption; this returns `exp(-absorption · thickness / cosθ_t)`, i.e. the
/// fraction of radiance surviving a single traversal at refracted cosine
/// `cos_theta_t`.  `absorption` and `thickness` are floored at `0`; the cosine
/// is floored at a small epsilon so a grazing path cannot divide by zero.  With
/// `absorption = 0` the transmittance is exactly `1` (a perfectly clear coat).
#[inline]
pub fn coat_transmittance(absorption: f32, thickness: f32, cos_theta_t: f32) -> f32 {
    let a = absorption.max(0.0);
    let t = thickness.max(0.0);
    if a == 0.0 || t == 0.0 {
        return 1.0;
    }
    let c = cos_theta_t.clamp(0.0, 1.0).max(1.0e-4);
    let tau = ops::exp(-a * t / c);
    if tau.is_finite() { tau.clamp(0.0, 1.0) } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_incidence_is_not_bent() {
        assert!((refracted_cosine(1.0, CLEARCOAT_IOR) - 1.0).abs() < 1e-6);
        assert!((weidlich_wilkie_cosine(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn refraction_bends_toward_normal() {
        // Entering a denser medium increases the cosine (smaller angle).
        for ci in [0.1f32, 0.3, 0.6, 0.9] {
            let ct = weidlich_wilkie_cosine(ci);
            assert!(ct >= ci - 1e-6, "ci={ci} ct={ct}");
            assert!((0.0..=1.0).contains(&ct), "ct={ct}");
        }
    }

    #[test]
    fn refraction_handles_nonphysical_ior() {
        assert_eq!(refracted_cosine(0.5, 0.0), 0.5);
        assert_eq!(refracted_cosine(0.5, -2.0), 0.5);
        assert!(refracted_cosine(0.5, f32::INFINITY).is_finite());
    }

    #[test]
    fn roughened_fresnel_reduces_to_schlick_when_smooth() {
        for c in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let rough0 = fresnel_schlick_roughened(CLEARCOAT_F0, c, 0.0);
            let plain = fresnel_coat(c);
            assert!((rough0 - plain).abs() < 1e-6, "c={c}");
        }
    }

    #[test]
    fn roughened_fresnel_damps_grazing_peak() {
        // At grazing a rough coat reflects less than a smooth one.
        let smooth = fresnel_schlick_roughened(CLEARCOAT_F0, 0.0, 0.0);
        let rough = fresnel_schlick_roughened(CLEARCOAT_F0, 0.0, 0.6);
        assert!(rough < smooth, "rough={rough} smooth={smooth}");
        // Normal incidence is unchanged by roughness.
        let n_smooth = fresnel_schlick_roughened(CLEARCOAT_F0, 1.0, 0.0);
        let n_rough = fresnel_schlick_roughened(CLEARCOAT_F0, 1.0, 0.6);
        assert!((n_smooth - n_rough).abs() < 1e-6);
    }

    #[test]
    fn fresnel_coat_endpoints() {
        assert!((fresnel_coat(1.0) - CLEARCOAT_F0).abs() < 1e-6);
        assert!((fresnel_coat(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn visibility_and_facing_clamps() {
        assert_eq!(visibility_clamp(-0.3), 0.0);
        assert_eq!(visibility_clamp(2.0), 1.0);
        assert!((visibility_clamp(0.4) - 0.4).abs() < 1e-6);
        assert_eq!(visibility_clamp(f32::NAN), 0.0);
        assert_eq!(facing_weight(0.5, 0.5), 1.0);
        assert_eq!(facing_weight(-0.1, 0.5), 0.0);
        assert_eq!(facing_weight(0.5, 0.0), 0.0);
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert!(refracted_cosine(f32::NAN, CLEARCOAT_IOR).is_finite());
        assert!(fresnel_schlick_roughened(f32::NAN, 0.5, 0.3).is_finite());
        assert!(fresnel_coat(f32::NEG_INFINITY).is_finite());
    }

    #[test]
    fn f0_from_ior_matches_clearcoat_constant() {
        assert!((f0_from_ior(CLEARCOAT_IOR) - CLEARCOAT_F0).abs() < 1e-4);
        // Higher IOR reflects more at normal incidence.
        assert!(f0_from_ior(2.0) > f0_from_ior(1.5));
        // Non-physical IOR falls back to the clearcoat constant.
        assert_eq!(f0_from_ior(0.0), CLEARCOAT_F0);
        assert_eq!(f0_from_ior(f32::NAN), CLEARCOAT_F0);
    }

    #[test]
    fn fresnel_from_ior_endpoints() {
        assert!((fresnel_from_ior(1.5, 1.0) - CLEARCOAT_F0).abs() < 1e-4);
        assert!((fresnel_from_ior(1.5, 0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn clear_coat_transmits_fully() {
        // Zero absorption or zero thickness => perfectly clear.
        assert_eq!(coat_transmittance(0.0, 1.0, 0.8), 1.0);
        assert_eq!(coat_transmittance(0.5, 0.0, 0.8), 1.0);
    }

    #[test]
    fn absorbing_coat_darkens_with_path_length() {
        // A steeper (smaller cosine) path travels further and absorbs more.
        let steep = coat_transmittance(1.0, 0.5, 0.2);
        let shallow = coat_transmittance(1.0, 0.5, 0.9);
        assert!(steep < shallow, "steep={steep} shallow={shallow}");
        assert!((0.0..=1.0).contains(&steep));
        assert!((0.0..=1.0).contains(&shallow));
    }

    #[test]
    fn transmittance_is_robust_to_degenerate_cosine() {
        assert!(coat_transmittance(1.0, 1.0, 0.0).is_finite());
        assert!(coat_transmittance(1.0, 1.0, f32::NAN).is_finite());
    }
}
