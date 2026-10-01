//! Clearcoat ↔ base energy coupling — top-layer attenuation of the base BRDF.
//!
//! A clearcoat is a semi-transparent dielectric film floating above the base
//! material.  Energy that reflects off the *top* of the coat (its Fresnel
//! reflectance `Fc`) never reaches the base layer, so the base contribution
//! must be attenuated by `(1 - Fc)` for the layered material to conserve energy.
//! This module provides that coupling together with the direction bookkeeping
//! needed when the clearcoat has its **own normal**, independent of the base
//! (geometric) normal — a common authoring feature for brushed/orange-peel
//! clearcoats where the coat's micro-geometry differs from the paint beneath.
//!
//! The layered evaluation this reference targets is
//!
//! ```text
//! Fc       = clearcoat_strength · F_Schlick(F0 = 0.04, NoV_coat)
//! layered  = base_brdf · (1 - Fc)  +  clearcoat_contribution
//! ```
//!
//! where `base_brdf` uses cosines against the **geometric** normal and the
//! clearcoat contribution uses cosines against the **clearcoat** normal.
//!
//! # Conventions
//! * `Fc` is evaluated at the *view* cosine `NoV` against the clearcoat normal,
//!   per the task convention.  A stricter two-sided variant
//!   [`attenuate_base_two_sided`] additionally accounts for the exit path
//!   `(1 - Fc(NoL))`; it is offered for callers that want the fuller
//!   Weidlich–Wilkie-style absorption but is *not* used by the default layering.
//! * All reflectances/weights are clamped to `[0, 1]`; all BRDF values are
//!   floored at `0`; every output is verified finite so coupling can never
//!   inject `NaN`/`inf` or negative energy.
//! * [`LayerCosines`] clamps every cosine to `[0, 1]` (upper hemisphere only),
//!   keeping the base and clearcoat normals strictly separated.
//! * Pure functions only: no RNG, I/O, GPU, globals or `unsafe`.
//!
//! # References
//! * Khronos `KHR_materials_clearcoat` — the `(1 - Fc)` base attenuation and the
//!   independent clearcoat-normal convention.
//! * Weidlich & Wilkie 2007, *Arbitrarily Layered Micro-Facet Surfaces* — the
//!   physical basis for layered Fresnel attenuation.
//! * Burley 2015, *Extending the Disney BRDF to a BSDF* — clearcoat layering.

use bevy_math::Vec3;

use crate::gi::spec_gi::ggx_lobe::fresnel_schlick_scalar;

use super::lobe::CLEARCOAT_F0;

/// Clamped cosine `max(0, n · w)` for a (nominally unit) normal and direction.
///
/// Both vectors are expected to be normalised; the dot product is clamped to
/// `[-1, 1]` to absorb floating-point drift and then floored at `0` so only the
/// upper hemisphere contributes.
#[inline]
pub fn clamp_cosine(normal: Vec3, w: Vec3) -> f32 {
    normal.dot(w).clamp(-1.0, 1.0).max(0.0)
}

/// Light/view cosines resolved against both the geometric and clearcoat normals.
///
/// The clearcoat may carry a normal distinct from the base/geometric normal, so
/// the base lobe and the clearcoat lobe see different incidence angles.  This
/// bundle keeps the four clamped cosines together so the high-level evaluator
/// never mixes them up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerCosines {
    /// Geometric-normal · view cosine (base layer).
    pub geo_n_dot_v: f32,
    /// Geometric-normal · light cosine (base layer).
    pub geo_n_dot_l: f32,
    /// Clearcoat-normal · view cosine (coat layer).
    pub coat_n_dot_v: f32,
    /// Clearcoat-normal · light cosine (coat layer).
    pub coat_n_dot_l: f32,
}

impl LayerCosines {
    /// Builds the four clamped cosines from world-space directions and normals.
    ///
    /// `wi` is the incoming/light direction and `wo` the outgoing/view
    /// direction, both pointing *away* from the surface; `geo_normal` and
    /// `coat_normal` are the base and clearcoat normals.  All cosines are
    /// clamped to the upper hemisphere via [`clamp_cosine`].
    #[inline]
    pub fn from_dirs(wi: Vec3, wo: Vec3, geo_normal: Vec3, coat_normal: Vec3) -> Self {
        Self {
            geo_n_dot_v: clamp_cosine(geo_normal, wo),
            geo_n_dot_l: clamp_cosine(geo_normal, wi),
            coat_n_dot_v: clamp_cosine(coat_normal, wo),
            coat_n_dot_l: clamp_cosine(coat_normal, wi),
        }
    }

    /// `true` when both layers face the light *and* the viewer, i.e. every
    /// cosine is strictly positive and the layered BRDF can carry energy.
    #[inline]
    pub fn is_front_facing(&self) -> bool {
        self.geo_n_dot_v > 0.0
            && self.geo_n_dot_l > 0.0
            && self.coat_n_dot_v > 0.0
            && self.coat_n_dot_l > 0.0
    }
}

/// Top-layer Fresnel reflectance `Fc = strength · F_Schlick(0.04, NoV)`.
///
/// `clearcoat_strength ∈ [0, 1]` scales the coat's presence (glTF
/// `clearcoatFactor`); `n_dot_v` is the view cosine against the clearcoat
/// normal.  The result is clamped to `[0, 1]` so it is always a valid
/// attenuation weight.
#[inline]
pub fn clearcoat_fresnel_reflectance(clearcoat_strength: f32, n_dot_v: f32) -> f32 {
    let strength = if clearcoat_strength.is_finite() {
        clearcoat_strength.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let f = fresnel_schlick_scalar(CLEARCOAT_F0, n_dot_v.max(0.0));
    let r = strength * f;
    if r.is_finite() { r.clamp(0.0, 1.0) } else { 0.0 }
}

/// Attenuates a base BRDF value by the clearcoat transmission `(1 - Fc)`.
///
/// `base_brdf` is floored at `0` and `fc` clamped to `[0, 1]`, so the result is
/// always a finite, non-negative value no larger than `base_brdf`.
#[inline]
pub fn attenuate_base(base_brdf: f32, fc: f32) -> f32 {
    let base = base_brdf.max(0.0);
    let transmit = (1.0 - fc.clamp(0.0, 1.0)).max(0.0);
    let r = base * transmit;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

/// Stricter two-sided attenuation `base · (1 - Fc_view) · (1 - Fc_light)`.
///
/// Accounts for Fresnel loss on *both* the entry (view) and exit (light) paths
/// through the coat, a closer approximation to Weidlich–Wilkie layering.  Not
/// used by the default [`combine_layers`] (which follows the single-sided task
/// convention) but available to callers wanting the fuller model.
#[inline]
pub fn attenuate_base_two_sided(base_brdf: f32, fc_view: f32, fc_light: f32) -> f32 {
    let base = base_brdf.max(0.0);
    let tv = (1.0 - fc_view.clamp(0.0, 1.0)).max(0.0);
    let tl = (1.0 - fc_light.clamp(0.0, 1.0)).max(0.0);
    let r = base * tv * tl;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

/// Combines the attenuated base and the clearcoat contribution:
/// `layered = base · (1 - Fc) + clearcoat_contribution`.
///
/// `clearcoat_contribution` is the already strength-scaled clearcoat lobe value.
/// Both terms are floored at `0` and the sum is verified finite, so the layered
/// BRDF is always a valid non-negative reflectance density.
#[inline]
pub fn combine_layers(base_brdf: f32, clearcoat_contribution: f32, fc: f32) -> f32 {
    let base = attenuate_base(base_brdf, fc);
    let coat = clearcoat_contribution.max(0.0);
    let r = base + coat;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

/// Coat transmission weight `1 - Fc`, i.e. the fraction of base radiance that
/// survives the top-layer Fresnel reflection.
///
/// `fc` is clamped to `[0, 1]`; the result is likewise in `[0, 1]`.
#[inline]
pub fn coat_transmission(fc: f32) -> f32 {
    (1.0 - fc.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Two-sided top-layer reflectance accounting for both the view and light
/// crossings of the coat: `1 - (1 - Fc_view)·(1 - Fc_light)`.
///
/// This is the effective fraction of base energy removed by the coat when both
/// the entry and exit Fresnel losses are modelled (the companion of
/// [`attenuate_base_two_sided`]).  All inputs are sanitised and the result is
/// clamped to `[0, 1]`.
#[inline]
pub fn clearcoat_fresnel_reflectance_two_sided(
    clearcoat_strength: f32,
    n_dot_v: f32,
    n_dot_l: f32,
) -> f32 {
    let fc_v = clearcoat_fresnel_reflectance(clearcoat_strength, n_dot_v);
    let fc_l = clearcoat_fresnel_reflectance(clearcoat_strength, n_dot_l);
    let r = 1.0 - (1.0 - fc_v) * (1.0 - fc_l);
    if r.is_finite() { r.clamp(0.0, 1.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn clamp_cosine_floors_backfacing() {
        let n = Vec3::Z;
        assert!((clamp_cosine(n, Vec3::Z) - 1.0).abs() < EPS);
        assert_eq!(clamp_cosine(n, Vec3::NEG_Z), 0.0);
        // Oblique.
        let w = Vec3::new(0.0, 0.6, 0.8).normalize();
        assert!((clamp_cosine(n, w) - 0.8).abs() < 1e-4);
    }

    #[test]
    fn layer_cosines_keep_normals_independent() {
        let wi = Vec3::new(0.0, 0.0, 1.0);
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let geo = Vec3::Z;
        // Tilted clearcoat normal sees a smaller cosine than the base normal.
        let coat = Vec3::new(0.0, 0.5, 0.866).normalize();
        let c = LayerCosines::from_dirs(wi, wo, geo, coat);
        assert!((c.geo_n_dot_v - 1.0).abs() < EPS);
        assert!(c.coat_n_dot_v < c.geo_n_dot_v);
        assert!(c.is_front_facing());
    }

    #[test]
    fn front_facing_requires_all_positive() {
        let c = LayerCosines {
            geo_n_dot_v: 0.5,
            geo_n_dot_l: 0.5,
            coat_n_dot_v: 0.0,
            coat_n_dot_l: 0.5,
        };
        assert!(!c.is_front_facing());
    }

    #[test]
    fn fresnel_reflectance_endpoints_and_strength() {
        // Normal incidence, full strength -> F0.
        assert!((clearcoat_fresnel_reflectance(1.0, 1.0) - CLEARCOAT_F0).abs() < 1e-6);
        // Grazing, full strength -> 1.
        assert!((clearcoat_fresnel_reflectance(1.0, 0.0) - 1.0).abs() < 1e-6);
        // Zero strength kills the coat entirely.
        assert_eq!(clearcoat_fresnel_reflectance(0.0, 0.3), 0.0);
        // Strength scales linearly.
        let full = clearcoat_fresnel_reflectance(1.0, 0.5);
        let half = clearcoat_fresnel_reflectance(0.5, 0.5);
        assert!((half - 0.5 * full).abs() < 1e-6);
    }

    #[test]
    fn attenuate_base_conserves_energy() {
        // (1 - Fc) scaling.
        assert!((attenuate_base(1.0, 0.25) - 0.75).abs() < 1e-6);
        // Never amplifies.
        assert!(attenuate_base(2.0, 0.1) <= 2.0);
        // Clamped weight.
        assert_eq!(attenuate_base(1.0, 2.0), 0.0);
        assert_eq!(attenuate_base(-5.0, 0.1), 0.0);
    }

    #[test]
    fn two_sided_is_never_brighter_than_single_sided() {
        let one = attenuate_base(1.0, 0.3);
        let two = attenuate_base_two_sided(1.0, 0.3, 0.2);
        assert!(two <= one + 1e-6, "two={two} one={one}");
        assert!(two >= 0.0);
    }

    #[test]
    fn combine_adds_attenuated_base_and_coat() {
        let base = 0.8f32;
        let coat = 0.3f32;
        let fc = 0.2f32;
        let expected = base * (1.0 - fc) + coat;
        assert!((combine_layers(base, coat, fc) - expected).abs() < 1e-6);
    }

    #[test]
    fn combine_with_zero_coat_is_just_attenuated_base() {
        assert!((combine_layers(0.5, 0.0, 0.1) - attenuate_base(0.5, 0.1)).abs() < 1e-6);
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert!(attenuate_base(f32::NAN, 0.1).is_finite());
        assert!(combine_layers(f32::INFINITY, 0.0, 0.1).is_finite());
        assert!(clearcoat_fresnel_reflectance(f32::NAN, 0.5).is_finite());
        let c = LayerCosines::from_dirs(Vec3::ZERO, Vec3::ZERO, Vec3::Z, Vec3::Z);
        assert!(c.geo_n_dot_v.is_finite());
    }

    #[test]
    fn coat_transmission_is_complement_of_fc() {
        assert!((coat_transmission(0.25) - 0.75).abs() < 1e-6);
        assert_eq!(coat_transmission(2.0), 0.0);
        assert_eq!(coat_transmission(-1.0), 1.0);
    }

    #[test]
    fn two_sided_reflectance_exceeds_single_sided() {
        let one = clearcoat_fresnel_reflectance(1.0, 0.5);
        let two = clearcoat_fresnel_reflectance_two_sided(1.0, 0.5, 0.5);
        assert!(two >= one - 1e-6, "two={two} one={one}");
        assert!((0.0..=1.0).contains(&two));
        // Consistent with the two-sided base attenuation.
        let base = 1.0f32;
        let fc_v = clearcoat_fresnel_reflectance(1.0, 0.5);
        let fc_l = clearcoat_fresnel_reflectance(1.0, 0.5);
        let via_atten = attenuate_base_two_sided(base, fc_v, fc_l);
        assert!((via_atten - (1.0 - two)).abs() < 1e-6, "atten={via_atten} 1-two={}", 1.0 - two);
    }
}
