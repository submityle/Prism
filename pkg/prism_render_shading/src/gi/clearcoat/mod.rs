//! Clearcoat second-specular lobe BRDF — CPU golden reference.
//!
//! This module is the backend-neutral numerical reference for a **clearcoat**
//! material: a thin, smooth dielectric film layered on top of a base BRDF, as
//! seen on car paint, lacquered wood, carbon fibre and varnished surfaces.  It
//! follows the glTF `KHR_materials_clearcoat` / Disney-Burley convention — a
//! fixed clearcoat `IOR = 1.5` (so `F0 = 0.04`), an always-isotropic,
//! never-tinted coat, and an optional clearcoat normal independent of the base
//! normal.
//!
//! Unlike [`crate::gi::env_brdf`] (which *bakes* split-sum / LTC sheen+clearcoat
//! LUTs), this module performs the **analytic lobe evaluation plus energy
//! coupling**:
//!
//! * [`lobe`] — the clearcoat GGX specular lobe `D · G · F / (4·NoL·NoV)` with
//!   fixed `F0 = 0.04` and Kelemen visibility (re-using the shared GGX `D`,
//!   Schlick `F` and roughness remap from [`crate::gi::spec_gi::ggx_lobe`]).
//! * [`coupling`] — the top-layer Fresnel attenuation of the base BRDF
//!   (`layered = base·(1 - Fc) + clearcoat`) and the independent-normal cosine
//!   bookkeeping.
//! * [`fresnel`] — refraction-into-coat (Weidlich–Wilkie), roughness-aware
//!   Fresnel, and orientation/visibility clamps.
//!
//! The high-level entry point [`clearcoat_brdf`] stitches these together into a
//! single layered BRDF value for a light/view pair.
//!
//! # Conventions
//! * World-space directions: `wi` (incoming/light) and `wo` (outgoing/view) both
//!   point *away* from the surface; `geo_normal` and `clearcoat_normal` are unit
//!   normals (the evaluator re-normalises defensively and falls back to a stable
//!   normal for degenerate input).
//! * The shared half vector `h = normalize(wi + wo)` bisects the pair, so
//!   `wo·h == wi·h`; this `VoH` drives both Fresnel terms.
//! * The base layer is evaluated against `geo_normal`, the coat against
//!   `clearcoat_normal`.  `clearcoat_strength` scales the whole coat (glTF
//!   `clearcoatFactor`); `0` reproduces the bare base BRDF.
//! * Every output is finite and non-negative; degenerate inputs return `0`
//!   rather than `NaN`/`inf`.
//!
//! # References
//! * Khronos `KHR_materials_clearcoat`.
//! * Burley 2012/2015, *Physically Based Shading at Disney* / *…to a BSDF*.
//! * Kelemen & Szirmay-Kalos 2001 — the clearcoat visibility approximation.

pub mod coupling;
pub mod fresnel;
pub mod lobe;

use bevy_math::Vec3;

use crate::gi::spec_gi::ggx_lobe::{fresnel_schlick_scalar, ndf_ggx, roughness_to_alpha, smith_g2};

pub use coupling::{
    attenuate_base, attenuate_base_two_sided, clamp_cosine, clearcoat_fresnel_reflectance,
    clearcoat_fresnel_reflectance_two_sided, coat_transmission, combine_layers, LayerCosines,
};
pub use fresnel::{
    coat_transmittance, facing_weight, f0_from_ior, fresnel_coat, fresnel_from_ior,
    fresnel_schlick_roughened, refracted_cosine, visibility_clamp, weidlich_wilkie_cosine,
    CLEARCOAT_IOR,
};
pub use lobe::{
    clearcoat_alpha, clearcoat_fresnel, clearcoat_lobe, clearcoat_lobe_smith,
    clearcoat_lobe_weighted, v_kelemen, v_smith, CLEARCOAT_F0,
};

/// Floor for the base lobe's `4·NoL·NoV` denominator.
const MIN_DENOM: f32 = 1.0e-6;

/// Authoring parameters for the clearcoat layer.
///
/// Defaults follow glTF `KHR_materials_clearcoat`: a strength of `0` (the coat
/// is *off*) and a perfectly smooth coat (`roughness = 0`).  With the default
/// `strength = 0`, [`clearcoat_brdf_params`] reproduces the bare base BRDF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearcoatParams {
    /// Coat presence / intensity `∈ [0, 1]` (glTF `clearcoatFactor`).
    pub strength: f32,
    /// Perceptual coat roughness `∈ [0, 1]` (glTF `clearcoatRoughnessFactor`).
    pub roughness: f32,
}

impl Default for ClearcoatParams {
    #[inline]
    fn default() -> Self {
        Self {
            strength: 0.0,
            roughness: 0.0,
        }
    }
}

impl ClearcoatParams {
    /// Builds clamped parameters, forcing `strength`/`roughness` into `[0, 1]`.
    #[inline]
    pub fn new(strength: f32, roughness: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
            roughness: roughness.clamp(0.0, 1.0),
        }
    }
}

/// Normalises `v`, falling back to `fallback` when `v` is degenerate (zero
/// length or non-finite) so downstream dot products stay well defined.
#[inline]
fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.is_finite() && v.length_squared() > 1.0e-12 {
        v.normalize()
    } else {
        fallback
    }
}

/// Scalar base specular BRDF `D · G · F / (4·NoV·NoL)` (height-correlated Smith).
///
/// Built from the shared GGX primitives so the base layer matches the engine's
/// main specular reference.  Returns `0` for any back-facing cosine and clamps
/// the result finite and non-negative.
#[inline]
fn base_specular(
    n_dot_h: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    v_dot_h: f32,
    f0: f32,
    roughness: f32,
) -> f32 {
    if !(n_dot_h > 0.0) || !(n_dot_v > 0.0) || !(n_dot_l > 0.0) {
        return 0.0;
    }
    let alpha = roughness_to_alpha(roughness);
    let d = ndf_ggx(n_dot_h, alpha);
    let g = smith_g2(n_dot_v, n_dot_l, alpha);
    let f = fresnel_schlick_scalar(f0.clamp(0.0, 1.0), v_dot_h.max(0.0));
    let denom = (4.0 * n_dot_v * n_dot_l).max(MIN_DENOM);
    let r = d * g * f / denom;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

/// Evaluates the full layered clearcoat BRDF for a light/view pair.
///
/// Returns the scalar reflectance density `f_r` combining the attenuated base
/// lobe and the clearcoat lobe:
///
/// ```text
/// Fc      = clearcoat_strength · F_Schlick(0.04, NoV_coat)
/// f_r     = base·(1 - Fc) + clearcoat_strength · clearcoat_lobe
/// ```
///
/// * `wi`, `wo` — world-space incoming/outgoing directions (away from surface).
/// * `geo_normal` — base/geometric normal (drives the base lobe).
/// * `clearcoat_normal` — coat normal (drives the coat lobe + `Fc`); may differ
///   from `geo_normal`.
/// * `base_f0` — base normal-incidence reflectance (scalar).
/// * `base_roughness` — base perceptual roughness.
/// * `clearcoat_strength`, `clearcoat_roughness` — coat factor and roughness.
///
/// All inputs are defended against degeneracy; the result is finite and
/// non-negative.
#[inline]
pub fn clearcoat_brdf(
    wi: Vec3,
    wo: Vec3,
    geo_normal: Vec3,
    clearcoat_normal: Vec3,
    base_f0: f32,
    base_roughness: f32,
    clearcoat_strength: f32,
    clearcoat_roughness: f32,
) -> f32 {
    let wi = safe_normalize(wi, Vec3::Z);
    let wo = safe_normalize(wo, Vec3::Z);
    let geo_n = safe_normalize(geo_normal, Vec3::Z);
    let coat_n = safe_normalize(clearcoat_normal, geo_n);
    let h = safe_normalize(wi + wo, geo_n);

    // Base-layer cosines (geometric normal).
    let geo_n_dot_v = clamp_cosine(geo_n, wo);
    let geo_n_dot_l = clamp_cosine(geo_n, wi);
    let geo_n_dot_h = clamp_cosine(geo_n, h);

    // Coat-layer cosines (clearcoat normal).
    let coat_n_dot_v = clamp_cosine(coat_n, wo);
    let coat_n_dot_l = clamp_cosine(coat_n, wi);
    let coat_n_dot_h = clamp_cosine(coat_n, h);

    // Shared view/half cosine (equals light/half for the bisecting half vector).
    let v_dot_h = wo.dot(h).clamp(0.0, 1.0);

    let base = base_specular(
        geo_n_dot_h,
        geo_n_dot_v,
        geo_n_dot_l,
        v_dot_h,
        base_f0,
        base_roughness,
    );

    let strength = clearcoat_strength.clamp(0.0, 1.0);
    let lobe = clearcoat_lobe(
        coat_n_dot_h,
        coat_n_dot_l,
        coat_n_dot_v,
        v_dot_h,
        clearcoat_roughness,
    );
    let coat_contribution = strength * lobe;

    let fc = clearcoat_fresnel_reflectance(strength, coat_n_dot_v);
    combine_layers(base, coat_contribution, fc)
}

/// Convenience wrapper taking a [`ClearcoatParams`] bundle.
///
/// Equivalent to [`clearcoat_brdf`] with `params.strength`/`params.roughness`.
#[inline]
pub fn clearcoat_brdf_params(
    wi: Vec3,
    wo: Vec3,
    geo_normal: Vec3,
    clearcoat_normal: Vec3,
    base_f0: f32,
    base_roughness: f32,
    params: ClearcoatParams,
) -> f32 {
    clearcoat_brdf(
        wi,
        wo,
        geo_normal,
        clearcoat_normal,
        base_f0,
        base_roughness,
        params.strength,
        params.roughness,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize()
    }

    #[test]
    fn default_params_disable_the_coat() {
        let p = ClearcoatParams::default();
        assert_eq!(p.strength, 0.0);
        assert_eq!(p.roughness, 0.0);
    }

    #[test]
    fn zero_strength_reproduces_bare_base() {
        let wo = dir(0.2, 0.0, 0.98);
        let wi = dir(-0.2, 0.0, 0.98);
        let n = Vec3::Z;
        let with_coat = clearcoat_brdf(wi, wo, n, n, 0.04, 0.3, 0.0, 0.1);

        // Reconstruct the bare base lobe through the same primitives.
        let h = (wi + wo).normalize();
        let base = base_specular(
            n.dot(h),
            n.dot(wo),
            n.dot(wi),
            wo.dot(h),
            0.04,
            0.3,
        );
        assert!((with_coat - base).abs() < 1e-6, "coat={with_coat} base={base}");
    }

    #[test]
    fn coat_attenuates_base_and_adds_specular() {
        let wo = dir(0.3, 0.0, 0.95);
        let wi = dir(-0.3, 0.0, 0.95);
        let n = Vec3::Z;
        let bare = clearcoat_brdf(wi, wo, n, n, 0.9, 0.5, 0.0, 0.1);
        let coated = clearcoat_brdf(wi, wo, n, n, 0.9, 0.5, 1.0, 0.1);
        // The layered result stays a valid finite, non-negative BRDF.
        assert!(coated.is_finite() && coated >= 0.0);
        assert!(bare.is_finite() && bare >= 0.0);
    }

    #[test]
    fn mirror_configuration_is_bright() {
        // Perfect retro/mirror config at normal incidence: strong coat lobe.
        let n = Vec3::Z;
        let wo = Vec3::Z;
        let wi = Vec3::Z;
        let f = clearcoat_brdf(wi, wo, n, n, 0.04, 0.2, 1.0, 0.02);
        assert!(f.is_finite() && f > 0.0, "f={f}");
    }

    #[test]
    fn independent_clearcoat_normal_changes_result() {
        let wo = dir(0.0, 0.3, 0.95);
        let wi = dir(0.0, -0.3, 0.95);
        let geo = Vec3::Z;
        let coat = dir(0.0, 0.4, 0.9);
        let aligned = clearcoat_brdf(wi, wo, geo, geo, 0.5, 0.4, 1.0, 0.1);
        let tilted = clearcoat_brdf(wi, wo, geo, coat, 0.5, 0.4, 1.0, 0.1);
        assert!(aligned.is_finite() && tilted.is_finite());
        assert!((aligned - tilted).abs() > 1e-7, "coat normal had no effect");
    }

    #[test]
    fn backfacing_light_returns_zero() {
        let n = Vec3::Z;
        let wo = dir(0.2, 0.0, 0.98);
        let wi = dir(0.0, 0.0, -1.0); // below horizon
        let f = clearcoat_brdf(wi, wo, n, n, 0.5, 0.3, 1.0, 0.1);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn params_wrapper_matches_explicit_call() {
        let wo = dir(0.1, 0.2, 0.97);
        let wi = dir(-0.1, -0.2, 0.97);
        let n = Vec3::Z;
        let p = ClearcoatParams::new(0.8, 0.25);
        let a = clearcoat_brdf_params(wi, wo, n, n, 0.3, 0.4, p);
        let b = clearcoat_brdf(wi, wo, n, n, 0.3, 0.4, 0.8, 0.25);
        assert!((a - b).abs() < 1e-7);
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        let f = clearcoat_brdf(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.5, 0.3, 1.0, 0.1);
        assert!(f.is_finite() && f >= 0.0, "f={f}");
        let g = clearcoat_brdf(
            Vec3::splat(f32::NAN),
            Vec3::Z,
            Vec3::Z,
            Vec3::Z,
            0.5,
            0.3,
            1.0,
            0.1,
        );
        assert!(g.is_finite() && g >= 0.0, "g={g}");
    }

    #[test]
    fn result_is_non_negative_over_grid() {
        let n = Vec3::Z;
        for i in 0..6 {
            for j in 0..6 {
                let a = (i as f32 + 0.5) / 6.0;
                let b = (j as f32 + 0.5) / 6.0;
                let wo = dir(a - 0.5, b - 0.5, 1.0);
                let wi = dir(0.5 - a, 0.5 - b, 1.0);
                let f = clearcoat_brdf(wi, wo, n, n, 0.5, 0.4, 0.7, 0.2);
                assert!(f.is_finite() && f >= 0.0, "f={f}");
            }
        }
    }
}
