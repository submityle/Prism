//! Clearcoat second-specular lobe — isotropic GGX at a fixed dielectric IOR.
//!
//! This module is the backend-neutral CPU golden reference for the *clearcoat*
//! specular lobe layered on top of a base material (automotive paint, lacquered
//! wood, carbon fibre, varnished surfaces).  A clearcoat is a thin, smooth-ish
//! dielectric film with a *fixed* index of refraction `IOR = 1.5`, so its
//! normal-incidence reflectance is the textbook dielectric value
//! `F0 = ((1 - 1.5) / (1 + 1.5))^2 = 0.04`.  Unlike the base layer the clearcoat
//! is always isotropic and never tinted, which is exactly the glTF
//! `KHR_materials_clearcoat` convention this reference targets.
//!
//! The lobe evaluates the classic Cook–Torrance microfacet form
//! `f_r = D · G · F / (4 · NoL · NoV)`, where:
//!
//! * `D` is the isotropic GGX / Trowbridge–Reitz normal distribution,
//! * `F` is the scalar Schlick Fresnel at the fixed `F0 = 0.04`, evaluated at
//!   the view/half-vector cosine `VoH`, and
//! * the masking–shadowing / visibility term is **Kelemen's approximation**
//!   `V = 1 / (4 · VoH²)` (see below).
//!
//! # Geometric term — Kelemen vs. Smith
//! Two geometric terms are provided and the caller must pick exactly one; the
//! high-level [`clearcoat_lobe`] uses **Kelemen** because it is the historical
//! Disney/Burley choice for clearcoat and matches the glTF sample viewer:
//!
//! * [`v_kelemen`] — the combined visibility `V = G / (4 · NoL · NoV)` is
//!   approximated directly as `1 / (4 · VoH²)`.  This folds the `1 / (4·NoL·NoV)`
//!   denominator of the Cook–Torrance form into the visibility, so the lobe is
//!   simply `D · V_Kelemen · F` with *no* extra division.  It is cheap, stable
//!   near grazing, and is what Disney 2012/2015 and Filament use for clearcoat.
//! * [`v_smith`] — a physically fuller height-correlated Smith visibility
//!   `V = G2 / (4 · NoL · NoV)` built from [`smith_g2`].  Offered for callers
//!   that want Smith parity with the base lobe; [`clearcoat_lobe_smith`] uses it.
//!
//! # Conventions
//! * All inputs are **cosines** already resolved against the *clearcoat* normal
//!   (which may differ from the base/geometric normal — see
//!   [`super::coupling`]).  `n_dot_h`, `n_dot_l`, `n_dot_v` are the half-, light-
//!   and view-vector cosines; `v_dot_h` is the view/half-vector cosine shared by
//!   the Fresnel and Kelemen terms.
//! * Only the upper hemisphere carries energy: any non-positive cosine returns
//!   `0` so a back-facing or grazing-past-horizon configuration is dark rather
//!   than negative or `NaN`.
//! * Perceptual `roughness ∈ [0, 1]` maps to the GGX width through the shared
//!   [`roughness_to_alpha`] (`alpha = roughness²`, floored), so the clearcoat and
//!   base layers agree on the roughness remap bit-for-bit.
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals or
//!   `unsafe`) that clamps denominators, floors negatives and verifies finiteness
//!   so the reference can never inject `NaN`/`inf` energy.
//!
//! # References
//! * Kelemen & Szirmay-Kalos 2001, *A Microfacet Based Coupled Specular-Matte
//!   BRDF Model* — the `1/(4·VoH²)` visibility approximation.
//! * Burley 2012/2015, *Physically Based Shading at Disney* — the clearcoat lobe
//!   with fixed `IOR = 1.5` and Kelemen visibility.
//! * Walter et al. 2007, *Microfacet Models for Refraction* — GGX `D`/Smith `G`.
//! * Khronos `KHR_materials_clearcoat` — the glTF clearcoat material convention.

use crate::gi::spec_gi::ggx_lobe::{fresnel_schlick_scalar, ndf_ggx, roughness_to_alpha, smith_g2};

/// Normal-incidence reflectance of a clearcoat dielectric with `IOR = 1.5`.
///
/// `F0 = ((1 - 1.5) / (1 + 1.5))^2 = 0.04`, the glTF/Disney clearcoat constant.
pub const CLEARCOAT_F0: f32 = 0.04;

/// Floor applied to the Kelemen `4·VoH²` denominator so a grazing half vector
/// (`VoH → 0`) yields a large-but-finite visibility instead of dividing by zero.
const MIN_DENOM: f32 = 1.0e-6;

/// Maps a perceptual clearcoat `roughness` to the GGX width `alpha`.
///
/// Thin wrapper over the shared [`roughness_to_alpha`] (`alpha = roughness²`,
/// clamped and floored) so the clearcoat layer re-uses the engine's single
/// roughness remap rather than duplicating it.
#[inline]
pub fn clearcoat_alpha(roughness: f32) -> f32 {
    roughness_to_alpha(roughness)
}

/// Scalar Schlick Fresnel of the clearcoat layer evaluated at `v_dot_h`.
///
/// Uses the fixed [`CLEARCOAT_F0`]; `v_dot_h` is clamped to `[0, 1]` so a
/// back-facing half vector returns the normal-incidence reflectance rather than
/// an out-of-range value.
#[inline]
pub fn clearcoat_fresnel(v_dot_h: f32) -> f32 {
    fresnel_schlick_scalar(CLEARCOAT_F0, v_dot_h.max(0.0))
}

/// Kelemen combined visibility `V = 1 / (4 · VoH²)`.
///
/// This approximates `G / (4 · NoL · NoV)` of the Cook–Torrance form, so the
/// full lobe is `D · V · F` with no further division.  `v_dot_h` is used by
/// magnitude; the denominator is floored at [`MIN_DENOM`] to stay finite near
/// grazing, and the result is guaranteed finite and non-negative.
#[inline]
pub fn v_kelemen(v_dot_h: f32) -> f32 {
    let voh = v_dot_h.abs();
    let denom = (4.0 * voh * voh).max(MIN_DENOM);
    let v = 1.0 / denom;
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Height-correlated Smith visibility `V = G2 / (4 · NoL · NoV)` for the
/// clearcoat, built from the shared [`smith_g2`].
///
/// The alternative to [`v_kelemen`]; returns `0` for any non-positive cosine and
/// floors the `4·NoL·NoV` denominator so the result stays finite.
#[inline]
pub fn v_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    if !(n_dot_v > 0.0) || !(n_dot_l > 0.0) {
        return 0.0;
    }
    let g2 = smith_g2(n_dot_v, n_dot_l, alpha);
    let denom = (4.0 * n_dot_v * n_dot_l).max(MIN_DENOM);
    let v = g2 / denom;
    if v.is_finite() { v.max(0.0) } else { 0.0 }
}

/// Evaluates the clearcoat specular lobe `D · V_Kelemen · F` (the default).
///
/// All cosines are taken against the *clearcoat* normal.  Returns `0` for a
/// back-facing light, view or half vector so no energy leaks below the horizon.
/// The result is clamped finite and non-negative.
///
/// * `n_dot_h` — clearcoat-normal · half-vector cosine (drives GGX `D`).
/// * `n_dot_l`, `n_dot_v` — light/view cosines, used only to gate the hemisphere
///   (Kelemen's visibility is expressed purely through `v_dot_h`).
/// * `v_dot_h` — view · half-vector cosine (drives Fresnel and Kelemen).
/// * `roughness` — perceptual clearcoat roughness, remapped via
///   [`clearcoat_alpha`].
#[inline]
pub fn clearcoat_lobe(
    n_dot_h: f32,
    n_dot_l: f32,
    n_dot_v: f32,
    v_dot_h: f32,
    roughness: f32,
) -> f32 {
    if !(n_dot_h > 0.0) || !(n_dot_l > 0.0) || !(n_dot_v > 0.0) {
        return 0.0;
    }
    let alpha = clearcoat_alpha(roughness);
    let d = ndf_ggx(n_dot_h, alpha);
    let v = v_kelemen(v_dot_h);
    let f = clearcoat_fresnel(v_dot_h);
    let r = d * v * f;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

/// Evaluates the clearcoat specular lobe `D · V_Smith · F` using the fuller
/// height-correlated Smith visibility instead of Kelemen's approximation.
///
/// Provided for callers that want Smith parity with the base lobe; behaves
/// identically to [`clearcoat_lobe`] at the hemisphere gate and clamping.
#[inline]
pub fn clearcoat_lobe_smith(
    n_dot_h: f32,
    n_dot_l: f32,
    n_dot_v: f32,
    v_dot_h: f32,
    roughness: f32,
) -> f32 {
    if !(n_dot_h > 0.0) || !(n_dot_l > 0.0) || !(n_dot_v > 0.0) {
        return 0.0;
    }
    let alpha = clearcoat_alpha(roughness);
    let d = ndf_ggx(n_dot_h, alpha);
    let v = v_smith(n_dot_v, n_dot_l, alpha);
    let f = clearcoat_fresnel(v_dot_h);
    let r = d * v * f;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

/// Cosine-weighted clearcoat lobe `clearcoat_lobe · NoL`.
///
/// Returns the clearcoat lobe multiplied by the (clamped) light cosine, i.e. the
/// quantity that is integrated against incident radiance when accumulating the
/// coat's contribution to outgoing radiance.  All gating/clamping of
/// [`clearcoat_lobe`] applies, and the extra `NoL` factor keeps the result
/// finite and non-negative.
#[inline]
pub fn clearcoat_lobe_weighted(
    n_dot_h: f32,
    n_dot_l: f32,
    n_dot_v: f32,
    v_dot_h: f32,
    roughness: f32,
) -> f32 {
    let lobe = clearcoat_lobe(n_dot_h, n_dot_l, n_dot_v, v_dot_h, roughness);
    let nl = n_dot_l.clamp(0.0, 1.0);
    let r = lobe * nl;
    if r.is_finite() { r.max(0.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearcoat_f0_is_the_dielectric_constant() {
        let ior = 1.5f32;
        let expected = ((1.0 - ior) / (1.0 + ior)).powi(2);
        assert!((CLEARCOAT_F0 - expected).abs() < 1e-4, "f0={expected}");
    }

    #[test]
    fn alpha_matches_shared_remap() {
        for r in [0.0f32, 0.25, 0.5, 1.0] {
            assert_eq!(clearcoat_alpha(r), roughness_to_alpha(r));
        }
    }

    #[test]
    fn fresnel_endpoints() {
        // Normal incidence returns F0.
        assert!((clearcoat_fresnel(1.0) - CLEARCOAT_F0).abs() < 1e-6);
        // Grazing tends to full reflection.
        assert!((clearcoat_fresnel(0.0) - 1.0).abs() < 1e-6);
        // Monotonic increase toward grazing.
        assert!(clearcoat_fresnel(0.2) > clearcoat_fresnel(0.9));
    }

    #[test]
    fn kelemen_is_finite_and_positive() {
        for voh in [0.0f32, 1.0e-4, 0.1, 0.5, 1.0] {
            let v = v_kelemen(voh);
            assert!(v.is_finite() && v >= 0.0, "voh={voh} v={v}");
        }
        // Decreasing visibility as VoH grows (less grazing).
        assert!(v_kelemen(0.2) > v_kelemen(0.9));
    }

    #[test]
    fn smith_visibility_is_bounded_and_gated() {
        let alpha = clearcoat_alpha(0.3);
        let v = v_smith(0.8, 0.6, alpha);
        assert!(v.is_finite() && v > 0.0, "v={v}");
        // Below-horizon cosines kill the term.
        assert_eq!(v_smith(-0.1, 0.5, alpha), 0.0);
        assert_eq!(v_smith(0.5, 0.0, alpha), 0.0);
    }

    #[test]
    fn lobe_is_non_negative_and_finite_over_grid() {
        for &nh in &[0.1f32, 0.5, 0.95, 1.0] {
            for &nl in &[0.2f32, 0.6, 1.0] {
                for &nv in &[0.2f32, 0.6, 1.0] {
                    for &vh in &[0.1f32, 0.5, 1.0] {
                        for &r in &[0.03f32, 0.3, 0.8] {
                            let f = clearcoat_lobe(nh, nl, nv, vh, r);
                            assert!(f.is_finite() && f >= 0.0, "f={f}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn lobe_gates_backfacing_configs() {
        assert_eq!(clearcoat_lobe(-0.1, 0.5, 0.5, 0.5, 0.1), 0.0);
        assert_eq!(clearcoat_lobe(0.5, -0.1, 0.5, 0.5, 0.1), 0.0);
        assert_eq!(clearcoat_lobe(0.5, 0.5, -0.1, 0.5, 0.1), 0.0);
    }

    #[test]
    fn smoother_coat_has_a_sharper_taller_peak() {
        // At the perfect-mirror configuration (NoH = 1) a smoother coat should
        // concentrate more energy, so its D (and thus the lobe) is larger.
        let smooth = clearcoat_lobe(1.0, 1.0, 1.0, 1.0, 0.05);
        let rough = clearcoat_lobe(1.0, 1.0, 1.0, 1.0, 0.6);
        assert!(smooth > rough, "smooth={smooth} rough={rough}");
    }

    #[test]
    fn kelemen_and_smith_variants_both_finite() {
        let k = clearcoat_lobe(0.9, 0.7, 0.6, 0.8, 0.2);
        let s = clearcoat_lobe_smith(0.9, 0.7, 0.6, 0.8, 0.2);
        assert!(k.is_finite() && k > 0.0, "k={k}");
        assert!(s.is_finite() && s > 0.0, "s={s}");
    }

    #[test]
    fn degenerate_inputs_never_nan() {
        assert_eq!(clearcoat_lobe(f32::NAN, 1.0, 1.0, 1.0, 0.1).max(0.0), 0.0);
        assert!(v_kelemen(f32::NAN).is_finite());
        assert!(v_smith(f32::INFINITY, 1.0, 0.3).is_finite());
    }

    #[test]
    fn weighted_lobe_is_lobe_times_nl() {
        let nh = 0.9f32;
        let nl = 0.7f32;
        let nv = 0.6f32;
        let vh = 0.8f32;
        let r = 0.2f32;
        let expected = clearcoat_lobe(nh, nl, nv, vh, r) * nl;
        assert!((clearcoat_lobe_weighted(nh, nl, nv, vh, r) - expected).abs() < 1e-7);
    }

    #[test]
    fn weighted_lobe_is_non_negative_and_gated() {
        assert_eq!(clearcoat_lobe_weighted(0.5, -0.1, 0.5, 0.5, 0.1), 0.0);
        let f = clearcoat_lobe_weighted(0.9, 0.8, 0.7, 0.9, 0.1);
        assert!(f.is_finite() && f >= 0.0, "f={f}");
    }
}
