//! Toksvig normal-map specular anti-aliasing — CPU golden reference.
//!
//! Normal-map minification averages many sub-texel normals into one filtered
//! normal.  The *length* of that averaged normal, `len = |avg_normal| ∈ [0, 1]`,
//! shrinks as the sub-texel normals disagree: a flat footprint keeps `len ≈ 1`,
//! while a footprint spanning a noisy, high-frequency bump field collapses
//! toward `len → 0`.  Toksvig 2005 observed that this shrinkage encodes the
//! sub-texel normal variance and can be folded back into the specular lobe so
//! that minified surfaces widen their highlight instead of flickering.
//!
//! The core relation maps a Blinn–Phong specular power (shininess) `s` to an
//! *effective* shininess through the Toksvig factor
//!
//! ```text
//! ft = len / (len + s · (1 - len)),      s_eff = ft · s.
//! ```
//!
//! Because this engine parameterises glossiness with a perceptual
//! `roughness ∈ [0, 1]` (GGX width `alpha = roughness²`), the module also
//! exposes the shininess ↔ `alpha` bridge `s = 2/alpha² − 2` (Beckmann slope
//! variance) and the equivalent *additive-variance* form.  Rewriting the
//! Toksvig factor through `alpha² ≈ 2/s` yields the closed form
//!
//! ```text
//! alpha_eff² = alpha² + 2·(1 − len)/len,
//! ```
//!
//! i.e. the averaged-normal shrinkage contributes an extra lobe variance
//! `≈ 2·(1 − len)/len` that is added to the base GGX variance — the same
//! additive-variance philosophy used by the derivative-based filters in
//! [`crate::gi::specular_aa::normal_variance`].  The two forms coincide in the
//! high-gloss limit (`s ≫ 1`, where `alpha² ≈ 2/s`) and the additive form
//! composes trivially with other variance sources; the exact factor form is
//! used by [`toksvig_roughness`] so faithful Toksvig output is always
//! available.
//!
//! # Conventions
//! * `no_std`: this file allocates nothing and imports no containers; every
//!   routine works on scalars.  Transcendental functions are not needed, so
//!   [`bevy_math::ops`] is not imported; only the inherent `f32::sqrt`,
//!   `max`/`min`/`clamp` appear.
//! * Perceptual `roughness ∈ [0, 1]` maps to the GGX width `alpha = roughness²`
//!   via [`roughness_to_alpha`]; this module never re-implements that mapping.
//! * The averaged-normal length `len` is clamped to `[MIN_LEN, 1]` so a fully
//!   decorrelated footprint (`len → 0`) produces a large-but-finite extra
//!   variance instead of a division by zero, and every result is finite,
//!   non-negative and clamped to a physical range (never `NaN`/`inf`).
//! * "Combining" roughness takes the *coarser* (larger) value so specular AA
//!   can only ever soften, never sharpen, the base material.
//!
//! # References
//! * Michael Toksvig 2005, *Mipmapping Normal Maps*, NVIDIA / Journal of
//!   Graphics Tools — the averaged-normal-length → effective-shininess factor.
//! * Han et al. 2007, *Frequency Domain Normal Map Filtering* — the variance
//!   interpretation that justifies the additive-`alpha²` closed form.

use crate::gi::spec_gi::ggx_lobe::{MIN_ALPHA, roughness_to_alpha};

/// Smallest averaged-normal length considered.  `len → 0` means a fully
/// decorrelated sub-texel normal field; flooring keeps `(1 − len)/len` finite.
pub const MIN_LEN: f32 = 1.0e-4;

/// Largest effective shininess returned by [`effective_shininess`].  A tiny
/// averaged-normal shrinkage on a near-mirror surface would otherwise produce
/// an astronomically large power; the clamp keeps the reference numerically
/// stable without visibly altering the highlight.
pub const MAX_SHININESS: f32 = 1.0e6;

/// Sanitises an averaged-normal length to `[MIN_LEN, 1]`.
///
/// Non-finite inputs collapse to [`MIN_LEN`] (treat garbage as maximally noisy).
#[inline]
fn sanitize_len(avg_normal_len: f32) -> f32 {
    if avg_normal_len.is_finite() {
        avg_normal_len.clamp(MIN_LEN, 1.0)
    } else {
        MIN_LEN
    }
}

/// Sanitises a Blinn–Phong shininess to `[0, MAX_SHININESS]`.
#[inline]
fn sanitize_shininess(shininess: f32) -> f32 {
    if shininess.is_finite() {
        shininess.clamp(0.0, MAX_SHININESS)
    } else {
        0.0
    }
}

/// Toksvig anti-aliasing factor `ft = len / (len + s·(1 − len))`.
///
/// `avg_normal_len` is the length of the mip-averaged normal and `shininess`
/// the Blinn–Phong specular power `s`.  The result lies in `(0, 1]`: it is `1`
/// for a flat footprint (`len = 1`) and shrinks toward `0` as the footprint
/// decorrelates, so `s_eff = ft · s` lowers the effective gloss.
#[inline]
pub fn toksvig_factor(avg_normal_len: f32, shininess: f32) -> f32 {
    let len = sanitize_len(avg_normal_len);
    let s = sanitize_shininess(shininess);
    let denom = (len + s * (1.0 - len)).max(MIN_LEN);
    let ft = len / denom;
    if ft.is_finite() { ft.clamp(0.0, 1.0) } else { 1.0 }
}

/// Effective (reduced) Blinn–Phong shininess `s_eff = ft · s` after Toksvig
/// minification filtering.
#[inline]
pub fn effective_shininess(base_shininess: f32, avg_normal_len: f32) -> f32 {
    let s = sanitize_shininess(base_shininess);
    let ft = toksvig_factor(avg_normal_len, s);
    (ft * s).clamp(0.0, MAX_SHININESS)
}

/// Converts a GGX width `alpha` to the equivalent Blinn–Phong shininess
/// `s = 2/alpha² − 2`.
///
/// This is the Beckmann slope-variance bridge (`alpha² ≈ 2/s`) used to move the
/// Toksvig factor between the shininess and roughness parameterisations.
/// `alpha` is floored at [`MIN_ALPHA`] and the result is non-negative.
#[inline]
pub fn shininess_from_alpha(alpha: f32) -> f32 {
    let a = alpha.max(MIN_ALPHA);
    let s = 2.0 / (a * a) - 2.0;
    if s.is_finite() { s.clamp(0.0, MAX_SHININESS) } else { MAX_SHININESS }
}

/// Inverse of [`shininess_from_alpha`]: `alpha = sqrt(2/(s + 2))`, floored at
/// [`MIN_ALPHA`].
#[inline]
pub fn alpha_from_shininess(shininess: f32) -> f32 {
    let s = sanitize_shininess(shininess);
    let a = (2.0 / (s + 2.0)).max(0.0).sqrt();
    a.max(MIN_ALPHA)
}

/// Additional GGX lobe variance `Δα² = 2·(1 − len)/len` contributed by the
/// averaged-normal shrinkage.
///
/// This is the high-gloss variance-domain approximation of the Toksvig factor
/// (`alpha² ≈ 2/s`) and is the quantity to *add* to a base `alpha²` when
/// composing with other variance sources.  A flat footprint (`len = 1`) adds
/// nothing; a fully decorrelated footprint adds a large-but-finite amount.
#[inline]
pub fn toksvig_delta_alpha_sq(avg_normal_len: f32) -> f32 {
    let len = sanitize_len(avg_normal_len);
    let delta = 2.0 * (1.0 - len) / len;
    if delta.is_finite() { delta.max(0.0) } else { 0.0 }
}

/// Combines a base GGX variance `alpha²` with the (high-gloss approximate)
/// Toksvig extra variance from `avg_normal_len`, returning the filtered
/// `alpha_eff²`.
///
/// `alpha_eff² = alpha² + 2·(1 − len)/len`, clamped to `[MIN_ALPHA², 1]` so the
/// result stays a valid, finite GGX width that is never sharper than the base.
/// For a faithful, non-approximate result prefer [`toksvig_roughness`].
#[inline]
pub fn effective_alpha_sq_from_len(base_alpha: f32, avg_normal_len: f32) -> f32 {
    let a = base_alpha.max(MIN_ALPHA);
    let base_sq = a * a;
    let eff = base_sq + toksvig_delta_alpha_sq(avg_normal_len);
    let floor = MIN_ALPHA * MIN_ALPHA;
    if eff.is_finite() { eff.clamp(floor, 1.0) } else { base_sq }
}

/// High-level helper: filters a perceptual `base_roughness` with the exact
/// Toksvig factor derived from `avg_normal_len`, returning the anti-aliased
/// perceptual roughness in `[0, 1]`.
///
/// Pipeline (exact, through the shininess domain):
/// `roughness → alpha → s → s' = ft·s → alpha' → roughness'`.  Because the
/// Toksvig factor `ft ∈ (0, 1]`, the effective shininess only decreases, so the
/// output is always `≥` the input (specular AA only softens).
#[inline]
pub fn toksvig_roughness(base_roughness: f32, avg_normal_len: f32) -> f32 {
    let alpha = roughness_to_alpha(base_roughness);
    let shininess = shininess_from_alpha(alpha);
    let eff_shininess = effective_shininess(shininess, avg_normal_len);
    let eff_alpha = alpha_from_shininess(eff_shininess);
    // roughness = sqrt(alpha), since alpha = roughness².
    let roughness = eff_alpha.max(0.0).sqrt();
    roughness.clamp(0.0, 1.0)
}

/// Combines two perceptual roughness values by taking the coarser (larger) one.
///
/// Used to merge a specular-AA roughness with the artist's base roughness so
/// filtering can only widen the lobe, never tighten it.  Non-finite inputs are
/// treated as `0` (fully smooth) before the comparison.
#[inline]
pub fn combine_roughness(a: f32, b: f32) -> f32 {
    let ca = if a.is_finite() { a.clamp(0.0, 1.0) } else { 0.0 };
    let cb = if b.is_finite() { b.clamp(0.0, 1.0) } else { 0.0 };
    ca.max(cb)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn flat_footprint_is_identity() {
        // len = 1 ⇒ no shrinkage, factor 1, zero extra variance.
        assert!((toksvig_factor(1.0, 100.0) - 1.0).abs() < EPS);
        assert!(toksvig_delta_alpha_sq(1.0).abs() < EPS);
        let r = toksvig_roughness(0.3, 1.0);
        assert!((r - 0.3).abs() < 1.0e-4, "r = {r}");
    }

    #[test]
    fn shrinking_length_lowers_shininess() {
        let s = 200.0;
        let full = effective_shininess(s, 1.0);
        let noisy = effective_shininess(s, 0.6);
        assert!((full - s).abs() < 1.0e-2);
        assert!(noisy < full, "noisy {noisy} should be < full {full}");
        assert!(noisy > 0.0);
    }

    #[test]
    fn delta_variance_monotonic_in_noise() {
        let a = toksvig_delta_alpha_sq(0.9);
        let b = toksvig_delta_alpha_sq(0.6);
        let c = toksvig_delta_alpha_sq(0.2);
        assert!(a < b && b < c, "{a} {b} {c}");
        assert!(a >= 0.0);
    }

    #[test]
    fn roughness_only_increases() {
        for &r in &[0.0_f32, 0.05, 0.2, 0.5, 0.9, 1.0] {
            let out = toksvig_roughness(r, 0.5);
            assert!(out >= r - 1.0e-4, "r {r} out {out}");
            assert!((0.0..=1.0).contains(&out));
        }
    }

    #[test]
    fn shininess_alpha_roundtrip() {
        for &alpha in &[0.05_f32, 0.1, 0.3, 0.6, 0.9] {
            let s = shininess_from_alpha(alpha);
            let back = alpha_from_shininess(s);
            assert!((alpha - back).abs() < 1.0e-3, "alpha {alpha} back {back}");
        }
    }

    #[test]
    fn additive_approximates_factor_in_high_gloss() {
        // In the high-gloss limit (s ≫ 1 ⇒ alpha² ≈ 2/s) the additive-variance
        // closed form tracks the exact shininess-domain Toksvig factor; the gap
        // only widens once the surface is pushed very rough.
        let base_roughness = 0.05_f32; // very glossy ⇒ large shininess
        let len = 0.985_f32; // mild shrinkage keeps s' large too
        let alpha = roughness_to_alpha(base_roughness);

        // Exact shininess-domain reference.
        let s = shininess_from_alpha(alpha);
        let s_eff = effective_shininess(s, len);
        let alpha_factor = alpha_from_shininess(s_eff);

        // Additive-variance approximation.
        let alpha_add = effective_alpha_sq_from_len(alpha, len).sqrt();

        assert!(
            (alpha_factor - alpha_add).abs() < 2.0e-2,
            "factor {alpha_factor} vs additive {alpha_add}"
        );
    }

    #[test]
    fn exact_and_additive_both_coarsen_consistently() {
        // Both pipelines must only ever increase roughness, and the exact form
        // must stay no coarser than the (over-estimating) additive form.
        let base = 0.2_f32;
        let len = 0.6_f32;
        let exact = toksvig_roughness(base, len);
        let alpha = roughness_to_alpha(base);
        let additive = effective_alpha_sq_from_len(alpha, len).sqrt().sqrt();
        assert!(exact >= base - 1.0e-4, "exact {exact} < base {base}");
        assert!(additive >= base - 1.0e-4, "additive {additive} < base {base}");
        assert!(additive >= exact - 1.0e-4, "additive {additive} < exact {exact}");
    }

    #[test]
    fn combine_takes_coarser() {
        assert!((combine_roughness(0.2, 0.5) - 0.5).abs() < EPS);
        assert!((combine_roughness(0.8, 0.1) - 0.8).abs() < EPS);
    }

    #[test]
    fn defends_against_garbage() {
        assert!(toksvig_factor(f32::NAN, 10.0).is_finite());
        assert!(toksvig_delta_alpha_sq(f32::INFINITY).is_finite());
        assert!(toksvig_roughness(f32::NAN, f32::NAN).is_finite());
        assert!(shininess_from_alpha(0.0).is_finite());
        assert!(alpha_from_shininess(f32::NAN) >= MIN_ALPHA);
        let out = toksvig_roughness(2.0, -5.0);
        assert!((0.0..=1.0).contains(&out));
    }

    #[test]
    fn zero_length_is_finite_and_coarse() {
        let d = toksvig_delta_alpha_sq(0.0);
        assert!(d.is_finite() && d > 0.0);
        let r = toksvig_roughness(0.1, 0.0);
        assert!(r.is_finite() && r > 0.1);
    }
}
