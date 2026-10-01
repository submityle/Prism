//! Corneal focusing gain ("eye caustic") — CPU golden reference.
//!
//! The cornea is a strong converging lens: it bends incoming light toward the
//! optical axis and, together with the index step into the aqueous humour,
//! concentrates radiance onto the iris.  The bright, slightly warm pooling of
//! light seen on a lit iris — strongest near the axis and fading toward the rim
//! — is this focusing.  Rather than trace a full caustic, this module provides
//! an analytic *gain* multiplier for the iris radiance built from two
//! physically grounded pieces:
//!
//! * **The `n²` law of radiance.** Basic radiometry says `L / n²` is invariant
//!   along a ray, so radiance crossing into a medium of higher index is scaled
//!   by `(n_t / n_i)²`.  For air→aqueous that is roughly `1.34² ≈ 1.79×` — a
//!   genuine brightening, independent of geometry.
//! * **Beam compression.** A curved refracting surface changes the
//!   cross-section of a pencil of rays.  The projected-area ratio
//!   `cos θ_i / cos θ_t` captures first-order how the on-axis beam narrows (a
//!   converging interface) and is folded in as a geometric focusing term.
//!
//! The two combine into a peak gain that is then feathered by a radial falloff
//! so the effect concentrates near the iris centre and relaxes to unity at the
//! rim, and modulated by `strength` for art-directability.  Beyond the critical
//! angle no light is transmitted and the gain is zero.
//!
//! # Conventions
//! * `cos_i` is the clamped incidence cosine at the cornea; `eta = n_i / n_t`
//!   is the relative index (air→aqueous `< 1`).  `radius_norm ∈ [0, 1]` is the
//!   distance from the iris centre (`0` = axis, `1` = rim).
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals
//!   or `unsafe`).  Results are clamped non-negative and finite; a total
//!   internal reflection configuration yields `0`, never `NaN`.
//! * `f32` arithmetic mirrors the WESL/GPU twin.  The radial falloff uses
//!   [`bevy_math::ops::exp`]; `sqrt` is inherent.
//!
//! # References
//! * Preetham/Pharr & Humphreys, *Physically Based Rendering* — the `n²`
//!   radiance scaling across a refractive interface.

use bevy_math::ops;

/// Smallest magnitude allowed for a denominator before flooring.
const MIN_DENOM: f32 = 1.0e-5;

/// Radiance gain from the `n²` law for a ray crossing an interface of relative
/// index `eta = n_i / n_t`.
///
/// Returns `(n_t / n_i)² = 1 / eta²`.  Entering a denser medium (`eta < 1`)
/// gives a gain `> 1`.  `eta` is floored at [`MIN_DENOM`] and the result is
/// clamped finite and non-negative.
#[inline]
pub fn radiance_n2_gain(eta: f32) -> f32 {
    let inv = 1.0 / eta.max(MIN_DENOM);
    let g = inv * inv;
    if g.is_finite() { g.max(0.0) } else { 0.0 }
}

/// First-order beam-compression factor `cos θ_i / cos θ_t`.
///
/// Describes how a pencil of rays narrows or widens across the interface.
/// Both cosines are clamped to `[0, 1]`; the transmitted cosine is floored so
/// the ratio stays finite.
#[inline]
pub fn beam_compression(cos_i: f32, cos_t: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let cos_t = cos_t.clamp(0.0, 1.0).max(MIN_DENOM);
    (cos_i / cos_t).max(0.0)
}

/// Transmitted cosine from Snell's law, or [`None`] under total internal
/// reflection.
///
/// `cos_i` is clamped to `[0, 1]`; `eta = n_i / n_t`.  Returns `cos θ_t` when
/// the ray transmits.
#[inline]
pub fn transmitted_cos(cos_i: f32, eta: f32) -> Option<f32> {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let eta = eta.max(MIN_DENOM);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        None
    } else {
        Some((1.0 - sin2_t).max(0.0).sqrt())
    }
}

/// Tunable parameters for the corneal focusing gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausticParams {
    /// Relative index `eta = n_incident / n_transmitted` of the cornea.
    pub eta: f32,
    /// Art-directable scale on the excess gain above unity, in `[0, ∞)`.
    pub strength: f32,
    /// Radial falloff rate; larger values confine the gain nearer the axis.
    pub falloff: f32,
}

impl Default for CausticParams {
    #[inline]
    fn default() -> Self {
        Self {
            // Air -> aqueous humour: the ray's effective destination medium.
            eta: super::cornea::IOR_AIR / super::cornea::IOR_AQUEOUS,
            strength: 1.0,
            falloff: 3.0,
        }
    }
}

impl CausticParams {
    /// Builds parameters, flooring `eta` and clamping `strength`/`falloff`
    /// non-negative so the gain can never diverge.
    #[inline]
    pub fn new(eta: f32, strength: f32, falloff: f32) -> Self {
        Self {
            eta: eta.max(MIN_DENOM),
            strength: strength.max(0.0),
            falloff: falloff.max(0.0),
        }
    }
}

/// Peak on-axis focusing gain for an incidence cosine, before radial feathering.
///
/// Combines the `n²` radiance law with the beam-compression factor:
/// `peak = (1 / eta²) · (cos θ_i / cos θ_t)`.  Returns `0` under total internal
/// reflection (no transmitted light to focus).
#[inline]
pub fn peak_focus_gain(cos_i: f32, eta: f32) -> f32 {
    match transmitted_cos(cos_i, eta) {
        Some(cos_t) => (radiance_n2_gain(eta) * beam_compression(cos_i, cos_t)).max(0.0),
        None => 0.0,
    }
}

/// Smooth radial feather in `[0, 1]`: `exp(-falloff · r²)`.
///
/// `1` on the optical axis, decaying toward the rim.  `radius_norm` is clamped
/// to `[0, 1]` and the exponent is bounded so the call stays finite.
#[inline]
pub fn radial_falloff(radius_norm: f32, falloff: f32) -> f32 {
    let r = radius_norm.clamp(0.0, 1.0);
    let falloff = falloff.max(0.0);
    let x = (-falloff * r * r).clamp(-80.0, 0.0);
    ops::exp(x).clamp(0.0, 1.0)
}

/// Full corneal focusing gain multiplier for the iris radiance.
///
/// Builds the peak gain from [`peak_focus_gain`], feathers it radially with
/// [`radial_falloff`] and blends from unity by `strength`:
///
/// ```text
/// gain = 1 + strength · (peak - 1)₊ · falloff(r)
/// ```
///
/// so the axis of a head-on eye is brightened by up to the full `n²`/beam gain
/// while the rim relaxes to `1`.  Under total internal reflection (`peak == 0`)
/// the excess term is clamped away and the gain is unity rather than darkening.
/// The result is always finite and non-negative.
#[inline]
pub fn corneal_focus_gain(cos_i: f32, radius_norm: f32, params: CausticParams) -> f32 {
    let peak = peak_focus_gain(cos_i, params.eta);
    // Only brightening is physical here; never let the term darken the iris.
    let excess = (peak - 1.0).max(0.0);
    let feather = radial_falloff(radius_norm, params.falloff);
    let gain = 1.0 + params.strength.max(0.0) * excess * feather;
    if gain.is_finite() { gain.max(0.0) } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn n2_gain_brightens_into_denser_medium() {
        // Air -> aqueous: eta < 1, so gain > 1.
        let g = radiance_n2_gain(super::super::cornea::IOR_AIR / super::super::cornea::IOR_AQUEOUS);
        assert!(g > 1.0, "g={g}");
        // Exactly (n_t/n_i)^2 = 1.336^2.
        let expect = super::super::cornea::IOR_AQUEOUS * super::super::cornea::IOR_AQUEOUS;
        assert!((g - expect).abs() < 1e-4, "g={g} expect={expect}");
    }

    #[test]
    fn n2_gain_is_unity_for_matched_media() {
        assert!((radiance_n2_gain(1.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn n2_gain_handles_degenerate_eta() {
        let g = radiance_n2_gain(0.0);
        assert!(g.is_finite());
    }

    #[test]
    fn beam_compression_is_unity_at_normal_incidence() {
        assert!((beam_compression(1.0, 1.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn beam_compression_finite_at_grazing() {
        let b = beam_compression(0.0, 0.0);
        assert!(b.is_finite(), "b={b}");
    }

    #[test]
    fn transmitted_cos_matches_snell() {
        // Normal incidence transmits straight: cos_t == 1.
        assert!((transmitted_cos(1.0, 0.75).unwrap() - 1.0).abs() < EPS);
    }

    #[test]
    fn transmitted_cos_none_under_tir() {
        // Dense -> rare beyond critical: eta > 1 and small cos_i.
        assert!(transmitted_cos(0.1, 1.5).is_none());
    }

    #[test]
    fn peak_gain_is_zero_under_tir() {
        assert_eq!(peak_focus_gain(0.1, 1.5), 0.0);
    }

    #[test]
    fn peak_gain_positive_on_axis() {
        let eta = super::super::cornea::IOR_AIR / super::super::cornea::IOR_AQUEOUS;
        let p = peak_focus_gain(1.0, eta);
        assert!(p > 1.0, "p={p}");
    }

    #[test]
    fn radial_falloff_peaks_on_axis() {
        assert!((radial_falloff(0.0, 3.0) - 1.0).abs() < EPS);
        assert!(radial_falloff(1.0, 3.0) < radial_falloff(0.5, 3.0));
        assert!(radial_falloff(0.5, 3.0) < 1.0);
    }

    #[test]
    fn radial_falloff_bounded() {
        for i in 0..=10 {
            let r = i as f32 / 10.0;
            let f = radial_falloff(r, 8.0);
            assert!((0.0..=1.0).contains(&f), "f={f}");
        }
    }

    #[test]
    fn focus_gain_center_brighter_than_rim() {
        let p = CausticParams::default();
        let center = corneal_focus_gain(1.0, 0.0, p);
        let rim = corneal_focus_gain(1.0, 1.0, p);
        assert!(center > rim, "center={center} rim={rim}");
        assert!(center > 1.0, "center={center}");
    }

    #[test]
    fn focus_gain_never_darkens() {
        let p = CausticParams::default();
        for i in 0..=10 {
            let r = i as f32 / 10.0;
            for j in 0..=10 {
                let cos_i = j as f32 / 10.0;
                let g = corneal_focus_gain(cos_i, r, p);
                assert!(g >= 1.0 - 1e-4, "g={g}");
                assert!(g.is_finite());
            }
        }
    }

    #[test]
    fn focus_gain_zero_strength_is_unity() {
        let p = CausticParams::new(0.75, 0.0, 3.0);
        assert!((corneal_focus_gain(1.0, 0.0, p) - 1.0).abs() < EPS);
    }

    #[test]
    fn focus_gain_under_tir_is_unity_not_nan() {
        // eta > 1 and grazing => TIR => peak 0 => gain clamps to 1.
        let p = CausticParams::new(1.5, 2.0, 3.0);
        let g = corneal_focus_gain(0.1, 0.0, p);
        assert!((g - 1.0).abs() < EPS, "g={g}");
    }

    #[test]
    fn strength_scales_excess() {
        let base = CausticParams::new(0.75, 1.0, 2.0);
        let strong = CausticParams::new(0.75, 2.0, 2.0);
        let gb = corneal_focus_gain(1.0, 0.0, base) - 1.0;
        let gs = corneal_focus_gain(1.0, 0.0, strong) - 1.0;
        assert!((gs - 2.0 * gb).abs() < 1e-4, "gb={gb} gs={gs}");
    }
}
