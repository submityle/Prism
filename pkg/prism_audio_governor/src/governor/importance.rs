//! Per-voice effective importance: the scalar the governor ranks voices by.
//!
//! When the budget forces culling, the governor must decide *which* voices to
//! demote. Design section 32 defines a voice's effective importance as the
//! product of four factors:
//!
//! ```text
//! importance = explicit_priority * distance_attenuation
//!            * perceptual_loudness * masking_factor
//! ```
//!
//! - **Explicit priority** is the designer's hint (a bus/category weight).
//! - **Distance attenuation** falls off with range, so far sources matter less.
//! - **Perceptual loudness** is a linear loudness estimate (design section 13's
//!   HDR window shares this measurement).
//! - **Masking factor** is `1.0` for an audible voice and a small penalty when
//!   the voice is masked by a louder neighbour (design section 33).
//!
//! The product is the single [`Importance`] scalar the voice pool compares
//! (design section 25), so lower layers never need to know how it was derived.
//!
//! # Determinism
//!
//! Distance attenuation uses [`bevy_math::ops`] for its one transcendental
//! (a power), so the result is bit-reproducible. Everything else is
//! multiplication and clamping.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Produces the [`Importance`] that [`crate::governor::lod::LodProfile`]'s
//! virtualisation threshold is compared against, and that the voice pool in
//! `prism_audio_core` ranks. The `masked` input is supplied by
//! [`crate::masking`].

use bevy_math::ops;
use prism_audio_core::math::Sample;
use prism_audio_core::voice::Importance;

/// Default multiplicative penalty applied to a masked voice's importance.
///
/// A masked voice is not silenced outright -- it is heavily deprioritised so it
/// is culled before any audible voice of similar raw loudness. The value is a
/// perceptual choice, not a physical constant.
pub const DEFAULT_MASKING_PENALTY: Sample = 0.15;

/// The four perceptual factors that combine into a voice's effective
/// importance.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ImportanceInputs {
    /// Designer-assigned priority weight (`>= 0`). Typically `1.0` for a normal
    /// voice; higher keeps a voice audible, `0.0` makes it maximally cullable.
    pub explicit_priority: Sample,
    /// Distance attenuation in `[0, 1]`: `1.0` at the listener, approaching
    /// `0.0` far away. Use [`inverse_distance_attenuation`] to derive it.
    pub distance_attenuation: Sample,
    /// Perceptual loudness estimate in `[0, 1]` (linear). Shares the HDR
    /// loudness measurement from design section 13.
    pub perceptual_loudness: Sample,
    /// Whether a louder neighbour masks this voice in its critical band.
    pub masked: bool,
}

impl ImportanceInputs {
    /// Creates inputs for an unmasked voice with explicit priority `1.0`.
    #[must_use]
    pub fn audible(distance_attenuation: Sample, perceptual_loudness: Sample) -> Self {
        Self {
            explicit_priority: 1.0,
            distance_attenuation,
            perceptual_loudness,
            masked: false,
        }
    }
}

/// Computes an effective importance from its four factors using
/// [`DEFAULT_MASKING_PENALTY`] for masked voices.
///
/// See [`effective_importance_with_penalty`] for control over the masking
/// penalty (the governor tightens it as budget shrinks).
#[must_use]
#[inline]
pub fn effective_importance(inputs: ImportanceInputs) -> Importance {
    effective_importance_with_penalty(inputs, DEFAULT_MASKING_PENALTY)
}

/// Computes an effective importance with an explicit masking penalty.
///
/// All factors are sanitised: non-finite inputs become `0.0`, priority and
/// loudness are floored at `0.0`, and the two normalised factors are clamped to
/// `[0, 1]`. `masking_penalty` is clamped to `[0, 1]`. The result is therefore
/// always a finite, non-negative scalar.
#[must_use]
pub fn effective_importance_with_penalty(
    inputs: ImportanceInputs,
    masking_penalty: Sample,
) -> Importance {
    let priority = sanitize_nonneg(inputs.explicit_priority);
    let distance = sanitize_unit(inputs.distance_attenuation);
    let loudness = sanitize_unit(inputs.perceptual_loudness);
    let penalty = sanitize_unit(masking_penalty);
    let mask = if inputs.masked { penalty } else { 1.0 };
    priority * distance * loudness * mask
}

/// Computes a distance attenuation factor in `[0, 1]` from a listener-relative
/// distance using an inverse-distance (rolloff) law.
///
/// Within `reference_distance` the factor is `1.0`; beyond it the factor is
/// `(reference / distance)^rolloff`, clamped to `[0, 1]`. A `rolloff` of `1.0`
/// is the physically motivated inverse law; larger exponents fall off faster.
///
/// Degenerate inputs degrade gracefully: a non-positive or non-finite
/// `reference_distance` yields `1.0` (no attenuation), and a non-finite
/// `distance` yields `0.0`.
///
/// # Examples
///
/// ```
/// # use prism_audio_governor::governor::importance::inverse_distance_attenuation;
/// // At the reference distance, no attenuation.
/// assert!((inverse_distance_attenuation(1.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
/// // Twice the reference distance with inverse law -> half.
/// assert!((inverse_distance_attenuation(2.0, 1.0, 1.0) - 0.5).abs() < 1e-6);
/// ```
#[must_use]
pub fn inverse_distance_attenuation(
    distance: Sample,
    reference_distance: Sample,
    rolloff: Sample,
) -> Sample {
    if !reference_distance.is_finite() || reference_distance <= 0.0 {
        return 1.0;
    }
    if !distance.is_finite() {
        return 0.0;
    }
    if distance <= reference_distance {
        return 1.0;
    }
    let ratio = reference_distance / distance;
    let exponent = if rolloff.is_finite() {
        rolloff.max(0.0)
    } else {
        1.0
    };
    ops::powf(ratio, exponent).clamp(0.0, 1.0)
}

/// Clamps a value into `[0, 1]`, mapping non-finite input to `0.0`.
#[inline]
fn sanitize_unit(x: Sample) -> Sample {
    if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }
}

/// Floors a value at `0.0`, mapping non-finite input to `0.0`.
#[inline]
fn sanitize_nonneg(x: Sample) -> Sample {
    if x.is_finite() { x.max(0.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn product_of_factors() {
        let inputs = ImportanceInputs {
            explicit_priority: 2.0,
            distance_attenuation: 0.5,
            perceptual_loudness: 0.4,
            masked: false,
        };
        // 2.0 * 0.5 * 0.4 * 1.0 = 0.4
        assert!((effective_importance(inputs) - 0.4).abs() < EPS);
    }

    #[test]
    fn masking_reduces_importance() {
        let base = ImportanceInputs::audible(1.0, 0.8);
        let masked = ImportanceInputs { masked: true, ..base };
        let i_base = effective_importance(base);
        let i_masked = effective_importance(masked);
        assert!(i_masked < i_base);
        assert!((i_masked - i_base * DEFAULT_MASKING_PENALTY).abs() < EPS);
    }

    #[test]
    fn tighter_penalty_demotes_further() {
        let masked = ImportanceInputs { masked: true, ..ImportanceInputs::audible(1.0, 0.8) };
        let loose = effective_importance_with_penalty(masked, 0.5);
        let tight = effective_importance_with_penalty(masked, 0.05);
        assert!(tight < loose);
    }

    #[test]
    fn inputs_are_sanitized() {
        let bad = ImportanceInputs {
            explicit_priority: Sample::NAN,
            distance_attenuation: 2.0,
            perceptual_loudness: -1.0,
            masked: false,
        };
        let i = effective_importance(bad);
        assert!(i.is_finite());
        assert!(i.abs() < EPS); // NaN priority -> 0, loudness floored -> 0
    }

    #[test]
    fn distance_within_reference_is_unity() {
        assert!((inverse_distance_attenuation(0.5, 1.0, 1.0) - 1.0).abs() < EPS);
        assert!((inverse_distance_attenuation(1.0, 1.0, 1.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn distance_inverse_law() {
        assert!((inverse_distance_attenuation(2.0, 1.0, 1.0) - 0.5).abs() < EPS);
        assert!((inverse_distance_attenuation(4.0, 1.0, 1.0) - 0.25).abs() < EPS);
    }

    #[test]
    fn steeper_rolloff_falls_faster() {
        let gentle = inverse_distance_attenuation(2.0, 1.0, 1.0);
        let steep = inverse_distance_attenuation(2.0, 1.0, 2.0);
        assert!(steep < gentle);
    }

    #[test]
    fn distance_degenerate_inputs() {
        assert!((inverse_distance_attenuation(5.0, 0.0, 1.0) - 1.0).abs() < EPS);
        assert!((inverse_distance_attenuation(5.0, -1.0, 1.0) - 1.0).abs() < EPS);
        assert!(inverse_distance_attenuation(Sample::INFINITY, 1.0, 1.0).abs() < EPS);
    }

    #[test]
    fn farther_voice_is_less_important() {
        let near = ImportanceInputs::audible(inverse_distance_attenuation(1.0, 1.0, 1.0), 0.6);
        let far = ImportanceInputs::audible(inverse_distance_attenuation(8.0, 1.0, 1.0), 0.6);
        assert!(effective_importance(far) < effective_importance(near));
    }
}
