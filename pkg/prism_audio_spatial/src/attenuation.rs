//! Distance attenuation: how a source's loudness falls off with range.
//!
//! This module implements the three classic *clamped* distance-attenuation
//! curves used across game and interactive audio: the **inverse-distance**,
//! **linear**, and **exponential** roll-off models. Each takes a reference
//! distance (inside which the source plays at unity gain), a maximum distance
//! (at and beyond which the curve stops changing), and a roll-off factor that
//! scales how aggressively gain decays between the two.
//!
//! # Provenance
//!
//! The curve formulae follow the *OpenAL 1.1 Specification* clamped distance
//! attenuation models (`AL_INVERSE_DISTANCE_CLAMPED`,
//! `AL_LINEAR_DISTANCE_CLAMPED`, and `AL_EXPONENT_DISTANCE_CLAMPED`), a public
//! specification. The same reference/max/roll-off parameterisation underlies
//! the attenuation curves exposed by middleware such as Wwise and FMOD. This
//! file contains **no Unreal Engine, Unity, Godot, Wwise, or FMOD source or
//! derived code**; it is implemented purely from that publicly documented
//! acoustics knowledge.
//!
//! # Determinism
//!
//! The only transcendental used here (the exponential model's power) routes
//! through [`bevy_math::ops::powf`] rather than an `f32` intrinsic, so results
//! are bit-reproducible across targets, as the workspace lints enforce. The
//! remaining arithmetic is plain multiply/add/divide plus range clamping.

use bevy_math::ops;
use prism_audio_core::math::Sample;

/// Smallest positive reference distance (in metres) permitted by
/// [`Attenuation::new`]. Keeping the reference strictly positive guarantees the
/// inverse and exponential models never divide by zero.
const MIN_REFERENCE_DISTANCE: Sample = 1.0e-6;

/// Selects which distance-attenuation curve an [`Attenuation`] evaluates.
///
/// All three variants are the *clamped* forms from the OpenAL 1.1
/// specification: distance is first restricted to
/// `[reference_distance, max_distance]` before the curve is applied, so gain is
/// flat at unity inside the reference radius and constant at (or beyond) the
/// maximum radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DistanceModel {
    /// Inverse-distance (a.k.a. `1/r`) roll-off. Physically motivated:
    /// free-field sound pressure from a point source falls off with the inverse
    /// of distance. This is the OpenAL default.
    Inverse,
    /// Linear roll-off from unity at the reference distance to zero at the
    /// maximum distance. Not physical, but predictable and easy to author.
    Linear,
    /// Exponential roll-off, `(d / reference)^(-rolloff)`, giving a steeper,
    /// more artistically controllable curve than the inverse model.
    Exponential,
}

/// A distance-attenuation descriptor: a curve choice plus its parameters.
///
/// This is plain, authoring-time description data (not a real-time node); it is
/// cheap to copy and, with the `serialize` feature, (de)serializable. Evaluate
/// it with [`Attenuation::gain`] to obtain a linear gain multiplier in
/// `[0, 1]`.
///
/// Construct instances through [`Attenuation::new`], which sanitises the
/// parameters so that [`gain`](Attenuation::gain) can never divide by zero or
/// produce a NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Attenuation {
    /// Which curve to evaluate.
    pub model: DistanceModel,
    /// Distance (metres) inside which gain is unity. Strictly positive.
    pub reference_distance: Sample,
    /// Distance (metres) at and beyond which the curve stops changing. Always
    /// at least [`reference_distance`](Attenuation::reference_distance).
    pub max_distance: Sample,
    /// Non-negative multiplier scaling how quickly gain decays. `0` disables
    /// attenuation entirely (constant unity gain).
    pub rolloff_factor: Sample,
}

impl Default for Attenuation {
    /// The OpenAL-style default: inverse-distance roll-off with a `1 m`
    /// reference, an effectively unbounded `10_000 m` maximum, and unit
    /// roll-off.
    #[inline]
    fn default() -> Self {
        Self {
            model: DistanceModel::Inverse,
            reference_distance: 1.0,
            max_distance: 10_000.0,
            rolloff_factor: 1.0,
        }
    }
}

impl Attenuation {
    /// Creates an attenuation descriptor, sanitising its parameters.
    ///
    /// The inputs are clamped into a self-consistent, safe-to-evaluate range:
    ///
    /// * `reference_distance` is raised to at least [`MIN_REFERENCE_DISTANCE`],
    ///   keeping it strictly positive so the inverse and exponential curves
    ///   never divide by zero.
    /// * `max_distance` is raised to at least the (already sanitised)
    ///   `reference_distance`, so the clamp range is never inverted.
    /// * `rolloff_factor` is clamped to be non-negative.
    #[must_use]
    #[inline]
    pub fn new(
        model: DistanceModel,
        reference_distance: Sample,
        max_distance: Sample,
        rolloff_factor: Sample,
    ) -> Self {
        let reference_distance = reference_distance.max(MIN_REFERENCE_DISTANCE);
        let max_distance = max_distance.max(reference_distance);
        let rolloff_factor = rolloff_factor.max(0.0);
        Self { model, reference_distance, max_distance, rolloff_factor }
    }

    /// Evaluates the linear gain multiplier for a source at `distance` metres.
    ///
    /// The result is always in `[0, 1]`, where `1` is unity (no attenuation)
    /// and `0` is silence. `distance` is first clamped to
    /// `[reference_distance, max_distance]`, so:
    ///
    /// * any distance at or below the reference returns `1.0`, and
    /// * any distance at or beyond the maximum returns the same value as the
    ///   maximum (the curve is flat outside the range).
    ///
    /// Within the range the curve depends on [`model`](Attenuation::model):
    ///
    /// * [`DistanceModel::Inverse`]:
    ///   `reference / (reference + rolloff * (d - reference))`
    /// * [`DistanceModel::Linear`]:
    ///   `1 - rolloff * (d - reference) / (max - reference)`
    ///   (a degenerate `max == reference` never reaches this branch because the
    ///   clamp forces `d == reference`)
    /// * [`DistanceModel::Exponential`]: `(d / reference)^(-rolloff)`
    ///
    /// A `rolloff_factor` of `0` yields constant unity gain in every model.
    #[must_use]
    #[inline]
    pub fn gain(&self, distance: Sample) -> Sample {
        let reference = self.reference_distance;
        let max = self.max_distance;

        // Clamp distance into the model's valid range first (the "clamped"
        // family). `new` guarantees `reference <= max`, so this is well-formed.
        let d = distance.clamp(reference, max);

        // Flat unity gain inside the reference radius; also covers the
        // degenerate `max == reference` case, where `d` is pinned to
        // `reference`.
        if d <= reference {
            return 1.0;
        }

        let raw = match self.model {
            DistanceModel::Inverse => {
                reference / (reference + self.rolloff_factor * (d - reference))
            }
            DistanceModel::Linear => {
                // `d > reference` here implies `max > reference` (otherwise the
                // clamp would have produced `d == reference`), so the divisor is
                // strictly positive.
                1.0 - self.rolloff_factor * (d - reference) / (max - reference)
            }
            DistanceModel::Exponential => ops::powf(d / reference, -self.rolloff_factor),
        };

        raw.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for float comparisons in these tests.
    const EPS: Sample = 1.0e-5;

    fn assert_close(a: Sample, b: Sample) {
        assert!((a - b).abs() <= EPS, "expected {a} ~= {b}");
    }

    #[test]
    fn new_sanitises_parameters() {
        // Non-positive reference is lifted to the positive minimum.
        let a = Attenuation::new(DistanceModel::Inverse, 0.0, -5.0, -3.0);
        assert!(a.reference_distance >= MIN_REFERENCE_DISTANCE);
        // Max is lifted to at least the sanitised reference.
        assert!(a.max_distance >= a.reference_distance);
        // Roll-off is clamped non-negative.
        assert!(a.rolloff_factor >= 0.0);
    }

    #[test]
    fn gain_at_and_within_reference_is_unity() {
        for model in [
            DistanceModel::Inverse,
            DistanceModel::Linear,
            DistanceModel::Exponential,
        ] {
            let a = Attenuation::new(model, 2.0, 50.0, 1.0);
            assert_close(a.gain(a.reference_distance), 1.0);
            // Anything closer than the reference is also unity.
            assert_close(a.gain(0.5), 1.0);
            assert_close(a.gain(0.0), 1.0);
        }
    }

    #[test]
    fn gain_is_monotonically_non_increasing() {
        for model in [
            DistanceModel::Inverse,
            DistanceModel::Linear,
            DistanceModel::Exponential,
        ] {
            let a = Attenuation::new(model, 1.0, 100.0, 1.0);
            let mut prev = a.gain(0.0);
            let mut d = 0.0;
            while d <= 120.0 {
                let g = a.gain(d);
                assert!(
                    g <= prev + EPS,
                    "gain increased for {model:?} at d={d}: {g} > {prev}"
                );
                assert!((0.0..=1.0).contains(&g), "gain out of range: {g}");
                prev = g;
                d += 0.5;
            }
        }
    }

    #[test]
    fn gain_is_clamped_beyond_max() {
        for model in [
            DistanceModel::Inverse,
            DistanceModel::Linear,
            DistanceModel::Exponential,
        ] {
            let a = Attenuation::new(model, 1.0, 25.0, 1.5);
            let at_max = a.gain(a.max_distance);
            // Anything past the maximum yields exactly the max-distance gain.
            assert_close(a.gain(a.max_distance + 1.0), at_max);
            assert_close(a.gain(1.0e6), at_max);
        }
    }

    #[test]
    fn inverse_model_sanity() {
        // reference = 1, rolloff = 1: gain(d) = 1 / d.
        let a = Attenuation::new(DistanceModel::Inverse, 1.0, 1000.0, 1.0);
        assert_close(a.gain(2.0), 0.5);
        assert_close(a.gain(4.0), 0.25);
        assert_close(a.gain(10.0), 0.1);
    }

    #[test]
    fn linear_model_sanity() {
        // reference = 0? no; reference = 1, max = 11, rolloff = 1:
        // gain(d) = 1 - (d - 1) / 10.
        let a = Attenuation::new(DistanceModel::Linear, 1.0, 11.0, 1.0);
        assert_close(a.gain(1.0), 1.0);
        assert_close(a.gain(6.0), 0.5);
        assert_close(a.gain(11.0), 0.0);
        // Half roll-off halves the slope, so the endpoint is 0.5.
        let half = Attenuation::new(DistanceModel::Linear, 1.0, 11.0, 0.5);
        assert_close(half.gain(11.0), 0.5);
    }

    #[test]
    fn exponential_model_sanity() {
        // reference = 1, rolloff = 2: gain(d) = d^-2.
        let a = Attenuation::new(DistanceModel::Exponential, 1.0, 1000.0, 2.0);
        assert_close(a.gain(2.0), 0.25);
        assert_close(a.gain(4.0), 0.0625);
        // rolloff = 1 collapses onto the inverse curve for reference = 1.
        let b = Attenuation::new(DistanceModel::Exponential, 1.0, 1000.0, 1.0);
        assert_close(b.gain(10.0), 0.1);
    }

    #[test]
    fn zero_rolloff_is_always_unity() {
        for model in [
            DistanceModel::Inverse,
            DistanceModel::Linear,
            DistanceModel::Exponential,
        ] {
            let a = Attenuation::new(model, 1.0, 100.0, 0.0);
            for &d in &[0.0, 1.0, 5.0, 50.0, 100.0, 1000.0] {
                assert_close(a.gain(d), 1.0);
            }
        }
    }

    #[test]
    fn degenerate_max_equals_reference_is_stable() {
        for model in [
            DistanceModel::Inverse,
            DistanceModel::Linear,
            DistanceModel::Exponential,
        ] {
            // Request max < reference; `new` lifts max up to reference.
            let a = Attenuation::new(model, 5.0, 1.0, 1.0);
            assert_close(a.reference_distance, 5.0);
            assert_close(a.max_distance, 5.0);
            for &d in &[0.0, 5.0, 5.0e3, 1.0e9] {
                let g = a.gain(d);
                assert!(!g.is_nan(), "NaN gain for {model:?} at d={d}");
                // Distance always clamps to the reference, so gain is unity.
                assert_close(g, 1.0);
            }
        }
    }

    #[test]
    fn default_matches_openal_defaults() {
        let a = Attenuation::default();
        assert_eq!(a.model, DistanceModel::Inverse);
        assert_close(a.reference_distance, 1.0);
        assert_close(a.max_distance, 10_000.0);
        assert_close(a.rolloff_factor, 1.0);
    }
}
