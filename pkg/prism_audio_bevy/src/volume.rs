//! The [`Volume`] value type: a loudness scalar expressed in either the linear
//! amplitude domain or the decibel domain, with lossless conversion between
//! them.
//!
//! Content authors habitually reason about loudness in decibels (a -6 dB trim,
//! a +3 dB boost) while the renderer multiplies linear amplitudes. [`Volume`]
//! keeps whichever representation the author chose and converts on demand, so
//! an authored `-6 dB` round-trips without silently collapsing to a float the
//! moment it is constructed. The ergonomic shape mirrors the component-based
//! source APIs common to entity engines, but the arithmetic and the decibel
//! mapping are original and use only [`bevy_math::ops`] for determinism.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Carried by [`crate::playback::PlaybackSettings`]; folded into the base
//! [`Importance`](prism_audio_core::voice::Importance) that
//! [`crate::player`] forwards to the runtime through
//! [`crate::emitter::AudioEmitter`].

use bevy_math::ops;

/// Natural logarithm of ten, used to convert decibels to a linear amplitude
/// ratio via `10^(db/20) = exp(db/20 * ln 10)`.
const LN_10: f32 = core::f32::consts::LN_10;

/// Linear amplitude at which a signal is treated as fully silent. Decibel
/// readings of amplitudes at or below this floor report negative infinity
/// rather than a huge negative finite value.
const SILENCE_FLOOR: f32 = 1.0e-9;

/// A loudness scalar in either the linear amplitude or decibel domain.
///
/// The two variants are interchangeable through [`Volume::to_linear`] and
/// [`Volume::to_decibels`]; equality and arithmetic are defined on the linear
/// amplitude so `Volume::Linear(0.5)` and `Volume::Decibels(-6.0206)` compare
/// and combine identically up to float precision.
#[derive(Debug, Clone, Copy)]
pub enum Volume {
    /// A linear amplitude multiplier: `1.0` is unity, `0.0` is silence, and
    /// values above one amplify. Negative inputs are treated as silence by the
    /// conversions.
    Linear(f32),
    /// A gain in decibels relative to unity: `0.0` is unity, negative values
    /// attenuate, positive values amplify. Negative infinity is exact silence.
    Decibels(f32),
}

impl Volume {
    /// Unity gain: the signal passes through unchanged.
    pub const UNITY: Self = Volume::Linear(1.0);

    /// Exact silence.
    pub const SILENT: Self = Volume::Linear(0.0);

    /// Returns the linear amplitude multiplier for this volume.
    ///
    /// Linear volumes are clamped at zero (negative amplitudes are treated as
    /// silence); decibel volumes map through `10^(db/20)`, with negative
    /// infinity producing exact zero.
    #[must_use]
    pub fn to_linear(self) -> f32 {
        match self {
            Volume::Linear(amplitude) => amplitude.max(0.0),
            Volume::Decibels(db) => {
                if db == f32::NEG_INFINITY {
                    0.0
                } else {
                    ops::exp(db * (LN_10 / 20.0))
                }
            }
        }
    }

    /// Returns the decibel gain for this volume.
    ///
    /// Decibel volumes are returned verbatim; linear amplitudes map through
    /// `20 * log10(amplitude)`, with amplitudes at or below the silence floor
    /// reporting negative infinity.
    #[must_use]
    pub fn to_decibels(self) -> f32 {
        match self {
            Volume::Decibels(db) => db,
            Volume::Linear(amplitude) => {
                if amplitude <= SILENCE_FLOOR {
                    f32::NEG_INFINITY
                } else {
                    20.0 * ops::log10(amplitude)
                }
            }
        }
    }

    /// Whether this volume is effectively silent (linear amplitude at or below
    /// the silence floor).
    #[must_use]
    #[inline]
    pub fn is_silent(self) -> bool {
        self.to_linear() <= SILENCE_FLOOR
    }

    /// Combines two volumes multiplicatively in the linear domain, returning a
    /// [`Volume::Linear`]. Combining is how a per-source trim stacks on top of
    /// a group trim: `-6 dB` times `-6 dB` is `-12 dB`.
    #[must_use]
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        Volume::Linear(self.to_linear() * other.to_linear())
    }

    /// Adjusts the volume by a signed percentage of its current linear
    /// amplitude. `+100.0` doubles the amplitude; `-50.0` halves it; values
    /// that would drive the amplitude below zero clamp to silence.
    #[must_use]
    #[inline]
    pub fn adjust_by_percentage(self, percent: f32) -> Self {
        let scaled = self.to_linear() * (1.0 + percent / 100.0);
        Volume::Linear(scaled.max(0.0))
    }
}

impl Default for Volume {
    /// Unity gain.
    #[inline]
    fn default() -> Self {
        Volume::UNITY
    }
}

impl PartialEq for Volume {
    /// Compares two volumes by their linear amplitude, so the two domains stay
    /// interchangeable under equality.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_linear() == other.to_linear()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Epsilon used instead of exact float equality in assertions.
    const EPS: f32 = 1.0e-4;

    /// Approximate float comparison.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn unity_round_trips_both_ways() {
        assert!(close(Volume::UNITY.to_linear(), 1.0));
        assert!(close(Volume::UNITY.to_decibels(), 0.0));
        assert!(close(Volume::Decibels(0.0).to_linear(), 1.0));
    }

    #[test]
    fn minus_six_decibels_is_half_amplitude() {
        let half = Volume::Decibels(-6.020_6);
        assert!(close(half.to_linear(), 0.5), "linear was {}", half.to_linear());
    }

    #[test]
    fn linear_to_decibels_matches_known_point() {
        let db = Volume::Linear(0.5).to_decibels();
        assert!(close(db, -6.020_6), "decibels were {db}");
    }

    #[test]
    fn silence_maps_to_negative_infinity_decibels() {
        assert_eq!(Volume::SILENT.to_decibels(), f32::NEG_INFINITY);
        assert!(Volume::SILENT.is_silent());
    }

    #[test]
    fn negative_infinity_decibels_is_exact_silence() {
        assert_eq!(Volume::Decibels(f32::NEG_INFINITY).to_linear(), 0.0);
    }

    #[test]
    fn negative_linear_amplitude_clamps_to_silence() {
        assert_eq!(Volume::Linear(-2.0).to_linear(), 0.0);
    }

    #[test]
    fn combine_stacks_in_linear_domain() {
        let stacked = Volume::Decibels(-6.020_6).combine(Volume::Decibels(-6.020_6));
        assert!(close(stacked.to_linear(), 0.25), "linear was {}", stacked.to_linear());
    }

    #[test]
    fn adjust_by_percentage_scales_amplitude() {
        assert!(close(Volume::Linear(1.0).adjust_by_percentage(100.0).to_linear(), 2.0));
        assert!(close(Volume::Linear(1.0).adjust_by_percentage(-50.0).to_linear(), 0.5));
        assert_eq!(Volume::Linear(1.0).adjust_by_percentage(-200.0).to_linear(), 0.0);
    }

    #[test]
    fn equality_is_domain_independent() {
        assert_eq!(Volume::Linear(0.5), Volume::Decibels(-6.020_6));
    }
}
