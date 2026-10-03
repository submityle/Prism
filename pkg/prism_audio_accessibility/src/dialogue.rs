//! Dialogue-priority boost that strengthens ducking of non-dialogue buses.
//!
//! [`DialogueBoost`] gives speech the highest priority in the mix. It does not
//! process audio itself; instead it emits a control-rate adjustment to the
//! parameters of the existing side-chain ducker so that music, ambience, and
//! effects are pushed further down (and recover a touch later) while dialogue
//! is present, which raises intelligibility for players who need it.
//!
//! The adjustment is expressed in decibels: a [`BoostStrength`] selects how
//! much extra attenuation (in dB) to add to the ducker range and how far to
//! lower the key threshold so the ducker engages earlier.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the dialogue-priority item of design section 23 by reusing the
//! ducking node and its high-dynamic-range window from design section 13. This
//! crate only transforms a [`DuckingParams`] value; the actual gain reduction
//! is still performed by the `prism_audio_core` ducking node.

use prism_audio_core::math::Sample;
use prism_audio_core::nodes::dynamics::ducking::DuckingParams;

/// Discrete intensity levels for the dialogue-priority boost.
///
/// Each level maps to an amount of extra ducking range and a key-threshold
/// offset. [`BoostStrength::Off`] is the identity: it changes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BoostStrength {
    /// No boost; the ducker is left untouched.
    Off,
    /// Gentle boost for mild intelligibility help.
    Low,
    /// Moderate boost suitable as a default accessibility setting.
    Medium,
    /// Aggressive boost for maximum dialogue clarity.
    High,
}

impl BoostStrength {
    /// Extra ducking range, in dB, added on top of the base ducker range.
    #[inline]
    #[must_use]
    pub const fn extra_range_db(self) -> Sample {
        match self {
            BoostStrength::Off => 0.0,
            BoostStrength::Low => 3.0,
            BoostStrength::Medium => 6.0,
            BoostStrength::High => 10.0,
        }
    }

    /// Key-threshold offset, in dB, applied so the ducker engages earlier.
    ///
    /// The value is non-positive: a more negative offset lowers the threshold
    /// and makes the ducker react to quieter dialogue.
    #[inline]
    #[must_use]
    pub const fn threshold_offset_db(self) -> Sample {
        match self {
            BoostStrength::Off => 0.0,
            BoostStrength::Low => -4.0,
            BoostStrength::Medium => -8.0,
            BoostStrength::High => -12.0,
        }
    }
}

/// Dialogue-priority accommodation that augments a side-chain ducker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DialogueBoost {
    /// Selected intensity level.
    strength: BoostStrength,
}

impl DialogueBoost {
    /// Creates a boost at the given `strength`.
    #[inline]
    #[must_use]
    pub const fn new(strength: BoostStrength) -> Self {
        Self { strength }
    }

    /// Returns the selected intensity level.
    #[inline]
    #[must_use]
    pub const fn strength(&self) -> BoostStrength {
        self.strength
    }

    /// Sets the intensity level.
    #[inline]
    pub const fn set_strength(&mut self, strength: BoostStrength) {
        self.strength = strength;
    }

    /// Returns `true` when the boost will change the ducker.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self.strength, BoostStrength::Off)
    }

    /// Returns the extra non-dialogue attenuation, in dB, this boost requests.
    ///
    /// The value is the additional ducking range beyond the base ducker; it is
    /// zero when the boost is [`BoostStrength::Off`].
    #[inline]
    #[must_use]
    pub fn extra_attenuation_db(&self) -> Sample {
        self.strength.extra_range_db()
    }

    /// Applies the boost to a base [`DuckingParams`], returning adjusted params.
    ///
    /// The ducker range grows by [`Self::extra_attenuation_db`] and the key
    /// threshold is lowered by the strength offset so dialogue engages the
    /// ducker sooner. All other fields are preserved. The threshold is clamped
    /// to a floor of `-80 dB` so it stays in a sensible range.
    #[must_use]
    pub fn apply(&self, base: DuckingParams) -> DuckingParams {
        let mut out = base;
        out.range_db = (base.range_db + self.strength.extra_range_db()).max(0.0);
        out.threshold_db = (base.threshold_db + self.strength.threshold_offset_db()).max(-80.0);
        out
    }
}

impl Default for DialogueBoost {
    fn default() -> Self {
        Self::new(BoostStrength::Off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn off_is_identity() {
        let base = DuckingParams::default();
        let out = DialogueBoost::new(BoostStrength::Off).apply(base);
        assert!((out.range_db - base.range_db).abs() < EPS);
        assert!((out.threshold_db - base.threshold_db).abs() < EPS);
    }

    #[test]
    fn strength_order_is_monotonic() {
        let low = BoostStrength::Low.extra_range_db();
        let medium = BoostStrength::Medium.extra_range_db();
        let high = BoostStrength::High.extra_range_db();
        assert!(low < medium);
        assert!(medium < high);
        // Threshold offsets become more negative as strength increases.
        assert!(BoostStrength::Low.threshold_offset_db() > BoostStrength::Medium.threshold_offset_db());
        assert!(BoostStrength::Medium.threshold_offset_db() > BoostStrength::High.threshold_offset_db());
    }

    #[test]
    fn apply_increases_range_and_lowers_threshold() {
        let base = DuckingParams::default();
        let boost = DialogueBoost::new(BoostStrength::High);
        let out = boost.apply(base);
        assert!((out.range_db - (base.range_db + 10.0)).abs() < EPS);
        assert!((out.threshold_db - (base.threshold_db - 12.0)).abs() < EPS);
        assert!(boost.is_active());
        assert!((boost.extra_attenuation_db() - 10.0).abs() < EPS);
    }

    #[test]
    fn threshold_has_a_floor() {
        let base = DuckingParams {
            threshold_db: -75.0,
            ..DuckingParams::default()
        };
        let out = DialogueBoost::new(BoostStrength::High).apply(base);
        assert!(out.threshold_db >= -80.0);
    }
}
