//! Declarative ducking rules between mix categories.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! CRIWARE, or Google Resonance Audio source or derived code; no AI/ML.
//! Side-chain ducking with attack/recovery ballistics is a classical mixing
//! technique; this is an independent implementation expressed as data.
//!
//! # Relationship
//! Implements the declarative rule of design section 46.8: "when category A has
//! active sources, category B attenuates by `x` dB using attack/recovery
//! curves". A rule is pure data; [`super::compiler`] lowers it onto design
//! section 12 modulation routes and the attack/release envelope of
//! [`super::activity`], so the runtime still resolves per-block `Smoothed`
//! gains and remains golden-reproducible.

use prism_audio_core::Sample;

use crate::modulation::Curve;

use super::category::CategoryId;

/// Default attack time in seconds applied when a trigger becomes active.
pub const DEFAULT_ATTACK_SECONDS: Sample = 0.01;

/// Default release time in seconds applied when a trigger goes idle.
pub const DEFAULT_RELEASE_SECONDS: Sample = 0.2;

/// A single declarative ducking relationship.
///
/// When the `trigger` category is active, the `target` category's gain is
/// reduced by up to `attenuation_db` decibels. The reduction follows the
/// `trigger`'s activity through an attack/release envelope (see
/// [`super::activity::ActivityEnvelope`]) and is shaped by `curve` before being
/// applied as a gain multiplier.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DuckingRule {
    /// Category whose activity drives the duck.
    pub trigger: CategoryId,
    /// Category whose gain is reduced while the trigger is active.
    pub target: CategoryId,
    /// Maximum gain reduction in decibels at full trigger activity.
    ///
    /// Expressed as a non-negative number of decibels of attenuation; `0`
    /// disables the duck and larger values reduce the target more.
    pub attenuation_db: Sample,
    /// Attack time in seconds as the reduction engages.
    pub attack_seconds: Sample,
    /// Release time in seconds as the reduction recovers.
    pub release_seconds: Sample,
    /// Response shape mapping normalized activity to normalized reduction.
    pub curve: Curve,
}

impl DuckingRule {
    /// Creates a rule reducing `target` by `attenuation_db` while `trigger` is
    /// active, using the default attack/release ballistics and a linear curve.
    #[must_use]
    pub fn new(trigger: CategoryId, target: CategoryId, attenuation_db: Sample) -> Self {
        Self {
            trigger,
            target,
            attenuation_db: attenuation_db.max(0.0),
            attack_seconds: DEFAULT_ATTACK_SECONDS,
            release_seconds: DEFAULT_RELEASE_SECONDS,
            curve: Curve::Linear,
        }
    }

    /// Sets the attack time in seconds (clamped to be non-negative).
    #[must_use]
    pub fn with_attack(mut self, attack_seconds: Sample) -> Self {
        self.attack_seconds = attack_seconds.max(0.0);
        self
    }

    /// Sets the release time in seconds (clamped to be non-negative).
    #[must_use]
    pub fn with_release(mut self, release_seconds: Sample) -> Self {
        self.release_seconds = release_seconds.max(0.0);
        self
    }

    /// Sets the response curve applied to the normalized activity.
    #[must_use]
    pub fn with_curve(mut self, curve: Curve) -> Self {
        self.curve = curve;
        self
    }

    /// Returns the linear gain multiplier at full reduction.
    ///
    /// This is `10^(-attenuation_db / 20)`, i.e. the floor gain the target
    /// reaches when the trigger is fully active: `1.0` for no attenuation and
    /// approaching `0.0` for very large attenuations.
    #[must_use]
    pub fn floor_gain(&self) -> Sample {
        decibels_to_linear(-self.attenuation_db)
    }

    /// Returns the maximum reduction fraction `1 - floor_gain` in `[0, 1)`.
    ///
    /// This is the depth the compiler installs on the modulation route: the
    /// amount subtracted from unity gain at full, fully shaped activity.
    #[must_use]
    pub fn reduction_depth(&self) -> Sample {
        1.0 - self.floor_gain()
    }
}

/// Converts a decibel value to a linear amplitude ratio.
///
/// Uses `10^(db/20) = exp(db * ln(10) / 20)` via [`bevy_math::ops::exp`] so the
/// result is bit-identical across platforms, matching the crate's determinism
/// contract (design section 24).
#[must_use]
pub fn decibels_to_linear(db: Sample) -> Sample {
    /// `ln(10) / 20`, the exponent scale from decibels to a natural-log power.
    const LN10_OVER_20: Sample = 0.115_129_255;
    bevy_math::ops::exp(db * LN10_OVER_20)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn zero_db_is_unity() {
        assert!(close(decibels_to_linear(0.0), 1.0));
    }

    #[test]
    fn minus_six_db_is_half_power_amplitude() {
        // -6.0206 dB is exactly 0.5 in amplitude; -6 dB is close.
        assert!(close(decibels_to_linear(-6.020_6), 0.5));
    }

    #[test]
    fn floor_gain_and_depth_are_consistent() {
        let rule = DuckingRule::new(CategoryId(0), CategoryId(1), 12.0);
        let g = rule.floor_gain();
        assert!(close(g, decibels_to_linear(-12.0)));
        assert!(close(rule.reduction_depth(), 1.0 - g));
        assert!(g > 0.0 && g < 1.0);
    }

    #[test]
    fn builders_clamp_and_set() {
        let rule = DuckingRule::new(CategoryId(0), CategoryId(1), -3.0)
            .with_attack(-1.0)
            .with_release(0.5);
        // Negative attenuation and attack are clamped to zero.
        assert!(close(rule.attenuation_db, 0.0));
        assert!(close(rule.attack_seconds, 0.0));
        assert!(close(rule.release_seconds, 0.5));
        // Zero attenuation leaves the target at unity gain.
        assert!(close(rule.floor_gain(), 1.0));
        assert!(close(rule.reduction_depth(), 0.0));
    }
}
