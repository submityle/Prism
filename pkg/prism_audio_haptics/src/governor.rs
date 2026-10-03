//! CPU-budget-driven quality scaling and low-tier bypass for haptics.
//!
//! Haptics are a non-essential, felt-only side-channel, so they are the first
//! thing to shed when the audio thread runs short of budget. The
//! [`HapticGovernor`] holds the active quality [`HapticTier`] and translates it
//! into concrete decisions: a scalar quality multiplier, whether generation is
//! bypassed entirely, and a scaled [`crate::transcode::TranscodeConfig`] (for
//! example a halved haptic rate at the reduced tier). A CPU-load reading can be
//! mapped straight onto a tier so the budget authority can drive the governor.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the governor of design section 36 and obeys the budget authority
//! of design section 32. It scales the [`crate::transcode::TranscodeConfig`]
//! consumed by [`crate::transcode::HapticTranscoder`].

use prism_audio_core::math::Sample;

use crate::transcode::TranscodeConfig;

/// Discrete haptic quality tiers selected by the budget authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum HapticTier {
    /// Full-rate, full-quality haptic generation.
    #[default]
    Full,
    /// Reduced quality: the haptic rate is halved to save budget.
    Reduced,
    /// Haptic generation is skipped entirely.
    Bypass,
}

/// Load fraction (0..1) at or above which the reduced tier engages.
const REDUCED_THRESHOLD: Sample = 0.75;

/// Load fraction (0..1) at or above which haptics are bypassed.
const BYPASS_THRESHOLD: Sample = 0.9;

/// Chooses and applies the active haptic quality tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HapticGovernor {
    tier: HapticTier,
}

impl HapticGovernor {
    /// Builds a governor fixed at `tier`.
    #[must_use]
    pub fn new(tier: HapticTier) -> Self {
        Self { tier }
    }

    /// Returns the active tier.
    #[inline]
    #[must_use]
    pub fn tier(&self) -> HapticTier {
        self.tier
    }

    /// Sets the active tier.
    #[inline]
    pub fn set_tier(&mut self, tier: HapticTier) {
        self.tier = tier;
    }

    /// Maps a CPU load fraction in `[0, 1]` to a tier.
    ///
    /// Below [`REDUCED_THRESHOLD`] the full tier is used; between the two
    /// thresholds the reduced tier; at or above [`BYPASS_THRESHOLD`] haptics
    /// are bypassed. Non-finite loads are treated as fully loaded.
    #[must_use]
    pub fn tier_for_load(load: Sample) -> HapticTier {
        if !load.is_finite() || load >= BYPASS_THRESHOLD {
            HapticTier::Bypass
        } else if load >= REDUCED_THRESHOLD {
            HapticTier::Reduced
        } else {
            HapticTier::Full
        }
    }

    /// Updates the active tier from a CPU load fraction.
    pub fn set_from_load(&mut self, load: Sample) {
        self.tier = Self::tier_for_load(load);
    }

    /// Returns `true` when haptic generation is skipped.
    #[inline]
    #[must_use]
    pub fn is_bypassed(&self) -> bool {
        matches!(self.tier, HapticTier::Bypass)
    }

    /// Returns the quality multiplier for the active tier (`1.0`, `0.5`, `0`).
    #[must_use]
    pub fn quality_scale(&self) -> Sample {
        match self.tier {
            HapticTier::Full => 1.0,
            HapticTier::Reduced => 0.5,
            HapticTier::Bypass => 0.0,
        }
    }

    /// Returns the transcode config scaled for the active tier.
    ///
    /// The full tier passes `config` through unchanged, the reduced tier halves
    /// the haptic rate (never below one hertz), and the bypass tier returns
    /// `None` to signal that no haptic signal should be generated.
    #[must_use]
    pub fn apply(&self, config: TranscodeConfig) -> Option<TranscodeConfig> {
        match self.tier {
            HapticTier::Full => Some(config),
            HapticTier::Reduced => {
                let mut scaled = config;
                scaled.haptic_rate_hz = (config.haptic_rate_hz / 2).max(1);
                Some(scaled)
            }
            HapticTier::Bypass => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_full() {
        let g = HapticGovernor::default();
        assert_eq!(g.tier(), HapticTier::Full);
        assert!(!g.is_bypassed());
    }

    #[test]
    fn load_maps_to_tiers() {
        assert_eq!(HapticGovernor::tier_for_load(0.1), HapticTier::Full);
        assert_eq!(HapticGovernor::tier_for_load(0.8), HapticTier::Reduced);
        assert_eq!(HapticGovernor::tier_for_load(0.95), HapticTier::Bypass);
        assert_eq!(HapticGovernor::tier_for_load(Sample::NAN), HapticTier::Bypass);
    }

    #[test]
    fn set_from_load_updates_tier() {
        let mut g = HapticGovernor::new(HapticTier::Full);
        g.set_from_load(0.99);
        assert!(g.is_bypassed());
    }

    #[test]
    fn quality_scale_matches_tier() {
        assert!((HapticGovernor::new(HapticTier::Full).quality_scale() - 1.0).abs() < 1e-6);
        assert!((HapticGovernor::new(HapticTier::Reduced).quality_scale() - 0.5).abs() < 1e-6);
        assert!(HapticGovernor::new(HapticTier::Bypass).quality_scale().abs() < 1e-6);
    }

    #[test]
    fn apply_full_is_identity() {
        let g = HapticGovernor::new(HapticTier::Full);
        let c = TranscodeConfig::default();
        let out = g.apply(c).unwrap();
        assert_eq!(out.haptic_rate_hz, c.haptic_rate_hz);
    }

    #[test]
    fn apply_reduced_halves_rate() {
        let g = HapticGovernor::new(HapticTier::Reduced);
        let c = TranscodeConfig {
            haptic_rate_hz: 1_000,
            ..TranscodeConfig::default()
        };
        let out = g.apply(c).unwrap();
        assert_eq!(out.haptic_rate_hz, 500);
    }

    #[test]
    fn apply_reduced_never_below_one() {
        let g = HapticGovernor::new(HapticTier::Reduced);
        let c = TranscodeConfig {
            haptic_rate_hz: 1,
            ..TranscodeConfig::default()
        };
        let out = g.apply(c).unwrap();
        assert_eq!(out.haptic_rate_hz, 1);
    }

    #[test]
    fn apply_bypass_is_none() {
        let g = HapticGovernor::new(HapticTier::Bypass);
        assert!(g.apply(TranscodeConfig::default()).is_none());
    }
}
