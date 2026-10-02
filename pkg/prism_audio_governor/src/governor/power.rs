//! Platform power profiles: a global cap layered over the quality ladder.
//!
//! Beyond the per-block closed loop, the platform imposes a *ceiling*. A
//! battery-powered handset should never climb to the reference tier even when
//! the CPU momentarily looks idle, because sustained high quality drains power
//! and overheats. Design section 32 calls for a mobile "power tier" that caps
//! the global sample-rate scale and the voice ceiling.
//!
//! This module models that as a [`PowerProfile`] (desktop, console, two mobile
//! tiers) that resolves to a [`PowerConstraints`] value. The governor clamps
//! its own tier to [`PowerConstraints::max_quality_tier`] and never exceeds the
//! physical-voice ceiling, regardless of what the budget loop would otherwise
//! allow.
//!
//! # Determinism
//!
//! Profiles map to constants; there is no floating-point state and the mapping
//! is a pure function.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Caps the [`crate::governor::lod::QualityTier`] chosen by
//! [`crate::governor::QualityGovernor`] and bounds the physical-voice budget
//! surfaced to the voice pool in `prism_audio_core`.

use prism_audio_core::math::Sample;

use crate::governor::lod::QualityTier;

/// A coarse platform/power class the game selects at startup (and may change if
/// the device enters a low-power state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PowerProfile {
    /// Mains-powered workstation: no artificial ceiling.
    Desktop,
    /// Home console: generous but bounded, matching a fixed thermal envelope.
    Console,
    /// Handheld in its performance power mode.
    MobileHigh,
    /// Handheld in a battery-saver / thermally throttled mode.
    MobileLow,
}

/// The concrete limits a [`PowerProfile`] imposes on the governor.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PowerConstraints {
    /// The richest quality tier the governor may ever select under this
    /// profile.
    pub max_quality_tier: QualityTier,
    /// The maximum number of simultaneously audible (physical) voices.
    pub max_physical_voices: u32,
    /// A global output sample-rate scale in `(0, 1]`: `1.0` is full rate, `0.5`
    /// is half rate for the deepest battery-saver mode.
    pub global_samplerate_scale: Sample,
}

impl PowerProfile {
    /// Resolves this profile to its concrete [`PowerConstraints`].
    #[must_use]
    pub fn constraints(self) -> PowerConstraints {
        match self {
            PowerProfile::Desktop => PowerConstraints {
                max_quality_tier: QualityTier(4),
                max_physical_voices: 256,
                global_samplerate_scale: 1.0,
            },
            PowerProfile::Console => PowerConstraints {
                max_quality_tier: QualityTier(4),
                max_physical_voices: 192,
                global_samplerate_scale: 1.0,
            },
            PowerProfile::MobileHigh => PowerConstraints {
                max_quality_tier: QualityTier(3),
                max_physical_voices: 96,
                global_samplerate_scale: 1.0,
            },
            PowerProfile::MobileLow => PowerConstraints {
                max_quality_tier: QualityTier(2),
                max_physical_voices: 48,
                global_samplerate_scale: 0.5,
            },
        }
    }
}

impl PowerConstraints {
    /// Returns the given tier clamped so it does not exceed
    /// [`max_quality_tier`](PowerConstraints::max_quality_tier).
    #[must_use]
    #[inline]
    pub fn clamp_tier(&self, tier: QualityTier) -> QualityTier {
        QualityTier(tier.0.min(self.max_quality_tier.0))
    }

    /// Returns the given physical-voice count clamped to
    /// [`max_physical_voices`](PowerConstraints::max_physical_voices).
    #[must_use]
    #[inline]
    pub fn clamp_voices(&self, requested: u32) -> u32 {
        requested.min(self.max_physical_voices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn desktop_is_unconstrained_tier() {
        let c = PowerProfile::Desktop.constraints();
        assert_eq!(c.max_quality_tier, QualityTier(4));
        assert!((c.global_samplerate_scale - 1.0).abs() < EPS);
    }

    #[test]
    fn mobile_low_caps_quality_and_rate() {
        let c = PowerProfile::MobileLow.constraints();
        assert_eq!(c.max_quality_tier, QualityTier(2));
        assert!((c.global_samplerate_scale - 0.5).abs() < EPS);
        assert!(c.max_physical_voices < PowerProfile::Desktop.constraints().max_physical_voices);
    }

    #[test]
    fn voice_ceiling_decreases_with_power() {
        let d = PowerProfile::Desktop.constraints().max_physical_voices;
        let mh = PowerProfile::MobileHigh.constraints().max_physical_voices;
        let ml = PowerProfile::MobileLow.constraints().max_physical_voices;
        assert!(d > mh);
        assert!(mh > ml);
    }

    #[test]
    fn tier_clamp_respects_ceiling() {
        let c = PowerProfile::MobileLow.constraints();
        assert_eq!(c.clamp_tier(QualityTier(4)), QualityTier(2));
        assert_eq!(c.clamp_tier(QualityTier(1)), QualityTier(1));
    }

    #[test]
    fn voice_clamp_respects_ceiling() {
        let c = PowerProfile::MobileHigh.constraints();
        assert_eq!(c.clamp_voices(1000), 96);
        assert_eq!(c.clamp_voices(10), 10);
    }
}
