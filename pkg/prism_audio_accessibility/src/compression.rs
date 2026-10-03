//! Night and hard-of-hearing dynamic-range compression profiles.
//!
//! A [`CompressionProfile`] narrows the dynamic-range window so that quiet
//! passages stay audible and loud passages stop startling the listener. It is a
//! control-rate preset: it carries a threshold, ratio, knee, and make-up gain
//! that are applied to the engine compressor, which still performs the actual
//! per-sample gain reduction.
//!
//! Three presets are provided: [`CompressionPreset::Off`] leaves the signal
//! untouched (unity ratio), [`CompressionPreset::Night`] is a moderate
//! late-night setting, and [`CompressionPreset::HardOfHearing`] applies a
//! stronger reduction with more make-up gain for listeners who need a tightly
//! bounded loudness range.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the dynamic-range-compression item of design section 23 by
//! narrowing the high-dynamic-range window of design section 13. The profile is
//! applied onto a [`CompressorParams`] value and the resulting parameters drive
//! the `prism_audio_core` compressor node, which is reused unchanged.

use prism_audio_core::math::Sample;
use prism_audio_core::nodes::dynamics::compressor::CompressorParams;

/// Named dynamic-range presets for accessibility listening modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum CompressionPreset {
    /// No compression; the dynamic range is left intact.
    Off,
    /// Moderate night-time compression.
    Night,
    /// Strong compression for hard-of-hearing listeners.
    HardOfHearing,
}

/// A dynamic-range compression accommodation resolved from a preset.
///
/// The profile stores the four parameters that define its reduction window and
/// can stamp them onto a base [`CompressorParams`] value.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressionProfile {
    /// Which preset this profile was built from.
    preset: CompressionPreset,
    /// Threshold in dBFS above which reduction begins.
    threshold_db: Sample,
    /// Compression ratio (`>= 1`).
    ratio: Sample,
    /// Soft-knee width in dB.
    knee_db: Sample,
    /// Make-up gain in dB applied after reduction.
    makeup_db: Sample,
}

impl CompressionProfile {
    /// Builds the profile for `preset`.
    #[must_use]
    pub const fn from_preset(preset: CompressionPreset) -> Self {
        match preset {
            CompressionPreset::Off => Self {
                preset,
                threshold_db: 0.0,
                ratio: 1.0,
                knee_db: 0.0,
                makeup_db: 0.0,
            },
            CompressionPreset::Night => Self {
                preset,
                threshold_db: -24.0,
                ratio: 3.0,
                knee_db: 8.0,
                makeup_db: 3.0,
            },
            CompressionPreset::HardOfHearing => Self {
                preset,
                threshold_db: -30.0,
                ratio: 5.0,
                knee_db: 10.0,
                makeup_db: 6.0,
            },
        }
    }

    /// Returns the preset this profile was built from.
    #[inline]
    #[must_use]
    pub const fn preset(&self) -> CompressionPreset {
        self.preset
    }

    /// Returns the threshold in dBFS.
    #[inline]
    #[must_use]
    pub const fn threshold_db(&self) -> Sample {
        self.threshold_db
    }

    /// Returns the compression ratio.
    #[inline]
    #[must_use]
    pub const fn ratio(&self) -> Sample {
        self.ratio
    }

    /// Returns the soft-knee width in dB.
    #[inline]
    #[must_use]
    pub const fn knee_db(&self) -> Sample {
        self.knee_db
    }

    /// Returns the make-up gain in dB.
    #[inline]
    #[must_use]
    pub const fn makeup_db(&self) -> Sample {
        self.makeup_db
    }

    /// Returns `true` when the profile will actually reduce dynamic range.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.ratio > 1.0
    }

    /// Applies the profile onto a base [`CompressorParams`].
    ///
    /// The threshold, ratio, knee, and make-up gain are replaced with the
    /// profile values; all other fields (attack, release, detection mode, mix)
    /// are preserved from `base`.
    #[must_use]
    pub fn apply(&self, base: CompressorParams) -> CompressorParams {
        let mut out = base;
        out.threshold_db = self.threshold_db;
        out.ratio = self.ratio.max(1.0);
        out.knee_db = self.knee_db.max(0.0);
        out.makeup_db = self.makeup_db;
        out
    }
}

impl Default for CompressionProfile {
    fn default() -> Self {
        Self::from_preset(CompressionPreset::Off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1e-6;

    #[test]
    fn off_is_inactive() {
        let p = CompressionProfile::from_preset(CompressionPreset::Off);
        assert!(!p.is_active());
        assert!((p.ratio() - 1.0).abs() < EPS);
    }

    #[test]
    fn stronger_preset_has_higher_ratio_and_makeup() {
        let night = CompressionProfile::from_preset(CompressionPreset::Night);
        let hoh = CompressionProfile::from_preset(CompressionPreset::HardOfHearing);
        assert!(night.is_active());
        assert!(hoh.is_active());
        assert!(hoh.ratio() > night.ratio());
        assert!(hoh.makeup_db() > night.makeup_db());
        assert!(hoh.threshold_db() < night.threshold_db());
    }

    #[test]
    fn apply_overrides_window_but_keeps_timing() {
        let base = CompressorParams {
            attack_ms: 7.0,
            release_ms: 333.0,
            ..CompressorParams::default()
        };
        let out = CompressionProfile::from_preset(CompressionPreset::Night).apply(base);
        assert!((out.attack_ms - 7.0).abs() < EPS);
        assert!((out.release_ms - 333.0).abs() < EPS);
        assert!((out.threshold_db - (-24.0)).abs() < EPS);
        assert!((out.ratio - 3.0).abs() < EPS);
        assert!((out.knee_db - 8.0).abs() < EPS);
        assert!((out.makeup_db - 3.0).abs() < EPS);
    }

    #[test]
    fn off_apply_is_unity_ratio() {
        let out = CompressionProfile::default().apply(CompressorParams::default());
        assert!((out.ratio - 1.0).abs() < EPS);
    }
}
