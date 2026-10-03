//! Aggregate accessibility profile composing every accommodation.
//!
//! [`AccessibilityProfile`] bundles the individual accommodations defined in
//! this crate into one applied configuration: whether captions and visual cues
//! are surfaced, whether the mono downmix is engaged, and the dialogue-boost and
//! compression settings. It exposes `apply`-style helpers that stamp the active
//! settings onto the relevant `prism_audio_core` parameter structs, and a
//! `compose` operation that merges two profiles by taking the stronger of each
//! accommodation.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Ties together the items of design section 23. The downmix feeds the output
//! stage of design section 48, the dialogue boost and compression feed the
//! dynamics stage of design section 13, and the caption / visual-cue toggles
//! gate what this crate reports to the telemetry ring of design section 26.

use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::nodes::dynamics::compressor::CompressorParams;
use prism_audio_core::nodes::dynamics::ducking::DuckingParams;

use crate::compression::{CompressionPreset, CompressionProfile};
use crate::dialogue::{BoostStrength, DialogueBoost};
use crate::downmix::{DownmixMatrix, MonoDownmix};

/// A composed set of accessibility accommodations ready to be applied.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AccessibilityProfile {
    /// Whether caption reports are surfaced to the UI.
    captions_enabled: bool,
    /// Whether visual sound cues are surfaced to the UI.
    visual_cues_enabled: bool,
    /// One-touch mono downmix accommodation.
    downmix: MonoDownmix,
    /// Dialogue-priority boost accommodation.
    dialogue_boost: DialogueBoost,
    /// Dynamic-range compression accommodation.
    compression: CompressionProfile,
}

impl AccessibilityProfile {
    /// Creates a profile with every accommodation disabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            captions_enabled: false,
            visual_cues_enabled: false,
            downmix: MonoDownmix::new(false),
            dialogue_boost: DialogueBoost::new(BoostStrength::Off),
            compression: CompressionProfile::from_preset(CompressionPreset::Off),
        }
    }

    /// Enables or disables caption reporting; returns the updated profile.
    #[must_use]
    pub fn with_captions(mut self, enabled: bool) -> Self {
        self.captions_enabled = enabled;
        self
    }

    /// Enables or disables visual cues; returns the updated profile.
    #[must_use]
    pub fn with_visual_cues(mut self, enabled: bool) -> Self {
        self.visual_cues_enabled = enabled;
        self
    }

    /// Sets the mono downmix accommodation; returns the updated profile.
    #[must_use]
    pub fn with_downmix(mut self, downmix: MonoDownmix) -> Self {
        self.downmix = downmix;
        self
    }

    /// Sets the dialogue-priority boost; returns the updated profile.
    #[must_use]
    pub fn with_dialogue_boost(mut self, dialogue_boost: DialogueBoost) -> Self {
        self.dialogue_boost = dialogue_boost;
        self
    }

    /// Sets the compression profile; returns the updated profile.
    #[must_use]
    pub fn with_compression(mut self, compression: CompressionProfile) -> Self {
        self.compression = compression;
        self
    }

    /// Returns whether caption reporting is enabled.
    #[inline]
    #[must_use]
    pub fn captions_enabled(&self) -> bool {
        self.captions_enabled
    }

    /// Returns whether visual cues are enabled.
    #[inline]
    #[must_use]
    pub fn visual_cues_enabled(&self) -> bool {
        self.visual_cues_enabled
    }

    /// Returns the mono downmix accommodation.
    #[inline]
    #[must_use]
    pub fn downmix(&self) -> MonoDownmix {
        self.downmix
    }

    /// Returns the dialogue-priority boost accommodation.
    #[inline]
    #[must_use]
    pub fn dialogue_boost(&self) -> DialogueBoost {
        self.dialogue_boost
    }

    /// Returns the compression accommodation.
    #[inline]
    #[must_use]
    pub fn compression(&self) -> CompressionProfile {
        self.compression
    }

    /// Returns `true` when at least one accommodation is active.
    #[must_use]
    pub fn is_any_active(&self) -> bool {
        self.captions_enabled
            || self.visual_cues_enabled
            || self.downmix.is_enabled()
            || self.dialogue_boost.is_active()
            || self.compression.is_active()
    }

    /// Returns the mono fold matrix when the downmix is enabled.
    #[must_use]
    pub fn downmix_matrix(&self, layout: ChannelLayout) -> Option<DownmixMatrix> {
        self.downmix.matrix(layout)
    }

    /// Applies the dialogue boost onto a base [`DuckingParams`].
    #[must_use]
    pub fn apply_ducking(&self, base: DuckingParams) -> DuckingParams {
        self.dialogue_boost.apply(base)
    }

    /// Applies the compression profile onto a base [`CompressorParams`].
    #[must_use]
    pub fn apply_compressor(&self, base: CompressorParams) -> CompressorParams {
        self.compression.apply(base)
    }

    /// Composes two profiles by taking the stronger of each accommodation.
    ///
    /// Boolean toggles are combined with logical OR; the dialogue boost and the
    /// compression profile take whichever operand has the higher intensity; and
    /// the downmix is enabled when either operand enables it.
    #[must_use]
    pub fn compose(&self, other: &AccessibilityProfile) -> AccessibilityProfile {
        let dialogue_strength = max_boost(
            self.dialogue_boost.strength(),
            other.dialogue_boost.strength(),
        );
        let compression = max_compression(self.compression, other.compression);
        Self {
            captions_enabled: self.captions_enabled || other.captions_enabled,
            visual_cues_enabled: self.visual_cues_enabled || other.visual_cues_enabled,
            downmix: MonoDownmix::new(self.downmix.is_enabled() || other.downmix.is_enabled()),
            dialogue_boost: DialogueBoost::new(dialogue_strength),
            compression,
        }
    }
}

impl Default for AccessibilityProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns the ordinal rank of a boost strength (higher is stronger).
#[inline]
#[must_use]
fn boost_rank(strength: BoostStrength) -> u8 {
    match strength {
        BoostStrength::Off => 0,
        BoostStrength::Low => 1,
        BoostStrength::Medium => 2,
        BoostStrength::High => 3,
    }
}

/// Returns the stronger of two boost strengths.
#[inline]
#[must_use]
fn max_boost(a: BoostStrength, b: BoostStrength) -> BoostStrength {
    if boost_rank(a) >= boost_rank(b) {
        a
    } else {
        b
    }
}

/// Returns the ordinal rank of a compression preset (higher is stronger).
#[inline]
#[must_use]
fn compression_rank(preset: CompressionPreset) -> u8 {
    match preset {
        CompressionPreset::Off => 0,
        CompressionPreset::Night => 1,
        CompressionPreset::HardOfHearing => 2,
    }
}

/// Returns the stronger of two compression profiles.
#[inline]
#[must_use]
fn max_compression(a: CompressionProfile, b: CompressionProfile) -> CompressionProfile {
    if compression_rank(a.preset()) >= compression_rank(b.preset()) {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_inactive() {
        let p = AccessibilityProfile::default();
        assert!(!p.is_any_active());
        assert!(p.downmix_matrix(ChannelLayout::Stereo).is_none());
    }

    #[test]
    fn builders_set_fields() {
        let p = AccessibilityProfile::new()
            .with_captions(true)
            .with_visual_cues(true)
            .with_downmix(MonoDownmix::new(true))
            .with_dialogue_boost(DialogueBoost::new(BoostStrength::Medium))
            .with_compression(CompressionProfile::from_preset(CompressionPreset::Night));
        assert!(p.captions_enabled());
        assert!(p.visual_cues_enabled());
        assert!(p.downmix().is_enabled());
        assert_eq!(p.dialogue_boost().strength(), BoostStrength::Medium);
        assert_eq!(p.compression().preset(), CompressionPreset::Night);
        assert!(p.is_any_active());
        assert!(p.downmix_matrix(ChannelLayout::Stereo).is_some());
    }

    #[test]
    fn apply_helpers_delegate() {
        let p = AccessibilityProfile::new()
            .with_dialogue_boost(DialogueBoost::new(BoostStrength::High))
            .with_compression(CompressionProfile::from_preset(CompressionPreset::HardOfHearing));
        let duck = p.apply_ducking(DuckingParams::default());
        assert!(duck.range_db > DuckingParams::default().range_db);
        let comp = p.apply_compressor(CompressorParams::default());
        assert!((comp.ratio - 5.0).abs() < 1e-6);
    }

    #[test]
    fn compose_takes_the_stronger() {
        let a = AccessibilityProfile::new()
            .with_captions(true)
            .with_dialogue_boost(DialogueBoost::new(BoostStrength::Low))
            .with_compression(CompressionProfile::from_preset(CompressionPreset::Night));
        let b = AccessibilityProfile::new()
            .with_visual_cues(true)
            .with_dialogue_boost(DialogueBoost::new(BoostStrength::High))
            .with_compression(CompressionProfile::from_preset(CompressionPreset::Off));
        let c = a.compose(&b);
        assert!(c.captions_enabled());
        assert!(c.visual_cues_enabled());
        assert_eq!(c.dialogue_boost().strength(), BoostStrength::High);
        assert_eq!(c.compression().preset(), CompressionPreset::Night);
    }
}
