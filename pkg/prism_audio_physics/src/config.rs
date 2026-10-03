//! Tunable configuration for the contact-audio translator.
//!
//! [`TranslatorConfig`] gathers the knobs that steer the physics-to-audio
//! translation: the restitution used to estimate impulse, the near-coincident
//! [`prism_audio_procedural::contact::MergeConfig`], the per-block impact budget
//! and far-field [`crate::cluster::ClusterConfig`] that bound the voice count,
//! and the fallbacks for surface roughness and body extent used when the
//! caller supplies no better value. The [`Default`] gives a sane,
//! voice-budget-friendly starting point.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Parameterises design section 47.1: consumed by
//! [`crate::translator::ContactAudioTranslator`] to drive impulse estimation,
//! merging, clustering, and budgeting.

use prism_audio_procedural::contact::MergeConfig;
use prism_audio_core::math::Sample;

use crate::cluster::ClusterConfig;

/// Top-level configuration of the contact-audio translator.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TranslatorConfig {
    /// Coefficient of restitution used to estimate collision impulse `[0, 1]`.
    pub restitution: Sample,
    /// Near-coincident impact merge window.
    pub merge: MergeConfig,
    /// Hard cap on impacts emitted per block after merge and cluster.
    pub max_impacts_per_block: usize,
    /// Far-field clustering parameters.
    pub cluster: ClusterConfig,
    /// Roughness used when a material pair offers no distinguishing value.
    pub default_roughness: Sample,
    /// Body extent used to normalise the strike position when unknown.
    pub default_body_extent: Sample,
}

impl Default for TranslatorConfig {
    #[inline]
    fn default() -> Self {
        Self {
            // Mildly bouncy: most everyday contacts are partly inelastic.
            restitution: 0.3,
            merge: MergeConfig::default(),
            // A comfortable per-block contact-voice budget.
            max_impacts_per_block: 32,
            cluster: ClusterConfig::default(),
            // Mid roughness reads as a generic surface.
            default_roughness: 0.5,
            // A ~1 m characteristic body size.
            default_body_extent: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = TranslatorConfig::default();
        assert!((0.0..=1.0).contains(&c.restitution));
        assert!((0.0..=1.0).contains(&c.default_roughness));
        assert!(c.max_impacts_per_block > 0);
        assert!(c.default_body_extent > 0.0);
        assert!(c.merge.window_samples > 0);
        assert!(c.cluster.cluster_radius > 0.0);
    }
}
