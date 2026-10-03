//! Default transition timing and curve for the snapshot mixer.
//!
//! When a caller requests a transition without specifying a duration or an
//! interpolation curve, the [`SnapshotMixer`](crate::mixer::SnapshotMixer)
//! falls back to the values in [`SnapshotConfig`]. The defaults are a short
//! half-second move along a symmetric S-curve, which reads as a smooth,
//! natural mix change rather than a linear ramp or an instant jump.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Supplies defaults consumed by [`crate::mixer::SnapshotMixer`]. The curve
//! type is `prism_audio_content::curve::Interpolation`.

use prism_audio_content::curve::Interpolation;

/// Tunable defaults applied when a transition omits a duration or curve.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotConfig {
    /// Default transition duration, in seconds, used when none is given.
    pub default_transition_secs: f32,
    /// Default interpolation curve used when none is given.
    pub default_interpolation: Interpolation,
}

impl SnapshotConfig {
    /// Builds a config from an explicit duration and curve.
    #[must_use]
    pub const fn new(default_transition_secs: f32, default_interpolation: Interpolation) -> Self {
        Self { default_transition_secs, default_interpolation }
    }
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self { default_transition_secs: 0.5, default_interpolation: Interpolation::SCurve }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn default_is_half_second_scurve() {
        let cfg = SnapshotConfig::default();
        assert!((cfg.default_transition_secs - 0.5).abs() < EPS);
        assert_eq!(cfg.default_interpolation, Interpolation::SCurve);
    }

    #[test]
    fn new_sets_fields() {
        let cfg = SnapshotConfig::new(1.25, Interpolation::Linear);
        assert!((cfg.default_transition_secs - 1.25).abs() < EPS);
        assert_eq!(cfg.default_interpolation, Interpolation::Linear);
    }
}
