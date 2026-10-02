//! The [`MasterGain`] resource: the desired master output gain, translated into
//! sample-accurate [`SetMasterGain`](prism_audio_rt::AudioCommand::SetMasterGain)
//! commands when it changes.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Change-detected by the master-gain system, which forwards it to the runtime.

use bevy_ecs::resource::Resource;

/// Desired master output gain for the runtime.
///
/// Mutating this resource (through `ResMut`) marks it changed, and the
/// master-gain system then enqueues one
/// [`SetMasterGain`](prism_audio_rt::AudioCommand::SetMasterGain) with
/// `at_frame = 0` and the configured `ramp_frames`.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct MasterGain {
    /// Linear gain multiplier (`1.0` is unity, `0.0` is silence). The runtime
    /// sanitizes non-finite or negative values to silence.
    pub linear: f32,
    /// Length, in frames, of the click-free ramp toward `linear`. `0` snaps
    /// immediately.
    pub ramp_frames: u32,
}

impl MasterGain {
    /// Builds a master gain that snaps immediately to `linear`.
    #[must_use]
    #[inline]
    pub fn new(linear: f32) -> Self {
        Self {
            linear,
            ramp_frames: 0,
        }
    }

    /// Sets the ramp length, in frames, used when the gain next changes.
    #[must_use]
    #[inline]
    pub fn with_ramp(mut self, ramp_frames: u32) -> Self {
        self.ramp_frames = ramp_frames;
        self
    }
}

impl Default for MasterGain {
    #[inline]
    fn default() -> Self {
        Self {
            linear: 1.0,
            ramp_frames: 0,
        }
    }
}
