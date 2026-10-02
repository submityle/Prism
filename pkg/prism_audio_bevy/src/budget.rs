//! The [`PhysicalVoiceBudget`] resource: the desired maximum number of
//! simultaneously audible voices, translated into
//! [`SetMaxPhysicalVoices`](prism_audio_rt::AudioCommand::SetMaxPhysicalVoices)
//! commands when it changes.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Change-detected by the budget system, which forwards it to the runtime and
//! mirrors it onto the client-side [`VoiceMirror`](crate::voice_registry::VoiceMirror).

use bevy_ecs::resource::Resource;

/// Desired physical-voice budget (maximum number of audible voices).
///
/// Mutating this resource enqueues one
/// [`SetMaxPhysicalVoices`](prism_audio_rt::AudioCommand::SetMaxPhysicalVoices)
/// and re-enforces the same budget on the client-side mirror pool so predicted
/// handles stay exact.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalVoiceBudget {
    /// Maximum number of simultaneously audible (physical) voices.
    pub max: usize,
}

impl PhysicalVoiceBudget {
    /// Builds a budget capping audible voices at `max`.
    #[must_use]
    #[inline]
    pub fn new(max: usize) -> Self {
        Self { max }
    }
}
