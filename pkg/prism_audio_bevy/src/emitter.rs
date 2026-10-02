//! The [`AudioEmitter`] component: an entity's desired voice and the handle of
//! its live voice, if one has been spawned on the runtime.
//!
//! An emitter declares *what* it wants to play (group, priority, base
//! importance, virtual behavior) and *where* it is heard from (its
//! [`GlobalTransform`](bevy_transform::components::GlobalTransform) relative to
//! the [`AudioListener`](crate::listener::AudioListener)). The spawn/stop/
//! importance systems read these fields, translate them into
//! [`AudioCommand`](prism_audio_rt::AudioCommand)s, and write the resulting
//! [`VoiceHandle`] back into [`AudioEmitter::voice`].
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Consumed by [`crate::systems`] and [`crate::commands`]; carries the
//! [`VoiceHandle`] mirrored from the runtime's
//! [`VoicePool`](prism_audio_core::voice::VoicePool).

use bevy_ecs::component::Component;
use prism_audio_core::voice::{Importance, VirtualBehavior, VoiceGroup, VoiceHandle};

/// Default reference distance, in world units, within which an emitter plays at
/// its full base importance.
const DEFAULT_REFERENCE_DISTANCE: f32 = 1.0;

/// Default maximum distance, in world units, beyond which an emitter's
/// attenuated importance reaches zero.
const DEFAULT_MAX_DISTANCE: f32 = 100.0;

/// A positional audio source attached to an entity.
///
/// The entity's
/// [`GlobalTransform`](bevy_transform::components::GlobalTransform) supplies the
/// emitter pose; the systems combine it with the listener pose to derive the
/// effective [`Importance`] forwarded to the runtime. The live voice, once the
/// runtime has allocated it, is mirrored into [`AudioEmitter::voice`].
///
/// Mutating any desired field through a `Query` marks the component changed so
/// the importance system can re-evaluate it; the systems never clear fields the
/// user owns, only the runtime-managed [`AudioEmitter::voice`] and
/// [`AudioEmitter::sent_importance`] bookkeeping.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct AudioEmitter {
    /// Group used by the runtime for per-source playback limiting.
    pub group: VoiceGroup,
    /// Caller priority, retained for tie-breaking and telemetry.
    pub priority: u8,
    /// Base (un-attenuated) importance at or inside
    /// [`AudioEmitter::reference_distance`]. Combined with distance to form the
    /// effective importance sent to the runtime.
    pub importance: Importance,
    /// Distance, in world units, within which the emitter plays at its full
    /// base importance.
    pub reference_distance: f32,
    /// Distance, in world units, at and beyond which the emitter's attenuated
    /// importance reaches zero.
    pub max_distance: f32,
    /// Behavior applied by the runtime if the voice is later culled.
    pub behavior: VirtualBehavior,
    /// Whether the emitter wants to be playing. Clearing it stops the live
    /// voice on the next stop pass.
    pub playing: bool,
    /// Explicit one-shot stop request. Set it to stop the live voice without
    /// clearing [`AudioEmitter::playing`]; the stop system clears it once the
    /// stop command is accepted.
    pub stop_requested: bool,
    /// Handle of the live voice mirrored from the runtime, or `None` when no
    /// voice is currently allocated for this emitter.
    pub voice: Option<VoiceHandle>,
    /// Last effective importance successfully forwarded to the runtime, used to
    /// suppress redundant [`SetVoiceImportance`](prism_audio_rt::AudioCommand::SetVoiceImportance)
    /// commands. `None` until the first update succeeds.
    pub sent_importance: Option<Importance>,
}

impl AudioEmitter {
    /// Builds an emitter in `group` with the given base `importance`, playing by
    /// default with [`ContinueVirtual`](VirtualBehavior::ContinueVirtual)
    /// behavior and the default distance band.
    #[must_use]
    #[inline]
    pub fn new(group: VoiceGroup, importance: Importance) -> Self {
        Self {
            group,
            priority: 0,
            importance,
            reference_distance: DEFAULT_REFERENCE_DISTANCE,
            max_distance: DEFAULT_MAX_DISTANCE,
            behavior: VirtualBehavior::ContinueVirtual,
            playing: true,
            stop_requested: false,
            voice: None,
            sent_importance: None,
        }
    }

    /// Sets the caller priority.
    #[must_use]
    #[inline]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the cull behavior.
    #[must_use]
    #[inline]
    pub fn with_behavior(mut self, behavior: VirtualBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Sets the reference and maximum distances defining the attenuation band.
    #[must_use]
    #[inline]
    pub fn with_distance(mut self, reference_distance: f32, max_distance: f32) -> Self {
        self.reference_distance = reference_distance;
        self.max_distance = max_distance;
        self
    }

    /// Sets whether the emitter wants to be playing.
    #[must_use]
    #[inline]
    pub fn playing(mut self, playing: bool) -> Self {
        self.playing = playing;
        self
    }

    /// Whether the emitter wants a voice spawned right now: it is playing, has
    /// no pending stop, and has no live voice yet.
    #[must_use]
    #[inline]
    pub fn wants_voice(&self) -> bool {
        self.playing && !self.stop_requested && self.voice.is_none()
    }

    /// Whether the live voice should be stopped: a voice exists and the emitter
    /// has either requested a stop or no longer wants to play.
    #[must_use]
    #[inline]
    pub fn wants_stop(&self) -> bool {
        self.voice.is_some() && (self.stop_requested || !self.playing)
    }
}
