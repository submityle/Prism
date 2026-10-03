//! The [`AudioPlayer`] component and the pure functions that derive an
//! [`AudioEmitter`](crate::emitter::AudioEmitter) from an
//! [`AudioPlayer`] plus its [`PlaybackSettings`](crate::playback::PlaybackSettings).
//!
//! [`AudioPlayer`] is the high-level, component-shaped way to make an entity
//! sound: attach an [`AudioPlayer`] and (optionally) a
//! [`PlaybackSettings`](crate::playback::PlaybackSettings), and the
//! [`sync_audio_players`](crate::player_systems::sync_audio_players) system
//! derives and maintains the lower-level
//! [`AudioEmitter`](crate::emitter::AudioEmitter) that drives the runtime. The
//! translation here is pure and deterministic so it is trivially testable in
//! isolation from the ECS schedule.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Consumed by [`crate::player_systems`]; produces and updates
//! [`crate::emitter::AudioEmitter`] from
//! [`crate::playback::PlaybackSettings`] and [`crate::volume::Volume`].

use bevy_ecs::component::Component;
use prism_audio_core::voice::VoiceGroup;

use crate::emitter::AudioEmitter;
use crate::playback::PlaybackSettings;

/// Reference and maximum distance written to a derived emitter when its
/// playback is non-spatial, placing the listener permanently inside the
/// full-volume radius so no distance attenuation is applied.
const NON_SPATIAL_DISTANCE: f32 = f32::MAX;

/// A high-level audio source attached to an entity.
///
/// An [`AudioPlayer`] names *which* logical source plays (its
/// [`VoiceGroup`], used by the runtime for per-source playback limiting) and
/// its scheduling [`priority`](AudioPlayer::priority). Pair it with a
/// [`PlaybackSettings`](crate::playback::PlaybackSettings) to describe loudness,
/// looping, and spatialization; the sync system then owns the derived
/// [`AudioEmitter`](crate::emitter::AudioEmitter).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioPlayer {
    /// Logical source group, forwarded to the runtime for playback limiting.
    pub group: VoiceGroup,
    /// Scheduling priority used for tie-breaking and telemetry.
    pub priority: u8,
    /// When set, the player requests that its sound stop; the disposition
    /// system then stops the voice and applies the
    /// [`PlaybackMode`](crate::playback::PlaybackMode) disposition (despawn the
    /// entity, remove the playback components, or nothing).
    pub stop_requested: bool,
}

impl AudioPlayer {
    /// Builds a player for logical source `group` at priority zero.
    #[must_use]
    #[inline]
    pub const fn new(group: VoiceGroup) -> Self {
        Self {
            group,
            priority: 0,
            stop_requested: false,
        }
    }

    /// Sets the scheduling priority.
    #[must_use]
    #[inline]
    pub const fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Requests that this player stop and its disposition be applied.
    #[inline]
    pub fn stop(&mut self) {
        self.stop_requested = true;
    }
}

/// Builds a fresh [`AudioEmitter`] from a player and its settings.
///
/// The effective volume (silent while paused or muted) becomes the emitter's
/// base importance; the mode chooses the cull behavior; a spatial player keeps
/// its authored attenuation band while a non-spatial player is pinned to full
/// volume regardless of listener distance.
#[must_use]
pub fn derive_emitter(player: &AudioPlayer, settings: &PlaybackSettings) -> AudioEmitter {
    let importance = settings.effective_volume().to_linear();
    let mut emitter = AudioEmitter::new(player.group, importance)
        .with_priority(player.priority)
        .with_behavior(settings.mode.virtual_behavior());
    emitter = if settings.spatial {
        emitter.with_distance(settings.reference_distance, settings.max_distance)
    } else {
        emitter.with_distance(NON_SPATIAL_DISTANCE, NON_SPATIAL_DISTANCE)
    };
    emitter
}

/// Updates the tunable fields of an existing emitter in place from a player and
/// its settings, preserving the runtime-managed bookkeeping
/// ([`voice`](AudioEmitter::voice), [`sent_importance`](AudioEmitter::sent_importance),
/// [`playing`](AudioEmitter::playing), and
/// [`stop_requested`](AudioEmitter::stop_requested)).
pub fn apply_to_emitter(player: &AudioPlayer, settings: &PlaybackSettings, emitter: &mut AudioEmitter) {
    emitter.group = player.group;
    emitter.priority = player.priority;
    emitter.importance = settings.effective_volume().to_linear();
    emitter.behavior = settings.mode.virtual_behavior();
    if settings.spatial {
        emitter.reference_distance = settings.reference_distance;
        emitter.max_distance = settings.max_distance;
    } else {
        emitter.reference_distance = NON_SPATIAL_DISTANCE;
        emitter.max_distance = NON_SPATIAL_DISTANCE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use prism_audio_core::voice::VirtualBehavior;

    use crate::playback::PlaybackMode;
    use crate::volume::Volume;

    /// Epsilon used instead of exact float equality.
    const EPS: f32 = 1.0e-4;

    /// Approximate float comparison.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn derive_folds_volume_into_importance() {
        let player = AudioPlayer::new(VoiceGroup(3));
        let settings = PlaybackSettings::ONCE.with_volume(Volume::Linear(0.5));
        let emitter = derive_emitter(&player, &settings);
        assert!(close(emitter.importance, 0.5), "importance was {}", emitter.importance);
        assert_eq!(emitter.group, VoiceGroup(3));
    }

    #[test]
    fn derive_maps_loop_to_continue_virtual() {
        let emitter = derive_emitter(&AudioPlayer::new(VoiceGroup(0)), &PlaybackSettings::LOOP);
        assert_eq!(emitter.behavior, VirtualBehavior::ContinueVirtual);
    }

    #[test]
    fn paused_player_derives_silent_importance() {
        let emitter = derive_emitter(
            &AudioPlayer::new(VoiceGroup(0)),
            &PlaybackSettings::ONCE.paused(),
        );
        assert_eq!(emitter.importance, 0.0);
    }

    #[test]
    fn non_spatial_player_ignores_distance() {
        let emitter = derive_emitter(
            &AudioPlayer::new(VoiceGroup(0)),
            &PlaybackSettings::ONCE.non_spatial(),
        );
        assert_eq!(emitter.reference_distance, NON_SPATIAL_DISTANCE);
        assert_eq!(emitter.max_distance, NON_SPATIAL_DISTANCE);
    }

    #[test]
    fn priority_propagates() {
        let player = AudioPlayer::new(VoiceGroup(1)).with_priority(7);
        let emitter = derive_emitter(&player, &PlaybackSettings::ONCE);
        assert_eq!(emitter.priority, 7);
    }

    #[test]
    fn apply_preserves_runtime_bookkeeping() {
        let player = AudioPlayer::new(VoiceGroup(2));
        let mut emitter = derive_emitter(&player, &PlaybackSettings::ONCE);
        emitter.sent_importance = Some(0.9);
        emitter.playing = true;

        let changed = PlaybackSettings::LOOP.with_volume(Volume::Linear(0.25));
        apply_to_emitter(&player, &changed, &mut emitter);

        assert!(close(emitter.importance, 0.25));
        assert_eq!(emitter.behavior, VirtualBehavior::ContinueVirtual);
        assert_eq!(emitter.sent_importance, Some(0.9), "bookkeeping must survive");
        assert!(emitter.playing);
    }

    #[test]
    fn stop_sets_the_request_flag() {
        let mut player = AudioPlayer::new(VoiceGroup(0));
        assert!(!player.stop_requested);
        player.stop();
        assert!(player.stop_requested);
    }

    #[test]
    fn mode_is_carried_for_disposition() {
        let player = AudioPlayer::new(VoiceGroup(0));
        let _ = derive_emitter(&player, &PlaybackSettings::DESPAWN);
        assert!(PlaybackMode::Despawn.despawns_entity());
    }
}
