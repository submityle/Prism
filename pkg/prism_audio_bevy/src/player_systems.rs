//! The ECS systems that project [`AudioPlayer`](crate::player::AudioPlayer)
//! components onto the lower-level [`AudioEmitter`](crate::emitter::AudioEmitter)
//! pipeline and apply end-of-playback disposition.
//!
//! [`sync_audio_players`] keeps each playing entity's derived
//! [`AudioEmitter`](crate::emitter::AudioEmitter) in step with its
//! [`AudioPlayer`](crate::player::AudioPlayer) and
//! [`PlaybackSettings`](crate::playback::PlaybackSettings): it inserts the
//! emitter the first time, then re-applies the tunable fields whenever either
//! source component changes, without disturbing the runtime bookkeeping the
//! emitter carries. [`apply_player_disposition`] consumes an
//! [`AudioPlayer::stop_requested`](crate::player::AudioPlayer::stop_requested)
//! and resolves the [`PlaybackMode`](crate::playback::PlaybackMode): despawn the
//! entity, remove the playback components, or simply stop the voice. In every
//! case the existing [`stop_removed_voices`](crate::systems::stop_removed_voices)
//! and [`stop_flagged_voices`](crate::systems::stop_flagged_voices) systems do
//! the actual voice teardown, so there is a single stop path.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Bridges [`crate::player`] onto [`crate::emitter`] and the
//! [`crate::systems`] command pipeline; emits structural changes through
//! [`bevy_ecs`] [`Commands`].

use bevy_ecs::prelude::{Changed, Commands, Entity, Or, Query};

use crate::emitter::AudioEmitter;
use crate::playback::{PlaybackMode, PlaybackSettings};
use crate::player::{apply_to_emitter, derive_emitter, AudioPlayer};

/// Inserts or updates the [`AudioEmitter`] derived from each changed
/// [`AudioPlayer`]/[`PlaybackSettings`] pair.
///
/// Entities missing a [`PlaybackSettings`] use [`PlaybackSettings::default`]
/// (play once at unity volume). An entity that already carries an emitter has
/// its tunable fields re-applied in place; one that does not has a freshly
/// derived emitter inserted so the next
/// [`spawn_voices`](crate::systems::spawn_voices) pass can allocate its voice.
pub fn sync_audio_players(
    mut commands: Commands,
    mut players: Query<
        (
            Entity,
            &AudioPlayer,
            Option<&PlaybackSettings>,
            Option<&mut AudioEmitter>,
        ),
        Or<(Changed<AudioPlayer>, Changed<PlaybackSettings>)>,
    >,
) {
    for (entity, player, settings, emitter) in &mut players {
        if player.stop_requested {
            continue;
        }
        let settings = settings.copied().unwrap_or_default();
        match emitter {
            Some(mut emitter) => apply_to_emitter(player, &settings, &mut emitter),
            None => {
                commands
                    .entity(entity)
                    .insert(derive_emitter(player, &settings));
            }
        }
    }
}

/// Resolves the disposition of every player that has requested a stop.
///
/// [`Despawn`](PlaybackMode::Despawn) despawns the entity;
/// [`Remove`](PlaybackMode::Remove) strips the playback components;
/// [`Once`](PlaybackMode::Once) and [`Loop`](PlaybackMode::Loop) flag the
/// emitter to stop and clear the request. The emitter teardown itself is left
/// to the shared stop systems, which observe either the removed
/// [`AudioEmitter`] or the raised stop flag.
pub fn apply_player_disposition(
    mut commands: Commands,
    mut players: Query<(
        Entity,
        &mut AudioPlayer,
        Option<&PlaybackSettings>,
        Option<&mut AudioEmitter>,
    )>,
) {
    for (entity, mut player, settings, emitter) in &mut players {
        if !player.stop_requested {
            continue;
        }
        let mode = settings.map(|settings| settings.mode).unwrap_or_default();
        match mode {
            PlaybackMode::Despawn => {
                commands.entity(entity).despawn();
            }
            PlaybackMode::Remove => {
                commands
                    .entity(entity)
                    .remove::<(AudioPlayer, PlaybackSettings, AudioEmitter)>();
            }
            PlaybackMode::Once | PlaybackMode::Loop => {
                if let Some(mut emitter) = emitter {
                    emitter.stop_requested = true;
                }
                player.stop_requested = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bevy_app::{App, Update};
    use bevy_ecs::schedule::IntoScheduleConfigs;
    use bevy_math::Vec3;
    use bevy_transform::components::GlobalTransform;
    use prism_audio_core::voice::VoiceGroup;

    use crate::plugin::AudioRuntimePlugin;
    use crate::volume::Volume;

    /// Minimal runtime configuration for deterministic headless tests.
    fn test_config() -> prism_audio_rt::AudioRuntimeConfig {
        prism_audio_rt::AudioRuntimeConfig {
            sample_rate: 48_000,
            max_block: 64,
            command_capacity: 64,
            telemetry_capacity: 16,
            retire_capacity: 8,
            voice_capacity: 16,
            max_physical_voices: 8,
        }
    }

    /// Builds an app with the audio plugin and the player systems scheduled.
    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(AudioRuntimePlugin::new(test_config()));
        app.add_systems(Update, (sync_audio_players, apply_player_disposition).chain());
        app
    }

    #[test]
    fn sync_inserts_emitter_from_player() {
        let mut app = test_app();
        let entity = app
            .world_mut()
            .spawn((
                AudioPlayer::new(VoiceGroup(1)),
                PlaybackSettings::LOOP.with_volume(Volume::Linear(0.5)),
                GlobalTransform::from_translation(Vec3::ZERO),
            ))
            .id();

        app.update();

        let emitter = app
            .world()
            .get::<AudioEmitter>(entity)
            .expect("emitter should be derived");
        assert_eq!(emitter.group, VoiceGroup(1));
        assert!((emitter.importance - 0.5).abs() < 1.0e-4);
    }

    #[test]
    fn changing_settings_updates_emitter_in_place() {
        let mut app = test_app();
        let entity = app
            .world_mut()
            .spawn((
                AudioPlayer::new(VoiceGroup(0)),
                PlaybackSettings::ONCE,
                GlobalTransform::from_translation(Vec3::ZERO),
            ))
            .id();
        app.update();

        app.world_mut()
            .get_mut::<PlaybackSettings>(entity)
            .expect("settings present")
            .volume = Volume::Linear(0.25);
        app.update();

        let emitter = app.world().get::<AudioEmitter>(entity).expect("emitter present");
        assert!((emitter.importance - 0.25).abs() < 1.0e-4);
    }

    #[test]
    fn despawn_disposition_removes_entity() {
        let mut app = test_app();
        let entity = app
            .world_mut()
            .spawn((
                AudioPlayer::new(VoiceGroup(0)),
                PlaybackSettings::DESPAWN,
                GlobalTransform::from_translation(Vec3::ZERO),
            ))
            .id();
        app.update();
        assert!(app.world().get::<AudioEmitter>(entity).is_some());

        app.world_mut()
            .get_mut::<AudioPlayer>(entity)
            .expect("player present")
            .stop();
        app.update();

        assert!(app.world().get_entity(entity).is_err(), "entity should be despawned");
    }

    #[test]
    fn remove_disposition_strips_components() {
        let mut app = test_app();
        let entity = app
            .world_mut()
            .spawn((
                AudioPlayer::new(VoiceGroup(0)),
                PlaybackSettings::REMOVE,
                GlobalTransform::from_translation(Vec3::ZERO),
            ))
            .id();
        app.update();

        app.world_mut()
            .get_mut::<AudioPlayer>(entity)
            .expect("player present")
            .stop();
        app.update();

        assert!(app.world().get::<AudioPlayer>(entity).is_none());
        assert!(app.world().get::<PlaybackSettings>(entity).is_none());
        assert!(app.world().get::<AudioEmitter>(entity).is_none());
        assert!(app.world().get_entity(entity).is_ok(), "entity itself should remain");
    }

    #[test]
    fn once_stop_flags_emitter_and_clears_request() {
        let mut app = test_app();
        let entity = app
            .world_mut()
            .spawn((
                AudioPlayer::new(VoiceGroup(0)),
                PlaybackSettings::ONCE,
                GlobalTransform::from_translation(Vec3::ZERO),
            ))
            .id();
        app.update();

        app.world_mut()
            .get_mut::<AudioPlayer>(entity)
            .expect("player present")
            .stop();
        app.update();

        assert!(!app.world().get::<AudioPlayer>(entity).expect("player present").stop_requested);
        assert!(app.world().get::<AudioEmitter>(entity).is_some(), "entity and emitter remain");
    }
}
