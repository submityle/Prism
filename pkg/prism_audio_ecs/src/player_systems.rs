//! Systems that project high-level players onto [`AudioEmitter`].

use alloc::vec::Vec;
use prism_ecs::prelude::{Component, Entity, World};

use crate::emitter::AudioEmitter;
use crate::playback::{PlaybackMode, PlaybackSettings};
use crate::player::{apply_to_emitter, derive_emitter, AudioPlayer};

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DerivedEmitter;

/// Projects players and settings (including settings removal) into emitters.
/// Removing a player removes only an emitter originally created by this bridge;
/// a pre-existing low-level emitter remains caller-owned.
pub fn sync_audio_players(world: &mut World) {
    let query = world.query::<(Entity, &DerivedEmitter, Option<&AudioPlayer>)>();
    let removed: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, _, player)| player.is_none().then_some(entity))
        .collect();
    for entity in removed {
        world.remove::<AudioEmitter>(entity);
        world.remove::<DerivedEmitter>(entity);
    }
    let query = world.query::<(Entity, &AudioPlayer, Option<&PlaybackSettings>)>();
    let players: Vec<_> = query
        .iter(world)
        .map(|(entity, player, settings)| (entity, *player, settings.copied().unwrap_or_default()))
        .collect();
    for (entity, player, settings) in players {
        if player.stop_requested {
            continue;
        }
        if let Some(emitter) = world.get::<AudioEmitter>(entity).copied() {
            let mut desired = emitter;
            apply_to_emitter(&player, &settings, &mut desired);
            if desired != emitter {
                *world
                    .get_mut::<AudioEmitter>(entity)
                    .expect("existing emitter") = desired;
            }
        } else {
            world.insert(entity, (derive_emitter(&player, &settings), DerivedEmitter));
        }
    }
}

/// Applies a player's explicitly requested stop disposition.
pub fn apply_player_disposition(world: &mut World) {
    let query = world.query::<(Entity, &AudioPlayer, Option<&PlaybackSettings>)>();
    let stopped: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, player, settings)| {
            player.stop_requested.then_some((
                entity,
                *player,
                settings.copied().unwrap_or_default(),
            ))
        })
        .collect();
    for (entity, player, settings) in stopped {
        let mode = settings.mode;
        match mode {
            PlaybackMode::Despawn => {
                world.despawn(entity);
            }
            PlaybackMode::Remove => {
                world.remove::<AudioPlayer>(entity);
                world.remove::<PlaybackSettings>(entity);
                world.remove::<AudioEmitter>(entity);
                world.remove::<DerivedEmitter>(entity);
            }
            PlaybackMode::Once | PlaybackMode::Loop => {
                if world.get::<AudioEmitter>(entity).is_none() {
                    world.insert(
                        entity,
                        (
                            derive_emitter(&player, &settings).playing(false),
                            DerivedEmitter,
                        ),
                    );
                }
                let mut emitter = world
                    .get_mut::<AudioEmitter>(entity)
                    .expect("emitter inserted");
                emitter.stop_requested = true;
                emitter.playing = false;
                world
                    .get_mut::<AudioPlayer>(entity)
                    .expect("queried player")
                    .stop_requested = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_app::prelude::App;
    use prism_audio_core::voice::VoiceGroup;
    use prism_transform::GlobalTransform;

    use crate::plugin::AudioRuntimePlugin;
    use crate::volume::Volume;

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(AudioRuntimePlugin::default());
        app
    }

    #[test]
    fn sync_inserts_and_updates_emitter() {
        let mut app = test_app();
        let entity = app.world_mut().spawn((
            AudioPlayer::new(VoiceGroup(1)),
            PlaybackSettings::LOOP.with_volume(Volume::Linear(0.5)),
            GlobalTransform::default(),
        ));
        app.update();
        assert_eq!(
            app.world()
                .get::<AudioEmitter>(entity)
                .expect("emitter")
                .group,
            VoiceGroup(1)
        );
        app.world_mut()
            .get_mut::<PlaybackSettings>(entity)
            .expect("settings")
            .volume = Volume::Linear(0.25);
        app.update();
        assert!(
            (app.world()
                .get::<AudioEmitter>(entity)
                .expect("emitter")
                .importance
                - 0.25)
                .abs()
                < 1.0e-4
        );
    }

    #[test]
    fn dispositions_are_applied() {
        let mut app = test_app();
        let remove = app
            .world_mut()
            .spawn((AudioPlayer::new(VoiceGroup(0)), PlaybackSettings::REMOVE));
        let despawn = app
            .world_mut()
            .spawn((AudioPlayer::new(VoiceGroup(0)), PlaybackSettings::DESPAWN));
        app.update();
        app.world_mut()
            .get_mut::<AudioPlayer>(remove)
            .expect("player")
            .stop();
        app.world_mut()
            .get_mut::<AudioPlayer>(despawn)
            .expect("player")
            .stop();
        app.update();
        assert!(app.world().get::<AudioPlayer>(remove).is_none());
        assert!(app.world().get_entity(despawn).is_err());
    }
}
