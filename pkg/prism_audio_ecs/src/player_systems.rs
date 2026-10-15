//! Systems that project high-level players onto [`AudioEmitter`].

use prism_ecs::prelude::{Changed, Commands, Entity, Or, Query};

use crate::emitter::AudioEmitter;
use crate::playback::{PlaybackMode, PlaybackSettings};
use crate::player::{apply_to_emitter, derive_emitter, AudioPlayer};

/// Inserts or updates the derived emitter for changed player settings.
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

/// Applies a player's requested end-of-playback disposition.
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
            PlaybackMode::Despawn => commands.entity(entity).despawn(),
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
    use prism_app::prelude::{App, Update};
    use prism_audio_core::voice::VoiceGroup;
    use prism_ecs::prelude::IntoSystemConfigs;
    use prism_transform::GlobalTransform;

    use crate::plugin::AudioRuntimePlugin;
    use crate::volume::Volume;

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(AudioRuntimePlugin::default());
        app.add_systems(
            Update,
            (sync_audio_players, apply_player_disposition).chain(),
        );
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
