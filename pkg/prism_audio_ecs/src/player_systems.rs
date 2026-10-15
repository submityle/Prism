//! Systems that project high-level players onto [`AudioEmitter`].

use alloc::vec::Vec;
use prism_ecs::prelude::{Component, Entity, World};

use crate::assets::AudioAssetStore;
use crate::emitter::AudioEmitter;
use crate::playback::{PlaybackMode, PlaybackSettings};
use crate::player::{apply_to_emitter, derive_emitter, AudioPlaybackState, AudioPlayer};

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
    let asset_store = world.get_resource::<AudioAssetStore>();
    let players: Vec<_> = query
        .iter(world)
        .map(|(entity, player, settings)| {
            let resolved = player.asset.is_some_and(|handle| {
                asset_store.is_some_and(|store| store.resolve(handle).is_some())
            });
            (
                entity,
                *player,
                settings.copied().unwrap_or_default(),
                resolved,
            )
        })
        .collect();
    for (entity, mut player, settings, asset_resolved) in players {
        if player.stop_requested {
            continue;
        }
        if player.asset.is_some() {
            if asset_resolved {
                // A published resident route owns the `Ready` state. Keep it
                // across frames instead of resetting it before the route
                // cache gets a chance to run; all other resolved assets are
                // still waiting for route materialization.
                if player.state != AudioPlaybackState::Ready {
                    player.state = AudioPlaybackState::AwaitingSourceRoute;
                }
            } else {
                player.state = AudioPlaybackState::WaitingForAsset;
            }
            // The current runtime command contract carries policy, not decoded
            // media. Do not turn an asset into an abstract fake-success voice.
            if world.get::<DerivedEmitter>(entity).is_some()
                && let Some(mut emitter) = world.get_mut::<AudioEmitter>(entity)
            {
                emitter.playing = false;
                emitter.stop_requested = true;
            }
            *world
                .get_mut::<AudioPlayer>(entity)
                .expect("queried player") = player;
            continue;
        }
        player.state = AudioPlaybackState::Ready;
        if let Some(emitter) = world.get::<AudioEmitter>(entity).copied() {
            let mut desired = emitter;
            apply_to_emitter(&player, &settings, &mut desired);
            if desired != emitter {
                if world.get::<DerivedEmitter>(entity).is_some() {
                    desired.playing = player.playing;
                }
                *world
                    .get_mut::<AudioEmitter>(entity)
                    .expect("existing emitter") = desired;
            }
        } else {
            world.insert(entity, (derive_emitter(&player, &settings), DerivedEmitter));
        }
        *world
            .get_mut::<AudioPlayer>(entity)
            .expect("queried player") = player;
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
    use crate::client_resource::AudioRuntimeHost;
    use alloc::string::String;
    use alloc::vec::Vec;
    use prism_app::prelude::App;
    use prism_audio_assets::asset::AudioAssetHandle;
    use prism_audio_assets::bank::{
        BankId, BankManifest, EntryId, MediaEntry, MediaFormat, Residency,
    };
    use prism_audio_assets::codec::CodecTag;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
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

    #[test]
    fn asset_player_waits_without_creating_a_voice() {
        let mut app = test_app();
        let entity = app.world_mut().spawn(AudioPlayer::from_asset(
            AudioAssetHandle::new(BankId(7), EntryId(1)),
            VoiceGroup(0),
        ));
        app.update();
        let player = app.world().get::<AudioPlayer>(entity).expect("player");
        assert_eq!(player.state, AudioPlaybackState::WaitingForAsset);
        assert_eq!(player.playhead, 0);
        assert!(app.world().get::<AudioEmitter>(entity).is_none());
    }

    #[test]
    fn resolved_asset_waits_for_source_route_without_fake_voice() {
        let mut app = test_app();
        app.world_mut()
            .resource_mut::<AudioAssetStore>()
            .load_bank(test_streaming_manifest(), &wav_mono(4))
            .expect("bank loads");
        let entity = app.world_mut().spawn(AudioPlayer::from_asset(
            AudioAssetHandle::new(BankId(7), EntryId(1)),
            VoiceGroup(0),
        ));
        app.update();
        let player = app.world().get::<AudioPlayer>(entity).expect("player");
        assert_eq!(player.state, AudioPlaybackState::AwaitingSourceRoute);
        assert_eq!(player.playhead, 0);
        assert!(app.world().get::<AudioEmitter>(entity).is_none());
    }

    #[test]
    fn player_playhead_tracks_rendered_runtime_frames() {
        let mut app = test_app();
        let entity = app
            .world_mut()
            .spawn((AudioPlayer::new(VoiceGroup(0)), GlobalTransform::default()));
        app.update();
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 64);
        app.world_mut()
            .get_non_send_mut::<AudioRuntimeHost>()
            .expect("runtime host")
            .pump_block(&mut output);
        app.update();
        assert_eq!(
            app.world()
                .get::<AudioPlayer>(entity)
                .expect("player")
                .playhead,
            64
        );
    }

    fn test_streaming_manifest() -> BankManifest {
        BankManifest::new(BankId(7), String::from("test"), 1).with_media(MediaEntry::new(
            EntryId(1),
            String::from("clip"),
            MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(4)),
            Residency::Streaming { prefetch_frames: 2 },
            0,
            52,
        ))
    }

    fn wav_mono(frames: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            data.extend_from_slice(&(frame as i16).to_le_bytes());
        }
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&48_000u32.to_le_bytes());
        wav.extend_from_slice(&96_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        wav
    }
}
