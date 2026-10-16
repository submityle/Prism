//! Adapter from scene-owned audio descriptors to the native ECS audio route.
//!
//! [`prism_scene::AudioSourceNode`] is intentionally data-only. This module is
//! the explicit runtime boundary: it resolves the stable bank/entry identity
//! into an [`AudioPlayer`] and [`PlaybackSettings`]. Asset loading and graph
//! publication remain separate steps, so an unresolved or streaming asset
//! cannot become a fake successful voice.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_audio_assets::asset::AudioAssetHandle;
use prism_audio_assets::bank::{BankId, EntryId};
use prism_audio_core::voice::VoiceGroup;
use prism_ecs::prelude::{Component, Entity, Resource, World};
use prism_scene::AudioSourceNode;

use crate::playback::PlaybackSettings;
use crate::player::AudioPlayer;
use crate::volume::Volume;

/// Marks playback controls that were materialized from a scene descriptor.
///
/// The marker is deliberately separate from [`AudioPlayer`]: a scene may be
/// despawned while an application-owned player on another entity is still
/// alive. Only controls carrying this marker are reclaimed by the adapter.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct SceneAudioBinding;

#[derive(Resource, Default)]
struct SceneAudioMaterializationState {
    descriptors: BTreeMap<Entity, AudioSourceNode>,
}

/// Materializes scene source descriptors into the audio ECS authoring facade.
///
/// The adapter is idempotent. Changing the bank/entry restarts the route from
/// frame zero; changing play, loop, or gain updates only the corresponding
/// control field. No decoder or device work occurs here.
pub fn materialize_scene_audio_sources(world: &mut World) {
    world.init_resource::<SceneAudioMaterializationState>();
    let sources: Vec<_> = world
        .query::<(Entity, &AudioSourceNode)>()
        .iter(world)
        .map(|(entity, source)| (entity, *source))
        .collect();

    // Scene instances can disappear through despawn, partition eviction, or a
    // structural override. Reconcile the previous descriptor set before
    // materializing the current one so the old route is removed on this frame
    // rather than leaving an orphaned player alive indefinitely.
    let current: BTreeMap<_, _> = sources.iter().copied().collect();
    let stale: Vec<_> = world
        .resource::<SceneAudioMaterializationState>()
        .descriptors
        .iter()
        .filter_map(|(&entity, _)| (!current.contains_key(&entity)).then_some(entity))
        .collect();
    for entity in stale {
        world
            .resource_mut::<SceneAudioMaterializationState>()
            .descriptors
            .remove(&entity);
        if world.remove::<SceneAudioBinding>(entity) {
            world.remove::<AudioPlayer>(entity);
            world.remove::<PlaybackSettings>(entity);
            world.remove::<crate::emitter::AudioEmitter>(entity);
            world.remove::<crate::player_systems::DerivedEmitter>(entity);
        }
    }

    for (entity, source) in sources {
        let changed = world
            .resource::<SceneAudioMaterializationState>()
            .descriptors
            .get(&entity)
            .copied()
            != Some(source);
        if !changed && world.get::<AudioPlayer>(entity).is_some() {
            continue;
        }
        world
            .resource_mut::<SceneAudioMaterializationState>()
            .descriptors
            .insert(entity, source);
        let handle = AudioAssetHandle::new(BankId(source.bank_id), EntryId(source.entry_id));
        let mode = if source.looping {
            crate::playback::PlaybackMode::Loop
        } else {
            crate::playback::PlaybackMode::Once
        };
        let settings = PlaybackSettings::with_mode(mode)
            .non_spatial()
            .with_volume(Volume::Linear(source.gain()));

        world.insert(entity, SceneAudioBinding);
        if world.get::<AudioPlayer>(entity).is_some() {
            {
                let mut player = world.get_mut::<AudioPlayer>(entity).expect("player");
                if player.asset != Some(handle) {
                    *player = player.with_asset(handle);
                }
                player.playing = source.playing;
                player.stop_requested = false;
            }
            if let Some(mut current) = world.get_mut::<PlaybackSettings>(entity) {
                *current = settings;
            } else {
                world.insert(entity, settings);
            }
        } else {
            let mut player = AudioPlayer::from_asset(handle, VoiceGroup(0));
            player.playing = source.playing;
            world.insert(entity, (player, settings));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_app::prelude::App;
    use prism_audio_core::voice::VoiceGroup;
    use prism_scene::AudioSourceNode;

    #[test]
    fn scene_descriptor_materializes_as_real_player_controls() {
        let mut app = App::new();
        let entity = app.world_mut().spawn(
            AudioSourceNode::new(7, 11)
                .with_looping(true)
                .with_playing(false)
                .with_gain(0.25),
        );
        materialize_scene_audio_sources(app.world_mut());
        let player = app.world().get::<AudioPlayer>(entity).expect("player");
        assert_eq!(
            player.asset,
            Some(AudioAssetHandle::new(BankId(7), EntryId(11)))
        );
        assert_eq!(player.group, VoiceGroup(0));
        assert!(!player.playing);
        assert_eq!(
            app.world()
                .get::<PlaybackSettings>(entity)
                .expect("settings")
                .mode,
            crate::playback::PlaybackMode::Loop
        );
        assert_eq!(
            app.world()
                .get::<PlaybackSettings>(entity)
                .expect("settings")
                .volume,
            Volume::Linear(0.25)
        );
    }
}
