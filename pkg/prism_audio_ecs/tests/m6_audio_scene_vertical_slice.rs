//! M6 scene/audio joint vertical slice.
//!
//! This test intentionally crosses the real scene transaction, bank decoder,
//! scene adapter, resident graph route, offline render callback, playhead
//! bridge, and scene teardown. It does not use a fake source or a synthetic
//! success state.

extern crate alloc;

use alloc::boxed::Box;
use alloc::string::String;

use prism_app::App;
use prism_asset::StableGuid;
use prism_audio_assets::asset::AudioAssetHandle;
use prism_audio_assets::bank::{BankId, BankManifest, EntryId, MediaEntry, MediaFormat, Residency};
use prism_audio_assets::codec::CodecTag;
use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_ecs::prelude::*;
use prism_audio_rt::AudioRuntimeConfig;
use prism_scene::prelude::{
    despawn_instance, spawn_scene, ArchetypeGroup, ArchetypeSignature, InstantiationPlan,
    LocalEntityId, ResidencyTier, SceneTemplate, TypedSpawnPlan,
};

fn config() -> AudioRuntimeConfig {
    AudioRuntimeConfig {
        sample_rate: 48_000,
        max_block: 64,
        command_capacity: 32,
        telemetry_capacity: 8,
        retire_capacity: 8,
        voice_capacity: 8,
        max_physical_voices: 4,
    }
}

fn wav_mono(samples: &[i16]) -> Vec<u8> {
    let mut data = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        data.extend_from_slice(&sample.to_le_bytes());
    }
    let mut wav = Vec::with_capacity(44 + data.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&48_000u32.to_le_bytes());
    wav.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);
    wav
}

fn template() -> SceneTemplate {
    SceneTemplate::new(
        StableGuid::from_u128(0x6d36_6175_6469_6f01),
        InstantiationPlan {
            entity_count: 1,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::default(),
                Box::new([LocalEntityId::new(0)]),
                Box::new([]),
            )]),
            entity_valued_fields: Box::new([]),
            hierarchy: Box::new([]),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        },
    )
}

fn load_bank(app: &mut App, bank: u32, entry: u32, samples: &[i16]) -> AudioAssetHandle {
    let blob = wav_mono(samples);
    let manifest =
        BankManifest::new(BankId(bank), String::from("m6-scene"), 1).with_media(MediaEntry::new(
            EntryId(entry),
            String::from("scene-source"),
            MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(samples.len() as u64)),
            Residency::Memory,
            0,
            blob.len() as u64,
        ));
    let handle = AudioAssetHandle::new(BankId(bank), EntryId(entry));
    app.world_mut()
        .resource_mut::<AudioAssetStore>()
        .load_bank(manifest, &blob)
        .expect("decoder-verified test bank should load");
    handle
}

fn pump(app: &mut App, frames: usize) -> AudioBuffer {
    let mut output = AudioBuffer::new(ChannelLayout::Stereo, frames);
    app.world_mut()
        .get_non_send_mut::<AudioRuntimeHost>()
        .expect("runtime host")
        .pump_block(&mut output);
    output
}

fn all_zero(output: &AudioBuffer) -> bool {
    output
        .channel(0)
        .iter()
        .chain(output.channel(1).iter())
        .all(|sample| *sample == 0.0)
}

#[test]
fn scene_audio_import_route_playhead_and_teardown_are_one_real_chain() {
    let mut app = App::new();
    app.add_plugins(AudioRuntimePlugin::new(config()));
    let handle = AudioAssetHandle::new(BankId(41), EntryId(7));

    let scene = template();
    let spawn = spawn_scene(
        app.world_mut(),
        &scene,
        TypedSpawnPlan::new().with_audio_source(
            LocalEntityId::new(0),
            prism_scene::AudioSourceNode::new(handle.bank.0, handle.entry.0)
                .with_looping(false)
                .with_playing(true)
                .with_gain(0.5),
        ),
        None,
        ResidencyTier::Interactive,
    )
    .expect("scene transaction should materialize the source descriptor");
    let root = spawn.root.expect("scene root");

    // Scene creation is independent from asset publication: the adapter must
    // materialize a real player, but the route must remain explicitly waiting
    // instead of allocating a fake voice while the bank is absent.
    app.update();
    let waiting = app.world().get::<AudioPlayer>(root).expect("scene player");
    assert_eq!(waiting.asset, Some(handle));
    assert_eq!(waiting.state, AudioPlaybackState::WaitingForAsset);
    assert!(all_zero(&pump(&mut app, 2)));

    // The normal resource path is a decoder-verified BankRegistry load. Once
    // it publishes the resident PCM, the existing scene player is routed into
    // the real graph without recreating the scene.
    load_bank(&mut app, 41, 7, &[16_384, 16_384, 16_384, 16_384]);
    app.update();
    let first = pump(&mut app, 2);
    assert!(
        first.channel(0)[0] > 0.0,
        "resident decoded PCM must reach the published graph"
    );

    app.update();
    let player = app
        .world()
        .get::<AudioPlayer>(root)
        .expect("scene adapter must create AudioPlayer");
    assert_eq!(player.asset, Some(handle));
    assert!(
        player.playhead > 0,
        "render progress must bind back to the player"
    );
    assert_eq!(player.state, AudioPlaybackState::Ready);

    assert_eq!(despawn_instance(app.world_mut(), &spawn), 1);
    app.update();
    assert!(app.world().get::<AudioPlayer>(root).is_none());
    let after_despawn = pump(&mut app, 2);
    assert!(
        all_zero(&after_despawn),
        "scene teardown must clear the resident route instead of leaving an orphan voice"
    );
    assert!(
        app.world()
            .resource::<AudioGraphRouteState>()
            .last_error()
            .is_none(),
        "teardown must not leave a stale route publication error"
    );
}

#[test]
fn unloading_and_reloading_scene_media_has_explicit_lifecycle_states() {
    let mut app = App::new();
    app.add_plugins(AudioRuntimePlugin::new(config()));
    let handle = load_bank(&mut app, 42, 8, &[16_384, 16_384, 16_384]);
    let scene = template();
    let spawn = spawn_scene(
        app.world_mut(),
        &scene,
        TypedSpawnPlan::new().with_audio_source(
            LocalEntityId::new(0),
            prism_scene::AudioSourceNode::new(handle.bank.0, handle.entry.0),
        ),
        None,
        ResidencyTier::Interactive,
    )
    .expect("scene should spawn");
    let root = spawn.root.expect("scene root");

    app.update();
    let playing = pump(&mut app, 1);
    assert!(playing.channel(0)[0] > 0.0);

    app.world_mut()
        .resource_mut::<AudioAssetStore>()
        .unload_bank(BankId(42))
        .expect("loaded bank should unload");
    app.update();
    let player = app
        .world()
        .get::<AudioPlayer>(root)
        .expect("player remains");
    assert_eq!(player.state, AudioPlaybackState::WaitingForAsset);
    assert!(all_zero(&pump(&mut app, 1)));

    let reloaded = load_bank(&mut app, 42, 8, &[8_192, 8_192, 8_192]);
    assert_eq!(reloaded, handle);
    app.update();
    assert_eq!(
        app.world().get::<AudioPlayer>(root).expect("player").state,
        AudioPlaybackState::Ready
    );
    assert!(pump(&mut app, 1).channel(0)[0] > 0.0);

    assert_eq!(despawn_instance(app.world_mut(), &spawn), 1);
    app.update();
    assert!(all_zero(&pump(&mut app, 1)));
}
