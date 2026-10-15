//! Compatibility-facade tests for the native Prism ECS audio bridge.
//!
//! The facade must expose the same public audio behavior while its runtime
//! implementation is owned by `prism_audio_ecs`. These tests intentionally use
//! only Prism-native app, ECS, math, and transform contracts.

use prism_app::App;
use prism_math::Vec3;
use prism_transform::{GlobalTransform, Transform};

use prism_audio_bevy::prelude::*;
use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::voice::{VoiceGroup, VoiceHandle};
use prism_audio_rt::{AudioRuntimeConfig, TelemetryFrame};

const EPS: f32 = 1.0e-4;

#[inline]
fn pose(position: Vec3) -> GlobalTransform {
    GlobalTransform::from(Transform::from_translation(position))
}

#[inline]
fn close(a: f32, b: f32) -> bool {
    let delta = a - b;
    (if delta < 0.0 { -delta } else { delta }) < EPS
}

fn config() -> AudioRuntimeConfig {
    AudioRuntimeConfig {
        sample_rate: 48_000,
        max_block: 64,
        command_capacity: 16,
        telemetry_capacity: 4,
        retire_capacity: 4,
        voice_capacity: 8,
        max_physical_voices: 4,
    }
}

fn app_with(config: AudioRuntimeConfig) -> App {
    let mut app = App::new();
    app.add_plugins(AudioRuntimePlugin::new(config));
    app
}

fn pump(app: &mut App, channels: ChannelLayout) -> TelemetryFrame {
    let mut output = AudioBuffer::new(channels, 64);
    app.world_mut()
        .get_non_send_mut::<AudioRuntimeHost>()
        .expect("audio runtime host")
        .pump_block(&mut output)
}

fn active_voices(app: &mut App) -> usize {
    app.world_mut()
        .get_non_send_mut::<AudioRuntimeHost>()
        .expect("audio runtime host")
        .runtime()
        .voices()
        .active_count()
}

fn importance(app: &mut App, handle: VoiceHandle) -> f32 {
    app.world_mut()
        .get_non_send_mut::<AudioRuntimeHost>()
        .expect("audio runtime host")
        .runtime()
        .voices()
        .get(handle)
        .expect("voice handle should resolve")
        .importance
}

#[test]
fn facade_projects_player_to_runtime_voice() {
    let mut app = app_with(config());
    app.world_mut().spawn((AudioListener, pose(Vec3::ZERO)));
    let entity = app.world_mut().spawn((
        AudioPlayer::new(VoiceGroup(7)),
        PlaybackSettings::ONCE.with_volume(Volume::Linear(0.75)),
        pose(Vec3::ZERO),
    ));

    app.update();

    let emitter = *app.world().get::<AudioEmitter>(entity).expect("emitter");
    assert_eq!(emitter.group, VoiceGroup(7));
    let handle = emitter.voice.expect("voice handle should be predicted");

    let frame = pump(&mut app, ChannelLayout::Stereo);
    assert_eq!(active_voices(&mut app), 1);
    assert_eq!(frame.physical_voices, 1);
    assert!(close(importance(&mut app, handle), 0.75));
}

#[test]
fn facade_preserves_spatial_and_non_spatial_behavior() {
    let mut spatial = app_with(config());
    spatial.world_mut().spawn((AudioListener, pose(Vec3::ZERO)));
    let spatial_entity = spatial.world_mut().spawn((
        AudioEmitter::new(VoiceGroup(0), 1.0).with_distance(1.0, 5.0),
        pose(Vec3::new(3.0, 0.0, 0.0)),
    ));
    spatial.update();
    let spatial_handle = spatial
        .world()
        .get::<AudioEmitter>(spatial_entity)
        .expect("spatial emitter")
        .voice
        .expect("spatial voice handle");
    pump(&mut spatial, ChannelLayout::Stereo);
    assert!(close(importance(&mut spatial, spatial_handle), 0.5));

    let mut non_spatial = app_with(config());
    non_spatial
        .world_mut()
        .spawn((AudioListener, pose(Vec3::new(1_000.0, 0.0, 0.0))));
    let non_spatial_entity = non_spatial.world_mut().spawn((
        AudioPlayer::new(VoiceGroup(1)),
        PlaybackSettings::ONCE
            .with_volume(Volume::Linear(0.6))
            .non_spatial(),
        pose(Vec3::ZERO),
    ));
    non_spatial.update();
    let non_spatial_handle = non_spatial
        .world()
        .get::<AudioEmitter>(non_spatial_entity)
        .expect("non-spatial emitter")
        .voice
        .expect("non-spatial voice handle");
    pump(&mut non_spatial, ChannelLayout::Mono);
    assert!(close(importance(&mut non_spatial, non_spatial_handle), 0.6));
}

#[test]
fn facade_removal_stops_runtime_voice() {
    let mut app = app_with(config());
    let entity = app
        .world_mut()
        .spawn((AudioEmitter::new(VoiceGroup(0), 1.0), pose(Vec3::ZERO)));

    app.update();
    pump(&mut app, ChannelLayout::Stereo);
    assert_eq!(active_voices(&mut app), 1);

    app.world_mut().remove::<AudioEmitter>(entity);
    app.update();
    pump(&mut app, ChannelLayout::Stereo);
    assert_eq!(active_voices(&mut app), 0);
    assert!(app.world().resource::<LiveVoices>().is_empty());
}

#[test]
fn facade_publishes_telemetry_through_native_resource() {
    let mut app = app_with(config());
    for _ in 0..3 {
        pump(&mut app, ChannelLayout::Stereo);
        app.update();
    }

    let telemetry = app.world().resource::<AudioTelemetry>();
    assert!(telemetry.has_data());
    assert!(telemetry.received() >= 3);
    assert!(telemetry.history().len() <= telemetry.capacity());
    assert_eq!(telemetry.latest().frames, 64);
}
