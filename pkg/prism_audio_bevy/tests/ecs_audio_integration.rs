//! AAA-grade whole-chain integration tests for the `prism_audio_bevy` ECS
//! audio front-end.
//!
//! These tests drive the crate's public control plane exactly as a game would:
//! they build a real [`bevy_app::App`], install [`AudioRuntimePlugin`], spawn
//! entities carrying the public components ([`AudioPlayer`],
//! [`PlaybackSettings`], [`AudioEmitter`], [`AudioListener`]), advance the real
//! [`bevy_ecs`] schedule with `App::update`, then pump the parked real-time
//! runtime off-thread through [`AudioRuntimeHost`]. Assertions reach all the way
//! into the runtime's authoritative [`VoicePool`](prism_audio_core::voice) and
//! the published [`TelemetryFrame`](prism_audio_rt::TelemetryFrame), so every
//! case exercises the genuine component to system to command-ring to DSP path
//! rather than any stub. Coverage spans the end-to-end data flow, the
//! client-side handle-prediction invariant, deterministic distance attenuation,
//! single-frame system scheduling order, physical-voice budgeting, master-gain
//! application through a compiled graph, telemetry history, ring-full retry, and
//! entity-disposition teardown.
//!
//! # Provenance
//!
//! Original test code. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code, and no code
//! derived from any such source. No AI/ML. Public standards (the decibel
//! mapping and inverse-style distance fade) informed only ideas, never copied
//! text or data.
//!
//! # Relationship
//!
//! Validates the integration layer documented in the crate root (the
//! client/control-plane half of engine design section 21). Depends on
//! [`bevy_app`], [`bevy_ecs`], [`bevy_math`], and [`bevy_transform`] for the
//! real ECS/app/pose plumbing, on [`prism_audio_core`] for the voice pool,
//! buffer, and graph primitives, and on [`prism_audio_rt`] for the runtime,
//! command ring, and telemetry ring the front-end bridges onto.

use bevy_app::App;
use bevy_math::Vec3;
use bevy_transform::components::GlobalTransform;

use prism_audio_bevy::prelude::*;
use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::graph::{AudioGraph, AudioNode, PortRef, ProcessIo, RenderContext};
use prism_audio_core::voice::{VoiceGroup, VoiceHandle};
use prism_audio_rt::{AudioRuntimeConfig, TelemetryFrame};

/// Approximate-equality tolerance used in place of exact float comparison.
const EPS: f32 = 1.0e-4;

/// Branch-free absolute value, avoiding `f32::abs` so the suite stays clear of
/// the forbidden std float-math surface.
#[inline]
fn fabs(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}

/// Whether two floats agree within [`EPS`].
#[inline]
fn close(a: f32, b: f32) -> bool {
    fabs(a - b) < EPS
}

/// A small, fully deterministic runtime configuration with tiny pools and
/// rings so budgeting and ring-full paths are easy to provoke.
fn base_config() -> AudioRuntimeConfig {
    AudioRuntimeConfig {
        sample_rate: 48_000,
        max_block: 64,
        command_capacity: 64,
        telemetry_capacity: 16,
        retire_capacity: 8,
        voice_capacity: 16,
        max_physical_voices: 8,
    }
}

/// Builds an [`App`] with only [`AudioRuntimePlugin`] installed (which already
/// schedules the full bridge, including the player systems).
fn app_with(config: AudioRuntimeConfig) -> App {
    let mut app = App::new();
    app.add_plugins(AudioRuntimePlugin::new(config));
    app
}

/// Renders exactly one offline block through the parked runtime and returns the
/// telemetry frame the audio thread would have published.
fn pump(app: &mut App, out: &mut AudioBuffer) -> TelemetryFrame {
    app.world_mut()
        .non_send_mut::<AudioRuntimeHost>()
        .pump_block(out)
}

/// Number of allocated (physical + virtual) voices held by the real runtime
/// pool, read straight from the audio-thread half.
fn runtime_active(app: &mut App) -> usize {
    app.world_mut()
        .non_send_mut::<AudioRuntimeHost>()
        .runtime()
        .voices()
        .active_count()
}

/// Number of audible (physical) voices held by the real runtime pool.
fn runtime_physical(app: &mut App) -> usize {
    app.world_mut()
        .non_send_mut::<AudioRuntimeHost>()
        .runtime()
        .voices()
        .physical_count()
}

/// Importance the runtime pool actually recorded for `handle`, i.e. the value
/// that survived the command ring into the DSP side.
fn runtime_importance(app: &mut App, handle: VoiceHandle) -> f32 {
    app.world_mut()
        .non_send_mut::<AudioRuntimeHost>()
        .runtime()
        .voices()
        .get(handle)
        .expect("handle should resolve in the runtime pool")
        .importance
}

/// A source node emitting a constant value on its single mono output.
struct Dc(f32);

impl AudioNode for Dc {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let channels = out.channels();
        for channel in 0..channels {
            for sample in out.channel_mut(channel) {
                *sample = self.0;
            }
        }
    }
}

/// A unity pass-through copying input 0 to output 0.
struct Passthrough;

impl AudioNode for Passthrough {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        output.copy_from(input);
    }
}

/// Publishes a `DC(1.0) -> passthrough -> master` graph so master-gain scaling
/// shows up as a measurable peak in telemetry.
fn publish_unity_dc_graph(app: &App) {
    let mut graph = AudioGraph::new(48_000, 64);
    let src = graph.add_node(Box::new(Dc(1.0)), Vec::new(), vec![ChannelLayout::Mono]);
    let pass = graph.add_node(
        Box::new(Passthrough),
        vec![ChannelLayout::Mono],
        vec![ChannelLayout::Mono],
    );
    graph
        .connect(PortRef::new(src, 0), PortRef::new(pass, 0))
        .expect("connect source to passthrough");
    graph
        .set_master(PortRef::new(pass, 0))
        .expect("set master port");
    graph.compile().expect("compile graph");
    app.world()
        .resource::<AudioClient>()
        .publish_graph(Box::new(graph));
}

/// End-to-end: a high-level [`AudioPlayer`] plus [`PlaybackSettings`] is
/// projected to an [`AudioEmitter`], spawned onto the runtime, and the handle
/// written back resolves inside the real pool, all in the public API surface.
#[test]
fn player_to_runtime_voice_round_trip() {
    let mut app = app_with(base_config());
    app.world_mut()
        .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));
    let entity = app
        .world_mut()
        .spawn((
            AudioPlayer::new(VoiceGroup(4)),
            PlaybackSettings::ONCE.with_volume(Volume::Linear(0.75)),
            GlobalTransform::from_translation(Vec3::ZERO),
        ))
        .id();

    app.update();

    let emitter = *app
        .world()
        .get::<AudioEmitter>(entity)
        .expect("sync should derive an emitter");
    assert_eq!(emitter.group, VoiceGroup(4), "group should propagate");
    let handle = emitter.voice.expect("a voice handle should be predicted");

    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    let frame = pump(&mut app, &mut out);

    assert_eq!(runtime_active(&mut app), 1, "runtime should hold one voice");
    assert_eq!(frame.physical_voices, 1, "telemetry should report one voice");
    assert!(
        close(runtime_importance(&mut app, handle), 0.75),
        "runtime importance should equal the authored volume",
    );
}

/// The whole chain is bit-for-bit deterministic: two independently built apps
/// running the identical scenario mint the identical [`VoiceHandle`] and
/// publish byte-identical telemetry.
#[test]
fn whole_chain_is_bit_identical_across_runs() {
    fn run() -> (VoiceHandle, u32, u32, u32) {
        let mut app = app_with(base_config());
        app.world_mut()
            .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));
        let entity = app
            .world_mut()
            .spawn((
                AudioEmitter::new(VoiceGroup(1), 0.5),
                GlobalTransform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
            ))
            .id();
        app.update();
        let handle = app
            .world()
            .get::<AudioEmitter>(entity)
            .expect("emitter present")
            .voice
            .expect("voice predicted");
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
        let frame = pump(&mut app, &mut out);
        (
            handle,
            frame.physical_voices,
            frame.master_peak.to_bits(),
            frame.master_rms.to_bits(),
        )
    }

    let first = run();
    let second = run();
    assert_eq!(
        (first.0.index(), first.0.generation()),
        (second.0.index(), second.0.generation()),
        "predicted handle must be identical across runs",
    );
    assert_eq!(first.1, second.1, "physical-voice counts must match");
    assert_eq!(first.2, second.2, "master peak bits must match");
    assert_eq!(first.3, second.3, "master rms bits must match");
}

/// Distance attenuation reaches the DSP side deterministically: the importance
/// the runtime recorded equals the hand-computed linear fade, and repeating the
/// run reproduces the exact same bits.
#[test]
fn distance_attenuation_reaches_runtime_deterministically() {
    fn importance_for(distance: f32) -> (f32, u32) {
        let mut app = app_with(base_config());
        app.world_mut()
            .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));
        let entity = app
            .world_mut()
            .spawn((
                AudioEmitter::new(VoiceGroup(0), 1.0).with_distance(1.0, 5.0),
                GlobalTransform::from_translation(Vec3::new(distance, 0.0, 0.0)),
            ))
            .id();
        app.update();
        let handle = app
            .world()
            .get::<AudioEmitter>(entity)
            .expect("emitter present")
            .voice
            .expect("voice predicted");
        // Drain the spawn command so the runtime pool actually allocates.
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
        pump(&mut app, &mut out);
        let importance = runtime_importance(&mut app, handle);
        (importance, importance.to_bits())
    }

    // Distance 3 with reference 1 and max 5: t = (3-1)/(5-1) = 0.5, so a base of
    // 1.0 fades to 0.5.
    let (value, bits) = importance_for(3.0);
    assert!(close(value, 0.5), "faded importance was {value}");

    let (repeat_value, repeat_bits) = importance_for(3.0);
    assert_eq!(bits, repeat_bits, "distance fade must be bit-deterministic");
    assert!(close(repeat_value, 0.5));
}

/// An emitter at or beyond its maximum distance is driven to exact silence, and
/// the zero reaches the runtime pool as a bit-exact `0.0`.
#[test]
fn beyond_max_distance_is_exact_silence() {
    let mut app = app_with(base_config());
    app.world_mut()
        .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));
    let entity = app
        .world_mut()
        .spawn((
            AudioEmitter::new(VoiceGroup(0), 1.0).with_distance(1.0, 5.0),
            GlobalTransform::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        ))
        .id();

    app.update();

    let emitter = *app.world().get::<AudioEmitter>(entity).expect("emitter");
    let handle = emitter.voice.expect("voice predicted even when silent");
    assert_eq!(
        emitter.sent_importance.map(f32::to_bits),
        Some(0.0_f32.to_bits()),
        "sent importance must be exact zero",
    );
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    pump(&mut app, &mut out);
    assert_eq!(
        runtime_importance(&mut app, handle).to_bits(),
        0.0_f32.to_bits(),
        "runtime importance must be exact zero",
    );
}

/// A non-spatial player ignores listener distance entirely: even with the
/// listener far away, the full authored volume reaches the runtime.
#[test]
fn non_spatial_player_ignores_listener_distance() {
    let mut app = app_with(base_config());
    app.world_mut().spawn((
        AudioListener,
        GlobalTransform::from_translation(Vec3::new(1_000.0, 0.0, 0.0)),
    ));
    let entity = app
        .world_mut()
        .spawn((
            AudioPlayer::new(VoiceGroup(2)),
            PlaybackSettings::ONCE
                .with_volume(Volume::Linear(0.6))
                .non_spatial(),
            GlobalTransform::from_translation(Vec3::ZERO),
        ))
        .id();

    app.update();

    let handle = app
        .world()
        .get::<AudioEmitter>(entity)
        .expect("emitter")
        .voice
        .expect("voice predicted");
    let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
    pump(&mut app, &mut out);
    assert!(
        close(runtime_importance(&mut app, handle), 0.6),
        "non-spatial importance should equal the authored volume",
    );
}

/// Single-frame scheduling invariant: because `sync_audio_players` is chained
/// ahead of `spawn_voices`, a freshly spawned [`AudioPlayer`] is derived into an
/// emitter and allocated on the runtime within the very same `update`, with no
/// extra frame of latency.
#[test]
fn player_spawns_voice_within_one_frame() {
    let mut app = app_with(base_config());
    let entity = app
        .world_mut()
        .spawn((
            AudioPlayer::new(VoiceGroup(0)),
            PlaybackSettings::ONCE,
            GlobalTransform::from_translation(Vec3::ZERO),
        ))
        .id();

    app.update();

    // The single-frame invariant: deriving the emitter and predicting its voice
    // both complete within one update because the player and spawn systems are
    // chained in that order.
    assert!(
        app.world()
            .get::<AudioEmitter>(entity)
            .and_then(|emitter| emitter.voice)
            .is_some(),
        "emitter and its voice must exist after a single frame",
    );

    // Draining the enqueued spawn then realizes the voice in the runtime pool.
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    pump(&mut app, &mut out);
    assert_eq!(
        runtime_active(&mut app),
        1,
        "runtime should hold the voice once the spawn is drained",
    );
}

/// The client-side [`VoiceMirror`] stays in lock-step with the runtime pool: its
/// predicted handles resolve in the real pool and its active count equals the
/// runtime's across several concurrent emitters.
#[test]
fn mirror_prediction_matches_runtime_pool() {
    let mut app = app_with(base_config());
    app.world_mut()
        .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));
    let mut handles = Vec::new();
    for index in 0..4u32 {
        let entity = app
            .world_mut()
            .spawn((
                AudioEmitter::new(VoiceGroup(index), 1.0),
                GlobalTransform::from_translation(Vec3::new(index as f32, 0.0, 0.0)),
            ))
            .id();
        handles.push(entity);
    }

    app.update();
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    pump(&mut app, &mut out);

    for &entity in &handles {
        let handle = app
            .world()
            .get::<AudioEmitter>(entity)
            .expect("emitter")
            .voice
            .expect("voice predicted");
        assert!(
            app.world_mut()
                .non_send_mut::<AudioRuntimeHost>()
                .runtime()
                .voices()
                .get(handle)
                .is_some(),
            "every predicted handle must resolve in the runtime pool",
        );
    }

    let mirror_active = app.world().resource::<VoiceMirror>().active_count();
    assert_eq!(
        mirror_active,
        runtime_active(&mut app),
        "mirror active count must equal the runtime pool",
    );
    assert_eq!(mirror_active, 4, "all four emitters should be live");
}

/// Lowering the physical-voice budget forces the runtime to keep only the
/// budgeted number audible while the rest become virtual, and telemetry agrees.
#[test]
fn physical_budget_caps_audible_voices() {
    let config = AudioRuntimeConfig {
        max_physical_voices: 2,
        ..base_config()
    };
    let mut app = app_with(config);
    app.world_mut().insert_resource(PhysicalVoiceBudget::new(2));
    app.world_mut()
        .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));

    for index in 0..5u32 {
        // Distinct importances so the pool has an unambiguous ranking.
        let importance = 0.1 + (index as f32) * 0.1;
        app.world_mut().spawn((
            AudioEmitter::new(VoiceGroup(index), importance),
            GlobalTransform::from_translation(Vec3::ZERO),
        ));
    }

    app.update();
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    let frame = pump(&mut app, &mut out);

    assert_eq!(runtime_active(&mut app), 5, "all five voices should allocate");
    assert_eq!(
        runtime_physical(&mut app),
        2,
        "only the budgeted number may be audible",
    );
    assert_eq!(
        frame.physical_voices, 2,
        "telemetry physical count must honor the budget",
    );
    assert_eq!(
        frame.virtual_voices, 3,
        "the remaining voices must be virtual",
    );
}

/// A master-gain change made through the [`MasterGain`] resource is forwarded to
/// the runtime and audibly scales the rendered block, observable as the master
/// peak reported in telemetry.
#[test]
fn master_gain_change_scales_rendered_output() {
    let mut app = app_with(base_config());
    publish_unity_dc_graph(&app);

    app.update();
    let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
    let unity = pump(&mut app, &mut out);
    assert!(
        close(unity.master_peak, 1.0),
        "unity peak was {}",
        unity.master_peak,
    );

    app.world_mut().resource_mut::<MasterGain>().linear = 0.25;
    app.update();
    let scaled = pump(&mut app, &mut out);
    assert!(
        close(scaled.master_peak, 0.25),
        "scaled peak was {}",
        scaled.master_peak,
    );
}

/// The telemetry pump drains every published frame into the [`AudioTelemetry`]
/// resource, advancing the received counter and bounding the history ring.
#[test]
fn telemetry_history_is_bounded_and_counted() {
    let mut app = app_with(base_config());
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);

    // Pump several blocks, each followed by an update that drains the ring.
    let blocks = 5u64;
    for _ in 0..blocks {
        pump(&mut app, &mut out);
        app.update();
    }

    let telemetry = app.world().resource::<AudioTelemetry>();
    assert!(telemetry.has_data(), "telemetry should hold data");
    assert!(
        telemetry.received() >= blocks,
        "received counter should advance with every pumped block",
    );
    assert!(
        telemetry.history().len() <= telemetry.capacity(),
        "history must never exceed its capacity",
    );
    assert_eq!(
        telemetry.latest().frames,
        64,
        "the latest frame should report the rendered block size",
    );
}

/// A full command ring never drops work: with a one-slot ring and more emitters
/// than it can admit in one frame, the backlog is retried on subsequent frames
/// until every emitter owns a live voice.
#[test]
fn full_command_ring_retries_until_drained() {
    let config = AudioRuntimeConfig {
        command_capacity: 1,
        ..base_config()
    };
    let mut app = app_with(config);
    app.world_mut()
        .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));

    let emitters = 4usize;
    let mut entities = Vec::new();
    for index in 0..emitters {
        let entity = app
            .world_mut()
            .spawn((
                AudioEmitter::new(VoiceGroup(index as u32), 1.0),
                GlobalTransform::from_translation(Vec3::ZERO),
            ))
            .id();
        entities.push(entity);
    }

    // One update can only admit a single spawn through the one-slot ring.
    app.update();
    let admitted_first = entities
        .iter()
        .filter(|&&entity| {
            app.world()
                .get::<AudioEmitter>(entity)
                .and_then(|emitter| emitter.voice)
                .is_some()
        })
        .count();
    assert!(
        admitted_first < emitters,
        "a one-slot ring must not admit every spawn in one frame (got {admitted_first})",
    );

    // Pump + update several times; each pump drains the ring so the next update
    // can admit another backlogged spawn.
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    for _ in 0..(emitters * 2) {
        pump(&mut app, &mut out);
        app.update();
    }

    let admitted_final = entities
        .iter()
        .filter(|&&entity| {
            app.world()
                .get::<AudioEmitter>(entity)
                .and_then(|emitter| emitter.voice)
                .is_some()
        })
        .count();
    assert_eq!(
        admitted_final, emitters,
        "every emitter must eventually acquire a voice despite the full ring",
    );
    assert_eq!(
        runtime_active(&mut app),
        emitters,
        "the runtime should ultimately hold one voice per emitter",
    );
}

/// Removing an [`AudioEmitter`] tears the voice down through the removal path:
/// the runtime frees the slot and the [`LiveVoices`] registry empties.
#[test]
fn emitter_removal_stops_voice() {
    let mut app = app_with(base_config());
    let entity = app
        .world_mut()
        .spawn((
            AudioEmitter::new(VoiceGroup(0), 1.0),
            GlobalTransform::from_translation(Vec3::ZERO),
        ))
        .id();

    app.update();
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    pump(&mut app, &mut out);
    assert_eq!(runtime_active(&mut app), 1, "voice should be live");

    app.world_mut().entity_mut(entity).remove::<AudioEmitter>();
    app.update();
    pump(&mut app, &mut out);

    assert_eq!(runtime_active(&mut app), 0, "removal must free the voice");
    assert!(
        app.world().resource::<LiveVoices>().is_empty(),
        "live-voice registry must be empty after removal",
    );
}

/// The despawn disposition tears the entity down and stops its voice: after the
/// player requests a stop, the entity is gone and the runtime pool is empty.
#[test]
fn despawn_disposition_tears_down_voice_and_entity() {
    let mut app = app_with(base_config());

    let entity = app
        .world_mut()
        .spawn((
            AudioPlayer::new(VoiceGroup(0)),
            PlaybackSettings::DESPAWN,
            GlobalTransform::from_translation(Vec3::ZERO),
        ))
        .id();

    app.update();
    let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
    pump(&mut app, &mut out);
    assert_eq!(runtime_active(&mut app), 1, "voice should be live first");

    app.world_mut()
        .get_mut::<AudioPlayer>(entity)
        .expect("player present")
        .stop();

    // Frame one: the disposition system despawns the entity (removing its
    // emitter) at the end of the chain.
    app.update();
    assert!(
        app.world().get_entity(entity).is_err(),
        "the entity should be despawned",
    );

    // Frame two: the removal is observed and the stop command enqueued; pumping
    // then frees the slot in the runtime pool.
    app.update();
    pump(&mut app, &mut out);
    assert_eq!(
        runtime_active(&mut app),
        0,
        "the despawned entity's voice must be freed",
    );
}
