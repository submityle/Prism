//! The [`AudioRuntimePlugin`]: constructs a runtime, installs the client and
//! bookkeeping resources, and schedules the bridge systems.
//!
//! The plugin owns the one-time wiring. It calls [`prism_audio_rt::runtime`],
//! keeps the clonable client in the [`AudioClient`] resource, parks the audio
//! thread half in the non-`Send` [`AudioRuntimeHost`], sizes the client-side
//! [`VoiceMirror`] to match, and chains the [`crate::systems`] in a fixed order
//! inside the [`AudioSystems`] set on [`Update`].
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; original ECS glue. No AI/ML.
//!
//! # Relationship
//!
//! Wires [`bevy_app`] startup to `prism_audio_rt` and registers the components,
//! resources, and systems defined across this crate.

use bevy_app::{App, Plugin, Update};
use bevy_ecs::schedule::IntoScheduleConfigs;
use prism_audio_rt::AudioRuntimeConfig;

use crate::budget::PhysicalVoiceBudget;
use crate::client_resource::{AudioClient, AudioRuntimeHost};
use crate::emitter::AudioEmitter;
use crate::listener::AudioListener;
use crate::master_gain::MasterGain;
use crate::playback::PlaybackSettings;
use crate::player::AudioPlayer;
use crate::player_systems::{apply_player_disposition, sync_audio_players};
use crate::systems::{
    apply_master_gain, apply_physical_budget, flush_pending_stops, pump_telemetry, spawn_voices,
    stop_flagged_voices, stop_removed_voices, update_importance, AudioSystems,
};
use crate::telemetry::AudioTelemetry;
use crate::voice_registry::{LiveVoices, PendingStops, VoiceMirror};

/// Bevy plugin wiring the ECS front-end to a freshly constructed audio runtime.
///
/// The [`AudioRuntimePlugin::config`] field is the single configuration knob;
/// it sizes the rings, the voice pool, and the physical-voice budget. Build the
/// plugin with [`AudioRuntimePlugin::new`] or rely on
/// [`AudioRuntimePlugin::default`] for the runtime defaults.
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioRuntimePlugin {
    /// Configuration passed verbatim to [`prism_audio_rt::runtime`].
    pub config: AudioRuntimeConfig,
}

impl AudioRuntimePlugin {
    /// Builds a plugin that will construct its runtime with `config`.
    #[must_use]
    #[inline]
    pub fn new(config: AudioRuntimeConfig) -> Self {
        Self { config }
    }
}

impl Plugin for AudioRuntimePlugin {
    fn build(&self, app: &mut App) {
        let config = self.config;
        let (runtime, client, collector) = prism_audio_rt::runtime(config);

        app.insert_resource(AudioClient::new(client));
        app.insert_non_send(AudioRuntimeHost::new(runtime, collector));
        app.insert_resource(VoiceMirror::new(
            config.voice_capacity,
            config.max_physical_voices,
        ));
        app.insert_resource(PhysicalVoiceBudget::new(config.max_physical_voices));
        app.init_resource::<LiveVoices>();
        app.init_resource::<PendingStops>();
        app.init_resource::<MasterGain>();
        app.init_resource::<AudioTelemetry>();

        {
            let world = app.world_mut();
            world.register_component::<AudioEmitter>();
            world.register_component::<AudioListener>();
            world.register_component::<AudioPlayer>();
            world.register_component::<PlaybackSettings>();
        }

        app.add_systems(
            Update,
            (
                sync_audio_players,
                flush_pending_stops,
                spawn_voices,
                update_importance,
                stop_flagged_voices,
                stop_removed_voices,
                apply_player_disposition,
                apply_master_gain,
                apply_physical_budget,
                pump_telemetry,
            )
                .chain()
                .in_set(AudioSystems),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::boxed::Box;
    use alloc::vec;

    use bevy_math::Vec3;
    use bevy_transform::components::GlobalTransform;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    use prism_audio_core::graph::{AudioGraph, AudioNode, PortRef, ProcessIo, RenderContext};
    use prism_audio_core::voice::VoiceGroup;

    /// Small deterministic configuration keeping the pools and rings tiny.
    fn test_config() -> AudioRuntimeConfig {
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

    /// Builds an `App` with the plugin installed.
    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(AudioRuntimePlugin::new(test_config()));
        app
    }

    /// Approximate float comparison used instead of exact equality.
    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-3
    }

    /// Renders one offline block and returns the published telemetry frame.
    fn pump(app: &mut App, out: &mut AudioBuffer) -> prism_audio_rt::TelemetryFrame {
        app.world_mut()
            .non_send_mut::<AudioRuntimeHost>()
            .pump_block(out)
    }

    /// Number of allocated voices currently held by the runtime pool.
    fn runtime_active(app: &mut App) -> usize {
        app.world_mut()
            .non_send_mut::<AudioRuntimeHost>()
            .runtime()
            .voices()
            .active_count()
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

    #[test]
    fn spawn_emits_spawn_voice() {
        let mut app = test_app();
        app.world_mut()
            .spawn((AudioListener, GlobalTransform::from_translation(Vec3::ZERO)));
        app.world_mut().spawn((
            AudioEmitter::new(VoiceGroup(0), 1.0),
            GlobalTransform::from_translation(Vec3::ZERO),
        ));

        app.update();

        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
        let frame = pump(&mut app, &mut out);

        assert_eq!(runtime_active(&mut app), 1, "runtime should allocate one voice");
        assert_eq!(frame.physical_voices, 1, "telemetry should report one physical voice");
    }

    #[test]
    fn removal_emits_stop_voice() {
        let mut app = test_app();
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
        assert_eq!(runtime_active(&mut app), 1, "voice should be live before removal");

        app.world_mut().entity_mut(entity).remove::<AudioEmitter>();
        app.update();
        pump(&mut app, &mut out);

        assert_eq!(runtime_active(&mut app), 0, "removal should stop the voice");
        assert!(
            app.world().resource::<LiveVoices>().is_empty(),
            "live-voice registry should be empty"
        );
    }

    #[test]
    fn explicit_stop_flag_emits_stop_voice() {
        let mut app = test_app();
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
        assert_eq!(runtime_active(&mut app), 1);

        app.world_mut()
            .get_mut::<AudioEmitter>(entity)
            .expect("emitter present")
            .stop_requested = true;
        app.update();
        pump(&mut app, &mut out);

        assert_eq!(runtime_active(&mut app), 0, "explicit stop should release the voice");
        let emitter = app.world().get::<AudioEmitter>(entity).expect("emitter present");
        assert!(emitter.voice.is_none(), "live handle should be cleared");
        assert!(!emitter.stop_requested, "stop flag should be consumed");
    }

    #[test]
    fn master_gain_change_emits_set_master_gain() {
        let mut app = test_app();

        {
            let mut graph = AudioGraph::new(48_000, 64);
            let src = graph.add_node(Box::new(Dc(1.0)), vec![], vec![ChannelLayout::Mono]);
            let pass = graph.add_node(
                Box::new(Passthrough),
                vec![ChannelLayout::Mono],
                vec![ChannelLayout::Mono],
            );
            graph
                .connect(PortRef::new(src, 0), PortRef::new(pass, 0))
                .expect("connect");
            graph.set_master(PortRef::new(pass, 0)).expect("set master");
            graph.compile().expect("compile");
            app.world()
                .resource::<AudioClient>()
                .publish_graph(Box::new(graph));
        }

        app.update();
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        let unity = pump(&mut app, &mut out);
        assert!(close(unity.master_peak, 1.0), "unity peak was {}", unity.master_peak);

        app.world_mut().resource_mut::<MasterGain>().linear = 0.25;
        app.update();
        let scaled = pump(&mut app, &mut out);
        assert!(close(scaled.master_peak, 0.25), "scaled peak was {}", scaled.master_peak);
    }

    #[test]
    fn telemetry_pump_populates_resource() {
        let mut app = test_app();

        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 64);
        pump(&mut app, &mut out);

        app.update();

        let telemetry = app.world().resource::<AudioTelemetry>();
        assert!(telemetry.has_data(), "telemetry resource should hold a frame");
        assert!(telemetry.received() >= 1, "received counter should advance");
        assert_eq!(telemetry.latest().frames, 64, "frame should report 64 rendered frames");
    }
}
