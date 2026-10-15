//! Plugin wiring for the Prism ECS audio bridge.

use alloc::sync::Arc;
use core::sync::atomic::AtomicBool;

use prism_app::prelude::{App, Plugin, Update};
use prism_audio_rt::AudioRuntimeConfig;
use prism_ecs::prelude::{IntoScheduleConfigs, LifecycleEvent};

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
use crate::voice_registry::{LiveVoices, PendingStops, RemovedAudioEmitters, VoiceMirror};

/// Plugin that constructs and wires a [`prism_audio_rt`] runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioRuntimePlugin {
    /// Runtime ring, pool, and output configuration.
    pub config: AudioRuntimeConfig,
}

impl AudioRuntimePlugin {
    /// Builds a plugin using `config`.
    #[must_use]
    #[inline]
    pub const fn new(config: AudioRuntimeConfig) -> Self {
        Self { config }
    }
}

fn record_removed_emitter(ctx: &mut prism_ecs::prelude::ObserverContext<'_>) {
    if let Some(removed) = ctx.world.get_resource_mut::<RemovedAudioEmitters>() {
        removed.push(ctx.entity);
    }
}

impl Plugin for AudioRuntimePlugin {
    fn build(&self, app: &mut App) {
        let config = self.config;
        let (runtime, client, collector) = prism_audio_rt::runtime(config);
        let open = Arc::new(AtomicBool::new(true));

        app.insert_resource(AudioClient::new(client, Arc::clone(&open)))
            .insert_resource(VoiceMirror::new(
                config.voice_capacity,
                config.max_physical_voices,
            ))
            .insert_resource(PhysicalVoiceBudget::new(config.max_physical_voices))
            .init_resource::<LiveVoices>()
            .init_resource::<PendingStops>()
            .init_resource::<RemovedAudioEmitters>()
            .init_resource::<MasterGain>()
            .init_resource::<AudioTelemetry>();

        let world = app.world_mut();
        world.insert_non_send(AudioRuntimeHost::new(runtime, collector, open));
        world.register_component::<AudioEmitter>();
        world.register_component::<AudioListener>();
        world.register_component::<AudioPlayer>();
        world.register_component::<PlaybackSettings>();
        world.observe::<AudioEmitter, _>(LifecycleEvent::Remove, record_removed_emitter);

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
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    use prism_audio_core::voice::VoiceGroup;
    use prism_transform::GlobalTransform;

    fn test_config() -> AudioRuntimeConfig {
        AudioRuntimeConfig {
            sample_rate: 48_000,
            max_block: 64,
            command_capacity: 32,
            telemetry_capacity: 16,
            retire_capacity: 8,
            voice_capacity: 8,
            max_physical_voices: 4,
        }
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(AudioRuntimePlugin::new(test_config()));
        app
    }

    fn pump(app: &mut App) {
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 64);
        app.world_mut()
            .get_non_send_mut::<AudioRuntimeHost>()
            .expect("runtime host")
            .pump_block(&mut output);
    }

    #[test]
    fn plugin_spawns_and_stops_a_voice() {
        let mut app = test_app();
        let entity = app.world_mut().spawn((
            AudioEmitter::new(VoiceGroup(0), 1.0),
            GlobalTransform::default(),
        ));
        app.update();
        pump(&mut app);
        assert_eq!(
            app.world()
                .get_non_send::<AudioRuntimeHost>()
                .expect("runtime host")
                .runtime()
                .voices()
                .active_count(),
            1
        );

        app.world_mut().remove::<AudioEmitter>(entity);
        app.update();
        pump(&mut app);
        assert_eq!(
            app.world()
                .get_non_send::<AudioRuntimeHost>()
                .expect("runtime host")
                .runtime()
                .voices()
                .active_count(),
            0
        );
    }

    #[test]
    fn telemetry_is_drained_into_resource() {
        let mut app = test_app();
        pump(&mut app);
        app.update();
        let telemetry = app.world().resource::<AudioTelemetry>();
        assert!(telemetry.has_data());
        assert_eq!(telemetry.latest().frames, 64);
    }
}
