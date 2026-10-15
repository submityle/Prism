//! Plugin wiring for the Prism ECS audio bridge.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::AtomicBool;

use prism_app::prelude::{App, Last, Plugin};
use prism_audio_rt::AudioRuntimeConfig;
use prism_ecs::{observer::ObserverId, prelude::*, world::WorldId};
use prism_transform::GlobalTransform;

use crate::assets::AudioAssetStore;
use crate::budget::PhysicalVoiceBudget;
use crate::client_resource::{AudioClient, AudioRuntimeHost};
use crate::emitter::AudioEmitter;
use crate::listener::AudioListener;
use crate::master_gain::MasterGain;
use crate::playback::PlaybackSettings;
use crate::player::AudioPlayer;
use crate::systems::{audio_frame, AudioSystems};
use crate::telemetry::AudioTelemetry;
use crate::voice_registry::{LiveVoices, PendingStops, RemovedAudioEmitters, VoiceMirror};

#[derive(Resource)]
pub(crate) struct BridgeState {
    pub world: WorldId,
    pub generation: u64,
    pub gain: Option<(u32, u32)>,
    pub budget: Option<usize>,
    observers: Vec<ObserverId>,
}

#[derive(Resource)]
struct AudioStopped;

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

fn record_removed_emitter(ctx: &mut ObserverContext<'_>) {
    if let Some(removed) = ctx.world.get_resource_mut::<RemovedAudioEmitters>() {
        removed.push(ctx.entity);
    }
}

/// Installs a runtime and lifecycle observers into a World, idempotently.
/// Preserves authoring gain/budget. Calling this after [`shutdown`] explicitly
/// restarts audio with a new pool; retained old clients stay closed.
///
/// This function does not install schedules: standalone Worlds call
/// [`audio_frame`]. Replacing an App's World also requires transferring its App
/// schedules (or constructing a new App/plugin), not just its audio resources.
pub fn install(world: &mut World, config: AudioRuntimeConfig) {
    if world
        .get_resource::<BridgeState>()
        .is_some_and(|state| state.world == world.id())
        && world
            .get_resource::<AudioClient>()
            .is_some_and(AudioClient::is_open)
    {
        return;
    }
    shutdown(world);
    world.remove_resource::<AudioStopped>();
    let (runtime, client, collector) = prism_audio_rt::runtime(config);
    let open = Arc::new(AtomicBool::new(true));
    let client = AudioClient::new(client, open);
    world.insert_resource(client.clone());
    world.insert_non_send(AudioRuntimeHost::new(runtime, collector, client));
    world.insert_resource(VoiceMirror::new(
        config.voice_capacity,
        config.max_physical_voices,
    ));
    world.insert_resource(LiveVoices::default());
    world.insert_resource(PendingStops::default());
    world.insert_resource(RemovedAudioEmitters::default());
    world.init_resource::<MasterGain>();
    world.init_resource::<AudioAssetStore>();
    if world.get_resource::<PhysicalVoiceBudget>().is_none() {
        world.insert_resource(PhysicalVoiceBudget::new(config.max_physical_voices));
    }
    // Preserve the caller's history capacity, but do not mix independent
    // runtime block/playhead epochs after an explicit restart or host loss.
    let capacity = world
        .get_resource::<AudioTelemetry>()
        .map(AudioTelemetry::capacity);
    if let Some(capacity) = capacity {
        world.insert_resource(AudioTelemetry::with_capacity(capacity));
    } else {
        world.init_resource::<AudioTelemetry>();
    }
    world.register_snapshot_component_equatable::<AudioEmitter>();
    world.register_snapshot_component_equatable::<AudioListener>();
    world.register_snapshot_component_equatable::<AudioPlayer>();
    world.register_snapshot_component_equatable::<PlaybackSettings>();
    world.register_snapshot_component_equatable::<crate::player_systems::DerivedEmitter>();
    world.register_snapshot_resource_equatable::<MasterGain>();
    world.register_snapshot_resource_equatable::<PhysicalVoiceBudget>();
    let observers = alloc::vec![
        world.observe::<AudioEmitter, _>(LifecycleEvent::Replace, record_removed_emitter),
        world.observe::<GlobalTransform, _>(LifecycleEvent::Remove, record_removed_emitter),
    ];
    world.insert_resource(BridgeState {
        world: world.id(),
        generation: world.archetype_generation(),
        gain: None,
        budget: None,
        observers,
    });
}

/// Stops this control plane, closes retained client clones, drops the non-send
/// runtime/collector, unregisters observers, and clears cached voice handles.
/// Idempotent. Pending commands are discarded by runtime destruction rather
/// than depending on a possibly full command ring to deliver shutdown stops.
pub fn shutdown(world: &mut World) {
    if let Some(client) = world.remove_resource::<AudioClient>() {
        client.close();
    }
    world.remove_non_send::<AudioRuntimeHost>();
    if let Some(state) = world.remove_resource::<BridgeState>() {
        for observer in state.observers {
            world.remove_observer(observer);
        }
    }
    world.remove_resource::<VoiceMirror>();
    world.remove_resource::<LiveVoices>();
    world.remove_resource::<PendingStops>();
    world.remove_resource::<RemovedAudioEmitters>();
    let query = world.query::<&mut AudioEmitter>();
    for mut emitter in query.iter_mut(world) {
        emitter.voice = None;
        emitter.sent_importance = None;
    }
    world.insert_resource(AudioStopped);
}

impl Plugin for AudioRuntimePlugin {
    fn build(&self, app: &mut App) {
        let config = self.config;
        install(app.world_mut(), config);
        app.add_systems(
            Last,
            (move |world: &mut World| {
                if world.get_resource::<AudioStopped>().is_some() {
                    return;
                }
                install(world, config);
                audio_frame(world);
            })
            .in_set(AudioSystems),
        );
    }

    fn shutdown(&self, app: &mut App) {
        shutdown(app.world_mut());
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
