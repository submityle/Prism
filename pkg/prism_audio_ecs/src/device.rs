//! Physical-device lifecycle host for the ECS audio facade.
//!
//! Each opened backend owns one `AudioRuntime` generation. A disconnect or
//! stream error invalidates that generation; the App thread creates a fresh
//! runtime/client/collector triplet, opens a fresh backend, swaps the retained
//! [`AudioClient`](crate::AudioClient) facade, and republishes authoritative
//! ECS graph/voice state. No recovery work occurs in the device callback.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use core::sync::atomic::AtomicBool;

use prism_app::prelude::{App, Last, Plugin};
use prism_audio_device::{
    AudioDeviceBackend, AudioDeviceFactory, DeviceError, DeviceEvent, DeviceFormat, DeviceRecovery,
    DeviceState,
};
use prism_audio_rt::{AudioRuntime, AudioRuntimeConfig, Collector};
use prism_ecs::prelude::{Resource, World};
use prism_ecs::schedule::IntoSystemConfigs;

use crate::client_resource::{AudioClient, AudioRuntimeMetadata};
use crate::emitter::AudioEmitter;
use crate::plugin::{initialize_control_plane, shutdown, BridgeState};
use crate::source_route::{AudioGraphRouteState, AudioOutputLayout};
use crate::systems::{audio_frame, AudioSystems};
use crate::telemetry::AudioTelemetry;
use crate::voice_registry::{LiveVoices, PendingStops, RemovedAudioEmitters, VoiceMirror};
use crate::PhysicalVoiceBudget;

/// Device callback and recovery policy.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioDeviceConfig {
    /// Fixed engine block size bridged into variable host callback buffers.
    pub block_frames: usize,
    /// Failed reopen attempts allowed before entering terminal failure.
    pub max_recovery_attempts: u32,
}

impl Default for AudioDeviceConfig {
    fn default() -> Self {
        Self {
            block_frames: 256,
            max_recovery_attempts: 3,
        }
    }
}

/// Observable state of the current physical-device generation.
#[derive(Resource, Debug, Clone, PartialEq, Eq)]
pub struct AudioDeviceStatus {
    /// Current externally observable lifecycle state.
    pub state: DeviceState,
    /// Negotiated format of the latest successfully opened generation.
    pub format: Option<DeviceFormat>,
    /// Monotonic count of successfully opened runtime/backend generations.
    pub generation: u64,
    /// Failed reopen attempts in the current recovery incident.
    pub recovery_attempts: u32,
    /// Most recent lifecycle event.
    pub last_event: Option<DeviceEvent>,
    /// Most recent backend error message.
    pub last_error: Option<String>,
}

impl Default for AudioDeviceStatus {
    fn default() -> Self {
        Self {
            state: DeviceState::Closed,
            format: None,
            generation: 0,
            recovery_attempts: 0,
            last_event: None,
            last_error: None,
        }
    }
}

/// Non-send owner of one live backend plus its off-thread collector.
pub struct AudioDeviceRuntimeHost {
    backend: Option<Box<dyn AudioDeviceBackend>>,
    factory: Box<dyn AudioDeviceFactory>,
    collector: Option<Collector>,
    client: AudioClient,
    runtime_config: AudioRuntimeConfig,
    device_config: AudioDeviceConfig,
    recovery: DeviceRecovery,
    desired_running: bool,
    format: Option<DeviceFormat>,
    generation: u64,
}

impl AudioDeviceRuntimeHost {
    fn new(
        factory: Box<dyn AudioDeviceFactory>,
        client: AudioClient,
        runtime_config: AudioRuntimeConfig,
        device_config: AudioDeviceConfig,
    ) -> Self {
        Self {
            backend: None,
            factory,
            collector: None,
            client,
            runtime_config,
            device_config,
            recovery: DeviceRecovery::new(device_config.max_recovery_attempts),
            desired_running: true,
            format: None,
            generation: 0,
        }
    }

    /// Collects deferred graph retirements on the App thread.
    pub fn collect(&self) -> usize {
        self.collector.as_ref().map_or(0, Collector::collect)
    }

    /// Current backend state, including the stopped state.
    #[must_use]
    pub fn state(&self) -> DeviceState {
        self.backend
            .as_ref()
            .map_or(self.recovery.state(), |backend| backend.state())
    }

    fn invalidate_generation(&mut self, world: &mut World, event: DeviceEvent) {
        self.client.suspend();
        if let Some(mut backend) = self.backend.take()
            && matches!(
                backend.state(),
                DeviceState::Running | DeviceState::Degraded
            )
        {
            let _ = backend.stop();
        }
        if let Some(collector) = self.collector.take() {
            let _ = collector.collect();
        }
        self.recovery.observe(event);
        reset_runtime_projection(world, self.runtime_config);
        let status = world.resource_mut::<AudioDeviceStatus>();
        status.state = self.recovery.state();
        status.last_event = Some(event);
        status.recovery_attempts = self.recovery.attempts();
    }

    fn open_generation(&mut self, world: &mut World) -> bool {
        if self.recovery.state() == DeviceState::Failed {
            return false;
        }
        let (runtime, replacement, collector) = prism_audio_rt::runtime(self.runtime_config);
        let metadata = AudioRuntimeMetadata {
            sample_rate: runtime.sample_rate(),
            max_block: runtime.max_block(),
        };
        match self.factory.open(runtime, self.device_config.block_frames) {
            Ok(mut backend) => {
                let format = backend.format();
                let start = if self.desired_running {
                    backend.start()
                } else {
                    Ok(())
                };
                if let Err(error) = start {
                    self.record_recovery_failure(world, error);
                    return false;
                }
                let event = match self.format {
                    Some(previous) if previous != format => DeviceEvent::FormatChanged {
                        previous,
                        current: format,
                    },
                    Some(_) => DeviceEvent::Reconnected,
                    None => DeviceEvent::Started,
                };
                self.client.replace(replacement);
                self.collector = Some(collector);
                self.backend = Some(backend);
                self.format = Some(format);
                self.generation = self.generation.saturating_add(1);
                self.recovery.recovered(match event {
                    DeviceEvent::Started => DeviceEvent::Reconnected,
                    other => other,
                });
                world.insert_resource(metadata);
                world.insert_resource(AudioOutputLayout(format.layout));
                let status = world.resource_mut::<AudioDeviceStatus>();
                status.state = if self.desired_running {
                    DeviceState::Running
                } else {
                    DeviceState::Stopped
                };
                status.format = Some(format);
                status.generation = self.generation;
                status.recovery_attempts = 0;
                status.last_event = Some(event);
                status.last_error = None;
                true
            }
            Err(error) => {
                self.record_recovery_failure(world, error);
                false
            }
        }
    }

    fn record_recovery_failure(&mut self, world: &mut World, error: DeviceError) {
        self.client.suspend();
        self.backend = None;
        self.collector = None;
        if !matches!(
            self.recovery.state(),
            DeviceState::Recovering | DeviceState::Failed
        ) {
            self.recovery.observe(DeviceEvent::StreamError);
        }
        self.recovery.observe(DeviceEvent::RecoveryFailed);
        let status = world.resource_mut::<AudioDeviceStatus>();
        status.state = self.recovery.state();
        status.recovery_attempts = self.recovery.attempts();
        status.last_event = Some(DeviceEvent::RecoveryFailed);
        status.last_error = Some(error.to_string());
    }

    fn install_initial_generation(
        &mut self,
        world: &mut World,
        runtime: AudioRuntime,
        collector: Collector,
    ) {
        match self.factory.open(runtime, self.device_config.block_frames) {
            Ok(mut backend) => match backend.start() {
                Ok(()) => {
                    let format = backend.format();
                    self.backend = Some(backend);
                    self.collector = Some(collector);
                    self.format = Some(format);
                    self.generation = 1;
                    self.recovery.observe(DeviceEvent::Started);
                    world.insert_resource(AudioOutputLayout(format.layout));
                    *world.resource_mut::<AudioDeviceStatus>() = AudioDeviceStatus {
                        state: DeviceState::Running,
                        format: Some(format),
                        generation: 1,
                        recovery_attempts: 0,
                        last_event: Some(DeviceEvent::Started),
                        last_error: None,
                    };
                }
                Err(error) => self.record_recovery_failure(world, error),
            },
            Err(error) => self.record_recovery_failure(world, error),
        }
    }
}

impl Drop for AudioDeviceRuntimeHost {
    fn drop(&mut self) {
        if let Some(backend) = self.backend.as_mut()
            && matches!(
                backend.state(),
                DeviceState::Running | DeviceState::Degraded
            )
        {
            let _ = backend.stop();
        }
        if let Some(collector) = self.collector.as_ref() {
            let _ = collector.collect();
        }
        self.client.close();
    }
}

/// Installs the ECS bridge and opens the first live device generation.
pub fn install_device(
    world: &mut World,
    runtime_config: AudioRuntimeConfig,
    device_config: AudioDeviceConfig,
    factory: Box<dyn AudioDeviceFactory>,
) {
    shutdown(world);
    let (runtime, raw_client, collector) = prism_audio_rt::runtime(runtime_config);
    let metadata = AudioRuntimeMetadata {
        sample_rate: runtime.sample_rate(),
        max_block: runtime.max_block(),
    };
    let client = AudioClient::new(raw_client, Arc::new(AtomicBool::new(true)));
    initialize_control_plane(world, runtime_config, client.clone(), metadata);
    world.insert_resource(device_config);
    world.insert_resource(AudioDeviceStatus::default());
    let mut host = AudioDeviceRuntimeHost::new(factory, client, runtime_config, device_config);
    host.install_initial_generation(world, runtime, collector);
    world.insert_non_send(host);
}

/// Starts a stopped device or resumes recovery when no generation is open.
pub fn start_audio_device(world: &mut World) -> Result<(), DeviceError> {
    let Some(mut host) = world.remove_non_send::<AudioDeviceRuntimeHost>() else {
        return Err(DeviceError::Backend(
            "audio device host is not installed".into(),
        ));
    };
    host.desired_running = true;
    let result = if let Some(backend) = host.backend.as_mut() {
        backend.start().map(|()| {
            host.recovery.observe(DeviceEvent::Started);
            let status = world.resource_mut::<AudioDeviceStatus>();
            status.state = DeviceState::Running;
            status.last_event = Some(DeviceEvent::Started);
            status.last_error = None;
        })
    } else if host.open_generation(world) {
        Ok(())
    } else {
        Err(DeviceError::Backend(
            world
                .resource::<AudioDeviceStatus>()
                .last_error
                .clone()
                .unwrap_or_else(|| "audio device recovery failed".into()),
        ))
    };
    world.insert_non_send(host);
    result
}

/// Pauses callback delivery without discarding the current runtime generation.
pub fn stop_audio_device(world: &mut World) -> Result<(), DeviceError> {
    let Some(mut host) = world.remove_non_send::<AudioDeviceRuntimeHost>() else {
        return Err(DeviceError::Backend(
            "audio device host is not installed".into(),
        ));
    };
    host.desired_running = false;
    let result = match host.backend.as_mut() {
        Some(backend) => backend.stop(),
        None => Err(DeviceError::InvalidState {
            operation: "stop",
            state: host.recovery.state(),
        }),
    };
    if result.is_ok() {
        let status = world.resource_mut::<AudioDeviceStatus>();
        status.state = DeviceState::Stopped;
        status.last_event = Some(DeviceEvent::Stopped);
        status.last_error = None;
    }
    world.insert_non_send(host);
    result
}

/// Drains backend events, performs bounded rebuild attempts, and collects
/// retired graph resources. Call once per App/control-thread frame.
pub fn drive_audio_device(world: &mut World) {
    let Some(mut host) = world.remove_non_send::<AudioDeviceRuntimeHost>() else {
        return;
    };
    let mut rebuild = host.backend.is_none()
        && host.desired_running
        && host.recovery.state() == DeviceState::Recovering;
    let mut terminal_event = None;
    if let Some(backend) = host.backend.as_mut() {
        while let Some(event) = backend.poll_event() {
            match event {
                DeviceEvent::Underrun { .. } => {
                    let status = world.resource_mut::<AudioDeviceStatus>();
                    status.state = DeviceState::Degraded;
                    status.last_event = Some(event);
                }
                DeviceEvent::Disconnected | DeviceEvent::StreamError => {
                    terminal_event = Some(event);
                    break;
                }
                _ => {
                    world.resource_mut::<AudioDeviceStatus>().last_event = Some(event);
                }
            }
        }
    }
    if let Some(event) = terminal_event {
        host.invalidate_generation(world, event);
        rebuild = host.desired_running;
    }
    if rebuild {
        host.open_generation(world);
    }
    let _ = host.collect();
    world.insert_non_send(host);
}

fn reset_runtime_projection(world: &mut World, config: AudioRuntimeConfig) {
    world.insert_resource(VoiceMirror::new(
        config.voice_capacity,
        config.max_physical_voices,
    ));
    world.insert_resource(LiveVoices::default());
    world.insert_resource(PendingStops::default());
    world.insert_resource(RemovedAudioEmitters::default());
    world.insert_resource(AudioGraphRouteState::default());
    let telemetry_capacity = world.resource::<AudioTelemetry>().capacity();
    world.insert_resource(AudioTelemetry::with_capacity(telemetry_capacity));
    let query = world.query::<&mut AudioEmitter>();
    for mut emitter in query.iter_mut(world) {
        emitter.voice = None;
        emitter.sent_importance = None;
    }
    {
        let generation = world.archetype_generation();
        let bridge = world.resource_mut::<BridgeState>();
        bridge.generation = generation;
        bridge.gain = None;
        bridge.budget = None;
    }
    let budget = world.resource::<PhysicalVoiceBudget>().max;
    world.resource_mut::<VoiceMirror>().set_max_physical(budget);
}

/// App plugin for a concrete, injectable device factory.
#[derive(Clone)]
pub struct AudioDevicePlugin<F> {
    /// Real-time runtime capacity and format configuration.
    pub runtime: AudioRuntimeConfig,
    /// Device callback and recovery policy.
    pub device: AudioDeviceConfig,
    /// Injectable physical or test backend factory.
    pub factory: F,
}

impl<F> AudioDevicePlugin<F> {
    /// Builds a plugin from runtime, device, and backend policies.
    #[must_use]
    pub const fn new(runtime: AudioRuntimeConfig, device: AudioDeviceConfig, factory: F) -> Self {
        Self {
            runtime,
            device,
            factory,
        }
    }
}

impl<F> Plugin for AudioDevicePlugin<F>
where
    F: AudioDeviceFactory + Clone + Send + Sync + 'static,
{
    fn build(&self, app: &mut App) {
        install_device(
            app.world_mut(),
            self.runtime,
            self.device,
            Box::new(self.factory.clone()),
        );
        app.add_systems(
            Last,
            (|world: &mut World| {
                drive_audio_device(world);
                audio_frame(world);
            })
            .in_set(AudioSystems),
        );
    }

    fn shutdown(&self, app: &mut App) {
        shutdown(app.world_mut());
    }
}

#[cfg(feature = "cpal-backend")]
/// Production plugin using the platform default CPAL output endpoint.
pub type CpalAudioDevicePlugin = AudioDevicePlugin<prism_audio_device::CpalOutputFactory>;

#[cfg(feature = "cpal-backend")]
impl Default for AudioDevicePlugin<prism_audio_device::CpalOutputFactory> {
    fn default() -> Self {
        Self::new(
            AudioRuntimeConfig::default(),
            AudioDeviceConfig::default(),
            prism_audio_device::CpalOutputFactory,
        )
    }
}
