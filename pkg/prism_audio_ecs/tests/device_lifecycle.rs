//! Injectable lifecycle tests for the ECS physical-device host.
//!
//! These tests exercise the production runtime, block renderer, retained
//! client facade, and rebuild state machine. They do not claim physical-device
//! coverage; the CPAL smoke test owns that opt-in boundary.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::voice::VoiceGroup;
use prism_audio_device::{
    AudioDeviceBackend, AudioDeviceFactory, BlockRenderer, DeviceError, DeviceEvent, DeviceFormat,
    DeviceSampleFormat, DeviceState, FakeAudioBackend, FakeAudioControl,
};
use prism_audio_ecs::prelude::*;
use prism_audio_ecs::systems::audio_frame;
use prism_audio_rt::{AudioCommand, AudioRuntime, AudioRuntimeConfig};
use prism_ecs::prelude::World;
use prism_transform::GlobalTransform;

#[derive(Clone)]
struct ScriptedFactory {
    shared: Arc<Mutex<FactoryState>>,
}

struct FactoryState {
    scripts: VecDeque<Result<DeviceFormat, &'static str>>,
    controls: Vec<FakeAudioControl>,
    metrics: Vec<Arc<BackendMetrics>>,
    block_frames: Vec<usize>,
    opens: usize,
}

#[derive(Default)]
struct BackendMetrics {
    starts: AtomicUsize,
    stops: AtomicUsize,
    drops: AtomicUsize,
}

struct InstrumentedBackend {
    inner: FakeAudioBackend,
    metrics: Arc<BackendMetrics>,
}

impl Drop for InstrumentedBackend {
    fn drop(&mut self) {
        self.metrics.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl AudioDeviceBackend for InstrumentedBackend {
    fn state(&self) -> DeviceState {
        self.inner.state()
    }

    fn format(&self) -> DeviceFormat {
        self.inner.format()
    }

    fn start(&mut self) -> Result<(), DeviceError> {
        let result = self.inner.start();
        if result.is_ok() {
            self.metrics.starts.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    fn stop(&mut self) -> Result<(), DeviceError> {
        let result = self.inner.stop();
        if result.is_ok() {
            self.metrics.stops.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    fn poll_event(&mut self) -> Option<DeviceEvent> {
        self.inner.poll_event()
    }
}

impl ScriptedFactory {
    fn new(scripts: impl IntoIterator<Item = Result<DeviceFormat, &'static str>>) -> Self {
        Self {
            shared: Arc::new(Mutex::new(FactoryState {
                scripts: scripts.into_iter().collect(),
                controls: Vec::new(),
                metrics: Vec::new(),
                block_frames: Vec::new(),
                opens: 0,
            })),
        }
    }

    fn control(&self, generation: usize) -> FakeAudioControl {
        self.shared.lock().unwrap().controls[generation].clone()
    }

    fn metrics(&self, generation: usize) -> Arc<BackendMetrics> {
        self.shared.lock().unwrap().metrics[generation].clone()
    }

    fn opens(&self) -> usize {
        self.shared.lock().unwrap().opens
    }
}

impl AudioDeviceFactory for ScriptedFactory {
    fn open(
        &mut self,
        runtime: AudioRuntime,
        block_frames: usize,
    ) -> Result<Box<dyn AudioDeviceBackend>, DeviceError> {
        let script = {
            let mut state = self.shared.lock().unwrap();
            state.opens += 1;
            state.block_frames.push(block_frames);
            state
                .scripts
                .pop_front()
                .unwrap_or(Err("unexpected extra recovery attempt"))
        };
        if block_frames == 0 {
            return Err(DeviceError::InvalidBlockFrames);
        }
        let format = script.map_err(|message| DeviceError::Backend(message.into()))?;
        let renderer = BlockRenderer::new(runtime, format.layout, block_frames);
        let (inner, control) = FakeAudioBackend::controlled_with_format(renderer, 1, format);
        let metrics = Arc::new(BackendMetrics::default());
        let backend = InstrumentedBackend {
            inner,
            metrics: metrics.clone(),
        };
        let mut state = self.shared.lock().unwrap();
        state.controls.push(control);
        state.metrics.push(metrics);
        Ok(Box::new(backend))
    }
}

fn runtime_config() -> AudioRuntimeConfig {
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

fn stereo(sample_format: DeviceSampleFormat) -> DeviceFormat {
    DeviceFormat {
        sample_rate: 48_000,
        layout: ChannelLayout::Stereo,
        sample_format,
    }
}

fn device_config(max_recovery_attempts: u32) -> AudioDeviceConfig {
    AudioDeviceConfig {
        block_frames: 32,
        max_recovery_attempts,
    }
}

#[test]
fn start_stop_disconnect_and_format_change_replace_the_runtime_generation() {
    let first = stereo(DeviceSampleFormat::F32);
    let changed = stereo(DeviceSampleFormat::I16);
    let factory = ScriptedFactory::new([Ok(first), Ok(first), Ok(changed)]);
    let mut world = World::new();
    install_device(
        &mut world,
        runtime_config(),
        device_config(3),
        Box::new(factory.clone()),
    );

    let retained = world.resource::<AudioClient>().clone();
    assert_eq!(
        world.resource::<AudioDeviceStatus>().state,
        DeviceState::Running
    );
    assert_eq!(world.resource::<AudioDeviceStatus>().generation, 1);
    assert_eq!(factory.metrics(0).starts.load(Ordering::SeqCst), 1);

    stop_audio_device(&mut world).unwrap();
    assert_eq!(
        world.resource::<AudioDeviceStatus>().state,
        DeviceState::Stopped
    );
    assert_eq!(factory.metrics(0).stops.load(Ordering::SeqCst), 1);
    start_audio_device(&mut world).unwrap();
    assert_eq!(
        world.resource::<AudioDeviceStatus>().state,
        DeviceState::Running
    );
    assert_eq!(factory.metrics(0).starts.load(Ordering::SeqCst), 2);

    let entity = world.spawn((
        AudioEmitter::new(VoiceGroup(7), 1.0),
        GlobalTransform::default(),
    ));
    audio_frame(&mut world);
    assert!(world.get::<AudioEmitter>(entity).unwrap().voice.is_some());

    factory.control(0).inject(DeviceEvent::Disconnected);
    drive_audio_device(&mut world);
    let status = world.resource::<AudioDeviceStatus>();
    assert_eq!(status.state, DeviceState::Running);
    assert_eq!(status.generation, 2);
    assert_eq!(status.last_event, Some(DeviceEvent::Reconnected));
    assert!(retained.is_open());
    assert!(world.get::<AudioEmitter>(entity).unwrap().voice.is_none());
    audio_frame(&mut world);
    assert!(world.get::<AudioEmitter>(entity).unwrap().voice.is_some());
    assert_eq!(factory.metrics(0).drops.load(Ordering::SeqCst), 1);

    factory.control(1).inject(DeviceEvent::StreamError);
    drive_audio_device(&mut world);
    let status = world.resource::<AudioDeviceStatus>();
    assert_eq!(status.state, DeviceState::Running);
    assert_eq!(status.generation, 3);
    assert_eq!(
        status.last_event,
        Some(DeviceEvent::FormatChanged {
            previous: first,
            current: changed,
        })
    );
    assert_eq!(
        world.resource::<AudioOutputLayout>().0,
        ChannelLayout::Stereo
    );
    retained
        .send(AudioCommand::SetMasterGain {
            linear: 0.5,
            at_frame: 0,
            ramp_frames: 0,
        })
        .unwrap();

    prism_audio_ecs::plugin::shutdown(&mut world);
    assert!(!retained.is_open());
    assert_eq!(factory.metrics(2).stops.load(Ordering::SeqCst), 1);
    assert_eq!(factory.metrics(2).drops.load(Ordering::SeqCst), 1);
}

#[test]
fn recovery_failures_are_bounded_and_failed_is_terminal() {
    let format = stereo(DeviceSampleFormat::F32);
    let factory = ScriptedFactory::new([
        Ok(format),
        Err("first reopen failed"),
        Err("second reopen failed"),
    ]);
    let mut world = World::new();
    install_device(
        &mut world,
        runtime_config(),
        device_config(2),
        Box::new(factory.clone()),
    );
    let retained = world.resource::<AudioClient>().clone();

    factory.control(0).inject(DeviceEvent::Disconnected);
    drive_audio_device(&mut world);
    assert_eq!(
        world.resource::<AudioDeviceStatus>().state,
        DeviceState::Recovering
    );
    assert_eq!(world.resource::<AudioDeviceStatus>().recovery_attempts, 1);
    assert!(!retained.is_open());

    drive_audio_device(&mut world);
    assert_eq!(
        world.resource::<AudioDeviceStatus>().state,
        DeviceState::Failed
    );
    assert_eq!(world.resource::<AudioDeviceStatus>().recovery_attempts, 2);
    assert_eq!(factory.opens(), 3);
    drive_audio_device(&mut world);
    drive_audio_device(&mut world);
    assert_eq!(factory.opens(), 3, "failed state must not retry forever");
    assert!(start_audio_device(&mut world).is_err());
    assert_eq!(factory.opens(), 3);

    prism_audio_ecs::plugin::shutdown(&mut world);
}

#[test]
fn zero_callback_block_size_is_rejected_instead_of_silently_corrected() {
    let factory = ScriptedFactory::new([Ok(stereo(DeviceSampleFormat::F32))]);
    let mut world = World::new();
    install_device(
        &mut world,
        runtime_config(),
        AudioDeviceConfig {
            block_frames: 0,
            max_recovery_attempts: 1,
        },
        Box::new(factory),
    );
    assert_eq!(
        world.resource::<AudioDeviceStatus>().state,
        DeviceState::Failed
    );
    assert!(!world.resource::<AudioClient>().is_open());
    prism_audio_ecs::plugin::shutdown(&mut world);
}
