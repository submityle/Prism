//! Backend-neutral live-device lifecycle and callback boundary.
//!
//! A live backend owns the [`BlockRenderer`] (and therefore the real-time
//! [`AudioRuntime`](prism_audio_rt::AudioRuntime)) for exactly one stream
//! generation. Device loss is handled on the control thread by dropping that
//! generation and asking an [`AudioDeviceFactory`] to open a new one. The
//! callback never locks, allocates, or attempts recovery.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use std::sync::Mutex;

use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::math::Sample;
use prism_audio_rt::AudioRuntime;

use crate::error::DeviceError;
use crate::render::BlockRenderer;

/// Sample representation negotiated with the physical device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSampleFormat {
    /// 32-bit IEEE-754 floating point.
    F32,
    /// 64-bit IEEE-754 floating point.
    F64,
    /// Signed 8-bit integer PCM.
    I8,
    /// Signed 16-bit integer PCM.
    I16,
    /// Signed 32-bit integer PCM.
    I32,
    /// Unsigned 8-bit integer PCM.
    U8,
    /// Unsigned 16-bit integer PCM.
    U16,
}

/// Backend-neutral description of one opened output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceFormat {
    /// Device sample rate in frames per second.
    pub sample_rate: u32,
    /// Device channel layout.
    pub layout: ChannelLayout,
    /// Device sample representation.
    pub sample_format: DeviceSampleFormat,
}

/// Device lifecycle state visible to the control thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    /// No stream has been opened.
    Closed,
    /// A stream is open but callback delivery is paused.
    Stopped,
    /// Callback delivery is active.
    Running,
    /// Playback continues after a recoverable xrun.
    Degraded,
    /// The current stream is unusable and must be rebuilt.
    Recovering,
    /// The configured recovery budget is exhausted.
    Failed,
}

/// Events emitted by a device callback or a control-thread rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceEvent {
    /// Callback delivery started.
    Started,
    /// Callback delivery stopped without discarding the stream.
    Stopped,
    /// The host reported an output underrun.
    Underrun {
        /// Affected frame count, or zero when the host does not expose it.
        frames: u32,
    },
    /// The endpoint disappeared or the stream was invalidated.
    Disconnected,
    /// The backend reported a non-disconnect stream failure.
    StreamError,
    /// A replacement generation opened with the same format.
    Reconnected,
    /// A replacement generation opened with a different format.
    FormatChanged {
        /// Last active generation's format.
        previous: DeviceFormat,
        /// Replacement generation's format.
        current: DeviceFormat,
    },
    /// One control-thread recovery attempt failed.
    RecoveryFailed,
}

/// Bounded recovery state machine shared by device hosts and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRecovery {
    state: DeviceState,
    attempts: u32,
    max_attempts: u32,
}

impl DeviceRecovery {
    /// Creates a closed state machine with a bounded recovery budget.
    #[must_use]
    pub const fn new(max_attempts: u32) -> Self {
        Self {
            state: DeviceState::Closed,
            attempts: 0,
            max_attempts,
        }
    }

    /// Returns the current lifecycle state.
    #[must_use]
    pub const fn state(self) -> DeviceState {
        self.state
    }

    /// Returns failed recovery attempts in the current incident.
    #[must_use]
    pub const fn attempts(self) -> u32 {
        self.attempts
    }

    /// Returns the configured failure budget.
    #[must_use]
    pub const fn max_attempts(self) -> u32 {
        self.max_attempts
    }

    /// Applies one observed lifecycle event.
    pub fn observe(&mut self, event: DeviceEvent) -> DeviceState {
        if self.state == DeviceState::Failed {
            return self.state;
        }
        match event {
            DeviceEvent::Started | DeviceEvent::Reconnected | DeviceEvent::FormatChanged { .. } => {
                self.state = DeviceState::Running;
                self.attempts = 0;
            }
            DeviceEvent::Stopped => self.state = DeviceState::Stopped,
            DeviceEvent::Underrun { .. } => {
                if self.state == DeviceState::Running {
                    self.state = DeviceState::Degraded;
                }
            }
            DeviceEvent::Disconnected | DeviceEvent::StreamError => {
                self.state = DeviceState::Recovering;
            }
            DeviceEvent::RecoveryFailed => {
                self.attempts = self.attempts.saturating_add(1);
                self.state = if self.attempts < self.max_attempts.max(1) {
                    DeviceState::Recovering
                } else {
                    DeviceState::Failed
                };
            }
        }
        self.state
    }

    /// Marks a newly opened generation as active.
    pub fn recovered(&mut self, event: DeviceEvent) -> DeviceState {
        debug_assert!(matches!(
            event,
            DeviceEvent::Reconnected | DeviceEvent::FormatChanged { .. }
        ));
        self.observe(event)
    }
}

/// One opened live output stream.
///
/// The trait intentionally has no callback method: a physical SDK invokes its
/// callback internally. Test backends may expose an explicit callback driver,
/// but that is not evidence that physical hardware was exercised.
pub trait AudioDeviceBackend {
    /// Returns the current stream lifecycle state.
    fn state(&self) -> DeviceState;
    /// Returns the negotiated format for this generation.
    fn format(&self) -> DeviceFormat;
    /// Starts or resumes callback delivery.
    fn start(&mut self) -> Result<(), DeviceError>;
    /// Stops callback delivery without discarding the generation.
    fn stop(&mut self) -> Result<(), DeviceError>;
    /// Drains one pending callback/control event without blocking.
    fn poll_event(&mut self) -> Option<DeviceEvent>;
}

/// Opens one backend generation around an owned real-time runtime.
pub trait AudioDeviceFactory {
    /// Opens one stopped backend around the supplied runtime generation.
    fn open(
        &mut self,
        runtime: AudioRuntime,
        block_frames: usize,
    ) -> Result<Box<dyn AudioDeviceBackend>, DeviceError>;
}

/// Cloneable control handle for deterministic backend event injection.
///
/// This is a state-machine test seam, not a physical-device implementation.
#[derive(Clone, Debug, Default)]
pub struct FakeAudioControl {
    events: Arc<Mutex<VecDeque<DeviceEvent>>>,
}

impl FakeAudioControl {
    /// Queues one event for the backend's control-thread poller.
    pub fn inject(&self, event: DeviceEvent) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(event);
    }
}

/// Deterministic callback backend using the production [`BlockRenderer`].
///
/// It validates callback ownership and lifecycle semantics in tests. It does
/// not claim to validate a host API, driver, or physical audio endpoint.
pub struct FakeAudioBackend {
    renderer: BlockRenderer,
    format: DeviceFormat,
    recovery: DeviceRecovery,
    control: FakeAudioControl,
    started: bool,
}

impl FakeAudioBackend {
    /// Creates a fake backend using the renderer's native format.
    #[must_use]
    pub fn new(renderer: BlockRenderer, max_recovery_attempts: u32) -> Self {
        Self::controlled(renderer, max_recovery_attempts).0
    }

    /// Creates a fake backend and a cloneable event-injection handle.
    #[must_use]
    pub fn controlled(
        renderer: BlockRenderer,
        max_recovery_attempts: u32,
    ) -> (Self, FakeAudioControl) {
        let format = DeviceFormat {
            sample_rate: renderer.runtime().sample_rate(),
            layout: renderer.layout(),
            sample_format: DeviceSampleFormat::F32,
        };
        Self::controlled_with_format(renderer, max_recovery_attempts, format)
    }

    /// Creates a controlled backend with an explicitly negotiated format.
    ///
    /// The format must agree with the renderer's sample rate and channel
    /// layout; only the sample representation may differ independently.
    #[must_use]
    pub fn controlled_with_format(
        renderer: BlockRenderer,
        max_recovery_attempts: u32,
        format: DeviceFormat,
    ) -> (Self, FakeAudioControl) {
        assert_eq!(format.sample_rate, renderer.runtime().sample_rate());
        assert_eq!(format.layout, renderer.layout());
        let control = FakeAudioControl::default();
        (
            Self {
                renderer,
                format,
                recovery: DeviceRecovery::new(max_recovery_attempts),
                control: control.clone(),
                started: false,
            },
            control,
        )
    }

    /// Queues one backend event without invoking the callback.
    pub fn inject(&self, event: DeviceEvent) {
        self.control.inject(event);
    }

    /// Returns the fake backend's local recovery state machine.
    #[must_use]
    pub const fn recovery(&self) -> DeviceRecovery {
        self.recovery
    }

    /// Drives the same renderer used by a physical callback.
    pub fn callback(&mut self, output: &mut [Sample]) -> Result<(), DeviceError> {
        if !self.started || !matches!(self.state(), DeviceState::Running | DeviceState::Degraded) {
            return Err(DeviceError::InvalidState {
                operation: "callback",
                state: self.state(),
            });
        }
        self.renderer.render_interleaved(output);
        Ok(())
    }
}

impl AudioDeviceBackend for FakeAudioBackend {
    fn state(&self) -> DeviceState {
        self.recovery.state()
    }

    fn format(&self) -> DeviceFormat {
        self.format
    }

    fn start(&mut self) -> Result<(), DeviceError> {
        if !matches!(self.state(), DeviceState::Closed | DeviceState::Stopped) {
            return Err(DeviceError::InvalidState {
                operation: "start",
                state: self.state(),
            });
        }
        self.started = true;
        self.recovery.observe(DeviceEvent::Started);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), DeviceError> {
        if !matches!(self.state(), DeviceState::Running | DeviceState::Degraded) {
            return Err(DeviceError::InvalidState {
                operation: "stop",
                state: self.state(),
            });
        }
        self.started = false;
        self.recovery.observe(DeviceEvent::Stopped);
        Ok(())
    }

    fn poll_event(&mut self) -> Option<DeviceEvent> {
        let event = self
            .control
            .events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()?;
        self.recovery.observe(event);
        match event {
            DeviceEvent::Started | DeviceEvent::Reconnected | DeviceEvent::FormatChanged { .. } => {
                self.started = true;
            }
            DeviceEvent::Stopped | DeviceEvent::Disconnected | DeviceEvent::StreamError => {
                self.started = false;
            }
            DeviceEvent::Underrun { .. } | DeviceEvent::RecoveryFailed => {}
        }
        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_core::buffer::ChannelLayout;
    use prism_audio_rt::{runtime, AudioRuntimeConfig};

    fn fake() -> FakeAudioBackend {
        let (runtime, _client, _collector) = runtime(AudioRuntimeConfig {
            max_block: 16,
            ..AudioRuntimeConfig::default()
        });
        FakeAudioBackend::new(BlockRenderer::new(runtime, ChannelLayout::Stereo, 8), 2)
    }

    #[test]
    fn callback_requires_running_and_stop_is_explicit() {
        let mut backend = fake();
        let mut output = [0.0; 16];
        assert!(backend.callback(&mut output).is_err());
        backend.start().unwrap();
        backend.callback(&mut output).unwrap();
        backend.stop().unwrap();
        assert_eq!(backend.state(), DeviceState::Stopped);
        assert!(backend.callback(&mut output).is_err());
        backend.start().unwrap();
        backend.callback(&mut output).unwrap();
    }

    #[test]
    fn callback_and_recovery_events_are_deterministic() {
        let mut backend = fake();
        backend.start().unwrap();
        backend.inject(DeviceEvent::Underrun { frames: 8 });
        assert_eq!(
            backend.poll_event(),
            Some(DeviceEvent::Underrun { frames: 8 })
        );
        assert_eq!(backend.state(), DeviceState::Degraded);
        backend.inject(DeviceEvent::Disconnected);
        assert_eq!(backend.poll_event(), Some(DeviceEvent::Disconnected));
        assert_eq!(backend.state(), DeviceState::Recovering);
        assert!(backend.callback(&mut [0.0; 16]).is_err());
    }

    #[test]
    fn repeated_recovery_failures_are_bounded_and_terminal() {
        let mut recovery = DeviceRecovery::new(2);
        recovery.observe(DeviceEvent::Disconnected);
        recovery.observe(DeviceEvent::RecoveryFailed);
        assert_eq!(recovery.state(), DeviceState::Recovering);
        recovery.observe(DeviceEvent::RecoveryFailed);
        assert_eq!(recovery.state(), DeviceState::Failed);
        recovery.observe(DeviceEvent::Reconnected);
        assert_eq!(recovery.state(), DeviceState::Failed);
    }

    #[test]
    fn controlled_handle_injects_stream_error() {
        let (runtime, _client, _collector) = runtime(AudioRuntimeConfig::default());
        let (mut backend, control) =
            FakeAudioBackend::controlled(BlockRenderer::new(runtime, ChannelLayout::Stereo, 64), 2);
        backend.start().unwrap();
        control.inject(DeviceEvent::StreamError);
        assert_eq!(backend.poll_event(), Some(DeviceEvent::StreamError));
        assert_eq!(backend.state(), DeviceState::Recovering);
    }
}
