//! Backend-neutral device callback and recovery boundary.
//!
//! Platform backends report callback failures to the control thread. Recovery
//! never reopens a stream from the callback itself: the callback remains a
//! bounded pull operation, while a controller decides when a new stream may be
//! opened. The fake backend below exercises the same state machine without
//! requiring a host audio device.

use alloc::collections::VecDeque;
use alloc::string::String;

use prism_audio_core::math::Sample;

use crate::render::BlockRenderer;

/// Device lifecycle state visible to the control thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    /// No stream is open.
    Closed,
    /// Callback is actively rendering.
    Running,
    /// A callback reported a recoverable glitch.
    Degraded,
    /// Control thread is allowed to attempt a reopen.
    Recovering,
    /// Recovery attempts are exhausted or a permanent error was reported.
    Failed,
}

/// Events emitted by a device callback or device enumeration layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceEvent {
    /// Stream opened successfully.
    Started,
    /// Callback could not provide the requested frames in time.
    Underrun {
        /// Number of frames reported by the host, or zero when the host only
        /// exposes a generic xrun notification.
        frames: u32,
    },
    /// Host removed or invalidated the device stream.
    Disconnected,
    /// A later enumeration observed a usable device again.
    Reconnected,
    /// A control-thread reopen attempt failed.
    RecoveryFailed,
}

/// Control-thread recovery state machine shared by real and fake backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRecovery {
    state: DeviceState,
    attempts: u32,
    max_attempts: u32,
}

impl DeviceRecovery {
    /// Creates a closed state machine with a bounded reopen budget.
    #[must_use]
    pub const fn new(max_attempts: u32) -> Self {
        Self {
            state: DeviceState::Closed,
            attempts: 0,
            max_attempts,
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub const fn state(self) -> DeviceState {
        self.state
    }

    /// Number of failed recovery attempts in the current incident.
    #[must_use]
    pub const fn attempts(self) -> u32 {
        self.attempts
    }

    /// Applies an observed event and returns the resulting state.
    pub fn observe(&mut self, event: DeviceEvent) -> DeviceState {
        match event {
            DeviceEvent::Started => {
                if self.state != DeviceState::Failed {
                    self.state = DeviceState::Running;
                    self.attempts = 0;
                }
            }
            DeviceEvent::Reconnected => {
                self.state = DeviceState::Running;
                self.attempts = 0;
            }
            DeviceEvent::Underrun { .. } => {
                if self.state == DeviceState::Running {
                    self.state = DeviceState::Degraded;
                }
            }
            DeviceEvent::Disconnected => {
                if self.state != DeviceState::Failed {
                    self.state = DeviceState::Recovering;
                }
            }
            DeviceEvent::RecoveryFailed => {
                if self.state != DeviceState::Failed {
                    self.attempts = self.attempts.saturating_add(1);
                    self.state = if self.attempts < self.max_attempts.max(1) {
                        DeviceState::Recovering
                    } else {
                        DeviceState::Failed
                    };
                }
            }
        }
        self.state
    }

    /// Records a successful control-thread reopen.
    pub fn recovered(&mut self) -> DeviceState {
        self.observe(DeviceEvent::Reconnected)
    }
}

/// Minimal backend contract for a fixed-block audio callback.
pub trait AudioDeviceBackend {
    /// Backend-specific error type.
    type Error;

    /// Current lifecycle state.
    fn state(&self) -> DeviceState;
    /// Starts or resumes callback delivery.
    fn start(&mut self) -> Result<(), Self::Error>;
    /// Pulls interleaved output in the host callback's requested size.
    fn callback(&mut self, output: &mut [Sample]) -> Result<(), Self::Error>;
    /// Drains callback/control events without blocking.
    fn poll_event(&mut self) -> Option<DeviceEvent>;
    /// Performs one bounded control-thread recovery attempt.
    fn recover(&mut self) -> Result<(), Self::Error>;
}

/// Deterministic backend used by tests and headless integration.
pub struct FakeAudioBackend {
    renderer: BlockRenderer,
    recovery: DeviceRecovery,
    events: VecDeque<DeviceEvent>,
    started: bool,
}

impl FakeAudioBackend {
    /// Creates a fake backend around the same renderer used by CPAL/offline.
    #[must_use]
    pub fn new(renderer: BlockRenderer, max_recovery_attempts: u32) -> Self {
        Self {
            renderer,
            recovery: DeviceRecovery::new(max_recovery_attempts),
            events: VecDeque::new(),
            started: false,
        }
    }

    /// Injects a deterministic host event.
    pub fn inject(&mut self, event: DeviceEvent) {
        self.events.push_back(event);
    }

    /// Borrows the recovery state machine for assertions/telemetry.
    #[must_use]
    pub const fn recovery(&self) -> DeviceRecovery {
        self.recovery
    }
}

impl AudioDeviceBackend for FakeAudioBackend {
    type Error = String;

    fn state(&self) -> DeviceState {
        self.recovery.state()
    }

    fn start(&mut self) -> Result<(), Self::Error> {
        if self.state() == DeviceState::Failed {
            return Err(String::from("fake device recovery budget exhausted"));
        }
        self.started = true;
        self.recovery.observe(DeviceEvent::Started);
        Ok(())
    }

    fn callback(&mut self, output: &mut [Sample]) -> Result<(), Self::Error> {
        if !self.started || !matches!(self.state(), DeviceState::Running | DeviceState::Degraded) {
            return Err(String::from("fake device is not running"));
        }
        self.renderer.render_interleaved(output);
        Ok(())
    }

    fn poll_event(&mut self) -> Option<DeviceEvent> {
        let event = self.events.pop_front()?;
        self.recovery.observe(event);
        match event {
            DeviceEvent::Started | DeviceEvent::Reconnected => self.started = true,
            DeviceEvent::Disconnected => self.started = false,
            DeviceEvent::Underrun { .. } | DeviceEvent::RecoveryFailed => {}
        }
        Some(event)
    }

    fn recover(&mut self) -> Result<(), Self::Error> {
        match self.recovery.state() {
            DeviceState::Failed => {
                return Err(String::from("fake recovery budget exhausted"));
            }
            DeviceState::Recovering => {}
            _ => return Err(String::from("fake device is not awaiting recovery")),
        }
        self.started = true;
        self.recovery.recovered();
        Ok(())
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
    fn callback_and_recovery_are_deterministic() {
        let mut backend = fake();
        backend.start().unwrap();
        let mut output = [0.0; 16];
        backend.callback(&mut output).unwrap();
        assert!(output.iter().all(|sample| *sample == 0.0));
        backend.inject(DeviceEvent::Underrun { frames: 8 });
        assert_eq!(
            backend.poll_event(),
            Some(DeviceEvent::Underrun { frames: 8 })
        );
        assert_eq!(backend.state(), DeviceState::Degraded);
        backend.inject(DeviceEvent::Disconnected);
        assert_eq!(backend.poll_event(), Some(DeviceEvent::Disconnected));
        assert_eq!(backend.state(), DeviceState::Recovering);
        backend.recover().unwrap();
        assert_eq!(backend.state(), DeviceState::Running);
    }

    #[test]
    fn repeated_recovery_failures_are_bounded() {
        let mut recovery = DeviceRecovery::new(2);
        recovery.observe(DeviceEvent::Disconnected);
        recovery.observe(DeviceEvent::RecoveryFailed);
        assert_eq!(recovery.state(), DeviceState::Recovering);
        recovery.observe(DeviceEvent::RecoveryFailed);
        assert_eq!(recovery.state(), DeviceState::Failed);
    }

    #[test]
    fn recovery_cannot_be_started_from_an_active_device() {
        let mut backend = fake();
        backend.start().unwrap();
        assert!(backend.recover().is_err());
        assert_eq!(backend.state(), DeviceState::Running);
    }

    #[test]
    fn reconnect_event_restores_callback_delivery() {
        let mut backend = fake();
        backend.start().unwrap();
        backend.inject(DeviceEvent::Disconnected);
        assert_eq!(backend.poll_event(), Some(DeviceEvent::Disconnected));
        let mut output = [0.0; 16];
        assert!(backend.callback(&mut output).is_err());

        backend.inject(DeviceEvent::Reconnected);
        assert_eq!(backend.poll_event(), Some(DeviceEvent::Reconnected));
        backend.callback(&mut output).unwrap();
        assert_eq!(backend.state(), DeviceState::Running);
    }

    #[test]
    fn failed_backend_cannot_bypass_the_recovery_budget_with_start() {
        let mut backend = fake();
        backend.start().unwrap();
        backend.inject(DeviceEvent::Disconnected);
        backend.poll_event();
        backend.inject(DeviceEvent::RecoveryFailed);
        backend.poll_event();
        backend.inject(DeviceEvent::RecoveryFailed);
        backend.poll_event();
        assert_eq!(backend.state(), DeviceState::Failed);
        assert!(backend.start().is_err());
        assert!(backend.callback(&mut [0.0; 16]).is_err());
    }
}
