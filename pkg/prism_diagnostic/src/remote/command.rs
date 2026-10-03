//! Remote commands and the runtime-tuning state they drive (`remote` feature).
//!
//! A remote panel steers a running engine by sending [`RemoteCommand`]s. Each
//! command is applied to a shared [`RuntimeControls`], which holds the live
//! tuning state (log level, whether sinks are enabled, sampling ratio, pending
//! capture requests) behind atomics so it can be read on the hot path from any
//! thread without locking. Applying a command returns a [`CommandOutcome`]
//! describing the effect, which the server can echo back to the panel.
//!
//! The commands here deliberately cover the design's remote-tuning surface
//! (design §15): change the log filter, toggle sinks, adjust sampling, trigger a
//! frame capture, request a snapshot, and shut the session down.

extern crate alloc;

use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::filter::{max_level, set_max_level};
use crate::model::Level;

/// A command sent from a remote panel to the running engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoteCommand {
    /// Change the runtime log-level threshold.
    SetLogLevel(Level),
    /// Enable or disable event sinks (log/metric fan-out).
    ToggleSink {
        /// `true` enables sinks, `false` silences them.
        enabled: bool,
    },
    /// Set a deterministic `numerator`-of-`denominator` sampling ratio.
    SetSampling {
        /// How many out of every `denominator` samples are kept.
        numerator: u32,
        /// Sampling window size (clamped to at least 1).
        denominator: u32,
    },
    /// Request that the engine capture the next frame's timeline.
    TriggerCapture,
    /// Ask the engine to push a fresh snapshot of its state.
    RequestSnapshot,
    /// Ask the session to shut down cleanly.
    Shutdown,
}

/// The effect of applying a [`RemoteCommand`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandOutcome {
    /// The log level changed from `old` to `new`.
    LogLevelChanged {
        /// Previous threshold.
        old: Level,
        /// New threshold.
        new: Level,
    },
    /// Sinks were toggled to the given enabled state.
    SinkToggled(bool),
    /// The sampling ratio changed to `numerator`/`denominator` (post-clamp).
    SamplingChanged {
        /// Kept-sample count per window.
        numerator: u32,
        /// Window size.
        denominator: u32,
    },
    /// A capture was requested; `pending` is the new outstanding-request count.
    CaptureRequested {
        /// Total outstanding capture requests after this one.
        pending: u64,
    },
    /// A state snapshot was requested.
    SnapshotRequested,
    /// The session is shutting down.
    ShuttingDown,
}

/// Shared, lock-free runtime tuning state driven by [`RemoteCommand`]s.
///
/// Cloneable semantics are provided by wrapping in an `Arc` at the call site;
/// the struct itself is not `Clone` because its atomics are the shared truth.
#[derive(Debug)]
pub struct RuntimeControls {
    sink_enabled: AtomicBool,
    capture_requests: AtomicU64,
    sampling_numerator: AtomicU32,
    sampling_denominator: AtomicU32,
    sample_counter: AtomicU64,
}

impl Default for RuntimeControls {
    fn default() -> Self {
        Self {
            sink_enabled: AtomicBool::new(true),
            capture_requests: AtomicU64::new(0),
            sampling_numerator: AtomicU32::new(1),
            sampling_denominator: AtomicU32::new(1),
            sample_counter: AtomicU64::new(0),
        }
    }
}

impl RuntimeControls {
    /// Create controls in the default "everything on, sample 1/1" state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether event sinks are currently enabled.
    pub fn sinks_enabled(&self) -> bool {
        self.sink_enabled.load(Ordering::Acquire)
    }

    /// Outstanding frame-capture request count.
    pub fn pending_captures(&self) -> u64 {
        self.capture_requests.load(Ordering::Acquire)
    }

    /// Consume one outstanding capture request if any exist, returning whether
    /// a capture should run this frame.
    pub fn take_capture_request(&self) -> bool {
        self.capture_requests
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                cur.checked_sub(1)
            })
            .is_ok()
    }

    /// Current sampling ratio as `(numerator, denominator)`.
    pub fn sampling(&self) -> (u32, u32) {
        (
            self.sampling_numerator.load(Ordering::Acquire),
            self.sampling_denominator.load(Ordering::Acquire),
        )
    }

    /// Deterministically decide whether the next sample is kept, advancing the
    /// internal window counter. With ratio `n/d`, exactly `n` of every `d`
    /// consecutive calls return `true`.
    pub fn should_sample(&self) -> bool {
        let (num, den) = self.sampling();
        if num >= den {
            return true;
        }
        if num == 0 {
            return false;
        }
        let idx = self.sample_counter.fetch_add(1, Ordering::AcqRel);
        (idx % den as u64) < num as u64
    }

    /// Apply `command`, mutating the shared state and returning its effect.
    pub fn apply(&self, command: RemoteCommand) -> CommandOutcome {
        match command {
            RemoteCommand::SetLogLevel(level) => {
                let old = max_level();
                set_max_level(level);
                CommandOutcome::LogLevelChanged { old, new: level }
            }
            RemoteCommand::ToggleSink { enabled } => {
                self.sink_enabled.store(enabled, Ordering::Release);
                CommandOutcome::SinkToggled(enabled)
            }
            RemoteCommand::SetSampling {
                numerator,
                denominator,
            } => {
                let den = denominator.max(1);
                let num = numerator.min(den);
                self.sampling_denominator.store(den, Ordering::Release);
                self.sampling_numerator.store(num, Ordering::Release);
                self.sample_counter.store(0, Ordering::Release);
                CommandOutcome::SamplingChanged {
                    numerator: num,
                    denominator: den,
                }
            }
            RemoteCommand::TriggerCapture => {
                let pending = self.capture_requests.fetch_add(1, Ordering::AcqRel) + 1;
                CommandOutcome::CaptureRequested { pending }
            }
            RemoteCommand::RequestSnapshot => CommandOutcome::SnapshotRequested,
            RemoteCommand::Shutdown => CommandOutcome::ShuttingDown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_log_level_reports_old_and_new_and_takes_effect() {
        set_max_level(Level::Info);
        let controls = RuntimeControls::new();
        let outcome = controls.apply(RemoteCommand::SetLogLevel(Level::Debug));
        assert_eq!(
            outcome,
            CommandOutcome::LogLevelChanged {
                old: Level::Info,
                new: Level::Debug,
            }
        );
        assert_eq!(max_level(), Level::Debug);
        set_max_level(Level::Info);
    }

    #[test]
    fn toggle_sink_updates_state() {
        let controls = RuntimeControls::new();
        assert!(controls.sinks_enabled());
        assert_eq!(
            controls.apply(RemoteCommand::ToggleSink { enabled: false }),
            CommandOutcome::SinkToggled(false)
        );
        assert!(!controls.sinks_enabled());
    }

    #[test]
    fn sampling_is_clamped_and_deterministic() {
        let controls = RuntimeControls::new();
        let outcome = controls.apply(RemoteCommand::SetSampling {
            numerator: 10,
            denominator: 4,
        });
        // numerator clamped to denominator -> keep everything.
        assert_eq!(
            outcome,
            CommandOutcome::SamplingChanged {
                numerator: 4,
                denominator: 4,
            }
        );
        for _ in 0..8 {
            assert!(controls.should_sample());
        }

        controls.apply(RemoteCommand::SetSampling {
            numerator: 1,
            denominator: 4,
        });
        let kept: usize = (0..8).filter(|_| controls.should_sample()).count();
        assert_eq!(kept, 2); // 1 of every 4, over 8 calls.

        controls.apply(RemoteCommand::SetSampling {
            numerator: 0,
            denominator: 4,
        });
        assert!(!controls.should_sample());
    }

    #[test]
    fn capture_requests_accumulate_and_drain() {
        let controls = RuntimeControls::new();
        assert_eq!(
            controls.apply(RemoteCommand::TriggerCapture),
            CommandOutcome::CaptureRequested { pending: 1 }
        );
        assert_eq!(
            controls.apply(RemoteCommand::TriggerCapture),
            CommandOutcome::CaptureRequested { pending: 2 }
        );
        assert_eq!(controls.pending_captures(), 2);
        assert!(controls.take_capture_request());
        assert!(controls.take_capture_request());
        assert!(!controls.take_capture_request());
        assert_eq!(controls.pending_captures(), 0);
    }

    #[test]
    fn snapshot_and_shutdown_outcomes() {
        let controls = RuntimeControls::new();
        assert_eq!(
            controls.apply(RemoteCommand::RequestSnapshot),
            CommandOutcome::SnapshotRequested
        );
        assert_eq!(
            controls.apply(RemoteCommand::Shutdown),
            CommandOutcome::ShuttingDown
        );
    }
}
