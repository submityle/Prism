//! Device-loss bridge: relay a lost GPU device from the render world to the app
//! (design §12 / §24.3).
//!
//! The GPU runtime lives in the render world; the [`AppLifecycle`] that must
//! pause simulation lives in the main world. The render world cannot write into
//! the main world — the one-way extract invariant (design §21) forbids it — so
//! recovery outcomes travel through a plugin-owned queue shared by both worlds.
//! The render driver records loss, failures and successful rebuilds; the main
//! world drains them into lifecycle events and [`DeviceRecoveryStatus`].
//!
//! [`AppLifecycle`]: prism_app::lifecycle::AppLifecycle

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::sync::Arc;
use std::sync::Mutex;

use prism_app::lifecycle::{DeviceLost, DeviceRestored};
use prism_ecs::event::Events;
use prism_ecs::prelude::Resource;
use prism_ecs::system::{Res, ResMut};
use prism_render_driver::DeviceLostReason;
use prism_render_runtime::RecoveryReport;

#[derive(Clone, Debug)]
enum RecoveryEvent {
    Lost(DeviceLostReason),
    Failed {
        attempts: u64,
        error: String,
    },
    Restored {
        attempts: u64,
        report: RecoveryReport,
    },
}

/// Main-world diagnostics for the most recent device-loss recovery.
#[derive(Clone, Debug, Default, PartialEq, Eq, Resource)]
pub struct DeviceRecoveryStatus {
    /// Whether rendering is paused pending a successful recovery.
    pub recovering: bool,
    /// Attempts made since the most recent loss, including the successful one.
    pub attempts: u64,
    /// Backend reason for the most recent loss.
    pub reason: Option<DeviceLostReason>,
    /// Most recent failure; cleared after a successful recovery or a new loss.
    pub last_error: Option<String>,
    /// Most recent successful recovery; cleared on a new loss.
    pub last_report: Option<RecoveryReport>,
}

/// A plugin-owned queue carrying recovery outcomes into the main world.
///
/// The lock is held only for a `take`/`report`, never across GPU work.
#[derive(Clone, Default)]
pub struct DeviceLossSignal(Arc<Mutex<VecDeque<RecoveryEvent>>>);

impl Resource for DeviceLossSignal {}

impl DeviceLossSignal {
    /// A fresh, empty signal.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a device loss without overwriting an unreported recovery result.
    ///
    /// Called by the render driver; a producer (a test, or
    /// a backend that surfaces loss out of band) can also report directly.
    pub fn report(&self, reason: DeviceLostReason) {
        lock(&self.0).push_back(RecoveryEvent::Lost(reason));
    }

    /// Take the oldest loss reason, preserving other recovery results.
    #[must_use]
    pub fn take(&self) -> Option<DeviceLostReason> {
        let mut events = lock(&self.0);
        let index = events
            .iter()
            .position(|event| matches!(event, RecoveryEvent::Lost(_)))?;
        match events.remove(index) {
            Some(RecoveryEvent::Lost(reason)) => Some(reason),
            _ => None,
        }
    }

    pub(crate) fn failed(&self, attempts: u64, error: String) {
        lock(&self.0).push_back(RecoveryEvent::Failed { attempts, error });
    }

    pub(crate) fn restored(&self, attempts: u64, report: RecoveryReport) {
        lock(&self.0).push_back(RecoveryEvent::Restored { attempts, report });
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Drain render recovery outcomes into main-world lifecycle events/diagnostics.
pub(crate) fn forward_device_recovery(
    signal: Res<DeviceLossSignal>,
    mut status: ResMut<DeviceRecoveryStatus>,
    mut lost: ResMut<Events<DeviceLost>>,
    mut restored: ResMut<Events<DeviceRestored>>,
) {
    let events = core::mem::take(&mut *lock(&signal.0));
    for event in events {
        match event {
            RecoveryEvent::Lost(reason) => {
                *status = DeviceRecoveryStatus {
                    recovering: true,
                    reason: Some(reason),
                    ..Default::default()
                };
                lost.send(DeviceLost);
            }
            RecoveryEvent::Failed { attempts, error } => {
                status.recovering = true;
                status.attempts = attempts;
                status.last_error = Some(error);
            }
            RecoveryEvent::Restored { attempts, report } => {
                status.recovering = false;
                status.attempts = attempts;
                status.last_report = Some(report);
                status.last_error = None;
                restored.send(DeviceRestored);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_round_trips_and_clears() {
        let signal = DeviceLossSignal::new();
        assert!(signal.take().is_none());
        signal.report(DeviceLostReason::Destroyed);
        assert_eq!(signal.take(), Some(DeviceLostReason::Destroyed));
        assert!(signal.take().is_none(), "take clears the cell");
    }

    #[test]
    fn loss_reasons_are_preserved_in_order() {
        let signal = DeviceLossSignal::new();
        signal.report(DeviceLostReason::Unknown);
        signal.report(DeviceLostReason::Destroyed);
        assert_eq!(signal.take(), Some(DeviceLostReason::Unknown));
        assert_eq!(signal.take(), Some(DeviceLostReason::Destroyed));
    }
}
