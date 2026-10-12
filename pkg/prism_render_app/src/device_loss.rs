//! Device-loss bridge: relay a lost GPU device from the render world to the app
//! (design §12 / §24.3).
//!
//! The GPU runtime lives in the render world; the [`AppLifecycle`] that must
//! pause simulation lives in the main world. The render world cannot write into
//! the main world — the one-way extract invariant (design §21) forbids it — so
//! the loss travels through a small plugin-owned cell both worlds hold a clone
//! of: a render-world system polls the runtime and records the reason, a
//! main-world system drains it and emits [`DeviceLost`].
//!
//! [`AppLifecycle`]: prism_app::lifecycle::AppLifecycle

use alloc::sync::Arc;
use std::sync::Mutex;

use prism_app::lifecycle::DeviceLost;
use prism_ecs::event::Events;
use prism_ecs::prelude::Resource;
use prism_ecs::system::{Res, ResMut};
use prism_render_driver::{DeviceLostReason, FrameSync, Instance};

use crate::runtime_resource::RenderRuntimeResource;

/// A plugin-owned cell carrying a device-loss reason from the render world to
/// the main world.
///
/// The lock is held only for a `take`/`report`, never across GPU work.
#[derive(Clone, Default)]
pub struct DeviceLossSignal(Arc<Mutex<Option<DeviceLostReason>>>);

impl Resource for DeviceLossSignal {}

impl DeviceLossSignal {
    /// A fresh, empty signal.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a device loss, overwriting any earlier unreported reason.
    ///
    /// Called by [`poll_device_loss`] each render frame; a producer (a test, or
    /// a backend that surfaces loss out of band) can also report directly.
    pub fn report(&self, reason: DeviceLostReason) {
        *lock(&self.0) = Some(reason);
    }

    /// Take the recorded reason, if any (clearing the cell).
    #[must_use]
    pub fn take(&self) -> Option<DeviceLostReason> {
        lock(&self.0).take()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Render-world system: poll the runtime for a lost device and record it.
///
/// A backend that cannot observe loss (null/headless) returns `None` from
/// [`RenderRuntime::device_lost`](prism_render_runtime::RenderRuntime::device_lost),
/// so this is a cheap check that never fires there.
pub(crate) fn poll_device_loss<I>(
    runtime: Res<RenderRuntimeResource<I>>,
    signal: Res<DeviceLossSignal>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    if let Some(reason) = runtime.runtime().device_lost() {
        signal.report(reason);
    }
}

/// Main-world system: drain the signal and emit [`DeviceLost`] into the app.
pub(crate) fn forward_device_loss(
    signal: Res<DeviceLossSignal>,
    mut events: ResMut<Events<DeviceLost>>,
) {
    if signal.take().is_some() {
        events.send(DeviceLost);
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
    fn a_newer_reason_overwrites_an_unreported_one() {
        let signal = DeviceLossSignal::new();
        signal.report(DeviceLostReason::Unknown);
        signal.report(DeviceLostReason::Destroyed);
        assert_eq!(signal.take(), Some(DeviceLostReason::Destroyed));
    }
}
