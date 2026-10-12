//! Application lifecycle: platform events and the graceful-shutdown path
//! (design §12, §21, §24.5).
//!
//! A console/mobile certification target must react to the host suspending the
//! process, reclaiming memory, or moving focus, and must shut down *cleanly* —
//! saving state, disconnecting, flushing to disk, and releasing devices — rather
//! than being killed mid-write. This module provides the two halves of that:
//!
//! 1. **Lifecycle events** — plain [`Event`] types
//!    ([`Suspended`], [`Resumed`], [`LowMemory`], [`FocusChanged`],
//!    [`WillRenderFirstFrame`], plus the GPU-recovery pair [`DeviceLost`] /
//!    [`DeviceRestored`]) a runner or render layer emits and systems read, plus
//!    an [`AppLifecycle`] resource tracking the coarse run state. The
//!    `drive_app_lifecycle` handler turns those edges into run-state
//!    transitions and pauses/unpauses the virtual clock.
//! 2. **Graceful shutdown** — [`App::run_shutdown`](crate::app::App::run_shutdown)
//!    runs the dedicated [`Shutdown`](crate::schedule::Shutdown) schedule once,
//!    drains deferred work, and tears plugins down in reverse registration order
//!    via [`Plugin::shutdown`](crate::plugin::Plugin::shutdown).
//!
//! # Who emits lifecycle events
//!
//! The windowed runner (design §10) forwards the real OS suspend / resume /
//! focus / low-memory notifications through [`Suspended`], [`Resumed`],
//! [`FocusChanged`], and [`LowMemory`]. [`DeviceLost`] / [`DeviceRestored`] are
//! emitted by the **render layer** — the RHI reports a lost device/context and
//! the render plugin relays it — so a GPU-resource recovery window pauses
//! simulation exactly as a suspend does. Any runner, plugin, or test can also
//! emit a lifecycle event through [`App::send_event`](crate::app::App::send_event).
//!
//! # `cleanup` vs. `shutdown`
//!
//! These are **two different hooks** and are deliberately not conflated:
//!
//! * [`Plugin::cleanup`](crate::plugin::Plugin::cleanup) runs once *after
//!   startup* (forward order) to drop build-time scratch resources.
//! * [`Plugin::shutdown`](crate::plugin::Plugin::shutdown) runs once *at exit*
//!   (reverse order) to persist and release.
//!
//! The design's "退出时逆序 cleanup" (design §12/§21) refers to teardown, which
//! this crate implements as the distinct `shutdown` hook rather than re-running
//! the post-startup `cleanup`.

use prism_ecs::event::{Event, EventCursor, Events};
use prism_ecs::resource::Resource;
use prism_ecs::system::{Local, Res, ResMut};

use crate::time::EngineClocks;

/// The host asked the process to suspend (mobile backgrounding, console sleep).
///
/// Systems reacting to this should pause simulation and release transient GPU
/// resources; the matching [`Resumed`] later rebuilds them. Emitted by a
/// platform runner (design §12).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Suspended;

impl Event for Suspended {}

/// The host resumed a previously [`Suspended`] process.
///
/// Systems reacting to this rebuild swap chains / graphics contexts and resume
/// simulation (design §12). The `drive_app_lifecycle` handler unpauses the
/// virtual clock on this edge; the accompanying first-frame delta clamp lives
/// in `prism_time` (design §25.1), which `prism_app` drives rather than
/// implements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resumed;

impl Event for Resumed {}

/// The host signalled memory pressure (mobile).
///
/// Systems reacting to this proactively unload streamed cells / caches to avoid
/// an OS kill (design §12).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowMemory;

impl Event for LowMemory {}

/// Window/application focus changed (design §12).
///
/// `focused` is `true` when the app gained focus and `false` when it lost
/// focus; a system might throttle work or mute audio while unfocused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FocusChanged {
    /// Whether the app now has focus.
    pub focused: bool,
}

impl Event for FocusChanged {}

/// The GPU device (or rendering context) was lost and must be rebuilt
/// (design §12 / §24.3 device-loss recovery).
///
/// Emitted by the render plugin when the RHI reports a lost device/context.
/// The `drive_app_lifecycle` handler moves [`AppLifecycle`] to
/// [`Recovering`](AppLifecycle::Recovering) and pauses the virtual clock while
/// GPU resources are rebuilt, then back to
/// [`Running`](AppLifecycle::Running) on the matching [`DeviceRestored`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLost;

impl Event for DeviceLost {}

/// The GPU device/context was rebuilt after a [`DeviceLost`]; rendering resumes.
///
/// Emitted by the render plugin once it has recreated the device and its
/// resources — the closing edge of the device-loss recovery window (design
/// §12 / §24.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceRestored;

impl Event for DeviceRestored {}

/// Emitted once, immediately before the first rendered frame (design §12).
///
/// A dismiss-the-splash / start-the-clock hook: systems that must run exactly
/// once at the simulation→presentation boundary (not at [`Startup`], which is
/// pre-first-frame) react here.
///
/// [`Startup`]: crate::schedule::Startup
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WillRenderFirstFrame;

impl Event for WillRenderFirstFrame {}

/// The coarse application run state (design §12).
///
/// Installed by [`App::add_lifecycle_events`](crate::app::App::add_lifecycle_events)
/// (defaulting to [`Running`](AppLifecycle::Running)) and advanced to
/// [`WillExit`](AppLifecycle::WillExit) by
/// [`App::run_shutdown`](crate::app::App::run_shutdown). The
/// `drive_app_lifecycle` handler (also registered by `add_lifecycle_events`)
/// moves it between [`Running`](AppLifecycle::Running),
/// [`Suspended`](AppLifecycle::Suspended), and
/// [`Recovering`](AppLifecycle::Recovering) as the suspend/resume and
/// device-lost/restored edges arrive; systems can read it to gate work (e.g.
/// skip rendering while suspended or while the GPU device is being rebuilt).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AppLifecycle {
    /// Not yet started (pre-first-frame).
    #[default]
    Idle,
    /// The normal running state.
    Running,
    /// Suspended by the host; simulation paused, transient resources released.
    Suspended,
    /// The GPU device/context was lost; simulation is paused while the render
    /// layer rebuilds it (design §12 / §24.3). Entered on [`DeviceLost`] and
    /// left on [`DeviceRestored`].
    Recovering,
    /// A graceful exit is in progress; shutdown systems and plugin teardown run.
    WillExit,
}

impl Resource for AppLifecycle {}

impl AppLifecycle {
    /// Whether the app is in the normal [`Running`](AppLifecycle::Running) state.
    #[must_use]
    pub fn is_running(self) -> bool {
        matches!(self, AppLifecycle::Running)
    }

    /// Whether the app is [`Suspended`](AppLifecycle::Suspended).
    #[must_use]
    pub fn is_suspended(self) -> bool {
        matches!(self, AppLifecycle::Suspended)
    }

    /// Whether the app is [`Recovering`](AppLifecycle::Recovering) from a lost
    /// GPU device.
    #[must_use]
    pub fn is_recovering(self) -> bool {
        matches!(self, AppLifecycle::Recovering)
    }

    /// Whether a graceful exit is in progress
    /// ([`WillExit`](AppLifecycle::WillExit)).
    #[must_use]
    pub fn is_exiting(self) -> bool {
        matches!(self, AppLifecycle::WillExit)
    }
}

/// Drive [`AppLifecycle`] and the main world's virtual clock from the
/// [`Suspended`] / [`Resumed`] platform events (design §12, §25.1).
///
/// Registered in the [`First`](crate::schedule::First) phase by
/// [`App::add_lifecycle_events`](crate::app::App::add_lifecycle_events), so it
/// runs once per frame after the event buffers rotate but before any user
/// phase observes the world. On each frame it folds the frame's lifecycle
/// events into a coarse run-state transition:
///
/// * A [`Suspended`] event (while [`Running`](AppLifecycle::Running)) moves the
///   state to [`Suspended`](AppLifecycle::Suspended) and **pauses** the virtual
///   clock, so simulation freezes: a paused virtual clock reports a zero delta,
///   which in turn feeds zero into the fixed-step accumulator, so no fixed
///   steps run while backgrounded.
/// * A [`Resumed`] event (while
///   [`Suspended`](AppLifecycle::Suspended)) moves the state back to
///   [`Running`](AppLifecycle::Running) and **unpauses** the virtual clock.
/// * A [`DeviceLost`] event moves the state to
///   [`Recovering`](AppLifecycle::Recovering) and pauses the virtual clock while
///   the render layer rebuilds the GPU device; the matching [`DeviceRestored`]
///   moves it back to [`Running`](AppLifecycle::Running) and unpauses.
///
/// A *restoring* edge (process [`Resumed`] or [`DeviceRestored`]) wins over a
/// *pausing* edge ([`Suspended`] or [`DeviceLost`]) in the same frame, so a
/// transient blip never leaves the app wedged in a paused state.
///
/// # Who owns the pause
///
/// The handler only unpauses a clock *it* paused: it remembers (in a
/// [`Local`]) whether the suspend transition is what paused the clock, and on
/// resume restores it only then. A clock the game had already paused for its
/// own reasons (a pause menu) is therefore left paused across a
/// suspend/resume, rather than being silently unpaused on resume.
///
/// # First-frame delta on resume
///
/// Pausing/unpausing is all this crate does; it does **not** itself clamp the
/// large real delta that accrues while backgrounded. That clamp is the virtual
/// clock's own max-delta guard in `prism_time` (design §25.1): on the resumed
/// frame the real clock reports the whole suspend gap, and the virtual clock
/// clamps it before it reaches the fixed accumulator. `prism_app` only relays
/// the suspend/resume edge; `prism_time` owns the delta policy.
///
/// # Both edges in one frame
///
/// If a frame somehow carries both a [`Suspended`] and a [`Resumed`] event,
/// resume wins (the app ends up [`Running`](AppLifecycle::Running)): a
/// transient background blip must never leave the app wedged in
/// [`Suspended`](AppLifecycle::Suspended). A graceful exit already in progress
/// ([`WillExit`](AppLifecycle::WillExit)) is never pulled back to a running or
/// suspended state.
pub(crate) fn drive_app_lifecycle(
    mut suspended_cursor: Local<EventCursor<Suspended>>,
    suspended_events: Res<Events<Suspended>>,
    mut resumed_cursor: Local<EventCursor<Resumed>>,
    resumed_events: Res<Events<Resumed>>,
    mut lost_cursor: Local<EventCursor<DeviceLost>>,
    lost_events: Res<Events<DeviceLost>>,
    mut restored_cursor: Local<EventCursor<DeviceRestored>>,
    restored_events: Res<Events<DeviceRestored>>,
    mut lifecycle: ResMut<AppLifecycle>,
    clocks: Option<ResMut<EngineClocks>>,
    mut we_paused: Local<bool>,
) {
    // Always drain every cursor so they stay current frame to frame, even on
    // frames where no transition results.
    let want_suspend = suspended_cursor.read(&suspended_events).count() > 0;
    let want_resume = resumed_cursor.read(&resumed_events).count() > 0;
    let want_lost = lost_cursor.read(&lost_events).count() > 0;
    let want_restored = restored_cursor.read(&restored_events).count() > 0;

    // A graceful shutdown already underway is terminal: do not resurrect it.
    if lifecycle.is_exiting() {
        return;
    }

    // Resolve this frame's target state. A *restoring* edge (process resume or
    // device restored) wins over a *pausing* edge in the same frame, so a
    // transient blip never leaves the app wedged in a paused state.
    // [`AppLifecycle::Suspended`] and [`AppLifecycle::Recovering`] are both
    // "paused" states with distinct causes.
    let target = if want_resume || want_restored {
        Some(AppLifecycle::Running)
    } else if want_lost {
        Some(AppLifecycle::Recovering)
    } else if want_suspend {
        Some(AppLifecycle::Suspended)
    } else {
        None
    };
    let Some(target) = target else {
        return;
    };
    if *lifecycle == target {
        return;
    }
    *lifecycle = target;

    let mut clocks = clocks;
    match target {
        AppLifecycle::Running => {
            // Only unpause a clock *we* paused; a game pause-menu pause is kept.
            if *we_paused {
                if let Some(clocks) = clocks.as_mut() {
                    clocks.virtual_time_mut().unpause();
                }
                *we_paused = false;
            }
        }
        AppLifecycle::Suspended | AppLifecycle::Recovering => {
            if let Some(clocks) = clocks.as_mut()
                && !clocks.virtual_time().is_paused()
            {
                clocks.virtual_time_mut().pause();
                *we_paused = true;
            }
        }
        AppLifecycle::Idle | AppLifecycle::WillExit => {}
    }
}
