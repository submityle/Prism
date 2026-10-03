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
//!    [`WillRenderFirstFrame`]) a platform runner emits and systems read, plus
//!    an [`AppLifecycle`] resource tracking the coarse run state.
//! 2. **Graceful shutdown** — [`App::run_shutdown`](crate::app::App::run_shutdown)
//!    runs the dedicated [`Shutdown`](crate::schedule::Shutdown) schedule once,
//!    drains deferred work, and tears plugins down in reverse registration order
//!    via [`Plugin::shutdown`](crate::plugin::Plugin::shutdown).
//!
//! # Who emits lifecycle events
//!
//! The events are *defined and deliverable* here, but the authoritative emitter
//! is a platform runner. The windowed `WinitRunner` (design §10) that forwards
//! real OS suspend/resume/focus/low-memory notifications is a later M4
//! increment and is **honestly absent** today — not stubbed. In the meantime a
//! runner, a test, or an integration layer can emit any lifecycle event through
//! [`App::send_event`](crate::app::App::send_event), and systems can react to it
//! exactly as they will under the real platform runner. The seam is real; only
//! the OS wiring is deferred and documented as such.
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

use prism_ecs::event::Event;
use prism_ecs::resource::Resource;

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
/// simulation (design §12). The accompanying first-frame delta clamp lives in
/// `prism_time` (design §25.1); this crate only relays the event.
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
/// [`App::run_shutdown`](crate::app::App::run_shutdown). A platform runner that
/// forwards [`Suspended`] / [`Resumed`] also moves it in and out of
/// [`Suspended`](AppLifecycle::Suspended); systems can read it to gate work
/// (e.g. skip rendering while suspended).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AppLifecycle {
    /// Not yet started (pre-first-frame).
    #[default]
    Idle,
    /// The normal running state.
    Running,
    /// Suspended by the host; simulation paused, transient resources released.
    Suspended,
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

    /// Whether a graceful exit is in progress
    /// ([`WillExit`](AppLifecycle::WillExit)).
    #[must_use]
    pub fn is_exiting(self) -> bool {
        matches!(self, AppLifecycle::WillExit)
    }
}
