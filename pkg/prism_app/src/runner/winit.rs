//! Windowed runner: drives an [`App`] on the OS event loop (design §10, M4).
//!
//! [`WinitRunner`] is the windowed counterpart to
//! [`HeadlessRunner`](super::HeadlessRunner). It binds the assembled [`App`] to
//! the real platform event loop through [`prism_window_winit`], requests the
//! requested window(s) on first resume (design §24.9 — one or more, for editor
//! viewports / multi-monitor), turns the frame's window events into lifecycle
//! events, runs [`App::update`] once per presented frame (paced by a
//! [`FramePacer`]), and stops when *every* window has closed or a system
//! requests exit through [`AppExitRequest`].
//!
//! # Testability
//!
//! The OS event loop only runs against a real display server, so the
//! event→action logic is factored into the pure [`window_actions`] function and
//! covered by unit tests; only [`WinitRunner::run`] touches the live loop. This
//! mirrors the split inside [`prism_window_winit`] itself.
//!
//! # Lifecycle forwarding
//!
//! The winit host trait surfaces a one-shot `resumed`, a repeatable
//! `suspended`, and a repeatable `low_memory`, so this runner forwards
//! [`Resumed`], [`Suspended`], and [`LowMemory`] (design §12/§25.1). A suspend
//! pauses virtual time and moves the app's
//! [`AppLifecycle`](crate::lifecycle::AppLifecycle) to
//! [`Suspended`](crate::lifecycle::AppLifecycle::Suspended); a memory warning
//! is forwarded so subsystems can release reclaimable resources.

use core::cell::Cell;
use std::rc::Rc;

use prism_window::WindowId;
use prism_window::envelope::WindowEventEnvelope;
use prism_window::event::WindowEvent as KernelWindowEvent;
use prism_window::geometry::PhysicalSize;
use prism_window_winit::runner::WinitApp;
use prism_window_winit::{
    FrameOutcome, WindowConfig, WinitBackend, WinitRunner as PlatformRunner,
};

use crate::app::App;
use crate::exit::{AppExit, AppExitRequest};
use crate::lifecycle::{FocusChanged, LowMemory, Resumed, Suspended, WillRenderFirstFrame};
use crate::pacing::{FrameLimit, FramePacer};

/// An action the windowed runner derives from a frame's window events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowAction {
    /// A managed window was closed or destroyed.
    Closed(WindowId),
    /// A managed window gained (`true`) or lost (`false`) focus.
    Focus(bool),
}

/// Translate a frame's kernel window events into runner [`WindowAction`]s.
///
/// Only events routed to one of the `managed` windows produce an action: a
/// `CloseRequested` or `Destroyed` maps to [`WindowAction::Closed`], and a
/// `Focused` maps to [`WindowAction::Focus`]. Events for unmanaged windows — and
/// resize / scale / cursor / occlusion events, which input and render plugins
/// consume directly — produce nothing here. Input order is preserved in the
/// output.
#[must_use]
pub fn window_actions(events: &[WindowEventEnvelope], managed: &[WindowId]) -> Vec<WindowAction> {
    let mut actions = Vec::new();
    for env in events {
        if !managed.contains(&env.window) {
            continue;
        }
        match env.event {
            KernelWindowEvent::CloseRequested | KernelWindowEvent::Destroyed => {
                actions.push(WindowAction::Closed(env.window));
            }
            KernelWindowEvent::Focused(focused) => actions.push(WindowAction::Focus(focused)),
            _ => {}
        }
    }
    actions
}

/// Drop every window named in `closed` from `open`, returning whether no window
/// is left (the runner exits when the last one goes).
fn close_windows(open: &mut Vec<WindowId>, closed: &[WindowId]) -> bool {
    open.retain(|window| !closed.contains(window));
    open.is_empty()
}

/// The window the runner requests on first resume.
#[derive(Clone, Debug)]
pub struct WindowSettings {
    /// Title bar / taskbar text.
    pub title: String,
    /// Initial inner (drawable) size in physical pixels.
    pub inner_size: PhysicalSize,
}

impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            title: "Prism".to_string(),
            inner_size: PhysicalSize::new(1280, 720),
        }
    }
}

/// The host the platform loop drives: an [`App`], the windows to open, and a
/// frame pacer.
struct WindowedHost {
    app: App,
    windows: Vec<WindowSettings>,
    /// Live windows; empty until the first resume opens them.
    open: Vec<WindowId>,
    started: bool,
    pacer: FramePacer,
    first_frame_sent: bool,
    exit: Rc<Cell<Option<AppExit>>>,
}

impl WinitApp for WindowedHost {
    fn resumed(&mut self, backend: &mut WinitBackend) {
        if !self.started {
            self.started = true;
            for settings in &self.windows {
                let config = WindowConfig::new(settings.title.clone(), settings.inner_size);
                self.open.push(backend.request_window(config));
            }
        }
        self.app.send_event(Resumed);
    }

    fn suspended(&mut self, _backend: &mut WinitBackend) {
        // A platform suspend pauses the app: `drive_app_lifecycle` (installed
        // by `App::add_lifecycle_events`) moves `AppLifecycle` to `Suspended`,
        // pauses virtual time, and stops advancing the fixed accumulator until
        // the matching `Resumed` (design §12/§25.1).
        self.app.send_event(Suspended);
    }

    fn low_memory(&mut self, _backend: &mut WinitBackend) {
        // Forward OS memory pressure so streaming/asset subsystems can drop
        // reclaimable caches (design §12, §24.7).
        self.app.send_event(LowMemory);
    }

    fn frame(&mut self, backend: &mut WinitBackend) -> FrameOutcome {
        let events = backend.drain_events();
        let actions = window_actions(&events, &self.open);
        let mut closed = Vec::new();
        for action in actions {
            match action {
                WindowAction::Closed(window) => closed.push(window),
                WindowAction::Focus(focused) => {
                    self.app.send_event(FocusChanged { focused });
                }
            }
        }
        if !closed.is_empty() && close_windows(&mut self.open, &closed) {
            // The last window is gone: request a clean exit (which still passes
            // through the §24.5 veto gate below).
            if let Some(request) = self.app.world_mut().get_resource_mut::<AppExitRequest>() {
                request.send_success();
            }
        }

        if !self.first_frame_sent {
            self.app.send_event(WillRenderFirstFrame);
            self.first_frame_sent = true;
        }

        // Async-ready gate (design §6): finish + start up once every plugin
        // reports ready; a still-initializing plugin is waited for.
        self.app.try_finish();
        self.app.update();

        if let Some(exit) = self.app.poll_exit() {
            // Honor the pipeline contract: bring the final in-flight render home
            // before the loop ends. A no-op when nothing is pipelined
            // (design §9 — runners sync once the frame loop ends).
            self.app.sync_sub_apps();
            self.exit.set(Some(exit));
            return FrameOutcome::Exit;
        }

        self.pacer.throttle();
        FrameOutcome::Continue
    }
}

/// A windowed runner that opens one or more windows and drives an [`App`].
///
/// ```no_run
/// use prism_app::{App, WinitRunner};
/// let mut app = App::new();
/// // ... add plugins and systems ...
/// let exit = WinitRunner::new(app).run();
/// # let _ = exit;
/// ```
///
/// Available only with the `winit` feature (it pulls in the platform backend).
pub struct WinitRunner {
    app: App,
    windows: Vec<WindowSettings>,
    limit: FrameLimit,
}

impl WinitRunner {
    /// A runner that opens one default 1280×720 window and runs uncapped.
    #[must_use]
    pub fn new(app: App) -> Self {
        Self {
            app,
            windows: vec![WindowSettings::default()],
            limit: FrameLimit::Off,
        }
    }

    /// Open exactly this window (title / size), replacing the default set.
    /// Builder-style.
    #[must_use]
    pub fn with_settings(mut self, settings: WindowSettings) -> Self {
        self.windows = vec![settings];
        self
    }

    /// Open several windows in order (design §24.9 — editor viewports /
    /// multi-monitor), replacing the default set. Builder-style. An empty list
    /// opens no window; the loop then runs until a system requests exit.
    #[must_use]
    pub fn with_windows(mut self, windows: Vec<WindowSettings>) -> Self {
        self.windows = windows;
        self
    }

    /// Cap the frame rate (design §13). Builder-style. The pacer is
    /// drift-free and resyncs its cadence after a long hitch.
    #[must_use]
    pub fn with_frame_limit(mut self, limit: FrameLimit) -> Self {
        self.limit = limit;
        self
    }

    /// Create the OS event loop and run the app until it exits.
    ///
    /// Must be called on the main thread. Returns the [`AppExit`] a system (or
    /// the primary window's close button) requested, or [`AppExit::error`] if
    /// the platform event loop could not be created or run.
    #[must_use]
    pub fn run(self) -> AppExit {
        let exit = Rc::new(Cell::new(None));
        let host = WindowedHost {
            app: self.app,
            windows: self.windows,
            open: Vec::new(),
            started: false,
            pacer: FramePacer::new(self.limit),
            first_frame_sent: false,
            exit: Rc::clone(&exit),
        };
        match PlatformRunner::new(host).run() {
            Ok(()) => exit.get().unwrap_or(AppExit::Success),
            Err(_) => AppExit::error(),
        }
    }
}

/// Runner function form for [`App::set_runner`](crate::app::App::set_runner):
/// `app.set_runner(run_windowed); app.run();`
///
/// Equivalent to [`WinitRunner::new(app).run()`](WinitRunner::run).
#[must_use]
pub fn run_windowed(app: App) -> AppExit {
    WinitRunner::new(app).run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_window::envelope::PlatformEventStamp;
    use prism_window::event::WindowEvent as Ev;

    fn env(window: u64, event: Ev) -> WindowEventEnvelope {
        WindowEventEnvelope::new(PlatformEventStamp::default(), WindowId(window), event)
    }

    #[test]
    fn close_on_a_managed_window_reports_it() {
        let events = [env(1, Ev::CloseRequested)];
        assert_eq!(
            window_actions(&events, &[WindowId(1)]),
            vec![WindowAction::Closed(WindowId(1))]
        );
    }

    #[test]
    fn close_on_an_unmanaged_window_is_ignored() {
        let events = [env(2, Ev::CloseRequested)];
        assert!(window_actions(&events, &[WindowId(1)]).is_empty());
    }

    #[test]
    fn focus_maps_to_focus_actions_in_order() {
        let events = [env(1, Ev::Focused(true)), env(1, Ev::Focused(false))];
        assert_eq!(
            window_actions(&events, &[WindowId(1)]),
            vec![WindowAction::Focus(true), WindowAction::Focus(false)]
        );
    }

    #[test]
    fn destroyed_window_and_unrelated_events_are_filtered() {
        let events = [
            env(1, Ev::CursorEntered),
            env(1, Ev::Destroyed),
            env(1, Ev::Minimized),
            env(1, Ev::Focused(true)),
        ];
        assert_eq!(
            window_actions(&events, &[WindowId(1)]),
            vec![WindowAction::Closed(WindowId(1)), WindowAction::Focus(true)]
        );
    }

    #[test]
    fn two_managed_windows_both_report_closes() {
        let events = [
            env(1, Ev::CloseRequested),
            env(2, Ev::CloseRequested),
            env(3, Ev::CloseRequested), // unmanaged
        ];
        assert_eq!(
            window_actions(&events, &[WindowId(1), WindowId(2)]),
            vec![
                WindowAction::Closed(WindowId(1)),
                WindowAction::Closed(WindowId(2)),
            ]
        );
    }

    #[test]
    fn close_windows_tracks_the_last_window() {
        let mut open = vec![WindowId(1), WindowId(2)];
        assert!(!close_windows(&mut open, &[WindowId(1)]));
        assert_eq!(open, vec![WindowId(2)]);
        assert!(close_windows(&mut open, &[WindowId(2)]));
        assert!(open.is_empty());
        // Closing an id that is not open is a no-op.
        let mut other = vec![WindowId(1)];
        assert!(!close_windows(&mut other, &[WindowId(9)]));
        assert_eq!(other, vec![WindowId(1)]);
    }

    #[test]
    fn platform_suspend_forwards_one_suspended_event() {
        let exit = Rc::new(Cell::new(None));
        let mut host = WindowedHost {
            app: App::new(),
            windows: Vec::new(),
            open: Vec::new(),
            started: false,
            pacer: FramePacer::new(FrameLimit::Off),
            first_frame_sent: false,
            exit,
        };
        let mut backend = WinitBackend::new();
        host.suspended(&mut backend);
        let events = host
            .app
            .world()
            .resource::<prism_ecs::event::Events<Suspended>>();
        assert_eq!(events.len(), 1, "suspend buffers one Suspended event");
    }

    #[test]
    fn platform_memory_warning_forwards_one_low_memory_event() {
        let exit = Rc::new(Cell::new(None));
        let mut host = WindowedHost {
            app: App::new(),
            windows: Vec::new(),
            open: Vec::new(),
            started: false,
            pacer: FramePacer::new(FrameLimit::Off),
            first_frame_sent: false,
            exit,
        };
        let mut backend = WinitBackend::new();
        host.low_memory(&mut backend);
        let events = host
            .app
            .world()
            .resource::<prism_ecs::event::Events<LowMemory>>();
        assert_eq!(events.len(), 1, "memory warning buffers one LowMemory event");
    }

    #[test]
    fn resume_after_suspend_re_delivers_resumed() {
        let exit = Rc::new(Cell::new(None));
        let mut host = WindowedHost {
            app: App::new(),
            windows: Vec::new(),
            open: Vec::new(),
            started: false,
            pacer: FramePacer::new(FrameLimit::Off),
            first_frame_sent: false,
            exit,
        };
        let mut backend = WinitBackend::new();
        // A suspend/resume cycle must leave the app able to run again: each
        // resume re-delivers `Resumed`, so a mobile background→foreground
        // transition is not stuck paused after the first resume.
        host.suspended(&mut backend);
        host.resumed(&mut backend);
        host.suspended(&mut backend);
        host.resumed(&mut backend);
        let resumed = host
            .app
            .world()
            .resource::<prism_ecs::event::Events<Resumed>>();
        let suspended = host
            .app
            .world()
            .resource::<prism_ecs::event::Events<Suspended>>();
        assert_eq!(resumed.len(), 2, "every resume re-delivers Resumed");
        assert_eq!(suspended.len(), 2, "every suspend delivers Suspended");
    }
}
