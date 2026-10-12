//! Windowed runner: drives an [`App`] on the OS event loop (design §10, M4).
//!
//! [`WinitRunner`] is the windowed counterpart to
//! [`HeadlessRunner`](super::HeadlessRunner). It binds the assembled [`App`] to
//! the real platform event loop through [`prism_window_winit`], requests the
//! primary window on first resume, turns the frame's window events into
//! lifecycle events, runs [`App::update`] once per presented frame (paced by a
//! [`FramePacer`]), and stops when the primary window closes or a system
//! requests exit through [`AppExitRequest`].
//!
//! # Testability
//!
//! The OS event loop only runs against a real display server, so the
//! event→action logic is factored into the pure [`window_actions`] function and
//! covered by unit tests; only [`WinitRunner::run`] touches the live loop. This
//! mirrors the split inside [`prism_window_winit`] itself.
//!
//! # Honestly deferred
//!
//! The winit host trait surfaces only a one-shot `resumed` callback, so this
//! runner forwards [`Resumed`] but has no platform *suspend* hook to forward
//! [`Suspended`]; likewise [`prism_window`] does not carry a low-memory signal
//! yet. Those remain M4 follow-ups rather than being faked here.

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
use crate::lifecycle::{FocusChanged, Resumed, WillRenderFirstFrame};
use crate::pacing::{FrameLimit, FramePacer};

/// An action the windowed runner derives from a frame's window events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowAction {
    /// The primary window was closed or destroyed: end the run.
    Exit,
    /// The primary window gained (`true`) or lost (`false`) focus.
    Focus(bool),
}

/// Translate a frame's kernel window events into runner [`WindowAction`]s.
///
/// Only events routed to `primary` produce an action: a `CloseRequested` or
/// `Destroyed` maps to [`WindowAction::Exit`], and a `Focused` maps to
/// [`WindowAction::Focus`]. Events for other windows — and resize / scale /
/// cursor / occlusion events, which input and render plugins consume directly —
/// produce nothing here. Input order is preserved in the output.
#[must_use]
pub fn window_actions(events: &[WindowEventEnvelope], primary: WindowId) -> Vec<WindowAction> {
    let mut actions = Vec::new();
    for env in events {
        if env.window != primary {
            continue;
        }
        match env.event {
            KernelWindowEvent::CloseRequested | KernelWindowEvent::Destroyed => {
                actions.push(WindowAction::Exit);
            }
            KernelWindowEvent::Focused(focused) => actions.push(WindowAction::Focus(focused)),
            _ => {}
        }
    }
    actions
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

/// The host the platform loop drives: an [`App`] plus the primary window and a
/// frame pacer.
struct WindowedHost {
    app: App,
    settings: WindowSettings,
    primary: Option<WindowId>,
    pacer: FramePacer,
    first_frame_sent: bool,
    exit: Rc<Cell<Option<AppExit>>>,
}

impl WinitApp for WindowedHost {
    fn resumed(&mut self, backend: &mut WinitBackend) {
        if self.primary.is_none() {
            let config = WindowConfig::new(self.settings.title.clone(), self.settings.inner_size);
            self.primary = Some(backend.request_window(config));
        }
        self.app.send_event(Resumed);
    }

    fn frame(&mut self, backend: &mut WinitBackend) -> FrameOutcome {
        let events = backend.drain_events();
        if let Some(primary) = self.primary {
            for action in window_actions(&events, primary) {
                match action {
                    WindowAction::Exit => {
                        if let Some(request) = self.app.world_mut().get_resource_mut::<AppExitRequest>()
                        {
                            request.send_success();
                        }
                    }
                    WindowAction::Focus(focused) => {
                        self.app.send_event(FocusChanged { focused });
                    }
                }
            }
        }

        if !self.first_frame_sent {
            self.app.send_event(WillRenderFirstFrame);
            self.first_frame_sent = true;
        }

        self.app.update();

        if let Some(exit) = self.app.poll_exit() {
            self.exit.set(Some(exit));
            return FrameOutcome::Exit;
        }

        self.pacer.throttle();
        FrameOutcome::Continue
    }
}

/// A windowed runner that opens one primary window and drives an [`App`].
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
    settings: WindowSettings,
    limit: FrameLimit,
}

impl WinitRunner {
    /// A runner that opens a default 1280×720 window and runs uncapped.
    #[must_use]
    pub fn new(app: App) -> Self {
        Self {
            app,
            settings: WindowSettings::default(),
            limit: FrameLimit::Off,
        }
    }

    /// Override the initial window (title / size). Builder-style.
    #[must_use]
    pub fn with_settings(mut self, settings: WindowSettings) -> Self {
        self.settings = settings;
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
            settings: self.settings,
            primary: None,
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
    fn close_requested_on_primary_requests_exit() {
        let events = [env(1, Ev::CloseRequested)];
        assert_eq!(window_actions(&events, WindowId(1)), vec![WindowAction::Exit]);
    }

    #[test]
    fn close_on_a_secondary_window_is_ignored() {
        let events = [env(2, Ev::CloseRequested)];
        assert!(window_actions(&events, WindowId(1)).is_empty());
    }

    #[test]
    fn focus_maps_to_focus_actions_in_order() {
        let events = [env(1, Ev::Focused(true)), env(1, Ev::Focused(false))];
        assert_eq!(
            window_actions(&events, WindowId(1)),
            vec![WindowAction::Focus(true), WindowAction::Focus(false)]
        );
    }

    #[test]
    fn destroyed_primary_exits_and_unrelated_events_are_skipped() {
        let events = [
            env(1, Ev::CursorEntered),
            env(1, Ev::Destroyed),
            env(1, Ev::Minimized),
            env(1, Ev::Focused(true)),
        ];
        assert_eq!(
            window_actions(&events, WindowId(1)),
            vec![WindowAction::Exit, WindowAction::Focus(true)]
        );
    }
}
