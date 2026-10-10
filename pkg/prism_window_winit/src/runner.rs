//! The real OS event loop: owns winit, drives the kernel stream, and applies
//! the three-state sync model once per frame (§9, §10).
//!
//! The split of responsibility is:
//!
//! * [`WinitBackend`] holds all mutable platform state — the live windows, the
//!   id map, per-window [`WindowSync`], the pending create/destroy queues, and
//!   the frame event buffer. It is the surface the host engine talks to.
//! * [`WinitRunner`] glues a host [`WinitApp`] to winit's
//!   [`ApplicationHandler`]. On each `about_to_wait` it coalesces the frame's
//!   events, lets the host read them and mutate desired state, then flushes the
//!   minimal command diff back to the OS.
//!
//! The host never touches winit types: it requests windows with
//! [`WindowConfig`], mutates [`WindowConfig`] through [`WinitBackend::config_mut`],
//! reads the kernel's `Copy` event stream, and asks for raw-window-handles when
//! it needs to build a render surface. Everything platform-specific is confined
//! to this crate.

use std::collections::HashMap;
use alloc::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent as WinitWindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window as WinitWindow, WindowId as WinitWindowId};

use prism_window::envelope::{
    EventSource, MonotonicTimestamp, PlatformEventSequence, PlatformEventStamp, WindowEventEnvelope,
};
use prism_window::event::WindowEvent as KernelWindowEvent;
use prism_window::WindowId;

use crate::command;
use crate::convert;
use crate::error::{BackendError, BackendResult};
use crate::id::WindowIdMap;
use crate::sync::{RealizedState, WindowConfig, WindowSync};
use crate::{batch, translate};

/// What the host wants the runner to do after a frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameOutcome {
    /// Keep pumping the event loop.
    Continue,
    /// Tear the event loop down and return from [`WinitRunner::run`].
    Exit,
}

/// A single live platform window and its synchronizer.
struct LiveWindow {
    /// The OS window. Reference-counted so the renderer can hold a handle for
    /// surface creation independently of the backend's bookkeeping.
    window: Arc<WinitWindow>,
    /// Desired/Applied/Realized state for this window.
    sync: WindowSync,
}

/// All mutable platform-backend state. This is what the host engine drives.
///
/// The host allocates logical windows with [`Self::request_window`], edits
/// their desired configuration via [`Self::config_mut`], consumes the kernel
/// event stream with [`Self::drain_events`], and obtains render-surface handles
/// through [`Self::window`]. Window creation and destruction are deferred to a
/// point where the winit [`ActiveEventLoop`] is available (the runner performs
/// them), so these calls only enqueue intent.
pub struct WinitBackend {
    ids: WindowIdMap,
    windows: HashMap<WindowId, LiveWindow>,
    pending_create: Vec<(WindowId, WindowConfig)>,
    pending_destroy: Vec<WindowId>,
    events: Vec<WindowEventEnvelope>,
    sequence: PlatformEventSequence,
    clock: Instant,
    next_id: u64,
}

impl Default for WinitBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WinitBackend {
    /// A fresh backend with no windows and an empty event buffer. The internal
    /// monotonic clock starts now; all event timestamps are nanoseconds since
    /// this point so a recording is byte-stable relative to its own start.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ids: WindowIdMap::new(),
            windows: HashMap::new(),
            pending_create: Vec::new(),
            pending_destroy: Vec::new(),
            events: Vec::new(),
            sequence: PlatformEventSequence(0),
            clock: Instant::now(),
            next_id: 1,
        }
    }

    /// Allocates a stable kernel [`WindowId`] and queues a window for creation
    /// from `config`. The OS window is built by the runner on the next pump
    /// where the event loop is available; the id is valid immediately so the
    /// host can refer to the window before it physically exists.
    pub fn request_window(&mut self, config: WindowConfig) -> WindowId {
        let id = WindowId(self.next_id);
        self.next_id += 1;
        self.pending_create.push((id, config));
        id
    }

    /// Queues a window for destruction. The OS window is dropped by the runner
    /// on the next pump. Unknown ids are ignored.
    pub fn request_destroy(&mut self, id: WindowId) {
        self.pending_destroy.push(id);
    }

    /// Mutable access to a live window's desired configuration. Returns `None`
    /// for ids that are not yet created (still pending) or already destroyed.
    pub fn config_mut(&mut self, id: WindowId) -> Option<&mut WindowConfig> {
        self.windows.get_mut(&id).map(|w| w.sync.desired_mut())
    }

    /// The OS-reported ground-truth state for a window, if it is live.
    #[must_use]
    pub fn realized(&self, id: WindowId) -> Option<&RealizedState> {
        self.windows.get(&id).map(|w| w.sync.realized())
    }

    /// A reference-counted handle to the live OS window, for building a render
    /// surface via `raw-window-handle`. Returns `None` if the window is not
    /// live.
    #[must_use]
    pub fn window(&self, id: WindowId) -> Option<Arc<WinitWindow>> {
        self.windows.get(&id).map(|w| Arc::clone(&w.window))
    }

    /// Borrows the current frame's (already-coalesced) event buffer without
    /// consuming it.
    #[must_use]
    pub fn events(&self) -> &[WindowEventEnvelope] {
        &self.events
    }

    /// Takes the current frame's event buffer, leaving it empty for the next
    /// frame. This is the host's single read point for the kernel stream.
    pub fn drain_events(&mut self) -> Vec<WindowEventEnvelope> {
        core::mem::take(&mut self.events)
    }

    /// Number of live OS windows.
    #[must_use]
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Whether there are no live OS windows (pending-create windows don't count).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    // --- internals driven by the runner -----------------------------------

    /// Builds the next platform event stamp (monotonic nanos + sequence).
    fn next_stamp(&mut self) -> PlatformEventStamp {
        let nanos = u64::try_from(self.clock.elapsed().as_nanos()).unwrap_or(u64::MAX);
        PlatformEventStamp::new(
            MonotonicTimestamp::from_nanos(nanos),
            self.sequence.post_increment(),
            EventSource::Window,
        )
    }

    /// Records a translated kernel event: folds it into the owning window's
    /// realized state immediately (so `realized()` is always fresh) and pushes
    /// a stamped envelope onto the frame buffer.
    fn record(&mut self, id: WindowId, event: KernelWindowEvent) {
        if let Some(live) = self.windows.get_mut(&id) {
            live.sync.observe(&event);
        }
        let stamp = self.next_stamp();
        self.events.push(WindowEventEnvelope::new(stamp, id, event));
    }

    /// Creates every queued window using the live event loop.
    fn process_pending_create(&mut self, event_loop: &ActiveEventLoop) {
        if self.pending_create.is_empty() {
            return;
        }
        for (kernel_id, config) in core::mem::take(&mut self.pending_create) {
            let attrs = command::build_attributes(&config);
            match event_loop.create_window(attrs) {
                Ok(window) => {
                    let window = Arc::new(window);
                    self.ids.insert(window.id(), kernel_id);
                    self.windows.insert(
                        kernel_id,
                        LiveWindow {
                            window,
                            sync: WindowSync::new(config),
                        },
                    );
                }
                Err(_) => {
                    // Creation failed (e.g. no display). The id is simply never
                    // bound; the host observes it never going live. We do not
                    // panic — a failed window must not take down the loop.
                }
            }
        }
    }

    /// Drops every queued window and unbinds its id.
    fn process_pending_destroy(&mut self) {
        if self.pending_destroy.is_empty() {
            return;
        }
        for id in core::mem::take(&mut self.pending_destroy) {
            self.ids.remove_by_kernel(id);
            // Dropping the LiveWindow drops the Arc<Window>; the OS window is
            // torn down once the last handle (renderer surface) is released.
            self.windows.remove(&id);
        }
    }

    /// For every live window, computes the minimal command diff, applies it to
    /// the OS window, and marks it applied so no field is commanded twice.
    fn flush_commands(&mut self) {
        for live in self.windows.values_mut() {
            let cmds = live.sync.diff();
            if cmds.is_empty() {
                continue;
            }
            for cmd in &cmds {
                command::apply(&live.window, cmd);
            }
            live.sync.mark_applied();
        }
    }
}

/// A host engine that the runner drives. The host sees only kernel types.
pub trait WinitApp {
    /// Called once, the first time the platform resumes, before any windows
    /// exist. The host should request its initial window(s) here.
    fn resumed(&mut self, backend: &mut WinitBackend);

    /// Called once per event-loop iteration after the frame's platform events
    /// have been collected and coalesced. The host reads events, mutates
    /// desired window state, and may request or destroy windows. Returning
    /// [`FrameOutcome::Exit`] tears the loop down.
    fn frame(&mut self, backend: &mut WinitBackend) -> FrameOutcome;
}

/// Binds a [`WinitApp`] host to winit's [`ApplicationHandler`] and owns the
/// backend state for the lifetime of the event loop.
pub struct WinitRunner<A: WinitApp> {
    backend: WinitBackend,
    app: A,
    resumed_once: bool,
}

impl<A: WinitApp> WinitRunner<A> {
    /// Wraps a host application in a runner with a fresh backend.
    #[must_use]
    pub fn new(app: A) -> Self {
        Self {
            backend: WinitBackend::new(),
            app,
            resumed_once: false,
        }
    }

    /// Read-only access to the backend (e.g. for inspection in tests).
    #[must_use]
    pub fn backend(&self) -> &WinitBackend {
        &self.backend
    }

    /// Mutable access to the backend before the loop starts.
    pub fn backend_mut(&mut self) -> &mut WinitBackend {
        &mut self.backend
    }

    /// Creates the OS event loop and runs it to completion on the calling
    /// thread. Blocks until the host returns [`FrameOutcome::Exit`] or the OS
    /// terminates the loop. Must be called on the main thread.
    ///
    /// # Errors
    /// Returns [`BackendError::EventLoopCreation`] if winit cannot build the
    /// event loop, or wraps a run-time loop error otherwise.
    pub fn run(mut self) -> BackendResult<()> {
        let event_loop =
            EventLoop::new().map_err(|e| BackendError::EventLoopCreation(e.to_string()))?;
        event_loop
            .run_app(&mut self)
            .map_err(|e| BackendError::EventLoopCreation(e.to_string()))
    }
}

impl<A: WinitApp> ApplicationHandler for WinitRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.resumed_once {
            self.resumed_once = true;
            self.app.resumed(&mut self.backend);
        }
        // Build any windows the host requested (either in `resumed` just now or
        // queued earlier). On Android a resume can recur; `request_window`
        // queueing plus this drain tolerates that.
        self.backend.process_pending_create(event_loop);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WinitWindowId,
        event: WinitWindowEvent,
    ) {
        let Some(kernel_id) = self.backend.ids.kernel(window_id) else {
            return;
        };

        match &event {
            // winit 0.30 reports the scale change without the new size inline
            // (it hands back an `InnerSizeWriter`); query the resolved inner
            // size from the live window and synthesize the kernel event.
            WinitWindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(live) = self.backend.windows.get(&kernel_id) {
                    let inner = convert::size_from_winit(live.window.inner_size());
                    let ev = translate::map_scale_factor_changed(*scale_factor, inner);
                    self.backend.record(kernel_id, ev);
                }
            }
            // Everything else goes through the pure translator; input-device
            // traffic and no-ops map to `None` and are dropped here (they are
            // `prism_input`'s concern).
            other => {
                if let Some(ev) = translate::translate(other) {
                    self.backend.record(kernel_id, ev);
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Collapse high-frequency resize/move/cursor/scale spam to last-wins
        // before the host sees the frame (§10.3).
        batch::coalesce(&mut self.backend.events);

        // Let the host read the frame and express new intent.
        let outcome = self.app.frame(&mut self.backend);

        // Service host-requested lifecycle changes, then push the minimal
        // command diff back to the OS.
        self.backend.process_pending_create(event_loop);
        self.backend.process_pending_destroy();
        self.backend.flush_commands();

        if matches!(outcome, FrameOutcome::Exit) {
            event_loop.exit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_window::geometry::PhysicalSize;

    #[test]
    fn request_window_allocates_distinct_ids() {
        let mut b = WinitBackend::new();
        let a = b.request_window(WindowConfig::new("A", PhysicalSize::new(100, 100)));
        let c = b.request_window(WindowConfig::new("B", PhysicalSize::new(200, 200)));
        assert_ne!(a, c);
        // Not yet live: creation is deferred to the runner.
        assert_eq!(b.window_count(), 0);
        assert!(b.config_mut(a).is_none());
    }

    #[test]
    fn drain_events_empties_the_buffer() {
        let mut b = WinitBackend::new();
        assert!(b.events().is_empty());
        let first = b.drain_events();
        assert!(first.is_empty());
    }

    #[test]
    fn next_stamp_is_strictly_increasing_in_sequence() {
        let mut b = WinitBackend::new();
        let s0 = b.next_stamp();
        let s1 = b.next_stamp();
        assert!(s1.sequence.get() > s0.sequence.get());
    }

    #[test]
    fn destroy_of_unknown_id_is_noop() {
        let mut b = WinitBackend::new();
        b.request_destroy(WindowId(999));
        b.process_pending_destroy();
        assert!(b.is_empty());
    }

    #[test]
    fn default_matches_new() {
        let b = WinitBackend::default();
        assert!(b.is_empty());
        assert_eq!(b.window_count(), 0);
    }
}
