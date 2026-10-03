//! Cross-thread sub-app pipelining (design §9, §24.3, §25.2).
//!
//! The default per-frame path ([`SubApps::update`](crate::sub_app::SubApps)
//! without pipelining) is **serial**: the main sub-app updates, then each
//! secondary sub-app extracts and updates on the same thread. That is exactly
//! the "serial extract first" step design §23 (risk #1) prescribes, and it
//! stays the only path unless a caller opts in.
//!
//! This module adds the *opt-in* refinement behind the `pipelined` feature: run
//! a secondary (e.g. render) sub-app for frame *N* on a worker thread **while**
//! the main sub-app simulates frame *N+1* on the calling thread, so simulation
//! and rendering overlap (design §9: "throughput near-doubles, latency traded
//! for throughput"). This mirrors Bevy's pipelined-rendering model without
//! copying its code.
//!
//! # Frame structure and the one-way invariant
//!
//! A pipelined frame driven by [`PipelinedExecutor::drive`] does, in order:
//!
//! 1. `main.update()` — simulate frame *N*. The previous frame's render
//!    (*N-1*) is still in flight on the worker thread, so this is the overlap.
//! 2. Join the in-flight render (*N-1*), returning its secondary sub-apps to
//!    the main thread.
//! 3. Run every secondary's [`ExtractFn`](crate::sub_app::ExtractFn) against
//!    the just-finished main world — the **synchronization point**. Extract
//!    runs on the main thread while no render thread is live, which is what
//!    keeps the one-way `main → sub` invariant (design §21) sound: the render
//!    thread never touches the main world concurrently.
//! 4. Move the secondaries onto a fresh worker thread and kick render *N*,
//!    then return. When [`drive`](PipelinedExecutor::drive) returns, render *N*
//!    is in flight and the secondaries are **not** resident on the main thread.
//!
//! Because step 3 is the only place a secondary reads the main world, and it
//! runs with no render thread live, the pipeline preserves the exact extract
//! semantics of the serial path — a secondary always reads a fully-simulated
//! frame, never a half-updated one — while overlapping the heavy work.
//!
//! # Resident-on-worker between frames
//!
//! Between frames the secondary sub-apps live on the worker thread, so they are
//! not reachable through [`SubApps::get`](crate::sub_app::SubApps::get) /
//! [`get_mut`](crate::sub_app::SubApps::get_mut). A caller that needs to
//! inspect or mutate a secondary must first bring it home with
//! [`PipelinedExecutor::sync`] (exposed as
//! [`App::sync_sub_apps`](crate::app::App::sync_sub_apps)). Runners call this
//! once the frame loop ends, and [`PipelinedExecutor`]'s [`Drop`] joins any
//! straggler so a dropped app never detaches or leaks the worker thread.

use std::thread::{self, JoinHandle};

use crate::sub_app::SubApp;
use crate::sub_app_label::BoxedSubAppLabel;

/// The owned list of secondary sub-apps that ferries between the main thread
/// and the render worker. Both halves — [`BoxedSubAppLabel`] (whose inner
/// `dyn SubAppLabel` is `Send + Sync`) and [`SubApp`] (whose `World` is `Send`)
/// — are `Send`, so the whole `Vec` can cross the thread boundary.
type Secondaries = Vec<(BoxedSubAppLabel, SubApp)>;

/// Drives the overlapped simulate/render pipeline for a [`SubApps`](crate::sub_app::SubApps) collection.
///
/// Holds the handle to the in-flight render frame (if any). Created by
/// [`SubApps::enable_pipelining`](crate::sub_app::SubApps::enable_pipelining)
/// and only ever touched while the owning [`SubApps`](crate::sub_app::SubApps) is borrowed mutably, so
/// it needs no interior synchronization of its own.
#[derive(Default)]
pub struct PipelinedExecutor {
    /// The render frame currently executing on a worker thread, carrying the
    /// secondary sub-apps it owns for the duration. `None` before the first
    /// frame and whenever the secondaries have been brought home by
    /// [`sync`](PipelinedExecutor::sync).
    in_flight: Option<JoinHandle<Secondaries>>,
}

impl PipelinedExecutor {
    /// Create an executor with nothing in flight.
    #[must_use]
    pub fn new() -> Self {
        Self { in_flight: None }
    }

    /// Whether a render frame is currently executing on the worker thread.
    #[must_use]
    pub fn has_in_flight(&self) -> bool {
        self.in_flight.is_some()
    }

    /// Drive one pipelined frame over `main` and `secondary` (design §9).
    ///
    /// Simulates `main` for the current frame — overlapping the previous
    /// frame's render still in flight — then joins that render, runs each
    /// secondary's extract against the freshly-simulated main world (the
    /// one-way synchronization point), and finally kicks this frame's render on
    /// a worker thread. On return, `secondary` is empty: its contents are owned
    /// by the newly spawned render frame until [`sync`](Self::sync) (or the
    /// next `drive`) brings them back.
    pub fn drive(&mut self, main: &mut SubApp, secondary: &mut Secondaries) {
        // 1. Simulate this frame while last frame's render overlaps.
        main.update();

        // 2. Reclaim last frame's secondaries from the render worker.
        self.join_in_flight(secondary);

        // 3. Extract on the main thread (one-way main → sub, design §21): the
        //    only point a secondary reads the main world, run with no render
        //    thread live so the read is of a complete frame.
        for (_, sub_app) in secondary.iter_mut() {
            sub_app.run_extract(&mut main.world);
        }

        // 4. Hand the secondaries to a worker thread and render this frame
        //    there, overlapping the next frame's simulation.
        let mut secs = std::mem::take(secondary);
        self.in_flight = Some(thread::spawn(move || {
            for (_, sub_app) in secs.iter_mut() {
                sub_app.update();
            }
            secs
        }));
    }

    /// Block until any in-flight render frame finishes, returning its secondary
    /// sub-apps to `secondary`. A no-op when nothing is in flight.
    ///
    /// After this returns the secondaries are resident on the calling thread
    /// and safe to inspect or mutate again.
    pub fn sync(&mut self, secondary: &mut Secondaries) {
        self.join_in_flight(secondary);
    }

    /// Join the in-flight render (if any) and move its secondaries into
    /// `secondary`. Propagates a render-thread panic on the calling thread so a
    /// failure is never silently swallowed.
    fn join_in_flight(&mut self, secondary: &mut Secondaries) {
        if let Some(handle) = self.in_flight.take() {
            match handle.join() {
                Ok(secs) => *secondary = secs,
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
    }
}

impl Drop for PipelinedExecutor {
    /// Join a straggling render frame so dropping a pipelined app never leaves
    /// a detached worker thread running against freed state. The reclaimed
    /// secondaries are simply dropped with the executor.
    fn drop(&mut self) {
        if let Some(handle) = self.in_flight.take() {
            // Best-effort: we are tearing down, so a render-thread panic here
            // is swallowed rather than double-panicking during unwind.
            let _ = handle.join();
        }
    }
}
