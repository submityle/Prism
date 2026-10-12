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
//!    (*N-1*) is still in flight on the render lane, so this is the overlap.
//! 2. Join the in-flight render (*N-1*), returning its secondary sub-apps to
//!    the main thread.
//! 3. Run every secondary's [`ExtractFn`](crate::sub_app::ExtractFn) against
//!    the just-finished main world — the **synchronization point**. Extract
//!    runs on the main thread while no render is live, which is what keeps the
//!    one-way `main → sub` invariant (design §21) sound: the render lane never
//!    touches the main world concurrently.
//! 4. Hand the secondaries to the render lane and kick render *N*, then return.
//!    When [`drive`](PipelinedExecutor::drive) returns, render *N* is in flight
//!    and the secondaries are **not** resident on the main thread.
//!
//! Because step 3 is the only place a secondary reads the main world, and it
//! runs with no render live, the pipeline preserves the exact extract semantics
//! of the serial path — a secondary always reads a fully-simulated frame, never
//! a half-updated one — while overlapping the heavy work.
//!
//! # A persistent render lane
//!
//! The render lane is a **single long-lived worker thread**, created lazily on
//! the first frame and reused for every subsequent one: [`drive`](PipelinedExecutor::drive)
//! hands it the frame's secondaries over a channel and collects the finished
//! frame back the same way. Spawning a fresh OS thread per frame would add
//! thread-creation latency (tens of microseconds, plus scheduling churn) to the
//! critical path *every* frame — precisely the cost the pipeline exists to
//! avoid — so the worker is created once and kept warm. A future refinement
//! (design §24.3) is to seat this lane on a `prism_tasks` worker group; today it
//! is a dedicated thread so the render cadence stays independent of the
//! simulation pool.
//!
//! # Resident-on-worker between frames
//!
//! Between frames the secondary sub-apps live in the render lane (on the worker
//! or in the channel), so they are not reachable through
//! [`SubApps::get`](crate::sub_app::SubApps::get) /
//! [`get_mut`](crate::sub_app::SubApps::get_mut). A caller that needs to
//! inspect or mutate a secondary must first bring it home with
//! [`PipelinedExecutor::sync`] (exposed as
//! [`App::sync_sub_apps`](crate::app::App::sync_sub_apps)). Runners call this
//! once the frame loop ends, and [`PipelinedExecutor`]'s [`Drop`] joins the
//! worker so a dropped app never detaches or leaks the thread.

use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use crate::sub_app::SubApp;
use crate::sub_app_label::BoxedSubAppLabel;

/// The owned list of secondary sub-apps that ferries between the main thread
/// and the render worker. Both halves — [`BoxedSubAppLabel`] (whose inner
/// `dyn SubAppLabel` is `Send + Sync`) and [`SubApp`] (whose `World` is `Send`)
/// — are `Send`, so the whole `Vec` can cross the thread boundary.
type Secondaries = Vec<(BoxedSubAppLabel, SubApp)>;

/// The result of one render frame on the worker, ferried back to the main
/// thread. A panic is captured (never lost) and re-raised on the main thread so
/// a render failure is not silently swallowed.
enum RenderOutcome {
    /// The frame rendered; the secondaries are handed back.
    Done(Secondaries),
    /// The frame panicked; the payload is re-raised by the collector.
    Panicked(Box<dyn core::any::Any + Send + 'static>),
}

/// The long-lived render lane: one OS worker thread that renders one frame at a
/// time, fed by a job channel and answering on a result channel.
struct RenderWorker {
    /// Sends a frame's secondaries to the worker. `None` once shutdown begins.
    jobs: Option<Sender<Secondaries>>,
    /// Receives each finished frame's outcome.
    results: Receiver<RenderOutcome>,
    /// The worker thread, joined on drop.
    handle: Option<JoinHandle<()>>,
}

impl RenderWorker {
    /// Spawn the render lane. The worker loops until the job channel closes
    /// (its only sender dropped), rendering each frame it is handed.
    fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Secondaries>();
        let (result_tx, result_rx) = mpsc::channel::<RenderOutcome>();
        let handle = thread::spawn(move || {
            while let Ok(mut secs) = job_rx.recv() {
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    for (_, sub_app) in secs.iter_mut() {
                        sub_app.update();
                    }
                    secs
                }));
                let message = match outcome {
                    Ok(secs) => RenderOutcome::Done(secs),
                    Err(payload) => RenderOutcome::Panicked(payload),
                };
                if result_tx.send(message).is_err() {
                    // The executor is gone; stop rendering.
                    break;
                }
            }
        });
        Self {
            jobs: Some(job_tx),
            results: result_rx,
            handle: Some(handle),
        }
    }

    /// Hand the worker a frame's secondaries. The channel is unbounded, so this
    /// never blocks the main thread.
    fn submit(&self, secs: Secondaries) {
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(secs);
        }
    }

    /// Block until the worker answers with the in-flight frame's outcome.
    fn collect(&self) -> Option<RenderOutcome> {
        self.results.recv().ok()
    }
}

impl Drop for RenderWorker {
    fn drop(&mut self) {
        // Close the job channel so the worker's `recv` unblocks and its loop
        // exits after finishing any in-flight frame, then join it. `results` is
        // still alive here, so a rendering worker can always deliver its last
        // frame before it stops.
        self.jobs = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Drives the overlapped simulate/render pipeline for a [`SubApps`](crate::sub_app::SubApps) collection.
///
/// Holds the persistent render lane (spawned on the first frame) and whether a
/// frame is currently in flight on it. Created by
/// [`SubApps::enable_pipelining`](crate::sub_app::SubApps::enable_pipelining)
/// and only ever touched while the owning [`SubApps`](crate::sub_app::SubApps) is borrowed mutably, so
/// it needs no interior synchronization of its own.
#[derive(Default)]
pub struct PipelinedExecutor {
    /// The render lane, created on the first [`drive`](Self::drive).
    worker: Option<RenderWorker>,
    /// Whether a render frame is currently in flight on the worker.
    in_flight: bool,
}

impl PipelinedExecutor {
    /// Create an executor with no worker yet and nothing in flight.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a render frame is currently executing on the worker thread.
    #[must_use]
    pub fn has_in_flight(&self) -> bool {
        self.in_flight
    }

    /// Drive one pipelined frame over `main` and `secondary` (design §9).
    ///
    /// Simulates `main` for the current frame — overlapping the previous
    /// frame's render still in flight — then joins that render, runs each
    /// secondary's extract against the freshly-simulated main world (the
    /// one-way synchronization point), and finally kicks this frame's render on
    /// the render lane. On return, `secondary` is empty: its contents are owned
    /// by the in-flight frame until [`sync`](Self::sync) (or the next `drive`)
    /// brings them back.
    pub fn drive(&mut self, main: &mut SubApp, secondary: &mut Secondaries) {
        use std::time::Instant;

        if self.worker.is_none() {
            self.worker = Some(RenderWorker::spawn());
        }

        // 1. Simulate this frame while last frame's render overlaps. If a
        //    render is in flight now, this whole `update` runs concurrently
        //    with it — the overlap window the pipeline exists to create.
        let had_in_flight = self.in_flight;
        let sim_start = Instant::now();
        main.update();
        let sim = sim_start.elapsed();

        // 2. Reclaim last frame's secondaries from the render worker. The time
        //    spent blocked here is the render tail that overran the simulation
        //    window above (zero when the render finished first).
        let wait_start = Instant::now();
        self.collect(secondary);
        let wait = wait_start.elapsed();

        // Observability (design §16: 流水线重叠率): record the overlap sample
        // only when a render was genuinely in flight during the simulate above,
        // so the first frame (nothing to overlap) is skipped rather than scored
        // as a spurious perfect overlap.
        if had_in_flight
            && let Some(diag) = main
                .world
                .get_resource_mut::<crate::diagnostics::FrameDiagnostics>()
        {
            diag.record_pipeline_overlap(sim, wait);
        }

        // 3. Extract on the main thread (one-way main → sub, design §21): the
        //    only point a secondary reads the main world, run with no render
        //    live so the read is of a complete frame.
        let extract_start = Instant::now();
        for (_, sub_app) in secondary.iter_mut() {
            sub_app.run_extract(&mut main.world);
        }
        let extract = extract_start.elapsed();
        if let Some(diag) = main
            .world
            .get_resource_mut::<crate::diagnostics::FrameDiagnostics>()
        {
            diag.record_extract(extract);
        }

        // 4. Hand the secondaries to the render lane and render this frame
        //    there, overlapping the next frame's simulation.
        let secs = std::mem::take(secondary);
        if let Some(worker) = self.worker.as_ref() {
            worker.submit(secs);
        }
        self.in_flight = true;
    }

    /// Block until any in-flight render frame finishes, returning its secondary
    /// sub-apps to `secondary`. A no-op when nothing is in flight.
    ///
    /// After this returns the secondaries are resident on the calling thread
    /// and safe to inspect or mutate again.
    pub fn sync(&mut self, secondary: &mut Secondaries) {
        self.collect(secondary);
    }

    /// Collect the in-flight render (if any) and move its secondaries into
    /// `secondary`. Propagates a render-thread panic on the calling thread so a
    /// failure is never silently swallowed.
    fn collect(&mut self, secondary: &mut Secondaries) {
        if !self.in_flight {
            return;
        }
        self.in_flight = false;
        let Some(worker) = self.worker.as_ref() else {
            return;
        };
        match worker.collect() {
            Some(RenderOutcome::Done(secs)) => *secondary = secs,
            Some(RenderOutcome::Panicked(payload)) => std::panic::resume_unwind(payload),
            // The worker stopped without answering; nothing to collect.
            None => {}
        }
    }
}
