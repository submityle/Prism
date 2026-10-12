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
//! # Render lanes: a dedicated thread (default) or a `prism_tasks` pool
//!
//! Frames are rendered on a *render lane* behind a small submit/collect
//! contract with two implementations:
//!
//! - **Dedicated** (feature `pipelined`, the default): a single long-lived
//!   worker thread, created lazily on the first frame and reused for every
//!   subsequent one. Spawning a fresh OS thread per frame would add
//!   thread-creation latency (tens of microseconds plus scheduling churn) to
//!   the critical path *every* frame — precisely the cost the pipeline exists
//!   to hide — so the worker is created once and kept warm. The render cadence
//!   stays independent of the simulation pool: the lowest-latency choice.
//! - **Task pool** (feature `pipelined-tasks`, design §24.3): the frame's render
//!   job is submitted to a [`prism_tasks`] work-stealing pool, so the pipeline
//!   runs on the engine's unified task graph and can share one pool with the
//!   rest of the engine. A pool may be created on demand or injected via
//!   [`PipelinedExecutor::with_task_pool`].
//!
//! Both lanes preserve the identical one-way extract semantics and observability
//! below, so switching a lane never changes *what* is simulated or rendered —
//! only *where* the render runs.
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
/// and the render lane. Both halves — [`BoxedSubAppLabel`] (whose inner
/// `dyn SubAppLabel` is `Send + Sync`) and [`SubApp`] (whose `World` is `Send`)
/// — are `Send`, so the whole `Vec` can cross the thread boundary.
type Secondaries = Vec<(BoxedSubAppLabel, SubApp)>;

/// The result of one render frame on a lane, ferried back to the main thread. A
/// panic is captured (never lost) and re-raised on the main thread so a render
/// failure is not silently swallowed.
enum RenderOutcome {
    /// The frame rendered; the secondaries are handed back.
    Done(Secondaries),
    /// The frame panicked; the payload is re-raised by the collector.
    Panicked(Box<dyn core::any::Any + Send + 'static>),
}

/// Runs one frame's secondaries, panicking on failure. Shared by both lanes so
/// their semantics cannot drift.
fn render_frame(mut secs: Secondaries) -> Secondaries {
    for (_, sub_app) in secs.iter_mut() {
        sub_app.update();
    }
    secs
}

/// A dedicated long-lived render thread: one OS worker that renders one frame
/// at a time, fed by a job channel and answering on a result channel.
struct DedicatedLane {
    /// Sends a frame's secondaries to the worker. `None` once shutdown begins.
    jobs: Option<Sender<Secondaries>>,
    /// Receives each finished frame's outcome.
    results: Receiver<RenderOutcome>,
    /// The worker thread, joined on drop.
    handle: Option<JoinHandle<()>>,
}

impl DedicatedLane {
    /// Spawn the render lane. The worker loops until the job channel closes
    /// (its only sender dropped), rendering each frame it is handed.
    fn spawn() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Secondaries>();
        let (result_tx, result_rx) = mpsc::channel::<RenderOutcome>();
        let handle = thread::spawn(move || {
            while let Ok(secs) = job_rx.recv() {
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| render_frame(secs)));
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

impl Drop for DedicatedLane {
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

/// A render lane that submits each frame as a job on a [`prism_tasks`] pool
/// (design §24.3: the pipeline runs on the engine's unified task graph).
#[cfg(feature = "pipelined-tasks")]
struct TaskLane {
    /// The shared work-stealing pool the render jobs run on.
    pool: prism_tasks::TaskPool,
    /// Tracks the outstanding render job (one at a time). We block on the
    /// result channel rather than the counter, but every submitted job must be
    /// registered so the pool's fork-join bookkeeping stays honest.
    counter: prism_tasks::Counter,
    /// Cloned into each render job to ship its outcome back.
    results_tx: Sender<RenderOutcome>,
    /// Receives each finished frame's outcome.
    results_rx: Receiver<RenderOutcome>,
}

#[cfg(feature = "pipelined-tasks")]
impl TaskLane {
    /// Wrap an existing pool as a render lane.
    fn new(pool: prism_tasks::TaskPool) -> Self {
        let (results_tx, results_rx) = mpsc::channel();
        Self {
            pool,
            counter: prism_tasks::Counter::new(),
            results_tx,
            results_rx,
        }
    }

    /// Submit one frame to the pool. The job renders and ships its outcome back
    /// on the result channel; `collect` blocks on that channel.
    fn submit(&self, secs: Secondaries) {
        let results_tx = self.results_tx.clone();
        self.pool.spawn(&self.counter, move || {
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| render_frame(secs)));
            let message = match outcome {
                Ok(secs) => RenderOutcome::Done(secs),
                Err(payload) => RenderOutcome::Panicked(payload),
            };
            let _ = results_tx.send(message);
        });
    }

    /// Block until the pool answers with the in-flight frame's outcome.
    fn collect(&self) -> Option<RenderOutcome> {
        self.results_rx.recv().ok()
    }
}

/// The render lane backing a [`PipelinedExecutor`]. Both variants share the
/// same submit/collect semantics; only the executor *where* the frame runs.
enum Lane {
    /// A dedicated long-lived thread (default).
    Dedicated(DedicatedLane),
    /// A `prism_tasks` work-stealing pool (design §24.3).
    #[cfg(feature = "pipelined-tasks")]
    Tasks(TaskLane),
}

impl Lane {
    /// Hand the lane a frame's secondaries and start rendering it.
    fn submit(&self, secs: Secondaries) {
        match self {
            Self::Dedicated(lane) => lane.submit(secs),
            #[cfg(feature = "pipelined-tasks")]
            Self::Tasks(lane) => lane.submit(secs),
        }
    }

    /// Block until the in-flight frame's outcome is ready.
    fn collect(&self) -> Option<RenderOutcome> {
        match self {
            Self::Dedicated(lane) => lane.collect(),
            #[cfg(feature = "pipelined-tasks")]
            Self::Tasks(lane) => lane.collect(),
        }
    }
}

/// Drives the overlapped simulate/render pipeline for a [`SubApps`](crate::sub_app::SubApps) collection.
///
/// Holds the render lane (created lazily on the first frame for the default
/// dedicated lane) and whether a frame is currently in flight on it. Created by
/// [`SubApps::enable_pipelining`](crate::sub_app::SubApps::enable_pipelining)
/// and only ever touched while the owning [`SubApps`](crate::sub_app::SubApps) is borrowed mutably, so
/// it needs no interior synchronization of its own.
#[derive(Default)]
pub struct PipelinedExecutor {
    /// The render lane, created on the first [`drive`](Self::drive) for the
    /// default dedicated lane (or eagerly for an injected task pool).
    lane: Option<Lane>,
    /// Whether a render frame is currently in flight on the lane.
    in_flight: bool,
}

impl PipelinedExecutor {
    /// Create an executor that renders on a **dedicated** lane (the default),
    /// spawned lazily on the first frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an executor that renders each frame on the given [`prism_tasks`]
    /// pool (design §24.3). Share the engine's pool here so the pipeline runs
    /// on the unified task graph rather than a private thread.
    #[cfg(feature = "pipelined-tasks")]
    #[must_use]
    pub fn with_task_pool(pool: prism_tasks::TaskPool) -> Self {
        Self {
            lane: Some(Lane::Tasks(TaskLane::new(pool))),
            in_flight: false,
        }
    }

    /// Ensure the default dedicated lane exists (no-op once created).
    fn ensure_lane(&mut self) {
        if self.lane.is_none() {
            self.lane = Some(Lane::Dedicated(DedicatedLane::spawn()));
        }
    }

    /// Whether a render frame is currently executing on the lane.
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

        self.ensure_lane();

        // 1. Simulate this frame while last frame's render overlaps. If a
        //    render is in flight now, this whole `update` runs concurrently
        //    with it — the overlap window the pipeline exists to create.
        let had_in_flight = self.in_flight;
        let sim_start = Instant::now();
        main.update();
        let sim = sim_start.elapsed();

        // 2. Reclaim last frame's secondaries from the render lane. The time
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
        if let Some(lane) = self.lane.as_ref() {
            lane.submit(secs);
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
        let Some(lane) = self.lane.as_ref() else {
            return;
        };
        match lane.collect() {
            Some(RenderOutcome::Done(secs)) => *secondary = secs,
            Some(RenderOutcome::Panicked(payload)) => std::panic::resume_unwind(payload),
            // The lane stopped without answering; nothing to collect.
            None => {}
        }
    }
}
