//! Thread-class separation: compute / I-O / main-thread lanes (design §24.2).
//!
//! Some work *must* run somewhere other than a general compute worker — GPU
//! submission and window/input callbacks on the main thread, and blocking
//! file/network I/O off the compute pool so a stalled syscall never starves (or
//! is starved by) the job graph. This module tags each job with a [`WorkClass`]
//! and routes it to the matching executor:
//!
//! - [`WorkClass::Compute`] → the work-stealing [`TaskPool`].
//! - [`WorkClass::Io`] → the off-pool blocking I/O lane
//!   ([`NamedThreads`]'s [`Io`](crate::ThreadCategory::Io) threads).
//! - [`WorkClass::MainThread`] → the main-thread queue, drained by
//!   [`ThreadClassPool::pump_main`] (never dispatched to a worker).
//!
//! It complements, rather than replaces, the §24.1 [`qos`](crate::qos) frame
//! scheduler: `qos` decides *when* committed vs. deferrable work runs within a
//! frame budget; this module decides *which thread class* runs a job at all.
//! It reuses the existing [`TaskPool`] and [`NamedThreads`] lanes wholesale —
//! no new thread machinery is introduced.
//!
//! # Determinism
//! The routing / isolation policy lives in [`ClassRouter`], a pure, clock-free,
//! `no_std`-friendly core (see [`route`](self::route)). Which class is served
//! next, and how many compute / I/O jobs are admitted per wave, is a
//! deterministic function of the queued work and the [`LaneBudget`]; the only
//! non-determinism is the order admitted jobs happen to finish on their
//! executors, which does not affect the returned [`DispatchReport`]. The core
//! is tested directly against a serial oracle.
//!
//! # Isolation
//! The three classes live in independent lanes, so an I/O backlog can never
//! consume a compute slot (or vice versa). Capping [`LaneBudget::io_slots`] to
//! `0` dispatches zero I/O yet leaves compute and main-thread dispatch
//! untouched — the concrete "blocking I/O does not starve compute" guarantee.
//!
//! # Single-threaded fallback
//! [`ThreadClassPool::inline`] (and [`TaskPool::thread_class_pool`] on a
//! single-threaded pool) runs compute and I/O jobs inline on the calling thread
//! in router-drain order, while main-thread jobs still queue for the pump. No
//! OS threads are involved, so the fallback is fully deterministic.

pub mod route;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::job::Job;
use crate::{Counter, NamedThreads, TaskPool, ThreadCategory};

pub use route::{
    admits, route, ClassRouter, ExecLane, LaneBudget, RoutePlan, RouteStep, WorkClass, CLASS_COUNT,
};

/// Summary of one [`ThreadClassPool::dispatch`] call.
///
/// Counts are over the jobs the router acted on this wave; the fields are a
/// deterministic function of the queued work and the [`LaneBudget`], so they
/// can be asserted exactly regardless of how the executors interleave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchReport {
    /// Compute jobs dispatched to the work-stealing pool this wave.
    pub compute_dispatched: usize,
    /// I/O jobs dispatched to the blocking I/O lane this wave.
    pub io_dispatched: usize,
    /// Main-thread jobs queued for the pump this wave.
    pub main_queued: usize,
    /// Compute jobs left queued because the budget was exhausted; they carry
    /// over to a later dispatch.
    pub compute_deferred: usize,
    /// I/O jobs left queued because the budget was exhausted; they carry over
    /// to a later dispatch.
    pub io_deferred: usize,
}

impl DispatchReport {
    /// Total jobs dispatched to an executor this wave (compute + I/O + main).
    #[must_use]
    pub fn dispatched(&self) -> usize {
        self.compute_dispatched + self.io_dispatched + self.main_queued
    }

    /// Total compute / I/O jobs deferred to a later wave.
    #[must_use]
    pub fn deferred(&self) -> usize {
        self.compute_deferred + self.io_deferred
    }
}

/// The executor backend a [`ThreadClassPool`] dispatches onto.
enum Backend {
    /// Real lanes: compute on the pool, I/O on the named blocking lane, main on
    /// the named main queue.
    Threaded {
        /// The work-stealing compute pool.
        compute: TaskPool,
        /// The named lanes owning the I/O threads and the main queue.
        named: NamedThreads,
    },
    /// Single-threaded fallback: compute and I/O run inline on dispatch; main
    /// jobs queue here for the pump.
    Inline {
        /// Main-thread jobs awaiting [`ThreadClassPool::pump_main`].
        main_queue: VecDeque<Job>,
    },
}

/// A thread-class-aware dispatcher that routes jobs to the compute pool, the
/// blocking I/O lane, or the main-thread queue (design §24.2).
///
/// Submit work with [`ThreadClassPool::submit_compute`],
/// [`ThreadClassPool::submit_io`], or [`ThreadClassPool::submit_main`], then
/// call [`ThreadClassPool::dispatch`] to route the queued work to its
/// executors. Compute and I/O run asynchronously — block for them with
/// [`ThreadClassPool::wait`] — while main-thread work waits for
/// [`ThreadClassPool::pump_main`] on the main thread.
pub struct ThreadClassPool {
    /// The per-class queues holding not-yet-dispatched jobs.
    router: ClassRouter<Job>,
    /// Per-wave admission budget enforcing lane isolation.
    budget: LaneBudget,
    /// Tracks outstanding compute + I/O jobs from the threaded backend so
    /// [`ThreadClassPool::wait`] can block on them.
    inflight: Counter,
    /// Where routed jobs actually run.
    backend: Backend,
}

impl ThreadClassPool {
    /// Build a dispatcher over an existing compute `pool` and `named` lanes,
    /// admitting every queued job per wave ([`LaneBudget::unbounded`]).
    #[must_use]
    pub fn new(pool: TaskPool, named: NamedThreads) -> Self {
        Self::with_budget(pool, named, LaneBudget::unbounded())
    }

    /// Build a dispatcher with an explicit per-wave [`LaneBudget`] (e.g. to cap
    /// how many compute / I/O jobs are admitted per dispatch for backpressure).
    #[must_use]
    pub fn with_budget(pool: TaskPool, named: NamedThreads, budget: LaneBudget) -> Self {
        Self {
            router: ClassRouter::new(),
            budget,
            inflight: Counter::new(),
            backend: Backend::Threaded {
                compute: pool,
                named,
            },
        }
    }

    /// Build the single-threaded fallback: compute and I/O run inline on
    /// [`ThreadClassPool::dispatch`] in router-drain order, main jobs queue for
    /// [`ThreadClassPool::pump_main`]. No OS threads are used.
    #[must_use]
    pub fn inline() -> Self {
        Self {
            router: ClassRouter::new(),
            budget: LaneBudget::unbounded(),
            inflight: Counter::new(),
            backend: Backend::Inline {
                main_queue: VecDeque::new(),
            },
        }
    }

    /// Whether this dispatcher runs compute / I/O inline (the single-threaded
    /// fallback).
    #[must_use]
    pub fn is_inline(&self) -> bool {
        matches!(self.backend, Backend::Inline { .. })
    }

    /// Replace the per-wave admission [`LaneBudget`].
    pub fn set_budget(&mut self, budget: LaneBudget) {
        self.budget = budget;
    }

    /// The current per-wave admission [`LaneBudget`].
    #[must_use]
    pub fn budget(&self) -> LaneBudget {
        self.budget
    }

    /// Queue `f` on `class`. Prefer the [`ThreadClassPool::submit_compute`] /
    /// [`ThreadClassPool::submit_io`] / [`ThreadClassPool::submit_main`]
    /// shorthands for the common cases.
    pub fn submit<F>(&mut self, class: WorkClass, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job: Job = Box::new(f);
        self.router.push(class, job);
    }

    /// Queue CPU-bound work for the compute pool.
    pub fn submit_compute<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(WorkClass::Compute, f);
    }

    /// Queue blocking file / network work for the off-pool I/O lane.
    pub fn submit_io<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(WorkClass::Io, f);
    }

    /// Queue main-thread-only work (GPU submit, platform callbacks) for the
    /// main pump.
    pub fn submit_main<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(WorkClass::MainThread, f);
    }

    /// Number of jobs currently queued across all classes (not yet dispatched).
    #[must_use]
    pub fn pending(&self) -> usize {
        self.router.len()
    }

    /// Number of jobs queued on a specific `class`.
    #[must_use]
    pub fn pending_in(&self, class: WorkClass) -> usize {
        self.router.len_in(class)
    }

    /// Whether no work is queued on any class.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.router.is_empty()
    }

    /// Borrow the underlying router (e.g. to inspect the next decision with
    /// [`ClassRouter::peek_plan`]).
    #[must_use]
    pub fn router(&self) -> &ClassRouter<Job> {
        &self.router
    }

    /// Route every admissible queued job to its executor under the current
    /// [`LaneBudget`], returning the deterministic [`DispatchReport`].
    ///
    /// Compute and I/O jobs run asynchronously (inline in the single-threaded
    /// fallback); block for their completion with [`ThreadClassPool::wait`].
    /// Main-thread jobs are queued and run only on
    /// [`ThreadClassPool::pump_main`]. Jobs deferred by an exhausted budget stay
    /// queued for the next dispatch.
    pub fn dispatch(&mut self) -> DispatchReport {
        let mut budget = self.budget;
        let mut report = DispatchReport::default();
        loop {
            match self.router.next_step(&mut budget) {
                RouteStep::Dispatch { class, payload, .. } => match class {
                    WorkClass::Compute => {
                        report.compute_dispatched += 1;
                        self.run_compute(payload);
                    }
                    WorkClass::Io => {
                        report.io_dispatched += 1;
                        self.run_io(payload);
                    }
                    WorkClass::MainThread => {
                        report.main_queued += 1;
                        self.enqueue_main(payload);
                    }
                },
                RouteStep::Idle {
                    compute_deferred,
                    io_deferred,
                } => {
                    report.compute_deferred = compute_deferred;
                    report.io_deferred = io_deferred;
                    break;
                }
            }
        }
        report
    }

    /// Dispatch and then block until the dispatched compute / I/O jobs finish
    /// (a convenience over [`ThreadClassPool::dispatch`] + [`wait`]). Main-thread
    /// work remains queued for [`ThreadClassPool::pump_main`].
    ///
    /// [`wait`]: ThreadClassPool::wait
    pub fn dispatch_blocking(&mut self) -> DispatchReport {
        let report = self.dispatch();
        self.wait();
        report
    }

    /// Block until every compute / I/O job dispatched so far has finished,
    /// helping the compute pool while waiting (no-op in the single-threaded
    /// fallback, where those jobs already ran inline).
    pub fn wait(&self) {
        if let Backend::Threaded { compute, .. } = &self.backend {
            compute.wait(&self.inflight);
        }
    }

    /// Run every main-thread job dispatched so far on the calling thread, then
    /// return the number executed. Call this from the application's main thread
    /// (e.g. once per frame).
    pub fn pump_main(&mut self) -> usize {
        match &mut self.backend {
            Backend::Threaded { named, .. } => named.run_main_pending(),
            Backend::Inline { main_queue } => {
                let batch: Vec<Job> = main_queue.drain(..).collect();
                let count = batch.len();
                for job in batch {
                    job();
                }
                count
            }
        }
    }

    /// Dispatch a compute job: onto the pool, or inline in the fallback.
    fn run_compute(&mut self, payload: Job) {
        match &self.backend {
            Backend::Threaded { compute, .. } => compute.spawn(&self.inflight, payload),
            Backend::Inline { .. } => payload(),
        }
    }

    /// Dispatch an I/O job: onto the blocking lane (tracked by `inflight`), or
    /// inline in the fallback.
    fn run_io(&mut self, payload: Job) {
        match &self.backend {
            Backend::Threaded { named, .. } => {
                self.inflight.add(1);
                let inflight = self.inflight.clone();
                let _lane = named.dispatch(ThreadCategory::Io, move || {
                    payload();
                    inflight.finish_one();
                });
            }
            Backend::Inline { .. } => payload(),
        }
    }

    /// Queue a main-thread job: on the named main queue, or in the inline
    /// fallback's own queue.
    fn enqueue_main(&mut self, payload: Job) {
        match &mut self.backend {
            Backend::Threaded { named, .. } => {
                let _lane = named.dispatch(ThreadCategory::Main, payload);
            }
            Backend::Inline { main_queue } => main_queue.push_back(payload),
        }
    }
}

impl TaskPool {
    /// Create a [`ThreadClassPool`] that dispatches compute work onto this pool
    /// and I/O / main work onto `named` (design §24.2).
    ///
    /// A single-threaded pool yields the inline fallback
    /// ([`ThreadClassPool::inline`]) so compute / I/O run inline deterministically.
    #[must_use]
    pub fn thread_class_pool(&self, named: &NamedThreads) -> ThreadClassPool {
        if self.is_single_threaded() {
            ThreadClassPool::inline()
        } else {
            ThreadClassPool::new(self.clone(), named.clone())
        }
    }
}
