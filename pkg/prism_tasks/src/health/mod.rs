//! Backpressure, deadlock prevention, and health monitoring (design §24.8).
//!
//! This module bundles the three §24.8 guards, each as a pure deterministic
//! core plus (where it makes sense) a thin execution façade over the existing
//! [`TaskPool`]:
//!
//! - **Backpressure** ([`backpressure`]): [`QueueBackpressure`] decides whether
//!   a submission is admitted, deferred (shedding deferrable
//!   [`Priority::Background`](crate::Priority::Background) work first), or
//!   rejected at a hard capacity cap. The [`BackpressureQueue`] façade wires
//!   that decision to real pool dispatch.
//! - **Deadlock prevention** ([`wait_graph`]): [`WaitGraph`] admits a
//!   wait-for dependency only if it keeps the dependency graph acyclic and
//!   within a bounded wait-chain depth. It is a decision structure with no
//!   execution façade.
//! - **Health monitoring** ([`monitor`]): [`PoolHealthMonitor`] turns raw
//!   observations into a deterministic [`HealthReport`] (starvation, steal
//!   failure rate, latency percentiles, longest job, peak queue depth). The
//!   [`HealthProbe`] façade samples *real* wall-clock job latencies and feeds
//!   them in.
//!
//! # Determinism and the honest boundary
//! Admission, cycle/chain checks, histogram bucketing, percentile math, and
//! every [`HealthReport`] field are pure integer functions of their inputs and
//! are tested against serial oracles. The only non-deterministic input is the
//! *wall-clock latency* [`HealthProbe::measure`] samples with
//! [`std::time::Instant`]: how long a job actually takes depends on real OS
//! scheduling. Once a latency is recorded, everything derived from it is
//! deterministic.

pub mod backpressure;
pub mod monitor;
pub mod wait_graph;

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;
use std::time::Instant;

pub use backpressure::{Admission, BackpressureLimits, QueueBackpressure};
pub use monitor::{
    HealthReport, LatencyHistogram, PoolHealthMonitor, StarvationDetector, StealStats,
};
pub use wait_graph::{DeadlockError, WaitGraph, WaitNodeId};

use crate::job::Job;
use crate::priority::Priority;
use crate::{Counter, TaskPool};

/// Summary of one [`BackpressureQueue::run`] call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackpressureRunReport {
    /// Number of admitted jobs dispatched and joined this run.
    pub ran: usize,
}

/// A backpressure-gated work queue bound to a [`TaskPool`] (design §24.8).
///
/// Submit work with [`BackpressureQueue::offer`], tagging each job with a
/// [`Priority`]; the gate admits, defers (shedding deferrable background work
/// under pressure), or rejects (at the hard capacity cap) per
/// [`QueueBackpressure`]. Admitted jobs buffer until [`BackpressureQueue::run`]
/// dispatches them onto the pool and joins them, draining the depth back down.
///
/// The admission *decision* is deterministic (a function of depth, watermarks,
/// and priority); only the order admitted jobs happen to finish on the workers
/// is not, and that does not affect the gate.
pub struct BackpressureQueue {
    /// The deterministic admission gate.
    gate: QueueBackpressure,
    /// Admitted-but-not-yet-dispatched jobs.
    pending: Vec<Job>,
    /// The pool admitted jobs are dispatched to.
    pool: TaskPool,
}

impl BackpressureQueue {
    /// Create a queue dispatching onto `pool` with the given `limits`.
    #[must_use]
    pub fn new(pool: TaskPool, limits: BackpressureLimits) -> Self {
        Self {
            gate: QueueBackpressure::new(limits),
            pending: Vec::new(),
            pool,
        }
    }

    /// Offer `f` at `priority`. On [`Admission::Admitted`] the job is buffered
    /// for the next [`BackpressureQueue::run`]; on [`Admission::Deferred`] or
    /// [`Admission::Rejected`] the job is dropped and the gate is unchanged.
    /// Returns the decision.
    pub fn offer<F>(&mut self, priority: Priority, f: F) -> Admission
    where
        F: FnOnce() + Send + 'static,
    {
        let decision = self.gate.offer(priority);
        if decision.is_admitted() {
            let job: Job = Box::new(f);
            self.pending.push(job);
        }
        decision
    }

    /// Current outstanding depth (buffered admitted jobs).
    #[must_use]
    #[inline]
    pub fn depth(&self) -> usize {
        self.gate.depth()
    }

    /// Number of buffered jobs awaiting the next run.
    #[must_use]
    #[inline]
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Whether background shedding is currently armed.
    #[must_use]
    #[inline]
    pub fn is_shedding(&self) -> bool {
        self.gate.is_shedding()
    }

    /// Borrow the underlying deterministic gate (e.g. to inspect
    /// [`QueueBackpressure::peek`] or the limits).
    #[must_use]
    #[inline]
    pub fn gate(&self) -> QueueBackpressure {
        self.gate
    }

    /// Dispatch every buffered admitted job onto the pool, join them all, and
    /// drain the depth back to zero. Returns a [`BackpressureRunReport`].
    pub fn run(&mut self) -> BackpressureRunReport {
        let jobs = core::mem::take(&mut self.pending);
        let ran = jobs.len();
        let counter = Counter::new();
        for job in jobs {
            self.pool.spawn(&counter, job);
        }
        self.pool.wait(&counter);
        self.gate.complete_many(ran);
        BackpressureRunReport { ran }
    }
}

impl TaskPool {
    /// Create a [`BackpressureQueue`] that dispatches onto this pool with the
    /// given `limits` (design §24.8).
    #[must_use]
    pub fn backpressure_queue(&self, limits: BackpressureLimits) -> BackpressureQueue {
        BackpressureQueue::new(self.clone(), limits)
    }
}

/// A cloneable, thread-safe handle that samples real job latencies and feeds a
/// shared [`PoolHealthMonitor`] (design §24.8).
///
/// Clone it into worker closures and wrap each unit of work in
/// [`HealthProbe::measure`]; the probe times the closure with
/// [`std::time::Instant`] and records the latency. Read the current indicators
/// with [`HealthProbe::report`].
///
/// # Honest boundary
/// The measured latency reflects real OS scheduling and is therefore not
/// reproducible run to run. Everything the underlying [`PoolHealthMonitor`]
/// derives from the recorded samples (percentiles, longest job, …) is
/// deterministic.
#[derive(Clone)]
pub struct HealthProbe {
    /// The shared monitor behind a mutex so worker threads can record into it.
    monitor: Arc<Mutex<PoolHealthMonitor>>,
}

impl HealthProbe {
    /// Create a probe for `worker_count` workers, flagging starvation at
    /// `starvation_threshold` consecutive idle observations and bucketing
    /// latencies with the ascending `latency_bounds_nanos`.
    #[must_use]
    pub fn new(
        worker_count: usize,
        starvation_threshold: u32,
        latency_bounds_nanos: &[u64],
    ) -> Self {
        Self {
            monitor: Arc::new(Mutex::new(PoolHealthMonitor::new(
                worker_count,
                starvation_threshold,
                latency_bounds_nanos,
            ))),
        }
    }

    /// Run `f`, timing it with [`std::time::Instant`] and recording the elapsed
    /// nanoseconds as a latency sample. Returns `f`'s result.
    pub fn measure<R>(&self, f: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let result = f();
        let nanos = elapsed_nanos(start);
        self.monitor.lock().unwrap().record_latency(nanos);
        result
    }

    /// Record that `worker` found no work this observation.
    pub fn record_idle(&self, worker: usize) {
        self.monitor.lock().unwrap().record_idle(worker);
    }

    /// Record that `worker` ran a job this observation.
    pub fn record_busy(&self, worker: usize) {
        self.monitor.lock().unwrap().record_busy(worker);
    }

    /// Record one steal attempt; `found_work` is `true` if it stole a job.
    pub fn record_steal(&self, found_work: bool) {
        self.monitor.lock().unwrap().record_steal(found_work);
    }

    /// Observe the current total queue depth, updating the running peak.
    pub fn observe_queue_depth(&self, depth: usize) {
        self.monitor.lock().unwrap().observe_queue_depth(depth);
    }

    /// Render the current deterministic [`HealthReport`].
    #[must_use]
    pub fn report(&self) -> HealthReport {
        self.monitor.lock().unwrap().report()
    }
}

/// Nanoseconds elapsed since `start`, saturating at [`u64::MAX`]. Factored out
/// so the clock dependency (the §24.8 honest boundary) has a single call site.
fn elapsed_nanos(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
