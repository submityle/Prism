//! Priority / `QoS` lanes with frame-budget-driven scheduling (design §24.1).
//!
//! A single FIFO queue cannot distinguish "must finish this frame" from "can
//! slip to idle time". This module adds **`QoS` lanes** keyed by [`Priority`]
//! plus a **frame-budget scheduler** that drains those lanes under an explicit
//! per-frame nanosecond budget:
//!
//! - The four foreground lanes ([`Priority::Low`]..=[`Priority::Critical`]) are
//!   committed work: every queued item runs this frame, highest lane first,
//!   FIFO within a lane.
//! - The [`Priority::Background`] lane (streaming, bakes, prefetch) is
//!   budget-gated: a background item runs this frame only while it still fits
//!   the frame's remaining headroom, and otherwise it is deferred to a later
//!   frame instead of blowing the frame time.
//!
//! The gating budget is exactly the `remaining_background_nanos` that
//! `prism_diagnostic`'s frame-budget registry reports (headroom left under the
//! frame target after measured foreground work). Feeding that value into
//! [`FrameScheduler::run_frame`] closes the loop the diagnostic crate documents
//! but does not drive: background jobs yield when the frame is tight.
//!
//! # Determinism
//! The policy lives in [`LaneQueues`], a pure, clock-free, `no_std`-friendly
//! core (see [`lane`]). Which items are admitted and which are deferred is a
//! deterministic function of the queue contents and the remaining headroom; the
//! only non-determinism is the order in which admitted jobs happen to finish on
//! the worker threads, which does not affect the returned
//! [`FrameRunReport`]. The core is tested directly against a serial oracle.
//!
//! # Layering
//! [`FrameScheduler`] is a thin façade over the existing [`TaskPool`]: admitted
//! jobs retain their explicit lanes through [`TaskPool::spawn_in_group`] and
//! join through [`TaskPool::wait`], so it inherits help-on-wait (no deadlock)
//! and the single-threaded inline fallback for free. The scheduling *decision* is
//! independent of the pool, so the core can also be embedded in subsystems that
//! own their own executor.

pub mod lane;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::job::Job;
use crate::priority::Priority;
use crate::TaskPool;

pub use lane::{admits_background, BudgetedItem, FrameStep, LanePlan, LaneQueues, LANE_COUNT};

/// Summary of one [`FrameScheduler::run_frame`] call.
///
/// Counts are over the items the scheduler acted on this frame; the fields are
/// a deterministic function of the queued work and the frame budget, so they
/// can be asserted exactly regardless of worker count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameRunReport {
    /// Foreground items (lanes `Low`..=`Critical`) run this frame.
    pub foreground_ran: usize,
    /// Background items admitted and run this frame.
    pub background_ran: usize,
    /// Background items left queued because they did not fit the budget; they
    /// carry over to a later frame.
    pub background_deferred: usize,
    /// Sum of the estimated costs of the admitted background items, in
    /// nanoseconds.
    pub background_nanos_consumed: u64,
    /// Background headroom still unspent when the frame ended, in nanoseconds.
    pub remaining_nanos_after: u64,
}

/// Jobs admitted from a [`FrameScheduler`] for one budget decision.
///
/// Admission and execution are deliberately separate: a caller that protects a
/// scheduler with a mutex can call [`FrameScheduler::prepare_frame`] while the
/// lock is held, release the lock, and only then execute this batch. Jobs in
/// the batch may therefore submit more work to the same scheduler without
/// deadlocking behind a runner that is waiting for them to finish.
#[must_use = "an admitted frame batch must be run or deliberately discarded"]
pub struct FrameBatch {
    pool: TaskPool,
    admitted: Vec<(Priority, Job)>,
    report: FrameRunReport,
}

impl FrameBatch {
    /// Deterministic admission report for this batch.
    #[must_use]
    pub fn report(&self) -> FrameRunReport {
        self.report
    }

    /// Whether this batch admitted no work.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.admitted.is_empty()
    }

    /// Dispatch every admitted job and wait for the batch to finish.
    ///
    /// The returned report is identical to [`FrameBatch::report`].
    pub fn run(self) -> FrameRunReport {
        let groups = [
            Priority::Background,
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
        ]
        .map(|lane| self.pool.priority_group(lane));
        for (lane, job) in self.admitted {
            self.pool.spawn_in_group(&groups[lane as usize], job);
        }
        for group in &groups {
            self.pool.wait(group.counter());
        }
        self.report
    }
}

/// A frame-budget-aware `QoS` scheduler bound to a [`TaskPool`].
///
/// Submit foreground work with [`FrameScheduler::submit_foreground`] and
/// deferrable work with [`FrameScheduler::submit_background`] (tagging each
/// background job with an estimated cost), then call
/// [`FrameScheduler::run_frame`] once per frame with the frame's remaining
/// background headroom. Deferred background work stays queued for the next
/// frame.
pub struct FrameScheduler {
    /// The lanes holding not-yet-run jobs.
    lanes: LaneQueues<Job>,
    /// The pool admitted jobs are dispatched to.
    pool: TaskPool,
}

impl FrameScheduler {
    /// Create a scheduler that dispatches onto `pool`.
    #[must_use]
    pub fn new(pool: TaskPool) -> Self {
        Self {
            lanes: LaneQueues::new(),
            pool,
        }
    }

    /// Redirect future admitted batches to `pool`, preserving every queued job.
    /// Already prepared batches retain their original pool.
    pub fn set_pool(&mut self, pool: TaskPool) {
        self.pool = pool;
    }

    /// Queue `f` on `lane` with estimated cost `est_nanos`. Prefer
    /// [`FrameScheduler::submit_foreground`] / [`FrameScheduler::submit_background`]
    /// for the common cases.
    pub fn submit<F>(&mut self, lane: Priority, est_nanos: u64, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job: Job = Box::new(f);
        self.lanes.push(lane, est_nanos, job);
    }

    /// Queue committed foreground work on `lane` (ignored budget cost). Panics
    /// if `lane` is [`Priority::Background`]; use
    /// [`FrameScheduler::submit_background`] for deferrable work.
    pub fn submit_foreground<F>(&mut self, lane: Priority, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        assert_ne!(
            lane,
            Priority::Background,
            "foreground submit on the background lane; use submit_background"
        );
        self.submit(lane, 0, f);
    }

    /// Queue deferrable background work with an estimated cost of `est_nanos`.
    pub fn submit_background<F>(&mut self, est_nanos: u64, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit(Priority::Background, est_nanos, f);
    }

    /// Number of items currently queued across all lanes.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.lanes.len()
    }

    /// Number of items queued on a specific `lane`.
    #[must_use]
    pub fn pending_in(&self, lane: Priority) -> usize {
        self.lanes.len_in(lane)
    }

    /// Whether no work is queued on any lane.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// Borrow the underlying lanes (e.g. to inspect the next decision with
    /// [`LaneQueues::peek_plan`]).
    #[must_use]
    pub fn lanes(&self) -> &LaneQueues<Job> {
        &self.lanes
    }

    /// Run one frame: drain every foreground lane and as much budget-gated
    /// background work as fits in `remaining_background_nanos`, dispatching each
    /// admitted job onto the pool and joining them before returning.
    ///
    /// `remaining_background_nanos` is the headroom the frame has for background
    /// work — pass `prism_diagnostic`'s `remaining_background_nanos`. Any
    /// background item that does not fit is left queued and reported in
    /// [`FrameRunReport::background_deferred`].
    pub fn run_frame(&mut self, remaining_background_nanos: u64) -> FrameRunReport {
        self.prepare_frame(remaining_background_nanos).run()
    }

    /// Decide which queued jobs fit this frame and move them into an executable
    /// batch without running them yet.
    ///
    /// This is the lock-friendly sibling of [`FrameScheduler::run_frame`]. The
    /// scheduler's FIFO queues are mutated only during admission; execution is
    /// owned by the returned [`FrameBatch`] and can happen after any outer lock
    /// protecting the scheduler has been released.
    pub fn prepare_frame(&mut self, remaining_background_nanos: u64) -> FrameBatch {
        self.prepare_batch(remaining_background_nanos, usize::MAX, true)
    }

    /// Admit all currently queued committed work without admitting Background.
    /// A wall-clock driver can execute this batch before measuring the headroom
    /// available for Background work.
    pub fn prepare_foreground(&mut self) -> FrameBatch {
        self.prepare_batch(0, usize::MAX, false)
    }

    /// Admit at most one job, leaving any Background head that does not fit
    /// queued. This lets a wall-clock driver recheck headroom between jobs.
    pub fn prepare_next(&mut self, remaining_background_nanos: u64) -> FrameBatch {
        self.prepare_batch(remaining_background_nanos, 1, true)
    }

    fn prepare_batch(
        &mut self,
        remaining_background_nanos: u64,
        limit: usize,
        include_background: bool,
    ) -> FrameBatch {
        let mut remaining = remaining_background_nanos;
        let mut report = FrameRunReport {
            remaining_nanos_after: remaining,
            ..FrameRunReport::default()
        };
        // Collect admitted jobs first so the deterministic decision is complete
        // before any worker starts; this keeps the report independent of
        // execution order.
        let mut admitted = Vec::new();
        while admitted.len() < limit {
            if !include_background
                && !matches!(
                    self.lanes.peek_plan(remaining),
                    LanePlan::Run { lane, .. } if lane != Priority::Background
                )
            {
                break;
            }
            match self.lanes.next_step(&mut remaining) {
                FrameStep::Ran {
                    lane,
                    est_nanos,
                    payload,
                } => {
                    if lane == Priority::Background {
                        report.background_ran += 1;
                        report.background_nanos_consumed += est_nanos;
                    } else {
                        report.foreground_ran += 1;
                    }
                    admitted.push((lane, payload));
                }
                FrameStep::DeferredBackground { pending, .. } => {
                    report.background_deferred = pending;
                    break;
                }
                FrameStep::Idle => break,
            }
        }
        report.remaining_nanos_after = remaining;
        FrameBatch {
            pool: self.pool.clone(),
            admitted,
            report,
        }
    }
}

impl TaskPool {
    /// Create a [`FrameScheduler`] that dispatches onto this pool (design
    /// §24.1).
    #[must_use]
    pub fn frame_scheduler(&self) -> FrameScheduler {
        FrameScheduler::new(self.clone())
    }
}
