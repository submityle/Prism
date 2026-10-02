//! The cooperative, preemptible work scheduler.
//!
//! [`Scheduler`] holds one FIFO queue per [`Lane`] and drains them under a
//! [`Deadline`]. Each outer iteration re-selects the most urgent non-empty
//! lane, so a unit enqueued on a higher lane mid-run preempts whatever lower
//! unit was about to resume — React Fiber's lane preemption, expressed as a
//! deterministic single-threaded loop. A unit that reports
//! [`StepOutcome::More`] is pushed to the back of its lane, giving sibling
//! units on the same lane a fair, round-robin slice before it runs again.

use alloc::boxed::Box;
use alloc::collections::VecDeque;

use crate::budget::{Clock, Deadline};
use crate::lane::{Lane, LaneMask};
use crate::work::{BoxWork, StepOutcome, Work};

/// Why a [`Scheduler::run`] call returned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StopReason {
    /// Every lane is empty; all scheduled work completed.
    Drained,
    /// The deadline expired with work still pending.
    YieldedToDeadline,
}

/// A summary of one [`Scheduler::run`] slice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RunReport {
    /// Why the slice ended.
    pub stop: StopReason,
    /// Number of [`Work::step`] calls made during the slice.
    pub steps: u32,
    /// Number of units that reported [`StepOutcome::Done`] during the slice.
    pub completed: u32,
    /// Lanes still holding work when the slice ended.
    pub pending: LaneMask,
}

impl RunReport {
    /// Whether all scheduled work finished.
    #[must_use]
    pub const fn is_drained(self) -> bool {
        matches!(self.stop, StopReason::Drained)
    }
}

/// A priority-laned, budget-driven reconciliation scheduler.
///
/// The scheduler owns no clock and spawns no threads; the host advances it by
/// calling [`run`](Scheduler::run) once per frame with that frame's deadline.
#[derive(Default)]
pub struct Scheduler {
    lanes: [VecDeque<BoxWork>; Lane::COUNT],
    mask: LaneMask,
}

impl Scheduler {
    /// Creates an empty scheduler.
    #[must_use]
    pub fn new() -> Scheduler {
        Scheduler {
            lanes: [const { VecDeque::new() }; Lane::COUNT],
            mask: LaneMask::EMPTY,
        }
    }

    /// Enqueues `work` on `lane`, to run after units already queued there.
    pub fn enqueue(&mut self, lane: Lane, work: BoxWork) {
        self.lanes[lane.index()].push_back(work);
        self.mask.insert(lane);
    }

    /// Enqueues a concrete [`Work`] value on `lane`, boxing it.
    pub fn schedule<W: Work + 'static>(&mut self, lane: Lane, work: W) {
        self.enqueue(lane, Box::new(work));
    }

    /// The set of lanes that currently hold work.
    #[must_use]
    pub fn pending_lanes(&self) -> LaneMask {
        self.mask
    }

    /// Whether no lane holds work.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.mask.is_empty()
    }

    /// The number of units waiting on `lane`.
    #[must_use]
    pub fn lane_len(&self, lane: Lane) -> usize {
        self.lanes[lane.index()].len()
    }

    /// The total number of units waiting across all lanes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lanes.iter().map(VecDeque::len).sum()
    }

    /// Whether the scheduler holds no units; see [`is_idle`](Scheduler::is_idle).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.is_idle()
    }

    /// Drains work by urgency until `deadline` expires or all lanes empty.
    ///
    /// Between every step the clock is consulted, so an expired deadline yields
    /// promptly while leaving unfinished units queued for the next frame. The
    /// most urgent non-empty lane is re-selected each iteration, so higher-lane
    /// units enqueued by a running step preempt lower-lane units.
    pub fn run(&mut self, clock: &impl Clock, deadline: Deadline) -> RunReport {
        let mut steps = 0u32;
        let mut completed = 0u32;
        loop {
            let Some(lane) = self.mask.highest() else {
                return RunReport {
                    stop: StopReason::Drained,
                    steps,
                    completed,
                    pending: self.mask,
                };
            };
            if deadline.is_expired(clock) {
                return RunReport {
                    stop: StopReason::YieldedToDeadline,
                    steps,
                    completed,
                    pending: self.mask,
                };
            }
            let Some(mut work) = self.lanes[lane.index()].pop_front() else {
                // Mask said this lane was non-empty; keep it consistent.
                self.mask.remove(lane);
                continue;
            };
            let outcome = work.step();
            steps = steps.saturating_add(1);
            match outcome {
                StepOutcome::More => self.lanes[lane.index()].push_back(work),
                StepOutcome::Done => completed = completed.saturating_add(1),
            }
            if self.lanes[lane.index()].is_empty() {
                self.mask.remove(lane);
            }
        }
    }

    /// Runs with [`Deadline::NEVER`], draining every lane to completion.
    ///
    /// Equivalent to a synchronous flush; useful for tests and for the final
    /// commit frame where partial work is not acceptable.
    pub fn run_to_completion(&mut self, clock: &impl Clock) -> RunReport {
        self.run(clock, Deadline::NEVER)
    }

    /// Drops every pending unit on `lane` and clears its mask bit.
    ///
    /// Models cancelling a superseded transition (for example a stale
    /// off-screen build after the viewport jumped elsewhere).
    pub fn cancel_lane(&mut self, lane: Lane) {
        self.lanes[lane.index()].clear();
        self.mask.remove(lane);
    }
}
