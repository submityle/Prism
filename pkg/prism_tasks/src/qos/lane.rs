//! Deterministic, `no_std`-friendly core for the `QoS` lane / frame-budget
//! scheduler (design §24.1).
//!
//! This module owns the *decision* half of §24.1 and holds no threads, no
//! clock, and no allocation beyond the per-lane queues. The scheduling policy
//! is a pure function of the current queue contents and the frame's remaining
//! background headroom, so it is directly unit-testable against a serial
//! oracle. The thread-pool façade that actually runs the admitted jobs lives in
//! the parent [`qos`](crate::qos) module.
//!
//! # Lanes
//! A lane is identified by [`Priority`]. The four foreground lanes
//! ([`Priority::Low`] through [`Priority::Critical`]) are *committed* work: the
//! scheduler always drains them this frame, highest-first, FIFO within a lane.
//! The single [`Priority::Background`] lane is *budget-gated*: a background item
//! runs this frame only while it still fits in the remaining headroom, and
//! otherwise it is deferred to a later frame.
//!
//! # Budget alignment
//! The remaining headroom passed to [`LaneQueues::next_step`] is exactly the
//! `remaining_background_nanos` reported by `prism_diagnostic`'s frame-budget
//! registry (headroom left under the frame target after measured foreground
//! work). Admitting a background item deducts its estimated cost from that
//! headroom, so later background items in the same frame see a shrinking budget
//! — the same "background yields when the frame is tight" contract, driven from
//! the scheduler side.

use alloc::collections::VecDeque;

use crate::priority::Priority;

/// Number of distinct [`Priority`] lanes (`Background`, `Low`, `Normal`,
/// `High`, `Critical`).
pub const LANE_COUNT: usize = 5;

/// Foreground lanes in descending service order. [`Priority::Background`] is
/// deliberately absent: it is the only budget-gated lane and is handled
/// separately by [`LaneQueues::next_step`].
const FOREGROUND_DESCENDING: [Priority; 4] = [
    Priority::Critical,
    Priority::High,
    Priority::Normal,
    Priority::Low,
];

/// Pure admission test for a background item: it may run this frame only if its
/// estimated cost fits within `remaining_nanos`.
///
/// A zero-cost item is always admitted, even when `remaining_nanos` is `0`, so
/// genuinely free background work never starves on an exhausted frame.
#[must_use]
#[inline]
pub fn admits_background(remaining_nanos: u64, est_nanos: u64) -> bool {
    est_nanos <= remaining_nanos
}

/// One queued item: an estimated cost in nanoseconds plus an arbitrary payload
/// (a job closure in the façade, a tag in tests).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetedItem<T> {
    /// Estimated cost of running this item, in nanoseconds. Only meaningful for
    /// the [`Priority::Background`] lane, where it gates admission; foreground
    /// lanes ignore it.
    pub est_nanos: u64,
    /// The payload to hand back when the item is served.
    pub payload: T,
}

/// A non-consuming view of what [`LaneQueues::next_step`] would do next, with no
/// payload moved out. Useful for inspection and for asserting the policy
/// without draining the queues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanePlan {
    /// The head of `lane` would be served next.
    Run {
        /// Lane whose head would run.
        lane: Priority,
        /// The head item's estimated cost in nanoseconds.
        est_nanos: u64,
    },
    /// All foreground lanes are empty and the background head does not fit the
    /// remaining headroom, so background work is deferred to a later frame.
    DeferBackground {
        /// Number of background items still waiting.
        pending: usize,
        /// Estimated cost of the background head that did not fit.
        head_est_nanos: u64,
    },
    /// Every lane is empty; there is nothing to run.
    Idle,
}

/// The outcome of a single [`LaneQueues::next_step`], carrying the served
/// payload when one was dequeued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStep<T> {
    /// An item was dequeued from `lane` and should run now. For the background
    /// lane, `est_nanos` has already been deducted from the caller's remaining
    /// headroom.
    Ran {
        /// Lane the item came from.
        lane: Priority,
        /// The item's estimated cost in nanoseconds.
        est_nanos: u64,
        /// The dequeued payload.
        payload: T,
    },
    /// Foreground work is exhausted and the background head does not fit the
    /// remaining headroom; the frame is done and background work is deferred.
    DeferredBackground {
        /// Number of background items still waiting (all deferred).
        pending: usize,
        /// Estimated cost of the background head that did not fit.
        head_est_nanos: u64,
        /// Headroom left when the deferral was decided, in nanoseconds.
        remaining_nanos: u64,
    },
    /// Every lane is empty.
    Idle,
}

/// Five FIFO lanes keyed by [`Priority`], drained highest-first with the
/// background lane gated by a per-frame nanosecond budget.
///
/// The container is generic over the payload `T`, so the exact same policy
/// drives both the real thread-pool façade (`T = Job`) and deterministic unit
/// tests (`T = u32` tags).
#[derive(Debug)]
pub struct LaneQueues<T> {
    /// One queue per lane, indexed by the [`Priority`] discriminant.
    lanes: [VecDeque<BudgetedItem<T>>; LANE_COUNT],
}

impl<T> Default for LaneQueues<T> {
    fn default() -> Self {
        Self {
            lanes: core::array::from_fn(|_| VecDeque::new()),
        }
    }
}

impl<T> LaneQueues<T> {
    /// Create an empty set of lanes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Immutable access to the queue backing `lane`.
    #[inline]
    fn lane(&self, lane: Priority) -> &VecDeque<BudgetedItem<T>> {
        &self.lanes[lane as usize]
    }

    /// Mutable access to the queue backing `lane`.
    #[inline]
    fn lane_mut(&mut self, lane: Priority) -> &mut VecDeque<BudgetedItem<T>> {
        &mut self.lanes[lane as usize]
    }

    /// Enqueue `payload` on `lane` with estimated cost `est_nanos`, behind any
    /// items already queued on that lane (FIFO within a lane).
    pub fn push(&mut self, lane: Priority, est_nanos: u64, payload: T) {
        self.lane_mut(lane)
            .push_back(BudgetedItem { est_nanos, payload });
    }

    /// Total number of queued items across every lane.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lanes.iter().map(VecDeque::len).sum()
    }

    /// Whether every lane is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lanes.iter().all(VecDeque::is_empty)
    }

    /// Number of items currently queued on `lane`.
    #[must_use]
    pub fn len_in(&self, lane: Priority) -> usize {
        self.lane(lane).len()
    }

    /// The highest-priority foreground lane that currently has work, if any.
    #[must_use]
    fn highest_foreground_nonempty(&self) -> Option<Priority> {
        FOREGROUND_DESCENDING
            .into_iter()
            .find(|&lane| !self.lane(lane).is_empty())
    }

    /// Estimated cost of the background head, if the background lane is
    /// non-empty.
    #[must_use]
    fn background_head_est(&self) -> Option<u64> {
        self.lane(Priority::Background)
            .front()
            .map(|item| item.est_nanos)
    }

    /// Decide, without dequeuing, what [`LaneQueues::next_step`] would do for
    /// the given `remaining_nanos` headroom.
    #[must_use]
    pub fn peek_plan(&self, remaining_nanos: u64) -> LanePlan {
        if let Some(lane) = self.highest_foreground_nonempty() {
            let est_nanos = self.lane(lane).front().map_or(0, |item| item.est_nanos);
            return LanePlan::Run { lane, est_nanos };
        }
        match self.background_head_est() {
            Some(est_nanos) if admits_background(remaining_nanos, est_nanos) => LanePlan::Run {
                lane: Priority::Background,
                est_nanos,
            },
            Some(head_est_nanos) => LanePlan::DeferBackground {
                pending: self.len_in(Priority::Background),
                head_est_nanos,
            },
            None => LanePlan::Idle,
        }
    }

    /// Dequeue and return the next item to run under the §24.1 policy, updating
    /// `remaining_nanos` in place when a background item is admitted.
    ///
    /// Foreground lanes are served highest-first and never touch
    /// `remaining_nanos`. Once foreground work is exhausted, the background head
    /// runs only while [`admits_background`] holds; otherwise the step reports
    /// [`FrameStep::DeferredBackground`] and leaves the background lane intact
    /// for a later frame.
    pub fn next_step(&mut self, remaining_nanos: &mut u64) -> FrameStep<T> {
        if let Some(lane) = self.highest_foreground_nonempty() {
            let item = self
                .lane_mut(lane)
                .pop_front()
                .expect("lane reported non-empty");
            return FrameStep::Ran {
                lane,
                est_nanos: item.est_nanos,
                payload: item.payload,
            };
        }
        match self.background_head_est() {
            Some(est_nanos) if admits_background(*remaining_nanos, est_nanos) => {
                *remaining_nanos -= est_nanos;
                let item = self
                    .lane_mut(Priority::Background)
                    .pop_front()
                    .expect("background head present");
                FrameStep::Ran {
                    lane: Priority::Background,
                    est_nanos: item.est_nanos,
                    payload: item.payload,
                }
            }
            Some(head_est_nanos) => FrameStep::DeferredBackground {
                pending: self.len_in(Priority::Background),
                head_est_nanos,
                remaining_nanos: *remaining_nanos,
            },
            None => FrameStep::Idle,
        }
    }
}
