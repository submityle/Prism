//! The wait-set: fibers suspended on a [`Counter`], plus the queue of fibers
//! made resumable once their counter reached zero.
//!
//! ## Protocol and the park-after-switch race
//! A fiber that calls `wait` on a non-complete counter does *not* register
//! itself before switching away. Instead it saves its context (switching back
//! to the worker's scheduler loop) and only *then* does the worker call
//! [`WaitSet::park`]. This ordering is essential: by the time the fiber appears
//! in the wait-set its machine context is fully saved, so another worker may
//! safely resume it without two threads ever touching one fiber stack.
//!
//! The window between "fiber decided to wait" and "worker parks it" is closed
//! by re-checking completion under the lock:
//! - [`WaitSet::park`] takes the parked-list lock and, if the counter is
//!   *already* complete, routes the fiber straight to the resume queue instead
//!   of parking it (so a completion that landed during the switch is not lost).
//! - [`WaitSet::flush`] takes the same lock and moves every now-complete waiter
//!   to the resume queue.
//!
//! Because `park` and `flush` serialize on the parked-list mutex and both
//! observe the counter under it, no wakeup can be missed: either `park` sees the
//! completion directly, or the fiber is parked and a subsequent `flush` sees it.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::FiberPtr;
use crate::Counter;

/// A fiber parked on a counter it is waiting to reach zero.
struct Waiter {
    fiber: FiberPtr,
    counter: Counter,
}

/// Suspended fibers waiting on counters, and the queue of ready-to-resume
/// fibers whose counters have since completed.
pub(crate) struct WaitSet {
    /// Fibers currently blocked on a non-complete counter.
    parked: Mutex<Vec<Waiter>>,
    /// Fibers whose counter is complete, awaiting pickup by a worker.
    resume: Mutex<VecDeque<FiberPtr>>,
}

impl WaitSet {
    /// Create an empty wait-set.
    pub(crate) fn new() -> Self {
        Self {
            parked: Mutex::new(Vec::new()),
            resume: Mutex::new(VecDeque::new()),
        }
    }

    /// Park `fiber` on `counter`, or route it directly to the resume queue if
    /// `counter` has already completed. Called by the worker *after* the fiber
    /// has fully switched out, so the fiber's context is saved and the fiber is
    /// safe to resume on any thread.
    ///
    /// Lock order is always parked → resume (shared with [`WaitSet::flush`]),
    /// so the two can never deadlock.
    pub(crate) fn park(&self, fiber: FiberPtr, counter: Counter) {
        let mut parked = self.parked.lock().unwrap();
        if counter.is_complete() {
            self.resume.lock().unwrap().push_back(fiber);
        } else {
            parked.push(Waiter { fiber, counter });
        }
    }

    /// Move every parked fiber whose counter is now complete to the resume
    /// queue. Returns `true` if at least one fiber became resumable.
    pub(crate) fn flush(&self) -> bool {
        let mut parked = self.parked.lock().unwrap();
        let mut ready: Vec<FiberPtr> = Vec::new();
        let mut i = 0;
        while i < parked.len() {
            if parked[i].counter.is_complete() {
                ready.push(parked.swap_remove(i).fiber);
            } else {
                i += 1;
            }
        }
        if ready.is_empty() {
            return false;
        }
        let mut resume = self.resume.lock().unwrap();
        for fiber in ready {
            resume.push_back(fiber);
        }
        true
    }

    /// Pop one ready-to-resume fiber, if any.
    pub(crate) fn pop_resume(&self) -> Option<FiberPtr> {
        self.resume.lock().unwrap().pop_front()
    }

    /// Whether the resume queue is currently empty.
    pub(crate) fn resume_is_empty(&self) -> bool {
        self.resume.lock().unwrap().is_empty()
    }

}
