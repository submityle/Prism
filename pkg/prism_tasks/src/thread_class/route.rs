//! Deterministic, `no_std`-friendly core for thread-class separation
//! (design §24.2).
//!
//! This module owns the *decision* half of §24.2 and holds no threads, no
//! clock, and no allocation beyond the per-class queues. Which executor a job
//! runs on, and how many jobs of each class are admitted in one dispatch wave,
//! is a pure function of the queued work and the per-lane admission budget, so
//! it is directly unit-testable against a serial oracle. The executor façade
//! that actually runs the routed jobs lives in the parent
//! [`thread_class`](crate::thread_class) module.
//!
//! # Classes and lanes
//! Work is tagged with a [`WorkClass`] describing *where it must run*, which
//! [`route`] maps to a physical [`ExecLane`]:
//!
//! - [`WorkClass::Compute`] → [`ExecLane::ComputePool`]: CPU-bound work for the
//!   work-stealing pool.
//! - [`WorkClass::Io`] → [`ExecLane::IoPool`]: blocking file / network work,
//!   kept off the compute pool so a stalled syscall never ties up a compute
//!   worker.
//! - [`WorkClass::MainThread`] → [`ExecLane::MainQueue`]: platform / GPU-submit
//!   work that only the main thread may run; it is queued for the main pump,
//!   never dispatched to a worker.
//!
//! # Isolation
//! The three classes live in independent FIFO lanes, so a job never leaves its
//! class: an I/O backlog cannot consume a compute slot and vice versa. The
//! per-wave [`LaneBudget`] caps how many compute / I/O jobs are admitted at
//! once (`usize::MAX` is unbounded); the main lane is always admitted because
//! queuing a main job costs no concurrency. Setting one lane's budget to `0`
//! leaves every other lane's dispatch untouched — the deterministic expression
//! of "blocking I/O does not starve compute".

use alloc::collections::VecDeque;

/// The kind of a unit of work, i.e. *which thread class* it must run on.
///
/// The discriminants are the stable lane indices used by [`ClassRouter`]; keep
/// [`WorkClass::Compute`] at `0` so the serve order and array indexing agree.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(usize)]
pub enum WorkClass {
    /// CPU-bound work for the work-stealing compute pool.
    Compute = 0,
    /// Blocking file / network I/O, run off the compute pool.
    Io = 1,
    /// Work pinned to the main thread (window / input events, GPU submission,
    /// main-thread-only platform APIs).
    MainThread = 2,
}

/// Number of distinct [`WorkClass`] lanes.
pub const CLASS_COUNT: usize = 3;

impl WorkClass {
    /// This class's stable lane index (`0..CLASS_COUNT`).
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The physical [`ExecLane`] this class routes to. Convenience wrapper over
    /// [`route`].
    #[must_use]
    #[inline]
    pub const fn exec_lane(self) -> ExecLane {
        route(self)
    }
}

/// The physical execution lane a [`WorkClass`] routes to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ExecLane {
    /// The work-stealing compute pool (`TaskPool`).
    ComputePool,
    /// The off-pool blocking I/O lane.
    IoPool,
    /// The main-thread queue, drained by the main pump.
    MainQueue,
}

/// The routing table: map a [`WorkClass`] to the [`ExecLane`] that runs it.
///
/// A total, side-effect-free function — the single source of truth both the
/// façade and the tests route through.
#[must_use]
#[inline]
pub const fn route(class: WorkClass) -> ExecLane {
    match class {
        WorkClass::Compute => ExecLane::ComputePool,
        WorkClass::Io => ExecLane::IoPool,
        WorkClass::MainThread => ExecLane::MainQueue,
    }
}

/// Per-wave admission budget: how many compute / I/O jobs may be dispatched in
/// a single [`ClassRouter::next_step`] drain. `usize::MAX` means unbounded.
///
/// The [`WorkClass::MainThread`] lane has no entry: queuing a main job costs no
/// concurrency, so it is always admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneBudget {
    /// Compute jobs admissible this wave.
    pub compute_slots: usize,
    /// I/O jobs admissible this wave.
    pub io_slots: usize,
}

impl LaneBudget {
    /// A budget that admits every queued job (`usize::MAX` per lane).
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            compute_slots: usize::MAX,
            io_slots: usize::MAX,
        }
    }

    /// A budget with explicit compute / I/O caps.
    #[must_use]
    pub const fn new(compute_slots: usize, io_slots: usize) -> Self {
        Self {
            compute_slots,
            io_slots,
        }
    }

    /// Remaining admissions for `class`. The main lane is always unbounded.
    #[must_use]
    #[inline]
    pub const fn slots_for(&self, class: WorkClass) -> usize {
        match class {
            WorkClass::Compute => self.compute_slots,
            WorkClass::Io => self.io_slots,
            WorkClass::MainThread => usize::MAX,
        }
    }
}

impl Default for LaneBudget {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Pure admission test: may another job of `class` be dispatched given the
/// slots still free in `budget`?
///
/// The [`WorkClass::MainThread`] lane is always admissible.
#[must_use]
#[inline]
pub fn admits(budget: &LaneBudget, class: WorkClass) -> bool {
    match class {
        WorkClass::MainThread => true,
        _ => budget.slots_for(class) > 0,
    }
}

/// Classes in the order [`ClassRouter::next_step`] serves them. Compute is
/// drained first (keep the cores fed), then I/O, then the main queue. The
/// lanes are independent, so this order only fixes the dispatch *sequence* for
/// deterministic testing, not any cross-lane priority.
const SERVE_ORDER: [WorkClass; CLASS_COUNT] =
    [WorkClass::Compute, WorkClass::Io, WorkClass::MainThread];

/// A non-consuming view of what [`ClassRouter::next_step`] would do next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutePlan {
    /// The head of `class` would be dispatched to `lane` next.
    Dispatch {
        /// Class whose head would be dispatched.
        class: WorkClass,
        /// Lane the head would run on.
        lane: ExecLane,
    },
    /// Nothing is admissible: either every lane is empty, or the only queued
    /// work is budget-capped compute / I/O with no slots left this wave.
    Idle {
        /// Compute jobs left queued (deferred to a later wave).
        compute_deferred: usize,
        /// I/O jobs left queued (deferred to a later wave).
        io_deferred: usize,
    },
}

/// The outcome of a single [`ClassRouter::next_step`], carrying the dequeued
/// payload when one was dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteStep<T> {
    /// `payload` was dequeued from `class` and should run on `lane` now. For a
    /// budget-capped lane, one slot has already been deducted from the caller's
    /// [`LaneBudget`].
    Dispatch {
        /// Class the payload came from.
        class: WorkClass,
        /// Lane the payload should run on.
        lane: ExecLane,
        /// The dequeued payload.
        payload: T,
    },
    /// Nothing more is admissible this wave; any still-queued compute / I/O is
    /// reported for carry-over to a later wave.
    Idle {
        /// Compute jobs left queued.
        compute_deferred: usize,
        /// I/O jobs left queued.
        io_deferred: usize,
    },
}

/// Three FIFO lanes keyed by [`WorkClass`], drained under a per-wave
/// [`LaneBudget`].
///
/// The container is generic over the payload `T`, so the same policy drives
/// both the real executor façade (`T = Job`) and deterministic unit tests
/// (`T = u32` tags).
#[derive(Debug)]
pub struct ClassRouter<T> {
    /// One queue per class, indexed by [`WorkClass::index`].
    lanes: [VecDeque<T>; CLASS_COUNT],
}

impl<T> Default for ClassRouter<T> {
    fn default() -> Self {
        Self {
            lanes: core::array::from_fn(|_| VecDeque::new()),
        }
    }
}

impl<T> ClassRouter<T> {
    /// Create an empty router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Immutable access to the queue backing `class`.
    #[inline]
    fn lane(&self, class: WorkClass) -> &VecDeque<T> {
        &self.lanes[class.index()]
    }

    /// Mutable access to the queue backing `class`.
    #[inline]
    fn lane_mut(&mut self, class: WorkClass) -> &mut VecDeque<T> {
        &mut self.lanes[class.index()]
    }

    /// Enqueue `payload` on `class`, behind any items already queued on that
    /// class (FIFO within a class).
    pub fn push(&mut self, class: WorkClass, payload: T) {
        self.lane_mut(class).push_back(payload);
    }

    /// Total number of queued items across every class.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lanes.iter().map(VecDeque::len).sum()
    }

    /// Whether every class queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lanes.iter().all(VecDeque::is_empty)
    }

    /// Number of items currently queued on `class`.
    #[must_use]
    pub fn len_in(&self, class: WorkClass) -> usize {
        self.lane(class).len()
    }

    /// The first class in serve order that has work *and* a free slot under
    /// `budget`, if any.
    #[must_use]
    fn next_admissible(&self, budget: &LaneBudget) -> Option<WorkClass> {
        SERVE_ORDER
            .into_iter()
            .find(|&class| !self.lane(class).is_empty() && admits(budget, class))
    }

    /// Decide, without dequeuing, what [`ClassRouter::next_step`] would do under
    /// `budget`.
    #[must_use]
    pub fn peek_plan(&self, budget: &LaneBudget) -> RoutePlan {
        match self.next_admissible(budget) {
            Some(class) => RoutePlan::Dispatch {
                class,
                lane: route(class),
            },
            None => RoutePlan::Idle {
                compute_deferred: self.len_in(WorkClass::Compute),
                io_deferred: self.len_in(WorkClass::Io),
            },
        }
    }

    /// Dequeue and return the next item to dispatch under the §24.2 policy,
    /// deducting one slot from `budget` when a compute / I/O item is admitted.
    ///
    /// Classes are served highest-first in [`SERVE_ORDER`]. A class with queued
    /// work but no free slot is skipped (its work carries over); the main lane
    /// is always admissible. When nothing is admissible the step reports
    /// [`RouteStep::Idle`] with the still-queued compute / I/O counts.
    pub fn next_step(&mut self, budget: &mut LaneBudget) -> RouteStep<T> {
        let Some(class) = self.next_admissible(budget) else {
            return RouteStep::Idle {
                compute_deferred: self.len_in(WorkClass::Compute),
                io_deferred: self.len_in(WorkClass::Io),
            };
        };
        match class {
            WorkClass::Compute => budget.compute_slots -= 1,
            WorkClass::Io => budget.io_slots -= 1,
            WorkClass::MainThread => {}
        }
        let payload = self
            .lane_mut(class)
            .pop_front()
            .expect("class reported non-empty");
        RouteStep::Dispatch {
            class,
            lane: route(class),
            payload,
        }
    }
}
