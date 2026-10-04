//! Handles and fired-event records for the deterministic
//! [`Scheduler`](super::Scheduler).

use crate::Duration;

/// A stable, cheap handle to a scheduled timer.
///
/// Returned by every `schedule_*` call and accepted by
/// [`Scheduler::cancel`](super::Scheduler::cancel),
/// [`Scheduler::reschedule`](super::Scheduler::reschedule), and
/// [`Scheduler::is_active`](super::Scheduler::is_active). It stays valid across
/// a [`reschedule`](super::Scheduler::reschedule); it is invalidated once the
/// timer is cancelled or (for a one-shot) has fired, after which its slot may
/// be reused by a later schedule under a fresh `generation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimerHandle {
    /// Index of the backing slot in the scheduler's slot table.
    pub(super) index: u32,
    /// Reuse generation of that slot; distinguishes a reused slot from the
    /// retired handle that previously owned it.
    pub(super) generation: u32,
}

impl TimerHandle {
    /// The backing slot index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The slot reuse generation this handle was issued for.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Whether a scheduled timer fires once or repeats on a fixed period.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimerKind {
    /// A one-shot timer: it fires exactly once, then its slot is retired.
    Once,
    /// A periodic timer: it re-arms for `+period` after each fire until
    /// cancelled.
    Repeat,
}

/// One fired timer event, produced by [`Scheduler::advance`](super::Scheduler::advance)
/// (or [`drain_due`](super::Scheduler::drain_due)).
///
/// Events are returned in deterministic order: ascending scheduled fire time
/// [`at`](Self::at), ties broken by original schedule order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fired<T> {
    /// Handle of the timer that fired. For a [`TimerKind::Repeat`] timer this
    /// handle is still live (re-armed); for [`TimerKind::Once`] it is now
    /// retired.
    pub handle: TimerHandle,
    /// The payload the timer was scheduled with (cloned on each fire, so a
    /// periodic timer yields one payload per occurrence).
    pub payload: T,
    /// The exact scheduled fire time, as elapsed scheduler time. This is the
    /// *scheduled* instant, which may be `<=` the scheduler's current elapsed
    /// time when a single advance crossed several periods.
    pub at: Duration,
    /// Whether this came from a one-shot or periodic timer.
    pub kind: TimerKind,
}
