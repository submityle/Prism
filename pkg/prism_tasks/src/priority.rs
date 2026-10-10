//! Priority lanes with priority inheritance (design §17 优先级继承, §24.1 `QoS`
//! 车道).
//!
//! A single FIFO queue cannot distinguish "must finish this frame" from "can
//! slip to idle time". A [`PriorityGroup`] tags its jobs with a [`Priority`]
//! and pushes them onto a priority inbox the scheduler drains **highest-first**
//! (FIFO within a level). The group's priority lives in a shared
//! [`PriorityCell`]; raising it retroactively promotes every not-yet-started
//! job in the group.
//!
//! That shared cell is how **priority inheritance** avoids priority inversion:
//! when a high-priority consumer must wait on a lower-priority group's results
//! ([`TaskPool::wait_inherited`]), it first boosts the group's cell to its own
//! priority, so the prerequisites it is blocked on jump ahead of unrelated
//! low-priority work instead of being starved behind it.
//!
//! Boosts are **monotonic**: a cell's effective priority only ever rises, so an
//! inheritance never accidentally demotes work that another waiter already
//! promoted.

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::job::Job;
use crate::{Counter, TaskPool};

/// Scheduling priority of a job group, ordered `Background < Low < Normal <
/// High < Critical`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Priority {
    /// Lowest: deferrable background work (e.g. prefetch, idle bakes).
    Background = 0,
    /// Below-normal work that should yield to interactive jobs.
    Low = 1,
    /// Default priority for ordinary work.
    #[default]
    Normal = 2,
    /// Above-normal work that should preempt background lanes.
    High = 3,
    /// Highest: must-run-now work (e.g. the frame's critical path).
    Critical = 4,
}

impl Priority {
    /// Reconstruct a [`Priority`] from its `u8` discriminant, saturating any
    /// out-of-range value to [`Priority::Critical`].
    fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Background,
            1 => Self::Low,
            2 => Self::Normal,
            3 => Self::High,
            _ => Self::Critical,
        }
    }
}

/// A shared, monotonically boostable priority level.
///
/// Every job in a [`PriorityGroup`] holds a clone of the same cell, so a single
/// [`PriorityCell::boost_to`] promotes the whole group at once. The level never
/// decreases.
#[derive(Debug)]
pub struct PriorityCell {
    /// The current effective priority as a [`Priority`] discriminant.
    bits: AtomicU8,
}

impl PriorityCell {
    /// Create a cell starting at `priority`.
    #[must_use]
    pub fn new(priority: Priority) -> Self {
        Self {
            bits: AtomicU8::new(priority as u8),
        }
    }

    /// The current effective priority.
    #[must_use]
    pub fn get(&self) -> Priority {
        Priority::from_bits(self.bits.load(Ordering::Acquire))
    }

    /// Raise the effective priority to `priority` if it is higher than the
    /// current level; never lowers it.
    pub fn boost_to(&self, priority: Priority) {
        let target = priority as u8;
        let mut current = self.bits.load(Ordering::Acquire);
        while target > current {
            match self.bits.compare_exchange_weak(
                current,
                target,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

/// One queued job plus the shared cell that gives it its effective priority.
struct PriorityEntry {
    /// Shared priority of the owning group.
    cell: Arc<PriorityCell>,
    /// Monotonic insertion sequence for FIFO tie-breaking within a level.
    seq: u64,
    /// The job to run.
    job: Job,
}

/// Scheduler-side priority queue drained highest-priority-first, FIFO within a
/// level. Kept `pub(crate)`: it is wired into the scheduler's `find_task`.
pub(crate) struct PriorityInbox {
    /// All pending prioritized jobs; scanned on pop (correctness over an O(1)
    /// heap, because boosts re-order entries in place and the inversion path is
    /// rare relative to the main deques).
    queue: Mutex<Vec<PriorityEntry>>,
    /// Insertion sequence source for FIFO tie-breaking.
    seq: AtomicU64,
    /// Cached length for a lock-free empty check on the hot `find_task` path.
    len: AtomicUsize,
}

impl PriorityInbox {
    /// Create an empty inbox.
    pub(crate) fn new() -> Self {
        Self {
            queue: Mutex::new(Vec::new()),
            seq: AtomicU64::new(0),
            len: AtomicUsize::new(0),
        }
    }

    /// Enqueue `job` under the group priority `cell`.
    pub(crate) fn push(&self, cell: Arc<PriorityCell>, job: Job) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let mut queue = self.queue.lock().unwrap();
        queue.push(PriorityEntry { cell, seq, job });
        self.len.store(queue.len(), Ordering::Release);
    }

    /// Pop the highest-priority job (FIFO within a level), or `None` if empty.
    pub(crate) fn pop_highest(&self) -> Option<Job> {
        // Lock-free fast path keeps the non-priority workload off the lock.
        if self.len.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut queue = self.queue.lock().unwrap();
        if queue.is_empty() {
            return None;
        }
        let mut best_idx = 0usize;
        let mut best_prio = queue[0].cell.get();
        let mut best_seq = queue[0].seq;
        for (idx, entry) in queue.iter().enumerate().skip(1) {
            let prio = entry.cell.get();
            if prio > best_prio || (prio == best_prio && entry.seq < best_seq) {
                best_idx = idx;
                best_prio = prio;
                best_seq = entry.seq;
            }
        }
        let entry = queue.swap_remove(best_idx);
        self.len.store(queue.len(), Ordering::Release);
        Some(entry.job)
    }
}

/// A group of jobs sharing one priority, with support for priority inheritance.
///
/// Create it with [`TaskPool::priority_group`], spawn into it with
/// [`TaskPool::spawn_in_group`], and wait on it (optionally boosting it to the
/// waiter's priority) with [`TaskPool::wait_inherited`].
pub struct PriorityGroup {
    /// Shared, boostable priority for every job in the group.
    cell: Arc<PriorityCell>,
    /// Fork-join counter tracking the group's jobs.
    counter: Counter,
    /// The group's base priority at creation.
    base: Priority,
}

impl PriorityGroup {
    /// The base priority the group was created with.
    #[must_use]
    pub fn base(&self) -> Priority {
        self.base
    }

    /// The current effective priority (base, possibly raised by inheritance).
    #[must_use]
    pub fn effective_priority(&self) -> Priority {
        self.cell.get()
    }

    /// The fork-join counter tracking this group's jobs.
    #[must_use]
    pub fn counter(&self) -> &Counter {
        &self.counter
    }

    /// Whether every job in the group has completed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.counter.is_complete()
    }

    /// Boost the group (and thus every pending job) to at least `priority`.
    pub fn boost_to(&self, priority: Priority) {
        self.cell.boost_to(priority);
    }
}

impl TaskPool {
    /// Create a [`PriorityGroup`] whose jobs start at `base` priority.
    #[must_use]
    pub fn priority_group(&self, base: Priority) -> PriorityGroup {
        PriorityGroup {
            cell: Arc::new(PriorityCell::new(base)),
            counter: Counter::new(),
            base,
        }
    }

    /// Spawn `f` into `group`, scheduling it on the priority inbox at the
    /// group's effective priority.
    ///
    /// In the single-threaded fallback the job runs inline, so priority
    /// ordering is moot but the group counter is still maintained.
    pub fn spawn_in_group<F>(&self, group: &PriorityGroup, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        group.counter.add(1);
        let counter = group.counter.clone();
        let job = move || {
            f();
            counter.finish_one();
        };
        if self.is_single_threaded() {
            job();
        } else {
            self.shared_state()
                .push_prioritized(Arc::clone(&group.cell), Box::new(job));
        }
    }

    /// Wait for `group` to finish, first boosting it to `waiter` priority so its
    /// pending jobs inherit the waiter's urgency (priority-inheritance,
    /// anti-inversion).
    ///
    /// The calling thread helps drive the pool while waiting, exactly like
    /// [`TaskPool::wait`], so this cannot deadlock. The boost is monotonic and
    /// takes effect for any of the group's jobs not yet picked up by a worker.
    pub fn wait_inherited(&self, group: &PriorityGroup, waiter: Priority) {
        group.cell.boost_to(waiter);
        self.wait(&group.counter);
    }
}

#[cfg(test)]
mod tests {
    use super::{Priority, PriorityCell, PriorityInbox};
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use std::sync::Mutex;

    #[test]
    fn priority_is_ordered() {
        assert!(Priority::Background < Priority::Low);
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn cell_boost_is_monotonic() {
        let cell = PriorityCell::new(Priority::Low);
        assert_eq!(cell.get(), Priority::Low);
        cell.boost_to(Priority::High);
        assert_eq!(cell.get(), Priority::High);
        // A lower boost does not demote.
        cell.boost_to(Priority::Normal);
        assert_eq!(cell.get(), Priority::High);
    }

    #[test]
    fn inbox_pops_highest_priority_first() {
        let inbox = PriorityInbox::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let push = |p: Priority, tag: u32| {
            let cell = Arc::new(PriorityCell::new(p));
            let order = Arc::clone(&order);
            inbox.push(cell, Box::new(move || order.lock().unwrap().push(tag)));
        };
        push(Priority::Low, 1);
        push(Priority::Critical, 2);
        push(Priority::Normal, 3);
        push(Priority::High, 4);

        while let Some(job) = inbox.pop_highest() {
            job();
        }
        assert_eq!(*order.lock().unwrap(), alloc::vec![2, 4, 3, 1]);
    }

    #[test]
    fn inbox_fifo_within_a_level() {
        let inbox = PriorityInbox::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        for tag in 0..5u32 {
            let cell = Arc::new(PriorityCell::new(Priority::Normal));
            let order = Arc::clone(&order);
            inbox.push(cell, Box::new(move || order.lock().unwrap().push(tag)));
        }
        while let Some(job) = inbox.pop_highest() {
            job();
        }
        assert_eq!(*order.lock().unwrap(), alloc::vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn boost_reorders_pending_entries() {
        let inbox = PriorityInbox::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let low_cell = Arc::new(PriorityCell::new(Priority::Low));
        {
            let order = Arc::clone(&order);
            inbox.push(
                Arc::clone(&low_cell),
                Box::new(move || order.lock().unwrap().push(1u32)),
            );
        }
        {
            let cell = Arc::new(PriorityCell::new(Priority::Normal));
            let order = Arc::clone(&order);
            inbox.push(cell, Box::new(move || order.lock().unwrap().push(2u32)));
        }
        // Inherit: boost the low entry above the normal one.
        low_cell.boost_to(Priority::High);
        while let Some(job) = inbox.pop_highest() {
            job();
        }
        assert_eq!(*order.lock().unwrap(), alloc::vec![1, 2]);
    }
}
