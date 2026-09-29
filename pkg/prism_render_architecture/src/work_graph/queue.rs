//! Single work-queue capacity and occupancy accounting.
//!
//! A work queue is a fixed-capacity ring of GPU-generated draw or dispatch
//! records. This module owns the `CPU`-side bookkeeping for one such queue: how
//! many items are in flight (occupancy), and cumulative lifetime counters for
//! items produced, consumed, and dropped.
//!
//! The accounting is deterministic and overflow-safe:
//!
//! * Occupancy is bounded by [`WorkQueueCapacity::items`]; a produce that would
//!   exceed it either drops the surplus or, when the queue is marked fatal on
//!   overflow, fails with [`QueueError::Overflow`] without mutating state.
//! * Cumulative counters use saturating addition, so a long-running session can
//!   never wrap a `u32` counter and silently under-report.
//!
//! Actual enqueue of records into a GPU-visible indirect buffer and submission
//! of the resulting commands are pending the GPU backend; this layer tracks
//! only the counts that let the `CPU` reason about pressure and loss.

use super::ratio::Ratio;

/// How large a work queue is and how it reacts to overflow.
///
/// `items` is the maximum simultaneous occupancy. When `overflow_is_fatal` is
/// set, a produce that cannot fit is rejected wholesale; otherwise the surplus
/// is dropped and counted.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WorkQueueCapacity {
    pub items: u32,
    pub overflow_is_fatal: bool,
}

impl WorkQueueCapacity {
    /// A non-fatal queue that drops surplus work beyond `items`.
    #[must_use]
    pub const fn lossy(items: u32) -> Self {
        Self {
            items,
            overflow_is_fatal: false,
        }
    }

    /// A queue that treats any overflow as a fatal accounting error.
    #[must_use]
    pub const fn fatal(items: u32) -> Self {
        Self {
            items,
            overflow_is_fatal: true,
        }
    }
}

/// Cumulative lifetime counters for one queue.
///
/// `produced` counts every item a producer offered to the queue, including
/// those later dropped for lack of room, so `dropped <= produced` always holds.
/// All three counters are monotonically non-decreasing and saturate at
/// [`u32::MAX`] rather than wrapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WorkQueueStats {
    pub produced: u32,
    pub consumed: u32,
    pub dropped: u32,
}

impl WorkQueueStats {
    /// Drop rate as an exact ratio of dropped over produced.
    ///
    /// Yields [`Ratio::ZERO`] before anything has been produced.
    #[must_use]
    pub fn drop_rate(self) -> Ratio {
        Ratio::new(u64::from(self.dropped), u64::from(self.produced))
    }
}

/// Why a queue operation could not be completed as requested.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QueueError {
    /// A produce would exceed capacity on a queue whose
    /// [`WorkQueueCapacity::overflow_is_fatal`] flag is set. No state changed.
    Overflow {
        /// Items requested by the failing produce.
        requested: u32,
        /// Items that could have been accepted before overflow.
        available: u32,
    },
}

/// Outcome of a successful [`WorkQueue::produce`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ProduceOutcome {
    /// Items admitted into the queue.
    pub accepted: u32,
    /// Items discarded because they did not fit (lossy queues only).
    pub dropped: u32,
}

/// Fixed-capacity accounting for a single GPU work queue.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct WorkQueue {
    capacity: WorkQueueCapacity,
    occupancy: u32,
    stats: WorkQueueStats,
}

impl WorkQueue {
    /// Creates an empty queue with the given capacity.
    #[must_use]
    pub const fn new(capacity: WorkQueueCapacity) -> Self {
        Self {
            capacity,
            occupancy: 0,
            stats: WorkQueueStats {
                produced: 0,
                consumed: 0,
                dropped: 0,
            },
        }
    }

    /// The queue's capacity configuration.
    #[must_use]
    pub const fn capacity(self) -> WorkQueueCapacity {
        self.capacity
    }

    /// Current in-flight occupancy: produced items not yet consumed or dropped.
    #[must_use]
    pub const fn occupancy(self) -> u32 {
        self.occupancy
    }

    /// Free slots remaining before the queue is full.
    #[must_use]
    pub const fn free(self) -> u32 {
        self.capacity.items - self.occupancy
    }

    /// Snapshot of the cumulative counters.
    #[must_use]
    pub const fn stats(self) -> WorkQueueStats {
        self.stats
    }

    /// Returns `true` when occupancy has reached capacity.
    #[must_use]
    pub const fn is_full(self) -> bool {
        self.occupancy >= self.capacity.items
    }

    /// Instantaneous utilization: occupancy over capacity, as an exact ratio.
    #[must_use]
    pub fn utilization(self) -> Ratio {
        Ratio::new(u64::from(self.occupancy), u64::from(self.capacity.items))
    }

    /// Produces `count` items into the queue.
    ///
    /// On a lossy queue the surplus beyond free capacity is dropped and counted;
    /// on a fatal queue an overflow leaves state untouched and returns
    /// [`QueueError::Overflow`]. Produced and dropped counters saturate.
    pub fn produce(&mut self, count: u32) -> Result<ProduceOutcome, QueueError> {
        let available = self.free();
        if count <= available {
            self.occupancy += count;
            self.stats.produced = self.stats.produced.saturating_add(count);
            return Ok(ProduceOutcome {
                accepted: count,
                dropped: 0,
            });
        }

        if self.capacity.overflow_is_fatal {
            return Err(QueueError::Overflow {
                requested: count,
                available,
            });
        }

        let dropped = count - available;
        self.occupancy += available;
        self.stats.produced = self.stats.produced.saturating_add(count);
        self.stats.dropped = self.stats.dropped.saturating_add(dropped);
        Ok(ProduceOutcome {
            accepted: available,
            dropped,
        })
    }

    /// Consumes up to `count` in-flight items, returning the number consumed.
    ///
    /// Consuming more than the occupancy simply drains the queue; the consumed
    /// counter saturates.
    pub fn consume(&mut self, count: u32) -> u32 {
        let consumed = count.min(self.occupancy);
        self.occupancy -= consumed;
        self.stats.consumed = self.stats.consumed.saturating_add(consumed);
        consumed
    }

    /// Explicitly drops up to `count` in-flight items, returning the number
    /// dropped. Used to model records discarded by culling after enqueue.
    pub fn drop_in_flight(&mut self, count: u32) -> u32 {
        let dropped = count.min(self.occupancy);
        self.occupancy -= dropped;
        self.stats.dropped = self.stats.dropped.saturating_add(dropped);
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produce_within_capacity() {
        let mut q = WorkQueue::new(WorkQueueCapacity::lossy(8));
        let out = q.produce(5).unwrap();
        assert_eq!(
            out,
            ProduceOutcome {
                accepted: 5,
                dropped: 0
            }
        );
        assert_eq!(q.occupancy(), 5);
        assert_eq!(q.free(), 3);
        assert_eq!(q.stats().produced, 5);
        assert!(!q.is_full());
    }

    #[test]
    fn lossy_overflow_drops_surplus() {
        let mut q = WorkQueue::new(WorkQueueCapacity::lossy(4));
        let out = q.produce(10).unwrap();
        assert_eq!(
            out,
            ProduceOutcome {
                accepted: 4,
                dropped: 6
            }
        );
        assert!(q.is_full());
        assert_eq!(q.stats().produced, 10);
        assert_eq!(q.stats().dropped, 6);
        assert_eq!(q.free(), 0);
    }

    #[test]
    fn fatal_overflow_returns_err_without_mutation() {
        let mut q = WorkQueue::new(WorkQueueCapacity::fatal(4));
        q.produce(3).unwrap();
        let before = q;
        let err = q.produce(5).unwrap_err();
        assert_eq!(
            err,
            QueueError::Overflow {
                requested: 5,
                available: 1
            }
        );
        assert_eq!(q, before, "fatal overflow must not mutate state");
    }

    #[test]
    fn fatal_exact_fit_succeeds() {
        let mut q = WorkQueue::new(WorkQueueCapacity::fatal(4));
        assert!(q.produce(4).is_ok());
        assert!(q.is_full());
        assert_eq!(q.produce(0), Ok(ProduceOutcome::default()));
    }

    #[test]
    fn consume_and_drop_reduce_occupancy() {
        let mut q = WorkQueue::new(WorkQueueCapacity::lossy(8));
        q.produce(6).unwrap();
        assert_eq!(q.consume(2), 2);
        assert_eq!(q.drop_in_flight(3), 3);
        assert_eq!(q.occupancy(), 1);
        assert_eq!(q.stats().consumed, 2);
        assert_eq!(q.stats().dropped, 3);
        // Draining more than occupancy is clamped.
        assert_eq!(q.consume(99), 1);
        assert_eq!(q.occupancy(), 0);
    }

    #[test]
    fn zero_capacity_drops_everything() {
        let mut q = WorkQueue::new(WorkQueueCapacity::lossy(0));
        let out = q.produce(7).unwrap();
        assert_eq!(
            out,
            ProduceOutcome {
                accepted: 0,
                dropped: 7
            }
        );
        assert!(q.is_full());
        assert_eq!(q.utilization(), Ratio::ZERO);
    }

    #[test]
    fn counters_saturate() {
        let mut q = WorkQueue::new(WorkQueueCapacity::lossy(u32::MAX));
        q.stats.produced = u32::MAX - 1;
        q.produce(10).unwrap();
        assert_eq!(q.stats().produced, u32::MAX);
    }

    #[test]
    fn ratios_are_exact() {
        let mut q = WorkQueue::new(WorkQueueCapacity::lossy(4));
        q.produce(3).unwrap();
        assert_eq!(q.utilization(), Ratio::new(3, 4));
        q.produce(10).unwrap(); // drops 9 (1 fit, so accepted 1, dropped 9)
        assert_eq!(q.stats().drop_rate(), Ratio::new(9, 13));
    }

    #[test]
    fn determinism_same_sequence_same_state() {
        let run = || {
            let mut q = WorkQueue::new(WorkQueueCapacity::lossy(5));
            q.produce(3).unwrap();
            q.consume(1);
            q.produce(4).unwrap();
            q.drop_in_flight(2);
            q
        };
        assert_eq!(run(), run());
    }
}
