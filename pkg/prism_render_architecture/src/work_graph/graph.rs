//! Multi-queue aggregation over a whole work graph.
//!
//! A frame drives many work queues at once: one per GPU-generated draw stream,
//! per culling pass, per dispatch fan-out. [`WorkGraph`] owns a set of
//! [`WorkQueue`]s keyed by [`WorkQueueId`] and forwards produce/consume/drop to
//! the addressed queue, so the `CPU` has one place to reason about total
//! pressure and loss across the frame.
//!
//! Queues are stored in a [`BTreeMap`], so iteration and aggregation always walk
//! ids in ascending order. Aggregate results are therefore independent of
//! insertion order and reproduce exactly across runs, which keeps captures and
//! golden comparisons stable.
//!
//! GPU submission of the aggregated work is pending the GPU backend; this layer
//! only sums the `CPU`-side counts.

use alloc::collections::BTreeMap;

use super::indirect::{IndirectBudget, IndirectKind, IndirectTally};
use super::queue::{ProduceOutcome, QueueError, WorkQueue, WorkQueueCapacity, WorkQueueStats};
use super::ratio::Ratio;

/// Stable identity of a work queue within a [`WorkGraph`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WorkQueueId(pub u32);

/// Addressed a queue id that is not registered in the graph.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphError {
    /// No queue with this id has been registered.
    UnknownQueue(WorkQueueId),
    /// The addressed queue rejected the operation.
    Queue(WorkQueueId, QueueError),
}

/// Aggregate counters and occupancy summed across every queue in a graph.
///
/// Totals are held in `u64` and accumulate with saturating addition, so summing
/// many near-full `u32` queues cannot wrap. Ratios are exact.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AggregateStats {
    pub queue_count: u32,
    pub total_capacity: u64,
    pub total_occupancy: u64,
    pub total_produced: u64,
    pub total_consumed: u64,
    pub total_dropped: u64,
}

impl AggregateStats {
    /// Graph-wide utilization: summed occupancy over summed capacity.
    #[must_use]
    pub fn utilization(self) -> Ratio {
        Ratio::new(self.total_occupancy, self.total_capacity)
    }

    /// Graph-wide drop rate: summed dropped over summed produced.
    #[must_use]
    pub fn drop_rate(self) -> Ratio {
        Ratio::new(self.total_dropped, self.total_produced)
    }
}

/// A collection of work queues addressed by [`WorkQueueId`], plus an indirect
/// command tally shared by the whole graph.
#[derive(Clone, Debug, Default)]
pub struct WorkGraph {
    queues: BTreeMap<u32, WorkQueue>,
    indirect: IndirectTally,
}

impl WorkGraph {
    /// Creates an empty graph whose indirect tally uses `budget`.
    #[must_use]
    pub fn new(budget: IndirectBudget) -> Self {
        Self {
            queues: BTreeMap::new(),
            indirect: IndirectTally::new(budget),
        }
    }

    /// Registers (or replaces) the queue at `id` with a fresh queue of the given
    /// capacity. Returns the previous queue, if any.
    pub fn register(&mut self, id: WorkQueueId, capacity: WorkQueueCapacity) -> Option<WorkQueue> {
        self.queues.insert(id.0, WorkQueue::new(capacity))
    }

    /// Number of registered queues.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queues.len()
    }

    /// Returns `true` when no queues are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queues.is_empty()
    }

    /// The shared indirect command tally.
    #[must_use]
    pub const fn indirect(&self) -> &IndirectTally {
        &self.indirect
    }

    /// Mutable access to the shared indirect command tally.
    pub fn indirect_mut(&mut self) -> &mut IndirectTally {
        &mut self.indirect
    }

    /// Immutable view of a registered queue.
    #[must_use]
    pub fn queue(&self, id: WorkQueueId) -> Option<&WorkQueue> {
        self.queues.get(&id.0)
    }

    /// Produces `count` items into the addressed queue.
    pub fn produce(&mut self, id: WorkQueueId, count: u32) -> Result<ProduceOutcome, GraphError> {
        let queue = self
            .queues
            .get_mut(&id.0)
            .ok_or(GraphError::UnknownQueue(id))?;
        queue
            .produce(count)
            .map_err(|err| GraphError::Queue(id, err))
    }

    /// Consumes up to `count` items from the addressed queue.
    pub fn consume(&mut self, id: WorkQueueId, count: u32) -> Result<u32, GraphError> {
        let queue = self
            .queues
            .get_mut(&id.0)
            .ok_or(GraphError::UnknownQueue(id))?;
        Ok(queue.consume(count))
    }

    /// Drops up to `count` in-flight items from the addressed queue.
    pub fn drop_in_flight(&mut self, id: WorkQueueId, count: u32) -> Result<u32, GraphError> {
        let queue = self
            .queues
            .get_mut(&id.0)
            .ok_or(GraphError::UnknownQueue(id))?;
        Ok(queue.drop_in_flight(count))
    }

    /// Reserves indirect commands on the shared tally.
    pub fn reserve_indirect(
        &mut self,
        kind: IndirectKind,
        count: u32,
    ) -> Result<super::indirect::ReserveOutcome, super::indirect::BudgetExceeded> {
        self.indirect.reserve(kind, count)
    }

    /// Sums every registered queue into an [`AggregateStats`], walking ids in
    /// ascending order for determinism.
    #[must_use]
    pub fn aggregate(&self) -> AggregateStats {
        let mut agg = AggregateStats {
            queue_count: self.queues.len() as u32,
            ..AggregateStats::default()
        };
        for queue in self.queues.values() {
            let stats: WorkQueueStats = queue.stats();
            agg.total_capacity = agg
                .total_capacity
                .saturating_add(u64::from(queue.capacity().items));
            agg.total_occupancy = agg
                .total_occupancy
                .saturating_add(u64::from(queue.occupancy()));
            agg.total_produced = agg.total_produced.saturating_add(u64::from(stats.produced));
            agg.total_consumed = agg.total_consumed.saturating_add(u64::from(stats.consumed));
            agg.total_dropped = agg.total_dropped.saturating_add(u64::from(stats.dropped));
        }
        agg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> WorkGraph {
        let mut g = WorkGraph::new(IndirectBudget::lossy(64, 64));
        g.register(WorkQueueId(2), WorkQueueCapacity::lossy(8));
        g.register(WorkQueueId(1), WorkQueueCapacity::fatal(4));
        g
    }

    #[test]
    fn register_and_len() {
        let mut g = graph();
        assert_eq!(g.len(), 2);
        assert!(!g.is_empty());
        let prev = g.register(WorkQueueId(1), WorkQueueCapacity::lossy(16));
        assert!(prev.is_some());
        assert_eq!(g.len(), 2);
        assert_eq!(g.queue(WorkQueueId(1)).unwrap().capacity().items, 16);
    }

    #[test]
    fn unknown_queue_errors() {
        let mut g = graph();
        assert_eq!(
            g.produce(WorkQueueId(9), 1),
            Err(GraphError::UnknownQueue(WorkQueueId(9)))
        );
        assert_eq!(
            g.consume(WorkQueueId(9), 1),
            Err(GraphError::UnknownQueue(WorkQueueId(9)))
        );
    }

    #[test]
    fn fatal_queue_error_is_forwarded() {
        let mut g = graph();
        g.produce(WorkQueueId(1), 3).unwrap();
        let err = g.produce(WorkQueueId(1), 5).unwrap_err();
        assert_eq!(
            err,
            GraphError::Queue(
                WorkQueueId(1),
                QueueError::Overflow {
                    requested: 5,
                    available: 1,
                }
            )
        );
    }

    #[test]
    fn aggregate_sums_all_queues() {
        let mut g = graph();
        g.produce(WorkQueueId(1), 4).unwrap(); // fatal cap 4, full
        g.produce(WorkQueueId(2), 10).unwrap(); // lossy cap 8: 8 in, 2 dropped
        g.consume(WorkQueueId(2), 3).unwrap();
        let agg = g.aggregate();
        assert_eq!(agg.queue_count, 2);
        assert_eq!(agg.total_capacity, 12);
        assert_eq!(agg.total_occupancy, 4 + 5);
        assert_eq!(agg.total_produced, 4 + 10);
        assert_eq!(agg.total_consumed, 3);
        assert_eq!(agg.total_dropped, 2);
        assert_eq!(agg.utilization(), Ratio::new(9, 12));
        assert_eq!(agg.drop_rate(), Ratio::new(2, 14));
    }

    #[test]
    fn empty_graph_aggregate_is_zero() {
        let g = WorkGraph::new(IndirectBudget::default());
        let agg = g.aggregate();
        assert_eq!(agg, AggregateStats::default());
        assert_eq!(agg.utilization(), Ratio::ZERO);
        assert_eq!(agg.drop_rate(), Ratio::ZERO);
    }

    #[test]
    fn shared_indirect_tally() {
        let mut g = graph();
        g.reserve_indirect(IndirectKind::Draw, 40).unwrap();
        assert_eq!(g.indirect().reserved(IndirectKind::Draw), 40);
        g.indirect_mut().release(IndirectKind::Draw, 10);
        assert_eq!(g.indirect().reserved(IndirectKind::Draw), 30);
    }

    #[test]
    fn aggregate_is_order_independent() {
        // Two graphs built with queues registered in opposite order must
        // aggregate identically because iteration is by ascending id.
        let mut a = WorkGraph::new(IndirectBudget::default());
        a.register(WorkQueueId(1), WorkQueueCapacity::lossy(8));
        a.register(WorkQueueId(2), WorkQueueCapacity::lossy(8));
        let mut b = WorkGraph::new(IndirectBudget::default());
        b.register(WorkQueueId(2), WorkQueueCapacity::lossy(8));
        b.register(WorkQueueId(1), WorkQueueCapacity::lossy(8));
        for g in [&mut a, &mut b] {
            g.produce(WorkQueueId(1), 5).unwrap();
            g.produce(WorkQueueId(2), 3).unwrap();
        }
        assert_eq!(a.aggregate(), b.aggregate());
    }
}
