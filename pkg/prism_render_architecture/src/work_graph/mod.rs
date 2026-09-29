//! GPU-generated draw and dispatch work.
//!
//! Modern GPU-driven rendering lets the GPU decide what to draw: a compute pass
//! culls and packs work into indirect buffers that later draw/dispatch commands
//! consume. The `CPU` no longer knows the exact counts, but it still owns the
//! buffers, the caps, and the pressure budget. This module is the deterministic,
//! `GPU`-independent accounting contract for that world.
//!
//! * [`ratio`] — exact integer [`Ratio`]s for utilization and drop-rate
//!   reporting, avoiding floating-point equality traps.
//! * [`queue`] — a single [`WorkQueue`]'s capacity/occupancy bookkeeping with
//!   saturating counters and lossy-or-fatal overflow.
//! * [`indirect`] — per-stream [`IndirectBudget`] and [`IndirectTally`] for
//!   counting GPU-generated draw and dispatch commands against their caps.
//! * [`graph`] — the multi-queue [`WorkGraph`] owner and its
//!   [`AggregateStats`].
//!
//! Every counter saturates rather than wraps, every ratio is exact, and every
//! aggregation walks queue ids in a fixed order, so results reproduce across
//! runs. Enqueueing records into GPU-visible indirect buffers and submitting the
//! resulting commands are pending the GPU backend; this layer tracks only the
//! `CPU`-side counts that let higher layers reason about pressure and loss.

pub mod graph;
pub mod indirect;
pub mod queue;
pub mod ratio;

pub use graph::{AggregateStats, GraphError, WorkGraph, WorkQueueId};
pub use indirect::{BudgetExceeded, IndirectBudget, IndirectKind, IndirectTally, ReserveOutcome};
pub use queue::{ProduceOutcome, QueueError, WorkQueue, WorkQueueCapacity, WorkQueueStats};
pub use ratio::Ratio;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_frame_accounting() {
        // A tiny frame: two queues plus an indirect budget, driven through a
        // produce/consume cycle, then aggregated.
        let mut graph = WorkGraph::new(IndirectBudget::lossy(128, 32));
        graph.register(WorkQueueId(0), WorkQueueCapacity::lossy(64));
        graph.register(WorkQueueId(1), WorkQueueCapacity::fatal(16));

        // GPU-driven culling reserves indirect draws, then produces records.
        graph.reserve_indirect(IndirectKind::Draw, 50).unwrap();
        graph.produce(WorkQueueId(0), 80).unwrap(); // 64 fit, 16 dropped
        graph.produce(WorkQueueId(1), 16).unwrap(); // exact fit
        graph.consume(WorkQueueId(0), 40).unwrap();

        let agg = graph.aggregate();
        assert_eq!(agg.queue_count, 2);
        assert_eq!(agg.total_capacity, 80);
        assert_eq!(agg.total_occupancy, (64 - 40) + 16);
        assert_eq!(agg.total_dropped, 16);
        assert_eq!(agg.total_produced, 80 + 16);
        assert_eq!(graph.indirect().reserved(IndirectKind::Draw), 50);
        assert!(agg.utilization() < Ratio::ONE);
    }

    #[test]
    fn public_type_paths_are_stable() {
        // The original stub exposed these three types at the module root; keep
        // those paths resolvable for dependents.
        let _id: WorkQueueId = WorkQueueId(0);
        let _cap: WorkQueueCapacity = WorkQueueCapacity::default();
        let _stats: WorkQueueStats = WorkQueueStats::default();
    }

    #[test]
    fn determinism_of_full_pipeline() {
        let run = || {
            let mut graph = WorkGraph::new(IndirectBudget::lossy(64, 64));
            graph.register(WorkQueueId(3), WorkQueueCapacity::lossy(10));
            graph.register(WorkQueueId(1), WorkQueueCapacity::lossy(10));
            graph.produce(WorkQueueId(1), 7).unwrap();
            graph.produce(WorkQueueId(3), 15).unwrap();
            graph.reserve_indirect(IndirectKind::Dispatch, 100).unwrap();
            graph.aggregate()
        };
        assert_eq!(run(), run());
    }
}
