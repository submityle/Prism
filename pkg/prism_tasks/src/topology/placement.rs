//! Workload-class-driven worker placement (design §24.6).
//!
//! Given a [`TopologyDescriptor`] and a [`WorkloadClass`], this module computes
//! a deterministic worker -> core assignment plus a NUMA-local memory-affinity
//! suggestion for each worker. The policy mirrors design §24.6:
//!
//! - [`WorkloadClass::LatencySensitive`] (the `Critical` lane) prefers
//!   performance (P) cores so latency-critical work runs on the fast cores.
//! - [`WorkloadClass::Background`] prefers efficiency (E) cores so background
//!   work stays off the P-cores and saves power on mobile parts.
//! - [`WorkloadClass::Throughput`] uses all cores, P-cores first, to saturate
//!   the machine for bulk parallel work.
//!
//! Within a class the cores are ordered P/E by preference, then **NUMA-local
//! first** (lower node index first) so that workers pack onto one node before
//! spilling to the next, keeping each worker's memory node-local. The suggested
//! memory node for a worker is simply the NUMA node of the core it landed on.
//!
//! Every function here is a pure, deterministic function of the descriptor and
//! the requested worker count — no clock, no threads — so a given machine +
//! request always yields the same plan, which is exactly what reproducible
//! frame scheduling needs.

use alloc::vec::Vec;

use crate::numa::{CoreClass, CoreInfo, NumaNodeId};

use super::descriptor::TopologyDescriptor;

/// The scheduling intent of a body of work, which drives P/E core preference.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WorkloadClass {
    /// Latency-critical work (input, simulation step, audio): prefer P-cores.
    LatencySensitive,
    /// Bulk parallel work that wants maximum throughput: use all cores,
    /// P-cores first.
    Throughput,
    /// Background / deferrable work (streaming, bakes): prefer E-cores to keep
    /// P-cores free and save power.
    Background,
}

impl WorkloadClass {
    /// The core-class preference order for this workload: the class to fill
    /// first, then the spill-over class. All three workloads can still run on
    /// either kind of core; this is only the *preferred* ordering.
    pub fn core_class_preference(self) -> [CoreClass; 2] {
        match self {
            WorkloadClass::LatencySensitive | WorkloadClass::Throughput => {
                [CoreClass::Performance, CoreClass::Efficiency]
            }
            WorkloadClass::Background => [CoreClass::Efficiency, CoreClass::Performance],
        }
    }

    #[inline]
    fn class_rank(self, class: CoreClass) -> u8 {
        let pref = self.core_class_preference();
        if class == pref[0] {
            0
        } else {
            1
        }
    }
}

/// Where a single worker was placed, with its NUMA-local memory suggestion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WorkerPlacement {
    /// Worker index within the pool.
    pub worker: usize,
    /// OS logical-core id the worker should pin to / run on.
    pub core: usize,
    /// NUMA node that core lives on.
    pub node: NumaNodeId,
    /// Performance class of the assigned core.
    pub class: CoreClass,
    /// Suggested NUMA node for this worker's node-local allocations (its
    /// arena/stacks). Equals [`WorkerPlacement::node`]; named separately so the
    /// memory-affinity intent is explicit at call sites.
    pub memory_node: NumaNodeId,
}

/// A deterministic worker -> core placement for one workload class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementPlan {
    placements: Vec<WorkerPlacement>,
}

impl PlacementPlan {
    /// Number of workers placed.
    pub fn len(&self) -> usize {
        self.placements.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// The placement for `worker`, if any.
    pub fn placement(&self, worker: usize) -> Option<WorkerPlacement> {
        self.placements.get(worker).copied()
    }

    /// All placements in worker order.
    pub fn placements(&self) -> &[WorkerPlacement] {
        &self.placements
    }

    /// Per-worker NUMA node vector (indexed by worker), as consumed by
    /// [`crate::numa::steal_order`].
    pub fn worker_nodes(&self) -> Vec<NumaNodeId> {
        self.placements.iter().map(|p| p.node).collect()
    }

    /// Per-worker suggested memory node vector (indexed by worker).
    pub fn memory_nodes(&self) -> Vec<NumaNodeId> {
        self.placements.iter().map(|p| p.memory_node).collect()
    }
}

/// Order `cores` for `class`: by core-class preference, then NUMA-local first
/// (ascending node), then by core id. Deterministic and total.
fn ordered_cores(cores: &[CoreInfo], class: WorkloadClass) -> Vec<CoreInfo> {
    let mut ordered = cores.to_vec();
    ordered.sort_by_key(|c| (class.class_rank(c.class), c.node.index(), c.id));
    ordered
}

fn plan_over(cores: &[CoreInfo], workers: usize) -> PlacementPlan {
    // `ordered_cores` never returns empty for a validated descriptor (it always
    // has >= 1 core), but guard anyway so this stays total.
    if cores.is_empty() || workers == 0 {
        return PlacementPlan {
            placements: Vec::new(),
        };
    }
    let placements = (0..workers)
        .map(|worker| {
            let core = cores[worker % cores.len()];
            WorkerPlacement {
                worker,
                core: core.id,
                node: core.node,
                class: core.class,
                memory_node: core.node,
            }
        })
        .collect();
    PlacementPlan { placements }
}

/// Compute a deterministic placement of `workers` workers for `class` over
/// `topology`.
///
/// Cores are ranked by the workload's P/E preference, then NUMA-local-first,
/// then by id; workers are assigned round-robin over that ranking. When there
/// are more workers than cores, several workers share a core (the honest
/// oversubscribed case). Each worker's `memory_node` is the NUMA node of its
/// assigned core.
pub fn plan_placement(
    topology: &TopologyDescriptor,
    class: WorkloadClass,
    workers: usize,
) -> PlacementPlan {
    let ordered = ordered_cores(topology.cores(), class);
    plan_over(&ordered, workers)
}

/// Partition the machine between a latency-sensitive pool and a background pool
/// along the P/E boundary (design §24.6: `Critical` -> P-cores,
/// `Background` -> E-cores), returning one plan each.
///
/// On a hybrid part the two plans use **disjoint** cores: the latency plan is
/// placed over the P-cores and the background plan over the E-cores. On a
/// homogeneous part (no P-cores, or no E-cores) the class whose preferred cores
/// are absent spills onto the available cores, which may then overlap the other
/// plan — the honest degenerate case flagged in the module/design docs.
pub fn plan_partitioned(
    topology: &TopologyDescriptor,
    latency_workers: usize,
    background_workers: usize,
) -> (PlacementPlan, PlacementPlan) {
    let p_cores: Vec<CoreInfo> = topology.cores_of_class(CoreClass::Performance).collect();
    let e_cores: Vec<CoreInfo> = topology.cores_of_class(CoreClass::Efficiency).collect();

    // Latency work goes on P-cores; fall back to E-cores if there are none.
    let latency_pool = if p_cores.is_empty() { &e_cores } else { &p_cores };
    // Background work goes on E-cores; fall back to P-cores if there are none.
    let background_pool = if e_cores.is_empty() {
        &p_cores
    } else {
        &e_cores
    };

    let latency = plan_over(
        &ordered_cores(latency_pool, WorkloadClass::LatencySensitive),
        latency_workers,
    );
    let background = plan_over(
        &ordered_cores(background_pool, WorkloadClass::Background),
        background_workers,
    );
    (latency, background)
}
