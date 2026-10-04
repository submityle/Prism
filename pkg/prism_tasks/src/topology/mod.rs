//! NUMA / hybrid-core topology data model and placement policy (design §24.6).
//!
//! This module layers the §24.6 pieces that the existing
//! [`numa`](crate::numa) and [`affinity`](crate::affinity) modules did not yet
//! cover onto their shared primitives ([`CoreInfo`](crate::numa::CoreInfo),
//! [`CoreClass`](crate::numa::CoreClass), [`NumaNodeId`](crate::numa::NumaNodeId)):
//!
//! - A **portable, injectable topology descriptor** ([`TopologyDescriptor`])
//!   that bundles cores, an inter-node [`NumaDistanceMatrix`], and a
//!   [`CacheTopology`]. It probes nothing; `prism_platform` injects the real
//!   values, and every query is a pure function of the stored data.
//! - **Workload-class placement** ([`WorkloadClass`], [`plan_placement`],
//!   [`plan_partitioned`]): map latency-sensitive work onto performance (P)
//!   cores, background work onto efficiency (E) cores, and throughput work
//!   across all cores, NUMA-local first, yielding a deterministic worker ->
//!   core plan plus a node-local memory suggestion per worker.
//! - **Distance-aware scheduling** ([`rank_nodes_by_distance`],
//!   [`nearest_node`], [`choose_balanced_node`]): pick the nearest NUMA node by
//!   the distance matrix, with a deterministic load-balancing tie-break.
//!
//! ## Relationship to `numa` / `affinity`
//! [`numa`](crate::numa) owns the single-penalty steal ordering and the
//! `Topology`/`detect` fallback; [`affinity`](crate::affinity) owns the actual
//! worker pinning. This module does **not** duplicate either: it adds the
//! richer distance matrix, cache hierarchy, and workload-class placement on top
//! of the same core types, and can be derived from a
//! [`Topology`](crate::numa::Topology) via
//! [`TopologyDescriptor::from_numa_topology`].
//!
//! ## Honest degradation
//! Real topology probing (NUMA distances, P/E maps, cache geometry) is the job
//! of `prism_platform`, which today exposes only a logical-core count. Until it
//! injects a full [`TopologyDescriptor`], callers build
//! [`TopologyDescriptor::uniform`] — one node, all performance cores, a
//! single-node distance matrix, no cache map — exactly the "缺失时退化为均匀池"
//! fallback §24.6 calls for. All the mechanism here (distance ranking, cache
//! closeness, class placement) is fully present and tested; it simply sees a
//! uniform machine until the platform layer describes more. No topology is ever
//! fabricated.

#![forbid(unsafe_code)]

mod cache;
mod descriptor;
mod distance;
mod placement;
mod schedule;

pub use cache::{shared_l3, CacheInfo, CacheLevel, CacheTopology};
pub use descriptor::{TopologyDescriptor, TopologyError};
pub use distance::{DistanceError, NumaDistanceMatrix, LOCAL_DISTANCE};
pub use placement::{
    plan_partitioned, plan_placement, PlacementPlan, WorkerPlacement, WorkloadClass,
};
pub use schedule::{choose_balanced_node, nearest_node, rank_nodes_by_distance};
