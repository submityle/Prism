//! NUMA / hybrid-core topology data model and placement policy (design §24.6).
//!
//! This module layers the §24.6 pieces that the existing
//! [`numa`](crate::numa) and [`affinity`](crate::affinity) modules did not yet
//! cover onto their shared primitives ([`CoreInfo`](crate::numa::CoreInfo),
//! [`CoreClass`](crate::numa::CoreClass), [`NumaNodeId`](crate::numa::NumaNodeId)):
//!
//! - A **portable, injectable topology descriptor** ([`TopologyDescriptor`])
//!   that bundles cores, an inter-node [`NumaDistanceMatrix`], and a
//!   [`CacheTopology`]. Detection delegates to `prism_platform`; every policy
//!   query is a pure function of the resulting data.
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
//! Linux platform detection retains allowed CPU ids, NUMA/SMT geometry, and
//! optional capacity/cache/distance evidence. Unsupported/failed probes are
//! explicitly marked by [`TopologyDescriptor::platform_source`]. Unknown core
//! classes use the ordinary scheduling lane, and unavailable distances use
//! synthetic policy costs (see [`TopologyDescriptor::has_platform_distances`]);
//! neither is presented as measured hardware. Placement's memory-node output
//! remains a suggestion, not an OS memory binding.

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
