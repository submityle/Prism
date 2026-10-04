//! The portable, injectable machine-topology descriptor (design §24.6).
//!
//! [`TopologyDescriptor`] is the single data model the scheduler reasons about:
//! a list of logical cores (with NUMA node and performance class), an inter-node
//! [`NumaDistanceMatrix`], and a [`CacheTopology`]. It is deliberately *data
//! only* — it probes nothing. On a real machine `prism_platform` would populate
//! it from the OS; this crate consumes it through the pure placement and
//! scheduling functions so that none of that logic needs a live machine to be
//! tested.
//!
//! This complements the existing [`Topology`](crate::numa::Topology) (which
//! carries cores + node count for steal ordering) by adding the §24.6 pieces
//! the scheduler was missing: the distance matrix and cache hierarchy. A
//! descriptor can be derived from a [`Topology`] with
//! [`TopologyDescriptor::from_numa_topology`] (uniform distances, no cache map)
//! for callers that only have the simpler model.

use alloc::vec::Vec;
use core::fmt;

use crate::numa::{CoreClass, CoreInfo, NumaNodeId, Topology};

use super::cache::CacheTopology;
use super::distance::NumaDistanceMatrix;

/// A validated, portable description of a machine's cores, NUMA distances, and
/// cache hierarchy.
#[derive(Clone, Debug)]
pub struct TopologyDescriptor {
    cores: Vec<CoreInfo>,
    node_count: usize,
    distances: NumaDistanceMatrix,
    caches: CacheTopology,
}

/// Why building a [`TopologyDescriptor`] failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TopologyError {
    /// No cores were supplied; a topology needs at least one core.
    NoCores,
    /// Two cores share the OS id `id`.
    DuplicateCoreId {
        /// The duplicated logical-core id.
        id: usize,
    },
    /// The distance matrix describes `matrix_nodes` nodes but the cores span
    /// `cores_nodes` nodes.
    DistanceNodeMismatch {
        /// Node count derived from the core list (`max(node) + 1`).
        cores_nodes: usize,
        /// Node count of the supplied distance matrix.
        matrix_nodes: usize,
    },
    /// A cache instance references core id `id`, which is not in the core list.
    CacheUnknownCore {
        /// The unknown logical-core id referenced by a cache.
        id: usize,
    },
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyError::NoCores => f.write_str("topology needs at least one core"),
            TopologyError::DuplicateCoreId { id } => write!(f, "duplicate core id {id}"),
            TopologyError::DistanceNodeMismatch {
                cores_nodes,
                matrix_nodes,
            } => write!(
                f,
                "distance matrix has {matrix_nodes} nodes but cores span {cores_nodes} nodes"
            ),
            TopologyError::CacheUnknownCore { id } => {
                write!(f, "cache references unknown core id {id}")
            }
        }
    }
}

fn derive_node_count(cores: &[CoreInfo]) -> usize {
    cores
        .iter()
        .map(|c| c.node.index() as usize + 1)
        .max()
        .unwrap_or(1)
}

impl TopologyDescriptor {
    /// A uniform descriptor: `cores` performance cores (ids `0..cores`) on a
    /// single NUMA node, a single-node distance matrix, and no cache map. The
    /// honest non-NUMA / non-hybrid fallback.
    pub fn uniform(cores: usize) -> Self {
        let cores = cores.max(1);
        let cores: Vec<CoreInfo> = (0..cores)
            .map(|id| CoreInfo {
                id,
                node: NumaNodeId::ZERO,
                class: CoreClass::Performance,
            })
            .collect();
        Self {
            cores,
            node_count: 1,
            distances: NumaDistanceMatrix::single(),
            caches: CacheTopology::empty(),
        }
    }

    /// Build and validate a descriptor from its parts.
    ///
    /// # Errors
    /// Returns [`TopologyError`] when the core list is empty, has duplicate
    /// ids, the distance matrix node count disagrees with the cores, or a cache
    /// references an unknown core id.
    pub fn new(
        cores: Vec<CoreInfo>,
        distances: NumaDistanceMatrix,
        caches: CacheTopology,
    ) -> Result<Self, TopologyError> {
        if cores.is_empty() {
            return Err(TopologyError::NoCores);
        }
        // Reject duplicate ids (deterministic: report the first duplicate by
        // ascending id).
        let mut ids: Vec<usize> = cores.iter().map(|c| c.id).collect();
        ids.sort_unstable();
        for pair in ids.windows(2) {
            if pair[0] == pair[1] {
                return Err(TopologyError::DuplicateCoreId { id: pair[0] });
            }
        }
        let node_count = derive_node_count(&cores);
        if node_count != distances.node_count() {
            return Err(TopologyError::DistanceNodeMismatch {
                cores_nodes: node_count,
                matrix_nodes: distances.node_count(),
            });
        }
        for cache in caches.caches() {
            for &id in &cache.cores {
                if !cores.iter().any(|c| c.id == id) {
                    return Err(TopologyError::CacheUnknownCore { id });
                }
            }
        }
        Ok(Self {
            cores,
            node_count,
            distances,
            caches,
        })
    }

    /// Derive a descriptor from the simpler [`Topology`] model: the same cores,
    /// a uniform distance matrix over its node count, and no cache map. The
    /// bridge for callers that only have a [`Topology`].
    pub fn from_numa_topology(topology: &Topology, remote_distance: u16) -> Self {
        let cores = topology.cores().to_vec();
        let node_count = topology.node_count();
        let distances = if node_count <= 1 {
            NumaDistanceMatrix::single()
        } else {
            NumaDistanceMatrix::uniform(node_count, remote_distance)
        };
        Self {
            cores,
            node_count,
            distances,
            caches: CacheTopology::empty(),
        }
    }

    /// All logical cores, in the order supplied.
    pub fn cores(&self) -> &[CoreInfo] {
        &self.cores
    }

    /// Number of logical cores.
    pub fn core_count(&self) -> usize {
        self.cores.len()
    }

    /// Number of distinct NUMA nodes (`1` on non-NUMA systems).
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// The inter-node distance matrix.
    pub fn distances(&self) -> &NumaDistanceMatrix {
        &self.distances
    }

    /// The cache hierarchy.
    pub fn caches(&self) -> &CacheTopology {
        &self.caches
    }

    /// Look up a core by its OS id.
    pub fn core(&self, id: usize) -> Option<CoreInfo> {
        self.cores.iter().copied().find(|c| c.id == id)
    }

    /// The NUMA node a core id resides on, if known.
    pub fn node_of_core(&self, id: usize) -> Option<NumaNodeId> {
        self.core(id).map(|c| c.node)
    }

    /// Iterate the cores on a given NUMA node.
    pub fn cores_on_node(&self, node: NumaNodeId) -> impl Iterator<Item = CoreInfo> + '_ {
        self.cores.iter().copied().filter(move |c| c.node == node)
    }

    /// Iterate the cores of a given performance class.
    pub fn cores_of_class(&self, class: CoreClass) -> impl Iterator<Item = CoreInfo> + '_ {
        self.cores.iter().copied().filter(move |c| c.class == class)
    }

    /// Number of performance (P) cores.
    pub fn performance_core_count(&self) -> usize {
        self.cores_of_class(CoreClass::Performance).count()
    }

    /// Number of efficiency (E) cores.
    pub fn efficiency_core_count(&self) -> usize {
        self.cores_of_class(CoreClass::Efficiency).count()
    }

    /// Whether this descriptor distinguishes performance and efficiency cores
    /// (i.e. it is a hybrid / big.LITTLE topology).
    pub fn is_hybrid(&self) -> bool {
        self.performance_core_count() > 0 && self.efficiency_core_count() > 0
    }
}
