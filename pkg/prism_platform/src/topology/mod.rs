//! Hybrid-core / NUMA / cache topology as portable data (design §24.3).
//!
//! AAA schedulers need to know the *shape* of the machine — which logical
//! cores are performance (`P`) vs efficiency (`E`), which share an SMT core,
//! which `NUMA` node and cache domain they live in — to build sensible pinning
//! and steal-ordering policies. This module exposes that shape as a validated,
//! `#![forbid(unsafe_code)]`, pure `core`+`alloc` **data model** so the whole
//! policy layer in `prism_tasks` can be exercised on any machine without a live
//! probe.
//!
//! The split of responsibilities mirrors the rest of this crate:
//!
//! - [`CpuTopology`] is *data only*. It is built with [`TopologyBuilder`] (used
//!   by a real OS probe, by tests, or by any caller that already knows the
//!   layout) and validated once so downstream code can trust it.
//! - [`CpuTopology::detect`] reads Linux sysfs and the calling thread's allowed
//!   CPU list. Other platforms and failed probes explicitly report a uniform
//!   fallback through [`CpuTopology::source`]. Optional capacity, cache and
//!   distance data are never invented; see [`linux`] for the injectable probe.
//!
//! See [`qos`] for mapping engine lanes onto OS quality-of-service classes and
//! [`power`] for the energy / thermal hooks that let a scheduler back off under
//! pressure.

use alloc::vec::Vec;
use core::fmt;

pub mod linux;
pub mod power;
pub mod qos;

/// Why a real topology could not be obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyFallback {
    /// No native backend is implemented for this operating system.
    UnsupportedPlatform,
    /// The Linux files were unavailable, inconsistent, or changed mid-probe.
    ProbeFailed,
    /// Operating-system probing is unavailable without `std`.
    NoStd,
}

/// Provenance of the topology, independent of optional metadata availability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologySource {
    /// A coherent Linux online/allowed CPU, SMT and NUMA sysfs snapshot.
    LinuxSysfs,
    /// Caller-supplied, validated data; not a live hardware probe.
    Explicit,
    /// Uniform unknown-class data, not a claim about physical geometry.
    Fallback(TopologyFallback),
}

/// Runtime capabilities associated with a topology snapshot.
///
/// The topology data and the operations that can act on it are separate
/// contracts: a platform may report CPU/node ids while still lacking either
/// CPU affinity or NUMA memory policy support. A `false` field is an explicit
/// capability gap, not permission to emulate success.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyCapabilities {
    /// Provenance of the topology data used by the caller.
    pub topology_source: TopologySource,
    /// Whether the current build can apply a real OS CPU-affinity mask.
    pub cpu_affinity: bool,
    /// Whether the current build can request real node-local memory policy.
    pub numa_memory_binding: bool,
}

impl TopologyCapabilities {
    /// Whether worker-to-CPU binding can be attempted through the platform.
    #[must_use]
    pub const fn can_bind_workers(self) -> bool {
        self.cpu_affinity
    }

    /// Whether an allocation can be made with a real NUMA node policy.
    #[must_use]
    pub const fn can_allocate_node_local(self) -> bool {
        self.numa_memory_binding
    }
}

/// An OS-reported data or unified cache, restricted to available logical CPUs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyCache {
    /// OS cache level (not restricted to L1/L2/L3).
    pub level: u8,
    /// Capacity in bytes.
    pub size_bytes: u64,
    /// Coherency line size in bytes.
    pub line_bytes: u32,
    /// Sorted OS logical CPU ids sharing the cache.
    pub cores: Vec<u32>,
}

/// The performance class of a logical core on a hybrid (big.LITTLE / Intel
/// hybrid) machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoreKind {
    /// A high-throughput performance (`P`) core. Prefer for critical-path work.
    Performance,
    /// A low-power efficiency (`E`) core. Prefer for background work.
    Efficiency,
    /// The class is unknown (symmetric machine, or not probed yet).
    Unknown,
}

impl CoreKind {
    /// Whether this is a known performance core.
    #[must_use]
    pub const fn is_performance(self) -> bool {
        matches!(self, CoreKind::Performance)
    }

    /// Whether this is a known efficiency core.
    #[must_use]
    pub const fn is_efficiency(self) -> bool {
        matches!(self, CoreKind::Efficiency)
    }
}

/// One logical core (an OS-schedulable hardware thread) and the domains it
/// belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyCore {
    /// The OS logical-core id used by affinity APIs.
    pub logical_id: u32,
    /// The physical core this logical core belongs to; two `SMT` siblings share
    /// one `physical_id`.
    pub physical_id: u32,
    /// The `NUMA` node this core's memory is local to.
    pub numa_node: u16,
    /// The last-level-cache (`LLC`, usually `L3`) domain id this core shares.
    pub llc_domain: u16,
    /// The performance class of this core.
    pub kind: CoreKind,
}

impl TopologyCore {
    /// Construct a core descriptor.
    #[must_use]
    pub const fn new(
        logical_id: u32,
        physical_id: u32,
        numa_node: u16,
        llc_domain: u16,
        kind: CoreKind,
    ) -> Self {
        Self {
            logical_id,
            physical_id,
            numa_node,
            llc_domain,
            kind,
        }
    }
}

/// Why building a [`CpuTopology`] failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TopologyError {
    /// No cores were supplied; a topology needs at least one core.
    NoCores,
    /// Two cores share the OS logical id `id`.
    DuplicateLogicalId {
        /// The duplicated logical-core id.
        id: u32,
    },
    /// Two cores share a `physical_id` (i.e. they are SMT siblings of one
    /// physical core) but disagree about its class, NUMA node, or LLC domain —
    /// a contradiction real hardware cannot present.
    InconsistentSmtSiblings {
        /// The shared physical-core id whose siblings disagree.
        physical_id: u32,
    },
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyError::NoCores => f.write_str("topology needs at least one core"),
            TopologyError::DuplicateLogicalId { id } => {
                write!(f, "duplicate logical-core id {id}")
            }
            TopologyError::InconsistentSmtSiblings { physical_id } => write!(
                f,
                "cores sharing physical id {physical_id} disagree on kind/NUMA/LLC"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TopologyError {}

/// A validated, portable description of a machine's logical cores grouped by
/// `NUMA` node, cache domain, `SMT` sibling, and performance class.
///
/// This is pure data: it probes nothing itself. Build a real one with
/// [`CpuTopology::detect`], or an explicit one with [`TopologyBuilder`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    cores: Vec<TopologyCore>,
    numa_node_count: u16,
    source: TopologySource,
    distances: Option<Vec<Vec<u16>>>,
    caches: Vec<TopologyCache>,
}

impl CpuTopology {
    /// Best-effort topology for the current process.
    ///
    /// Linux reads online CPUs intersected with the current thread's allowed
    /// list, physical siblings and NUMA membership. Missing optional capacity,
    /// distances or caches stay unknown. Probe errors and other OSes return
    /// an explicitly marked fallback; use [`linux::probe`] for diagnostics.
    /// Without `std` the fallback has one core. This does not bind memory or
    /// pin threads, nor promise that CPU hotplug cannot occur after returning.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(all(feature = "std", any(target_os = "linux", target_os = "android")))]
        {
            linux::probe_or_fallback(
                |path| std::fs::read_to_string(path).ok(),
                logical_core_count(),
            )
        }
        #[cfg(not(all(feature = "std", any(target_os = "linux", target_os = "android"))))]
        {
            #[cfg(not(feature = "std"))]
            let reason = TopologyFallback::NoStd;
            #[cfg(feature = "std")]
            let reason = TopologyFallback::UnsupportedPlatform;
            Self::fallback(logical_core_count().max(1), reason)
        }
    }

    fn fallback(count: usize, reason: TopologyFallback) -> Self {
        let cores = (0..count)
            .map(|i| {
                let id = i as u32;
                TopologyCore::new(id, id, 0, 0, CoreKind::Unknown)
            })
            .collect();
        Self {
            cores,
            numa_node_count: 1,
            source: TopologySource::Fallback(reason),
            distances: None,
            caches: Vec::new(),
        }
    }

    /// How this topology was obtained (explicit data is not a hardware probe).
    #[must_use]
    pub fn source(&self) -> TopologySource {
        self.source
    }

    /// Return the operation capabilities for this topology snapshot.
    ///
    /// `topology_source` describes only the data provenance. CPU affinity and
    /// NUMA memory binding are queried independently because detecting a node
    /// id does not imply that either operation is available.
    #[must_use]
    pub fn capabilities(&self) -> TopologyCapabilities {
        TopologyCapabilities {
            topology_source: self.source,
            cpu_affinity: cpu_affinity_supported(),
            numa_memory_binding: numa_memory_binding_supported(),
        }
    }

    /// Distinct OS NUMA ids in ascending order. Distance rows/columns use this
    /// order, not `max(node_id) + 1`; memory-only/inaccessible nodes are omitted.
    #[must_use]
    pub fn node_ids(&self) -> Vec<u16> {
        let mut ids: Vec<_> = self.cores.iter().map(|core| core.numa_node).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// OS distance rows in [`Self::node_ids`] order, or `None` when unavailable
    /// or invalid. No synthetic remote distance is presented as measured data.
    #[must_use]
    pub fn numa_distances(&self) -> Option<&[Vec<u16>]> {
        self.distances.as_deref()
    }

    /// OS-reported caches. Missing/inconsistent cache records are omitted.
    #[must_use]
    pub fn caches(&self) -> &[TopologyCache] {
        &self.caches
    }

    /// All logical cores, in the order they were supplied.
    #[must_use]
    pub fn cores(&self) -> &[TopologyCore] {
        &self.cores
    }

    /// The number of logical cores.
    #[must_use]
    pub fn logical_core_count(&self) -> usize {
        self.cores.len()
    }

    /// The number of distinct physical cores (counting `SMT` siblings once).
    #[must_use]
    pub fn physical_core_count(&self) -> usize {
        let mut ids: Vec<u32> = self.cores.iter().map(|c| c.physical_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// The number of `NUMA` nodes.
    #[must_use]
    pub fn numa_node_count(&self) -> u16 {
        self.numa_node_count
    }

    /// Whether geometry was probed or explicitly supplied, rather than a
    /// uniform fallback. Optional class/cache/distance information may still
    /// be unknown. Use [`Self::source`] to distinguish live from explicit data.
    #[must_use]
    pub fn is_probed(&self) -> bool {
        !matches!(self.source, TopologySource::Fallback(_))
    }

    /// Whether this machine mixes performance and efficiency cores.
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        let has_p = self.cores.iter().any(|c| c.kind.is_performance());
        let has_e = self.cores.iter().any(|c| c.kind.is_efficiency());
        has_p && has_e
    }

    /// Iterate over the cores of a given performance class.
    pub fn cores_of_kind(&self, kind: CoreKind) -> impl Iterator<Item = &TopologyCore> {
        self.cores.iter().filter(move |c| c.kind == kind)
    }

    /// Iterate over the cores local to a given `NUMA` node.
    pub fn cores_on_node(&self, node: u16) -> impl Iterator<Item = &TopologyCore> {
        self.cores.iter().filter(move |c| c.numa_node == node)
    }

    /// Whether a logical core has at least one `SMT` sibling (another logical
    /// core sharing its physical core).
    #[must_use]
    pub fn has_smt_sibling(&self, logical_id: u32) -> bool {
        let Some(target) = self.cores.iter().find(|c| c.logical_id == logical_id) else {
            return false;
        };
        self.cores
            .iter()
            .filter(|c| c.physical_id == target.physical_id)
            .count()
            > 1
    }
}

fn cpu_affinity_supported() -> bool {
    #[cfg(feature = "std")]
    {
        crate::thread::affinity_supported()
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

fn numa_memory_binding_supported() -> bool {
    #[cfg(feature = "std")]
    {
        crate::vm::NumaAllocation::supported()
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

#[cfg(feature = "std")]
fn logical_core_count() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
}

#[cfg(not(feature = "std"))]
fn logical_core_count() -> usize {
    1
}

/// Builder for an explicit [`CpuTopology`] — used by a real OS probe, by tests,
/// or by any caller that already knows the machine shape.
#[derive(Clone, Debug, Default)]
pub struct TopologyBuilder {
    cores: Vec<TopologyCore>,
}

impl TopologyBuilder {
    /// Start an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self { cores: Vec::new() }
    }

    /// Add a logical core.
    #[must_use]
    pub fn core(mut self, core: TopologyCore) -> Self {
        self.cores.push(core);
        self
    }

    /// Add many logical cores.
    #[must_use]
    pub fn with_cores<I: IntoIterator<Item = TopologyCore>>(mut self, cores: I) -> Self {
        self.cores.extend(cores);
        self
    }

    /// Validate and finish. The resulting topology reports
    /// [`CpuTopology::is_probed`] as `true` because its fields were supplied
    /// explicitly rather than defaulted.
    ///
    /// # Errors
    ///
    /// Returns [`TopologyError::NoCores`] when empty, or
    /// [`TopologyError::DuplicateLogicalId`] when two cores share a logical id.
    pub fn build(self) -> Result<CpuTopology, TopologyError> {
        if self.cores.is_empty() {
            return Err(TopologyError::NoCores);
        }
        let mut seen: Vec<u32> = self.cores.iter().map(|c| c.logical_id).collect();
        seen.sort_unstable();
        for pair in seen.windows(2) {
            if pair[0] == pair[1] {
                return Err(TopologyError::DuplicateLogicalId { id: pair[0] });
            }
        }
        // Cores sharing a physical id are SMT siblings of one physical core, so
        // they must agree on its class, NUMA node, and LLC domain. A
        // contradiction would hand the scheduler impossible NUMA/affinity input
        // (CR-found).
        for (i, core) in self.cores.iter().enumerate() {
            let disagrees = self.cores[i + 1..].iter().any(|other| {
                other.physical_id == core.physical_id
                    && (other.kind != core.kind
                        || other.numa_node != core.numa_node
                        || other.llc_domain != core.llc_domain)
            });
            if disagrees {
                return Err(TopologyError::InconsistentSmtSiblings {
                    physical_id: core.physical_id,
                });
            }
        }
        // Count the *distinct* NUMA node ids: `max + 1` over-counted a sparse
        // node set and overflowed for `u16::MAX` (CR-found).
        let mut node_ids: Vec<u16> = self.cores.iter().map(|c| c.numa_node).collect();
        node_ids.sort_unstable();
        node_ids.dedup();
        let numa_node_count = node_ids.len().min(u16::MAX as usize) as u16;
        Ok(CpuTopology {
            cores: self.cores,
            numa_node_count,
            source: TopologySource::Explicit,
            distances: None,
            caches: Vec::new(),
        })
    }
}
