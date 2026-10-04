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
//! - [`CpuTopology::detect`] returns a **best-effort** topology for the current
//!   process. Under `std` it fills in the logical-core count and architecture;
//!   it does **not** claim to know the `P`/`E` split, `SMT` siblings, `NUMA`
//!   geometry, or cache domains, because those require per-OS syscalls that are
//!   not wired yet. [`CpuTopology::is_probed`] reports whether the richer fields
//!   are real (`true`) or the uniform fallback (`false`), so callers never
//!   mistake "one flat node of unknown cores" for a genuine single-socket read.
//!
//! See [`qos`] for mapping engine lanes onto OS quality-of-service classes and
//! [`power`] for the energy / thermal hooks that let a scheduler back off under
//! pressure.

use alloc::vec::Vec;
use core::fmt;

pub mod power;
pub mod qos;

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
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyError::NoCores => f.write_str("topology needs at least one core"),
            TopologyError::DuplicateLogicalId { id } => {
                write!(f, "duplicate logical-core id {id}")
            }
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
    probed: bool,
}

impl CpuTopology {
    /// Best-effort topology for the current process.
    ///
    /// Under `std` this reports the real logical-core count (via
    /// [`std::thread::available_parallelism`]) as a single flat `NUMA` node of
    /// [`CoreKind::Unknown`] cores, each its own physical core and cache domain.
    /// It makes **no** claim about the `P`/`E` split, `SMT` topology, `NUMA`
    /// geometry, or cache sharing — those need per-OS syscalls that are not
    /// wired yet, so [`CpuTopology::is_probed`] returns `false` here. Without
    /// `std` it falls back to a single core.
    #[must_use]
    pub fn detect() -> Self {
        let count = logical_core_count().max(1);
        let cores = (0..count)
            .map(|i| {
                let id = i as u32;
                TopologyCore::new(id, id, 0, 0, CoreKind::Unknown)
            })
            .collect();
        // `probed = false`: this is the uniform fallback, not a genuine OS read
        // of the hybrid/NUMA/cache geometry.
        Self {
            cores,
            numa_node_count: 1,
            probed: false,
        }
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

    /// Whether the richer fields (`P`/`E` class, `SMT`, `NUMA`, cache domains)
    /// are a genuine OS read (`true`) or the uniform [`CpuTopology::detect`]
    /// fallback (`false`).
    #[must_use]
    pub fn is_probed(&self) -> bool {
        self.probed
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
        let numa_node_count = self
            .cores
            .iter()
            .map(|c| c.numa_node)
            .max()
            .map_or(1, |m| m + 1);
        Ok(CpuTopology {
            cores: self.cores,
            numa_node_count,
            probed: true,
        })
    }
}
