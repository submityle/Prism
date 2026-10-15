//! NUMA proximity and hybrid-core topology (design §14, §24.6).
//!
//! Two related ideas live here, both about *where* work and memory sit:
//!
//! 1. **NUMA proximity + cross-node steal penalty.** Each worker's frame arena
//!    and fiber stacks are associated with the worker's NUMA node (design §14:
//!    "每 worker 竞技场/栈在其所在 NUMA 节点分配"). When a worker runs dry it
//!    prefers to steal from a *same-node* victim; a cross-node steal is still
//!    allowed but ranked behind local victims via a penalty, cutting remote
//!    memory traffic. [`steal_order`] and [`steal_penalty`] implement this
//!    ordering as pure, deterministic functions, independent of the live
//!    scheduler, so the policy is unit-testable on any machine.
//! 2. **big.LITTLE / hybrid cores.** [`CoreClass`] distinguishes performance
//!    (P) cores from efficiency (E) cores so latency-sensitive work can prefer
//!    P-cores and background work E-cores (design §24.6).
//!
//! ## Probe and fallback
//! [`Topology::detect`] consumes [`prism_platform::topology::CpuTopology`],
//! retaining OS CPU/node ids, optional cache/distance data and provenance.
//! Linux probes sysfs and the calling thread's allowed CPUs; unsupported or
//! failed probes explicitly fall back. Unknown hardware classes use the
//! ordinary Performance *scheduling lane*, not a claim that P cores were found.
//! Inspect [`Topology::platform`] for the original class and probe source.
//!
//! The `numa` feature enables local-first stealing for affinity-planned workers.
//! Pinning is best-effort; neither topology nor stealing binds physical memory
//! or guarantees locality when a pin fails. Unpinned pools retain flat stealing.

use core::fmt;

/// OS NUMA node identifier; ids may be sparse and need not include zero.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct NumaNodeId(u16);

impl NumaNodeId {
    /// The conventional single-node fallback id, not necessarily a real node.
    pub const ZERO: NumaNodeId = NumaNodeId(0);

    /// Construct a node id from a raw index.
    pub const fn new(index: u16) -> Self {
        NumaNodeId(index)
    }

    /// The raw index.
    pub const fn index(self) -> u16 {
        self.0
    }
}

impl fmt::Display for NumaNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node{}", self.0)
    }
}

/// Performance class of a physical core on a hybrid (big.LITTLE) CPU.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CoreClass {
    /// A performance ("big" / P) core: higher throughput and clocks. Preferred
    /// for latency-sensitive and `Critical`-lane work.
    Performance,
    /// An efficiency ("LITTLE" / E) core: lower power. Preferred for background
    /// and `Low`-priority work to keep P-cores free (design §24.6).
    Efficiency,
}

/// Description of a single logical core: which NUMA node it belongs to and its
/// performance class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CoreInfo {
    /// OS logical-core index (the value passed to affinity pinning).
    pub id: usize,
    /// NUMA node this core resides on.
    pub node: NumaNodeId,
    /// Performance class (P/E).
    pub class: CoreClass,
}

/// The penalty added to a steal candidate that lives on a different NUMA node.
///
/// Same-node victims score 0; cross-node victims score this. The absolute value
/// only needs to order local before remote — larger would still just mean
/// "after all local victims" in [`steal_order`].
pub const CROSS_NODE_STEAL_PENALTY: u32 = 1;

/// Resolved machine topology used by the scheduler for placement decisions.
#[derive(Clone, Debug)]
pub struct Topology {
    cores: Vec<CoreInfo>,
    node_count: usize,
    numa_enabled: bool,
    platform: Option<prism_platform::topology::CpuTopology>,
}

impl Topology {
    /// Probe through the platform backend, retaining explicit fallback status.
    pub fn detect() -> Self {
        Self::from_platform(prism_platform::topology::CpuTopology::detect())
    }

    /// Consume a validated platform probe (or injected topology). Unknown
    /// classes use the neutral ordinary lane; original evidence is retained.
    pub fn from_platform(platform: prism_platform::topology::CpuTopology) -> Self {
        use prism_platform::topology::CoreKind;
        let mut topo = Self::from_cores(
            platform
                .cores()
                .iter()
                .map(|core| CoreInfo {
                    id: core.logical_id as usize,
                    node: NumaNodeId::new(core.numa_node),
                    class: match core.kind {
                        CoreKind::Efficiency => CoreClass::Efficiency,
                        CoreKind::Performance | CoreKind::Unknown => CoreClass::Performance,
                    },
                })
                .collect(),
        );
        topo.numa_enabled = cfg!(feature = "numa");
        topo.platform = Some(platform);
        topo
    }

    /// Original platform evidence, including unknown classes/fallback reason.
    /// Explicit [`Self::from_cores`] data has no platform probe attached.
    pub fn platform(&self) -> Option<&prism_platform::topology::CpuTopology> {
        self.platform.as_ref()
    }

    /// Sorted distinct OS node ids, not a dense `0..node_count` range.
    pub fn node_ids(&self) -> Vec<NumaNodeId> {
        let mut nodes: Vec<_> = self.cores.iter().map(|core| core.node).collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    /// A uniform topology: `cores` performance cores (ids `0..cores`), all on
    /// NUMA node 0. The non-NUMA / non-hybrid fallback.
    pub fn uniform(cores: usize) -> Self {
        let cores = cores.max(1);
        let cores = (0..cores)
            .map(|id| CoreInfo {
                id,
                node: NumaNodeId::ZERO,
                class: CoreClass::Performance,
            })
            .collect();
        Self {
            cores,
            node_count: 1,
            numa_enabled: false,
            platform: None,
        }
    }

    /// Build a topology from an explicit core list (used by tests and by any
    /// platform adapter). Counts distinct nodes without renumbering OS ids.
    ///
    /// Panics if `cores` is empty or contains duplicate logical ids.
    pub fn from_cores(cores: Vec<CoreInfo>) -> Self {
        assert!(!cores.is_empty(), "topology needs at least one core");
        let mut ids: Vec<_> = cores.iter().map(|c| c.id).collect();
        ids.sort_unstable();
        assert!(
            !ids.windows(2).any(|pair| pair[0] == pair[1]),
            "duplicate logical core id"
        );
        let mut nodes: Vec<_> = cores.iter().map(|c| c.node).collect();
        nodes.sort_unstable();
        nodes.dedup();
        let node_count = nodes.len();
        Self {
            cores,
            node_count,
            numa_enabled: node_count > 1,
            platform: None,
        }
    }

    /// Number of logical cores.
    pub fn core_count(&self) -> usize {
        self.cores.len()
    }

    /// Number of distinct NUMA nodes (`1` on non-NUMA systems).
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Whether NUMA-aware scheduling was requested *and* more than one node is
    /// described. Honest: `false` on a uniform fallback even when the `numa`
    /// feature is on, because there is only one node to be near.
    pub fn numa_enabled(&self) -> bool {
        self.numa_enabled && self.node_count > 1
    }

    /// All cores, in id order as supplied.
    pub fn cores(&self) -> &[CoreInfo] {
        &self.cores
    }

    /// Look up a core by OS id.
    pub fn core(&self, id: usize) -> Option<CoreInfo> {
        self.cores.iter().copied().find(|c| c.id == id)
    }

    /// The NUMA node of a core id, if known.
    pub fn node_of_core(&self, id: usize) -> Option<NumaNodeId> {
        self.core(id).map(|c| c.node)
    }

    /// Iterate the cores on a given node.
    pub fn cores_on_node(&self, node: NumaNodeId) -> impl Iterator<Item = CoreInfo> + '_ {
        self.cores.iter().copied().filter(move |c| c.node == node)
    }

    /// Iterate the cores of a given performance class.
    pub fn cores_of_class(&self, class: CoreClass) -> impl Iterator<Item = CoreInfo> + '_ {
        self.cores.iter().copied().filter(move |c| c.class == class)
    }
}

/// The NUMA node a steal would reach from `thief_node` to `victim_node`:
/// `0` for a same-node (local) steal, [`CROSS_NODE_STEAL_PENALTY`] otherwise.
pub fn steal_penalty(thief_node: NumaNodeId, victim_node: NumaNodeId) -> u32 {
    if thief_node == victim_node {
        0
    } else {
        CROSS_NODE_STEAL_PENALTY
    }
}

/// Produce the order in which `thief` should probe other workers when stealing,
/// given each worker's NUMA node in `worker_nodes` (indexed by worker id).
///
/// Victims are ranked by `(steal_penalty, worker_index)`: all same-node workers
/// first (in ascending index order), then cross-node workers. The thief itself
/// is excluded. The result is fully deterministic — the property the design
/// relies on so steal order never perturbs results (§21).
pub fn steal_order(worker_nodes: &[NumaNodeId], thief: usize) -> Vec<usize> {
    let thief_node = worker_nodes.get(thief).copied().unwrap_or(NumaNodeId::ZERO);
    let mut victims: Vec<usize> = (0..worker_nodes.len()).filter(|&w| w != thief).collect();
    victims.sort_by_key(|&w| (steal_penalty(thief_node, worker_nodes[w]), w));
    victims
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pe(id: usize, node: u16, class: CoreClass) -> CoreInfo {
        CoreInfo {
            id,
            node: NumaNodeId::new(node),
            class,
        }
    }

    #[test]
    fn detect_preserves_platform_evidence() {
        let topo = Topology::detect();
        assert!(topo.core_count() >= 1);
        let platform = topo.platform().unwrap();
        assert_eq!(topo.node_count(), platform.node_ids().len());
        assert_eq!(
            topo.numa_enabled(),
            cfg!(feature = "numa") && topo.node_count() > 1
        );
        assert_eq!(topo.core_count(), platform.logical_core_count());
        if !platform.is_probed() {
            assert_eq!(topo.node_count(), 1);
            assert!(topo
                .cores()
                .iter()
                .all(|c| c.class == CoreClass::Performance));
        }
    }

    #[test]
    fn from_cores_derives_node_count() {
        let topo = Topology::from_cores(vec![
            pe(0, 0, CoreClass::Performance),
            pe(1, 0, CoreClass::Performance),
            pe(2, 1, CoreClass::Efficiency),
            pe(3, 1, CoreClass::Efficiency),
        ]);
        assert_eq!(topo.node_count(), 2);
        assert_eq!(topo.core_count(), 4);
        assert!(topo.numa_enabled());
        assert_eq!(topo.node_of_core(2), Some(NumaNodeId::new(1)));
        assert_eq!(topo.cores_on_node(NumaNodeId::new(1)).count(), 2);
        assert_eq!(topo.cores_of_class(CoreClass::Efficiency).count(), 2);
    }

    #[test]
    fn penalty_is_zero_same_node() {
        assert_eq!(steal_penalty(NumaNodeId::new(1), NumaNodeId::new(1)), 0);
        assert_eq!(
            steal_penalty(NumaNodeId::new(0), NumaNodeId::new(1)),
            CROSS_NODE_STEAL_PENALTY
        );
    }

    #[test]
    fn steal_order_prefers_same_node_then_index() {
        // workers: 0,1 on node 0; 2,3 on node 1.
        let nodes = [
            NumaNodeId::new(0),
            NumaNodeId::new(0),
            NumaNodeId::new(1),
            NumaNodeId::new(1),
        ];
        // Thief = worker 0 (node 0): local victim is 1, then cross-node 2,3.
        assert_eq!(steal_order(&nodes, 0), vec![1, 2, 3]);
        // Thief = worker 3 (node 1): local victim is 2, then cross-node 0,1.
        assert_eq!(steal_order(&nodes, 3), vec![2, 0, 1]);
    }

    #[test]
    fn steal_order_is_deterministic_and_total() {
        let nodes = [
            NumaNodeId::new(1),
            NumaNodeId::new(0),
            NumaNodeId::new(1),
            NumaNodeId::new(0),
        ];
        let a = steal_order(&nodes, 2);
        let b = steal_order(&nodes, 2);
        assert_eq!(a, b);
        // Covers every other worker exactly once.
        let mut seen = a.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 3]);
        // Same-node victim (0) comes before cross-node victims (1, 3).
        assert_eq!(a[0], 0);
    }

    #[test]
    fn uniform_single_node_steal_order_is_index_order() {
        let nodes = vec![NumaNodeId::ZERO; 4];
        assert_eq!(steal_order(&nodes, 1), vec![0, 2, 3]);
    }
}
