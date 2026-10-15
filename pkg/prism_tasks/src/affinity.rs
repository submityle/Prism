//! Worker core-pinning and hybrid-core worker placement (design §11, §14, §24.6).
//!
//! Pinning each compute worker to a fixed OS core reduces cross-core migration
//! jitter and keeps a worker's cache (and its NUMA-local frame arena/stacks)
//! hot (design §14: "worker 绑核减少迁移抖动与 cache 失效"). This module turns
//! a [`Topology`] into a concrete worker→core plan and performs the pin through
//! [`prism_platform`]'s affinity API.
//!
//! ## Hybrid-core awareness (big.LITTLE)
//! When the topology distinguishes performance (P) and efficiency (E) cores,
//! [`plan_worker_cores`] places low-numbered workers on P-cores first so that
//! the latency-sensitive lanes (which the scheduler keeps at low worker ids)
//! run on big cores, and spills later workers onto E-cores (design §24.6).
//!
//! ## Honest degradation
//! Pinning is best-effort. On platforms with no per-core pinning (macOS, and
//! anything [`prism_platform::affinity_supported`] reports `false`),
//! [`WorkerCorePlan::pin_current`] returns [`AffinityError::Unsupported`] and
//! the caller simply runs unpinned — the pool stays correct, it just forgoes
//! the locality win. No pinning is ever faked.

use prism_platform::AffinityError;

use crate::numa::{CoreClass, NumaNodeId, Topology};

/// Whether this build can actually pin threads to cores (forwarded from
/// [`prism_platform`]). `false` on macOS and other unsupported platforms.
pub fn affinity_supported() -> bool {
    prism_platform::affinity_supported()
}

/// Pin the calling thread to a single OS core, best-effort.
///
/// Returns [`AffinityError::Unsupported`] where the platform cannot pin, rather
/// than pretending to succeed.
pub fn pin_current_thread_to_core(core: usize) -> Result<(), AffinityError> {
    prism_platform::set_current_thread_affinity(core)
}

/// The core a particular worker is assigned to, with its locality metadata.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CoreAssignment {
    /// Worker index within the pool.
    pub worker: usize,
    /// OS logical-core id to pin to.
    pub core: usize,
    /// NUMA node that core lives on (drives arena/stack placement and steal
    /// preference).
    pub node: NumaNodeId,
    /// Performance class of the core (P/E).
    pub class: CoreClass,
}

/// How to spread workers across the available core classes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CoreClassPolicy {
    /// Fill performance cores first (low worker ids), then efficiency cores.
    /// The default: keeps latency-sensitive low-id workers on big cores.
    #[default]
    PerformanceFirst,
    /// Ignore class; assign by core id order only (round-robin over all cores).
    Flat,
}

/// A deterministic mapping from worker index to a core assignment.
#[derive(Clone, Debug)]
pub struct WorkerCorePlan {
    assignments: Vec<CoreAssignment>,
}

/// Result of attempting to bind the calling thread to a planned worker core.
///
/// The planned assignment is kept separate from the OS outcome. In particular,
/// [`WorkerBindingOutcome::Unbound`] is not a successful binding: callers may
/// continue safely without affinity, but must not report that the worker is
/// pinned.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkerBindingOutcome {
    /// The OS accepted the affinity request for this assignment.
    Bound {
        /// Assignment that was successfully applied.
        assignment: CoreAssignment,
    },
    /// The worker has a valid assignment, but the OS did not apply it.
    Unbound {
        /// Assignment that was requested.
        assignment: CoreAssignment,
        /// Explicit capability gap or OS rejection.
        error: AffinityError,
    },
    /// No assignment exists for the requested worker index.
    InvalidWorker {
        /// Worker index that was requested.
        worker: usize,
    },
}

impl WorkerBindingOutcome {
    /// Whether the OS accepted a real CPU-affinity binding.
    #[must_use]
    pub const fn is_bound(self) -> bool {
        matches!(self, Self::Bound { .. })
    }

    /// The planned assignment, if the worker index was valid.
    #[must_use]
    pub const fn assignment(self) -> Option<CoreAssignment> {
        match self {
            Self::Bound { assignment } | Self::Unbound { assignment, .. } => Some(assignment),
            Self::InvalidWorker { .. } => None,
        }
    }

    /// The platform error when a valid assignment was not applied.
    #[must_use]
    pub const fn error(self) -> Option<AffinityError> {
        match self {
            Self::Unbound { error, .. } => Some(error),
            Self::Bound { .. } | Self::InvalidWorker { .. } => None,
        }
    }
}

impl WorkerCorePlan {
    /// Number of workers covered.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// The assignment for `worker`, if any.
    pub fn assignment(&self, worker: usize) -> Option<CoreAssignment> {
        self.assignments.get(worker).copied()
    }

    /// All assignments in worker order.
    pub fn assignments(&self) -> &[CoreAssignment] {
        &self.assignments
    }

    /// Per-worker NUMA node vector (indexed by worker), as consumed by
    /// [`crate::numa::steal_order`].
    pub fn worker_nodes(&self) -> Vec<NumaNodeId> {
        self.assignments.iter().map(|a| a.node).collect()
    }

    /// The NUMA node assigned to `worker` (node 0 if out of range).
    pub fn node_of_worker(&self, worker: usize) -> NumaNodeId {
        self.assignment(worker)
            .map(|a| a.node)
            .unwrap_or(NumaNodeId::ZERO)
    }

    /// Pin the *calling* thread to the core assigned to `worker`, best-effort.
    ///
    /// Returns [`AffinityError::InvalidCore`] if `worker` has no assignment,
    /// [`AffinityError::Unsupported`] on platforms that cannot pin, or whatever
    /// the OS reports. A worker should treat any error as "run unpinned".
    pub fn pin_current(&self, worker: usize) -> Result<(), AffinityError> {
        match self.bind_current(worker) {
            WorkerBindingOutcome::Bound { .. } => Ok(()),
            WorkerBindingOutcome::Unbound { error, .. } => Err(error),
            WorkerBindingOutcome::InvalidWorker { .. } => Err(AffinityError::InvalidCore),
        }
    }

    /// Bind the calling thread to the core assigned to `worker` and preserve
    /// the complete outcome for diagnostics and capability-aware callers.
    ///
    /// `Unbound` is the safe degradation path: the worker remains correct and
    /// may run without pinning, but no binding is claimed. An unsupported
    /// platform is reported as [`AffinityError::Unsupported`] inside that
    /// outcome rather than being converted into success.
    #[must_use]
    pub fn bind_current(&self, worker: usize) -> WorkerBindingOutcome {
        let Some(assignment) = self.assignment(worker) else {
            return WorkerBindingOutcome::InvalidWorker { worker };
        };
        match pin_current_thread_to_core(assignment.core) {
            Ok(()) => WorkerBindingOutcome::Bound { assignment },
            Err(error) => WorkerBindingOutcome::Unbound { assignment, error },
        }
    }
}

/// Build a deterministic worker→core plan for `workers` workers over
/// `topology`, honoring `policy` for hybrid cores.
///
/// Workers are assigned round-robin over the (policy-ordered) core list, so a
/// pool with more workers than cores still produces a total assignment (several
/// workers may share a core — the OS scheduler then time-slices them, which is
/// the honest behavior when oversubscribed).
pub fn plan_worker_cores(
    topology: &Topology,
    workers: usize,
    policy: CoreClassPolicy,
) -> WorkerCorePlan {
    // Order cores per policy. `PerformanceFirst` ranks P-cores ahead of E-cores
    // (ties broken by id); `Flat` keeps id order.
    let mut cores: Vec<_> = topology.cores().to_vec();
    match policy {
        CoreClassPolicy::PerformanceFirst => cores.sort_by_key(|c| {
            let class_rank = match c.class {
                CoreClass::Performance => 0u8,
                CoreClass::Efficiency => 1u8,
            };
            (class_rank, c.id)
        }),
        CoreClassPolicy::Flat => cores.sort_by_key(|c| c.id),
    }

    let assignments = (0..workers)
        .map(|worker| {
            let core = cores[worker % cores.len()];
            CoreAssignment {
                worker,
                core: core.id,
                node: core.node,
                class: core.class,
            }
        })
        .collect();

    WorkerCorePlan { assignments }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numa::CoreInfo;

    fn core(id: usize, node: u16, class: CoreClass) -> CoreInfo {
        CoreInfo {
            id,
            node: NumaNodeId::new(node),
            class,
        }
    }

    #[test]
    fn flat_plan_is_round_robin_by_id() {
        let topo = Topology::uniform(4);
        let plan = plan_worker_cores(&topo, 6, CoreClassPolicy::Flat);
        assert_eq!(plan.len(), 6);
        let cores: Vec<_> = plan.assignments().iter().map(|a| a.core).collect();
        assert_eq!(cores, vec![0, 1, 2, 3, 0, 1]);
        // Uniform topology => every worker on node 0, performance class.
        assert!(plan
            .assignments()
            .iter()
            .all(|a| a.node == NumaNodeId::ZERO));
    }

    #[test]
    fn performance_first_places_low_workers_on_p_cores() {
        // 2 P-cores (ids 0,1) + 2 E-cores (ids 2,3).
        let topo = Topology::from_cores(vec![
            core(0, 0, CoreClass::Performance),
            core(1, 0, CoreClass::Performance),
            core(2, 0, CoreClass::Efficiency),
            core(3, 0, CoreClass::Efficiency),
        ]);
        let plan = plan_worker_cores(&topo, 4, CoreClassPolicy::PerformanceFirst);
        assert_eq!(plan.assignment(0).unwrap().class, CoreClass::Performance);
        assert_eq!(plan.assignment(1).unwrap().class, CoreClass::Performance);
        assert_eq!(plan.assignment(2).unwrap().class, CoreClass::Efficiency);
        assert_eq!(plan.assignment(3).unwrap().class, CoreClass::Efficiency);
    }

    #[test]
    fn worker_nodes_follow_core_nodes() {
        let topo = Topology::from_cores(vec![
            core(0, 0, CoreClass::Performance),
            core(1, 1, CoreClass::Performance),
        ]);
        let plan = plan_worker_cores(&topo, 2, CoreClassPolicy::Flat);
        assert_eq!(
            plan.worker_nodes(),
            vec![NumaNodeId::new(0), NumaNodeId::new(1)]
        );
        assert_eq!(plan.node_of_worker(1), NumaNodeId::new(1));
    }

    #[test]
    fn pin_current_reports_honestly() {
        let topo = Topology::uniform(2);
        let plan = plan_worker_cores(&topo, 2, CoreClassPolicy::Flat);
        // Out-of-range worker is InvalidCore regardless of platform support.
        assert_eq!(plan.pin_current(99), Err(AffinityError::InvalidCore));
        // In-range: either succeeds (Linux/Windows) or honestly reports
        // Unsupported (macOS). Never a silent fake-success on an unsupported OS.
        let result = plan.pin_current(0);
        if affinity_supported() {
            // On a supporting platform core 0 always exists.
            assert!(result.is_ok() || matches!(result, Err(AffinityError::SystemError(_))));
        } else {
            assert_eq!(result, Err(AffinityError::Unsupported));
        }

        match plan.bind_current(99) {
            WorkerBindingOutcome::InvalidWorker { worker } => assert_eq!(worker, 99),
            other => panic!("unexpected invalid-worker outcome: {other:?}"),
        }
        match plan.bind_current(0) {
            WorkerBindingOutcome::Bound { assignment } => {
                assert!(affinity_supported());
                assert_eq!(assignment.worker, 0);
            }
            WorkerBindingOutcome::Unbound { assignment, error } => {
                assert_eq!(assignment.worker, 0);
                if affinity_supported() {
                    assert!(matches!(error, AffinityError::SystemError(_)));
                } else {
                    assert_eq!(error, AffinityError::Unsupported);
                }
            }
            WorkerBindingOutcome::InvalidWorker { .. } => unreachable!(),
        }
    }
}
