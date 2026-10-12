//! Deterministic dependency order for runtime reassembly.
//!
//! A reload pass must reassemble dependencies before the assets that consume
//! them. [`ReloadPlan`] computes that order for only the transitive closure of
//! one or more roots, leaving scheduling and execution to its caller. Keeping
//! this as a pure graph operation makes it safe to use from synchronous and
//! asynchronous reload implementations alike.

use crate::dependency::DependencyGraph;
use crate::id::UntypedAssetId;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

/// An error produced while planning a runtime reassembly pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReloadPlanError {
    /// The selected dependency closure contains a cycle.
    ///
    /// `participants` contains every node in the selected closure that could
    /// not be ordered by Kahn's algorithm. This is deterministic and includes
    /// any nodes downstream of a cycle, which are also impossible to
    /// reassemble safely during this pass.
    Cycle {
        /// Nodes that could not be placed in dependency-first order.
        participants: Vec<UntypedAssetId>,
    },
}

impl fmt::Display for ReloadPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cycle { participants } => write!(
                f,
                "reload dependency cycle involving {} unorderable asset(s)",
                participants.len()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReloadPlanError {}

/// A dependency-first, deterministic runtime reassembly order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReloadPlan {
    order: Vec<UntypedAssetId>,
}

impl ReloadPlan {
    /// Builds a plan for the transitive dependency closure of `roots`.
    ///
    /// Roots are merged and deduplicated. The planner always selects the
    /// smallest ready asset id, so the result is independent of root and edge
    /// insertion order. A root that is not currently registered in the graph
    /// is treated as an isolated node; this lets callers plan a newly reserved
    /// runtime asset without a separate graph mutation.
    pub fn build(
        graph: &DependencyGraph,
        roots: impl IntoIterator<Item = UntypedAssetId>,
    ) -> Result<Self, ReloadPlanError> {
        let roots: BTreeSet<_> = roots.into_iter().collect();
        if roots.is_empty() {
            return Ok(Self::default());
        }

        let closure = dependency_closure(graph, &roots);
        let order = topological_order(graph, &closure)?;
        Ok(Self { order })
    }

    /// Returns the planned order as a slice without consuming the plan.
    #[must_use]
    pub fn as_slice(&self) -> &[UntypedAssetId] {
        &self.order
    }

    /// Returns the number of assets in the plan.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns whether the plan contains no assets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Iterates over assets in dependency-first order.
    pub fn iter(&self) -> core::slice::Iter<'_, UntypedAssetId> {
        self.order.iter()
    }

    /// Consumes the plan and returns its dependency-first order.
    #[must_use]
    pub fn into_order(self) -> Vec<UntypedAssetId> {
        self.order
    }
}

impl AsRef<[UntypedAssetId]> for ReloadPlan {
    fn as_ref(&self) -> &[UntypedAssetId] {
        self.as_slice()
    }
}

impl IntoIterator for ReloadPlan {
    type Item = UntypedAssetId;
    type IntoIter = alloc::vec::IntoIter<UntypedAssetId>;

    fn into_iter(self) -> Self::IntoIter {
        self.order.into_iter()
    }
}

impl<'a> IntoIterator for &'a ReloadPlan {
    type Item = &'a UntypedAssetId;
    type IntoIter = core::slice::Iter<'a, UntypedAssetId>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Builds a runtime reload plan without requiring callers to name the plan
/// type explicitly.
pub fn plan_reload(
    graph: &DependencyGraph,
    roots: impl IntoIterator<Item = UntypedAssetId>,
) -> Result<ReloadPlan, ReloadPlanError> {
    ReloadPlan::build(graph, roots)
}

fn dependency_closure(
    graph: &DependencyGraph,
    roots: &BTreeSet<UntypedAssetId>,
) -> BTreeSet<UntypedAssetId> {
    let mut closure = roots.clone();
    let mut pending: Vec<_> = roots.iter().copied().collect();

    while let Some(asset) = pending.pop() {
        for dependency in graph.dependencies(asset) {
            if closure.insert(dependency) {
                pending.push(dependency);
            }
        }
    }

    closure
}

fn topological_order(
    graph: &DependencyGraph,
    closure: &BTreeSet<UntypedAssetId>,
) -> Result<Vec<UntypedAssetId>, ReloadPlanError> {
    let mut unmet = BTreeMap::new();
    let mut dependents: BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>> = BTreeMap::new();

    for &asset in closure {
        let dependencies = graph.dependencies(asset);
        let count = dependencies
            .iter()
            .filter(|dependency| closure.contains(dependency))
            .count();
        unmet.insert(asset, count);

        for dependency in dependencies {
            if closure.contains(&dependency) {
                dependents.entry(dependency).or_default().insert(asset);
            }
        }
    }

    let mut ready: BTreeSet<_> = unmet
        .iter()
        .filter_map(|(&asset, &count)| (count == 0).then_some(asset))
        .collect();
    let mut order = Vec::with_capacity(closure.len());

    while let Some(&asset) = ready.iter().next() {
        ready.remove(&asset);
        order.push(asset);

        if let Some(dependent_assets) = dependents.get(&asset) {
            for &dependent in dependent_assets {
                let count = unmet
                    .get_mut(&dependent)
                    .expect("dependent must be in the selected closure");
                *count -= 1;
                if *count == 0 {
                    ready.insert(dependent);
                }
            }
        }
    }

    if order.len() == closure.len() {
        return Ok(order);
    }

    let ordered: BTreeSet<_> = order.into_iter().collect();
    let participants = closure.difference(&ordered).copied().collect();
    Err(ReloadPlanError::Cycle { participants })
}
