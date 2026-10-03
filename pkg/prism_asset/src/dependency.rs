//! Asset dependency tracking with deterministic topological ordering.

use crate::id::UntypedAssetId;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

/// An error produced while ordering a [`DependencyGraph`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DependencyError {
    /// The graph contains a dependency cycle; `participants` lists every asset
    /// that could not be ordered, sorted for determinism.
    Cycle {
        /// The assets involved in (or downstream of) the cycle.
        participants: Vec<UntypedAssetId>,
    },
}

impl fmt::Display for DependencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cycle { participants } => {
                write!(
                    f,
                    "dependency cycle involving {} asset(s)",
                    participants.len()
                )
            }
        }
    }
}

/// A directed graph of asset→dependency edges.
///
/// Edges point from an asset to the assets it depends on. The graph answers two
/// questions the loader needs: the direct dependencies/dependents of an asset,
/// and a load order in which every dependency precedes the assets that need it.
/// All internal storage is ordered (`BTreeMap`/`BTreeSet`), so queries and the
/// topological order are fully deterministic across runs.
#[derive(Clone, Default)]
pub struct DependencyGraph {
    nodes: BTreeSet<UntypedAssetId>,
    deps: BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>>,
    dependents: BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>>,
}

impl DependencyGraph {
    /// Creates an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an asset node with no edges. Idempotent.
    pub fn add_asset(&mut self, asset: UntypedAssetId) {
        self.nodes.insert(asset);
    }

    /// Records that `asset` depends on `dependency`, inserting both nodes if
    /// absent. Self-edges are ignored, since an asset cannot depend on itself.
    pub fn add_dependency(&mut self, asset: UntypedAssetId, dependency: UntypedAssetId) {
        self.nodes.insert(asset);
        self.nodes.insert(dependency);
        if asset == dependency {
            return;
        }
        self.deps.entry(asset).or_default().insert(dependency);
        self.dependents
            .entry(dependency)
            .or_default()
            .insert(asset);
    }

    /// Removes an asset and every edge touching it.
    pub fn remove_asset(&mut self, asset: UntypedAssetId) {
        self.nodes.remove(&asset);
        if let Some(deps) = self.deps.remove(&asset) {
            for dep in deps {
                if let Some(set) = self.dependents.get_mut(&dep) {
                    set.remove(&asset);
                }
            }
        }
        if let Some(dependents) = self.dependents.remove(&asset) {
            for dependent in dependents {
                if let Some(set) = self.deps.get_mut(&dependent) {
                    set.remove(&asset);
                }
            }
        }
    }

    /// The direct dependencies of `asset`, sorted.
    #[must_use]
    pub fn dependencies(&self, asset: UntypedAssetId) -> Vec<UntypedAssetId> {
        self.deps
            .get(&asset)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// The assets that directly depend on `asset`, sorted.
    #[must_use]
    pub fn dependents(&self, asset: UntypedAssetId) -> Vec<UntypedAssetId> {
        self.dependents
            .get(&asset)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// The number of asset nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Produces a load order in which every dependency appears before the
    /// assets that depend on it.
    ///
    /// Uses Kahn's algorithm, always selecting the smallest ready id to keep
    /// the output deterministic. Returns [`DependencyError::Cycle`] listing the
    /// unorderable nodes when the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<UntypedAssetId>, DependencyError> {
        let mut unmet: BTreeMap<UntypedAssetId, usize> = BTreeMap::new();
        for &node in &self.nodes {
            let count = self.deps.get(&node).map_or(0, BTreeSet::len);
            unmet.insert(node, count);
        }

        let mut ready: BTreeSet<UntypedAssetId> = unmet
            .iter()
            .filter_map(|(&node, &count)| (count == 0).then_some(node))
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(&next) = ready.iter().next() {
            ready.remove(&next);
            order.push(next);
            if let Some(dependents) = self.dependents.get(&next) {
                for &dependent in dependents {
                    if let Some(count) = unmet.get_mut(&dependent) {
                        *count -= 1;
                        if *count == 0 {
                            ready.insert(dependent);
                        }
                    }
                }
            }
        }

        if order.len() == self.nodes.len() {
            Ok(order)
        } else {
            let ordered: BTreeSet<UntypedAssetId> = order.into_iter().collect();
            let participants = self.nodes.difference(&ordered).copied().collect();
            Err(DependencyError::Cycle { participants })
        }
    }
}
