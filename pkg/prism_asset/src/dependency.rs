//! Asset dependency tracking with deterministic topological ordering.

use crate::id::UntypedAssetId;
use crate::load_state::{LoadState, RecursiveDependencyLoadState};
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
/// Edges point from an asset to the assets it depends on. The graph answers the
/// questions the loader needs: the direct dependencies/dependents of an asset,
/// a deterministic load order (dependency before dependent), and — maintained
/// *incrementally* (design §8.1/§8.2) — the recursive readiness of each asset's
/// full dependency closure plus hot-reload invalidation propagation (§12).
///
/// All internal storage is ordered (`BTreeMap`/`BTreeSet`), so queries, the
/// topological order, and the set of propagation-affected nodes are fully
/// deterministic across runs — the basis for regression tests and networked
/// consistency.
///
/// ## Incremental readiness
/// Rather than re-folding each asset's whole closure on every query, the graph
/// caches a per-node [`RecursiveDependencyLoadState`]. Setting a node's own
/// [`LoadState`] with [`set_self_state`](Self::set_self_state) recomputes that
/// node and pushes the change along `dependent` edges only as far as values
/// actually move — `O(affected subgraph)`, not `O(closure)`. The returned set
/// of newly-ready nodes is exactly the set for which the loader should emit
/// [`AssetEvent::LoadedWithDependencies`](crate::AssetEvent::LoadedWithDependencies).
#[derive(Clone, Default)]
pub struct DependencyGraph {
    nodes: BTreeSet<UntypedAssetId>,
    deps: BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>>,
    dependents: BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>>,
    /// Each node's own load state (independent of its dependencies). Absent
    /// entries read as [`LoadState::NotLoaded`].
    self_state: BTreeMap<UntypedAssetId, LoadState>,
    /// Cached recursive closure state per node, maintained incrementally.
    rec_state: BTreeMap<UntypedAssetId, RecursiveDependencyLoadState>,
    /// Nodes invalidated by a source change and awaiting reassembly (§12).
    stale: BTreeSet<UntypedAssetId>,
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
        self.dependents.entry(dependency).or_default().insert(asset);
        // The new edge can only change `asset`'s closure (it gained a dep), so
        // recompute from there and let it ripple to its own dependents.
        let _ = self.propagate_from([asset]);
    }

    /// Like [`add_dependency`](Self::add_dependency) but refuses an edge that
    /// would close a cycle, detected *at assembly time* (design §8.3) rather
    /// than deferred to [`topological_order`](Self::topological_order). The
    /// returned [`DependencyError::Cycle`] lists exactly the nodes on the
    /// offending cycle (deterministically sorted). A self-edge is a no-op `Ok`.
    pub fn try_add_dependency(
        &mut self,
        asset: UntypedAssetId,
        dependency: UntypedAssetId,
    ) -> Result<(), DependencyError> {
        self.nodes.insert(asset);
        self.nodes.insert(dependency);
        if asset == dependency {
            return Ok(());
        }
        // Adding `asset -> dependency` cycles iff `dependency` can already
        // reach `asset` along existing dependency edges.
        if self.reachable(dependency, &self.deps).contains(&asset) {
            let forward = self.reachable(dependency, &self.deps);
            let backward = self.reachable(asset, &self.dependents);
            let participants = forward.intersection(&backward).copied().collect();
            return Err(DependencyError::Cycle { participants });
        }
        self.add_dependency(asset, dependency);
        Ok(())
    }

    /// Removes an asset and every edge touching it, then incrementally repairs
    /// the recursive readiness of everything that used to depend on it.
    pub fn remove_asset(&mut self, asset: UntypedAssetId) {
        self.nodes.remove(&asset);
        self.self_state.remove(&asset);
        self.rec_state.remove(&asset);
        self.stale.remove(&asset);
        if let Some(deps) = self.deps.remove(&asset) {
            for dep in deps {
                if let Some(set) = self.dependents.get_mut(&dep) {
                    set.remove(&asset);
                }
            }
        }
        // Capture former dependents before dropping the edge set so their
        // closures can be recomputed now that one dependency is gone.
        let former_dependents = self.dependents.remove(&asset).unwrap_or_default();
        for &dependent in &former_dependents {
            if let Some(set) = self.deps.get_mut(&dependent) {
                set.remove(&asset);
            }
        }
        let _ = self.propagate_from(former_dependents);
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

    // ---- incremental readiness helpers (private) --------------------------

    /// This node's own [`LoadState`], defaulting to `NotLoaded` when unset.
    fn self_of(&self, node: UntypedAssetId) -> LoadState {
        self.self_state.get(&node).copied().unwrap_or_default()
    }

    /// This node's cached recursive closure state, defaulting to `NotLoaded`.
    fn recursive_of(&self, node: UntypedAssetId) -> RecursiveDependencyLoadState {
        self.rec_state.get(&node).copied().unwrap_or_default()
    }

    /// Recomputes a node's recursive state from its own [`LoadState`] folded
    /// with the cached recursive state of each direct dependency (worst-wins).
    /// Reads only direct neighbours, so it is `O(outdegree)` given an
    /// up-to-date cache — the invariant [`propagate_from`](Self::propagate_from)
    /// maintains.
    fn compute_rec(&self, node: UntypedAssetId) -> RecursiveDependencyLoadState {
        let mut acc = RecursiveDependencyLoadState::from(self.self_of(node));
        if let Some(deps) = self.deps.get(&node) {
            for &dep in deps {
                acc = acc.combine(self.recursive_of(dep));
            }
        }
        acc
    }

    /// Every node reachable from `start` (inclusive) by following `map` edges.
    /// Iterative DFS over ordered sets, so the result is deterministic.
    fn reachable(
        &self,
        start: UntypedAssetId,
        map: &BTreeMap<UntypedAssetId, BTreeSet<UntypedAssetId>>,
    ) -> BTreeSet<UntypedAssetId> {
        let mut seen = BTreeSet::new();
        seen.insert(start);
        let mut stack = Vec::new();
        stack.push(start);
        while let Some(node) = stack.pop() {
            if let Some(next) = map.get(&node) {
                for &n in next {
                    if seen.insert(n) {
                        stack.push(n);
                    }
                }
            }
        }
        seen
    }

    /// Recomputes recursive state starting from `seeds`, rippling along
    /// `dependent` edges only while values actually change. Returns the set of
    /// nodes whose recursive state moved. Terminates because the dependency
    /// graph is a DAG — cycles are rejected at assembly time by
    /// [`try_add_dependency`](Self::try_add_dependency).
    fn propagate_from(
        &mut self,
        seeds: impl IntoIterator<Item = UntypedAssetId>,
    ) -> BTreeSet<UntypedAssetId> {
        let mut changed = BTreeSet::new();
        let mut stack: Vec<UntypedAssetId> = seeds.into_iter().collect();
        while let Some(node) = stack.pop() {
            let new = self.compute_rec(node);
            if new != self.recursive_of(node) {
                self.rec_state.insert(node, new);
                changed.insert(node);
                if let Some(deps) = self.dependents.get(&node) {
                    stack.extend(deps.iter().copied());
                }
            }
        }
        changed
    }

    // ---- incremental readiness + invalidation API (public) ----------------

    /// Sets a node's own [`LoadState`] and incrementally repairs the recursive
    /// readiness of it and everything that transitively depends on it.
    ///
    /// Returns the set of nodes whose [`RecursiveDependencyLoadState`] changed
    /// as a result — exactly the nodes for which the loader should re-evaluate
    /// and, when they reach `Loaded`, emit
    /// [`AssetEvent::LoadedWithDependencies`](crate::AssetEvent::LoadedWithDependencies).
    pub fn set_self_state(
        &mut self,
        asset: UntypedAssetId,
        state: LoadState,
    ) -> BTreeSet<UntypedAssetId> {
        self.nodes.insert(asset);
        self.self_state.insert(asset, state);
        self.propagate_from([asset])
    }

    /// This asset's own load state, independent of its dependencies.
    #[must_use]
    pub fn self_state(&self, asset: UntypedAssetId) -> LoadState {
        self.self_of(asset)
    }

    /// This asset's recursive load state over its full dependency closure.
    ///
    /// Reads the incrementally-maintained cache; falls back to an on-demand
    /// fold for nodes whose state has never been set (so a never-touched node
    /// still reports a correct `NotLoaded`-dominated closure).
    #[must_use]
    pub fn recursive_state(&self, asset: UntypedAssetId) -> RecursiveDependencyLoadState {
        self.rec_state
            .get(&asset)
            .copied()
            .unwrap_or_else(|| self.compute_rec(asset))
    }

    /// Marks `asset` and every asset that transitively depends on it as stale,
    /// i.e. awaiting reassembly after a source change (design §12 hot reload).
    /// Dependents are stale because their composed/derived data was built from
    /// the now-changed input. Returns the affected nodes, sorted.
    pub fn invalidate(&mut self, asset: UntypedAssetId) -> Vec<UntypedAssetId> {
        let affected = self.reachable(asset, &self.dependents);
        for &node in &affected {
            self.stale.insert(node);
        }
        affected.into_iter().collect()
    }

    /// Whether `asset` is currently marked stale.
    #[must_use]
    pub fn is_stale(&self, asset: UntypedAssetId) -> bool {
        self.stale.contains(&asset)
    }

    /// Clears the stale mark on `asset`, returning whether it had been set.
    pub fn clear_stale(&mut self, asset: UntypedAssetId) -> bool {
        self.stale.remove(&asset)
    }

    /// Drains every stale node, returning them sorted — the reload pass uses
    /// this to collect its work set and reset the pending marks in one step.
    pub fn take_stale(&mut self) -> Vec<UntypedAssetId> {
        core::mem::take(&mut self.stale).into_iter().collect()
    }

    /// The number of nodes currently marked stale.
    #[must_use]
    pub fn stale_count(&self) -> usize {
        self.stale.len()
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
