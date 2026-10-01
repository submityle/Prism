//! Read-only introspection of a [`Runtime`]'s reactive dependency graph.
//!
//! A running [`Runtime`] keeps its graph private so that the propagation
//! algorithm stays the single source of truth. This module exposes a frozen,
//! owned *view* of that graph — a [`GraphSnapshot`] of [`NodeInfo`] records — so
//! tooling can inspect shape (which signals feed which memos and effects)
//! without being able to mutate or drive the runtime.
//!
//! Snapshots are deterministic: nodes are ordered by [`NodeId`] and every edge
//! list is sorted, so the same graph always yields byte-identical output. This
//! is the data the Loom inspector's dependency-graph tooling consumes.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::node::{NodeId, NodeKind};
use crate::Runtime;

/// The role a node plays in the reactive graph, mirroring the runtime's internal
/// node kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKindInfo {
    /// A writable source value with no computation.
    Signal,
    /// A derived, lazily recomputed, cached value.
    Memo,
    /// A side effect that re-runs when its sources change.
    Effect,
}

/// An owned description of a single reactive node and its edges.
///
/// `sources` are the nodes this node reads (its dependencies); `observers` are
/// the nodes that read this node (its dependents). Both lists are sorted
/// ascending for deterministic output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeInfo {
    /// The node's stable identifier (its slot index in the runtime).
    pub id: NodeId,
    /// The node's role.
    pub kind: NodeKindInfo,
    /// Nodes this node depends on, sorted ascending.
    pub sources: Vec<NodeId>,
    /// Nodes that depend on this node, sorted ascending.
    pub observers: Vec<NodeId>,
}

/// An owned, deterministic snapshot of a runtime's dependency graph.
///
/// Produced by [`Runtime::graph_snapshot`]. Nodes are ordered by [`NodeId`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphSnapshot {
    /// Every live node, ordered by [`NodeId`].
    pub nodes: Vec<NodeInfo>,
}

impl GraphSnapshot {
    /// Number of live nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of directed dependency edges.
    ///
    /// Counted as the sum of every node's source count; this equals the sum of
    /// every node's observer count for a consistent graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.nodes.iter().map(|node| node.sources.len()).sum()
    }

    /// Returns the node with identifier `id`, if present.
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&NodeInfo> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Returns every signal node, in [`NodeId`] order.
    #[must_use]
    pub fn signals(&self) -> Vec<&NodeInfo> {
        self.of_kind(NodeKindInfo::Signal)
    }

    /// Returns every memo node, in [`NodeId`] order.
    #[must_use]
    pub fn memos(&self) -> Vec<&NodeInfo> {
        self.of_kind(NodeKindInfo::Memo)
    }

    /// Returns every effect node, in [`NodeId`] order.
    #[must_use]
    pub fn effects(&self) -> Vec<&NodeInfo> {
        self.of_kind(NodeKindInfo::Effect)
    }

    /// Returns the dependencies (sources) of `id`, if the node exists.
    #[must_use]
    pub fn sources_of(&self, id: NodeId) -> Option<&[NodeId]> {
        self.get(id).map(|node| node.sources.as_slice())
    }

    /// Returns the dependents (observers) of `id`, if the node exists.
    #[must_use]
    pub fn observers_of(&self, id: NodeId) -> Option<&[NodeId]> {
        self.get(id).map(|node| node.observers.as_slice())
    }

    /// Returns `true` if the source graph contains a directed cycle.
    ///
    /// A well-formed reactive graph is acyclic; this walks the `sources` edges
    /// with a depth-first three-colour search to prove it.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let mut visited = BTreeSet::new();
        let mut on_stack = BTreeSet::new();
        for node in &self.nodes {
            if !visited.contains(&node.id)
                && self.detect_cycle(node.id, &mut visited, &mut on_stack)
            {
                return true;
            }
        }
        false
    }

    /// Filters nodes by kind, preserving [`NodeId`] order.
    fn of_kind(&self, kind: NodeKindInfo) -> Vec<&NodeInfo> {
        self.nodes.iter().filter(|node| node.kind == kind).collect()
    }

    /// Depth-first cycle probe over the `sources` edges.
    fn detect_cycle(
        &self,
        id: NodeId,
        visited: &mut BTreeSet<NodeId>,
        on_stack: &mut BTreeSet<NodeId>,
    ) -> bool {
        visited.insert(id);
        on_stack.insert(id);
        if let Some(node) = self.get(id) {
            for &source in &node.sources {
                if !visited.contains(&source) {
                    if self.detect_cycle(source, visited, on_stack) {
                        return true;
                    }
                } else if on_stack.contains(&source) {
                    return true;
                }
            }
        }
        on_stack.remove(&id);
        false
    }
}

/// Maps an internal [`NodeKind`] to its public [`NodeKindInfo`].
fn map_kind(kind: NodeKind) -> NodeKindInfo {
    match kind {
        NodeKind::Signal => NodeKindInfo::Signal,
        NodeKind::Memo => NodeKindInfo::Memo,
        NodeKind::Effect => NodeKindInfo::Effect,
    }
}

impl Runtime {
    /// Captures a deterministic, owned snapshot of the current dependency graph.
    ///
    /// The snapshot is a read-only view: nodes are ordered by [`NodeId`] and
    /// every edge list is sorted ascending. It never drives propagation or
    /// mutates the runtime.
    ///
    /// # Example
    ///
    /// ```
    /// use prism_ui_reactive::Runtime;
    ///
    /// let rt = Runtime::new();
    /// let a = rt.signal(1i32);
    /// let doubled = rt.memo({
    ///     let a = a.clone();
    ///     move || a.get() * 2
    /// });
    /// let _ = doubled.get();
    ///
    /// let graph = rt.graph_snapshot();
    /// assert_eq!(graph.node_count(), 2);
    /// assert_eq!(graph.edge_count(), 1);
    /// assert!(!graph.has_cycle());
    /// ```
    #[must_use]
    pub fn graph_snapshot(&self) -> GraphSnapshot {
        let inner = self.0.borrow();
        let mut nodes = Vec::new();
        for (id, slot) in inner.node_slots().iter().enumerate() {
            let Some(node) = slot else {
                continue;
            };
            let mut sources = node.sources.clone();
            sources.sort_unstable();
            let mut observers = node.observers.clone();
            observers.sort_unstable();
            nodes.push(NodeInfo {
                id,
                kind: map_kind(node.kind),
                sources,
                observers,
            });
        }
        GraphSnapshot { nodes }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use crate::introspect::{NodeInfo, NodeKindInfo};
    use crate::Runtime;

    #[test]
    fn empty_runtime_has_empty_graph() {
        let rt = Runtime::new();
        let graph = rt.graph_snapshot();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(!graph.has_cycle());
    }

    #[test]
    fn single_signal_has_no_edges() {
        let rt = Runtime::new();
        let _a = rt.signal(1i32);
        let graph = rt.graph_snapshot();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.signals().len(), 1);
        assert_eq!(graph.memos().len(), 0);
        assert_eq!(graph.effects().len(), 0);
        assert_eq!(graph.nodes[0].kind, NodeKindInfo::Signal);
    }

    #[test]
    fn signal_memo_effect_chain_shape() {
        let rt = Runtime::new();
        let a = rt.signal(2i32);
        let doubled = rt.memo({
            let a = a.clone();
            move || a.get() * 2
        });
        let seen: alloc::rc::Rc<core::cell::RefCell<Vec<i32>>> =
            alloc::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
        let _effect = rt.effect({
            let doubled = doubled.clone();
            let seen = seen.clone();
            move || seen.borrow_mut().push(doubled.get())
        });

        let graph = rt.graph_snapshot();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.signals().len(), 1);
        assert_eq!(graph.memos().len(), 1);
        assert_eq!(graph.effects().len(), 1);

        let signal_id = graph.signals()[0].id;
        let memo_id = graph.memos()[0].id;
        let effect_id = graph.effects()[0].id;

        // signal <- memo <- effect
        assert_eq!(graph.sources_of(memo_id), Some([signal_id].as_slice()));
        assert_eq!(graph.sources_of(effect_id), Some([memo_id].as_slice()));
        assert_eq!(graph.observers_of(signal_id), Some([memo_id].as_slice()));
        assert_eq!(graph.observers_of(memo_id), Some([effect_id].as_slice()));
        assert_eq!(graph.sources_of(signal_id), Some([].as_slice()));
        assert_eq!(graph.observers_of(effect_id), Some([].as_slice()));
    }

    #[test]
    fn edge_count_counts_all_dependencies() {
        let rt = Runtime::new();
        let a = rt.signal(1i32);
        let b = rt.signal(2i32);
        let sum = rt.memo({
            let a = a.clone();
            let b = b.clone();
            move || a.get() + b.get()
        });
        let _ = sum.get();
        let graph = rt.graph_snapshot();
        // memo depends on both signals -> 2 edges.
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn chain_has_no_cycle() {
        let rt = Runtime::new();
        let a = rt.signal(1i32);
        let m = rt.memo({
            let a = a.clone();
            move || a.get() + 1
        });
        let _ = m.get();
        assert!(!rt.graph_snapshot().has_cycle());
    }

    #[test]
    fn get_returns_matching_node() {
        let rt = Runtime::new();
        let a = rt.signal(7i32);
        let graph = rt.graph_snapshot();
        let id = graph.signals()[0].id;
        let info = graph.get(id).unwrap();
        assert_eq!(info.kind, NodeKindInfo::Signal);
        assert_eq!(info.id, id);
        assert!(graph.get(9999).is_none());
        // Keep the signal alive until after inspection.
        let _ = a;
    }

    #[test]
    fn fan_out_observers_sorted() {
        let rt = Runtime::new();
        let a = rt.signal(1i32);
        let m1 = rt.memo({
            let a = a.clone();
            move || a.get() + 1
        });
        let m2 = rt.memo({
            let a = a.clone();
            move || a.get() + 2
        });
        let _ = m1.get();
        let _ = m2.get();
        let graph = rt.graph_snapshot();
        let signal_id = graph.signals()[0].id;
        let observers = graph.observers_of(signal_id).unwrap();
        assert_eq!(observers.len(), 2);
        // Sorted ascending for determinism.
        assert!(observers.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn snapshot_is_owned_and_detached() {
        let rt = Runtime::new();
        let a = rt.signal(1i32);
        let graph = rt.graph_snapshot();
        // Dropping the runtime handle must not affect the owned snapshot.
        drop(rt);
        drop(a);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn node_info_equality() {
        let left = NodeInfo {
            id: 0,
            kind: NodeKindInfo::Signal,
            sources: Vec::new(),
            observers: alloc::vec![1usize],
        };
        let right = left.clone();
        assert_eq!(left, right);
    }

    #[test]
    fn snapshot_reflects_disposed_slot_absence() {
        let rt = Runtime::new();
        let a = rt.signal(1i32);
        let m = rt.memo({
            let a = a.clone();
            move || a.get() + 1
        });
        let _ = m.get();
        assert_eq!(rt.graph_snapshot().node_count(), 2);
        // Disposing a memo frees its slot; the snapshot should drop that node.
        m.dispose();
        assert_eq!(rt.graph_snapshot().node_count(), 1);
        let _ = a;
    }
}
