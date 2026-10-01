//! Analysing a reactive dependency graph captured from a [`Runtime`].
//!
//! [`DependencyGraph`] consumes a [`GraphSnapshot`] produced by
//! [`prism_ui_reactive::Runtime::graph_snapshot`] and answers structural
//! questions about it: the transitive dependents and dependencies of a node, a
//! topological order (or `None` when the graph has a cycle), its roots
//! (dependency-free signals) and leaves (observer-free sinks, usually effects),
//! and renderings as Graphviz DOT or a human-readable indented tree.
//!
//! All results are deterministically ordered and computed with integer
//! arithmetic only, so output is stable across runs.
//!
//! [`Runtime`]: prism_ui_reactive::Runtime

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use prism_ui_reactive::{GraphSnapshot, NodeId, NodeKindInfo};

/// A node's role plus its edges, as retained by a [`DependencyGraph`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphNode {
    kind: NodeKindInfo,
    sources: Vec<NodeId>,
    observers: Vec<NodeId>,
}

/// An analysable, owned view of a reactive dependency graph.
///
/// Build one with [`DependencyGraph::from`]. Nodes are keyed by [`NodeId`] and
/// every stored edge list is sorted ascending for deterministic traversal.
///
/// # Example
///
/// ```
/// use prism_ui_inspector::DependencyGraph;
/// use prism_ui_reactive::Runtime;
///
/// let rt = Runtime::new();
/// let count = rt.signal(1i32);
/// let doubled = rt.memo({
///     let count = count.clone();
///     move || count.get() * 2
/// });
/// // Reading the memo wires it to the signal.
/// assert_eq!(doubled.get(), 2);
///
/// let graph = DependencyGraph::from(&rt.graph_snapshot());
/// let signal_id = rt.graph_snapshot().signals()[0].id;
/// let memo_id = rt.graph_snapshot().memos()[0].id;
///
/// assert_eq!(graph.roots(), vec![signal_id]);
/// assert_eq!(graph.dependents_of(signal_id), vec![memo_id]);
/// assert_eq!(graph.topo_order(), Some(vec![signal_id, memo_id]));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DependencyGraph {
    nodes: BTreeMap<NodeId, GraphNode>,
}

impl DependencyGraph {
    /// Builds a dependency graph from a reactive [`GraphSnapshot`].
    #[must_use]
    pub fn from(snapshot: &GraphSnapshot) -> Self {
        let mut nodes = BTreeMap::new();
        for info in &snapshot.nodes {
            let mut sources = info.sources.clone();
            sources.sort_unstable();
            let mut observers = info.observers.clone();
            observers.sort_unstable();
            nodes.insert(
                info.id,
                GraphNode {
                    kind: info.kind,
                    sources,
                    observers,
                },
            );
        }
        Self { nodes }
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when `id` is present in the graph.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Returns the kind of node `id`, if present.
    #[must_use]
    pub fn kind_of(&self, id: NodeId) -> Option<NodeKindInfo> {
        self.nodes.get(&id).map(|node| node.kind)
    }

    /// Returns every node transitively observing `id` (its dependents), sorted.
    ///
    /// These are the nodes reachable by following `observers` edges; `id` itself
    /// is excluded. Returns an empty vector when `id` is absent.
    #[must_use]
    pub fn dependents_of(&self, id: NodeId) -> Vec<NodeId> {
        self.transitive(id, Direction::Observers)
    }

    /// Returns every node `id` transitively depends on (its dependencies),
    /// sorted.
    ///
    /// These are the nodes reachable by following `sources` edges; `id` itself
    /// is excluded. Returns an empty vector when `id` is absent.
    #[must_use]
    pub fn dependencies_of(&self, id: NodeId) -> Vec<NodeId> {
        self.transitive(id, Direction::Sources)
    }

    /// Returns the dependency-free nodes (no sources), sorted.
    ///
    /// In a reactive graph these are the signals that feed everything else.
    #[must_use]
    pub fn roots(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.sources.is_empty())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns the observer-free nodes (no observers), sorted.
    ///
    /// These are the sinks of the graph, typically effects.
    #[must_use]
    pub fn leaves(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.observers.is_empty())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns a topological order (dependencies before dependents), or `None`
    /// when the graph contains a cycle.
    ///
    /// Ties are broken by ascending [`NodeId`], so the order is deterministic.
    #[must_use]
    pub fn topo_order(&self) -> Option<Vec<NodeId>> {
        let mut in_degree: BTreeMap<NodeId, usize> = BTreeMap::new();
        for (id, node) in &self.nodes {
            in_degree.insert(*id, node.sources.len());
        }
        let mut ready: BTreeSet<NodeId> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(&next) = ready.iter().next() {
            ready.remove(&next);
            order.push(next);
            if let Some(node) = self.nodes.get(&next) {
                for observer in &node.observers {
                    if let Some(degree) = in_degree.get_mut(observer) {
                        *degree -= 1;
                        if *degree == 0 {
                            ready.insert(*observer);
                        }
                    }
                }
            }
        }
        if order.len() == self.nodes.len() {
            Some(order)
        } else {
            None
        }
    }

    /// Renders the graph as Graphviz DOT text.
    ///
    /// Nodes are labelled with their id and kind; edges run from each source to
    /// its dependent. Output is deterministic.
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        out.push_str("digraph reactive {\n");
        for (id, node) in &self.nodes {
            let _ = writeln!(out, "  n{id} [label=\"#{id} {}\"];", kind_label(node.kind));
        }
        let mut edges: BTreeSet<(NodeId, NodeId)> = BTreeSet::new();
        for (id, node) in &self.nodes {
            for source in &node.sources {
                edges.insert((*source, *id));
            }
        }
        for (from, to) in &edges {
            let _ = writeln!(out, "  n{from} -> n{to};");
        }
        out.push_str("}\n");
        out
    }

    /// Renders the graph as a human-readable, indented dependency tree.
    ///
    /// Each root is printed and its dependents are listed recursively, indented
    /// by two spaces per level. A node revisited along the current path is
    /// marked `(cycle)` and not expanded further, so the output is always
    /// finite.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        let roots = self.roots();
        let starts = if roots.is_empty() {
            self.nodes.keys().copied().collect()
        } else {
            roots
        };
        for id in starts {
            let mut path = BTreeSet::new();
            self.render_node(id, 0, &mut path, &mut out);
        }
        out
    }

    /// Shared transitive-reachability walk in a chosen edge direction.
    fn transitive(&self, id: NodeId, direction: Direction) -> Vec<NodeId> {
        let mut visited = BTreeSet::new();
        let mut stack = Vec::new();
        if self.nodes.contains_key(&id) {
            stack.push(id);
        }
        while let Some(current) = stack.pop() {
            if let Some(node) = self.nodes.get(&current) {
                let neighbours = match direction {
                    Direction::Sources => &node.sources,
                    Direction::Observers => &node.observers,
                };
                for &next in neighbours {
                    if visited.insert(next) {
                        stack.push(next);
                    }
                }
            }
        }
        visited.remove(&id);
        visited.into_iter().collect()
    }

    /// Recursive helper backing [`DependencyGraph::render`].
    fn render_node(&self, id: NodeId, depth: usize, path: &mut BTreeSet<NodeId>, out: &mut String) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        let kind = self
            .nodes
            .get(&id)
            .map_or("?", |node| kind_label(node.kind));
        if !path.insert(id) {
            let _ = writeln!(out, "#{id} {kind} (cycle)");
            return;
        }
        let _ = writeln!(out, "#{id} {kind}");
        if let Some(node) = self.nodes.get(&id) {
            for observer in &node.observers {
                self.render_node(*observer, depth + 1, path, out);
            }
        }
        path.remove(&id);
    }
}

/// Which edge set a transitive walk should follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    /// Follow `sources` edges (dependencies).
    Sources,
    /// Follow `observers` edges (dependents).
    Observers,
}

/// Returns the DOT/text label for a node kind.
fn kind_label(kind: NodeKindInfo) -> &'static str {
    match kind {
        NodeKindInfo::Signal => "Signal",
        NodeKindInfo::Memo => "Memo",
        NodeKindInfo::Effect => "Effect",
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    extern crate alloc;

    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    use prism_ui_reactive::Runtime;

    use super::DependencyGraph;

    /// Builds a signal -> memo -> effect chain and returns the graph plus ids.
    fn chain() -> (DependencyGraph, usize, usize, usize) {
        let rt = Runtime::new();
        let a = rt.signal(1i32);
        let doubled = rt.memo({
            let a = a.clone();
            move || a.get() * 2
        });
        let log = Rc::new(RefCell::new(Vec::new()));
        let _effect = rt.effect({
            let doubled = doubled.clone();
            let log = log.clone();
            move || log.borrow_mut().push(doubled.get())
        });
        let snap = rt.graph_snapshot();
        let signal_id = snap.signals()[0].id;
        let memo_id = snap.memos()[0].id;
        let effect_id = snap.effects()[0].id;
        (DependencyGraph::from(&snap), signal_id, memo_id, effect_id)
    }

    #[test]
    fn from_snapshot_node_count() {
        let (graph, _, _, _) = chain();
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn dependents_of_signal_are_memo_and_effect() {
        let (graph, signal_id, memo_id, effect_id) = chain();
        let mut expected = alloc::vec![memo_id, effect_id];
        expected.sort_unstable();
        assert_eq!(graph.dependents_of(signal_id), expected);
    }

    #[test]
    fn dependents_of_effect_is_empty() {
        let (graph, _, _, effect_id) = chain();
        assert!(graph.dependents_of(effect_id).is_empty());
    }

    #[test]
    fn dependencies_of_effect_are_memo_and_signal() {
        let (graph, signal_id, memo_id, effect_id) = chain();
        let mut expected = alloc::vec![signal_id, memo_id];
        expected.sort_unstable();
        assert_eq!(graph.dependencies_of(effect_id), expected);
    }

    #[test]
    fn dependencies_of_signal_is_empty() {
        let (graph, signal_id, _, _) = chain();
        assert!(graph.dependencies_of(signal_id).is_empty());
    }

    #[test]
    fn roots_are_signals() {
        let (graph, signal_id, _, _) = chain();
        assert_eq!(graph.roots(), alloc::vec![signal_id]);
    }

    #[test]
    fn leaves_are_effects() {
        let (graph, _, _, effect_id) = chain();
        assert_eq!(graph.leaves(), alloc::vec![effect_id]);
    }

    #[test]
    fn topo_order_respects_dependencies() {
        let (graph, signal_id, memo_id, effect_id) = chain();
        let order = graph.topo_order().expect("acyclic");
        assert_eq!(order.len(), 3);
        let pos = |id| order.iter().position(|x| *x == id).unwrap();
        assert!(pos(signal_id) < pos(memo_id));
        assert!(pos(memo_id) < pos(effect_id));
    }

    #[test]
    fn topo_order_is_deterministic() {
        let (graph, _, _, _) = chain();
        assert_eq!(graph.topo_order(), graph.topo_order());
    }

    #[test]
    fn to_dot_contains_nodes_and_edges() {
        let (graph, signal_id, memo_id, _) = chain();
        let dot = graph.to_dot();
        assert!(dot.starts_with("digraph reactive {"));
        assert!(dot.contains("Signal"));
        assert!(dot.contains("Memo"));
        assert!(dot.contains("Effect"));
        assert!(dot.contains(&alloc::format!("n{signal_id} -> n{memo_id};")));
        assert!(dot.trim_end().ends_with('}'));
    }

    #[test]
    fn render_lists_chain_with_indentation() {
        let (graph, signal_id, memo_id, effect_id) = chain();
        let text = graph.render();
        assert!(text.contains(&alloc::format!("#{signal_id} Signal")));
        assert!(text.contains(&alloc::format!("  #{memo_id} Memo")));
        assert!(text.contains(&alloc::format!("    #{effect_id} Effect")));
    }

    #[test]
    fn diamond_dependencies() {
        let rt = Runtime::new();
        let a = rt.signal(1i32);
        let left = rt.memo({
            let a = a.clone();
            move || a.get() + 1
        });
        let right = rt.memo({
            let a = a.clone();
            move || a.get() + 2
        });
        let sum = rt.memo({
            let left = left.clone();
            let right = right.clone();
            move || left.get() + right.get()
        });
        let _ = sum.get();

        let snap = rt.graph_snapshot();
        let graph = DependencyGraph::from(&snap);
        assert_eq!(graph.node_count(), 4);

        let a_id = snap.signals()[0].id;
        // Everything downstream of the signal: both branches plus the sum.
        assert_eq!(graph.dependents_of(a_id).len(), 3);

        let order = graph.topo_order().expect("acyclic");
        let pos = |id| order.iter().position(|x| *x == id).unwrap();
        let sum_id = snap.memos().iter().map(|n| n.id).max().expect("has memos");
        // The join node comes last.
        assert_eq!(pos(sum_id), order.len() - 1);
    }

    #[test]
    fn empty_graph_renders_empty() {
        let rt = Runtime::new();
        let graph = DependencyGraph::from(&rt.graph_snapshot());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.topo_order(), Some(Vec::new()));
        assert!(graph.render().is_empty());
        assert_eq!(graph.to_dot(), "digraph reactive {\n}\n");
    }

    #[test]
    fn absent_node_queries_are_empty() {
        let (graph, _, _, _) = chain();
        assert!(graph.dependents_of(9999).is_empty());
        assert!(graph.dependencies_of(9999).is_empty());
        assert!(!graph.contains(9999));
    }
}
