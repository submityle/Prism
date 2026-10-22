//! Push reaction graph (design §10): the fine-grained, glitch-free reactive
//! kernel that turns "poll every frame" into "recompute only what changed".
//!
//! This is the data-truth / signal layer borrowed in *form* from Our Machinery's
//! "The Truth" and SolidJS's fine-grained reactivity (design §2, §10): a
//! directed acyclic graph of **source** nodes (raw change signals) and
//! **derived** nodes (recompute callbacks) connected by dependency edges. When a
//! source changes you [`mark_dirty`](ReactionGraph::mark_dirty) it; the dirt
//! propagates transitively to every dependent, and a single
//! [`flush`](ReactionGraph::flush) recomputes each dirty derived node **exactly
//! once**, in dependency order (every upstream dependency is recomputed before
//! the node that reads it).
//!
//! # Why topological order (glitch-free)
//!
//! Consider a diamond `A → {B, C} → D`. If `D` recomputed as soon as *either*
//! `B` or `C` updated, it would observe a half-updated world — a *glitch* — and
//! might recompute twice. By ordering the flush so that both `B` and `C` run
//! before `D`, `D` recomputes once, reading fully consistent inputs. This is
//! the structural guarantee that makes "cost ∝ amount-of-change" hold for
//! *derived* computation, not just for the extraction path (design §10, §17).
//!
//! # Relationship to observers / hooks
//!
//! Observers and component hooks (design §12, [`crate::observer`]) are the
//! *imperative* reactive bus — they fire callbacks at structural-change sites.
//! The reaction graph is the *declarative* dataflow layer that sits beside
//! them. [`World::reaction_source`](crate::world::World::reaction_source) binds
//! component observations to sources, and the schedule settles their effects
//! after collecting all source batches. Explicit graph users can also mark
//! sources and flush with their own context. The graph is `World`-independent so it
//! can be unit-tested in isolation and reused for UI signals
//! (`prism_ui_reactive`), spatial-index invalidation, and GPU dirty tracking.
//!
//! # Context
//!
//! Recompute callbacks are `FnMut(&mut C)` for a caller-chosen context `C`.
//! The context is the recompute's read/write surface — in the engine it is a
//! façade over the `World` (or a scratch cache); in tests it is a plain struct.
//! The graph never inspects `C`; it only guarantees *when* and *how often* each
//! callback runs.
//!
//! Dirt propagation maintains each pending node's count of dirty dependencies.
//! A non-empty flush sorts only the `d` dirty node indices, seeds Kahn ordering
//! from those counts, and walks outgoing edges of dirty nodes; it never scans
//! all incoming edges of a dirty node. Propagation plus ordering is
//! `O(d log d + e_d)`, where `e_d` counts outgoing adjacency entries reached
//! from the dirty subgraph. A clean frame is O(1). Callbacks are `Send + Sync`,
//! allowing an owning World to cross task lanes.

use alloc::boxed::Box;
use alloc::collections::{BTreeSet, VecDeque};
use alloc::vec::Vec;

/// Opaque, stable handle to a node in a [`ReactionGraph`].
///
/// Returned by [`ReactionGraph::add_source`] / [`ReactionGraph::add_derived`]
/// and accepted by the mutation and query methods. Handles are dense indices
/// and remain valid for the lifetime of the graph (nodes are never removed).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(usize);

impl NodeId {
    /// The raw dense index of this node within its owning graph.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// A derived node's recompute callback: an effectful closure over the
/// caller-chosen context `C`.
type Recompute<C> = Box<dyn FnMut(&mut C) + Send + Sync>;

/// One graph node. Private: constructed only through the graph's builder API.
struct Node<C> {
    /// Upstream nodes this node reads from (its dependencies). A node is
    /// recomputed only after every entry here has been recomputed.
    deps: Vec<usize>,
    /// Downstream nodes that read from this node (its dependents). Dirt flows
    /// along these edges.
    dependents: Vec<usize>,
    /// Whether this node must be recomputed on the next [`ReactionGraph::flush`].
    dirty: bool,
    /// Number of dependencies that are currently dirty. Maintained as dirt
    /// propagates and nodes settle, so ordering never scans clean incoming
    /// edges of a high-indegree dirty node.
    dirty_indegree: usize,
    /// Position in the current flush's sorted dirty-node array. Meaningful
    /// only while `dirty` is true; the dirty bit is the membership marker.
    order_position: usize,
    /// Recompute callback, or `None` for a pure source signal (sources carry a
    /// dirty bit but have nothing to recompute).
    recompute: Option<Recompute<C>>,
}

/// A push-driven, glitch-free reactive dependency graph (design §10).
///
/// Build the graph once with [`add_source`](Self::add_source) /
/// [`add_derived`](Self::add_derived), then each frame (or stage) push changes
/// with [`mark_dirty`](Self::mark_dirty) and settle them with
/// [`flush`](Self::flush). The flush recomputes every transitively-dirtied
/// derived node exactly once in dependency order.
///
/// The graph is generic over the recompute context `C` so it stays decoupled
/// from `World`; the engine passes a `World` façade, tests pass a scratch
/// struct.
pub struct ReactionGraph<C> {
    /// Dense node storage; indices are [`NodeId`]s.
    nodes: Vec<Node<C>>,
    /// Only dirty nodes; clean frames never scan the node array.
    pending: Vec<usize>,
    /// Deterministic work counters for sparse-flush complexity regressions.
    #[cfg(test)]
    stats: ReactionStats,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReactionStats {
    pub(crate) ordering_nodes: usize,
    /// Dirty-subgraph dependent adjacency entries inspected during ordering.
    pub(crate) ordering_edges: usize,
    pub(crate) executed_nodes: usize,
}

impl<C> Default for ReactionGraph<C> {
    #[inline]
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            pending: Vec::new(),
            #[cfg(test)]
            stats: ReactionStats::default(),
        }
    }
}

impl<C> ReactionGraph<C> {
    #[cfg(test)]
    pub(crate) fn stats(&self) -> ReactionStats {
        self.stats
    }

    /// Create an empty reaction graph.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of nodes (sources + derived) in the graph.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of dirty source and derived nodes awaiting the next flush.
    #[inline]
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.pending.len()
    }

    /// Add a **source** node: a raw change signal with no recompute logic.
    ///
    /// Mark it dirty with [`mark_dirty`](Self::mark_dirty) whenever its backing
    /// datum changes; the dirt then propagates to every derived dependent on
    /// the next [`flush`](Self::flush).
    pub fn add_source(&mut self) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            deps: Vec::new(),
            dependents: Vec::new(),
            dirty: false,
            dirty_indegree: 0,
            order_position: usize::MAX,
            recompute: None,
        });
        NodeId(id)
    }

    /// Add a **derived** node that reads from `deps` and recomputes by calling
    /// `recompute(&mut C)`.
    ///
    /// Dependency edges are deduplicated, so listing the same dependency twice
    /// is harmless. The node starts clean; call [`mark_dirty`](Self::mark_dirty)
    /// on it (or on any upstream dependency) to schedule its first recompute.
    pub fn add_derived<F>(&mut self, deps: &[NodeId], recompute: F) -> NodeId
    where
        F: FnMut(&mut C) + Send + Sync + 'static,
    {
        assert!(
            deps.iter().all(|dep| dep.0 < self.nodes.len()),
            "unknown reaction dependency"
        );
        let id = self.nodes.len();
        self.nodes.push(Node {
            deps: Vec::new(),
            dependents: Vec::new(),
            dirty: false,
            dirty_indegree: 0,
            order_position: usize::MAX,
            recompute: Some(Box::new(recompute)),
        });
        for &dep in deps {
            self.add_edge(dep.0, id);
        }
        NodeId(id)
    }

    /// Declare that `node` additionally depends on `depends_on` (edge
    /// `depends_on → node`). Idempotent for an edge that already exists.
    ///
    /// # Panics
    /// Panics if either node is unknown or the edge would create a dependency
    /// cycle. Rejection leaves the graph unchanged.
    #[inline]
    pub fn add_dependency(&mut self, node: NodeId, depends_on: NodeId) {
        self.add_edge(depends_on.0, node.0);
    }

    /// Insert the directed edge `dep → node` (dep is recomputed before node),
    /// deduplicating so in-degree accounting stays correct.
    fn add_edge(&mut self, dep: usize, node: usize) {
        assert!(
            dep < self.nodes.len() && node < self.nodes.len(),
            "unknown reaction node"
        );
        if self.nodes[node].deps.contains(&dep) {
            return;
        }
        assert!(
            !self.edge_would_create_cycle(dep, node),
            "reaction dependency would create a cycle"
        );
        self.nodes[node].deps.push(dep);
        self.nodes[dep].dependents.push(node);
        // Adding an edge after a source was dirtied must not lose the new
        // dependent merely because the source is already in the pending set.
        if self.nodes[dep].dirty {
            if self.nodes[node].dirty {
                self.nodes[node].dirty_indegree += 1;
            } else {
                self.mark_dirty(NodeId(node));
            }
        }
    }

    /// Whether adding `dep → node` would close a path back to `dep`.
    /// This runs only while constructing the graph, never during a flush. The
    /// visited set bounds work to the affected reachable subgraph even when
    /// multiple dependency paths converge.
    fn edge_would_create_cycle(&self, dep: usize, node: usize) -> bool {
        let mut stack = alloc::vec![node];
        let mut visited = BTreeSet::new();
        while let Some(current) = stack.pop() {
            if current == dep {
                return true;
            }
            if !visited.insert(current) {
                continue;
            }
            stack.extend(self.nodes[current].dependents.iter().copied());
        }
        false
    }

    /// Whether `node` is currently scheduled to recompute on the next flush.
    #[inline]
    #[must_use]
    pub fn is_dirty(&self, node: NodeId) -> bool {
        self.nodes[node.0].dirty
    }

    /// Mark `node` and **every transitive dependent** dirty.
    ///
    /// This is the "push": one changed source dirties exactly the sub-graph
    /// that reads from it, and nothing else — the structural root of
    /// "cost ∝ amount-of-change". An already-dirty node is never revisited,
    /// so converging dependency paths are traversed only once.
    pub fn mark_dirty(&mut self, node: NodeId) {
        if self.nodes[node.0].dirty {
            return;
        }
        // `node` may be explicitly dirtied while one or more upstream nodes
        // are already pending. Count those edges once at the root; all
        // subsequently dirtied nodes receive their count from propagation.
        self.nodes[node.0].dirty_indegree = self.nodes[node.0]
            .deps
            .iter()
            .filter(|&&dependency| self.nodes[dependency].dirty)
            .count();
        let mut stack = Vec::new();
        stack.push(node.0);
        while let Some(i) = stack.pop() {
            if self.nodes[i].dirty {
                continue;
            }
            self.nodes[i].dirty = true;
            self.pending.push(i);
            for k in 0..self.nodes[i].dependents.len() {
                let d = self.nodes[i].dependents[k];
                self.nodes[d].dirty_indegree += 1;
                if !self.nodes[d].dirty {
                    stack.push(d);
                }
            }
        }
    }

    /// Recompute every dirty derived node exactly once, in dependency order,
    /// clearing dirty bits as it goes. Returns the number of **derived** nodes
    /// recomputed (source nodes are cleared but not counted).
    ///
    /// Because dirt was already propagated eagerly by
    /// [`mark_dirty`](Self::mark_dirty) and the walk is topological, every
    /// node's dependencies are settled before it runs (glitch-free) and no node
    /// runs more than once (dedup).
    pub fn flush(&mut self, ctx: &mut C) -> usize {
        self.flush_while(ctx, |_| true)
    }

    /// Stop before the next node if the context no longer belongs to this graph.
    /// Completed nodes stay clean; unvisited nodes remain available for retry.
    /// The World bridge uses this to stop after a callback replaces its World.
    pub(crate) fn flush_while(
        &mut self,
        ctx: &mut C,
        mut keep_running: impl FnMut(&C) -> bool,
    ) -> usize {
        if self.pending.is_empty() {
            return 0;
        }
        self.order_pending();
        let mut recomputed = 0;
        while let Some(&i) = self.pending.last() {
            if !keep_running(ctx) {
                break;
            }
            // Keep both the callback and its dirty entry until it succeeds.
            // An unwind therefore leaves this node and its downstream work
            // retryable instead of silently discarding the callback.
            if let Some(cb) = self.nodes[i].recompute.as_mut() {
                cb(ctx);
                recomputed += 1;
                #[cfg(test)]
                {
                    self.stats.executed_nodes += 1;
                }
            }
            self.nodes[i].dirty = false;
            for k in 0..self.nodes[i].dependents.len() {
                let dependent = self.nodes[i].dependents[k];
                if self.nodes[dependent].dirty {
                    debug_assert!(self.nodes[dependent].dirty_indegree > 0);
                    self.nodes[dependent].dirty_indegree -= 1;
                }
            }
            self.pending.pop();
        }
        recomputed
    }

    /// Order only the dirty subgraph. A sparse change must not trigger a scan
    /// of unrelated clean nodes merely because the graph was recently built
    /// or extended. Kahn's queue is seeded in node-index order and dependent
    /// edges retain insertion order, so delivery remains deterministic.
    /// The dirty bit is the O(1) membership test: every dirty node is present
    /// exactly once in the pending set until its callback succeeds.
    fn order_pending(&mut self) {
        let mut affected = core::mem::take(&mut self.pending);
        affected.sort_unstable();
        let count = affected.len();
        let mut indegree = alloc::vec![0usize; count];
        for (position, &node) in affected.iter().enumerate() {
            self.nodes[node].order_position = position;
            indegree[position] = self.nodes[node].dirty_indegree;
        }
        #[cfg(test)]
        {
            self.stats.ordering_nodes += count;
        }
        let mut queue = VecDeque::new();
        for (position, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(position);
            }
        }
        let mut ordered = Vec::with_capacity(count);
        let mut emitted = alloc::vec![false; count];
        while let Some(position) = queue.pop_front() {
            let node = affected[position];
            emitted[position] = true;
            ordered.push(node);
            for &dependent in &self.nodes[node].dependents {
                #[cfg(test)]
                {
                    self.stats.ordering_edges += 1;
                }
                if self.nodes[dependent].dirty {
                    // The dirty/pending invariant also holds while pending is
                    // temporarily owned by this ordering pass.
                    let dependent_position = self.nodes[dependent].order_position;
                    debug_assert!(dependent_position < count);
                    if indegree[dependent_position] > 0 {
                        indegree[dependent_position] -= 1;
                        if indegree[dependent_position] == 0 {
                            queue.push_back(dependent_position);
                        }
                    }
                }
            }
        }
        debug_assert!(
            emitted.iter().all(|was_emitted| *was_emitted),
            "reaction graph must remain acyclic"
        );
        // flush_while pops from the end; reverse to execute dependency-first.
        ordered.reverse();
        self.pending = ordered;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn assert_dirty_pending_invariant<C>(graph: &ReactionGraph<C>) {
        let mut pending = alloc::vec![false; graph.nodes.len()];
        for &node in &graph.pending {
            assert!(graph.nodes[node].dirty, "pending node must be dirty");
            assert!(!pending[node], "pending nodes must be unique");
            pending[node] = true;
        }
        for (index, node) in graph.nodes.iter().enumerate() {
            assert_eq!(node.dirty, pending[index], "node {index}");
            assert_eq!(
                node.dirty_indegree,
                node.deps
                    .iter()
                    .filter(|&&dependency| graph.nodes[dependency].dirty)
                    .count(),
                "dirty indegree for node {index}"
            );
        }
    }

    /// Scratch recompute context recording per-node values and the order /
    /// count of recomputes, so tests can assert glitch-freedom and dedup.
    #[derive(Default)]
    struct Recorder {
        /// Source value feeding the diamond.
        a: i32,
        /// `B = A + 1`.
        b: i32,
        /// `C = A + 2`.
        c: i32,
        /// `D = B + C`.
        d: i32,
        /// Number of times the `D` node recomputed (must be 1 per settle).
        d_count: u32,
        /// Recompute order log of node labels, in flush order.
        order: Vec<u32>,
    }

    #[test]
    fn diamond_is_glitch_free_and_dedups() {
        let mut g = ReactionGraph::<Recorder>::new();
        let a = g.add_source();
        let b = g.add_derived(&[a], |r: &mut Recorder| {
            r.b = r.a + 1;
            r.order.push(1);
        });
        let c = g.add_derived(&[a], |r: &mut Recorder| {
            r.c = r.a + 2;
            r.order.push(2);
        });
        let _d = g.add_derived(&[b, c], |r: &mut Recorder| {
            r.d = r.b + r.c;
            r.d_count += 1;
            r.order.push(3);
        });

        let mut ctx = Recorder {
            a: 10,
            ..Recorder::default()
        };
        g.mark_dirty(a);
        let recomputed = g.flush(&mut ctx);

        // All three derived nodes recomputed, exactly once each.
        assert_eq!(recomputed, 3);
        assert_eq!(ctx.d_count, 1, "D must recompute exactly once (no glitch)");
        // Fully consistent inputs: D saw the updated B and C.
        assert_eq!(ctx.b, 11);
        assert_eq!(ctx.c, 12);
        assert_eq!(ctx.d, 23);
        // Dependency order: both B(1) and C(2) run before D(3).
        let pos_b = ctx.order.iter().position(|&x| x == 1).unwrap();
        let pos_c = ctx.order.iter().position(|&x| x == 2).unwrap();
        let pos_d = ctx.order.iter().position(|&x| x == 3).unwrap();
        assert!(pos_b < pos_d && pos_c < pos_d);
        assert_eq!(ctx.order.iter().filter(|&&x| x == 3).count(), 1);
    }

    #[test]
    fn repeated_mark_dirty_still_recomputes_once() {
        let mut g = ReactionGraph::<Recorder>::new();
        let a = g.add_source();
        let _b = g.add_derived(&[a], |r: &mut Recorder| {
            r.b += 1;
            r.order.push(1);
        });
        let mut ctx = Recorder::default();
        g.mark_dirty(a);
        g.mark_dirty(a);
        g.mark_dirty(a);
        let recomputed = g.flush(&mut ctx);
        assert_eq!(recomputed, 1);
        assert_eq!(ctx.b, 1, "coalesced: three marks, one recompute");
    }

    #[test]
    fn mark_dirty_propagates_only_downstream() {
        // s1 → d1,  s2 → d2 : changing s1 must not recompute d2.
        let mut g = ReactionGraph::<Recorder>::new();
        let s1 = g.add_source();
        let s2 = g.add_source();
        let d1 = g.add_derived(&[s1], |r: &mut Recorder| r.order.push(1));
        let d2 = g.add_derived(&[s2], |r: &mut Recorder| r.order.push(2));

        g.mark_dirty(s1);
        assert!(g.is_dirty(s1));
        assert!(g.is_dirty(d1));
        assert!(!g.is_dirty(s2));
        assert!(!g.is_dirty(d2));

        let mut ctx = Recorder::default();
        let recomputed = g.flush(&mut ctx);
        assert_eq!(recomputed, 1, "cost proportional to change: only d1 ran");
        assert_eq!(ctx.order, alloc::vec![1]);
        assert!(!g.is_dirty(d1), "flush clears dirty bits");
    }

    #[test]
    fn sparse_flush_does_not_scan_clean_incoming_edges() {
        const INDEGREE: usize = 4096;
        let mut graph = ReactionGraph::<Recorder>::new();
        let sources: Vec<_> = (0..INDEGREE).map(|_| graph.add_source()).collect();
        let effect_calls = alloc::sync::Arc::new(core::sync::atomic::AtomicUsize::new(0));
        let effect = {
            let effect_calls = effect_calls.clone();
            graph.add_derived(&sources, move |ctx| {
                effect_calls.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                ctx.order.push(1);
            })
        };
        let tail = graph.add_derived(&[effect], |ctx| ctx.order.push(2));

        graph.mark_dirty(sources[INDEGREE / 2]);
        assert_eq!(graph.nodes[effect.index()].dirty_indegree, 1);
        assert_eq!(graph.nodes[tail.index()].dirty_indegree, 1);
        assert_dirty_pending_invariant(&graph);

        let mut context = Recorder::default();
        assert_eq!(graph.flush(&mut context), 2);
        assert_eq!(context.order, alloc::vec![1, 2]);
        assert_eq!(effect_calls.load(core::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(graph.stats.ordering_nodes, 3);
        assert_eq!(
            graph.stats.ordering_edges, 2,
            "only dirty outgoing edges participate in Kahn ordering"
        );
        assert_dirty_pending_invariant(&graph);
    }

    #[test]
    fn transitive_chain_dirties_whole_tail() {
        // a → b → c → d
        let mut g = ReactionGraph::<Recorder>::new();
        let a = g.add_source();
        let b = g.add_derived(&[a], |r: &mut Recorder| r.order.push(1));
        let c = g.add_derived(&[b], |r: &mut Recorder| r.order.push(2));
        let d = g.add_derived(&[c], |r: &mut Recorder| r.order.push(3));
        g.mark_dirty(a);
        assert!(g.is_dirty(b) && g.is_dirty(c) && g.is_dirty(d));
        let mut ctx = Recorder::default();
        assert_eq!(g.flush(&mut ctx), 3);
        assert_eq!(ctx.order, alloc::vec![1, 2, 3]);
    }

    #[test]
    fn cycle_edge_is_rejected_without_mutating_the_dag() {
        let mut g = ReactionGraph::<Recorder>::new();
        let n0 = g.add_derived(&[], |r: &mut Recorder| {
            r.d_count += 1;
            r.order.push(0);
        });
        let n1 = g.add_derived(&[n0], |r: &mut Recorder| {
            r.d_count += 1;
            r.order.push(1);
        });
        let n2 = g.add_derived(&[n1], |r: &mut Recorder| {
            r.d_count += 1;
            r.order.push(2);
        });

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            g.add_dependency(n0, n2); // n2 → n0 would close the cycle
        }))
        .is_err());
        assert!(g.nodes[n0.index()].deps.is_empty());
        assert_eq!(g.nodes[n1.index()].deps, alloc::vec![n0.index()]);
        assert_eq!(g.nodes[n2.index()].deps, alloc::vec![n1.index()]);

        g.mark_dirty(n0);
        assert_dirty_pending_invariant(&g);
        let mut ctx = Recorder::default();
        let recomputed = g.flush(&mut ctx);
        assert_eq!(recomputed, 3);
        assert_eq!(ctx.d_count, 3);
        assert_eq!(ctx.order, alloc::vec![0, 1, 2]);
        assert_dirty_pending_invariant(&g);
    }

    #[test]
    fn clean_graph_flush_is_a_no_op() {
        let mut g = ReactionGraph::<Recorder>::new();
        let a = g.add_source();
        let _b = g.add_derived(&[a], |r: &mut Recorder| r.b += 1);
        let mut ctx = Recorder::default();
        assert_eq!(g.flush(&mut ctx), 0);
        assert_eq!(ctx.b, 0);
    }

    #[test]
    fn interrupted_flush_counts_completed_effects_and_retains_unvisited_work() {
        let mut graph = ReactionGraph::<Recorder>::new();
        let source = graph.add_source();
        let first = graph.add_derived(&[source], |ctx| ctx.order.push(1));
        let second = graph.add_derived(&[first], |ctx| ctx.order.push(2));
        graph.mark_dirty(source);
        assert_dirty_pending_invariant(&graph);
        let mut ctx = Recorder::default();
        assert_eq!(graph.flush_while(&mut ctx, |_| false), 0);
        assert_dirty_pending_invariant(&graph);
        assert_eq!(graph.dirty_count(), 3);
        assert_eq!(graph.flush_while(&mut ctx, |ctx| ctx.order.is_empty()), 1);
        assert_dirty_pending_invariant(&graph);
        assert_eq!(ctx.order, alloc::vec![1]);
        assert!(!graph.is_dirty(source));
        assert!(!graph.is_dirty(first));
        assert!(graph.is_dirty(second));
        assert_eq!(graph.dirty_count(), 1);
        assert_eq!(graph.flush(&mut ctx), 1);
        assert_dirty_pending_invariant(&graph);
        assert_eq!(ctx.order, alloc::vec![1, 2]);
        assert_eq!(graph.flush(&mut ctx), 0);
    }

    #[test]
    fn edges_are_deduplicated() {
        let mut g = ReactionGraph::<Recorder>::new();
        let a = g.add_source();
        // Same dependency listed twice must not corrupt in-degree accounting
        // (which would otherwise deadlock Kahn's algorithm and drop the node).
        let _b = g.add_derived(&[a, a], |r: &mut Recorder| r.order.push(1));
        g.mark_dirty(a);
        let mut ctx = Recorder::default();
        assert_eq!(g.flush(&mut ctx), 1);
        assert_eq!(ctx.order, alloc::vec![1]);
    }

    #[test]
    fn a_dependency_added_after_marking_includes_its_new_dependent() {
        let mut graph = ReactionGraph::<Recorder>::new();
        let source = graph.add_source();
        let derived = graph.add_derived(&[], |ctx| ctx.order.push(1));
        graph.mark_dirty(source);
        graph.add_dependency(derived, source);
        assert_dirty_pending_invariant(&graph);
        assert_eq!(graph.dirty_count(), 2);
        let mut ctx = Recorder::default();
        assert_eq!(graph.flush(&mut ctx), 1);
        assert_eq!(ctx.order, alloc::vec![1]);
        assert_eq!(graph.dirty_count(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn invalid_dependencies_leave_the_existing_graph_unchanged() {
        let mut graph = ReactionGraph::<Recorder>::new();
        let source = graph.add_source();
        let invalid = NodeId(20);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.add_derived(&[source, invalid], |_| {});
        }))
        .is_err());
        assert_eq!(graph.len(), 1);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.add_dependency(source, invalid);
        }))
        .is_err());
        graph.mark_dirty(source);
        assert_eq!(graph.flush(&mut Recorder::default()), 0);
        assert_eq!(graph.dirty_count(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn panic_preserves_the_callback_without_replaying_completed_nodes() {
        let mut graph = ReactionGraph::<Recorder>::new();
        let source = graph.add_source();
        let before = graph.add_derived(&[source], |ctx| ctx.order.push(1));
        let mut fail_once = true;
        let failed = graph.add_derived(&[before], move |ctx| {
            assert!(!core::mem::take(&mut fail_once), "effect failed");
            ctx.order.push(2);
        });
        graph.add_derived(&[failed], |ctx| ctx.order.push(3));
        let mut ctx = Recorder::default();
        graph.mark_dirty(source);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            graph.flush(&mut ctx);
        }))
        .is_err());
        assert_dirty_pending_invariant(&graph);
        assert_eq!(ctx.order, alloc::vec![1]);
        assert_eq!(graph.dirty_count(), 2);
        assert_eq!(graph.flush(&mut ctx), 2);
        assert_dirty_pending_invariant(&graph);
        assert_eq!(ctx.order, alloc::vec![1, 2, 3]);
        assert_eq!(graph.flush(&mut ctx), 0);
    }
}
