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
//! them: a system (or an observer) marks source nodes dirty as components are
//! written, then flushes the graph once per stage to settle all derived values
//! with no redundant work. The graph is deliberately `World`-independent so it
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

use alloc::boxed::Box;
use alloc::collections::VecDeque;
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
type Recompute<C> = Box<dyn FnMut(&mut C)>;

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
    /// Cached topological order of *all* node indices (dependencies first).
    topo: Vec<usize>,
    /// Whether [`topo`](Self::topo) is up to date with the current edge set.
    topo_valid: bool,
}

impl<C> Default for ReactionGraph<C> {
    #[inline]
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            topo: Vec::new(),
            topo_valid: true,
        }
    }
}

impl<C> ReactionGraph<C> {
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
            recompute: None,
        });
        // A fresh isolated node must still appear in the flush order.
        self.topo_valid = false;
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
        F: FnMut(&mut C) + 'static,
    {
        let id = self.nodes.len();
        self.nodes.push(Node {
            deps: Vec::new(),
            dependents: Vec::new(),
            dirty: false,
            recompute: Some(Box::new(recompute)),
        });
        for &dep in deps {
            self.add_edge(dep.0, id);
        }
        self.topo_valid = false;
        NodeId(id)
    }

    /// Declare that `node` additionally depends on `depends_on` (edge
    /// `depends_on → node`). Idempotent for an edge that already exists.
    #[inline]
    pub fn add_dependency(&mut self, node: NodeId, depends_on: NodeId) {
        self.add_edge(depends_on.0, node.0);
    }

    /// Insert the directed edge `dep → node` (dep is recomputed before node),
    /// deduplicating so in-degree accounting stays correct.
    fn add_edge(&mut self, dep: usize, node: usize) {
        if self.nodes[node].deps.contains(&dep) {
            return;
        }
        self.nodes[node].deps.push(dep);
        self.nodes[dep].dependents.push(node);
        self.topo_valid = false;
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
    /// "cost ∝ amount-of-change". The walk is cycle-safe: an already-dirty node
    /// is never revisited, so dependency cycles terminate.
    pub fn mark_dirty(&mut self, node: NodeId) {
        let mut stack = Vec::new();
        stack.push(node.0);
        while let Some(i) = stack.pop() {
            if self.nodes[i].dirty {
                continue;
            }
            self.nodes[i].dirty = true;
            for k in 0..self.nodes[i].dependents.len() {
                let d = self.nodes[i].dependents[k];
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
        self.ensure_topo();
        let mut recomputed = 0;
        for idx in 0..self.topo.len() {
            let i = self.topo[idx];
            if !self.nodes[i].dirty {
                continue;
            }
            self.nodes[i].dirty = false;
            // Take the callback out so the call cannot alias the node slot; a
            // recompute only touches `ctx`, never the graph, but this keeps the
            // borrow trivially sound and re-entrancy-safe.
            if let Some(mut cb) = self.nodes[i].recompute.take() {
                cb(ctx);
                self.nodes[i].recompute = Some(cb);
                recomputed += 1;
            }
        }
        recomputed
    }

    /// Rebuild [`topo`](Self::topo) via Kahn's algorithm if the edge set
    /// changed since the last build.
    ///
    /// Nodes are emitted in dependency order (every dependency precedes its
    /// dependents). Any nodes left inside a dependency cycle (never reaching
    /// in-degree zero) are appended afterwards in index order, so a flush over
    /// a cyclic graph still terminates and still visits each node at most once.
    fn ensure_topo(&mut self) {
        if self.topo_valid {
            return;
        }
        let n = self.nodes.len();
        let mut indegree = Vec::with_capacity(n);
        for node in &self.nodes {
            indegree.push(node.deps.len());
        }
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, &deg) in indegree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }
        let mut topo = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for k in 0..self.nodes[u].dependents.len() {
                let v = self.nodes[u].dependents[k];
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        if topo.len() < n {
            // Cycle fallback: append the remaining (cyclic) nodes deterministically.
            let mut seen = Vec::with_capacity(n);
            seen.resize(n, false);
            for &u in &topo {
                seen[u] = true;
            }
            for (i, &was_seen) in seen.iter().enumerate() {
                if !was_seen {
                    topo.push(i);
                }
            }
        }
        self.topo = topo;
        self.topo_valid = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

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
    fn cycle_is_safe_and_each_runs_once() {
        // n0 → n1 → n2 → n0 (a 3-cycle of derived nodes).
        let mut g = ReactionGraph::<Recorder>::new();
        let n0 = g.add_derived(&[], |r: &mut Recorder| r.d_count += 1);
        let n1 = g.add_derived(&[n0], |r: &mut Recorder| r.d_count += 1);
        let n2 = g.add_derived(&[n1], |r: &mut Recorder| r.d_count += 1);
        g.add_dependency(n0, n2); // close the cycle

        g.mark_dirty(n0); // must terminate despite the cycle
        assert!(g.is_dirty(n0) && g.is_dirty(n1) && g.is_dirty(n2));

        let mut ctx = Recorder::default();
        let recomputed = g.flush(&mut ctx); // must terminate
        assert_eq!(recomputed, 3, "every node in the cycle runs exactly once");
        assert_eq!(ctx.d_count, 3);
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
}
