//! Heterogeneous sub-job **dependency DAG** for one system's internal work
//! (design §8.3 "Fiber 作业图：原子计数依赖 + 工作窃取 + 近零同步点").
//!
//! Where [`JobGraph::par_for_each`](crate::system::job_graph::JobGraph::par_for_each)
//! and [`JobGraph::par_chunks`](crate::system::job_graph::JobGraph::par_chunks)
//! fan **one** query's rows out *homogeneously* (every sub-job runs the same
//! body over a different slice), a real DOOM / Naughty-Dog fiber job graph also
//! wants to split a heavy system into a handful of **named, heterogeneous**
//! sub-jobs wired by dependency edges — "run `broadphase` and `integrate` in
//! parallel, then `resolve` once both finish, and `emit_events` as soon as
//! `broadphase` alone is done". A plain phase barrier (nested
//! [`scope`](prism_tasks::TaskPool::scope)) can express that only by forcing a
//! full join between phases, which serialises independent chains: in the
//! diamond `A → {B, C} → D` with an extra `A → E`, phasing blocks `E` behind
//! the longer of `B`/`C` even though `E` depends on neither.
//!
//! [`JobDag`] removes those false sync points. Each node carries an atomic
//! *pending-predecessor* count; a node is dispatched onto the shared pool the
//! instant its last predecessor finishes (dataflow scheduling), so independent
//! sub-chains overlap freely and the only real synchronisation is each true
//! edge. This is the atomic-counter dependency form §8.3 cites.
//!
//! ```ignore
//! fn physics_step(mut q: Query<(&mut Vel, &Mass)>, jobs: JobGraph) {
//!     let bodies = SharedState::default();
//!     jobs.dag(|dag| {
//!         let broad = dag.add(|| bodies.build_broadphase());
//!         let integ = dag.add(|| bodies.integrate_forces());
//!         // `resolve` waits for BOTH `broad` and `integ`:
//!         let _res = dag.add_after(&[broad, integ], || bodies.resolve_contacts());
//!         // `events` waits only for `broad`, so it overlaps `integ`/`resolve`:
//!         let _ev = dag.add_after(&[broad], || bodies.emit_broadphase_events());
//!     }); // returns once every node has run
//! }
//! ```
//!
//! # Where the machinery lives (design §24.1)
//!
//! The kernel does **not** build its own thread pool or scheduler. [`JobDag`]
//! is a thin, *safe* orchestrator layered on `prism_tasks`'
//! [`TaskPool::scope`](prism_tasks::TaskPool::scope): the work-stealing
//! execution, help-on-wait join barrier, and borrowed-lifetime erasure all live
//! in `prism_tasks`. The ECS side only *declares the sub-job order* — exactly
//! the "访问集 + 顺序 + 车道标注" split §24.1 mandates — and contains no `unsafe`.
//!
//! # Semantics & guarantees
//!
//! * **Happens-before.** A node runs strictly after *all* its declared
//!   predecessors have returned, on some pool worker (or the calling thread,
//!   which participates while waiting). Producers hand values to consumers
//!   through `'env` interior mutability (a [`Mutex`], atomic, or one-shot cell);
//!   the edge supplies the ordering, as in [`TaskPool::scope`].
//! * **Fork-join.** [`dispatch`](JobDag::dispatch) (and the
//!   [`JobGraph::dag`](crate::system::job_graph::JobGraph::dag) entry point)
//!   returns only after every node has run. No node, and none of the `'env`
//!   borrows it captured, can outlive the call.
//! * **Panic propagation.** A panicking node is caught by the underlying scope
//!   and re-raised on the dispatching thread after the graph joins, so a sub-job
//!   failure never poisons the pool (mirrors
//!   [`prism_tasks::JobHandle`](prism_tasks::JobHandle)).
//! * **Determinism of *results*.** The graph is pure dataflow: outputs depend
//!   only on the declared edges, never on worker count or steal order. The
//!   single-threaded fallback runs a topological order inline.
//! * **Cycles.** A dependency cycle is a programming error and makes dataflow
//!   dispatch impossible; it is detected up front and panics with a clear
//!   message instead of deadlocking.
//!
//! Enabled by the `multi_thread` feature.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use prism_tasks::{Scope, TaskPool};

/// An opaque handle to a sub-job added to a [`JobDag`], used to declare that
/// later sub-jobs depend on it.
///
/// This is the ECS-side analogue of the `JobHandle` design §8.3 asks a system
/// to be able to name: it identifies a node so dependency edges can be drawn to
/// it. (`prism_tasks`' own [`JobHandle`](prism_tasks::JobHandle) carries a
/// *result* across a detached join; here nodes communicate through `'env`
/// interior mutability and the DAG edge, so a lightweight id is all that's
/// needed.)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct JobId(usize);

impl JobId {
    /// The node's dense index within its owning [`JobDag`] (0-based, in
    /// insertion order).
    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

/// One node: its (not-yet-run) body plus the predecessors it waits on.
struct Node<'env> {
    /// `Some` until the node is dispatched, then taken exactly once.
    job: Option<Box<dyn FnOnce() + Send + 'env>>,
    /// Indices of predecessor nodes that must finish before this one runs.
    deps: Vec<usize>,
}

/// A builder for a heterogeneous sub-job dependency DAG (design §8.3).
///
/// Add nodes with [`add`](JobDag::add) (no predecessors) or
/// [`add_after`](JobDag::add_after) (waits on the given handles), optionally
/// wire extra edges with [`order`](JobDag::order), then run the whole graph on
/// a pool with [`dispatch`](JobDag::dispatch). Usually you do not construct this
/// directly — take a [`JobGraph`](crate::system::job_graph::JobGraph) system
/// param and call [`JobGraph::dag`](crate::system::job_graph::JobGraph::dag),
/// which builds, dispatches, and joins for you.
///
/// `'env` is the lifetime of data the node closures borrow from the enclosing
/// stack frame; every node is joined before dispatch returns, so borrowing
/// local system state is sound (the same contract as
/// [`TaskPool::scope`](prism_tasks::TaskPool::scope)).
pub struct JobDag<'env> {
    nodes: Vec<Node<'env>>,
}

impl<'env> Default for JobDag<'env> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<'env> JobDag<'env> {
    /// Create an empty graph.
    #[inline]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Create an empty graph with capacity for `n` nodes.
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n),
        }
    }

    /// The number of nodes added so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether no nodes have been added.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Add a root sub-job that depends on nothing and may start immediately.
    #[inline]
    pub fn add<F>(&mut self, f: F) -> JobId
    where
        F: FnOnce() + Send + 'env,
    {
        self.push(Vec::new(), f)
    }

    /// Add a sub-job that runs only after every handle in `deps` has finished.
    ///
    /// Each dependency must be a handle returned by an earlier `add`/`add_after`
    /// on *this* graph (enforced by a bounds assertion); referencing only
    /// earlier nodes keeps the graph acyclic by construction. Duplicate entries
    /// in `deps` are collapsed to a single edge.
    pub fn add_after<F>(&mut self, deps: &[JobId], f: F) -> JobId
    where
        F: FnOnce() + Send + 'env,
    {
        let len = self.nodes.len();
        let mut edges: Vec<usize> = Vec::with_capacity(deps.len());
        for &d in deps {
            assert!(
                d.0 < len,
                "JobDag::add_after: dependency {} is not a node of this graph (len {len})",
                d.0
            );
            if !edges.contains(&d.0) {
                edges.push(d.0);
            }
        }
        self.push(edges, f)
    }

    /// Draw an extra dependency edge: `after` will not start until `before`
    /// finishes. Both handles must already belong to this graph.
    ///
    /// Edges that form a cycle are rejected at [`dispatch`](JobDag::dispatch)
    /// time with a panic rather than deadlocking.
    pub fn order(&mut self, before: JobId, after: JobId) {
        let len = self.nodes.len();
        assert!(
            before.0 < len && after.0 < len,
            "JobDag::order: both handles must be nodes of this graph (len {len})"
        );
        assert!(
            before.0 != after.0,
            "JobDag::order: a node cannot depend on itself (node {})",
            before.0
        );
        let deps = &mut self.nodes[after.0].deps;
        if !deps.contains(&before.0) {
            deps.push(before.0);
        }
    }

    /// Shared node-insertion path.
    #[inline]
    fn push<F>(&mut self, deps: Vec<usize>, f: F) -> JobId
    where
        F: FnOnce() + Send + 'env,
    {
        let id = JobId(self.nodes.len());
        self.nodes.push(Node {
            job: Some(Box::new(f)),
            deps,
        });
        id
    }

    /// Run the whole graph on `pool`, returning only once every node has
    /// executed.
    ///
    /// Nodes with no outstanding predecessors are dispatched immediately; each
    /// finishing node decrements its successors' atomic pending counts and
    /// dispatches any that reach zero, so independent chains overlap and the
    /// only synchronisation is each real edge. Panics in nodes are re-raised on
    /// the caller after the graph joins.
    ///
    /// # Panics
    ///
    /// Panics if the declared edges contain a cycle (dataflow dispatch would
    /// otherwise be impossible).
    pub fn dispatch(self, pool: &TaskPool) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        // Build the successor adjacency and in-degrees from the per-node
        // predecessor lists.
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indeg: Vec<usize> = vec![0; n];
        for (i, node) in self.nodes.iter().enumerate() {
            for &d in &node.deps {
                succ[d].push(i);
                indeg[i] += 1;
            }
        }

        // Validate acyclicity (and derive a topological order for the
        // single-threaded fallback) via Kahn's algorithm.
        let order = kahn_order(&succ, &indeg)
            .expect("JobDag::dispatch: dependency cycle detected — the graph must be a DAG");

        // Single-threaded fallback: run the topological order inline. This
        // avoids deep inline-spawn recursion and yields the same result the
        // multi-threaded dataflow dispatch would.
        if pool.is_single_threaded() {
            let mut jobs: Vec<Option<Box<dyn FnOnce() + Send + 'env>>> =
                self.nodes.into_iter().map(|node| node.job).collect();
            for i in order {
                let job = jobs[i].take().expect("JobDag: node dispatched more than once");
                job();
            }
            return;
        }

        // Multi-threaded dataflow dispatch. The shared state is allocated here —
        // *outside* the scope — so references handed to spawned sub-jobs outlive
        // the scope's `'scope` region.
        let state = DagState {
            jobs: self
                .nodes
                .into_iter()
                .map(|node| Mutex::new(node.job))
                .collect(),
            succ,
            pending: indeg.into_iter().map(AtomicUsize::new).collect(),
        };

        pool.scope(|scope| {
            for (i, p) in state.pending.iter().enumerate() {
                if p.load(Ordering::Acquire) == 0 {
                    spawn_node(scope, &state, i);
                }
            }
        });
    }
}

/// Shared, `Sync` dispatch state borrowed by every spawned sub-job.
struct DagState<'env> {
    /// Each node's body, taken exactly once when the node is dispatched.
    jobs: Vec<Mutex<Option<Box<dyn FnOnce() + Send + 'env>>>>,
    /// `succ[i]` = nodes that depend on node `i`.
    succ: Vec<Vec<usize>>,
    /// `pending[i]` = number of node `i`'s predecessors that have not finished.
    pending: Vec<AtomicUsize>,
}

/// Take node `i`'s body and spawn it on `scope`; when it finishes, release its
/// successors and recursively spawn any whose last predecessor just completed.
///
/// `state` is borrowed for `'scope` (it is allocated outside the scope in
/// [`JobDag::dispatch`]), so the sub-job closures — which must be `'scope` —
/// may hold a reference to it. The node bodies live for the graph's own `'data`
/// environment, kept separate from the scope's internal `'env` and only
/// required to outlive the scope (`'data: 'scope`); this decoupling is what lets
/// `state` be a `dispatch`-local whose `'data` closures outlive each spawn.
fn spawn_node<'scope, 'env, 'data>(
    scope: &'scope Scope<'scope, 'env>,
    state: &'scope DagState<'data>,
    i: usize,
) where
    'data: 'scope,
{
    let job = state.jobs[i]
        .lock()
        .expect("JobDag: dispatch state mutex poisoned")
        .take()
        .expect("JobDag: node dispatched more than once");

    scope.spawn(move || {
        job();
        // Release successors: whoever drives a successor's pending count to zero
        // dispatches it. `AcqRel` pairs the predecessor's writes with the
        // successor's reads so the happens-before edge holds.
        for &s in &state.succ[i] {
            if state.pending[s].fetch_sub(1, Ordering::AcqRel) == 1 {
                spawn_node(scope, state, s);
            }
        }
    });
}

/// Kahn's algorithm: return a topological order of the DAG, or `None` if the
/// graph contains a cycle (fewer than `n` nodes become schedulable).
fn kahn_order(succ: &[Vec<usize>], indeg_in: &[usize]) -> Option<Vec<usize>> {
    let n = succ.len();
    let mut indeg = indeg_in.to_vec();
    let mut order: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut head = 0;
    while head < order.len() {
        let i = order[head];
        head += 1;
        for &s in &succ[i] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                order.push(s);
            }
        }
    }
    (order.len() == n).then_some(order)
}
