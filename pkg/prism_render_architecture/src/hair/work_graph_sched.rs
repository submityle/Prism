//! `GPU` `Work Graphs` scheduling sim for the strand solver (design doc §8.5
//! item14).
//!
//! The `XPBD` strand solver runs a handful of substeps per frame, and each
//! substep solves its distance / bending / collision constraints in
//! graph-colored batches: one color batch is internally parallel, but batches
//! run in sequence because neighboring batches share particles. Dispatching
//! that as a flat list of `GPU` passes bursts the command stream into sharp
//! peaks — a wide predict pass, then one dispatch per color batch, then a
//! velocity write-back, repeated per substep. The `GPU` `Work Graphs` answer is
//! to describe the whole frame as a producer-consumer graph and let the device
//! self-schedule it: a node launches its successors as soon as its own records
//! retire, so the hardware overlaps the tail of one batch with the head of the
//! next and flattens those peaks without a `CPU` round-trip.
//!
//! This module is the deterministic, panic-free *contract* layer for that
//! schedule. It is pure topology — array in, graph out, `golden`-comparable, no
//! device state and (deliberately) no floating point at all. It mirrors the
//! contract discipline of [`crate::hair::cluster`]: the real `GPU` self-schedule
//! is owned by `prism_render_scene`, and this layer only produces the `CPU`
//! description that dispatch is validated against. It does four jobs:
//!
//! * **Node description** — [`WorkNodeKind`] tags a node as a `Predict`,
//!   `SolveColorBatch`, `Finalize`, or an explicit `SubstepBoundary` marker, and
//!   [`WorkNode`] carries its substep index, optional color-batch index, its
//!   producer dependency indices, and an estimated record/`launch` count (how
//!   many particles or constraints it drives).
//! * **Graph assembly** — [`build_sim_work_graph`] expands a substep count and a
//!   per-color-batch constraint-count list into the canonical
//!   `Predict → SolveColorBatch chain → Finalize` `DAG` per substep, chaining
//!   each substep's `Predict` onto the previous substep's `Finalize`.
//! * **Validity & ordering** — [`WorkGraph`] checks dependency indices are in
//!   bounds and the graph is acyclic, and [`topological_order`] runs a
//!   deterministic `Kahn` sweep (ties broken by ascending index) that returns a
//!   safe prefix instead of panicking on a cyclic or malformed graph.
//! * **Bucketing** — [`bin_work_graph`] fans a list of node indices into
//!   per-[`WorkNodeKind`] buckets in one order-preserving pass, reusing the
//!   `push`/`total`/`is_empty` bucket shape of
//!   [`crate::virtual_geometry::bins`]; out-of-range indices are skipped.

use alloc::vec::Vec;

/// Hard upper bound on the number of substeps a single frame's work graph may
/// expand to. A requested substep count above this is clamped, so a bad caller
/// cannot explode the node count.
pub const MAX_WORK_GRAPH_SUBSTEPS: usize = 64;

/// Hard upper bound on the number of color batches emitted per substep. Guards
/// the node count the same way [`MAX_WORK_GRAPH_SUBSTEPS`] does.
pub const MAX_WORK_GRAPH_COLOR_BATCHES: usize = 256;

/// The role a [`WorkNode`] plays in one substep of the strand solve.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WorkNodeKind {
    /// Semi-implicit predict: integrate unconstrained positions for every
    /// particle before any constraint is solved. One per substep.
    Predict,
    /// Solve one graph-colored constraint batch. Internally parallel, but
    /// sequenced after the previous batch because batches share particles.
    SolveColorBatch,
    /// Velocity write-back: derive velocities from the solved positions at the
    /// end of a substep. One per substep.
    Finalize,
    /// Explicit substep barrier marker. Not emitted by
    /// [`build_sim_work_graph`] (which chains `Predict` directly onto the
    /// previous `Finalize`), but a valid node kind a caller can insert and bin
    /// when it wants an explicit synchronization point.
    SubstepBoundary,
}

/// One node in the solver work graph.
///
/// A node is a pure description: what it does ([`WorkNodeKind`]), which substep
/// and (for a `SolveColorBatch`) which color batch it belongs to, which
/// producer nodes it consumes (`deps`, by node index), and how much work it is
/// estimated to record — particle count for `Predict`/`Finalize`, constraint
/// count for a `SolveColorBatch`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkNode {
    /// This node's role in the solve.
    pub kind: WorkNodeKind,
    /// Zero-based substep this node belongs to.
    pub substep: usize,
    /// Zero-based color-batch index for a `SolveColorBatch`; `None` for every
    /// other kind, which is not tied to a specific batch.
    pub color_batch: Option<usize>,
    /// Producer node indices this node consumes (incoming edges).
    pub deps: Vec<usize>,
    /// Estimated record/`launch` count: how many particles or constraints this
    /// node drives.
    pub work_items: usize,
}

impl WorkNode {
    /// Builds a node with the given kind, substep, color batch, dependencies,
    /// and estimated work-item count.
    #[must_use]
    pub fn new(
        kind: WorkNodeKind,
        substep: usize,
        color_batch: Option<usize>,
        deps: Vec<usize>,
        work_items: usize,
    ) -> Self {
        Self {
            kind,
            substep,
            color_batch,
            deps,
            work_items,
        }
    }
}

/// A solver work graph: a node list plus the producer→consumer edges those
/// nodes' dependency sets encode.
///
/// Each node stores its incoming edges in [`WorkNode::deps`], so an edge
/// `d → n` exists exactly when `n`'s dependency set contains `d`. The graph
/// owns no device state; it is the `CPU` contract the real `GPU` self-schedule
/// is checked against.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorkGraph {
    /// The graph's nodes in assembly order.
    pub nodes: Vec<WorkNode>,
}

impl WorkGraph {
    /// Wraps a node list as a work graph.
    #[must_use]
    pub fn new(nodes: Vec<WorkNode>) -> Self {
        Self { nodes }
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Total number of producer→consumer edges, counting only dependency
    /// indices that reference a real node.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        let n = self.nodes.len();
        let mut total = 0;
        for node in &self.nodes {
            for &d in &node.deps {
                if d < n {
                    total += 1;
                }
            }
        }
        total
    }

    /// Returns `true` when every dependency index references a real node.
    #[must_use]
    pub fn deps_in_bounds(&self) -> bool {
        let n = self.nodes.len();
        for node in &self.nodes {
            for &d in &node.deps {
                if d >= n {
                    return false;
                }
            }
        }
        true
    }

    /// Returns `true` when the graph is acyclic, i.e. a `Kahn` sweep over its
    /// in-bounds edges can order every node. Out-of-bounds edges are ignored
    /// here; use [`WorkGraph::deps_in_bounds`] to detect those.
    #[must_use]
    pub fn is_acyclic(&self) -> bool {
        topological_order(self).len() == self.nodes.len()
    }

    /// Returns `true` when the graph is a well-formed `DAG`: every dependency
    /// index is in bounds and the graph is acyclic.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.deps_in_bounds() && self.is_acyclic()
    }
}

/// Tuning for [`build_sim_work_graph`].
///
/// All fields are plain counts; [`WorkGraphParams::sanitized`] clamps them into
/// safe ranges before assembly, so a malformed value can never panic or explode
/// the node count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkGraphParams {
    /// Upper bound on substeps actually expanded, clamped to
    /// `1..=`[`MAX_WORK_GRAPH_SUBSTEPS`] by [`WorkGraphParams::sanitized`].
    pub max_substeps: usize,
    /// Upper bound on color batches emitted per substep, clamped to
    /// `1..=`[`MAX_WORK_GRAPH_COLOR_BATCHES`] by [`WorkGraphParams::sanitized`].
    pub max_color_batches: usize,
    /// Particle count used as the estimated work-item count for every
    /// `Predict` and `Finalize` node.
    pub particle_count: usize,
}

impl Default for WorkGraphParams {
    fn default() -> Self {
        Self {
            max_substeps: 8,
            max_color_batches: 32,
            particle_count: 0,
        }
    }
}

impl WorkGraphParams {
    /// Returns a copy with every field clamped into its valid range: both caps
    /// are forced to at least `1` and no more than their hard limit, so a `0`
    /// or oversized cap cannot drop every node or run away.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            max_substeps: clamp_usize(self.max_substeps, 1, MAX_WORK_GRAPH_SUBSTEPS),
            max_color_batches: clamp_usize(self.max_color_batches, 1, MAX_WORK_GRAPH_COLOR_BATCHES),
            particle_count: self.particle_count,
        }
    }
}

/// Clamps `value` into `[lo, hi]`. `lo` is assumed `<= hi`.
#[must_use]
fn clamp_usize(value: usize, lo: usize, hi: usize) -> usize {
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
    }
}

/// Expands a frame of the strand solve into its canonical work graph.
///
/// For each of the `substeps` substeps (capped at
/// [`WorkGraphParams::max_substeps`]) the graph gets a `Predict` node, then a
/// `SolveColorBatch` node per non-empty entry of `color_batch_sizes` wired into
/// a dependency chain (`Predict → batch 0 → batch 1 → …`), then a `Finalize`
/// node depending on the last batch (or directly on `Predict` when a substep
/// has no batches). The next substep's `Predict` depends on the previous
/// substep's `Finalize`, serializing substeps while leaving each color-batch
/// chain free to overlap under the real `GPU` self-schedule.
///
/// Empty color batches (size `0`) are skipped, and emitted batches are capped at
/// [`WorkGraphParams::max_color_batches`]; emitted batches are renumbered
/// densely from `0`. A `substeps` of `0` yields an empty graph. The expansion is
/// fully deterministic: identical inputs always yield an identical graph.
#[must_use]
pub fn build_sim_work_graph(
    substeps: usize,
    color_batch_sizes: &[usize],
    params: WorkGraphParams,
) -> WorkGraph {
    let params = params.sanitized();
    let effective_substeps = clamp_usize(substeps, 0, params.max_substeps);

    let mut nodes: Vec<WorkNode> = Vec::new();
    let mut prev_finalize: Option<usize> = None;

    for substep in 0..effective_substeps {
        // Predict: depends on the previous substep's Finalize, if any.
        let predict_idx = nodes.len();
        let mut predict_deps: Vec<usize> = Vec::new();
        if let Some(pf) = prev_finalize {
            predict_deps.push(pf);
        }
        nodes.push(WorkNode::new(
            WorkNodeKind::Predict,
            substep,
            None,
            predict_deps,
            params.particle_count,
        ));

        // Color batches: a serial chain hanging off Predict.
        let mut last = predict_idx;
        let mut emitted_batches = 0usize;
        for &size in color_batch_sizes {
            if emitted_batches >= params.max_color_batches {
                break;
            }
            if size == 0 {
                continue;
            }
            let idx = nodes.len();
            nodes.push(WorkNode::new(
                WorkNodeKind::SolveColorBatch,
                substep,
                Some(emitted_batches),
                alloc::vec![last],
                size,
            ));
            last = idx;
            emitted_batches += 1;
        }

        // Finalize: depends on the last node of this substep's chain.
        let finalize_idx = nodes.len();
        nodes.push(WorkNode::new(
            WorkNodeKind::Finalize,
            substep,
            None,
            alloc::vec![last],
            params.particle_count,
        ));
        prev_finalize = Some(finalize_idx);
    }

    WorkGraph::new(nodes)
}

/// Returns a deterministic topological ordering of the graph's node indices via
/// `Kahn`'s algorithm.
///
/// Among nodes that are currently ready (in-degree `0`, not yet emitted) the
/// smallest index is always chosen next, so the order is stable and
/// reproducible. Only in-bounds dependency edges are considered; out-of-bounds
/// edges are dropped. If the graph has a cycle (or an edge whose target can
/// never become ready), the returned ordering is the safe prefix of nodes that
/// could be ordered rather than a panic — so `len() < node_count` signals a
/// non-`DAG`.
#[must_use]
pub fn topological_order(graph: &WorkGraph) -> Vec<usize> {
    let n = graph.nodes.len();
    let mut indegree: Vec<usize> = alloc::vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];

    for (ni, node) in graph.nodes.iter().enumerate() {
        for &d in &node.deps {
            if d < n {
                indegree[ni] += 1;
                successors[d].push(ni);
            }
        }
    }

    let mut emitted: Vec<bool> = alloc::vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);

    loop {
        // Pick the smallest-index ready node for a stable, deterministic order.
        let mut pick: Option<usize> = None;
        for i in 0..n {
            if !emitted[i] && indegree[i] == 0 {
                pick = Some(i);
                break;
            }
        }
        let Some(i) = pick else {
            break;
        };
        emitted[i] = true;
        order.push(i);
        for &s in &successors[i] {
            if indegree[s] > 0 {
                indegree[s] -= 1;
            }
        }
    }

    order
}

/// A node index list partitioned by [`WorkNodeKind`].
///
/// Each bucket holds node indices in the order they were seen, so the
/// partition is deterministic and preserves input order within a kind, matching
/// the bucket discipline of [`crate::virtual_geometry::bins`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorkNodeBins {
    /// Indices of `Predict` nodes.
    pub predict: Vec<usize>,
    /// Indices of `SolveColorBatch` nodes.
    pub solve_color_batch: Vec<usize>,
    /// Indices of `Finalize` nodes.
    pub finalize: Vec<usize>,
    /// Indices of `SubstepBoundary` nodes.
    pub substep_boundary: Vec<usize>,
}

impl WorkNodeBins {
    /// Mutable handle to the bucket backing a given node kind.
    fn bucket_mut(&mut self, kind: WorkNodeKind) -> &mut Vec<usize> {
        match kind {
            WorkNodeKind::Predict => &mut self.predict,
            WorkNodeKind::SolveColorBatch => &mut self.solve_color_batch,
            WorkNodeKind::Finalize => &mut self.finalize,
            WorkNodeKind::SubstepBoundary => &mut self.substep_boundary,
        }
    }

    /// Shared handle to the bucket backing a given node kind.
    #[must_use]
    pub fn bucket(&self, kind: WorkNodeKind) -> &[usize] {
        match kind {
            WorkNodeKind::Predict => &self.predict,
            WorkNodeKind::SolveColorBatch => &self.solve_color_batch,
            WorkNodeKind::Finalize => &self.finalize,
            WorkNodeKind::SubstepBoundary => &self.substep_boundary,
        }
    }

    /// Appends a node index to the bucket for `kind`.
    pub fn push(&mut self, kind: WorkNodeKind, index: usize) {
        self.bucket_mut(kind).push(index);
    }

    /// Total number of indices across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.predict.len()
            + self.solve_color_batch.len()
            + self.finalize.len()
            + self.substep_boundary.len()
    }

    /// Returns `true` when no index landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.predict.is_empty()
            && self.solve_color_batch.is_empty()
            && self.finalize.is_empty()
            && self.substep_boundary.is_empty()
    }
}

/// Partitions the given node `indices` into per-[`WorkNodeKind`] buckets.
///
/// Each index is looked up in `graph`; an index that falls outside the node
/// list is skipped rather than panicking, so a stale index list cannot crash
/// bucketing. Input order is preserved within each bucket.
#[must_use]
pub fn bin_work_graph(graph: &WorkGraph, indices: &[usize]) -> WorkNodeBins {
    let mut bins = WorkNodeBins::default();
    for &i in indices {
        if let Some(node) = graph.nodes.get(i) {
            bins.push(node.kind, i);
        }
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn empty_substeps_build_empty_graph() {
        let g = build_sim_work_graph(0, &[3, 4], WorkGraphParams::default());
        assert!(g.is_empty());
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(g.is_valid());
        assert_eq!(topological_order(&g), Vec::<usize>::new());
    }

    #[test]
    fn single_substep_no_batches_is_predict_then_finalize() {
        let params = WorkGraphParams {
            particle_count: 100,
            ..WorkGraphParams::default()
        };
        let g = build_sim_work_graph(1, &[], params);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.nodes[0].kind, WorkNodeKind::Predict);
        assert_eq!(g.nodes[0].deps, Vec::<usize>::new());
        assert_eq!(g.nodes[0].work_items, 100);
        assert_eq!(g.nodes[1].kind, WorkNodeKind::Finalize);
        assert_eq!(g.nodes[1].deps, vec![0]);
        assert_eq!(g.nodes[1].work_items, 100);
        assert!(g.is_valid());
        assert_eq!(topological_order(&g), vec![0, 1]);
    }

    #[test]
    fn single_substep_chains_color_batches() {
        let params = WorkGraphParams {
            particle_count: 10,
            ..WorkGraphParams::default()
        };
        let g = build_sim_work_graph(1, &[3, 5, 2], params);
        // Predict, Solve(3), Solve(5), Solve(2), Finalize.
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.nodes[1].kind, WorkNodeKind::SolveColorBatch);
        assert_eq!(g.nodes[1].color_batch, Some(0));
        assert_eq!(g.nodes[1].work_items, 3);
        assert_eq!(g.nodes[1].deps, vec![0]);
        assert_eq!(g.nodes[2].color_batch, Some(1));
        assert_eq!(g.nodes[2].work_items, 5);
        assert_eq!(g.nodes[2].deps, vec![1]);
        assert_eq!(g.nodes[3].color_batch, Some(2));
        assert_eq!(g.nodes[3].work_items, 2);
        assert_eq!(g.nodes[3].deps, vec![2]);
        assert_eq!(g.nodes[4].kind, WorkNodeKind::Finalize);
        assert_eq!(g.nodes[4].deps, vec![3]);
        assert_eq!(g.edge_count(), 4);
        assert!(g.is_valid());
    }

    #[test]
    fn next_substep_predict_depends_on_previous_finalize() {
        let g = build_sim_work_graph(2, &[4], WorkGraphParams::default());
        // s0: Predict(0), Solve(1), Finalize(2); s1: Predict(3), Solve(4), Finalize(5).
        assert_eq!(g.node_count(), 6);
        assert_eq!(g.nodes[3].kind, WorkNodeKind::Predict);
        assert_eq!(g.nodes[3].substep, 1);
        assert_eq!(g.nodes[3].deps, vec![2]);
        assert_eq!(g.nodes[2].kind, WorkNodeKind::Finalize);
        assert_eq!(g.nodes[2].substep, 0);
        assert!(g.is_valid());
        // The chain is already in dependency order, so topo is the identity.
        assert_eq!(topological_order(&g), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn skips_empty_batches_and_renumbers_densely() {
        let g = build_sim_work_graph(1, &[0, 4, 0, 2], WorkGraphParams::default());
        // Predict, Solve(4) as batch 0, Solve(2) as batch 1, Finalize.
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.nodes[1].color_batch, Some(0));
        assert_eq!(g.nodes[1].work_items, 4);
        assert_eq!(g.nodes[2].color_batch, Some(1));
        assert_eq!(g.nodes[2].work_items, 2);
        assert!(g.is_valid());
    }

    #[test]
    fn substeps_are_capped_at_max_substeps() {
        let params = WorkGraphParams {
            max_substeps: 2,
            ..WorkGraphParams::default()
        };
        let g = build_sim_work_graph(10, &[], params);
        // Two substeps, each Predict + Finalize.
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.nodes[3].substep, 1);
    }

    #[test]
    fn color_batches_are_capped_at_max_color_batches() {
        let params = WorkGraphParams {
            max_color_batches: 1,
            ..WorkGraphParams::default()
        };
        let g = build_sim_work_graph(1, &[3, 4, 5], params);
        // Predict, one Solve, Finalize.
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.nodes[1].kind, WorkNodeKind::SolveColorBatch);
        assert_eq!(g.nodes[1].color_batch, Some(0));
        assert_eq!(g.nodes[2].kind, WorkNodeKind::Finalize);
    }

    #[test]
    fn sanitize_clamps_caps_into_range() {
        let p = WorkGraphParams {
            max_substeps: 0,
            max_color_batches: 100_000,
            particle_count: 42,
        }
        .sanitized();
        assert_eq!(p.max_substeps, 1);
        assert_eq!(p.max_color_batches, MAX_WORK_GRAPH_COLOR_BATCHES);
        assert_eq!(p.particle_count, 42);
    }

    #[test]
    fn topological_order_breaks_ties_by_ascending_index() {
        // Three independent roots plus one node depending on all of them.
        let nodes = vec![
            WorkNode::new(WorkNodeKind::Predict, 0, None, vec![], 0),
            WorkNode::new(WorkNodeKind::Predict, 0, None, vec![], 0),
            WorkNode::new(WorkNodeKind::Predict, 0, None, vec![], 0),
            WorkNode::new(WorkNodeKind::Finalize, 0, None, vec![0, 1, 2], 0),
        ];
        let g = WorkGraph::new(nodes);
        assert!(g.is_valid());
        assert_eq!(topological_order(&g), vec![0, 1, 2, 3]);
    }

    #[test]
    fn cyclic_graph_does_not_panic_and_returns_prefix() {
        // 0 -> 1 -> 2 -> 0 is a pure cycle; node 3 is an independent root.
        let nodes = vec![
            WorkNode::new(WorkNodeKind::SolveColorBatch, 0, Some(0), vec![2], 1),
            WorkNode::new(WorkNodeKind::SolveColorBatch, 0, Some(1), vec![0], 1),
            WorkNode::new(WorkNodeKind::SolveColorBatch, 0, Some(2), vec![1], 1),
            WorkNode::new(WorkNodeKind::Predict, 0, None, vec![], 0),
        ];
        let g = WorkGraph::new(nodes);
        assert!(!g.is_acyclic());
        assert!(!g.is_valid());
        // Only the acyclic root can be ordered.
        assert_eq!(topological_order(&g), vec![3]);
    }

    #[test]
    fn out_of_bounds_dependency_is_flagged_but_safe() {
        let nodes = vec![
            WorkNode::new(WorkNodeKind::Predict, 0, None, vec![], 0),
            WorkNode::new(WorkNodeKind::Finalize, 0, None, vec![99], 0),
        ];
        let g = WorkGraph::new(nodes);
        assert!(!g.deps_in_bounds());
        assert!(!g.is_valid());
        // The dangling edge is dropped, so both nodes still order safely.
        assert_eq!(topological_order(&g), vec![0, 1]);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn bin_routes_by_kind_and_preserves_order() {
        let g = build_sim_work_graph(1, &[3, 5], WorkGraphParams::default());
        // Predict(0), Solve(1), Solve(2), Finalize(3).
        let order = topological_order(&g);
        let bins = bin_work_graph(&g, &order);
        assert_eq!(bins.total(), 4);
        assert_eq!(bins.predict, vec![0]);
        assert_eq!(bins.solve_color_batch, vec![1, 2]);
        assert_eq!(bins.finalize, vec![3]);
        assert!(bins.substep_boundary.is_empty());
        assert_eq!(bins.bucket(WorkNodeKind::SolveColorBatch), &[1, 2]);
    }

    #[test]
    fn bin_skips_out_of_range_indices() {
        let g = build_sim_work_graph(1, &[], WorkGraphParams::default());
        // Only indices 0 and 1 exist; 2 and 99 are out of range.
        let bins = bin_work_graph(&g, &[1, 99, 0, 2]);
        assert_eq!(bins.total(), 2);
        assert_eq!(bins.finalize, vec![1]);
        assert_eq!(bins.predict, vec![0]);
    }

    #[test]
    fn bin_empty_indices_is_empty() {
        let g = build_sim_work_graph(2, &[4], WorkGraphParams::default());
        let bins = bin_work_graph(&g, &[]);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }

    #[test]
    fn substep_boundary_bins_when_present() {
        let nodes = vec![
            WorkNode::new(WorkNodeKind::SubstepBoundary, 0, None, vec![], 0),
            WorkNode::new(WorkNodeKind::Predict, 0, None, vec![0], 0),
        ];
        let g = WorkGraph::new(nodes);
        let bins = bin_work_graph(&g, &[0, 1]);
        assert_eq!(bins.substep_boundary, vec![0]);
        assert_eq!(bins.predict, vec![1]);
    }

    #[test]
    fn build_is_deterministic_bit_exact() {
        let params = WorkGraphParams {
            max_substeps: 6,
            max_color_batches: 10,
            particle_count: 2048,
        };
        let a = build_sim_work_graph(3, &[7, 0, 11, 4], params);
        let b = build_sim_work_graph(3, &[7, 0, 11, 4], params);
        assert_eq!(a, b);
        // Topological orders also match bit for bit.
        assert_eq!(topological_order(&a), topological_order(&b));
    }
}
