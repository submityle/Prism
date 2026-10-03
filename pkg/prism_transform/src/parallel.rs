//! Multi-threaded hierarchy propagation via `prism_tasks`.
//!
//! The serial pass in [`crate::propagation`] walks a single parent-before-child
//! order. Parallelism exploits a structural fact of a forest: a node's world
//! transform depends only on its parent's, and a parent's depth is strictly
//! less than its child's. So if we group nodes by **depth level**, every node
//! in a level can be computed independently once all shallower levels are done:
//!
//! ```text
//! level 0 (roots):   g[n] = local[n].affine()
//! level d (d > 0):   g[n] = g[parent(n)] * local[n]   // parent is in level < d
//! ```
//!
//! Within a level the writes target disjoint nodes and the reads touch only
//! already-finalized shallower levels, so a level is embarrassingly parallel.
//! We process levels in order; each level is split into adaptive chunks and run
//! through [`TaskPool::scope`]. The computation is **identical** to the serial
//! result bit-for-bit (same `Affine3` composition, same operand order), so a
//! parallel run equals a serial run.
//!
//! Granularity adaptivity: a level smaller than [`PARALLEL_THRESHOLD`] runs
//! inline on the calling thread, because the fork/join cost would dwarf the
//! work. Only levels with enough nodes pay for task scheduling.
//!
//! This module is `std`-only (it needs the `prism_tasks` thread pool) and is
//! gated behind the crate's `std` feature.

use alloc::vec::Vec;

use prism_tasks::TaskPool;

use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
use crate::{GlobalTransform, Transform};

/// Levels below this node count are propagated serially on the calling thread;
/// the fork/join overhead is not worth it for small levels.
pub const PARALLEL_THRESHOLD: usize = 256;

/// A precomputed depth-bucketed traversal plan for a fixed hierarchy topology.
///
/// Building the plan is `O(n)`; it stays valid as long as the hierarchy's
/// parent/child edges are unchanged (only the [`Transform`] *values* change).
/// Reuse it across frames to amortize the ordering cost — recompute only when
/// the topology changes.
#[derive(Clone, Debug, Default)]
pub struct LevelPlan {
    /// All node ids grouped by depth, concatenated shallow-to-deep.
    nodes: Vec<NodeId>,
    /// `ranges[d]` is the `(start, end)` slice of `nodes` holding depth `d`.
    ranges: Vec<(usize, usize)>,
}

impl LevelPlan {
    /// Build a plan from `hierarchy`.
    ///
    /// # Errors
    /// [`HierarchyError::Cycle`] if the hierarchy is not a forest.
    pub fn build(hierarchy: &Hierarchy) -> Result<Self, HierarchyError> {
        let order = hierarchy.compute_order()?;
        let n = hierarchy.len();
        // Depth of each node; parents precede children in `order`, so a single
        // forward pass resolves every depth from its parent's.
        let mut depth = alloc::vec![0u32; n];
        let mut max_depth = 0u32;
        for &node in &order {
            let d = match hierarchy.parent(node) {
                None => 0,
                Some(parent) => depth[parent.index()] + 1,
            };
            depth[node.index()] = d;
            max_depth = max_depth.max(d);
        }

        // Counting sort of nodes by depth into contiguous level buckets.
        let num_levels = (max_depth as usize) + 1;
        let mut counts = alloc::vec![0usize; num_levels];
        for &node in &order {
            counts[depth[node.index()] as usize] += 1;
        }
        let mut ranges = Vec::with_capacity(num_levels);
        let mut acc = 0usize;
        for &c in &counts {
            ranges.push((acc, acc + c));
            acc += c;
        }
        let mut cursor: Vec<usize> = ranges.iter().map(|&(start, _)| start).collect();
        let mut nodes = alloc::vec![NodeId::new(0); n];
        for &node in &order {
            let d = depth[node.index()] as usize;
            nodes[cursor[d]] = node;
            cursor[d] += 1;
        }

        Ok(Self { nodes, ranges })
    }

    /// Number of depth levels.
    #[inline]
    pub fn level_count(&self) -> usize {
        self.ranges.len()
    }

    /// Total number of nodes covered by the plan.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the plan covers no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The node ids at depth `level`.
    #[inline]
    fn level(&self, level: usize) -> &[NodeId] {
        let (start, end) = self.ranges[level];
        &self.nodes[start..end]
    }
}

/// Compute one level's world transforms into `out`, reading parents from the
/// already-finalized `globals`. Pure function of its inputs; no aliasing.
#[inline]
fn compute_level(
    nodes: &[NodeId],
    hierarchy: &Hierarchy,
    locals: &[Transform],
    globals: &[GlobalTransform],
    out: &mut [GlobalTransform],
) {
    for (slot, &node) in out.iter_mut().zip(nodes) {
        let local = &locals[node.index()];
        *slot = match hierarchy.parent(node) {
            None => GlobalTransform::from_transform(local),
            Some(parent) => globals[parent.index()].mul_transform(local),
        };
    }
}

/// Run a full propagation pass in parallel on `pool`, writing a world transform
/// for every node. The result is identical to [`crate::propagate`].
///
/// Builds a fresh [`LevelPlan`] each call; prefer
/// [`propagate_parallel_with_plan`] with a cached plan for a static topology.
///
/// # Errors
/// - [`HierarchyError::LengthMismatch`] if either buffer length differs from
///   the number of nodes.
/// - [`HierarchyError::Cycle`] if the hierarchy is not a forest.
pub fn propagate_parallel(
    pool: &TaskPool,
    hierarchy: &Hierarchy,
    locals: &[Transform],
    globals: &mut [GlobalTransform],
) -> Result<(), HierarchyError> {
    if locals.len() != hierarchy.len() || globals.len() != hierarchy.len() {
        return Err(HierarchyError::LengthMismatch);
    }
    let plan = LevelPlan::build(hierarchy)?;
    propagate_parallel_with_plan(pool, &plan, hierarchy, locals, globals);
    Ok(())
}

/// Run a full propagation pass in parallel using a precomputed [`LevelPlan`].
///
/// # Panics
/// Panics if `locals`/`globals` are not both the hierarchy's node count, or if
/// the plan was built for a different (larger) hierarchy.
pub fn propagate_parallel_with_plan(
    pool: &TaskPool,
    plan: &LevelPlan,
    hierarchy: &Hierarchy,
    locals: &[Transform],
    globals: &mut [GlobalTransform],
) {
    assert_eq!(
        locals.len(),
        hierarchy.len(),
        "locals length must equal the hierarchy node count"
    );
    assert_eq!(
        globals.len(),
        hierarchy.len(),
        "globals length must equal the hierarchy node count"
    );

    // Scratch output for the current level; reused (grown) across levels so we
    // never scatter into `globals` while a parallel read of it is in flight.
    let mut scratch: Vec<GlobalTransform> = Vec::new();

    for level in 0..plan.level_count() {
        let nodes = plan.level(level);
        if nodes.is_empty() {
            continue;
        }

        if nodes.len() < PARALLEL_THRESHOLD || pool.worker_count() <= 1 {
            // Small level (or single-worker pool): compute in place, serially.
            // Reads of `globals` touch only shallower, already-written levels,
            // and writes target this level's disjoint nodes, so we can scatter
            // directly without a scratch buffer.
            for &node in nodes {
                let local = &locals[node.index()];
                let world = match hierarchy.parent(node) {
                    None => GlobalTransform::from_transform(local),
                    Some(parent) => globals[parent.index()].mul_transform(local),
                };
                globals[node.index()] = world;
            }
            continue;
        }

        // Parallel level: compute into disjoint scratch chunks reading the
        // shared, immutable `globals`, then scatter back serially.
        scratch.clear();
        scratch.resize(nodes.len(), GlobalTransform::IDENTITY);

        let grain = adaptive_grain(nodes.len(), pool.worker_count());
        let globals_ref: &[GlobalTransform] = globals;
        pool.scope(|s| {
            for (nchunk, ochunk) in nodes.chunks(grain).zip(scratch.chunks_mut(grain)) {
                s.spawn(move || compute_level(nchunk, hierarchy, locals, globals_ref, ochunk));
            }
        });

        for (&node, &world) in nodes.iter().zip(scratch.iter()) {
            globals[node.index()] = world;
        }
    }
}

/// Chunk size for a parallel level: several chunks per worker, with a floor so
/// scheduling overhead stays amortized.
#[inline]
fn adaptive_grain(len: usize, workers: usize) -> usize {
    let workers = workers.max(1);
    if workers == 1 {
        return len.max(1);
    }
    let target_chunks = workers.saturating_mul(8).max(1);
    len.div_ceil(target_chunks).max(64).min(len.max(1))
}
