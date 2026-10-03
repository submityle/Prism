//! Propagation statistics: the observability hooks for a propagation pass
//! (design doc §16).
//!
//! The incremental propagator's whole value proposition is "cost proportional
//! to what actually moved". [`PropagationStats`] makes that measurable: it
//! reports how many nodes were recomputed versus skipped, how many independent
//! dirty subtrees there were (the natural parallel grain), and how deep the
//! forest is (long chains are a pathology worth flagging). These are real
//! counts derived from a pass, not estimates — a static frame reports
//! `nodes_visited == 0`, and an editor/profiler can surface "why was this frame
//! expensive" directly.
//!
//! Timing is exposed as a hook rather than measured here: the core is
//! `no_std` and clockless, so [`PropagationStats::elapsed_nanos`] is filled by
//! the caller via [`PropagationStats::with_elapsed_nanos`] from whatever clock
//! the host owns (e.g. `prism_time`). It defaults to `0`, meaning "unmeasured".

use alloc::vec;

use crate::dirty::DirtyStats;
use crate::hierarchy::Hierarchy;

/// Per-pass propagation metrics.
///
/// Build one from a dirty pass with [`PropagationStats::from_incremental`], then
/// optionally attach a measured duration with
/// [`PropagationStats::with_elapsed_nanos`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PropagationStats {
    /// Total nodes in the forest this pass ran over.
    pub nodes_total: usize,
    /// Nodes whose world transform was (re)computed this pass. `0` for a static
    /// frame; equal to `nodes_total` for a full rebuild.
    pub nodes_visited: usize,
    /// Nodes left untouched because they sat in a clean subtree
    /// (`nodes_total - nodes_visited`). This is the incremental win made
    /// explicit.
    pub dirty_skipped: usize,
    /// Number of minimal dirty roots (changed nodes with no changed ancestor).
    pub dirty_roots: usize,
    /// Depth-level count of the forest: the length of its longest root-to-leaf
    /// chain. A sudden spike flags a degenerate deep hierarchy. `0` for an
    /// empty forest.
    pub levels: usize,
    /// Independent subtrees that could run concurrently — the by-root parallel
    /// grain of design §8. Equal to [`PropagationStats::dirty_roots`] for an
    /// incremental pass (disjoint dirty subtrees carry no cross-dependency).
    pub parallel_chunks: usize,
    /// Wall-clock duration of the pass in nanoseconds, or `0` when unmeasured.
    /// Filled by the caller via [`PropagationStats::with_elapsed_nanos`].
    pub elapsed_nanos: u64,
}

impl PropagationStats {
    /// Derive the metrics of an incremental pass from its [`DirtyStats`] and the
    /// `hierarchy` it ran over.
    #[must_use]
    pub fn from_incremental(hierarchy: &Hierarchy, dirty: DirtyStats) -> Self {
        let nodes_total = hierarchy.len();
        let nodes_visited = dirty.recomputed;
        Self {
            nodes_total,
            nodes_visited,
            dirty_skipped: nodes_total.saturating_sub(nodes_visited),
            dirty_roots: dirty.dirty_roots,
            levels: depth_levels(hierarchy),
            parallel_chunks: dirty.dirty_roots,
            elapsed_nanos: 0,
        }
    }

    /// Attach a measured duration (from the caller's own clock).
    #[inline]
    #[must_use]
    pub fn with_elapsed_nanos(mut self, nanos: u64) -> Self {
        self.elapsed_nanos = nanos;
        self
    }
}

/// The forest's depth-level count: the length of its longest root-to-leaf
/// chain, i.e. `max_depth + 1`. Returns `0` for an empty forest, and `0` if the
/// hierarchy is not a forest (so a cycle cannot panic this diagnostic helper).
#[must_use]
pub fn depth_levels(hierarchy: &Hierarchy) -> usize {
    let Ok(order) = hierarchy.compute_order() else {
        return 0;
    };
    let n = hierarchy.len();
    if n == 0 {
        return 0;
    }
    // Parents precede children in `order`, so one forward pass resolves every
    // depth from its parent's.
    let mut depth = vec![0usize; n];
    let mut max_depth = 0usize;
    for &node in &order {
        let d = match hierarchy.parent(node) {
            None => 0,
            Some(parent) => depth[parent.index()] + 1,
        };
        depth[node.index()] = d;
        if d > max_depth {
            max_depth = d;
        }
    }
    max_depth + 1
}
