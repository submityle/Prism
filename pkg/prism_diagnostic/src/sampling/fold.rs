//! Hotspot folding: flat profiles, call trees, and collapsed stacks (§24.5).
//!
//! The raw sample buffer ([`StackSample`]s) is reconstructed into the three
//! shapes a profiler UI consumes, all as pure deterministic folds:
//!
//! 1. **Flat profile** ([`flat_profile`]): per-frame self (exclusive, leaf-hit)
//!    and inclusive (appears-anywhere) sample counts, converted to estimated
//!    nanoseconds via the sampling interval. This is the "which function is hot"
//!    table.
//! 2. **Call tree** ([`call_tree`]): a merged tree of stacks. [`FoldDirection::TopDown`]
//!    merges outermost-first (the classic caller→callee flame graph);
//!    [`FoldDirection::BottomUp`] merges leaf-first (the inverted "who called
//!    this hot leaf" view).
//! 3. **Collapsed stacks** ([`collapsed_stacks`]): one line per distinct full
//!    stack with its summed weight — the folded format a flame-graph renderer
//!    reads directly.
//!
//! A frame's estimated time is `samples * interval_nanos`: a well-known
//! statistical-profiling identity (a frame seen in `k` of `n` ticks ran roughly
//! `k/n` of the wall time). Counts dedupe per sample, so a recursive frame that
//! appears twice in one stack still contributes `1` to its inclusive count for
//! that sample. Everything is `core`/`alloc` integer arithmetic — no `unsafe`.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::sample::StackSample;
use super::symbol::{FrameId, SymbolTable};

/// Which direction to merge stacks when building a [`CallTree`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldDirection {
    /// Merge outermost-first (root → leaf): the classic caller→callee flame
    /// graph, where the root is the common entry point.
    TopDown,
    /// Merge leaf-first (leaf → root): the inverted flame graph, where each root
    /// is a hot leaf and children are its callers.
    BottomUp,
}

/// One frame's aggregated statistics in a [`FlatProfile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStat {
    /// The frame this row describes.
    pub frame: FrameId,
    /// Summed weight of samples whose *leaf* is this frame (exclusive / self).
    pub self_samples: u64,
    /// Summed weight of samples where this frame appears *anywhere* in the
    /// stack, counted once per sample (inclusive / total).
    pub inclusive_samples: u64,
    /// Estimated exclusive time, `self_samples * interval_nanos`.
    pub self_nanos: u64,
    /// Estimated inclusive time, `inclusive_samples * interval_nanos`.
    pub inclusive_nanos: u64,
}

/// A flat per-frame hotspot table, sorted by self time descending (the hottest
/// leaf first), ties broken by ascending [`FrameId`] for determinism.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FlatProfile {
    /// Per-frame rows, hottest self time first.
    rows: Vec<FrameStat>,
    /// Total weight across all samples (the inclusive count of the whole run).
    total_samples: u64,
    /// Sampling interval used to convert counts to nanoseconds.
    interval_nanos: u64,
}

impl FlatProfile {
    /// Borrow the per-frame rows (hottest self time first).
    #[inline]
    #[must_use]
    pub fn rows(&self) -> &[FrameStat] {
        &self.rows
    }

    /// Total sample weight folded into this profile.
    #[inline]
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// The sampling interval (nanoseconds per tick) used for this profile.
    #[inline]
    #[must_use]
    pub fn interval_nanos(&self) -> u64 {
        self.interval_nanos
    }

    /// Total estimated wall time, `total_samples * interval_nanos`.
    #[inline]
    #[must_use]
    pub fn total_nanos(&self) -> u64 {
        self.total_samples.saturating_mul(self.interval_nanos)
    }

    /// Look up a frame's row by id, if present.
    #[must_use]
    pub fn get(&self, frame: FrameId) -> Option<&FrameStat> {
        self.rows.iter().find(|row| row.frame == frame)
    }

    /// The hottest frame by self time, if any.
    #[inline]
    #[must_use]
    pub fn hottest(&self) -> Option<&FrameStat> {
        self.rows.first()
    }

    /// A frame's self time as a fraction of the whole run in `[0, 1]`.
    #[must_use]
    pub fn self_fraction(&self, frame: FrameId) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.get(frame).map_or(0.0, |row| {
            row.self_samples as f64 / self.total_samples as f64
        })
    }
}

/// Build a [`FlatProfile`] from a sample buffer and the sampling interval.
#[must_use]
pub fn flat_profile(samples: &[StackSample], interval_nanos: u64) -> FlatProfile {
    let interval_nanos = interval_nanos.max(1);
    let mut self_counts: BTreeMap<FrameId, u64> = BTreeMap::new();
    let mut inclusive_counts: BTreeMap<FrameId, u64> = BTreeMap::new();
    let mut total_samples = 0u64;
    // Reused per-sample dedup set for the inclusive count (recursion-safe).
    let mut seen: Vec<FrameId> = Vec::new();

    for sample in samples {
        total_samples = total_samples.saturating_add(sample.weight);
        if let Some(leaf) = sample.leaf() {
            *self_counts.entry(leaf).or_insert(0) += sample.weight;
        }
        seen.clear();
        for &frame in &sample.stack {
            if !seen.contains(&frame) {
                seen.push(frame);
                *inclusive_counts.entry(frame).or_insert(0) += sample.weight;
            }
        }
    }

    let mut rows: Vec<FrameStat> = inclusive_counts
        .iter()
        .map(|(&frame, &inclusive)| {
            let self_samples = self_counts.get(&frame).copied().unwrap_or(0);
            FrameStat {
                frame,
                self_samples,
                inclusive_samples: inclusive,
                self_nanos: self_samples.saturating_mul(interval_nanos),
                inclusive_nanos: inclusive.saturating_mul(interval_nanos),
            }
        })
        .collect();

    // Hottest self time first; ties by ascending FrameId for a stable order.
    rows.sort_by(|a, b| {
        b.self_samples
            .cmp(&a.self_samples)
            .then_with(|| a.frame.cmp(&b.frame))
    });

    FlatProfile {
        rows,
        total_samples,
        interval_nanos,
    }
}

/// One node in a merged [`CallTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallNode {
    /// The frame at this node, or `None` for the synthetic root.
    pub frame: Option<FrameId>,
    /// Summed weight of samples passing through this node (inclusive).
    pub inclusive_samples: u64,
    /// Summed weight of samples whose merge path *ends* at this node (self).
    pub self_samples: u64,
    /// Child node indices into [`CallTree::nodes`], sorted by inclusive weight
    /// descending (ties by ascending child [`FrameId`]).
    pub children: Vec<usize>,
}

impl CallNode {
    /// Estimated inclusive time at this node for the given interval.
    #[inline]
    #[must_use]
    pub fn inclusive_nanos(&self, interval_nanos: u64) -> u64 {
        self.inclusive_samples.saturating_mul(interval_nanos)
    }

    /// Estimated self time at this node for the given interval.
    #[inline]
    #[must_use]
    pub fn self_nanos(&self, interval_nanos: u64) -> u64 {
        self.self_samples.saturating_mul(interval_nanos)
    }
}

/// A merged call tree built from a sample buffer in one [`FoldDirection`].
///
/// Node `0` is always the synthetic root; its [`CallNode::inclusive_samples`]
/// equals the total folded weight. Children are stored by index and are sorted
/// deterministically, so two equal sample buffers build byte-identical trees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallTree {
    /// All nodes; index `0` is the synthetic root.
    nodes: Vec<CallNode>,
    /// Direction this tree was folded in.
    direction: FoldDirection,
    /// Sampling interval for nanosecond conversion.
    interval_nanos: u64,
}

impl CallTree {
    /// Borrow every node (index `0` is the synthetic root).
    #[inline]
    #[must_use]
    pub fn nodes(&self) -> &[CallNode] {
        &self.nodes
    }

    /// The synthetic root node.
    #[inline]
    #[must_use]
    pub fn root(&self) -> &CallNode {
        &self.nodes[0]
    }

    /// The direction this tree was folded in.
    #[inline]
    #[must_use]
    pub fn direction(&self) -> FoldDirection {
        self.direction
    }

    /// The sampling interval (nanoseconds per tick) for this tree.
    #[inline]
    #[must_use]
    pub fn interval_nanos(&self) -> u64 {
        self.interval_nanos
    }

    /// Borrow a node by index, if in range.
    #[inline]
    #[must_use]
    pub fn node(&self, index: usize) -> Option<&CallNode> {
        self.nodes.get(index)
    }

    /// Total node count including the synthetic root.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Always `false`: a tree always has the synthetic root. Present so the
    /// `len`/`is_empty` pair satisfies the usual lint.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The maximum depth of any node below the root (`0` when only the root
    /// exists). Depth counts edges, so a single `root → a` path has depth `1`.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.depth_from(0, 0)
    }

    fn depth_from(&self, index: usize, depth: usize) -> usize {
        let node = &self.nodes[index];
        node.children
            .iter()
            .map(|&child| self.depth_from(child, depth + 1))
            .max()
            .unwrap_or(depth)
    }
}

/// Build a merged [`CallTree`] from a sample buffer in the given direction.
#[must_use]
pub fn call_tree(
    samples: &[StackSample],
    direction: FoldDirection,
    interval_nanos: u64,
) -> CallTree {
    let interval_nanos = interval_nanos.max(1);
    let mut nodes: Vec<CallNode> = Vec::with_capacity(samples.len() + 1);
    nodes.push(CallNode {
        frame: None,
        inclusive_samples: 0,
        self_samples: 0,
        children: Vec::new(),
    });
    // (parent index, child frame) -> child node index, for merge lookups.
    let mut index: BTreeMap<(usize, FrameId), usize> = BTreeMap::new();
    // Scratch buffer for the direction-ordered stack view.
    let mut ordered: Vec<FrameId> = Vec::new();

    for sample in samples {
        if sample.stack.is_empty() {
            // An empty stack still counts toward the root's inclusive total.
            nodes[0].inclusive_samples = nodes[0].inclusive_samples.saturating_add(sample.weight);
            continue;
        }
        ordered.clear();
        ordered.extend_from_slice(&sample.stack);
        if matches!(direction, FoldDirection::BottomUp) {
            ordered.reverse();
        }

        let mut current = 0usize;
        nodes[0].inclusive_samples = nodes[0].inclusive_samples.saturating_add(sample.weight);
        let last = ordered.len() - 1;
        for (depth, &frame) in ordered.iter().enumerate() {
            let child = match index.get(&(current, frame)) {
                Some(&existing) => existing,
                None => {
                    let new_index = nodes.len();
                    nodes.push(CallNode {
                        frame: Some(frame),
                        inclusive_samples: 0,
                        self_samples: 0,
                        children: Vec::new(),
                    });
                    nodes[current].children.push(new_index);
                    index.insert((current, frame), new_index);
                    new_index
                }
            };
            nodes[child].inclusive_samples =
                nodes[child].inclusive_samples.saturating_add(sample.weight);
            if depth == last {
                nodes[child].self_samples = nodes[child].self_samples.saturating_add(sample.weight);
            }
            current = child;
        }
    }

    // Deterministic child ordering: inclusive descending, ties by frame id.
    // Snapshot the inclusive weights first so the sort key borrow is clean.
    let inclusive: Vec<u64> = nodes.iter().map(|n| n.inclusive_samples).collect();
    let frames: Vec<Option<FrameId>> = nodes.iter().map(|n| n.frame).collect();
    for node in &mut nodes {
        node.children.sort_by(|&a, &b| {
            inclusive[b]
                .cmp(&inclusive[a])
                .then_with(|| frames[a].cmp(&frames[b]))
        });
    }

    CallTree {
        nodes,
        direction,
        interval_nanos,
    }
}

/// One collapsed stack: a full root→leaf frame path and its summed weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollapsedStack {
    /// The full stack, outermost frame first, leaf last.
    pub frames: Vec<FrameId>,
    /// Summed weight of samples with exactly this stack.
    pub samples: u64,
}

impl CollapsedStack {
    /// Render this stack in Brendan Gregg's folded format using `symbols`:
    /// `root;mid;leaf <samples>`. Unresolved ids render as `?<id>`.
    #[must_use]
    pub fn to_folded_string(&self, symbols: &SymbolTable) -> String {
        use alloc::string::String;
        use core::fmt::Write as _;

        let mut line = String::new();
        for (i, &frame) in self.frames.iter().enumerate() {
            if i > 0 {
                line.push(';');
            }
            match symbols.resolve(frame) {
                Some(name) => line.push_str(name),
                None => {
                    // Fall back to a stable placeholder for an unknown id.
                    let _ = write!(line, "?{}", frame.0);
                }
            }
        }
        let _ = write!(line, " {}", self.samples);
        line
    }
}

/// Fold a sample buffer into collapsed stacks, one per distinct full stack,
/// ordered lexicographically by frame-id path for determinism.
#[must_use]
pub fn collapsed_stacks(samples: &[StackSample]) -> Vec<CollapsedStack> {
    let mut folded: BTreeMap<Vec<FrameId>, u64> = BTreeMap::new();
    for sample in samples {
        let entry = folded.entry(sample.stack.clone()).or_insert(0);
        *entry = entry.saturating_add(sample.weight);
    }
    folded
        .into_iter()
        .map(|(frames, samples)| CollapsedStack { frames, samples })
        .collect()
}
