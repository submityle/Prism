//! One-dimensional augmented interval tree for particle interval-set queries.
//!
//! A particle engine routinely holds a *collection* of one-dimensional (`1D`)
//! intervals and must answer spatial queries against the whole set: which
//! lifetime windows are alive at frame time `p` (a `stabbing` query), which
//! emitter bursts overlap a scheduled window `q`, how many axis spans cover a
//! given slab coordinate. Doing that with a linear scan is `O(n)` per query;
//! the classic Cormen-Leiserson-Rivest-Stein (`CLRS`) answer is the *interval
//! tree*: a balanced binary-search tree (`BST`) keyed on each interval's low
//! endpoint, augmented at every node with the maximum high endpoint found in
//! its subtree. That `max`-endpoint augmentation is what lets a query prune an
//! entire subtree in one comparison and report all `k` matches in
//! `O(log n + k)` time.
//!
//! This module is the self-contained reference for that structure. It builds a
//! static, balanced tree once from a slice of intervals (recursive median
//! split over the low-endpoint-sorted order, so the in-order walk is sorted and
//! the height is `O(log n)`), then serves three read-only queries: point
//! `stabbing`, interval overlap, and a `stabbing` count.
//!
//! # Scope boundary (read before extending)
//!
//! This is deliberately **not** the same module as the sibling
//! `interval_overlap_1d`. That sibling owns the *algebra of a single pair* of
//! intervals: intersect, union, hull, touch, membership -- the closed-form set
//! operations on two intervals. This module owns the *spatial query index over
//! `N` intervals*: a tree / augmented search structure (`CLRS` interval tree)
//! that reports which of many stored intervals satisfy a `stabbing` or overlap
//! predicate. Pair algebra answers "how do these two intervals combine"; this
//! index answers "which of my `N` intervals are hit". Keep set-combination
//! algebra in the sibling and keep the search structure here.
//!
//! Coordinates are integer `i64` so every comparison is exact: there is no
//! floating-point `==` hazard, no epsilon, and no transcendental call anywhere
//! in the module. Intervals are closed, `[lo, hi]`.

use alloc::vec::Vec;

/// Sentinel node handle meaning "no child" / "empty subtree".
///
/// Node links are plain `usize` indices into the flat node array; this
/// `usize::MAX` sentinel marks the absence of a child so the structure needs no
/// `Option` boxing and stays trivially copyable.
const NIL: usize = usize::MAX;

/// A closed one-dimensional (`1D`) integer interval `[lo, hi]`.
///
/// Both endpoints are inclusive. The canonical form keeps `lo <= hi`; the
/// [`Interval::new`] constructor enforces that by swapping reversed endpoints,
/// so callers may pass bounds in either order. Integer coordinates make every
/// comparison exact, so endpoint-touch cases (adjacent intervals sharing a
/// coordinate) are decided without any epsilon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Interval {
    /// Lower (inclusive) endpoint.
    pub lo: i64,
    /// Upper (inclusive) endpoint.
    pub hi: i64,
}

impl Interval {
    /// Builds a closed interval, swapping the endpoints when `lo > hi`.
    ///
    /// The result is always in canonical `lo <= hi` form.
    #[must_use]
    pub fn new(lo: i64, hi: i64) -> Self {
        if lo > hi {
            Self { lo: hi, hi: lo }
        } else {
            Self { lo, hi }
        }
    }

    /// Returns `true` when the closed interval covers point `p`.
    ///
    /// Uses an inclusive range membership test so both endpoints count as
    /// inside.
    #[must_use]
    pub fn contains(&self, p: i64) -> bool {
        (self.lo..=self.hi).contains(&p)
    }

    /// Returns `true` when this interval shares at least one point with `other`.
    ///
    /// Two closed intervals overlap exactly when neither lies strictly before
    /// the other, i.e. `self.lo <= other.hi && self.hi >= other.lo`. Touching
    /// at a single shared endpoint counts as overlap because the intervals are
    /// closed.
    #[must_use]
    pub fn overlaps(&self, other: Interval) -> bool {
        self.lo <= other.hi && self.hi >= other.lo
    }
}

/// One node of the augmented interval `BST`.
///
/// `max_hi` is the augmentation: the maximum `hi` over this node's whole
/// subtree. It is the single value a query compares against to decide that an
/// entire subtree can be skipped.
#[derive(Clone, Copy, Debug)]
struct Node {
    /// Low endpoint of this node's interval (the `BST` key).
    lo: i64,
    /// High endpoint of this node's interval.
    hi: i64,
    /// Original index of this interval in the slice passed to [`IntervalTree::build`].
    idx: usize,
    /// Maximum `hi` over the entire subtree rooted at this node.
    max_hi: i64,
    /// Left child handle, or [`NIL`].
    left: usize,
    /// Right child handle, or [`NIL`].
    right: usize,
}

/// A static, balanced, `max`-endpoint-augmented interval tree (`CLRS` style).
///
/// Build it once from a slice of [`Interval`]s with [`IntervalTree::build`];
/// then issue read-only `stabbing` and overlap queries. All reported values are
/// original indices into the slice that was built from, so callers can map a
/// hit straight back to their own storage.
#[derive(Clone, Debug)]
pub struct IntervalTree {
    /// Flat node pool; the in-order walk is sorted by `lo`.
    nodes: Vec<Node>,
    /// Handle of the root node, or [`NIL`] for an empty tree.
    root: usize,
}

/// Recursively builds a balanced subtree over `order[start..end]`.
///
/// `order` holds interval indices sorted by low endpoint. The median element
/// becomes the subtree root, keeping the tree height `O(log n)`; children are
/// built from the two halves. Returns the handle of the subtree root, or
/// [`NIL`] when the range is empty, and fills in each node's `max_hi`
/// augmentation on the way back up.
fn build_subtree(
    intervals: &[Interval],
    order: &[usize],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> usize {
    if start >= end {
        return NIL;
    }
    let mid = start + (end - start) / 2;
    let iv = intervals[order[mid]];
    let node_pos = nodes.len();
    nodes.push(Node {
        lo: iv.lo,
        hi: iv.hi,
        idx: order[mid],
        max_hi: iv.hi,
        left: NIL,
        right: NIL,
    });
    let left = build_subtree(intervals, order, start, mid, nodes);
    let right = build_subtree(intervals, order, mid + 1, end, nodes);
    let mut m = nodes[node_pos].hi;
    if left != NIL {
        m = m.max(nodes[left].max_hi);
    }
    if right != NIL {
        m = m.max(nodes[right].max_hi);
    }
    nodes[node_pos].left = left;
    nodes[node_pos].right = right;
    nodes[node_pos].max_hi = m;
    node_pos
}

impl IntervalTree {
    /// Builds the tree from a slice of intervals.
    ///
    /// The intervals are sorted by low endpoint (ties broken by high endpoint,
    /// then original index for a deterministic layout) and a balanced `BST` is
    /// assembled by recursive median split, so the height is `O(log n)`. The
    /// input slice is not mutated; reported indices refer to positions in it.
    #[must_use]
    pub fn build(intervals: &[Interval]) -> Self {
        let mut order: Vec<usize> = (0..intervals.len()).collect();
        order.sort_by(|&a, &b| {
            intervals[a]
                .lo
                .cmp(&intervals[b].lo)
                .then(intervals[a].hi.cmp(&intervals[b].hi))
                .then(a.cmp(&b))
        });
        let mut nodes: Vec<Node> = Vec::with_capacity(intervals.len());
        let root = build_subtree(intervals, &order, 0, order.len(), &mut nodes);
        Self { nodes, root }
    }

    /// Returns the number of intervals stored in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the tree holds no intervals.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Reports the indices of all intervals that contain point `p`.
    ///
    /// This is a `stabbing` query: it collects every stored interval `[lo, hi]`
    /// with `lo <= p <= hi`. Returned indices are not guaranteed in sorted
    /// order; sort them if a canonical comparison is required. Runs in
    /// `O(log n + k)` for `k` reported hits.
    #[must_use]
    pub fn query_point(&self, p: i64) -> Vec<usize> {
        self.query_overlap(Interval { lo: p, hi: p })
    }

    /// Reports the indices of all intervals that overlap the query interval `q`.
    ///
    /// Two closed intervals overlap when they share at least one point,
    /// including a single touching endpoint. Returned indices are not
    /// guaranteed in sorted order. Runs in `O(log n + k)`.
    #[must_use]
    pub fn query_overlap(&self, q: Interval) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        self.collect_overlap(self.root, q.lo, q.hi, &mut out);
        out
    }

    /// Counts the intervals that contain point `p` without allocating a result.
    ///
    /// Equivalent to `self.query_point(p).len()` but walks the tree once and
    /// keeps only a counter.
    #[must_use]
    pub fn count_point(&self, p: i64) -> usize {
        self.count_stab(self.root, p)
    }

    /// Recursive overlap collector with `max_hi` subtree pruning.
    ///
    /// Walks in-order (left, node, right) and uses the `max_hi` augmentation to
    /// cut whole subtrees: if every interval in a subtree ends before `qlo`, the
    /// subtree is skipped; if the current node's `lo` already exceeds `qhi`, the
    /// right subtree (whose keys are all `>= lo`) cannot overlap and is skipped.
    fn collect_overlap(&self, node: usize, qlo: i64, qhi: i64, out: &mut Vec<usize>) {
        if node == NIL {
            return;
        }
        let n = self.nodes[node];
        if n.max_hi < qlo {
            return;
        }
        self.collect_overlap(n.left, qlo, qhi, out);
        if n.lo <= qhi && n.hi >= qlo {
            out.push(n.idx);
        }
        if n.lo <= qhi {
            self.collect_overlap(n.right, qlo, qhi, out);
        }
    }

    /// Recursive `stabbing` counter mirroring [`IntervalTree::collect_overlap`].
    fn count_stab(&self, node: usize, p: i64) -> usize {
        if node == NIL {
            return 0;
        }
        let n = self.nodes[node];
        if n.max_hi < p {
            return 0;
        }
        let mut c = self.count_stab(n.left, p);
        if (n.lo..=n.hi).contains(&p) {
            c += 1;
        }
        if n.lo <= p {
            c += self.count_stab(n.right, p);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sorts a result vector so `stabbing`/overlap hits compare canonically.
    #[cfg(test)]
    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    /// Builds the shared reference interval set used by the hand-computed tests.
    ///
    /// Index map:
    /// `0:[15,20] 1:[10,30] 2:[17,19] 3:[5,20] 4:[12,15] 5:[30,40] 6:[25,30] 7:[0,3]`.
    #[cfg(test)]
    fn reference() -> IntervalTree {
        let data = [
            Interval::new(15, 20),
            Interval::new(10, 30),
            Interval::new(17, 19),
            Interval::new(5, 20),
            Interval::new(12, 15),
            Interval::new(30, 40),
            Interval::new(25, 30),
            Interval::new(0, 3),
        ];
        IntervalTree::build(&data)
    }

    #[test]
    fn build_empty_tree_has_no_nodes() {
        let t = IntervalTree::build(&[]);
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn build_single_interval_len_one() {
        let t = IntervalTree::build(&[Interval::new(2, 7)]);
        assert!(!t.is_empty());
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn single_interval_point_inside() {
        let t = IntervalTree::build(&[Interval::new(2, 7)]);
        assert_eq!(sorted(t.query_point(5)), [0]);
    }

    #[test]
    fn single_interval_point_outside_below() {
        let t = IntervalTree::build(&[Interval::new(2, 7)]);
        assert!(t.query_point(1).is_empty());
    }

    #[test]
    fn single_interval_point_outside_above() {
        let t = IntervalTree::build(&[Interval::new(2, 7)]);
        assert!(t.query_point(8).is_empty());
    }

    #[test]
    fn single_interval_count_inside_and_outside() {
        let t = IntervalTree::build(&[Interval::new(2, 7)]);
        assert_eq!(t.count_point(5), 1);
        assert_eq!(t.count_point(9), 0);
    }

    #[test]
    fn empty_tree_query_point_is_empty() {
        let t = IntervalTree::build(&[]);
        assert!(t.query_point(0).is_empty());
    }

    #[test]
    fn empty_tree_query_overlap_is_empty() {
        let t = IntervalTree::build(&[]);
        assert!(t.query_overlap(Interval::new(-5, 5)).is_empty());
    }

    #[test]
    fn empty_tree_count_point_is_zero() {
        let t = IntervalTree::build(&[]);
        assert_eq!(t.count_point(42), 0);
    }

    #[test]
    fn reference_stab_at_15() {
        let t = reference();
        assert_eq!(sorted(t.query_point(15)), [0, 1, 3, 4]);
    }

    #[test]
    fn reference_stab_at_30_upper_endpoints() {
        let t = reference();
        assert_eq!(sorted(t.query_point(30)), [1, 5, 6]);
    }

    #[test]
    fn reference_stab_at_17() {
        let t = reference();
        assert_eq!(sorted(t.query_point(17)), [0, 1, 2, 3]);
    }

    #[test]
    fn reference_stab_at_25() {
        let t = reference();
        assert_eq!(sorted(t.query_point(25)), [1, 6]);
    }

    #[test]
    fn reference_stab_at_3_touch_upper() {
        let t = reference();
        assert_eq!(sorted(t.query_point(3)), [7]);
    }

    #[test]
    fn reference_stab_at_0_lower_endpoint() {
        let t = reference();
        assert_eq!(sorted(t.query_point(0)), [7]);
    }

    #[test]
    fn reference_stab_at_40_upper_endpoint() {
        let t = reference();
        assert_eq!(sorted(t.query_point(40)), [5]);
    }

    #[test]
    fn reference_stab_at_4_gap_is_empty() {
        let t = reference();
        assert!(t.query_point(4).is_empty());
    }

    #[test]
    fn reference_stab_above_all_is_empty() {
        let t = reference();
        assert!(t.query_point(41).is_empty());
    }

    #[test]
    fn reference_stab_below_all_is_empty() {
        let t = reference();
        assert!(t.query_point(-5).is_empty());
    }

    #[test]
    fn reference_count_at_15() {
        let t = reference();
        assert_eq!(t.count_point(15), 4);
    }

    #[test]
    fn reference_count_at_30() {
        let t = reference();
        assert_eq!(t.count_point(30), 3);
    }

    #[test]
    fn reference_count_at_17() {
        let t = reference();
        assert_eq!(t.count_point(17), 4);
    }

    #[test]
    fn reference_count_at_4_is_zero() {
        let t = reference();
        assert_eq!(t.count_point(4), 0);
    }

    #[test]
    fn reference_overlap_16_18() {
        let t = reference();
        assert_eq!(sorted(t.query_overlap(Interval::new(16, 18))), [0, 1, 2, 3]);
    }

    #[test]
    fn reference_overlap_3_5_touches_both_ends() {
        let t = reference();
        assert_eq!(sorted(t.query_overlap(Interval::new(3, 5))), [3, 7]);
    }

    #[test]
    fn reference_overlap_21_24_only_wide_span() {
        let t = reference();
        assert_eq!(sorted(t.query_overlap(Interval::new(21, 24))), [1]);
    }

    #[test]
    fn reference_overlap_full_range_hits_all() {
        let t = reference();
        assert_eq!(
            sorted(t.query_overlap(Interval::new(0, 40))),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn reference_overlap_below_all_is_empty() {
        let t = reference();
        assert!(t.query_overlap(Interval::new(-20, -1)).is_empty());
    }

    #[test]
    fn reference_overlap_above_all_is_empty() {
        let t = reference();
        assert!(t.query_overlap(Interval::new(100, 200)).is_empty());
    }

    #[test]
    fn reference_count_matches_query_len() {
        let t = reference();
        for p in -10..=45 {
            assert_eq!(t.count_point(p), t.query_point(p).len());
        }
    }

    #[test]
    fn nested_intervals_stab_center_hits_all() {
        let data = [
            Interval::new(0, 100),
            Interval::new(10, 90),
            Interval::new(20, 80),
            Interval::new(40, 60),
        ];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_point(50)), [0, 1, 2, 3]);
    }

    #[test]
    fn nested_intervals_stab_edge_hits_outermost() {
        let data = [
            Interval::new(0, 100),
            Interval::new(10, 90),
            Interval::new(20, 80),
            Interval::new(40, 60),
        ];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_point(5)), [0]);
        assert_eq!(sorted(t.query_point(95)), [0]);
    }

    #[test]
    fn adjacent_intervals_shared_endpoint_both_stabbed() {
        // `[0,5]` and `[5,10]` touch exactly at 5; a closed-interval stab hits both.
        let data = [Interval::new(0, 5), Interval::new(5, 10)];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_point(5)), [0, 1]);
    }

    #[test]
    fn adjacent_intervals_interior_points_split() {
        let data = [Interval::new(0, 5), Interval::new(5, 10)];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_point(4)), [0]);
        assert_eq!(sorted(t.query_point(6)), [1]);
    }

    #[test]
    fn fully_overlapping_identical_span_all_reported() {
        let data = [
            Interval::new(1, 9),
            Interval::new(1, 9),
            Interval::new(1, 9),
        ];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_point(5)), [0, 1, 2]);
        assert_eq!(t.count_point(5), 3);
    }

    #[test]
    fn disjoint_set_each_point_hits_one() {
        let data = [
            Interval::new(0, 2),
            Interval::new(5, 7),
            Interval::new(10, 12),
        ];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_point(1)), [0]);
        assert_eq!(sorted(t.query_point(6)), [1]);
        assert_eq!(sorted(t.query_point(11)), [2]);
        assert!(t.query_point(3).is_empty());
    }

    #[test]
    fn disjoint_set_gap_overlap_is_empty() {
        let data = [
            Interval::new(0, 2),
            Interval::new(5, 7),
            Interval::new(10, 12),
        ];
        let t = IntervalTree::build(&data);
        assert!(t.query_overlap(Interval::new(3, 4)).is_empty());
    }

    #[test]
    fn duplicate_intervals_all_distinct_indices() {
        let data = [
            Interval::new(2, 8),
            Interval::new(2, 8),
            Interval::new(2, 8),
            Interval::new(2, 8),
        ];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_overlap(Interval::new(4, 5))), [0, 1, 2, 3]);
    }

    #[test]
    fn negative_coordinates_stab_and_overlap() {
        let data = [
            Interval::new(-50, -40),
            Interval::new(-45, -5),
            Interval::new(-10, 10),
        ];
        let t = IntervalTree::build(&data);
        assert_eq!(sorted(t.query_point(-45)), [0, 1]);
        assert_eq!(sorted(t.query_point(-8)), [1, 2]);
        assert_eq!(sorted(t.query_overlap(Interval::new(-42, -42))), [0, 1]);
    }

    #[test]
    fn interval_new_swaps_reversed_endpoints() {
        let a = Interval::new(9, 2);
        assert_eq!(a.lo, 2);
        assert_eq!(a.hi, 9);
    }

    #[test]
    fn interval_contains_both_endpoints() {
        let a = Interval::new(3, 7);
        assert!(a.contains(3));
        assert!(a.contains(7));
        assert!(a.contains(5));
        assert!(!a.contains(2));
        assert!(!a.contains(8));
    }

    #[test]
    fn interval_overlaps_touch_counts() {
        let a = Interval::new(0, 5);
        assert!(a.overlaps(Interval::new(5, 10)));
        assert!(a.overlaps(Interval::new(-3, 0)));
        assert!(!a.overlaps(Interval::new(6, 9)));
        assert!(!a.overlaps(Interval::new(-4, -1)));
    }

    #[test]
    fn point_interval_query_single_coordinate() {
        // A degenerate `[k,k]` query interval is exactly a point stab.
        let t = reference();
        assert_eq!(
            sorted(t.query_overlap(Interval::new(25, 25))),
            sorted(t.query_point(25))
        );
    }

    #[test]
    fn large_sequential_overlap_window() {
        // 64 unit-length intervals `[i,i]`; a window `[10,13]` hits exactly 4 of them.
        let data: Vec<Interval> = (0..64_i64).map(|i| Interval::new(i, i)).collect();
        let t = IntervalTree::build(&data);
        assert_eq!(
            sorted(t.query_overlap(Interval::new(10, 13))),
            [10, 11, 12, 13]
        );
    }

    #[test]
    fn large_staircase_stab_counts() {
        // Interval i covers `[0, i]`, so point p is covered by indices p..=n-1.
        let n = 40_i64;
        let data: Vec<Interval> = (0..n).map(|i| Interval::new(0, i)).collect();
        let t = IntervalTree::build(&data);
        // Point 10 is covered by every interval whose hi >= 10, i.e. indices 10..=39.
        assert_eq!(t.count_point(10), 30);
        assert_eq!(sorted(t.query_point(39)), [39]);
    }

    #[test]
    fn build_is_order_independent_for_counts() {
        let forward = [
            Interval::new(0, 10),
            Interval::new(5, 15),
            Interval::new(12, 20),
        ];
        let reversed = [
            Interval::new(12, 20),
            Interval::new(5, 15),
            Interval::new(0, 10),
        ];
        let tf = IntervalTree::build(&forward);
        let tr = IntervalTree::build(&reversed);
        // Index assignment differs between layouts, so compare covered counts.
        assert_eq!(tf.count_point(7), tr.count_point(7));
        assert_eq!(tf.count_point(13), tr.count_point(13));
    }
}
