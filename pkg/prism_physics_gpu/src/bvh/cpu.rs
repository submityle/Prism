//! Sequential golden twin for the whole `LBVH` build.
//!
//! [`cpu_build_lbvh`] runs the exact stages the `GPU` pipeline runs, in the same
//! order and with the same integer arithmetic: quantise each leaf centroid to a
//! 30-bit Morton code, stably sort the primitive indices by code, build Karras'
//! binary radix tree over the sorted codes, then union child boxes bottom-up
//! into internal-node bounds. Because every stage is either an integer
//! permutation or an exact `min`/`max` reduction, a real-device parity test that
//! matches this twin is bit-for-bit evidence the kernels are a faithful port.
//!
//! # Node layout
//!
//! For `n` leaves the tree has `n - 1` internal nodes and `n` leaves. Nodes are
//! addressed in one *encoded* index space: an id below [`Lbvh::num_internal`] is
//! internal node `id`; an id at or above it is leaf `id - num_internal`. Child
//! links in [`Lbvh::left`]/[`Lbvh::right`] and every entry of [`Lbvh::parent`]
//! use this encoding; [`Lbvh::root`] is the encoded id of the tree root. The
//! root's parent is [`NO_PARENT`].
//!
//! # Provenance
//!
//! The binary radix tree construction is Karras, "Maximizing Parallelism in the
//! Construction of BVHs, Octrees, and k-d Trees" (High Performance Graphics
//! 2012). No Unreal Engine source or derived code.

use glam::Vec3;

use super::config::{Aabb, SceneBounds};
use super::morton::cpu_morton_code;

/// Sentinel parent for the root node (it has no parent).
pub const NO_PARENT: u32 = u32::MAX;

/// The complete linear `BVH`: a Morton ordering plus a binary radix tree with
/// per-node bounds.
///
/// See the [module docs](self) for the encoded index space that child and
/// parent links use.
#[derive(Clone, Debug, PartialEq)]
pub struct Lbvh {
    /// Number of leaves (input primitives).
    pub num_leaves: usize,
    /// Number of internal nodes (`num_leaves - 1`, or `0` when empty).
    pub num_internal: usize,
    /// Encoded id of the root node, or [`NO_PARENT`] for an empty tree.
    pub root: u32,
    /// Primitive indices in Morton order; `sorted_indices[i]` is the input
    /// primitive that became leaf `i`.
    pub sorted_indices: Vec<u32>,
    /// Morton codes aligned with [`Lbvh::sorted_indices`] (ascending).
    pub sorted_codes: Vec<u32>,
    /// Left child (encoded id) of each internal node.
    pub left: Vec<u32>,
    /// Right child (encoded id) of each internal node.
    pub right: Vec<u32>,
    /// Parent (encoded id) of every node, indexed by encoded id; the root's
    /// entry is [`NO_PARENT`].
    pub parent: Vec<u32>,
    /// Bounds of each internal node, indexed by internal id.
    pub internal_aabb: Vec<Aabb>,
    /// Leaf bounds in Morton order; `leaf_aabb[i]` is the box of leaf `i`.
    pub leaf_aabb: Vec<Aabb>,
}

impl Lbvh {
    /// The empty tree (no leaves).
    #[must_use]
    fn empty() -> Lbvh {
        Lbvh {
            num_leaves: 0,
            num_internal: 0,
            root: NO_PARENT,
            sorted_indices: Vec::new(),
            sorted_codes: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            parent: Vec::new(),
            internal_aabb: Vec::new(),
            leaf_aabb: Vec::new(),
        }
    }

    /// Whether an encoded child id refers to a leaf rather than an internal
    /// node.
    #[must_use]
    pub fn is_leaf(&self, encoded: u32) -> bool {
        (encoded as usize) >= self.num_internal
    }
}

/// Builds the complete `LBVH` over `boxes`, the golden twin of the `GPU` build.
///
/// Returns an empty tree for an empty slice and a single-leaf tree (no internal
/// nodes, `root` naming leaf 0) for one box.
#[must_use]
pub fn cpu_build_lbvh(boxes: &[Aabb]) -> Lbvh {
    let n = boxes.len();
    if n == 0 {
        return Lbvh::empty();
    }

    // Stage 1: Morton code per centroid within the scene bounds.
    let bounds = SceneBounds::of(boxes).unwrap_or(SceneBounds {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
    });
    let codes: Vec<u32> = boxes
        .iter()
        .map(|b| cpu_morton_code(b.centroid(), &bounds))
        .collect();

    // Stage 2: stable ascending sort of primitive indices by code. Rust's
    // `sort_by_key` is stable, so equal codes keep ascending primitive order,
    // exactly matching the stable radix sort of an identity payload on device.
    let mut sorted_indices: Vec<u32> = (0..n as u32).collect();
    sorted_indices.sort_by_key(|&i| codes[i as usize]);
    let sorted_codes: Vec<u32> = sorted_indices.iter().map(|&i| codes[i as usize]).collect();
    let leaf_aabb: Vec<Aabb> = sorted_indices.iter().map(|&i| boxes[i as usize]).collect();

    let num_internal = n - 1;
    let total_nodes = num_internal + n;
    let mut parent = vec![NO_PARENT; total_nodes];

    // Single-leaf tree: no internal nodes; the root is leaf 0.
    if num_internal == 0 {
        return Lbvh {
            num_leaves: n,
            num_internal,
            root: 0,
            sorted_indices,
            sorted_codes,
            left: Vec::new(),
            right: Vec::new(),
            parent,
            internal_aabb: Vec::new(),
            leaf_aabb,
        };
    }

    // Stage 3: Karras binary radix tree. Each internal node derives its range,
    // split, and children independently from the sorted code array.
    let mut left = vec![0u32; num_internal];
    let mut right = vec![0u32; num_internal];
    for i in 0..num_internal {
        let (lc, rc) = karras_children(&sorted_codes, i, num_internal);
        left[i] = lc;
        right[i] = rc;
        parent[lc as usize] = i as u32;
        parent[rc as usize] = i as u32;
    }

    // Stage 4: bottom-up bounds via the same atomic parent-climb the kernel
    // runs. Only the second child to reach a node unions the two subtree boxes,
    // so each internal box is written exactly once from finished children.
    let internal_aabb = climb_bounds(num_internal, &left, &right, &parent, &leaf_aabb);

    Lbvh {
        num_leaves: n,
        num_internal,
        root: 0,
        sorted_indices,
        sorted_codes,
        left,
        right,
        parent,
        internal_aabb,
        leaf_aabb,
    }
}

/// The bounds of an encoded child, reading from the leaf boxes or the
/// internal-node boxes computed so far.
#[must_use]
fn child_aabb(encoded: u32, num_internal: usize, internal: &[Aabb], leaf: &[Aabb]) -> Aabb {
    let idx = encoded as usize;
    if idx >= num_internal {
        leaf[idx - num_internal]
    } else {
        internal[idx]
    }
}

/// Fills every internal node's bounds by unioning child boxes bottom-up.
///
/// Each leaf climbs toward the root; a node is finished (and its box written)
/// only by the second child to arrive, so both subtrees are complete first.
#[must_use]
fn climb_bounds(
    num_internal: usize,
    left: &[u32],
    right: &[u32],
    parent: &[u32],
    leaf_aabb: &[Aabb],
) -> Vec<Aabb> {
    let zero = Aabb {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
    };
    let mut internal = vec![zero; num_internal];
    let mut ready = vec![0u32; num_internal];
    for leaf in 0..leaf_aabb.len() {
        let leaf_id = (num_internal + leaf) as u32;
        let mut node = parent[leaf_id as usize];
        loop {
            let idx = node as usize;
            let arrived = ready[idx];
            ready[idx] = arrived + 1;
            if arrived == 0 {
                // First child here; the second finishes this node later.
                break;
            }
            let a = child_aabb(left[idx], num_internal, &internal, leaf_aabb);
            let b = child_aabb(right[idx], num_internal, &internal, leaf_aabb);
            internal[idx] = a.union(&b);
            let p = parent[idx];
            if p == NO_PARENT {
                break;
            }
            node = p;
        }
    }
    internal
}

/// Karras' delta: the length of the common Morton-code prefix of `sorted[i]`
/// and `sorted[j]`, or `-1` when `j` is out of range.
///
/// Equal codes are broken by the 32-bit index so every leaf pair has a distinct
/// delta, which keeps the tree well defined under duplicate Morton codes.
#[must_use]
fn delta(sorted: &[u32], i: i64, j: i64) -> i64 {
    let n = sorted.len() as i64;
    if j < 0 || j >= n {
        return -1;
    }
    let ki = sorted[i as usize];
    let kj = sorted[j as usize];
    if ki == kj {
        32 + i64::from((i as u32 ^ j as u32).leading_zeros())
    } else {
        i64::from((ki ^ kj).leading_zeros())
    }
}

/// Computes the encoded left and right children of internal node `i`.
#[must_use]
fn karras_children(sorted: &[u32], i: usize, num_internal: usize) -> (u32, u32) {
    let ii = i as i64;

    // Direction of the range this node covers (+1 forwards, -1 backwards).
    let d = (delta(sorted, ii, ii + 1) - delta(sorted, ii, ii - 1)).signum();

    // Upper bound on the range length, then binary search for its far end.
    let delta_min = delta(sorted, ii, ii - d);
    let mut l_max: i64 = 2;
    while delta(sorted, ii, ii + l_max * d) > delta_min {
        l_max *= 2;
    }
    let mut l: i64 = 0;
    let mut t = l_max / 2;
    while t >= 1 {
        if delta(sorted, ii, ii + (l + t) * d) > delta_min {
            l += t;
        }
        t /= 2;
    }
    let j = ii + l * d;

    // Binary search for the split position within the range.
    let delta_node = delta(sorted, ii, j);
    let mut s: i64 = 0;
    let mut t = (l + 1) / 2;
    loop {
        if delta(sorted, ii, ii + (s + t) * d) > delta_node {
            s += t;
        }
        if t == 1 {
            break;
        }
        t = (t + 1) / 2;
    }
    let gamma = ii + s * d + d.min(0);

    // A child is a leaf when its side of the split reaches the range end.
    let range_lo = ii.min(j);
    let range_hi = ii.max(j);
    let num_internal = num_internal as i64;
    let left = if range_lo == gamma {
        (num_internal + gamma) as u32
    } else {
        gamma as u32
    };
    let right = if range_hi == gamma + 1 {
        (num_internal + gamma + 1) as u32
    } else {
        (gamma + 1) as u32
    };
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::{cpu_build_lbvh, Lbvh, NO_PARENT};
    use crate::bvh::config::Aabb;
    use glam::Vec3;

    /// A unit box centred on `(x, y, z)`.
    fn box_at(x: f32, y: f32, z: f32) -> Aabb {
        let c = Vec3::new(x, y, z);
        let h = Vec3::splat(0.5);
        Aabb::new(c - h, c + h)
    }

    /// Recomputes the union of every leaf box the naive way.
    fn union_all(boxes: &[Aabb]) -> Aabb {
        let mut u = boxes[0];
        for b in &boxes[1..] {
            u = u.union(b);
        }
        u
    }

    /// The bounds of an encoded child within a built tree.
    fn child_box(tree: &Lbvh, encoded: u32) -> Aabb {
        if tree.is_leaf(encoded) {
            tree.leaf_aabb[encoded as usize - tree.num_internal]
        } else {
            tree.internal_aabb[encoded as usize]
        }
    }

    /// Walks the tree from `root`, checking every leaf is reached exactly once
    /// and that each internal box contains both children's boxes.
    fn check_tree(tree: &Lbvh) {
        let mut seen = vec![false; tree.num_leaves];
        let mut stack = vec![tree.root];
        let mut visited_internal = 0usize;
        while let Some(node) = stack.pop() {
            if tree.is_leaf(node) {
                let leaf = node as usize - tree.num_internal;
                assert!(!seen[leaf], "leaf {leaf} visited twice");
                seen[leaf] = true;
                continue;
            }
            visited_internal += 1;
            let idx = node as usize;
            let l = tree.left[idx];
            let r = tree.right[idx];
            assert_eq!(tree.parent[l as usize], node);
            assert_eq!(tree.parent[r as usize], node);
            let bb = tree.internal_aabb[idx];
            let lb = child_box(tree, l);
            let rb = child_box(tree, r);
            assert_eq!(bb, lb.union(&rb));
            stack.push(l);
            stack.push(r);
        }
        assert!(seen.iter().all(|&s| s), "every leaf reachable");
        assert_eq!(visited_internal, tree.num_internal);
    }

    #[test]
    fn empty_input_is_empty_tree() {
        let tree = cpu_build_lbvh(&[]);
        assert_eq!(tree.num_leaves, 0);
        assert_eq!(tree.num_internal, 0);
        assert_eq!(tree.root, NO_PARENT);
    }

    #[test]
    fn single_leaf_root_is_the_leaf() {
        let b = box_at(1.0, 2.0, 3.0);
        let tree = cpu_build_lbvh(&[b]);
        assert_eq!(tree.num_leaves, 1);
        assert_eq!(tree.num_internal, 0);
        assert_eq!(tree.root, 0);
        assert!(tree.is_leaf(tree.root));
        assert_eq!(tree.leaf_aabb, vec![b]);
        assert_eq!(tree.parent, vec![NO_PARENT]);
    }

    #[test]
    fn two_leaves_make_one_internal_root() {
        let boxes = [box_at(0.0, 0.0, 0.0), box_at(4.0, 0.0, 0.0)];
        let tree = cpu_build_lbvh(&boxes);
        assert_eq!(tree.num_internal, 1);
        assert_eq!(tree.root, 0);
        assert!(!tree.is_leaf(tree.root));
        check_tree(&tree);
        assert_eq!(tree.internal_aabb[0], union_all(&boxes));
    }

    #[test]
    fn line_of_boxes_builds_valid_tree() {
        let boxes: Vec<Aabb> = (0..17).map(|k| box_at(k as f32, 0.0, 0.0)).collect();
        let tree = cpu_build_lbvh(&boxes);
        assert_eq!(tree.num_leaves, 17);
        assert_eq!(tree.num_internal, 16);
        check_tree(&tree);
        assert_eq!(tree.internal_aabb[tree.root as usize], union_all(&boxes));
        // Morton codes are non-decreasing along the sort.
        for w in tree.sorted_codes.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn duplicate_codes_still_reach_every_leaf() {
        // All boxes coincide, so all Morton codes are identical: the index
        // tiebreak in `delta` must still yield a valid binary tree.
        let boxes: Vec<Aabb> = (0..9).map(|_| box_at(2.0, 2.0, 2.0)).collect();
        let tree = cpu_build_lbvh(&boxes);
        check_tree(&tree);
        assert_eq!(tree.internal_aabb[tree.root as usize], boxes[0]);
    }

    #[test]
    fn degenerate_axis_builds_valid_tree() {
        // A perfectly flat scene (z extent zero) must still quantise and build.
        let boxes: Vec<Aabb> = (0..12)
            .map(|k| {
                let c = Vec3::new(k as f32, (k % 3) as f32, 0.0);
                Aabb::new(c, c)
            })
            .collect();
        let tree = cpu_build_lbvh(&boxes);
        check_tree(&tree);
    }
}
