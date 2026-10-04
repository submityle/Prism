//! Karras (2012) binary radix tree over sorted `Morton` keys.
//!
//! Given `n` primitives sorted by `Morton` key, the hierarchy has exactly
//! `n - 1` internal nodes and `n` leaves. Each internal node `i` owns a
//! contiguous range of the sorted array; its split position partitions that
//! range into the two child subtrees. The construction is embarrassingly
//! parallel — every internal node is derived independently from the key array —
//! which is why it maps directly onto a one-thread-per-node `GPU` kernel. This
//! module is the `CPU` reference for that kernel.
//!
//! Ties between equal keys are resolved by augmenting the longest-common-prefix
//! metric with the sorted position, exactly as the reference formulation does,
//! so duplicate `Morton` keys still yield a well-defined, balanced tree.

use alloc::vec;
use alloc::vec::Vec;

/// Reference to a child, distinguishing an internal node from a leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Child {
    /// Internal node at this index into [`RadixTree::internal`].
    Internal(u32),
    /// Leaf covering the single sorted primitive at this position.
    Leaf(u32),
}

/// One internal node: the two children of a split range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InternalNode {
    /// Left (lower sorted position) child.
    pub left: Child,
    /// Right (higher sorted position) child.
    pub right: Child,
}

/// The constructed binary radix tree.
///
/// For `n >= 2` the root is internal node `0`. For `n == 1` there are no
/// internal nodes and the single leaf is the whole tree; callers handle that
/// degenerate shape before building a tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadixTree {
    /// `n - 1` internal nodes; empty when `n < 2`.
    pub internal: Vec<InternalNode>,
    /// Leaf count `n`.
    pub leaf_count: u32,
}

/// Longest common prefix length of the keys at sorted positions `i` and `j`.
///
/// Returns `-1` when `j` is out of range, which makes the range-finding loops
/// terminate at the array boundary. When the two keys are identical the metric
/// is extended by the common prefix of the positions themselves, guaranteeing a
/// strict ordering even for duplicate `Morton` keys.
fn delta(keys: &[u32], i: usize, j: isize) -> i32 {
    if j < 0 {
        return -1;
    }
    let j = j as usize;
    if j >= keys.len() {
        return -1;
    }
    let ki = keys[i];
    let kj = keys[j];
    if ki == kj {
        // Identical keys: break the tie with the 32-bit positions so the metric
        // stays strictly monotonic. Add 32 (the key width used here) so any
        // real key difference always outranks a position-only difference.
        32 + ((i as u32) ^ (j as u32)).leading_zeros() as i32
    } else {
        (ki ^ kj).leading_zeros() as i32
    }
}

/// Determines the inclusive sorted range `[first, last]` owned by internal
/// node `i`, following the reference range-finding procedure.
fn determine_range(keys: &[u32], i: usize) -> (usize, usize) {
    let n = keys.len();
    // Direction of the range: +1 extends to higher positions, -1 to lower.
    let delta_right = delta(keys, i, i as isize + 1);
    let delta_left = delta(keys, i, i as isize - 1);
    let direction: isize = if delta_right > delta_left { 1 } else { -1 };

    // Lower bound on the common-prefix length inside this node's range.
    let delta_min = delta(keys, i, i as isize - direction);

    // Exponentially probe for an upper bound on the range length.
    let mut l_max: isize = 2;
    while delta(keys, i, i as isize + l_max * direction) > delta_min {
        l_max *= 2;
    }

    // Binary search the exact range length.
    let mut l: isize = 0;
    let mut t = l_max / 2;
    while t >= 1 {
        if delta(keys, i, i as isize + (l + t) * direction) > delta_min {
            l += t;
        }
        t /= 2;
    }

    let j = i as isize + l * direction;
    let first = i.min(j as usize);
    let last = i.max(j as usize);
    debug_assert!(last < n);
    (first, last)
}

/// Finds the split position of the inclusive range `[first, last]`: the largest
/// position `s` with `first <= s < last` such that the left subtree covers
/// `[first, s]` and the right subtree covers `[s + 1, last]`.
fn find_split(keys: &[u32], first: usize, last: usize) -> usize {
    let common_prefix = delta(keys, first, last as isize);
    let mut split = first;
    let mut step = last - first;
    loop {
        step = step.div_ceil(2);
        let candidate = split + step;
        if candidate < last && delta(keys, first, candidate as isize) > common_prefix {
            split = candidate;
        }
        if step <= 1 {
            break;
        }
    }
    split
}

/// Builds the radix tree over `keys`, which must be sorted ascending.
///
/// Requires `keys.len() >= 2`; callers handle the empty and single-primitive
/// cases directly because those shapes have no internal nodes.
#[must_use]
pub fn build(keys: &[u32]) -> RadixTree {
    let n = keys.len();
    debug_assert!(n >= 2);
    let mut internal = vec![
        InternalNode {
            left: Child::Leaf(0),
            right: Child::Leaf(0),
        };
        n - 1
    ];
    for (i, node) in internal.iter_mut().enumerate() {
        let (first, last) = determine_range(keys, i);
        let split = find_split(keys, first, last);
        let left = if split == first {
            Child::Leaf(split as u32)
        } else {
            Child::Internal(split as u32)
        };
        let right = if split + 1 == last {
            Child::Leaf((split + 1) as u32)
        } else {
            Child::Internal((split + 1) as u32)
        };
        *node = InternalNode { left, right };
    }
    RadixTree {
        internal,
        leaf_count: n as u32,
    }
}
