//! Keyed list reconciliation with minimal moves.
//!
//! When a reactive `For`/list re-renders, we want to **reuse** existing child
//! nodes whose key is unchanged instead of destroying and recreating them
//! (which would churn ECS archetypes and discard per-item state). This module
//! computes, from the old key order and the new key order, the smallest set of
//! operations that turns one into the other.
//!
//! The move set is minimised the same way Vue 3 and `SolidJS` do it: find the
//! longest increasing subsequence (LIS) of reused items by their old position;
//! those items are already in the right relative order and stay put, and only
//! the remaining reused items need to move. This keeps "cost ∝ change", one of
//! Loom's core performance contracts.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// One step of a reconciliation plan, emitted in **new** order (one entry per
/// item in the new key list).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffOp {
    /// Reuse the old item at `old_index`; its relative order is unchanged so no
    /// move is required.
    Keep {
        /// Index into the old key list.
        old_index: usize,
    },
    /// Reuse the old item at `old_index`, but it must be repositioned to land
    /// at this slot.
    Move {
        /// Index into the old key list.
        old_index: usize,
    },
    /// Create a brand new item for the new key at `new_index`.
    Create {
        /// Index into the new key list.
        new_index: usize,
    },
}

/// The full plan produced by [`diff_keyed`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Diff {
    /// One op per new slot, in new order.
    pub ops: Vec<DiffOp>,
    /// Old indices whose key is absent from the new list; their nodes should be
    /// removed.
    pub removals: Vec<usize>,
}

impl Diff {
    /// Number of reused items that must be moved.
    pub fn move_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, DiffOp::Move { .. }))
            .count()
    }

    /// Number of newly created items.
    pub fn create_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, DiffOp::Create { .. }))
            .count()
    }
}

/// Computes a minimal-move reconciliation plan from `old_keys` to `new_keys`.
///
/// Keys are matched by equality and are assumed unique within each list; if a
/// key repeats, the last occurrence in `old_keys` wins the match. The returned
/// [`Diff`] reuses every key present in both lists, creates keys new to
/// `new_keys`, and removes keys dropped from `old_keys`.
pub fn diff_keyed<K: Ord + Clone>(old_keys: &[K], new_keys: &[K]) -> Diff {
    // Map each old key to its position.
    let mut old_index_of: BTreeMap<K, usize> = BTreeMap::new();
    for (i, key) in old_keys.iter().enumerate() {
        old_index_of.insert(key.clone(), i);
    }

    // For every new slot, resolve the old index it reuses (if any).
    let mut new_to_old: Vec<Option<usize>> = Vec::with_capacity(new_keys.len());
    let mut reused_old: Vec<bool> = alloc::vec![false; old_keys.len()];
    for key in new_keys {
        match old_index_of.get(key) {
            Some(&old_index) => {
                new_to_old.push(Some(old_index));
                reused_old[old_index] = true;
            }
            None => new_to_old.push(None),
        }
    }

    // Anything not reused is removed, reported in ascending old order.
    let removals: Vec<usize> = (0..old_keys.len()).filter(|&i| !reused_old[i]).collect();

    // Positions (in new order) that reuse an old node, paired with that old
    // index. The LIS over the old indices is the stay-put set.
    let reused_positions: Vec<usize> = (0..new_to_old.len())
        .filter(|&p| new_to_old[p].is_some())
        .collect();
    let reused_old_indices: Vec<usize> = reused_positions
        .iter()
        .map(|&p| new_to_old[p].expect("filtered to Some"))
        .collect();
    let lis = longest_increasing_subsequence(&reused_old_indices);
    // Mark which reused positions are part of the LIS (keep), the rest move.
    let mut keep = alloc::vec![false; new_to_old.len()];
    for &lis_idx in &lis {
        keep[reused_positions[lis_idx]] = true;
    }

    let mut ops = Vec::with_capacity(new_to_old.len());
    for (new_index, slot) in new_to_old.iter().enumerate() {
        match slot {
            Some(old_index) => {
                if keep[new_index] {
                    ops.push(DiffOp::Keep {
                        old_index: *old_index,
                    });
                } else {
                    ops.push(DiffOp::Move {
                        old_index: *old_index,
                    });
                }
            }
            None => ops.push(DiffOp::Create { new_index }),
        }
    }

    Diff { ops, removals }
}

/// Returns the indices (into `values`) of a longest strictly-increasing
/// subsequence, in ascending order. `O(n log n)`.
fn longest_increasing_subsequence(values: &[usize]) -> Vec<usize> {
    if values.is_empty() {
        return Vec::new();
    }
    // `tails[k]` = index into `values` of the smallest tail of an increasing
    // subsequence of length `k + 1`. `prev` reconstructs the chain.
    let mut tails: Vec<usize> = Vec::new();
    let mut prev: Vec<Option<usize>> = alloc::vec![None; values.len()];

    for i in 0..values.len() {
        let v = values[i];
        // Binary search for the first tail whose value is >= v.
        let mut lo = 0usize;
        let mut hi = tails.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if values[tails[mid]] < v {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > 0 {
            prev[i] = Some(tails[lo - 1]);
        }
        if lo == tails.len() {
            tails.push(i);
        } else {
            tails[lo] = i;
        }
    }

    // Reconstruct from the last tail backwards.
    let mut result = Vec::with_capacity(tails.len());
    let mut cursor = tails.last().copied();
    while let Some(i) = cursor {
        result.push(i);
        cursor = prev[i];
    }
    result.reverse();
    result
}
