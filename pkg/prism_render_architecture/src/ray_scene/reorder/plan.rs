//! Deterministic reorder plan: the `CPU` golden a `GPU` radix sort reproduces.
//!
//! [`plan_reorder`] takes the per-ray [`CoherenceKey`]s produced by
//! [`super::sort_key`] and emits a **stable** permutation that groups coherent
//! rays into contiguous runs, plus the batch table a dispatcher consumes to
//! launch one coherent shading group per run.
//!
//! The permutation is produced by a *stable* sort on the key value, so rays
//! that share a key keep their original (ascending-index) relative order. This
//! determinism is the whole point: a hardware Shader-Execution-Reordering pass
//! or a `GPU` radix sort can be validated element-by-element against this plan
//! (same inputs ⇒ identical `order`), and replays are bit-reproducible across
//! platforms because only integer key comparisons are involved.
//!
//! # Batching
//!
//! Rays are grouped by the high bits of their key: `prefix = raw >> batch_shift`.
//! Choosing `batch_shift == layout.material_shift()` groups purely by
//! material / hit-group (the coarsest, highest-value coherence axis); a smaller
//! shift refines batches by direction and then spatial locality. A shift of
//! `0` makes every distinct key its own batch; a shift `>= 64` collapses every
//! ray into one batch (the whole stream shares the empty prefix `0`).
//!
//! # References
//! - NVIDIA, *Shader Execution Reordering* (Ada / `OptiX`) white paper.
//! - Meister et al., *A Survey on Bounding Volume Hierarchies for Ray Tracing*,
//!   EG 2021 (ray sorting / stream compaction for coherence).

use super::sort_key::CoherenceKey;

/// One contiguous run of coherent rays in the reordered stream.
///
/// `start .. start + len` indexes into [`ReorderPlan::order`]; every ray in the
/// run shares the same high-bit `key_prefix`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoherentBatch {
    /// Shared high-bit key prefix (`raw >> batch_shift`) identifying the batch.
    pub key_prefix: u64,
    /// Offset of the first ray of this batch within [`ReorderPlan::order`].
    pub start: u32,
    /// Number of rays in this batch.
    pub len: u32,
}

/// Aggregate statistics describing a [`ReorderPlan`] (telemetry / budgeting).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReorderStats {
    /// Total rays planned (equals `order.len()`).
    pub ray_count: u32,
    /// Number of coherent batches produced.
    pub batch_count: u32,
    /// Largest single batch size (0 when there are no rays).
    pub largest_batch: u32,
    /// Count of batches holding exactly one ray (coherence-divergence proxy).
    pub singleton_batches: u32,
}

/// The deterministic result of [`plan_reorder`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReorderPlan {
    /// Stable-sorted permutation of the input indices (`order[i]` is the source
    /// ray index placed at reordered slot `i`).
    pub order: Vec<u32>,
    /// Coherent runs over [`ReorderPlan::order`], in ascending `key_prefix`.
    pub batches: Vec<CoherentBatch>,
    /// Aggregate statistics.
    pub stats: ReorderStats,
}

impl ReorderPlan {
    /// `true` when the plan carries no rays.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Iterates the source ray indices of `batch` in reordered order.
    ///
    /// Returns an empty slice if the batch bounds fall outside `order` (a
    /// defensive guard; well-formed plans never trip it).
    #[must_use]
    pub fn batch_indices(&self, batch: &CoherentBatch) -> &[u32] {
        let start = batch.start as usize;
        let end = start.saturating_add(batch.len as usize);
        self.order.get(start..end).unwrap_or(&[])
    }
}

/// Drops the low `batch_shift` bits of a key to form its batch prefix.
///
/// A shift `>= 64` would be undefined behaviour for the `>>` operator, so it is
/// clamped to a single all-rays bucket (prefix `0`).
#[must_use]
fn batch_prefix(raw: u64, batch_shift: u32) -> u64 {
    if batch_shift >= 64 {
        0
    } else {
        raw >> batch_shift
    }
}

/// Builds the deterministic reorder plan for `keys`, grouping by the key prefix
/// `raw >> batch_shift`.
///
/// * The permutation is a **stable** sort on the key value: equal keys keep
///   their ascending source-index order, so the output is reproducible.
/// * `batch_shift` selects the coherence granularity (see the module docs);
///   `>= 64` collapses everything into one batch, `0` splits by exact key.
/// * An empty input yields an empty plan (empty `order`, no batches, zero
///   stats).
#[must_use]
pub fn plan_reorder(keys: &[CoherenceKey], batch_shift: u32) -> ReorderPlan {
    if keys.is_empty() {
        return ReorderPlan::default();
    }

    // Stable sort of indices by key value. `sort_by_key` is a stable sort, so
    // ties retain ascending source index — the determinism contract.
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    order.sort_by_key(|&i| keys[i as usize].raw());

    finalize_plan(keys, order, batch_shift)
}

/// Builds the batch table + stats for an already-ordered permutation.
///
/// Shared by [`plan_reorder`] (comparison sort) and
/// [`super::radix::plan_reorder_radix`] (`GPU`-faithful radix sort): both must
/// produce the *same* [`ReorderPlan`] from the same stable `order`, so the
/// batching lives in one place. `order` must be a permutation of
/// `0..keys.len()` and must be non-empty (callers short-circuit empty input).
#[must_use]
pub(super) fn finalize_plan(
    keys: &[CoherenceKey],
    order: Vec<u32>,
    batch_shift: u32,
) -> ReorderPlan {
    // Walk the sorted order grouping consecutive equal prefixes into runs.
    let mut batches: Vec<CoherentBatch> = Vec::new();
    let mut largest_batch: u32 = 0;
    let mut singleton_batches: u32 = 0;

    let mut run_start: u32 = 0;
    let mut run_prefix = batch_prefix(keys[order[0] as usize].raw(), batch_shift);

    let flush = |batches: &mut Vec<CoherentBatch>,
                 largest: &mut u32,
                 singletons: &mut u32,
                 prefix: u64,
                 start: u32,
                 end: u32| {
        let len = end - start;
        if len == 1 {
            *singletons += 1;
        }
        if len > *largest {
            *largest = len;
        }
        batches.push(CoherentBatch {
            key_prefix: prefix,
            start,
            len,
        });
    };

    for slot in 1..order.len() as u32 {
        let prefix = batch_prefix(keys[order[slot as usize] as usize].raw(), batch_shift);
        if prefix != run_prefix {
            flush(
                &mut batches,
                &mut largest_batch,
                &mut singleton_batches,
                run_prefix,
                run_start,
                slot,
            );
            run_start = slot;
            run_prefix = prefix;
        }
    }
    flush(
        &mut batches,
        &mut largest_batch,
        &mut singleton_batches,
        run_prefix,
        run_start,
        order.len() as u32,
    );

    let stats = ReorderStats {
        ray_count: order.len() as u32,
        batch_count: batches.len() as u32,
        largest_batch,
        singleton_batches,
    };

    ReorderPlan {
        order,
        batches,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::super::sort_key::{CoherenceKey, CoherenceKeyLayout};
    use super::*;

    fn layout() -> CoherenceKeyLayout {
        CoherenceKeyLayout::balanced([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).unwrap()
    }

    fn keys(raws: &[u64]) -> Vec<CoherenceKey> {
        raws.iter().copied().map(CoherenceKey).collect()
    }

    fn is_permutation(order: &[u32], n: usize) -> bool {
        if order.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for &i in order {
            let i = i as usize;
            if i >= n || seen[i] {
                return false;
            }
            seen[i] = true;
        }
        true
    }

    #[test]
    fn empty_input_yields_empty_plan() {
        let plan = plan_reorder(&[], 0);
        assert!(plan.is_empty());
        assert!(plan.order.is_empty());
        assert!(plan.batches.is_empty());
        assert_eq!(plan.stats, ReorderStats::default());
    }

    #[test]
    fn order_is_a_permutation() {
        let k = keys(&[7, 3, 3, 9, 1, 3]);
        let plan = plan_reorder(&k, 0);
        assert!(is_permutation(&plan.order, k.len()));
    }

    #[test]
    fn order_is_non_decreasing_by_key() {
        let k = keys(&[7, 3, 3, 9, 1, 3]);
        let plan = plan_reorder(&k, 0);
        let sorted: Vec<u64> = plan.order.iter().map(|&i| k[i as usize].raw()).collect();
        assert!(sorted.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn equal_keys_keep_ascending_source_order() {
        // Three rays share key 3 at source indices 1, 2, 5; a stable sort must
        // emit them in that order.
        let k = keys(&[7, 3, 3, 9, 1, 3]);
        let plan = plan_reorder(&k, 0);
        let threes: Vec<u32> = plan
            .order
            .iter()
            .copied()
            .filter(|&i| k[i as usize].raw() == 3)
            .collect();
        assert_eq!(threes, vec![1, 2, 5]);
    }

    #[test]
    fn exact_key_batches_when_shift_zero() {
        let k = keys(&[7, 3, 3, 9, 1, 3]);
        let plan = plan_reorder(&k, 0);
        // Distinct keys: 1, 3, 7, 9 -> four batches; the 3-batch holds three.
        assert_eq!(plan.batches.len(), 4);
        assert_eq!(plan.stats.batch_count, 4);
        let three = plan
            .batches
            .iter()
            .find(|b| b.key_prefix == 3)
            .expect("key-3 batch");
        assert_eq!(three.len, 3);
        assert_eq!(plan.batch_indices(three), &[1, 2, 5]);
    }

    #[test]
    fn stats_track_largest_and_singletons() {
        let k = keys(&[7, 3, 3, 9, 1, 3]);
        let plan = plan_reorder(&k, 0);
        assert_eq!(plan.stats.ray_count, 6);
        assert_eq!(plan.stats.largest_batch, 3); // the three 3s
        assert_eq!(plan.stats.singleton_batches, 3); // 1, 7, 9
    }

    #[test]
    fn huge_shift_collapses_to_single_batch() {
        let k = keys(&[7, 3, 3, 9, 1, 3]);
        let plan = plan_reorder(&k, 64);
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.batches[0].key_prefix, 0);
        assert_eq!(plan.batches[0].len, 6);
        assert_eq!(plan.stats.singleton_batches, 0);
        // Order is still a stably-sorted permutation.
        assert!(is_permutation(&plan.order, k.len()));
    }

    #[test]
    fn material_shift_groups_by_material() {
        let l = layout();
        // Two materials, two rays each with differing direction/space so the
        // low bits differ but the material prefix matches.
        let k = vec![
            l.encode(2, [0.0, 1.0, 0.0], [0.1, 0.1, 0.1]),
            l.encode(1, [1.0, 0.0, 0.0], [0.9, 0.2, 0.3]),
            l.encode(2, [0.0, -1.0, 0.0], [0.8, 0.8, 0.8]),
            l.encode(1, [-1.0, 0.0, 0.0], [0.2, 0.5, 0.7]),
        ];
        let plan = plan_reorder(&k, l.material_shift());
        assert_eq!(plan.batches.len(), 2);
        assert_eq!(plan.batches[0].key_prefix, 1);
        assert_eq!(plan.batches[0].len, 2);
        assert_eq!(plan.batches[1].key_prefix, 2);
        assert_eq!(plan.batches[1].len, 2);
        // Every ray in batch 0 really is material 1.
        for &i in plan.batch_indices(&plan.batches[0]) {
            assert_eq!(k[i as usize].raw() >> l.material_shift(), 1);
        }
    }

    #[test]
    fn batches_cover_order_contiguously() {
        let k = keys(&[7, 3, 3, 9, 1, 3, 7, 1]);
        let plan = plan_reorder(&k, 0);
        let mut cursor = 0u32;
        for b in &plan.batches {
            assert_eq!(b.start, cursor);
            cursor += b.len;
        }
        assert_eq!(cursor, plan.order.len() as u32);
    }

    #[test]
    fn single_ray_plan() {
        let plan = plan_reorder(&keys(&[42]), 0);
        assert_eq!(plan.order, vec![0]);
        assert_eq!(plan.batches.len(), 1);
        assert_eq!(plan.batches[0].key_prefix, 42);
        assert_eq!(plan.stats.largest_batch, 1);
        assert_eq!(plan.stats.singleton_batches, 1);
    }
}
