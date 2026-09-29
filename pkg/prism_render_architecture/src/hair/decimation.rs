//! Importance-weighted, pop-free strand decimation order for density LOD.
//!
//! Continuous hair LOD keeps fewer render strands as a groom recedes. To avoid
//! popping, the strands kept at a low count *must be a subset* of those kept at
//! a higher count — the selection has to be **nested**. The stride sample in
//! [`super::interpolation::decimate_bindings`] is deterministic and cheap but
//! only nested when target counts share integer factors; at arbitrary counts it
//! swaps which strands survive and visibly pops.
//!
//! This module fixes that by precomputing a single **decimation order**: a
//! stable permutation of strand indices whose every prefix is a valid kept set.
//! Keeping the first `k` entries for any `k` is, by construction, a subset of
//! keeping the first `k + 1`, so density LOD can slide continuously with zero
//! pop. The order is also **importance-weighted** — long, high-curvature, or
//! authored-priority strands are ranked to survive to the lowest counts, the
//! same density-LOD bias `UE5` Groom and `HairWorks` apply — while a per-strand
//! hash jitter decorrelates the ranking spatially so heavy decimation thins the
//! groom evenly instead of carving bald patches.
//!
//! Everything is deterministic, array-in / array-out, and panic-free: empty or
//! mismatched inputs yield an empty order rather than crashing. The crate has no
//! `libm`, so importance uses only `+`, `*`, and `sqrt` (via vector length) —
//! no transcendental calls.

use alloc::vec::Vec;

use super::interpolation::{hash_to_unit, resolved_render_count, RenderStrandBinding, Vec3};
use super::lod::HairLodDecision;

/// Hash sub-key salt for the per-strand decimation jitter, kept distinct from
/// the clump/curl/jitter salts in [`super::interpolation`] so the orderings do
/// not correlate.
const DECIMATION_JITTER_KEY: u32 = 0x00DE_C1AA;

/// Relative blend weights that fold per-strand metrics into one importance
/// scalar. Each weight scales its already-normalized metric; the blend is then
/// renormalized by the weight sum, so only the *ratios* between weights matter.
#[derive(Clone, Copy, Debug)]
pub struct ImportanceWeights {
    /// Weight on normalized arc length (longer strands read more on silhouette).
    pub length: f32,
    /// Weight on normalized accumulated curvature (curls / flyaways matter).
    pub curvature: f32,
    /// Weight on the authored per-strand priority (artist override).
    pub authored: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        // Length and authored priority dominate; curvature is a moderate boost.
        Self {
            length: 1.0,
            curvature: 0.5,
            authored: 1.0,
        }
    }
}

/// Accumulated arc length of a strand polyline (`0` for fewer than two points).
#[must_use]
pub fn strand_arc_length(points: &[Vec3]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for i in 1..points.len() {
        total += (points[i] - points[i - 1]).length();
    }
    total
}

/// Accumulated turning of a strand polyline, transcendental-free.
///
/// At each interior vertex the incoming and outgoing unit tangents contribute
/// `1 - dot(t_in, t_out)` (in `0..=2`, rising with the bend). The sum is `0` for
/// a straight strand and grows with curls; fewer than three points yield `0`.
#[must_use]
pub fn strand_curvature(points: &[Vec3]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for i in 1..points.len() - 1 {
        let t_in = (points[i] - points[i - 1]).normalize_or(Vec3::ZERO);
        let t_out = (points[i + 1] - points[i]).normalize_or(Vec3::ZERO);
        total += 1.0 - t_in.dot(t_out);
    }
    total
}

/// Folds per-strand metrics into a normalized importance in `0..=1`.
///
/// `lengths`, `curvatures`, and `authored` must share a length; a mismatch
/// yields an empty vector. Length and curvature are each normalized by their
/// maximum across the set (so scene scale drops out); `authored` is taken as an
/// already-normalized artist priority and clamped to `0..=1`. The three are
/// blended by `weights` and renormalized by the weight sum. When every metric
/// or weight is zero the importance is `0`, which the ranking treats as the
/// lowest priority.
#[must_use]
pub fn compute_importance(
    lengths: &[f32],
    curvatures: &[f32],
    authored: &[f32],
    weights: ImportanceWeights,
) -> Vec<f32> {
    let n = lengths.len();
    if curvatures.len() != n || authored.len() != n {
        return Vec::new();
    }
    let mut max_len = 0.0_f32;
    let mut max_curv = 0.0_f32;
    for i in 0..n {
        if lengths[i] > max_len {
            max_len = lengths[i];
        }
        if curvatures[i] > max_curv {
            max_curv = curvatures[i];
        }
    }
    let weight_sum = weights.length + weights.curvature + weights.authored;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if weight_sum <= 0.0 {
            out.push(0.0);
            continue;
        }
        let len_n = if max_len > 0.0 {
            lengths[i] / max_len
        } else {
            0.0
        };
        let curv_n = if max_curv > 0.0 {
            curvatures[i] / max_curv
        } else {
            0.0
        };
        let auth = authored[i].clamp(0.0, 1.0);
        let blended = weights.length * len_n + weights.curvature * curv_n + weights.authored * auth;
        out.push((blended / weight_sum).clamp(0.0, 1.0));
    }
    out
}

/// Ranking priority of one strand: importance plus a symmetric hash jitter.
///
/// `jitter` is the peak decorrelation amplitude added to `importance`; the
/// per-strand hash lands in `[-0.5, 0.5) * jitter`. Higher priority survives to
/// lower render counts. `jitter = 0` gives a pure importance ranking (which
/// tends to cluster kept strands); a small positive jitter spreads the thinning
/// evenly while preserving the coarse importance ordering.
#[must_use]
pub fn decimation_priority(seed: u32, importance: f32, jitter: f32) -> f32 {
    let h = hash_to_unit(seed, DECIMATION_JITTER_KEY) - 0.5;
    importance + jitter * h
}

/// Builds the nested decimation order: strand indices sorted by descending
/// priority (ties broken by ascending index for a total, stable order).
///
/// `seeds` and `importances` must share a length; a mismatch yields an empty
/// order. The result is a permutation of `0..seeds.len()`; its first `k` entries
/// are the strands kept at render count `k`, and every prefix nests inside the
/// next.
#[must_use]
pub fn build_decimation_order(seeds: &[u32], importances: &[f32], jitter: f32) -> Vec<u32> {
    let n = seeds.len();
    if importances.len() != n {
        return Vec::new();
    }
    let mut ranked: Vec<(f32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let priority = decimation_priority(seeds[i], importances[i], jitter);
        ranked.push((priority, i as u32));
    }
    // Descending priority; ties resolved by ascending index for a total order.
    ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.into_iter().map(|(_, idx)| idx).collect()
}

/// Selects the first `target` strands of a decimation `order`, emitted in
/// ascending original-index order for coherent downstream processing.
///
/// `out` is cleared first. The selected *set* is `order`'s length-`target`
/// prefix, so it nests across target counts regardless of the ascending
/// re-sort; a `target` of `0` yields an empty list and a `target` at or beyond
/// `order.len()` keeps every index.
pub fn select_nested(order: &[u32], target: usize, out: &mut Vec<u32>) {
    out.clear();
    let keep = target.min(order.len());
    out.extend_from_slice(&order[..keep]);
    out.sort_unstable();
}

/// Importance-weighted, pop-free replacement for
/// [`super::interpolation::decimate_bindings`].
///
/// Ranks the render-strand bindings by [`build_decimation_order`] using each
/// binding's `seed` and the parallel `importances`, then keeps
/// [`resolved_render_count`] of them, emitted in original order. When
/// `importances` does not match `bindings` in length it degrades gracefully to
/// a uniform hash ranking (importance `0` for all), which is still nested and
/// deterministic. `out` is cleared first; a proxy tier (count `0`) yields an
/// empty list and a count at or above the input length keeps every binding in
/// order.
pub fn decimate_bindings_importance(
    bindings: &[RenderStrandBinding],
    importances: &[f32],
    jitter: f32,
    decision: &HairLodDecision,
    out: &mut Vec<RenderStrandBinding>,
) {
    out.clear();
    let target = resolved_render_count(decision) as usize;
    if target == 0 || bindings.is_empty() {
        return;
    }
    let n = bindings.len();
    let mut seeds = Vec::with_capacity(n);
    for binding in bindings {
        seeds.push(binding.seed);
    }
    // Graceful degrade to uniform ranking when importance data is absent.
    let uniform;
    let importance_slice: &[f32] = if importances.len() == n {
        importances
    } else {
        uniform = alloc::vec![0.0_f32; n];
        &uniform
    };
    let order = build_decimation_order(&seeds, importance_slice, jitter);
    let mut kept = Vec::new();
    select_nested(&order, target, &mut kept);
    for &idx in &kept {
        if let Some(&binding) = bindings.get(idx as usize) {
            out.push(binding);
        }
    }
}

/// Default jitter strength for the geometry-free nested decimation path.
///
/// With uniform importance the per-strand priority is `jitter * (hash - 0.5)`,
/// so any positive jitter yields a fully hash-driven ranking; `1.0` keeps the
/// spread in its natural `[-0.5, 0.5]` range.
pub const DEFAULT_DECIMATION_JITTER: f32 = 1.0;

/// Geometry-free, pop-free strand decimation — a drop-in upgrade over the
/// stride sample formerly used by [`super::interpolation::decimate_bindings`].
///
/// Ranks bindings purely by a per-strand hash of their `seed` (uniform
/// importance), producing a nested decimation order whose every prefix is a
/// valid kept set. Unlike a stride sample it never swaps which strands survive
/// as the target count slides, so continuous density LOD stays pop-free at
/// *any* count — not only counts that share integer factors. `out` is cleared
/// first; a proxy tier (count `0`) yields an empty list and a count at or above
/// the input length keeps every binding in order. Use
/// [`decimate_bindings_importance`] instead when strand geometry is available
/// and importance weighting is wanted.
pub fn decimate_bindings_nested(
    bindings: &[RenderStrandBinding],
    decision: &HairLodDecision,
    out: &mut Vec<RenderStrandBinding>,
) {
    decimate_bindings_importance(bindings, &[], DEFAULT_DECIMATION_JITTER, decision, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::{HairGroupHandle, HairLodTier};

    const EPS: f32 = 1e-5;

    fn strand_decision(tier: HairLodTier, render_strands: u32) -> HairLodDecision {
        HairLodDecision {
            handle: HairGroupHandle(0),
            tier,
            render_strands,
            segments_per_strand: 8,
        }
    }

    fn binding(seed: u32) -> RenderStrandBinding {
        RenderStrandBinding {
            guides: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
            root_uv: (0.0, 0.0),
            seed,
        }
    }

    #[test]
    fn arc_length_and_curvature_basics() {
        let straight = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        assert!((strand_arc_length(&straight) - 2.0).abs() < EPS);
        assert!(
            strand_curvature(&straight).abs() < EPS,
            "straight has no turn"
        );

        let bent = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        // 90-degree turn: 1 - dot(+x, +y) = 1.
        assert!((strand_curvature(&bent) - 1.0).abs() < EPS);

        // Degenerate inputs never panic and read as zero.
        assert!(strand_arc_length(&[Vec3::ZERO]).abs() < EPS);
        assert!(strand_curvature(&[Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)]).abs() < EPS);
    }

    #[test]
    fn importance_is_normalized_and_length_biased() {
        let lengths = [1.0, 2.0, 4.0];
        let curvatures = [0.0, 0.0, 0.0];
        let authored = [0.0, 0.0, 0.0];
        let imp = compute_importance(
            &lengths,
            &curvatures,
            &authored,
            ImportanceWeights::default(),
        );
        assert_eq!(imp.len(), 3);
        for &v in &imp {
            assert!((0.0..=1.0).contains(&v), "importance out of range");
        }
        // Longest strand is most important; monotonic with length here.
        assert!(imp[2] > imp[1] && imp[1] > imp[0]);
    }

    #[test]
    fn importance_rejects_length_mismatch() {
        let imp = compute_importance(
            &[1.0, 2.0],
            &[0.0],
            &[0.0, 0.0],
            ImportanceWeights::default(),
        );
        assert!(imp.is_empty());
    }

    #[test]
    fn order_is_a_permutation() {
        let seeds = [10, 20, 30, 40, 50];
        let importances = [0.1, 0.9, 0.5, 0.2, 0.7];
        let order = build_decimation_order(&seeds, &importances, 0.25);
        assert_eq!(order.len(), 5);
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, alloc::vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn pure_importance_order_ranks_by_importance() {
        let seeds = [1, 2, 3, 4];
        let importances = [0.2, 0.8, 0.5, 0.1];
        // Zero jitter => pure importance ranking.
        let order = build_decimation_order(&seeds, &importances, 0.0);
        assert_eq!(order[0], 1, "highest importance first");
        assert_eq!(order[3], 3, "lowest importance last");
    }

    #[test]
    fn selection_is_nested_across_counts() {
        let seeds = [7, 11, 13, 17, 19, 23, 29, 31];
        let importances = [0.3, 0.9, 0.1, 0.6, 0.4, 0.8, 0.2, 0.5];
        let order = build_decimation_order(&seeds, &importances, 0.2);
        let mut smaller = Vec::new();
        let mut larger = Vec::new();
        for (k1, k2) in [(1usize, 3usize), (3, 5), (2, 7), (5, 8)] {
            select_nested(&order, k1, &mut smaller);
            select_nested(&order, k2, &mut larger);
            assert_eq!(smaller.len(), k1);
            assert_eq!(larger.len(), k2);
            // Every kept strand at the smaller count survives at the larger.
            for idx in &smaller {
                assert!(larger.contains(idx), "decimation popped a strand");
            }
            // Output is ascending original-index order.
            let mut sorted = larger.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, larger);
        }
    }

    #[test]
    fn order_is_deterministic() {
        let seeds = [3, 1, 4, 1, 5, 9, 2, 6];
        let importances = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let a = build_decimation_order(&seeds, &importances, 0.3);
        let b = build_decimation_order(&seeds, &importances, 0.3);
        assert_eq!(a, b);
    }

    #[test]
    fn order_rejects_length_mismatch() {
        assert!(build_decimation_order(&[1, 2, 3], &[0.5, 0.5], 0.1).is_empty());
    }

    #[test]
    fn decimate_bindings_respects_count_and_nesting() {
        let bindings: Vec<RenderStrandBinding> = (0..10u32)
            .map(|i| binding(i.wrapping_mul(2_654_435_761)))
            .collect();
        let importances: Vec<f32> = (0..10).map(|i| i as f32 / 9.0).collect();

        // Proxy tier keeps nothing.
        let mut out = Vec::new();
        let proxy = strand_decision(HairLodTier::Cards, 4);
        decimate_bindings_importance(&bindings, &importances, 0.2, &proxy, &mut out);
        assert!(out.is_empty(), "card tier keeps no strands");

        // Strand tier keeps exactly the resolved count, in original order.
        let d3 = strand_decision(HairLodTier::Strands, 3);
        let mut out3 = Vec::new();
        decimate_bindings_importance(&bindings, &importances, 0.2, &d3, &mut out3);
        assert_eq!(out3.len(), 3);
        let seeds3: Vec<u32> = out3.iter().map(|b| b.seed).collect();

        let d6 = strand_decision(HairLodTier::Strands, 6);
        let mut out6 = Vec::new();
        decimate_bindings_importance(&bindings, &importances, 0.2, &d6, &mut out6);
        assert_eq!(out6.len(), 6);
        let seeds6: Vec<u32> = out6.iter().map(|b| b.seed).collect();
        // Nested: the 3 kept strands survive into the 6-strand selection.
        for s in &seeds3 {
            assert!(seeds6.contains(s), "count change popped a strand");
        }

        // Count at or above input keeps everything in original order.
        let full = strand_decision(HairLodTier::Strands, 32);
        let mut out_full = Vec::new();
        decimate_bindings_importance(&bindings, &importances, 0.2, &full, &mut out_full);
        assert_eq!(out_full.len(), bindings.len());
        for (i, b) in out_full.iter().enumerate() {
            assert_eq!(b.seed, bindings[i].seed);
        }
    }

    #[test]
    fn decimate_degrades_gracefully_on_missing_importance() {
        let bindings: Vec<RenderStrandBinding> = (0..5u32).map(binding).collect();
        // Mismatched importance length => uniform ranking, still deterministic.
        let d = strand_decision(HairLodTier::Strands, 2);
        let mut a = Vec::new();
        let mut b = Vec::new();
        decimate_bindings_importance(&bindings, &[0.5], 0.2, &d, &mut a);
        decimate_bindings_importance(&bindings, &[0.5], 0.2, &d, &mut b);
        assert_eq!(a.len(), 2);
        let sa: Vec<u32> = a.iter().map(|x| x.seed).collect();
        let sb: Vec<u32> = b.iter().map(|x| x.seed).collect();
        assert_eq!(sa, sb, "uniform fallback must stay deterministic");
    }

    #[test]
    fn empty_inputs_are_safe() {
        let empty: [u32; 0] = [];
        assert!(build_decimation_order(&empty, &[], 0.2).is_empty());
        let order = build_decimation_order(&[1, 2, 3], &[0.1, 0.2, 0.3], 0.1);
        let mut out = Vec::new();
        select_nested(&order, 0, &mut out);
        assert!(out.is_empty());
        let bindings: Vec<RenderStrandBinding> = Vec::new();
        let d = strand_decision(HairLodTier::Strands, 4);
        let mut bout = Vec::new();
        decimate_bindings_importance(&bindings, &[], 0.2, &d, &mut bout);
        assert!(bout.is_empty());
    }

    #[test]
    fn nested_geometry_free_decimation_is_pop_free_and_deterministic() {
        let bindings: Vec<RenderStrandBinding> = (0..40u32).map(binding).collect();

        // Keep 7 then 13: with a stride sample these survivor sets would not
        // nest (40/7 = 5, 40/13 = 3), so the groom would pop. The nested path
        // must keep the 7 as a strict subset of the 13.
        let mut low = Vec::new();
        decimate_bindings_nested(
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 7),
            &mut low,
        );
        let mut high = Vec::new();
        decimate_bindings_nested(
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 13),
            &mut high,
        );
        assert_eq!(low.len(), 7);
        assert_eq!(high.len(), 13);
        let high_seeds: Vec<u32> = high.iter().map(|b| b.seed).collect();
        for kept in &low {
            assert!(high_seeds.contains(&kept.seed), "kept set must nest");
        }
        // Emitted in ascending original order for coherent downstream work.
        for w in high.windows(2) {
            assert!(w[0].seed < w[1].seed);
        }
        // Determinism across passes.
        let mut again = Vec::new();
        decimate_bindings_nested(
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 7),
            &mut again,
        );
        let a: Vec<u32> = low.iter().map(|b| b.seed).collect();
        let b: Vec<u32> = again.iter().map(|b| b.seed).collect();
        assert_eq!(a, b);
        // Proxy tier empties; count at/above length keeps all in order.
        let mut proxy = Vec::new();
        decimate_bindings_nested(
            &bindings,
            &strand_decision(HairLodTier::Cards, 0),
            &mut proxy,
        );
        assert!(proxy.is_empty());
        let mut all = Vec::new();
        decimate_bindings_nested(
            &bindings,
            &strand_decision(HairLodTier::Strands, 999),
            &mut all,
        );
        assert_eq!(all.len(), 40);
        assert_eq!(all[0].seed, 0);
        assert_eq!(all[39].seed, 39);
    }
}
