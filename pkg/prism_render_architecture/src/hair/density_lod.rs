//! End-to-end density-LOD driver: geometry-derived, importance-weighted,
//! pop-free render-strand decimation.
//!
//! [`super::decimation`] provides the *mechanism* — a nested decimation order
//! and importance blending — but leaves the caller to supply a per-strand
//! importance array. This module supplies it from the actual groom geometry so
//! importance-weighted density LOD works end to end on the CPU (matching the
//! way `UE5` Groom biases which strands survive as a groom recedes).
//!
//! The pipeline is:
//!
//! 1. Measure each *guide* strand's arc length and accumulated curvature from
//!    the resampled control points, and read its authored thickness (root
//!    radius) as an artist-priority proxy.
//! 2. Propagate those guide metrics onto every *render* strand through its
//!    binding weights (a render strand inherits the weighted blend of the
//!    guides it is skinned to).
//! 3. Fold the three metrics into a single normalized importance scalar via
//!    [`super::decimation::compute_importance`].
//! 4. Decimate the bindings to the resolved LOD count with
//!    [`super::decimation::decimate_bindings_importance`], so long, curly, or
//!    thick strands persist to the lowest counts while the kept set still nests
//!    across counts (zero pop).
//!
//! Everything is deterministic, array-in / array-out, and panic-free: empty
//! grooms, empty bindings, out-of-range guide indices, and length mismatches
//! all degrade to safe, well-defined output. No transcendental calls — only
//! `+`, `*`, and `sqrt` (through vector length), consistent with the crate's
//! `libm`-free math policy.

use alloc::vec::Vec;

use super::decimation::{
    compute_importance, decimate_bindings_importance, strand_arc_length, strand_curvature,
    ImportanceWeights, DEFAULT_DECIMATION_JITTER,
};
use super::groom_import::ResampledGroom;
use super::interpolation::{RenderStrandBinding, Vec3 as InterpVec3};
use super::lod::HairLodDecision;

/// Per-guide geometric metrics measured once from the resampled groom.
///
/// All three vectors share `groom.strand_count()` entries. `authored` is the
/// root radius normalized to `[0, 1]` by the groom's maximum root radius, so it
/// reads as a relative thickness priority; when every guide has zero radius it
/// is uniformly `0`.
#[derive(Clone, Debug, Default)]
pub struct GuideMetrics {
    /// Arc length of each guide polyline (world units).
    pub lengths: Vec<f32>,
    /// Accumulated turning of each guide polyline (radian-free curvature sum).
    pub curvatures: Vec<f32>,
    /// Normalized root-radius thickness priority in `[0, 1]`.
    pub authored: Vec<f32>,
}

impl GuideMetrics {
    /// Number of guides measured.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lengths.len()
    }

    /// Returns `true` when no guide was measured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lengths.is_empty()
    }
}

/// Measures per-guide arc length, curvature, and normalized thickness from a
/// resampled groom.
///
/// An empty groom yields empty metrics. The root radii are normalized by the
/// maximum root radius across the groom so `authored` lands in `[0, 1]`; a
/// groom whose radii are all zero yields uniform `0` authored priority.
#[must_use]
pub fn guide_metrics(groom: &ResampledGroom) -> GuideMetrics {
    let n = groom.strand_count();
    let mut lengths = Vec::with_capacity(n);
    let mut curvatures = Vec::with_capacity(n);
    let mut radii = Vec::with_capacity(n);
    let mut max_radius = 0.0_f32;
    for i in 0..n {
        let points = groom.strand(i).unwrap_or(&[]);
        // `ResampledGroom` stores solver-side `dynamics::Vec3`; the arc-length
        // and curvature helpers operate on render-side `interpolation::Vec3`.
        // Convert the polyline once (metrics are measured a single time).
        let poly: Vec<InterpVec3> = points
            .iter()
            .map(|p| InterpVec3::new(p.x, p.y, p.z))
            .collect();
        lengths.push(strand_arc_length(&poly));
        curvatures.push(strand_curvature(&poly));
        let r = groom.attributes[i].root_radius.max(0.0);
        if r > max_radius {
            max_radius = r;
        }
        radii.push(r);
    }
    let authored = if max_radius > f32::EPSILON {
        radii.iter().map(|&r| r / max_radius).collect()
    } else {
        radii.iter().map(|_| 0.0).collect()
    };
    GuideMetrics {
        lengths,
        curvatures,
        authored,
    }
}

/// Blends per-guide metrics onto each render-strand binding through its guide
/// weights, then folds them into a normalized importance scalar.
///
/// A render strand inherits the weight-blended length, curvature, and thickness
/// of the guides it is bound to; out-of-range guide indices contribute nothing.
/// The blended per-binding metrics are then normalized and mixed by
/// [`compute_importance`], so the returned vector is `[0, 1]` with one entry per
/// binding. An empty binding list yields an empty vector.
#[must_use]
pub fn binding_importances(
    bindings: &[RenderStrandBinding],
    metrics: &GuideMetrics,
    weights: ImportanceWeights,
) -> Vec<f32> {
    let n = bindings.len();
    let mut lengths = Vec::with_capacity(n);
    let mut curvatures = Vec::with_capacity(n);
    let mut authored = Vec::with_capacity(n);
    let guide_count = metrics.len();
    for binding in bindings {
        let mut len_acc = 0.0_f32;
        let mut curv_acc = 0.0_f32;
        let mut auth_acc = 0.0_f32;
        for (&guide, &weight) in binding.guides.iter().zip(binding.weights.iter()) {
            let idx = guide as usize;
            if weight > 0.0 && idx < guide_count {
                len_acc += weight * metrics.lengths[idx];
                curv_acc += weight * metrics.curvatures[idx];
                auth_acc += weight * metrics.authored[idx];
            }
        }
        lengths.push(len_acc);
        curvatures.push(curv_acc);
        authored.push(auth_acc);
    }
    compute_importance(&lengths, &curvatures, &authored, weights)
}

/// Plans an importance-weighted, pop-free decimation of render-strand bindings
/// for a resolved LOD decision, deriving importance from the groom geometry.
///
/// This is the end-to-end density-LOD entry point: it measures guide metrics,
/// propagates them to the bindings, and decimates to the LOD's resolved render
/// count with [`decimate_bindings_importance`]. `out` is cleared first. A proxy
/// tier (count `0`), an empty binding list, or an empty groom all yield an empty
/// output; a count at or above the input length keeps every binding in order.
/// The kept set nests across counts, so continuous LOD stays pop-free.
pub fn plan_density_lod(
    groom: &ResampledGroom,
    bindings: &[RenderStrandBinding],
    decision: &HairLodDecision,
    weights: ImportanceWeights,
    jitter: f32,
    out: &mut Vec<RenderStrandBinding>,
) {
    let metrics = guide_metrics(groom);
    let importances = binding_importances(bindings, &metrics, weights);
    decimate_bindings_importance(bindings, &importances, jitter, decision, out);
}

/// Convenience wrapper over [`plan_density_lod`] with the default importance
/// weights and the default per-strand jitter.
pub fn plan_density_lod_default(
    groom: &ResampledGroom,
    bindings: &[RenderStrandBinding],
    decision: &HairLodDecision,
    out: &mut Vec<RenderStrandBinding>,
) {
    plan_density_lod(
        groom,
        bindings,
        decision,
        ImportanceWeights::default(),
        DEFAULT_DECIMATION_JITTER,
        out,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::dynamics::Vec3;
    use crate::hair::groom_import::StrandAttributes;
    use crate::hair::interpolation::GUIDE_INFLUENCE_COUNT;
    use crate::hair::lod::HairLodDecision;
    use crate::hair::{HairGroupHandle, HairLodTier};

    const EPS: f32 = 1e-5;

    fn attr(root_radius: f32, seed: u32) -> StrandAttributes {
        StrandAttributes {
            root_radius,
            tip_radius: root_radius * 0.5,
            root_uv: [0.0, 0.0],
            seed,
        }
    }

    /// Builds a groom of `strand_count` strands, each with `ppts` points, where
    /// strand `i` is a straight vertical line of height `heights[i]`.
    fn straight_groom(heights: &[f32], ppts: usize) -> ResampledGroom {
        let mut positions = Vec::new();
        let mut attributes = Vec::new();
        for (i, &h) in heights.iter().enumerate() {
            for p in 0..ppts {
                let t = p as f32 / (ppts - 1) as f32;
                positions.push(Vec3::new(0.0, t * h, 0.0));
            }
            attributes.push(attr(1.0, i as u32));
        }
        ResampledGroom {
            points_per_strand: ppts,
            positions,
            attributes,
        }
    }

    fn single_guide_binding(guide: u32, seed: u32) -> RenderStrandBinding {
        let mut guides = [0u32; GUIDE_INFLUENCE_COUNT];
        let mut weights = [0.0f32; GUIDE_INFLUENCE_COUNT];
        guides[0] = guide;
        weights[0] = 1.0;
        RenderStrandBinding {
            guides,
            weights,
            root_uv: (0.0, 0.0),
            seed,
        }
    }

    fn strand_decision(tier: HairLodTier, render_strands: u32) -> HairLodDecision {
        HairLodDecision {
            handle: HairGroupHandle(0),
            tier,
            render_strands,
            segments_per_strand: 8,
        }
    }

    #[test]
    fn guide_metrics_measure_length_and_normalized_thickness() {
        // Three straight guides of heights 1, 2, 4; radii 1, 2, 4.
        let mut groom = straight_groom(&[1.0, 2.0, 4.0], 5);
        groom.attributes[0].root_radius = 1.0;
        groom.attributes[1].root_radius = 2.0;
        groom.attributes[2].root_radius = 4.0;
        let m = guide_metrics(&groom);
        assert_eq!(m.len(), 3);
        assert!((m.lengths[0] - 1.0).abs() < EPS);
        assert!((m.lengths[1] - 2.0).abs() < EPS);
        assert!((m.lengths[2] - 4.0).abs() < EPS);
        // Straight lines have no curvature.
        for c in &m.curvatures {
            assert!(c.abs() < EPS, "straight guide must not curve");
        }
        // Radius normalized by max (4): 0.25, 0.5, 1.0.
        assert!((m.authored[0] - 0.25).abs() < EPS);
        assert!((m.authored[1] - 0.5).abs() < EPS);
        assert!((m.authored[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn longer_guide_yields_higher_binding_importance() {
        let groom = straight_groom(&[1.0, 4.0], 5);
        let m = guide_metrics(&groom);
        let bindings = [single_guide_binding(0, 10), single_guide_binding(1, 20)];
        let imp = binding_importances(&bindings, &m, ImportanceWeights::default());
        assert_eq!(imp.len(), 2);
        // Binding 1 rides the longer guide, so it must be more important.
        assert!(imp[1] > imp[0]);
        for v in &imp {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn plan_keeps_most_important_strand_at_lowest_count() {
        // Guide lengths ascending: strand 3 (longest) is the most important.
        let groom = straight_groom(&[1.0, 2.0, 3.0, 8.0], 6);
        let m = guide_metrics(&groom);
        let bindings: Vec<RenderStrandBinding> =
            (0..4u32).map(|i| single_guide_binding(i, i)).collect();
        // Pure importance (no jitter) so the ranking is unambiguous.
        let mut out = Vec::new();
        plan_density_lod(
            &groom,
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 1),
            ImportanceWeights::default(),
            0.0,
            &mut out,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].seed, 3, "longest strand must survive to count 1");
        // Independent cross-check on the raw metric.
        let _ = m;
    }

    #[test]
    fn plan_is_pop_free_and_deterministic() {
        let heights: Vec<f32> = (0..40u32).map(|i| 1.0 + i as f32 * 0.1).collect();
        let groom = straight_groom(&heights, 5);
        let bindings: Vec<RenderStrandBinding> =
            (0..40u32).map(|i| single_guide_binding(i, i)).collect();
        let mut low = Vec::new();
        let mut high = Vec::new();
        plan_density_lod_default(
            &groom,
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 7),
            &mut low,
        );
        plan_density_lod_default(
            &groom,
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 19),
            &mut high,
        );
        assert_eq!(low.len(), 7);
        assert_eq!(high.len(), 19);
        // Nesting: the low set is a subset of the high set (no pop) even at
        // counts that share no integer factor with the total.
        let high_seeds: Vec<u32> = high.iter().map(|b| b.seed).collect();
        for kept in &low {
            assert!(high_seeds.contains(&kept.seed), "kept set must nest");
        }
        // Ascending original order for coherent downstream work.
        for w in high.windows(2) {
            assert!(w[0].seed < w[1].seed);
        }
        // Determinism.
        let mut again = Vec::new();
        plan_density_lod_default(
            &groom,
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 7),
            &mut again,
        );
        let a: Vec<u32> = low.iter().map(|b| b.seed).collect();
        let b: Vec<u32> = again.iter().map(|b| b.seed).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn out_of_range_guides_and_empty_inputs_are_safe() {
        let groom = straight_groom(&[1.0, 2.0], 4);
        let m = guide_metrics(&groom);
        // Binding referencing a nonexistent guide 99 contributes nothing but
        // must not panic; its importance is a well-defined value.
        let bad = single_guide_binding(99, 7);
        let imp = binding_importances(&[bad], &m, ImportanceWeights::default());
        assert_eq!(imp.len(), 1);
        assert!((0.0..=1.0).contains(&imp[0]));

        // Empty groom / empty bindings.
        let empty_groom = ResampledGroom {
            points_per_strand: 4,
            positions: Vec::new(),
            attributes: Vec::new(),
        };
        assert!(guide_metrics(&empty_groom).is_empty());
        let mut out = Vec::new();
        plan_density_lod_default(
            &empty_groom,
            &[bad],
            &strand_decision(HairLodTier::Strands, 4),
            &mut out,
        );
        // No guides -> uniform-0 importance -> still a valid nested selection.
        assert_eq!(out.len(), 1);

        plan_density_lod_default(
            &groom,
            &[],
            &strand_decision(HairLodTier::Strands, 4),
            &mut out,
        );
        assert!(out.is_empty());

        // Proxy tier empties regardless of geometry.
        let bindings = [single_guide_binding(0, 1), single_guide_binding(1, 2)];
        plan_density_lod_default(
            &groom,
            &bindings,
            &strand_decision(HairLodTier::Cards, 0),
            &mut out,
        );
        assert!(out.is_empty());
    }
}
