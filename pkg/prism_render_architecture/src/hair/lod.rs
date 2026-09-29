//! Screen-coverage hair LOD selection and strand-dynamics binding.
//!
//! A groom cannot render every render strand at every distance: a character
//! filling the screen needs full strands, while the same character across the
//! map should collapse to cards or a mesh shell. This module maps a hair
//! group's screen coverage to a [`HairLodTier`], resolves how many render
//! strands and control points that tier keeps, and — only for strand-based
//! tiers — emits the [`DeformationRequest`] that charges strand dynamics
//! against the shared deformation budget.
//!
//! Coverage is supplied by the caller (projected screen fraction in `0..=1`),
//! so this layer stays a pure, deterministic classification with no projection
//! or transcendental math. Tier thresholds are ordered coarsest-last; strand
//! counts decimate by fixed integer factors so results are exactly
//! reproducible frame to frame.

use alloc::vec::Vec;

use super::{HairGroup, HairGroupHandle, HairLodTier};
use crate::deformation::schedule::DeformationRequest;
use crate::deformation::DeformationKind;

/// Coverage boundaries at which a groom drops to the next coarser tier.
///
/// Coverage is a screen fraction in `0..=1`. The invariant
/// `reduced_strands_below >= cards_below >= mesh_below` is expected; if it is
/// violated the classification still terminates deterministically by testing
/// boundaries in order.
#[derive(Clone, Copy, Debug)]
pub struct HairLodThresholds {
    /// Below this coverage, drop from full strands to reduced strands.
    pub reduced_strands_below: f32,
    /// Below this coverage, drop from reduced strands to cards.
    pub cards_below: f32,
    /// Below this coverage, drop from cards to a static mesh shell.
    pub mesh_below: f32,
}

/// The resolved LOD for one hair group this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HairLodDecision {
    /// Groom this decision applies to.
    pub handle: HairGroupHandle,
    /// Selected tier.
    pub tier: HairLodTier,
    /// Render strands kept at this tier (`0` for card/mesh proxies).
    pub render_strands: u32,
    /// Control points per strand at this tier (`0` for card/mesh proxies).
    pub segments_per_strand: u32,
}

/// Classifies a screen coverage into a hair LOD tier.
#[must_use]
pub fn select_hair_lod_tier(coverage: f32, thresholds: HairLodThresholds) -> HairLodTier {
    if coverage >= thresholds.reduced_strands_below {
        HairLodTier::Strands
    } else if coverage >= thresholds.cards_below {
        HairLodTier::ReducedStrands
    } else if coverage >= thresholds.mesh_below {
        HairLodTier::Cards
    } else {
        HairLodTier::Mesh
    }
}

/// Resolves the render-strand and segment budget for a group at a coverage.
///
/// Full strands keep the authored counts; reduced strands decimate to a quarter
/// of the strands with half the control points (never below one); card and mesh
/// proxies keep no per-strand geometry.
///
/// The coverage-selected tier is clamped to be no finer than the group's
/// [`HairGroup::native_form`], so a card-authored (for example NPR/anime) groom
/// is never promoted to strands it does not own, no matter how much screen it
/// covers.
#[must_use]
pub fn resolve_hair_lod(
    group: HairGroup,
    coverage: f32,
    thresholds: HairLodThresholds,
) -> HairLodDecision {
    let tier = select_hair_lod_tier(coverage, thresholds).coarser_of(group.native_form);
    let (render_strands, segments_per_strand) = match tier {
        HairLodTier::Strands => (group.max_render_strands, group.segments_per_strand),
        HairLodTier::ReducedStrands => (
            (group.max_render_strands / 4).max(1),
            (group.segments_per_strand / 2).max(1),
        ),
        HairLodTier::Cards | HairLodTier::Mesh => (0, 0),
    };
    HairLodDecision {
        handle: group.handle,
        tier,
        render_strands,
        segments_per_strand,
    }
}

/// Builds the strand-dynamics deformation request for a resolved LOD.
///
/// Only strand-based tiers simulate; card and mesh proxies return [`None`]. The
/// charged vertex count is the interpolated control points this tier drives
/// (`render_strands * segments_per_strand`), which is the dominant per-frame
/// deformation cost. Ray-traced grooms need their acceleration structure
/// refit after deformation, so `needs_blas_refit` is set.
#[must_use]
pub fn hair_deformation_request(
    group: HairGroup,
    decision: HairLodDecision,
    priority: u32,
) -> Option<DeformationRequest> {
    if !decision.tier.is_strand_based() {
        return None;
    }
    let vertex_count = decision
        .render_strands
        .saturating_mul(decision.segments_per_strand);
    Some(DeformationRequest {
        handle: group.deformation,
        kind: DeformationKind::Hair,
        vertex_count,
        priority,
        needs_blas_refit: true,
    })
}

/// Hair groups partitioned by the LOD tier selected for them this frame.
///
/// The renderer consumes one bucket per pass: strand-based tiers feed the
/// compute strand rasterizer and strand dynamics, cards feed the billboard
/// pass, and mesh proxies fall into the ordinary opaque path. Preserving input
/// order within each bucket keeps draw submission deterministic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HairLodPlan {
    /// Grooms rendered at full strand density.
    pub strands: Vec<HairLodDecision>,
    /// Grooms rendered as decimated strands.
    pub reduced_strands: Vec<HairLodDecision>,
    /// Grooms rendered as camera-facing cards.
    pub cards: Vec<HairLodDecision>,
    /// Grooms rendered as a static mesh shell.
    pub mesh: Vec<HairLodDecision>,
}

/// Tier iteration order for building indirect passes.
pub const HAIR_LOD_ORDER: [HairLodTier; 4] = [
    HairLodTier::Strands,
    HairLodTier::ReducedStrands,
    HairLodTier::Cards,
    HairLodTier::Mesh,
];

impl HairLodPlan {
    /// Total grooms across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.strands.len() + self.reduced_strands.len() + self.cards.len() + self.mesh.len()
    }

    /// Returns `true` when no groom landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strands.is_empty()
            && self.reduced_strands.is_empty()
            && self.cards.is_empty()
            && self.mesh.is_empty()
    }

    /// The bucket backing a given tier.
    #[must_use]
    pub fn bucket(&self, tier: HairLodTier) -> &[HairLodDecision] {
        match tier {
            HairLodTier::Strands => &self.strands,
            HairLodTier::ReducedStrands => &self.reduced_strands,
            HairLodTier::Cards => &self.cards,
            HairLodTier::Mesh => &self.mesh,
        }
    }

    /// Number of grooms routed to a given tier.
    #[must_use]
    pub fn count_of_tier(&self, tier: HairLodTier) -> usize {
        self.bucket(tier).len()
    }

    /// Appends a decision to the bucket for its tier.
    pub fn push(&mut self, decision: HairLodDecision) {
        match decision.tier {
            HairLodTier::Strands => self.strands.push(decision),
            HairLodTier::ReducedStrands => self.reduced_strands.push(decision),
            HairLodTier::Cards => self.cards.push(decision),
            HairLodTier::Mesh => self.mesh.push(decision),
        }
    }
}

/// Resolves and bins a set of grooms by their per-group screen coverage.
///
/// `coverage[i]` is the screen fraction for `groups[i]`. A group with no
/// matching coverage entry is skipped rather than panicking, so a stale or
/// short coverage slice cannot crash LOD selection. Input order is preserved
/// within each bucket.
#[must_use]
pub fn bin_hair_lod(
    groups: &[HairGroup],
    coverage: &[f32],
    thresholds: HairLodThresholds,
) -> HairLodPlan {
    let mut plan = HairLodPlan::default();
    for (index, &group) in groups.iter().enumerate() {
        let Some(&cov) = coverage.get(index) else {
            continue;
        };
        plan.push(resolve_hair_lod(group, cov, thresholds));
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformation::DeformationHandle;

    const THRESHOLDS: HairLodThresholds = HairLodThresholds {
        reduced_strands_below: 0.5,
        cards_below: 0.2,
        mesh_below: 0.05,
    };

    fn group(handle: u32) -> HairGroup {
        HairGroup {
            handle: HairGroupHandle(handle),
            guide_strand_count: 512,
            max_render_strands: 40_000,
            segments_per_strand: 8,
            deformation: DeformationHandle(handle),
            native_form: HairLodTier::Strands,
        }
    }

    fn card_authored_group(handle: u32) -> HairGroup {
        HairGroup {
            native_form: HairLodTier::Cards,
            ..group(handle)
        }
    }

    #[test]
    fn tier_thresholds_are_ordered() {
        assert_eq!(select_hair_lod_tier(0.9, THRESHOLDS), HairLodTier::Strands);
        assert_eq!(
            select_hair_lod_tier(0.3, THRESHOLDS),
            HairLodTier::ReducedStrands
        );
        assert_eq!(select_hair_lod_tier(0.1, THRESHOLDS), HairLodTier::Cards);
        assert_eq!(select_hair_lod_tier(0.01, THRESHOLDS), HairLodTier::Mesh);
    }

    #[test]
    fn full_strands_keep_authored_counts() {
        let decision = resolve_hair_lod(group(0), 0.8, THRESHOLDS);
        assert_eq!(decision.tier, HairLodTier::Strands);
        assert_eq!(decision.render_strands, 40_000);
        assert_eq!(decision.segments_per_strand, 8);
    }

    #[test]
    fn reduced_strands_decimate_by_fixed_factors() {
        let decision = resolve_hair_lod(group(0), 0.3, THRESHOLDS);
        assert_eq!(decision.tier, HairLodTier::ReducedStrands);
        assert_eq!(decision.render_strands, 10_000);
        assert_eq!(decision.segments_per_strand, 4);
    }

    #[test]
    fn proxy_tiers_drop_strand_geometry() {
        let cards = resolve_hair_lod(group(0), 0.1, THRESHOLDS);
        assert_eq!(cards.render_strands, 0);
        assert_eq!(cards.segments_per_strand, 0);
        let mesh = resolve_hair_lod(group(0), 0.0, THRESHOLDS);
        assert_eq!(mesh.tier, HairLodTier::Mesh);
        assert_eq!(mesh.render_strands, 0);
    }

    #[test]
    fn strand_tiers_emit_deformation_request() {
        let decision = resolve_hair_lod(group(7), 0.8, THRESHOLDS);
        let request = hair_deformation_request(group(7), decision, 3).expect("strand tier sims");
        assert_eq!(request.handle, DeformationHandle(7));
        assert_eq!(request.kind, DeformationKind::Hair);
        assert_eq!(request.vertex_count, 40_000 * 8);
        assert_eq!(request.priority, 3);
        assert!(request.needs_blas_refit);
    }

    #[test]
    fn proxy_tiers_emit_no_deformation_request() {
        let decision = resolve_hair_lod(group(0), 0.1, THRESHOLDS);
        assert!(hair_deformation_request(group(0), decision, 1).is_none());
    }

    #[test]
    fn binning_routes_and_preserves_order() {
        let groups = [group(2), group(0), group(1)];
        let coverage = [0.9, 0.3, 0.9];
        let plan = bin_hair_lod(&groups, &coverage, THRESHOLDS);
        assert_eq!(plan.total(), 3);
        assert_eq!(plan.count_of_tier(HairLodTier::Strands), 2);
        assert_eq!(plan.strands[0].handle, HairGroupHandle(2));
        assert_eq!(plan.strands[1].handle, HairGroupHandle(1));
        assert_eq!(plan.reduced_strands[0].handle, HairGroupHandle(0));
    }

    #[test]
    fn short_coverage_slice_skips_extra_groups() {
        let groups = [group(0), group(1)];
        let coverage = [0.9];
        let plan = bin_hair_lod(&groups, &coverage, THRESHOLDS);
        assert_eq!(plan.total(), 1);
        assert_eq!(plan.strands[0].handle, HairGroupHandle(0));
    }

    #[test]
    fn empty_input_is_empty_plan() {
        let plan = bin_hair_lod(&[], &[], THRESHOLDS);
        assert!(plan.is_empty());
        assert_eq!(plan.total(), 0);
    }

    #[test]
    fn card_authored_groom_is_never_promoted_to_strands() {
        // Full screen coverage would otherwise select strands, but a groom
        // authored as cards (NPR/anime look) has no strand geometry and must
        // stay at cards regardless of how close the camera is.
        let decision = resolve_hair_lod(card_authored_group(0), 0.99, THRESHOLDS);
        assert_eq!(decision.tier, HairLodTier::Cards);
        assert_eq!(decision.render_strands, 0);
        assert_eq!(decision.segments_per_strand, 0);
    }

    #[test]
    fn card_authored_groom_still_coarsens_with_distance() {
        // Native form clamps the finest tier, but distance may still drop a
        // card groom to the mesh shell.
        let decision = resolve_hair_lod(card_authored_group(0), 0.0, THRESHOLDS);
        assert_eq!(decision.tier, HairLodTier::Mesh);
    }

    #[test]
    fn card_authored_groom_emits_no_deformation_request() {
        let decision = resolve_hair_lod(card_authored_group(0), 0.99, THRESHOLDS);
        assert!(hair_deformation_request(card_authored_group(0), decision, 1).is_none());
    }

    #[test]
    fn coarser_of_returns_the_higher_rank_tier() {
        assert_eq!(
            HairLodTier::Strands.coarser_of(HairLodTier::Cards),
            HairLodTier::Cards
        );
        assert_eq!(
            HairLodTier::Cards.coarser_of(HairLodTier::Strands),
            HairLodTier::Cards
        );
        assert_eq!(
            HairLodTier::Mesh.coarser_of(HairLodTier::Strands),
            HairLodTier::Mesh
        );
    }
}
