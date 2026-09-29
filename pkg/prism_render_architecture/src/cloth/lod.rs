//! Screen-coverage cloth LOD selection and sim-dynamics binding.
//!
//! A garment cannot simulate every sim-mesh vertex at every distance: a hero
//! character filling the screen needs the full solve, while the same character
//! across the map should collapse to a reduced solve or a plain skinned shell.
//! This module maps a cloth piece's screen coverage to a [`ClothLodTier`],
//! resolves how many sim vertices and constraints that tier keeps, and — only
//! for simulated tiers — emits the [`DeformationRequest`] that charges cloth
//! dynamics against the shared deformation budget.
//!
//! Coverage is supplied by the caller (projected screen fraction in `0..=1`),
//! so this layer stays a pure, deterministic classification with no projection
//! or transcendental math. Tier thresholds are ordered coarsest-last; sim
//! counts decimate by fixed integer factors so results are exactly reproducible
//! frame to frame. This mirrors `hair/lod.rs`: cloth and hair are symmetric
//! subsystems that share only the deformation-budget arbitration layer.

use alloc::vec::Vec;

use super::{ClothLodTier, ClothPiece, ClothPieceHandle};
use crate::deformation::schedule::DeformationRequest;
use crate::deformation::DeformationKind;

/// Coverage boundaries at which a garment drops to the next coarser tier.
///
/// Coverage is a screen fraction in `0..=1`. The invariant
/// `reduced_sim_below >= skinned_below` is expected; if it is violated the
/// classification still terminates deterministically by testing boundaries in
/// order.
#[derive(Clone, Copy, Debug)]
pub struct ClothLodThresholds {
    /// Below this coverage, drop from full simulation to reduced simulation.
    pub reduced_sim_below: f32,
    /// Below this coverage, drop from reduced simulation to a skinned proxy.
    pub skinned_below: f32,
}

/// The resolved LOD for one cloth piece this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClothLodDecision {
    /// Garment this decision applies to.
    pub handle: ClothPieceHandle,
    /// Selected tier.
    pub tier: ClothLodTier,
    /// Sim-mesh vertices solved at this tier (`0` for the skinned proxy).
    pub sim_vertices: u32,
    /// Constraints solved at this tier (`0` for the skinned proxy).
    pub constraints: u32,
}

/// Classifies a screen coverage into a cloth LOD tier.
#[must_use]
pub fn select_cloth_lod_tier(coverage: f32, thresholds: ClothLodThresholds) -> ClothLodTier {
    if coverage >= thresholds.reduced_sim_below {
        ClothLodTier::FullSim
    } else if coverage >= thresholds.skinned_below {
        ClothLodTier::ReducedSim
    } else {
        ClothLodTier::SkinnedProxy
    }
}

/// Resolves the sim-vertex and constraint budget for a piece at a coverage.
///
/// Full simulation keeps the authored counts; reduced simulation decimates to a
/// quarter of the sim vertices and a quarter of the constraints (never below
/// one, so a simulated tier always has something to solve); the skinned proxy
/// keeps no sim geometry.
///
/// The coverage-selected tier is clamped to be no finer than the piece's
/// [`ClothPiece::native_form`], so a garment authored to only ever skin (a
/// background NPC outfit) is never promoted to a simulation it does not own, no
/// matter how much screen it covers.
#[must_use]
pub fn resolve_cloth_lod(
    piece: ClothPiece,
    coverage: f32,
    thresholds: ClothLodThresholds,
) -> ClothLodDecision {
    let tier = select_cloth_lod_tier(coverage, thresholds).coarser_of(piece.native_form);
    let (sim_vertices, constraints) = match tier {
        ClothLodTier::FullSim => (piece.sim_vertex_count, piece.constraint_count),
        ClothLodTier::ReducedSim => (
            (piece.sim_vertex_count / 4).max(1),
            (piece.constraint_count / 4).max(1),
        ),
        ClothLodTier::SkinnedProxy => (0, 0),
    };
    ClothLodDecision {
        handle: piece.handle,
        tier,
        sim_vertices,
        constraints,
    }
}

/// Builds the cloth-dynamics deformation request for a resolved LOD.
///
/// Only simulated tiers solve; the skinned proxy returns [`None`]. The charged
/// vertex count is the sim vertices this tier drives, which is the dominant
/// per-frame deformation cost. Deformed cloth needs its ray-tracing
/// acceleration structure refit afterwards, so `needs_blas_refit` is set; the
/// budget layer may still defer the refit while running the deformation.
#[must_use]
pub fn cloth_deformation_request(
    piece: ClothPiece,
    decision: ClothLodDecision,
    priority: u32,
) -> Option<DeformationRequest> {
    if !decision.tier.is_simulated() {
        return None;
    }
    Some(DeformationRequest {
        handle: piece.deformation,
        kind: DeformationKind::Cloth,
        vertex_count: decision.sim_vertices,
        priority,
        needs_blas_refit: true,
    })
}

/// Cloth pieces partitioned by the LOD tier selected for them this frame.
///
/// The renderer consumes one bucket at a time: full-sim and reduced-sim pieces
/// feed the cloth solver and its deformation dispatch, while skinned proxies
/// fall into the ordinary skinned-mesh path. Preserving input order within each
/// bucket keeps draw and dispatch submission deterministic, matching the
/// golden binning pattern in `virtual_geometry/bins.rs`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClothLodPlan {
    /// Pieces simulated at full resolution.
    pub full_sim: Vec<ClothLodDecision>,
    /// Pieces simulated at reduced resolution.
    pub reduced_sim: Vec<ClothLodDecision>,
    /// Pieces rendered as a static skinned proxy.
    pub skinned_proxy: Vec<ClothLodDecision>,
}

/// Tier iteration order for building indirect passes.
pub const CLOTH_LOD_ORDER: [ClothLodTier; 3] = [
    ClothLodTier::FullSim,
    ClothLodTier::ReducedSim,
    ClothLodTier::SkinnedProxy,
];

impl ClothLodPlan {
    /// Total pieces across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.full_sim.len() + self.reduced_sim.len() + self.skinned_proxy.len()
    }

    /// Returns `true` when no piece landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.full_sim.is_empty() && self.reduced_sim.is_empty() && self.skinned_proxy.is_empty()
    }

    /// The bucket backing a given tier.
    #[must_use]
    pub fn bucket(&self, tier: ClothLodTier) -> &[ClothLodDecision] {
        match tier {
            ClothLodTier::FullSim => &self.full_sim,
            ClothLodTier::ReducedSim => &self.reduced_sim,
            ClothLodTier::SkinnedProxy => &self.skinned_proxy,
        }
    }

    /// Number of pieces routed to a given tier.
    #[must_use]
    pub fn count_of_tier(&self, tier: ClothLodTier) -> usize {
        self.bucket(tier).len()
    }

    /// Appends a decision to the bucket for its tier.
    pub fn push(&mut self, decision: ClothLodDecision) {
        match decision.tier {
            ClothLodTier::FullSim => self.full_sim.push(decision),
            ClothLodTier::ReducedSim => self.reduced_sim.push(decision),
            ClothLodTier::SkinnedProxy => self.skinned_proxy.push(decision),
        }
    }
}

/// Resolves and bins a set of cloth pieces by their per-piece screen coverage.
///
/// `coverage[i]` is the screen fraction for `pieces[i]`. A piece with no
/// matching coverage entry is skipped rather than panicking, so a stale or
/// short coverage slice cannot crash LOD selection. Input order is preserved
/// within each bucket.
#[must_use]
pub fn bin_cloth_lod(
    pieces: &[ClothPiece],
    coverage: &[f32],
    thresholds: ClothLodThresholds,
) -> ClothLodPlan {
    let mut plan = ClothLodPlan::default();
    for (index, &piece) in pieces.iter().enumerate() {
        let Some(&cov) = coverage.get(index) else {
            continue;
        };
        plan.push(resolve_cloth_lod(piece, cov, thresholds));
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformation::DeformationHandle;

    const THRESHOLDS: ClothLodThresholds = ClothLodThresholds {
        reduced_sim_below: 0.5,
        skinned_below: 0.1,
    };

    fn piece(handle: u32) -> ClothPiece {
        ClothPiece {
            handle: ClothPieceHandle(handle),
            sim_vertex_count: 4096,
            render_vertex_count: 65_536,
            constraint_count: 12_000,
            deformation: DeformationHandle(handle),
            native_form: ClothLodTier::FullSim,
        }
    }

    fn skin_authored_piece(handle: u32) -> ClothPiece {
        ClothPiece {
            native_form: ClothLodTier::SkinnedProxy,
            ..piece(handle)
        }
    }

    #[test]
    fn tier_thresholds_are_ordered() {
        assert_eq!(
            select_cloth_lod_tier(0.9, THRESHOLDS),
            ClothLodTier::FullSim
        );
        assert_eq!(
            select_cloth_lod_tier(0.3, THRESHOLDS),
            ClothLodTier::ReducedSim
        );
        assert_eq!(
            select_cloth_lod_tier(0.01, THRESHOLDS),
            ClothLodTier::SkinnedProxy
        );
    }

    #[test]
    fn boundary_coverage_selects_the_finer_tier() {
        // A coverage exactly on a threshold keeps the finer tier (>=).
        assert_eq!(
            select_cloth_lod_tier(0.5, THRESHOLDS),
            ClothLodTier::FullSim
        );
        assert_eq!(
            select_cloth_lod_tier(0.1, THRESHOLDS),
            ClothLodTier::ReducedSim
        );
    }

    #[test]
    fn full_sim_keeps_authored_counts() {
        let decision = resolve_cloth_lod(piece(0), 0.8, THRESHOLDS);
        assert_eq!(decision.tier, ClothLodTier::FullSim);
        assert_eq!(decision.sim_vertices, 4096);
        assert_eq!(decision.constraints, 12_000);
    }

    #[test]
    fn reduced_sim_decimates_by_fixed_factors() {
        let decision = resolve_cloth_lod(piece(0), 0.3, THRESHOLDS);
        assert_eq!(decision.tier, ClothLodTier::ReducedSim);
        assert_eq!(decision.sim_vertices, 1024);
        assert_eq!(decision.constraints, 3000);
    }

    #[test]
    fn reduced_sim_never_drops_below_one() {
        let tiny = ClothPiece {
            sim_vertex_count: 2,
            constraint_count: 1,
            ..piece(0)
        };
        let decision = resolve_cloth_lod(tiny, 0.3, THRESHOLDS);
        assert_eq!(decision.sim_vertices, 1);
        assert_eq!(decision.constraints, 1);
    }

    #[test]
    fn skinned_proxy_drops_sim_geometry() {
        let decision = resolve_cloth_lod(piece(0), 0.01, THRESHOLDS);
        assert_eq!(decision.tier, ClothLodTier::SkinnedProxy);
        assert_eq!(decision.sim_vertices, 0);
        assert_eq!(decision.constraints, 0);
    }

    #[test]
    fn simulated_tiers_emit_deformation_request() {
        let decision = resolve_cloth_lod(piece(7), 0.8, THRESHOLDS);
        let request =
            cloth_deformation_request(piece(7), decision, 3).expect("simulated tier sims");
        assert_eq!(request.handle, DeformationHandle(7));
        assert_eq!(request.kind, DeformationKind::Cloth);
        assert_eq!(request.vertex_count, 4096);
        assert_eq!(request.priority, 3);
        assert!(request.needs_blas_refit);
    }

    #[test]
    fn reduced_tier_charges_decimated_vertex_count() {
        let decision = resolve_cloth_lod(piece(1), 0.3, THRESHOLDS);
        let request = cloth_deformation_request(piece(1), decision, 2).expect("reduced sims");
        assert_eq!(request.vertex_count, 1024);
    }

    #[test]
    fn skinned_proxy_emits_no_deformation_request() {
        let decision = resolve_cloth_lod(piece(0), 0.01, THRESHOLDS);
        assert!(cloth_deformation_request(piece(0), decision, 1).is_none());
    }

    #[test]
    fn binning_routes_and_preserves_order() {
        let pieces = [piece(2), piece(0), piece(1)];
        let coverage = [0.9, 0.3, 0.9];
        let plan = bin_cloth_lod(&pieces, &coverage, THRESHOLDS);
        assert_eq!(plan.total(), 3);
        assert_eq!(plan.count_of_tier(ClothLodTier::FullSim), 2);
        assert_eq!(plan.full_sim[0].handle, ClothPieceHandle(2));
        assert_eq!(plan.full_sim[1].handle, ClothPieceHandle(1));
        assert_eq!(plan.reduced_sim[0].handle, ClothPieceHandle(0));
    }

    #[test]
    fn short_coverage_slice_skips_extra_pieces() {
        let pieces = [piece(0), piece(1)];
        let coverage = [0.9];
        let plan = bin_cloth_lod(&pieces, &coverage, THRESHOLDS);
        assert_eq!(plan.total(), 1);
        assert_eq!(plan.full_sim[0].handle, ClothPieceHandle(0));
    }

    #[test]
    fn empty_input_is_empty_plan() {
        let plan = bin_cloth_lod(&[], &[], THRESHOLDS);
        assert!(plan.is_empty());
        assert_eq!(plan.total(), 0);
    }

    #[test]
    fn skin_authored_piece_is_never_promoted_to_sim() {
        // Full screen coverage would otherwise select full sim, but a garment
        // authored to only skin has no sim geometry and must stay a proxy.
        let decision = resolve_cloth_lod(skin_authored_piece(0), 0.99, THRESHOLDS);
        assert_eq!(decision.tier, ClothLodTier::SkinnedProxy);
        assert_eq!(decision.sim_vertices, 0);
        assert!(cloth_deformation_request(skin_authored_piece(0), decision, 1).is_none());
    }

    #[test]
    fn native_form_clamp_limits_finest_tier_to_reduced() {
        // A piece authored as reduced-sim never reaches full sim, even at full
        // coverage, but still coarsens to a proxy at distance.
        let reduced_authored = ClothPiece {
            native_form: ClothLodTier::ReducedSim,
            ..piece(0)
        };
        let close = resolve_cloth_lod(reduced_authored, 0.99, THRESHOLDS);
        assert_eq!(close.tier, ClothLodTier::ReducedSim);
        let far = resolve_cloth_lod(reduced_authored, 0.01, THRESHOLDS);
        assert_eq!(far.tier, ClothLodTier::SkinnedProxy);
    }

    #[test]
    fn lod_order_lists_finest_to_coarsest() {
        assert_eq!(
            CLOTH_LOD_ORDER,
            [
                ClothLodTier::FullSim,
                ClothLodTier::ReducedSim,
                ClothLodTier::SkinnedProxy
            ]
        );
    }
}
