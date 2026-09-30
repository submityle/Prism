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
    cloth_lod_budget(piece, tier)
}

/// Resolves the sim-vertex and constraint budget a cloth piece keeps at an
/// already-selected tier.
///
/// Full simulation keeps the authored counts; reduced simulation decimates to a
/// quarter of the sim vertices and a quarter of the constraints (never below
/// one, so a simulated tier always has something to solve); the skinned proxy
/// keeps no sim geometry. This is the single authoritative decimation rule: both
/// the stateless [`resolve_cloth_lod`] gate and the hysteretic scene-side gate
/// build their decision through it, so a tier chosen by either path charges an
/// identical budget. The tier is taken as already resolved (and, where relevant,
/// already clamped to a piece's native form), so this function performs no
/// classification of its own.
#[must_use]
pub fn cloth_lod_budget(piece: ClothPiece, tier: ClothLodTier) -> ClothLodDecision {
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

/// Classifies a screen coverage into a cloth LOD tier *with hysteresis*, so a
/// garment hovering on a threshold does not oscillate ("pop") between tiers
/// frame to frame.
///
/// `hysteresis` is a symmetric coverage dead-band applied around each authored
/// boundary: starting from the `current` tier, a garment must fall a full
/// `hysteresis` *below* a boundary to drop to the coarser tier, and rise a full
/// `hysteresis` *above* it to climb back to the finer tier. Between those two
/// edges the `current` tier is held, which is exactly the continuous, pop-free
/// LOD transition the cloth design doc calls for (mirroring the bias/hysteresis
/// band UE's Chaos cloth LOD uses).
///
/// A `hysteresis` of `0.0` collapses both edges onto the authored boundary and
/// reproduces [`select_cloth_lod_tier`] exactly for *every* `current` tier, so a
/// garment that never opts into hysteresis is bit-identical to the stateless
/// gate. The band is clamped non-negative, and every boundary is retested
/// against the new coverage, so a garment that jumps several tiers in one frame
/// (a hard camera cut) still resolves directly to the correct distant tier
/// rather than stepping one tier per frame.
#[must_use]
pub fn select_cloth_lod_tier_hysteretic(
    coverage: f32,
    thresholds: ClothLodThresholds,
    hysteresis: f32,
    current: ClothLodTier,
) -> ClothLodTier {
    let band = hysteresis.max(0.0);
    // "down" edges must be crossed (strictly below) to coarsen; "up" edges must
    // be reached (`>=`) to refine. Separating them by `2 * band` is the dead-band
    // that suppresses boundary popping. With `band == 0` both edges coincide with
    // the authored boundary and this reduces to `select_cloth_lod_tier`.
    let reduced_down = thresholds.reduced_sim_below - band;
    let reduced_up = thresholds.reduced_sim_below + band;
    let skinned_down = thresholds.skinned_below - band;
    let skinned_up = thresholds.skinned_below + band;
    match current {
        ClothLodTier::FullSim => {
            if coverage < skinned_down {
                ClothLodTier::SkinnedProxy
            } else if coverage < reduced_down {
                ClothLodTier::ReducedSim
            } else {
                ClothLodTier::FullSim
            }
        }
        ClothLodTier::ReducedSim => {
            if coverage >= reduced_up {
                ClothLodTier::FullSim
            } else if coverage < skinned_down {
                ClothLodTier::SkinnedProxy
            } else {
                ClothLodTier::ReducedSim
            }
        }
        ClothLodTier::SkinnedProxy => {
            if coverage >= reduced_up {
                ClothLodTier::FullSim
            } else if coverage >= skinned_up {
                ClothLodTier::ReducedSim
            } else {
                ClothLodTier::SkinnedProxy
            }
        }
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
    fn hysteresis_zero_matches_stateless_for_every_current_tier() {
        // With no dead-band, the hysteretic gate must be bit-identical to the
        // stateless classifier regardless of which tier the garment currently
        // holds: the default (opt-out) behavior is unchanged.
        for tier in CLOTH_LOD_ORDER {
            for step in 0..=100 {
                let coverage = step as f32 / 100.0;
                assert_eq!(
                    select_cloth_lod_tier_hysteretic(coverage, THRESHOLDS, 0.0, tier),
                    select_cloth_lod_tier(coverage, THRESHOLDS),
                    "coverage {coverage} from {tier:?} diverged from the stateless gate",
                );
            }
        }
    }

    #[test]
    fn hysteresis_holds_the_current_tier_across_the_boundary_band() {
        // A garment sitting just above the reduced boundary does not drop until
        // coverage falls a full band below it, and once reduced it does not climb
        // back until coverage rises a full band above it. The band around 0.5
        // with hysteresis 0.05 is [0.45, 0.55).
        let band = 0.05;
        // Coming from full sim, coverage in [0.45, 0.5) holds full sim even though
        // the stateless gate would already have dropped to reduced.
        assert_eq!(
            select_cloth_lod_tier_hysteretic(0.47, THRESHOLDS, band, ClothLodTier::FullSim),
            ClothLodTier::FullSim,
        );
        assert_eq!(
            select_cloth_lod_tier(0.47, THRESHOLDS),
            ClothLodTier::ReducedSim,
        );
        // Falling below 0.45 finally drops to reduced.
        assert_eq!(
            select_cloth_lod_tier_hysteretic(0.44, THRESHOLDS, band, ClothLodTier::FullSim),
            ClothLodTier::ReducedSim,
        );
        // Coming from reduced, coverage in (0.5, 0.55) holds reduced even though
        // the stateless gate would already have climbed to full sim.
        assert_eq!(
            select_cloth_lod_tier_hysteretic(0.53, THRESHOLDS, band, ClothLodTier::ReducedSim),
            ClothLodTier::ReducedSim,
        );
        assert_eq!(select_cloth_lod_tier(0.53, THRESHOLDS), ClothLodTier::FullSim);
        // Rising above 0.55 finally climbs back to full sim.
        assert_eq!(
            select_cloth_lod_tier_hysteretic(0.56, THRESHOLDS, band, ClothLodTier::ReducedSim),
            ClothLodTier::FullSim,
        );
    }

    #[test]
    fn hysteresis_suppresses_popping_across_a_dithering_sweep() {
        // Coverage dithering by +/- one band step around the reduced boundary must
        // not produce a single tier flip once the dead-band is entered: the whole
        // point of hysteresis. Sweep coverage back and forth inside the band and
        // assert the tier never changes.
        let band = 0.08;
        let mut tier = ClothLodTier::FullSim;
        // Seed just inside the upper edge so we start held at full sim.
        tier = select_cloth_lod_tier_hysteretic(0.5 + band - 1.0e-3, THRESHOLDS, band, tier);
        assert_eq!(tier, ClothLodTier::FullSim);
        // Dither strictly inside (down_edge, up_edge) = (0.42, 0.58): no flips.
        for &coverage in &[0.57, 0.43, 0.55, 0.45, 0.5, 0.44, 0.56] {
            let next = select_cloth_lod_tier_hysteretic(coverage, THRESHOLDS, band, tier);
            assert_eq!(next, ClothLodTier::FullSim, "coverage {coverage} popped the tier");
            tier = next;
        }
    }

    #[test]
    fn hysteresis_still_allows_a_multi_tier_jump_on_a_hard_cut() {
        // A hard camera cut that drops coverage from full-screen to almost nothing
        // must collapse straight to the skinned proxy in one step, not walk down
        // one tier per frame.
        assert_eq!(
            select_cloth_lod_tier_hysteretic(0.001, THRESHOLDS, 0.05, ClothLodTier::FullSim),
            ClothLodTier::SkinnedProxy,
        );
        // ...and the reverse: from a distant proxy straight back to full sim when
        // the garment snaps to fill the screen.
        assert_eq!(
            select_cloth_lod_tier_hysteretic(0.99, THRESHOLDS, 0.05, ClothLodTier::SkinnedProxy),
            ClothLodTier::FullSim,
        );
    }

    #[test]
    fn cloth_lod_budget_matches_resolve_for_the_same_tier() {
        // The extracted budget helper must agree with the stateless resolver
        // whenever they land on the same tier, so routing a hysteretic tier
        // through the budget helper charges an identical cost.
        for (coverage, tier) in [
            (0.9, ClothLodTier::FullSim),
            (0.3, ClothLodTier::ReducedSim),
            (0.01, ClothLodTier::SkinnedProxy),
        ] {
            let resolved = resolve_cloth_lod(piece(3), coverage, THRESHOLDS);
            assert_eq!(resolved.tier, tier);
            assert_eq!(cloth_lod_budget(piece(3), tier), resolved);
        }
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
