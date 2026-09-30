//! Screen-coverage cloth LOD gate for the scene-side solve.
//!
//! The architecture layer (`prism_render_architecture::cloth::lod`) owns the
//! deterministic coverage->tier classification, the coarser-of-`native_form`
//! clamp and the sim-vertex/constraint budget decimation. This module is the
//! scene-side bridge: it turns an authored
//! [`ClothGarment`](super::garment::ClothGarment)'s LOD description into an
//! architecture [`ClothPiece`], resolves the tier for the frame's screen
//! coverage through the golden [`resolve_cloth_lod`], and exposes the decision
//! the prepare stage gates on.
//!
//! The device consequences this gate lands are honest and verifiable. A garment
//! whose coverage collapses it to [`ClothLodTier::SkinnedProxy`] builds no
//! resident GPU piece and therefore records no compute pass — the same "no work,
//! no dispatch" contract an empty garment already honors. At
//! [`ClothLodTier::ReducedSim`] a garment that carries a pre-authored coarse
//! simulation mesh ([`super::lod_mesh`]) solves *that* mesh — fewer particles,
//! constraints and dispatched work-items — so the reduced tier delivers a real
//! per-frame cost reduction rather than merely recording a smaller budget; the
//! tier-aware mesh selection lives in
//! [`ClothGarment::as_solve_input_for_tier`](super::garment::ClothGarment::as_solve_input_for_tier)
//! and the prepare stage solves whichever view it returns. A reduced-tier garment
//! with no authored coarse mesh keeps the honest fallback of re-solving the full
//! mesh (never a fabricated decimation). The resolved budget counts are surfaced
//! (see [`ClothLodDecision`]) so the renderer can bin and account for pieces by
//! tier.

use prism_render_architecture::cloth::lod::{
    resolve_cloth_lod, select_cloth_lod_tier_hysteretic, ClothLodDecision, ClothLodThresholds,
};
use prism_render_architecture::cloth::{ClothLodTier, ClothPiece, ClothPieceHandle};
use prism_render_architecture::deformation::DeformationHandle;

use super::garment::ClothGarment;

/// Builds the architecture-layer [`ClothPiece`] describing a garment at its
/// finest authored LOD.
///
/// The simulated counts come straight from the resident garment: `positions`
/// sizes the sim-vertex count, `constraints` the constraint count, and the
/// authored `render_vertex_count` carries through unchanged. Deriving the piece
/// from the live buffers (rather than a duplicated authored count) keeps the
/// LOD budget an exact reflection of the mesh the solver would actually run.
#[must_use]
pub(crate) fn garment_cloth_piece(garment: &ClothGarment) -> ClothPiece {
    ClothPiece {
        handle: ClothPieceHandle(garment.lod_piece_id()),
        sim_vertex_count: garment.sim_vertex_count(),
        render_vertex_count: garment.render_vertex_count(),
        constraint_count: garment.constraint_count(),
        deformation: DeformationHandle(garment.lod_piece_id()),
        native_form: garment.native_form(),
    }
}

/// The LOD thresholds a garment authored, in the architecture layer's form.
#[must_use]
pub(crate) fn garment_thresholds(garment: &ClothGarment) -> ClothLodThresholds {
    ClothLodThresholds {
        reduced_sim_below: garment.lod_reduced_sim_below(),
        skinned_below: garment.lod_skinned_below(),
    }
}

/// Resolves the LOD decision for a garment at its current screen coverage.
///
/// Reuses the golden [`resolve_cloth_lod`] end to end (classification, the
/// coarser-of-`native_form` clamp and the budget decimation), so the scene tier
/// is bit-identical to the architecture layer for the same inputs. The scene
/// never re-derives the thresholds or the decimation factor: this is a thin
/// adapter over the one authoritative source.
#[must_use]
pub(crate) fn resolve_garment_lod(garment: &ClothGarment) -> ClothLodDecision {
    resolve_cloth_lod(
        garment_cloth_piece(garment),
        garment.coverage(),
        garment_thresholds(garment),
    )
}

/// Advances a garment's LOD tier for the frame through the *hysteretic* gate.
///
/// Unlike [`resolve_garment_lod`] (which reclassifies the coverage from scratch
/// every frame and would pop across a threshold), this feeds the garment's
/// authored coverage dead-band and its last-frame [`current_tier`] into the
/// golden [`select_cloth_lod_tier_hysteretic`], then clamps the result no finer
/// than the garment's native form — the same clamp the stateless gate applies.
/// The coverage system calls this each frame and stores the result back on the
/// garment so [`ClothGarment::lod_decision`] charges the held tier.
///
/// With a zero dead-band this is bit-identical to the tier
/// [`resolve_garment_lod`] selects, so a garment that never opts into hysteresis
/// keeps the stateless behavior.
///
/// [`current_tier`]: ClothGarment::current_tier
#[must_use]
pub(crate) fn resolve_garment_tier_hysteretic(garment: &ClothGarment) -> ClothLodTier {
    select_cloth_lod_tier_hysteretic(
        garment.coverage(),
        garment_thresholds(garment),
        garment.lod_hysteresis(),
        garment.current_tier(),
    )
    .coarser_of(garment.native_form())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::{ClothGarment, ClothGarmentBuilder};
    use prism_render_architecture::cloth::{ClothLodTier, ClothParticle, Vec3};

    /// The 4-particle sim mesh shared by the LOD seam tests.
    fn lod_particles() -> [ClothParticle; 4] {
        [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ]
    }

    /// Builds the shared LOD garment at a given screen coverage: it drops to a
    /// reduced sim below 0.5 coverage and to a skinned proxy below 0.1 coverage,
    /// and carries the stable LOD piece id `7`.
    fn lod_garment_at(coverage: f32) -> ClothGarment {
        ClothGarmentBuilder::from_particles(&lod_particles())
            .lod_thresholds(0.5, 0.1)
            .lod_piece_id(7)
            .coverage(coverage)
            .build()
    }

    #[test]
    fn full_coverage_keeps_full_sim_and_authored_budget() {
        let garment = lod_garment_at(1.0);
        let decision = resolve_garment_lod(&garment);
        assert_eq!(decision.tier, ClothLodTier::FullSim);
        assert_eq!(decision.handle, ClothPieceHandle(7));
        assert_eq!(decision.sim_vertices, 4);
        // No authored constraints on the raw-particle path, so the piece has a
        // zero constraint budget; the point is the full-sim path keeps whatever
        // the mesh actually has.
        assert_eq!(decision.constraints, garment.constraint_count());
        assert!(decision.tier.is_simulated());
    }

    #[test]
    fn mid_coverage_selects_reduced_sim_and_still_simulates() {
        let garment = lod_garment_at(0.3);
        let decision = resolve_garment_lod(&garment);
        assert_eq!(decision.tier, ClothLodTier::ReducedSim);
        // Reduced sim decimates the sim budget to a quarter, clamped to at least
        // one: 4 / 4 == 1 here.
        assert_eq!(decision.sim_vertices, 1);
        assert!(decision.tier.is_simulated());
    }

    #[test]
    fn low_coverage_collapses_to_skinned_proxy_and_skips_the_solve() {
        let garment = lod_garment_at(0.05);
        let decision = resolve_garment_lod(&garment);
        assert_eq!(decision.tier, ClothLodTier::SkinnedProxy);
        assert_eq!(decision.sim_vertices, 0);
        assert_eq!(decision.constraints, 0);
        assert!(!decision.tier.is_simulated());
    }

    #[test]
    fn native_skinned_form_is_never_promoted_to_a_simulation() {
        // A background outfit authored to only ever skin must stay skinned even
        // when it fills the screen: the coverage-selected finer tier is clamped
        // to the coarser native form.
        let garment = ClothGarmentBuilder::from_particles(&lod_particles())
            .lod_thresholds(0.5, 0.1)
            .lod_piece_id(7)
            .native_form(ClothLodTier::SkinnedProxy)
            .coverage(1.0)
            .build();
        let decision = resolve_garment_lod(&garment);
        assert_eq!(decision.tier, ClothLodTier::SkinnedProxy);
        assert!(!decision.tier.is_simulated());
    }

    #[test]
    fn default_garment_disables_lod_and_always_full_sims() {
        // The default LOD envelope (both thresholds at zero) can never trigger a
        // reduction because coverage is always >= 0, so an author who never sets
        // LOD keeps the pre-LOD behavior: full simulation regardless of coverage.
        let particles = [ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0)];
        for coverage in [0.0, 0.001, 0.5, 1.0] {
            let garment = ClothGarmentBuilder::from_particles(&particles)
                .coverage(coverage)
                .build();
            let decision = resolve_garment_lod(&garment);
            assert_eq!(decision.tier, ClothLodTier::FullSim);
        }
    }

    #[test]
    fn scene_decision_matches_architecture_golden_bit_for_bit() {
        // The scene adapter must be a pure pass-through to the golden resolver:
        // build the same piece/thresholds by hand and require an identical
        // decision across the full coverage sweep.
        for coverage in [0.0, 0.05, 0.1, 0.3, 0.49, 0.5, 0.9, 1.0] {
            let garment = lod_garment_at(coverage);
            let piece = garment_cloth_piece(&garment);
            let thresholds = garment_thresholds(&garment);
            let scene = resolve_garment_lod(&garment);
            let golden = resolve_cloth_lod(piece, coverage, thresholds);
            assert_eq!(scene, golden, "coverage {coverage} diverged from golden");
        }
    }

    #[test]
    fn hysteretic_tier_with_zero_band_matches_the_stateless_gate() {
        // A garment that never opts into hysteresis must resolve exactly the
        // tier the stateless gate would pick, across the whole coverage sweep.
        for coverage in [0.0, 0.05, 0.1, 0.3, 0.49, 0.5, 0.9, 1.0] {
            let garment = lod_garment_at(coverage);
            assert_eq!(
                resolve_garment_tier_hysteretic(&garment),
                resolve_garment_lod(&garment).tier,
                "coverage {coverage} diverged from the stateless tier",
            );
        }
    }

    #[test]
    fn hysteretic_tier_holds_the_current_tier_inside_the_dead_band() {
        // Thresholds 0.5/0.1 with a 0.1 band: the reduced boundary drops at 0.4
        // and refines at 0.6. A garment currently at full sim sitting at 0.45 is
        // inside the band, so it must hold full sim rather than pop to reduced.
        let garment = ClothGarmentBuilder::from_particles(&lod_particles())
            .lod_thresholds(0.5, 0.1)
            .lod_hysteresis(0.1)
            .coverage(0.45)
            .build();
        // Seeded from the stateless gate: 0.45 < 0.5 classifies as reduced.
        assert_eq!(garment.current_tier(), ClothLodTier::ReducedSim);

        // Now pin the current tier to full sim and re-resolve at 0.45: the band
        // spans (0.4, 0.6) around the reduced boundary, so full sim is held.
        let mut held = garment.clone();
        held.set_current_tier(ClothLodTier::FullSim);
        assert_eq!(
            resolve_garment_tier_hysteretic(&held),
            ClothLodTier::FullSim,
            "0.45 is inside the reduced dead-band and must not pop",
        );

        // Drop coverage below the lower edge (< 0.4): the garment finally
        // coarsens to reduced sim.
        held.set_coverage(0.35);
        assert_eq!(
            resolve_garment_tier_hysteretic(&held),
            ClothLodTier::ReducedSim,
        );
    }

    #[test]
    fn hysteretic_tier_never_refines_past_the_native_form() {
        // A background outfit authored to only ever skin must stay skinned even
        // when it fills the screen, exactly like the stateless clamp.
        let garment = ClothGarmentBuilder::from_particles(&lod_particles())
            .lod_thresholds(0.5, 0.1)
            .lod_hysteresis(0.2)
            .native_form(ClothLodTier::SkinnedProxy)
            .coverage(1.0)
            .build();
        assert_eq!(
            resolve_garment_tier_hysteretic(&garment),
            ClothLodTier::SkinnedProxy,
        );
    }
}
