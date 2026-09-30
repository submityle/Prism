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
//! The one *device* consequence this slice lands is honest and verifiable: a
//! garment whose coverage collapses it to [`ClothLodTier::SkinnedProxy`] builds
//! no resident GPU piece and therefore records no compute pass — the same
//! "no work, no dispatch" contract an empty garment already honors. Simulated
//! tiers (full and reduced) still solve the authored mesh; swapping a reduced
//! tier onto a pre-authored, decimated LOD mesh is an asset-pipeline follow-up
//! tracked in the cloth design doc, so this slice deliberately does not fake a
//! decimation it cannot yet run. The resolved budget counts are still surfaced
//! (see [`ClothLodDecision`]) so the renderer can bin and account for pieces by
//! tier today.

use prism_render_architecture::cloth::lod::{
    resolve_cloth_lod, ClothLodDecision, ClothLodThresholds,
};
use prism_render_architecture::cloth::{ClothPiece, ClothPieceHandle};
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
}
