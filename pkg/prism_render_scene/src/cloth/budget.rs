//! Scene-side cloth deformation-budget arbitration.
//!
//! Cloth is one of several subsystems (skinning, morph targets, hair, vertex
//! animation) that deform cached geometry on the `GPU` every frame, and they
//! all draw from one shared per-frame budget: how many vertices the deformation
//! cache can process and how many acceleration-structure refits ray tracing can
//! afford. The architecture layer owns the deterministic, priority-greedy
//! arbiter that decides which requested jobs run this frame and which slip to
//! the next ([`plan_deformations`]); this module is the scene-side bridge that
//! turns the resident garments into that arbiter's input and reads its verdict
//! back out as a per-garment "simulate this frame?" mask.
//!
//! The one *device* consequence this slice lands is honest and verifiable: a
//! simulated garment whose sim-vertex charge does not fit the frame's remaining
//! budget builds no resident `GPU` piece this frame and therefore records no
//! compute pass — the exact same "no work, no dispatch" contract the screen
//! coverage LOD gate ([`super::lod`]) already honors for a skinned proxy. The
//! garment is not dropped: its render mesh keeps last frame's embedded pose and
//! the arbiter re-offers it next frame, so deferral is a temporal smoothing of
//! the solver load, never a permanent loss. The single highest-priority garment
//! is always admitted even if it alone blows the budget, so no garment starves.
//!
//! Priority is the live screen coverage the estimator already writes each frame:
//! a garment filling more of the screen is more visually important, so it is
//! admitted first and the distant background garments defer. This reuses the
//! same coverage signal that drives the LOD tier, keeping the whole
//! coverage -> tier -> budget path a single coherent story.
//!
//! The default budget is effectively unlimited, mirroring the LOD thresholds'
//! disabled-by-default policy: the mechanism is real and fully exercised by the
//! unit tests, and an author opts a scene into a finite deformation budget by
//! lowering [`ClothDeformationBudget::budget`]. Until then every simulated
//! garment is admitted and the gate is a transparent pass-through.

use bevy_ecs::resource::Resource;

use prism_render_architecture::cloth::lod::cloth_deformation_request;
use prism_render_architecture::deformation::schedule::{
    plan_deformations, DeformationPlan, DeformationRequest,
};
use prism_render_architecture::deformation::{
    DeformationBudget, DeformationHandle, DeformationKind,
};

use super::garment::ClothGarment;
use super::lod::garment_cloth_piece;

/// Scale mapping a garment's `0..=1` screen coverage onto the arbiter's integer
/// priority. A million steps give sub-`0.0001`-coverage resolution while
/// keeping the largest priority (`1_000_000`) far below `u32::MAX`, so ties are
/// only ever broken by the arbiter's ascending-handle rule for two garments of
/// genuinely equal coverage.
const COVERAGE_PRIORITY_SCALE: f32 = 1_000_000.0;

/// The shared per-frame deformation budget cloth solves are charged against.
///
/// Wraps the architecture-layer [`DeformationBudget`] as a render-world
/// resource so a scene can tune the cloth solver's per-frame ceiling. The
/// default is effectively unlimited (`u32::MAX` on both axes), which keeps the
/// budget gate a transparent pass-through until an author opts in with a finite
/// ceiling — the same disabled-by-default policy the LOD thresholds use.
#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct ClothDeformationBudget {
    /// The per-frame vertex and BLAS-refit ceiling cloth dynamics competes for.
    pub(crate) budget: DeformationBudget,
}

impl Default for ClothDeformationBudget {
    fn default() -> Self {
        Self {
            budget: DeformationBudget {
                vertices_per_frame: u32::MAX,
                blas_refits_per_frame: u32::MAX,
            },
        }
    }
}

/// Maps a garment's `0..=1` screen coverage to the arbiter's integer priority.
///
/// Larger on screen means more visually important, so a higher coverage yields
/// a higher priority and is admitted first. Non-finite coverage (which the
/// garment API already prevents, but is guarded here so the arbiter input is
/// always well-formed) maps to the lowest priority.
#[must_use]
pub(crate) fn coverage_priority(coverage: f32) -> u32 {
    let clamped = if coverage.is_finite() {
        coverage.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (clamped * COVERAGE_PRIORITY_SCALE) as u32
}

/// Builds the deformation requests for every simulated garment in `garments`.
///
/// Each request is stamped with a per-frame handle equal to the garment's slot
/// index, so the arbiter identity is robust even when two garments share an
/// authored LOD piece id. Non-simulated garments (skinned proxies) and
/// zero-vertex garments emit no request: they build no resident piece, so
/// charging them against the budget would be meaningless. The vertex charge and
/// the `needs_blas_refit` flag come straight from the architecture-layer golden
/// [`cloth_deformation_request`], keeping the scene-side charge identical to the
/// value the deformation scheduler is designed around.
#[must_use]
pub(crate) fn collect_cloth_requests(garments: &[ClothGarment]) -> Vec<DeformationRequest> {
    let mut requests = Vec::new();
    for (index, garment) in garments.iter().enumerate() {
        let decision = garment.lod_decision();
        if decision.sim_vertices == 0 {
            continue;
        }
        // Stamp the per-frame handle with the slot index so the admitted mask
        // maps back to the exact garment regardless of the authored piece id.
        let mut piece = garment_cloth_piece(garment);
        piece.deformation = DeformationHandle(index as u32);
        if let Some(request) =
            cloth_deformation_request(piece, decision, coverage_priority(garment.coverage()))
        {
            requests.push(request);
        }
    }
    requests
}

/// Runs the deformation arbiter over `garments` for one frame under `budget`.
#[must_use]
pub(crate) fn plan_cloth_frame(
    garments: &[ClothGarment],
    budget: DeformationBudget,
) -> DeformationPlan {
    plan_deformations(&collect_cloth_requests(garments), budget)
}

/// Resolves, per garment slot, whether the deformation budget admits it this
/// frame.
///
/// The returned mask is indexed by the garment's position in `garments`: a
/// `true` slot is admitted and builds its resident piece, a `false` slot either
/// is a non-simulated proxy (never requested) or was deferred to a later frame
/// because it did not fit the vertex budget. The prepare stage gates each
/// garment on its slot so a deferred garment records no compute pass this frame.
#[must_use]
pub(crate) fn admitted_garment_mask(
    garments: &[ClothGarment],
    budget: DeformationBudget,
) -> Vec<bool> {
    let plan = plan_cloth_frame(garments, budget);
    let mut mask = vec![false; garments.len()];
    for handle in plan.handles_of_kind(DeformationKind::Cloth) {
        if let Some(slot) = mask.get_mut(handle.0 as usize) {
            *slot = true;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::ClothLodTier;

    /// A simulated garment with `vertices` particles at the given coverage. LOD
    /// stays at full simulation (default thresholds are disabled) so the only
    /// gate under test is the deformation budget.
    fn simulated(vertices: usize, coverage: f32) -> ClothGarment {
        ClothGarment {
            positions: vec![[0.0; 4]; vertices],
            coverage,
            ..Default::default()
        }
    }

    /// A garment authored to only ever skin: its native form is the skinned
    /// proxy, so LOD never promotes it to a simulation and it emits no request.
    /// The frame-state tier is seeded to the skinned proxy to match what the
    /// builder's stateless seed and the coverage system's native-form clamp
    /// would resolve for such an outfit.
    fn skinned(vertices: usize) -> ClothGarment {
        ClothGarment {
            positions: vec![[0.0; 4]; vertices],
            coverage: 1.0,
            native_form: ClothLodTier::SkinnedProxy,
            current_tier: ClothLodTier::SkinnedProxy,
            ..Default::default()
        }
    }

    const TIGHT: DeformationBudget = DeformationBudget {
        vertices_per_frame: 1000,
        blas_refits_per_frame: 8,
    };

    #[test]
    fn coverage_priority_is_monotonic_and_clamped() {
        assert!(coverage_priority(1.0) > coverage_priority(0.5));
        assert!(coverage_priority(0.5) > coverage_priority(0.0));
        // Out-of-range coverage clamps rather than overflowing the scale.
        assert_eq!(coverage_priority(2.0), coverage_priority(1.0));
        assert_eq!(coverage_priority(-1.0), coverage_priority(0.0));
        // Non-finite coverage maps to the lowest priority.
        assert_eq!(coverage_priority(f32::NAN), 0);
    }

    #[test]
    fn requests_skip_skinned_and_empty_garments() {
        let garments = [simulated(4, 1.0), skinned(4), simulated(0, 1.0)];
        let requests = collect_cloth_requests(&garments);
        // Only the first garment is a non-empty simulated piece.
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].handle, DeformationHandle(0));
        assert_eq!(requests[0].vertex_count, 4);
        assert_eq!(requests[0].kind, DeformationKind::Cloth);
    }

    #[test]
    fn request_handle_tracks_the_slot_index() {
        // A skinned garment in slot 0 emits nothing, so the simulated garment in
        // slot 1 must still carry handle 1, not 0.
        let garments = [skinned(4), simulated(4, 1.0)];
        let requests = collect_cloth_requests(&garments);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].handle, DeformationHandle(1));
    }

    #[test]
    fn unlimited_budget_admits_every_simulated_garment() {
        let garments = [simulated(500, 0.9), simulated(500, 0.1), skinned(500)];
        let mask = admitted_garment_mask(&garments, ClothDeformationBudget::default().budget);
        // Both simulated garments are admitted; the skinned proxy is never a
        // simulated candidate so its slot stays false.
        assert_eq!(mask, vec![true, true, false]);
    }

    #[test]
    fn tight_budget_defers_lower_coverage_garments() {
        // Two 800-vertex garments cannot both fit a 1000-vertex budget. The
        // higher-coverage garment (slot 0) wins; the lower-coverage one defers.
        let garments = [simulated(800, 0.9), simulated(800, 0.2)];
        let mask = admitted_garment_mask(&garments, TIGHT);
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn higher_coverage_wins_regardless_of_slot_order() {
        // Slot order is reversed (low coverage first) but the arbiter still
        // admits the higher-coverage garment in slot 1 and defers slot 0.
        let garments = [simulated(800, 0.2), simulated(800, 0.9)];
        let mask = admitted_garment_mask(&garments, TIGHT);
        assert_eq!(mask, vec![false, true]);
    }

    #[test]
    fn oversized_top_priority_garment_never_starves() {
        // A single garment larger than the whole budget is still admitted so the
        // solver always makes forward progress.
        let garments = [simulated(5000, 1.0)];
        let mask = admitted_garment_mask(&garments, TIGHT);
        assert_eq!(mask, vec![true]);
    }
}
