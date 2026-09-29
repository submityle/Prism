//! Per-frame assembly of a directional light's shadow page requests.
//!
//! This is the shadow counterpart to the virtual-geometry frame planner: it
//! turns the frame's visible shadow casters into the exact set of clipmap pages
//! that must be resident. Each caster arrives in world space with the shadow
//! texel size its on-screen size warrants; this planner projects the shared
//! camera and every caster's bounds onto the light plane via
//! [`DirectionalLightBasis`], marks the clip pages each footprint overlaps with
//! [`mark_receiver_footprint`], and returns a [`ShadowFramePlan`] carrying the
//! coalesced [`ShadowRequestBatch`]. The backend flushes that batch into a
//! [`ShadowResidencyTable`](super::residency::ShadowResidencyTable) to drive
//! streaming and eviction.
//!
//! It composes only the deterministic, GPU-independent decision layers, so the
//! whole world-space-casters-to-page-requests path is unit-testable end to end.

use super::clipmap::ClipmapConfig;
use super::coverage::mark_receiver_footprint;
use super::light_space::DirectionalLightBasis;
use super::residency::ShadowRequestBatch;
use crate::gpu_scene::SceneBounds;

/// One visible shadow caster the planner must cover this frame.
///
/// `bounds` are the caster's world-space bounds; `priority` is its screen
/// importance, forwarded so a page shared by several casters streams in at the
/// highest urgency any of them reported; `required_texel_size` is the
/// world-space shadow texel size its on-screen size warrants and selects the
/// clip level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowCaster {
    /// World-space bounds of the caster.
    pub bounds: SceneBounds,
    /// Screen-importance priority forwarded to the request batch.
    pub priority: f32,
    /// World-space shadow texel size the caster needs; picks the clip level.
    pub required_texel_size: f32,
}

/// The shadow-page requests a directional light needs for one frame.
///
/// `requests` is the coalesced set of pages every visible caster overlaps, each
/// at the highest priority any caster reported; `marked_pages` is the total
/// page overlaps recorded before coalescing (a coarse cost signal), always at
/// least [`request_count`](Self::request_count).
#[derive(Clone, Debug, Default)]
pub struct ShadowFramePlan {
    /// Coalesced clip-page requests for this frame.
    pub requests: ShadowRequestBatch,
    /// Total page overlaps recorded across all casters before coalescing.
    pub marked_pages: usize,
}

impl ShadowFramePlan {
    /// Number of distinct pages requested this frame.
    #[must_use]
    pub fn request_count(&self) -> usize {
        self.requests.len()
    }

    /// Returns `true` when no caster overlapped any in-range clip page.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// Assembles the full per-frame shadow page-request plan for one directional
/// light.
///
/// Projects `camera_world` and each caster's world bounds onto the light plane
/// with `basis`, then marks the clip pages every caster footprint overlaps into
/// one shared [`ShadowRequestBatch`]. Casters whose footprint falls entirely
/// outside the selected level's grid contribute nothing. The returned plan's
/// batch is ready to flush into a residency table.
#[must_use]
pub fn plan_shadow_frame(
    config: &ClipmapConfig,
    basis: &DirectionalLightBasis,
    camera_world: [f32; 3],
    casters: &[ShadowCaster],
) -> ShadowFramePlan {
    let camera = basis.project_point(camera_world);
    let mut requests = ShadowRequestBatch::new();
    let mut marked_pages = 0usize;
    for caster in casters {
        let (min, max) = basis.project_bounds(&caster.bounds);
        marked_pages += mark_receiver_footprint(
            config,
            camera,
            min,
            max,
            caster.required_texel_size,
            caster.priority,
            &mut requests,
        );
    }
    ShadowFramePlan {
        requests,
        marked_pages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClipmapConfig {
        ClipmapConfig {
            light: 5,
            level_count: 4,
            resolution: 8,
            page_texel_dim: 128,
            level0_page_size: 4.0,
        }
    }

    // Top-down light so the shadow plane is world XZ and reasoning is easy.
    fn top_down() -> DirectionalLightBasis {
        DirectionalLightBasis::from_direction([0.0, -1.0, 0.0]).expect("basis")
    }

    fn caster(center: [f32; 3], half: [f32; 3], texel: f32, prio: f32) -> ShadowCaster {
        ShadowCaster {
            bounds: SceneBounds {
                center,
                radius: 0.0,
                half_extents: half,
                _padding: 0.0,
            },
            priority: prio,
            required_texel_size: texel,
        }
    }

    #[test]
    fn empty_casters_yield_empty_plan() {
        let plan = plan_shadow_frame(&config(), &top_down(), [0.0, 0.0, 0.0], &[]);
        assert!(plan.is_empty());
        assert_eq!(plan.request_count(), 0);
        assert_eq!(plan.marked_pages, 0);
    }

    #[test]
    fn single_point_caster_requests_one_page() {
        let c = config();
        let basis = top_down();
        // A tiny caster at the camera resolves to the finest level, one page.
        let casters = [caster([0.0, 3.0, 0.0], [0.0, 0.0, 0.0], 0.001, 1.0)];
        let plan = plan_shadow_frame(&c, &basis, [0.0, 0.0, 0.0], &casters);
        assert_eq!(plan.request_count(), 1);
        assert_eq!(plan.marked_pages, 1);
    }

    #[test]
    fn casters_sharing_a_page_coalesce_to_max_priority() {
        let c = config();
        let basis = top_down();
        // Two point casters at the same spot: one page, highest priority wins.
        let casters = [
            caster([0.0, 3.0, 0.0], [0.0, 0.0, 0.0], 0.001, 1.0),
            caster([0.0, 9.0, 0.0], [0.0, 0.0, 0.0], 0.001, 7.0),
        ];
        let plan = plan_shadow_frame(&c, &basis, [0.0, 0.0, 0.0], &casters);
        assert_eq!(plan.request_count(), 1);
        // Both casters overlapped a page, so two marks coalesced into one.
        assert_eq!(plan.marked_pages, 2);
        let key = c
            .page_of([0.0, 0.0], basis.project_point([0.0, 3.0, 0.0]), 0)
            .expect("in range");
        assert_eq!(plan.requests.priority(key), Some(7.0));
    }

    #[test]
    fn wide_caster_requests_multiple_pages() {
        let c = config();
        let basis = top_down();
        let ps = c.page_size(0);
        // A box two pages wide in X spans at least two columns.
        let casters = [caster([0.0, 3.0, 0.0], [ps, 0.0, 0.0], 0.001, 1.0)];
        let plan = plan_shadow_frame(&c, &basis, [0.0, 0.0, 0.0], &casters);
        assert!(plan.request_count() >= 2);
    }

    #[test]
    fn caster_outside_coverage_contributes_nothing() {
        let c = config();
        let basis = top_down();
        // Far beyond the finest level's coverage.
        let casters = [caster([10_000.0, 3.0, 0.0], [1.0, 0.0, 1.0], 0.001, 1.0)];
        let plan = plan_shadow_frame(&c, &basis, [0.0, 0.0, 0.0], &casters);
        assert!(plan.is_empty());
    }
}
