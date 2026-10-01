//! Two-phase raster routing of culled work items.
//!
//! A Nanite-style visibility pipeline rasterizes in two passes around an HZB
//! rebuild. The *early* pass draws instances that passed the previous frame's
//! depth hierarchy; their depth then rebuilds the current-frame HZB. The *late*
//! pass draws the instances that were deferred by the early test but revealed
//! by that fresh HZB (see
//! [`resolve_two_phase_occlusion`](crate::resolve_two_phase_occlusion) and
//! [`cull_view_two_phase`](crate::cull_view_two_phase), which stamps the two
//! groups with [`EARLY`](crate::VisibilityStageMask::EARLY) and
//! [`LATE_RETEST`](crate::VisibilityStageMask::LATE_RETEST) |
//! [`LATE_VISIBLE`](crate::VisibilityStageMask::LATE_VISIBLE) respectively).
//!
//! [`plan_two_phase_raster`] turns a flat work list into the two ordered draw
//! lists the rasterizer consumes, assigning each item to exactly one phase so
//! nothing is drawn twice. It is the direct downstream consumer of the stage
//! tags and changes no existing API.

use crate::{GpuRenderWorkItem, VisibilityStageMask};
use alloc::vec::Vec;

/// The raster phase a work item belongs to in a two-phase visibility pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RasterPhase {
    /// Drawn before the current-frame HZB is rebuilt, from instances that
    /// passed the previous-frame HZB.
    Early,
    /// Drawn after the HZB rebuild, from instances deferred by the early test
    /// and revealed by the current-frame HZB.
    Late,
}

/// Classifies a stage mask into the single raster phase that must draw it.
///
/// [`EARLY`](crate::VisibilityStageMask::EARLY) takes precedence: an instance
/// already drawn in the early pass is never redrawn late, even if a later test
/// also marked it [`LATE_VISIBLE`](crate::VisibilityStageMask::LATE_VISIBLE).
/// An item that is neither early nor late-visible (for example a bare
/// [`LATE_RETEST`](crate::VisibilityStageMask::LATE_RETEST) that failed the
/// current HZB) is not rastered and returns `None`; such items are normally
/// already removed by [`cull_view`](crate::cull_view).
pub fn raster_phase_of(stages: VisibilityStageMask) -> Option<RasterPhase> {
    if stages.contains(VisibilityStageMask::EARLY) {
        Some(RasterPhase::Early)
    } else if stages.contains(VisibilityStageMask::LATE_VISIBLE) {
        Some(RasterPhase::Late)
    } else {
        None
    }
}

/// Ordered draw lists for the two raster phases.
///
/// Each list preserves the relative order of the input work, so a prior
/// `sort_by_key(sort_key)` is respected within each phase.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TwoPhaseRasterPlan {
    /// Items drawn before the HZB rebuild.
    pub early: Vec<GpuRenderWorkItem>,
    /// Items drawn after the HZB rebuild.
    pub late: Vec<GpuRenderWorkItem>,
}

impl TwoPhaseRasterPlan {
    /// Total items routed into either phase (never more than the input length,
    /// since each item lands in at most one phase).
    pub fn drawn_count(&self) -> usize {
        self.early.len() + self.late.len()
    }
}

/// Partitions `work` into the early and late raster phases by stage mask.
///
/// Each item is routed by [`raster_phase_of`] into exactly one phase (or
/// dropped when unclassified), so the two lists are disjoint and never redraw
/// an instance. The traversal is linear and allocates only the two output
/// lists.
pub fn plan_two_phase_raster(work: &[GpuRenderWorkItem]) -> TwoPhaseRasterPlan {
    let mut plan = TwoPhaseRasterPlan::default();
    for item in work {
        match raster_phase_of(item.visibility_stages) {
            Some(RasterPhase::Early) => plan.early.push(*item),
            Some(RasterPhase::Late) => plan.late.push(*item),
            None => {}
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RenderPassMask, WorkSortKey};
    use alloc::vec;
    use prism_render_architecture::abi::GenerationalHandle;

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle {
            index,
            generation: 1,
        }
    }

    fn item(scene_index: u32, stages: VisibilityStageMask) -> GpuRenderWorkItem {
        GpuRenderWorkItem {
            scene: handle(scene_index),
            geometry: handle(scene_index),
            material: handle(scene_index),
            lod_or_cluster: 0,
            pass_mask: RenderPassMask::OPAQUE,
            visibility_stages: stages,
            sort_key: WorkSortKey::default(),
        }
    }

    #[test]
    fn early_only_and_deferred_revealed_route_to_distinct_phases() {
        let work = vec![
            item(1, VisibilityStageMask::EARLY),
            item(
                2,
                VisibilityStageMask::LATE_RETEST | VisibilityStageMask::LATE_VISIBLE,
            ),
            item(3, VisibilityStageMask::EARLY),
        ];
        let plan = plan_two_phase_raster(&work);
        assert_eq!(plan.early.len(), 2);
        assert_eq!(plan.late.len(), 1);
        assert_eq!(plan.early[0].scene, handle(1));
        assert_eq!(plan.early[1].scene, handle(3));
        assert_eq!(plan.late[0].scene, handle(2));
        assert_eq!(plan.drawn_count(), work.len());
    }

    #[test]
    fn early_visible_item_is_not_redrawn_in_the_late_pass() {
        // An item flagged both early and late-visible was already drawn early;
        // precedence keeps it out of the late list to avoid a double draw.
        let both = VisibilityStageMask::EARLY | VisibilityStageMask::LATE_VISIBLE;
        assert_eq!(raster_phase_of(both), Some(RasterPhase::Early));
        let plan = plan_two_phase_raster(&[item(1, both)]);
        assert_eq!(plan.early.len(), 1);
        assert!(plan.late.is_empty());
    }

    #[test]
    fn bare_late_retest_without_reveal_is_not_rastered() {
        // A candidate deferred early but still occluded late carries only
        // LATE_RETEST; it is drawn by neither phase.
        assert_eq!(raster_phase_of(VisibilityStageMask::LATE_RETEST), None);
        let plan = plan_two_phase_raster(&[item(1, VisibilityStageMask::LATE_RETEST)]);
        assert!(plan.early.is_empty());
        assert!(plan.late.is_empty());
        assert_eq!(plan.drawn_count(), 0);
    }

    #[test]
    fn input_order_is_preserved_within_each_phase() {
        let late = VisibilityStageMask::LATE_RETEST | VisibilityStageMask::LATE_VISIBLE;
        let work = vec![
            item(10, late),
            item(20, VisibilityStageMask::EARLY),
            item(30, late),
            item(40, VisibilityStageMask::EARLY),
        ];
        let plan = plan_two_phase_raster(&work);
        assert_eq!(
            plan.early.iter().map(|i| i.scene.index).collect::<Vec<_>>(),
            vec![20, 40]
        );
        assert_eq!(
            plan.late.iter().map(|i| i.scene.index).collect::<Vec<_>>(),
            vec![10, 30]
        );
    }
}
