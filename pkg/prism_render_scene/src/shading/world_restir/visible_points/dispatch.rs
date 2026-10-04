//! `Core3d` scheduling system recording the world-space `ReSTIR` visible-point
//! producer dispatch for every view.
//!
//! Gate on the enable, resolve the view, recover the inverse projection and the
//! view->world matrix from the [`ExtractedView`], build the
//! [`GpuVisiblePointsParams`] immediate block from the resident tile grid, then
//! dispatch one 2-D workgroup block per [`VISIBLE_POINTS_WORKGROUP_SIZE`] tiles
//! on each axis. `visible_points_main` bounds-checks every invocation against
//! the live `(tiles_x, tiles_y)` and `point_count`, so the rounded extent never
//! writes a record past the list.
//!
//! The producer runs before the inject pass (it fills the list inject hashes),
//! so it is the first world-space `ReSTIR` pass on the render graph each frame.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::super::settings::PrismWorldRestirSettings;
use super::abi::{GpuVisiblePointsParams, VISIBLE_POINTS_TILE_SIZE, VISIBLE_POINTS_WORKGROUP_SIZE};
use super::bind_groups::ViewWorldRestirVisiblePointsBindGroup;
use super::pipeline::WorldRestirVisiblePointsPipeline;
use super::resources::ViewWorldRestirVisiblePoints;

/// Number of 1-D workgroups the producer dispatch records along one tile axis
/// of `tiles` tiles: one workgroup per [`VISIBLE_POINTS_WORKGROUP_SIZE`] tiles,
/// rounded up, with a floor of one so a degenerate `tiles == 0` axis still
/// issues a single (fully bounds-checked, no-op) workgroup.
fn producer_workgroups(tiles: u32) -> u32 {
    tiles.max(1).div_ceil(VISIBLE_POINTS_WORKGROUP_SIZE)
}

/// `Core3d` scheduling system recording the `visible_points_main` dispatch for
/// every view whose resident visible-point list and bind group are live.
pub(crate) fn world_restir_visible_points_pass(
    settings: Res<PrismWorldRestirSettings>,
    view: ViewQuery<(
        &ViewWorldRestirVisiblePoints,
        &ViewWorldRestirVisiblePointsBindGroup,
        &ExtractedView,
    )>,
    pipeline: Res<WorldRestirVisiblePointsPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (points, bind_group, extracted) = view.into_inner();

    // The producer pipeline must be resident before the pass runs.
    let Some(producer_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    // Inverse reverse-Z projection reconstructs the view-space shading point
    // from a tile-centre device depth; the view->world matrix then lifts the
    // point and its normal into world space for the `SHARC` hash.
    let view_from_clip = extracted.clip_from_view.inverse();
    let world_from_view = extracted.world_from_view.to_matrix();
    let params = GpuVisiblePointsParams::new(
        view_from_clip,
        world_from_view,
        points.screen(),
        points.tiles(),
        VISIBLE_POINTS_TILE_SIZE,
        points.point_count(),
    );

    let groups_x = producer_workgroups(points.tiles().x);
    let groups_y = producer_workgroups(points.tiles().y);

    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism world-space ReSTIR visible points"),
        timestamp_writes: None,
    });
    pass.set_pipeline(producer_pipeline);
    pass.set_bind_group(0, bind_group.group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(groups_x, groups_y, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn producer_workgroups_rounds_tiles_up_to_the_workgroup_size() {
        assert_eq!(producer_workgroups(VISIBLE_POINTS_WORKGROUP_SIZE), 1);
        assert_eq!(producer_workgroups(VISIBLE_POINTS_WORKGROUP_SIZE + 1), 2);
        assert_eq!(producer_workgroups(2 * VISIBLE_POINTS_WORKGROUP_SIZE), 2);
    }

    #[test]
    fn producer_workgroups_floors_degenerate_axis_at_one() {
        assert_eq!(producer_workgroups(0), 1);
        assert_eq!(producer_workgroups(1), 1);
    }
}
