//! `Core3d` node recording the three motion-blur dispatches for every view.
//!
//! Mirrors [`super::super::ssr::reconstruct::ssr_reconstruct_pass`] and the TAA
//! resolve node: gate on the enable, resolve the view, then record the chain in
//! order. The three passes share one immediate block and run:
//!
//! 1. `tile_max` — one workgroup per 8x8 *tile* block, reducing every tile to
//!    its longest velocity;
//! 2. `neighbor_max` — same tile-grid dispatch, dilating that field by one
//!    ring; and
//! 3. `reconstruct` — one workgroup per 8x8 *pixel* block, gathering the blur.
//!
//! Every dispatch bounds-checks its invocations in the shader. Placed after the
//! post-composite tonemapping inputs are ready (it reads the composited HDR
//! `scene_color` and the SSR device depth) and gated on
//! [`PrismMotionBlurSettings::enabled`].

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::{MotionBlurParams, MOTION_BLUR_WORKGROUP_SIZE};
use super::bind_groups::ViewMotionBlurBindGroups;
use super::pipeline::MotionBlurPipeline;
use super::resources::ViewMotionBlur;
use super::settings::PrismMotionBlurSettings;

/// `Core3d` node recording the `TileMax` -> `NeighborMax` -> reconstruction chain
/// for every view whose motion-blur textures and bind groups are resident.
pub(crate) fn motion_blur_pass(
    settings: Res<PrismMotionBlurSettings>,
    view: ViewQuery<(&ViewMotionBlur, &ViewMotionBlurBindGroups, &ExtractedView)>,
    pipeline: Res<MotionBlurPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (motion_blur, groups, extracted) = view.into_inner();

    // All three pipelines must be resident before any pass runs; the chain is
    // meaningless with a missing link.
    let Some(tile_max_pipeline) = cache.get_compute_pipeline(pipeline.tile_max()) else {
        return;
    };
    let Some(neighbor_max_pipeline) = cache.get_compute_pipeline(pipeline.neighbor_max()) else {
        return;
    };
    let Some(reconstruct_pipeline) = cache.get_compute_pipeline(pipeline.reconstruct()) else {
        return;
    };

    let size = motion_blur.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Inverse projection reconstructs the same linear view-space depth the SSR
    // resolve uses for its bilateral term, so the soft-depth ordering is
    // scale-correct. The live settings drive the sample count and tunables.
    let view_from_clip = extracted.clip_from_view.inverse();
    let params = MotionBlurParams::from_settings(view_from_clip, size, &settings);

    let tiles = motion_blur.tiles;
    let tile_groups_x = tiles.x.div_ceil(MOTION_BLUR_WORKGROUP_SIZE);
    let tile_groups_y = tiles.y.div_ceil(MOTION_BLUR_WORKGROUP_SIZE);
    let pixel_groups_x = size.x.div_ceil(MOTION_BLUR_WORKGROUP_SIZE);
    let pixel_groups_y = size.y.div_ceil(MOTION_BLUR_WORKGROUP_SIZE);

    let immediates = bytemuck::bytes_of(&params);
    let encoder = ctx.command_encoder();

    // Pass 1: TileMax over the tile grid.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism motion blur tile max"),
            timestamp_writes: None,
        });
        pass.set_pipeline(tile_max_pipeline);
        pass.set_bind_group(0, groups.tile_max(), &[]);
        pass.set_immediates(0, immediates);
        pass.dispatch_workgroups(tile_groups_x, tile_groups_y, 1);
    }

    // Pass 2: NeighborMax over the same tile grid.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism motion blur neighbor max"),
            timestamp_writes: None,
        });
        pass.set_pipeline(neighbor_max_pipeline);
        pass.set_bind_group(0, groups.neighbor_max(), &[]);
        pass.set_immediates(0, immediates);
        pass.dispatch_workgroups(tile_groups_x, tile_groups_y, 1);
    }

    // Pass 3: reconstruction over the full-resolution pixel grid.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism motion blur reconstruct"),
            timestamp_writes: None,
        });
        pass.set_pipeline(reconstruct_pipeline);
        pass.set_bind_group(0, groups.reconstruct(), &[]);
        pass.set_immediates(0, immediates);
        pass.dispatch_workgroups(pixel_groups_x, pixel_groups_y, 1);
    }
}
