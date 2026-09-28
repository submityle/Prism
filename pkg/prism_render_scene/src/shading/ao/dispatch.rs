//! `Core3d` graph node recording the GTAO geometry-prepass dispatch.
//!
//! Runs after the visibility raster (so the ids/metadata targets are filled)
//! and before the GTAO kernel that consumes its outputs.  For each view with a
//! resident [`ViewGtaoPrepassBindGroups`] it binds the two groups and dispatches
//! one workgroup per 8x8 pixel tile, letting the shader's per-pixel guards drop
//! background and stale samples.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::{GpuGtaoPrepassParams, GTAO_PREPASS_WORKGROUP_SIZE};
use super::bind_groups::ViewGtaoPrepassBindGroups;
use super::pipeline::GtaoPrepassPipeline;
use super::resources::ViewGtaoTextures;

pub(crate) fn gtao_prepass_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewGtaoTextures, &ViewGtaoPrepassBindGroups, &ExtractedView)>,
    pipeline: Res<GtaoPrepassPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_gtao {
        return;
    }
    let (textures, groups, extracted) = view.into_inner();

    let Some(prepass) = cache.get_compute_pipeline(pipeline.prepass) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // World-space vertices decoded from the visibility buffer are pushed into
    // the RH, camera-at-origin view frame the kernel integrates in.
    let view_from_world = extracted.world_from_view.to_matrix().inverse();
    let params = GpuGtaoPrepassParams {
        view_from_world: view_from_world.to_cols_array(),
        width: size.x,
        height: size.y,
        _pad0: 0,
        _pad1: 0,
    };

    let workgroups_x = size.x.div_ceil(GTAO_PREPASS_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(GTAO_PREPASS_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism GTAO prepass"),
            timestamp_writes: None,
        });
    pass.set_pipeline(prepass);
    pass.set_bind_group(0, &groups.view, &[]);
    pass.set_bind_group(1, &groups.scene, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
