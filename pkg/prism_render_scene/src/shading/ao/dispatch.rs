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

use super::abi::{
    GpuGtaoConfig, GpuGtaoPrepassParams, GTAO_KERNEL_WORKGROUP_SIZE, GTAO_PREPASS_WORKGROUP_SIZE,
};
use super::bind_groups::{ViewGtaoKernelBindGroup, ViewGtaoPrepassBindGroups};
use super::pipeline::{GtaoKernelPipeline, GtaoPrepassPipeline};
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


/// `Core3d` graph node recording the GTAO kernel dispatch.
///
/// Runs after [`gtao_prepass_pass`] (so the linear-depth/view-normal targets
/// are populated) and before the shading resolve that samples the resulting
/// ambient-visibility target. For each view with a resident
/// [`ViewGtaoKernelBindGroup`] it uploads the projection-derived config and
/// dispatches one workgroup per 8x8 pixel tile.
pub(crate) fn gtao_compute_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewGtaoTextures, &ViewGtaoKernelBindGroup, &ExtractedView)>,
    pipeline: Res<GtaoKernelPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_gtao {
        return;
    }
    let (textures, group, extracted) = view.into_inner();

    let Some(kernel) = cache.get_compute_pipeline(pipeline.kernel) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // The kernel reconstructs view positions from the perspective diagonal
    // exactly like the golden `GtaoCamera::from_projection`.
    let clip_from_view = extracted.clip_from_view;
    let config = GpuGtaoConfig::from_projection(
        clip_from_view.x_axis.x,
        clip_from_view.y_axis.y,
        settings.gtao_world_radius,
        settings.gtao_falloff,
        settings.gtao_power,
        settings.gtao_slice_count,
        settings.gtao_steps_per_slice,
    );

    let workgroups_x = size.x.div_ceil(GTAO_KERNEL_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(GTAO_KERNEL_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism GTAO kernel"),
            timestamp_writes: None,
        });
    pass.set_pipeline(kernel);
    pass.set_bind_group(0, &group.view, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
