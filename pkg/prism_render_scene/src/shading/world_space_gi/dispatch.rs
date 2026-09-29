//! `Core3d` scheduling system recording the two world-space GI dispatches for
//! every view.
//!
//! Mirrors [`super::super::ssgi::trace`]'s dispatch: gate on the enable,
//! resolve the view, recover the inverse projection + near plane from the
//! [`ExtractedView`], then run the probe-update pass (one workgroup per 8x8
//! *probe* block) followed by the resolve pass (one workgroup per 8x8 *pixel*
//! block). Both entry points bounds-check every invocation.
//!
//! Unlike the `scene_color` post-processors, the GI result is an *export*
//! buffer (`gi_out`) a downstream resolve samples, so there is no copy back
//! over `scene_color`.

use bevy_ecs::prelude::*;
use bevy_math::Vec4;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::WORLD_SPACE_GI_WORKGROUP_SIZE;
use super::bind_groups::ViewWorldSpaceGiBindGroups;
use super::pipeline::WorldSpaceGiPipeline;
use super::resources::ViewWorldSpaceGi;
use super::settings::PrismWorldSpaceGiSettings;

/// `Core3d` scheduling system recording the `probe_update_main` +
/// `resolve_main` dispatches for every view whose GI resources and bind groups
/// are resident.
pub(crate) fn world_space_gi_pass(
    settings: Res<PrismWorldSpaceGiSettings>,
    view: ViewQuery<(
        &ViewWorldSpaceGi,
        &ViewWorldSpaceGiBindGroups,
        &ExtractedView,
    )>,
    pipeline: Res<WorldSpaceGiPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (gi, groups, extracted) = view.into_inner();

    // Both pipelines must be resident before the pass runs.
    let Some(probe_update_pipeline) = cache.get_compute_pipeline(pipeline.probe_update()) else {
        return;
    };
    let Some(resolve_pipeline) = cache.get_compute_pipeline(pipeline.resolve()) else {
        return;
    };

    let size = gi.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Inverse projection (clip -> view) used to reconstruct view-space
    // positions in both passes; near plane recovered the same way as SSGI.
    let clip_from_view = extracted.clip_from_view;
    let view_from_clip = clip_from_view.inverse();
    // Reverse-Z: device depth 1.0 is the near plane, so inverse-projecting
    // clip (0, 0, 1, 1) yields a view-space point at `-near` along `-Z`.
    let near_view = view_from_clip * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let near = if near_view.w.abs() > f32::EPSILON {
        (near_view.z / near_view.w).abs().max(1.0e-3)
    } else {
        1.0e-3
    };

    let probe_grid = gi.probe_grid;
    let probe_params = settings.probe_params(size, probe_grid, view_from_clip, near);
    let resolve_params = settings.resolve_params(size, probe_grid, view_from_clip);

    let probe_groups_x = probe_grid.x.div_ceil(WORLD_SPACE_GI_WORKGROUP_SIZE);
    let probe_groups_y = probe_grid.y.div_ceil(WORLD_SPACE_GI_WORKGROUP_SIZE);
    let resolve_groups_x = size.x.div_ceil(WORLD_SPACE_GI_WORKGROUP_SIZE);
    let resolve_groups_y = size.y.div_ceil(WORLD_SPACE_GI_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    // Pass 1: capture one screen probe per tile into the probe storage buffer.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism world-space GI probe update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(probe_update_pipeline);
        pass.set_bind_group(0, groups.probe_update_group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&probe_params));
        pass.dispatch_workgroups(probe_groups_x, probe_groups_y, 1);
    }

    // Pass 2: interpolate the four surrounding probes per pixel into gi_out.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism world-space GI resolve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resolve_pipeline);
        pass.set_bind_group(0, groups.resolve_group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&resolve_params));
        pass.dispatch_workgroups(resolve_groups_x, resolve_groups_y, 1);
    }
}
