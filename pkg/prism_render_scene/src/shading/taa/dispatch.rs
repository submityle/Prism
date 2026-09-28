//! `Core3d` node recording the TAA resolve dispatch for every view.
//!
//! Mirrors [`super::super::ssr::temporal`]'s dispatch node. Runs after
//! `ssr_composite` (which folds reflections into `scene_color`) and before the
//! main pass, so it resolves the fully composited opaque HDR radiance while the
//! forward-transparency composite stays sharp afterwards. Dispatches one
//! workgroup per 8x8 pixel tile; the shader bounds-checks every invocation and
//! falls back to the current frame wherever the reprojection or the history is
//! invalid. Gated on [`PrismShadingSettings::enable_taa`].

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::abi::{GpuTaaResolveParams, TAA_WORKGROUP_SIZE};
use super::super::resources::ViewVisibilityBuffer;
use super::bind_groups::ViewTaaBindGroup;
use super::pipeline::TaaResolvePipeline;
use super::resources::ViewTaa;

/// `Core3d` node recording the TAA resolve dispatch for every view.
///
/// Reads the composited `scene_color` + the motion-vector G-buffer + last
/// frame's history and writes this frame's resolved output into the ping-pong
/// write slot the shading composite then reads in place of `scene_color`.
pub(crate) fn taa_resolve_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewVisibilityBuffer, &ViewTaaBindGroup, &ViewTaa)>,
    pipeline: Res<TaaResolvePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_taa {
        return;
    }
    let (visibility, group, taa) = view.into_inner();

    let Some(resolve) = cache.get_compute_pipeline(pipeline.resolve()) else {
        return;
    };

    let size = visibility.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // The reprojection reads the motion-vector G-buffer on the GPU, so the
    // params only carry the framebuffer extent and the history-valid flag.
    let params = GpuTaaResolveParams::new(size.x, size.y, taa.valid());

    let workgroups_x = size.x.div_ceil(TAA_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(TAA_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism TAA resolve"),
            timestamp_writes: None,
        });
    pass.set_pipeline(resolve);
    pass.set_bind_group(0, group.group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
