//! `Core3d` graph node recording the SSR geometry-prepass dispatch.
//!
//! Runs after the visibility raster (so the ids/metadata targets are filled)
//! and before the shading resolve that will ultimately consume the reflection
//! buffer. For each view with a resident [`ViewSsrPrepassBindGroups`] it binds
//! the two groups and dispatches one workgroup per 8x8 pixel tile, letting the
//! shader's per-pixel guards drop background and stale samples. Mirrors
//! [`super::super::ao::gtao_prepass_pass`].

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::{GpuSsrPrepassParams, SSR_WORKGROUP_SIZE};
use super::bind_groups::ViewSsrPrepassBindGroups;
use super::pipeline::SsrPrepassPipeline;
use super::resources::ViewSsrTextures;

pub(crate) fn ssr_prepass_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrPrepassBindGroups, &ExtractedView)>,
    pipeline: Res<SsrPrepassPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
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
    // the RH, camera-at-origin view frame; the projection then yields the
    // reverse-Z device depth the screen-space march compares against.
    let view_from_world = extracted.world_from_view.to_matrix().inverse();
    let params = GpuSsrPrepassParams::new(
        view_from_world,
        extracted.clip_from_view,
        size.x,
        size.y,
    );

    let workgroups_x = size.x.div_ceil(SSR_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSR_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSR prepass"),
            timestamp_writes: None,
        });
    pass.set_pipeline(prepass);
    pass.set_bind_group(0, &groups.view, &[]);
    pass.set_bind_group(1, &groups.scene, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
