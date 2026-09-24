//! `Core3d` graph node compositing the resolved HDR radiance onto the view.
//!
//! Terminal step of the visibility -> classify -> shade -> *composite* chain.
//! It runs *after* [`Core3dSystems::MainPass`](bevy_core_pipeline::core_3d::graph)
//! (which cleared the view target and drew any non-Prism geometry) and *before*
//! Bevy's tonemapping node.  For every covered pixel it copies the resolved
//! linear radiance from `scene_color` into the view target; uncovered pixels are
//! `discard`ed in `composite.wesl`, so the main pass' clear/background survives.
//!
//! All heavy lifting was done upstream: [`super::pipeline`] already specialized
//! the render pipeline for the view's target format, and [`super::bind_groups`]
//! already built the group-0 bind group.  This node only records the fullscreen
//! draw, skipping any view without both — plus the same `enable_visibility_buffer`
//! / single-sample gate the raster and resolve passes use, so the composite can
//! never run for a frame the rest of the chain sat out.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{PipelineCache, RenderPassDescriptor},
    renderer::{RenderContext, ViewQuery},
    view::{Msaa, ViewTarget},
};

use super::super::runtime::PrismShadingSettings;
use super::bind_groups::ViewCompositeBindGroup;
use super::pipeline::ViewCompositePipelineId;

pub(crate) fn composite_shading(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(
        &ViewTarget,
        &ViewCompositeBindGroup,
        &ViewCompositePipelineId,
        Option<&Msaa>,
    )>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let (target, bind_group, pipeline_id, msaa) = view.into_inner();
    // The Prism visibility path is single-sample only; a multisampled view never
    // wrote `scene_color`, so there is nothing to composite.
    if msaa.is_some_and(|value| value.samples() != 1) {
        return;
    }
    let Some(pipeline) = cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    // The main pass already ran, so `get_color_attachment` loads (never clears)
    // the target: covered pixels overwrite it, uncovered pixels are discarded,
    // preserving the cleared background/non-Prism geometry underneath.
    let color_attachment = target.get_color_attachment();
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism shading composite"),
        color_attachments: &[Some(color_attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group.0, &[]);
    // Fullscreen triangle: three vertices, one instance, no vertex/index buffer.
    pass.draw(0..3, 0..1);
}
