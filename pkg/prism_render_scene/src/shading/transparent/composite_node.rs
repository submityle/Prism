//! `Core3d` graph node blending resolved WBOIT transparency over the view.
//!
//! Runs *after* [`super::super::composite::composite_shading`] (which copied the
//! resolved opaque HDR radiance onto the view target) and *before* Bevy
//! tonemapping node. For every pixel it emits `(average_rgb, coverage)` from
//! `oit.wesl`, and the pipeline fixed-function `SrcAlpha`/`OneMinusSrcAlpha`
//! blend lays that over the opaque view target, reproducing
//! `OitAccumulation::resolve`. Pixels with no transparent coverage emit
//! coverage 0 and leave the target untouched.
//!
//! All heavy lifting was done upstream: [`super::composite_pipeline`] specialized
//! the render pipeline for the view target format, and
//! [`super::composite_bind_groups`] built the group-0 bind group. This node only
//! records the fullscreen draw, skipping any view without both - plus the same
//! `enable_visibility_buffer` / single-sample gate the rest of the chain uses.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{PipelineCache, RenderPassDescriptor},
    renderer::{RenderContext, ViewQuery},
    view::{Msaa, ViewTarget},
};

use super::super::runtime::PrismShadingSettings;
use super::composite_bind_groups::ViewOitCompositeBindGroup;
use super::composite_pipeline::ViewOitCompositePipelineId;

pub(crate) fn oit_composite(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(
        &ViewTarget,
        &ViewOitCompositeBindGroup,
        &ViewOitCompositePipelineId,
        Option<&Msaa>,
    )>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let (target, bind_group, pipeline_id, msaa) = view.into_inner();
    // The Prism visibility path is single-sample only.
    if msaa.is_some_and(|value| value.samples() != 1) {
        return;
    }
    let Some(pipeline) = cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    // The opaque composite already ran, so `get_color_attachment` loads (never
    // clears) the target; the enabled blend lays transparency over it.
    let color_attachment = target.get_color_attachment();
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("prism oit composite"),
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
