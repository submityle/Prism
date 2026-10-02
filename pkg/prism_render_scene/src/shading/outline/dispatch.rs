//! `Core3d` scheduling system recording the outline dispatch for every view.
//!
//! Mirrors [`super::super::color_grade::dispatch`]: gate on the enable, resolve
//! the view, upload the immediate block, run one workgroup per 8x8 *pixel* block
//! and copy the outlined output back over `scene_color`.
//!
//! The dispatch bounds-checks its invocations in the shader. Placed after the
//! composited HDR `scene_color` and the SSR device depth / view-space normal are
//! ready and gated on [`PrismOutlineSettings::enabled`]. The output is copied
//! back over `scene_color` so the downstream post chain reads the outlined
//! image.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::OUTLINE_WORKGROUP_SIZE;
use super::bind_groups::ViewOutlineBindGroup;
use super::pipeline::OutlinePipeline;
use super::resources::ViewOutline;
use super::settings::PrismOutlineSettings;

/// `Core3d` scheduling system recording the `outline_main` dispatch for every
/// view whose outline texture and bind group are resident.
pub(crate) fn outline_pass(
    settings: Res<PrismOutlineSettings>,
    view: ViewQuery<(
        &ViewOutline,
        &ViewOutlineBindGroup,
        &ViewVisibilityBuffer,
        &ExtractedView,
    )>,
    pipeline: Res<OutlinePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (outline, group, visibility, extracted) = view.into_inner();

    // The pipeline must be resident before the pass runs.
    let Some(outline_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let size = outline.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Inverse projection reconstructs the same linear view-space depth the SSR
    // prepass validated, so the golden depth edge sees the distances the CPU
    // reference was checked against. The live settings drive the line colour and
    // the edge thresholds.
    let view_from_clip = extracted.clip_from_view.inverse();
    let params = settings.params(view_from_clip, size);

    let groups_x = size.x.div_ceil(OUTLINE_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(OUTLINE_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism outline"),
            timestamp_writes: None,
        });
        pass.set_pipeline(outline_pipeline);
        pass.set_bind_group(0, group.group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the outlined result back over scene_color so the downstream post
    // chain (which samples scene_color, not outline_out) reads the outlined
    // image. The pass cannot write scene_color in place: it binds scene_color as
    // a sampled read and samples neighbouring texels across the cross, so the
    // chain ping-pongs into a dedicated target and then blits back. Same extent
    // + same format (SCENE_COLOR_FORMAT), single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: outline.outline_out_texture(),
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: visibility.scene_color_texture(),
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
}
