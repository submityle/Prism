//! `Core3d` scheduling system recording the posterize dispatch for every view.
//!
//! Mirrors [`super::super::color_grade::dispatch`]: gate on the enable, resolve
//! the view, upload the immediate block, run one workgroup per 8x8 *pixel* block
//! and copy the posterized output back over `scene_color`.
//!
//! The dispatch bounds-checks its invocations in the shader. Placed after the
//! composited `HDR` `scene_color` is ready and gated on
//! [`PrismPosterizeSettings::enabled`]. The output is copied back over
//! `scene_color` so the downstream post chain reads the posterized image.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::POSTERIZE_WORKGROUP_SIZE;
use super::bind_groups::ViewPosterizeBindGroup;
use super::pipeline::PosterizePipeline;
use super::resources::ViewPosterize;
use super::settings::PrismPosterizeSettings;

/// `Core3d` scheduling system recording the `posterize_main` dispatch for every
/// view whose posterize texture and bind group are resident.
pub(crate) fn posterize_pass(
    settings: Res<PrismPosterizeSettings>,
    view: ViewQuery<(
        &ViewPosterize,
        &ViewPosterizeBindGroup,
        &ViewVisibilityBuffer,
    )>,
    pipeline: Res<PosterizePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (posterize, group, visibility) = view.into_inner();

    // The pipeline must be resident before the pass runs.
    let Some(posterize_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let size = posterize.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = settings.params(size);

    let groups_x = size.x.div_ceil(POSTERIZE_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(POSTERIZE_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism posterize"),
            timestamp_writes: None,
        });
        pass.set_pipeline(posterize_pipeline);
        pass.set_bind_group(0, group.group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the posterized result back over `scene_color` so the downstream post
    // chain (which samples `scene_color`, not `posterize_out`) reads the
    // posterized image. The pass cannot write `scene_color` in place because it
    // binds it as a sampled read, so the chain ping-pongs into a dedicated
    // target and then blits back. Same extent + same format
    // (SCENE_COLOR_FORMAT), single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: posterize.posterize_out_texture(),
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
