//! `Core3d` scheduling system recording the cross-hatching dispatch for every
//! view.
//!
//! Mirrors [`super::super::color_grade::dispatch`]: gate on the enable, resolve
//! the view, upload the immediate block, run one workgroup per 8x8 *pixel* block
//! and copy the filtered output back over `scene_color`.
//!
//! The dispatch bounds-checks its invocations in the shader. Placed after the
//! composited HDR `scene_color` is ready and gated on
//! [`PrismHatchingSettings::enabled`]. The output is copied back over
//! `scene_color` so the downstream post chain reads the filtered image.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::abi::HATCHING_WORKGROUP_SIZE;
use super::bind_groups::ViewHatchingBindGroup;
use super::pipeline::HatchingPipeline;
use super::resources::ViewHatching;
use super::settings::PrismHatchingSettings;
use super::super::resources::ViewVisibilityBuffer;

/// `Core3d` scheduling system recording the `hatching_main` dispatch for every
/// view whose hatching texture and bind group are resident.
pub(crate) fn hatching_pass(
    settings: Res<PrismHatchingSettings>,
    view: ViewQuery<(
        &ViewHatching,
        &ViewHatchingBindGroup,
        &ViewVisibilityBuffer,
    )>,
    pipeline: Res<HatchingPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (hatching, group, visibility) = view.into_inner();

    // The pipeline must be resident before the pass runs.
    let Some(hatching_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let size = hatching.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = settings.params(size);

    let groups_x = size.x.div_ceil(HATCHING_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(HATCHING_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism hatching"),
            timestamp_writes: None,
        });
        pass.set_pipeline(hatching_pipeline);
        pass.set_bind_group(0, group.group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the filtered result back over scene_color so the downstream post
    // chain (which samples scene_color, not hatching_out) reads the filtered
    // image. The pass cannot write scene_color in place because it binds
    // scene_color as a sampled read, so the chain ping-pongs into a dedicated
    // target and then blits back. Same extent + same format (SCENE_COLOR_FORMAT),
    // single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: hatching.hatching_out_texture(),
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
