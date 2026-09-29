//! `Core3d` scheduling system recording the chromatic-aberration dispatch for
//! every view.
//!
//! Mirrors [`super::super::dof::dispatch`]: gate on the enable, resolve the
//! view, upload the single immediate block, dispatch one workgroup per 8x8
//! *pixel* block, then copy the aberrated output back over `scene_color` so the
//! downstream post chain reads the fringed image. The pass cannot write
//! `scene_color` in place: each pixel fetches three radially split neighbours of
//! the scene colour, so an in-place write would feed already-aberrated texels
//! back into later pixels. It therefore writes a dedicated target and blits back
//! (same extent + same `SCENE_COLOR_FORMAT`, single mip/layer). The shader
//! bounds-checks every invocation against `screen_size`.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::abi::CHROMATIC_ABERRATION_WORKGROUP_SIZE;
use super::bind_groups::ViewChromaticAberrationBindGroups;
use super::pipeline::ChromaticAberrationPipeline;
use super::resources::ViewChromaticAberration;
use super::settings::PrismChromaticAberrationSettings;
use super::super::resources::ViewVisibilityBuffer;

/// `Core3d` scheduling system recording the chromatic-aberration dispatch for
/// every view whose aberration texture and bind group are resident, then
/// copying the aberrated output back over `scene_color`.
pub(crate) fn chromatic_aberration_pass(
    settings: Res<PrismChromaticAberrationSettings>,
    view: ViewQuery<(
        &ViewChromaticAberration,
        &ViewChromaticAberrationBindGroups,
        &ViewVisibilityBuffer,
    )>,
    pipeline: Res<ChromaticAberrationPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (chromatic, groups, visibility) = view.into_inner();

    let Some(compute_pipeline) = cache.get_compute_pipeline(pipeline.id()) else {
        return;
    };

    let size = chromatic.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = settings.params(size);
    let groups_x = size.x.div_ceil(CHROMATIC_ABERRATION_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(CHROMATIC_ABERRATION_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism chromatic aberration"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute_pipeline);
        pass.set_bind_group(0, groups.group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the aberrated result back over scene_color so the downstream post
    // chain (which samples scene_color, not chromatic_out) reads the fringed
    // image. Same extent + same format (SCENE_COLOR_FORMAT), single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: chromatic.chromatic_out_texture(),
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
