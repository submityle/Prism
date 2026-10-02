//! `Core3d` scheduling system recording the tone-map dispatch for every view.
//!
//! Mirrors [`super::super::gamut_map::dispatch`]: gate on the enable, resolve the
//! view, upload the immediate block, run one workgroup per 8x8 *pixel* block and
//! copy the tone-mapped output back over `scene_color`.
//!
//! Tone mapping compresses the open-domain pre-exposed linear HDR radiance into
//! the display-referred `[0, 1]` range. It is scheduled late in the pre-`MainPass`
//! post chain — after `lens_flare_pass` and *before* `gamut_map_pass`, since
//! gamut mapping is conventionally applied after the tone-map curve. Gated on
//! [`PrismTonemapSettings::enabled`] (opt-in, off by default); the output is
//! copied back over `scene_color` so the downstream chain reads the tone-mapped
//! image. When enabled the camera must be pinned to `Tonemapping::None` so the
//! Bevy `PostProcess` tone-map node does not double-map (see
//! [`super::settings`]).

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::TONEMAP_WORKGROUP_SIZE;
use super::bind_groups::ViewTonemapBindGroup;
use super::pipeline::TonemapPipeline;
use super::resources::ViewTonemap;
use super::settings::PrismTonemapSettings;

/// `Core3d` scheduling system recording the `tonemap_main` dispatch for every
/// view whose tone-map texture and bind group are resident.
pub(crate) fn tonemap_pass(
    settings: Res<PrismTonemapSettings>,
    view: ViewQuery<(&ViewTonemap, &ViewTonemapBindGroup, &ViewVisibilityBuffer)>,
    pipeline: Res<TonemapPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (tonemap, group, visibility) = view.into_inner();

    // The pipeline must be resident before the pass runs.
    let Some(tonemap_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let size = tonemap.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = settings.params(size);

    let groups_x = size.x.div_ceil(TONEMAP_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(TONEMAP_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism tonemap"),
            timestamp_writes: None,
        });
        pass.set_pipeline(tonemap_pipeline);
        pass.set_bind_group(0, group.group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the tone-mapped result back over scene_color so the downstream post
    // chain (which samples scene_color, not tonemap_out) reads the tone-mapped
    // image. The pass cannot write scene_color in place because it binds
    // scene_color as a sampled read, so the chain ping-pongs into a dedicated
    // target and then blits back. Same extent + same format (SCENE_COLOR_FORMAT),
    // single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: tonemap.tonemap_out_texture(),
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
