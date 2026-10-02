//! `Core3d` scheduling system recording the ordered-dither dispatch for every
//! view.
//!
//! Mirrors [`super::super::gamut_map::dispatch`]: gate on the enable, resolve
//! the view, upload the immediate block, run one workgroup per 8x8 *pixel* block
//! and copy the dithered output back over `scene_color`.
//!
//! Ordered dithering is display-referred and conventionally runs *after* the
//! display tone-map curve (it snaps to the final display's palette / bit depth).
//! This pipeline currently keeps tone mapping in the Bevy `MainPass`-and-after
//! node chain, so every pre-`MainPass` post effect (this one included) operates
//! on the linear HDR `scene_color`; the pass is placed late in that linear chain
//! and this ordering may be revisited once the tone-map ownership is decided.
//! Gated on [`PrismOrderedDitherSettings::enabled`]; the output is copied back
//! over `scene_color` so the downstream post chain reads the dithered image.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::ORDERED_DITHER_WORKGROUP_SIZE;
use super::bind_groups::ViewOrderedDitherBindGroup;
use super::pipeline::OrderedDitherPipeline;
use super::resources::ViewOrderedDither;
use super::settings::PrismOrderedDitherSettings;

/// `Core3d` scheduling system recording the `ordered_dither_main` dispatch for
/// every view whose ordered-dither texture and bind group are resident.
pub(crate) fn ordered_dither_pass(
    settings: Res<PrismOrderedDitherSettings>,
    view: ViewQuery<(
        &ViewOrderedDither,
        &ViewOrderedDitherBindGroup,
        &ViewVisibilityBuffer,
    )>,
    pipeline: Res<OrderedDitherPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (ordered_dither, group, visibility) = view.into_inner();

    // The pipeline must be resident before the pass runs.
    let Some(ordered_dither_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let size = ordered_dither.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = settings.params(size);

    let groups_x = size.x.div_ceil(ORDERED_DITHER_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(ORDERED_DITHER_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism ordered dither"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ordered_dither_pipeline);
        pass.set_bind_group(0, group.group(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the dithered result back over scene_color so the downstream post
    // chain (which samples scene_color, not ordered_dither_out) reads the
    // dithered image. The pass cannot write scene_color in place because it
    // binds scene_color as a sampled read, so the chain ping-pongs into a
    // dedicated target and then blits back. Same extent + same format
    // (SCENE_COLOR_FORMAT), single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: ordered_dither.ordered_dither_out_texture(),
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
