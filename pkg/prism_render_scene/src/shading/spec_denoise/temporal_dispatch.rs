//! `Core3d` dispatch nodes recording the specular-GI *temporal denoise* passes
//! (reproject + history-clamp) for every view.
//!
//! These two compute nodes sit between the `spec_gi` reuse resolve and the
//! spatial pre-filter, forming the ReBLUR/ReLAX-style temporal accumulator:
//!
//! * [`spec_denoise_reproject_pass`] first stages the current frame's SSR
//!   reverse-Z `scene_depth` into the view's persistent write-slot depth plane
//!   (so next frame it is the `prev_depth` the world-space disocclusion guard
//!   reads), then dispatches `spec_denoise_reproject.wesl`, which follows each
//!   pixel's specular *virtual* reflection point back into last frame's
//!   accumulated history and writes the reprojected history/meta/luma planes.
//! * [`spec_denoise_history_clamp_pass`] dispatches
//!   `spec_denoise_history_clamp.wesl`, fusing that reprojected history with the
//!   current-frame noisy `spec_gi` resolve under an AABB colour clamp, advancing
//!   the age + dual-rate luminance EMAs, and writing both the next-frame history
//!   planes and the `denoised` plane the spatial pass filters in place of the
//!   raw resolve.
//!
//! Like [`super::dispatch::spec_denoise_spatial_pass`], each kernel reads its
//! config from the bound uniform the bind-group slice uploaded, so these nodes
//! record **no** `set_immediates`. Both dispatch one workgroup per
//! [`SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE`]×[`SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE`]
//! tile and the kernels bounds-check every invocation against the config extent.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        ComputePassDescriptor, Extent3d, Origin3d, PipelineCache, TexelCopyTextureInfo,
        TextureAspect,
    },
    renderer::{RenderContext, ViewQuery},
};

use super::super::ssr::ViewSsrTextures;
use super::temporal_abi::SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE;
use super::temporal_bind_groups::{
    ViewSpecDenoiseHistoryClampBindGroup, ViewSpecDenoiseReprojectBindGroup,
};
use super::temporal_pipeline::{SpecDenoiseHistoryClampPipeline, SpecDenoiseReprojectPipeline};
use super::temporal_resources::ViewSpecDenoiseTemporal;

/// `Core3d` node recording the temporal reproject dispatch for every view whose
/// temporal planes and reproject bind group are resident.
///
/// Gated on `enable_spec_gi`; the per-view [`ViewSpecDenoiseReprojectBindGroup`]
/// is only present when the subsystem's full gate (SSR + visibility buffer +
/// single-sample) already held in resource prep, so a resident bind group is
/// sufficient to dispatch.
///
/// Before the compute pass it copies the current frame's SSR reverse-Z
/// `scene_depth` into the view's write-slot depth plane (`COPY_SRC` →
/// `COPY_DST`, same `R32Float` extent/format); the two textures are distinct, so
/// the copy only stages next frame's `prev_depth` and does not alias this
/// frame's reproject reads.
pub(crate) fn spec_denoise_reproject_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(
        &ViewSpecDenoiseTemporal,
        &ViewSsrTextures,
        &ViewSpecDenoiseReprojectBindGroup,
    )>,
    pipeline: Res<SpecDenoiseReprojectPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_spec_gi {
        return;
    }
    let (temporal, ssr, bind_group) = view.into_inner();

    let Some(reproject) = cache.get_compute_pipeline(pipeline.reproject()) else {
        return;
    };

    let size = temporal.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Stage this frame's SSR reverse-Z depth into the write-slot plane so next
    // frame it is the `prev_depth` the world-space disocclusion guard reads.
    ctx.command_encoder().copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: ssr.scene_depth_texture(),
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
            texture: temporal.curr_depth_texture(),
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

    let workgroups_x = size.x.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism spec_denoise reproject"),
            timestamp_writes: None,
        });
    pass.set_pipeline(reproject);
    pass.set_bind_group(0, bind_group.group(), &[]);
    // No `set_immediates`: the config travels in the bound uniform at binding 0.
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

/// `Core3d` node recording the temporal history-clamp dispatch for every view
/// whose temporal planes and history-clamp bind group are resident.
///
/// Gated on `enable_spec_gi`; runs after [`spec_denoise_reproject_pass`] so the
/// reprojected planes it clamps are this frame's. Unlike the reproject node it
/// records no copy — its depth input is the same current-frame SSR plane.
pub(crate) fn spec_denoise_history_clamp_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(
        &ViewSpecDenoiseTemporal,
        &ViewSpecDenoiseHistoryClampBindGroup,
    )>,
    pipeline: Res<SpecDenoiseHistoryClampPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_spec_gi {
        return;
    }
    let (temporal, bind_group) = view.into_inner();

    let Some(history_clamp) = cache.get_compute_pipeline(pipeline.history_clamp()) else {
        return;
    };

    let size = temporal.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let workgroups_x = size.x.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism spec_denoise history_clamp"),
            timestamp_writes: None,
        });
    pass.set_pipeline(history_clamp);
    pass.set_bind_group(0, bind_group.group(), &[]);
    // No `set_immediates`: the config travels in the bound uniform at binding 0.
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_count_rounds_up_to_cover_every_texel() {
        // Both temporal dispatches must cover a partial trailing tile: a
        // 1920x1080 target at the 8x8 workgroup needs ceil(1920/8)=240 x
        // ceil(1080/8)=135 groups, and a non-multiple extent still rounds up so
        // no edge texels drop.
        assert_eq!(1920u32.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE), 240);
        assert_eq!(1080u32.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE), 135);
        assert_eq!(1u32.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE), 1);
        assert_eq!(9u32.div_ceil(SPEC_DENOISE_TEMPORAL_WORKGROUP_SIZE), 2);
    }
}
