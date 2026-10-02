//! `Core3d` scheduling system recording the three depth-of-field dispatches for
//! every view.
//!
//! Mirrors [`super::super::motion_blur::dispatch`]: gate on the enable, resolve
//! the view, then record the chain in order. Unlike motion blur, each pass
//! uploads its *own* immediate block (the shader carries three `var<immediate>`
//! globals, one per entry point). All three run one workgroup per 8x8 *pixel*
//! block:
//!
//! 1. `dof_coc` — reconstructs the view distance and writes the near/far `CoC`
//!    radii;
//! 2. `dof_gather` — disk-bokeh blurs the scene colour by that field; and
//! 3. `dof_composite` — blends sharp toward blurred by the `CoC`.
//!
//! Every dispatch bounds-checks its invocations in the shader. Placed after the
//! composited HDR `scene_color` and the SSR device depth are ready and gated on
//! [`PrismDofSettings::enabled`]. The composited output is copied back over
//! `scene_color` so the downstream post chain reads the defocused image.

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
use super::abi::{GpuDofCocParams, GpuDofCompositeParams, GpuDofGatherParams, DOF_WORKGROUP_SIZE};
use super::bind_groups::ViewDofBindGroups;
use super::pipeline::DofPipeline;
use super::resources::ViewDof;
use super::settings::PrismDofSettings;

/// `Core3d` scheduling system recording the
/// `dof_coc` -> `dof_gather` -> `dof_composite` chain for every view whose `DoF`
/// textures and bind groups are resident.
pub(crate) fn dof_pass(
    settings: Res<PrismDofSettings>,
    view: ViewQuery<(
        &ViewDof,
        &ViewDofBindGroups,
        &ViewVisibilityBuffer,
        &ExtractedView,
    )>,
    pipeline: Res<DofPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (dof, groups, visibility, extracted) = view.into_inner();

    // All three pipelines must be resident before any pass runs; the chain is
    // meaningless with a missing link.
    let Some(coc_pipeline) = cache.get_compute_pipeline(pipeline.coc()) else {
        return;
    };
    let Some(gather_pipeline) = cache.get_compute_pipeline(pipeline.gather()) else {
        return;
    };
    let Some(composite_pipeline) = cache.get_compute_pipeline(pipeline.composite()) else {
        return;
    };

    let size = dof.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Inverse projection reconstructs the same linear view-space distance the
    // SSR prepass validated, so the golden optics see the distances the CPU
    // reference was checked against. The live settings drive the optics, the
    // gather tap budget and the blend.
    let view_from_clip = extracted.clip_from_view.inverse();
    let coc_params = GpuDofCocParams::from_settings(view_from_clip, size, &settings);
    let gather_params = GpuDofGatherParams::from_settings(size, &settings);
    let composite_params = GpuDofCompositeParams::from_settings(size, &settings);

    let groups_x = size.x.div_ceil(DOF_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(DOF_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();

    // Pass 1: circle-of-confusion prepass.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism dof coc"),
            timestamp_writes: None,
        });
        pass.set_pipeline(coc_pipeline);
        pass.set_bind_group(0, groups.coc(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&coc_params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Pass 2: near/far separated disk-bokeh gather.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism dof gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(gather_pipeline);
        pass.set_bind_group(0, groups.gather(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&gather_params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Pass 3: composite the blurred image over the sharp one by the CoC blend.
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism dof composite"),
            timestamp_writes: None,
        });
        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, groups.composite(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&composite_params));
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Copy the composited result back over scene_color so the downstream post
    // chain (which samples scene_color, not dof_out) reads the defocused image.
    // The composite cannot write scene_color in place: `dof_gather` reads
    // neighbouring scene_color texels across each disc, so the chain ping-pongs
    // into a dedicated target and then blits back. Same extent + same format
    // (SCENE_COLOR_FORMAT), single mip/layer.
    encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
            texture: dof.dof_out_texture(),
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
