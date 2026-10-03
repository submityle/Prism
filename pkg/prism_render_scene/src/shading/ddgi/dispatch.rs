//! `Core3d` scheduling system recording the DDGI sample dispatch for every
//! view.
//!
//! Mirrors [`super::super::world_space_gi::dispatch`]: gate on the enable,
//! resolve the view, recover the clip->world transform and near plane from the
//! [`ExtractedView`], then run `sample_main` (one workgroup per 8x8 *pixel*
//! block). The entry point bounds-checks every invocation.
//!
//! Unlike the `scene_color` post-processors, the GI result is an *export*
//! buffer (`gi_out`) a downstream composite samples, so there is no copy back
//! over `scene_color`.

use bevy_ecs::prelude::*;
use bevy_math::Vec4;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::{GpuDdgiSampleParams, GpuDdgiUpdateParams, DDGI_WORKGROUP_SIZE};
use super::bind_groups::ViewDdgiBindGroups;
use super::pipeline::DdgiPipeline;
use super::resources::ViewDdgi;
use super::settings::PrismDdgiSettings;

/// `Core3d` scheduling system recording the `sample_main` dispatch for every
/// view whose DDGI resources and bind group are resident.
pub(crate) fn ddgi_sample_pass(
    settings: Res<PrismDdgiSettings>,
    view: ViewQuery<(&ViewDdgi, &ViewDdgiBindGroups, &ExtractedView)>,
    pipeline: Res<DdgiPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (gi, groups, extracted) = view.into_inner();

    let Some(sample_pipeline) = cache.get_compute_pipeline(pipeline.sample()) else {
        return;
    };

    let size = gi.size();
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Clip -> world reconstruction transform: compose the inverse projection
    // (clip -> view) with the camera's world placement (view -> world). The
    // sample kernel multiplies clip-space `(ndc, depth, 1)` by this to recover
    // the shading point's world position.
    let view_from_clip = extracted.clip_from_view.inverse();
    let world_from_view = extracted.world_from_view.to_matrix();
    let clip_to_world = world_from_view * view_from_clip;

    // Reverse-Z: device depth 1.0 is the near plane, so inverse-projecting
    // clip (0, 0, 1, 1) yields a view-space point at `-near` along `-Z`.
    let near_view = view_from_clip * Vec4::new(0.0, 0.0, 1.0, 1.0);
    let near = if near_view.w.abs() > f32::EPSILON {
        (near_view.z / near_view.w).abs().max(1.0e-3)
    } else {
        1.0e-3
    };

    let params = GpuDdgiSampleParams::new(
        clip_to_world,
        [size.x as f32, size.y as f32],
        near,
        settings.intensity,
    );

    let groups_x = size.x.div_ceil(DDGI_WORKGROUP_SIZE);
    let groups_y = size.y.div_ceil(DDGI_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism DDGI sample"),
        timestamp_writes: None,
    });
    pass.set_pipeline(sample_pipeline);
    pass.set_bind_group(0, groups.sample_group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(groups_x, groups_y, 1);
}

/// `Core3d` scheduling system recording the `probe_update_main` dispatch for
/// every view whose DDGI resources and bind group are resident.
///
/// One workgroup per probe (64 cooperative threads): trace 64 rays against the
/// screen-space G-buffer, temporally blend the octahedral irradiance + depth
/// moments against the per-probe history, relocate / classify the probe and
/// write both octahedral atlases. Scheduled *before* `ddgi_sample_pass` so the
/// atlases the sample pass reads are populated this frame.
pub(crate) fn ddgi_probe_update_pass(
    settings: Res<PrismDdgiSettings>,
    view: ViewQuery<(&ViewDdgi, &ViewDdgiBindGroups, &ExtractedView)>,
    pipeline: Res<DdgiPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (gi, groups, extracted) = view.into_inner();

    let probe_count = gi.probe_count();
    if probe_count == 0 {
        return;
    }

    let Some(update_pipeline) = cache.get_compute_pipeline(pipeline.probe_update()) else {
        return;
    };

    // Reverse-Z clip<->world transforms. `clip_to_world` reconstructs the world
    // position of a sampled G-buffer texel; `world_to_clip` projects a traced
    // ray endpoint back to screen space for the depth / colour fetch.
    let view_from_clip = extracted.clip_from_view.inverse();
    let world_from_view = extracted.world_from_view.to_matrix();
    let clip_to_world = world_from_view * view_from_clip;
    let world_to_clip = clip_to_world.inverse();

    // Rays march up to two probe spacings before giving up — far enough to find
    // the neighbouring surface, short enough to stay in the screen-space cache.
    let max_ray_distance = settings.spacing.length() * 2.0;

    let params = GpuDdgiUpdateParams::new(
        world_to_clip,
        clip_to_world,
        settings.backface_threshold,
        settings.min_frontface_distance,
        settings.activity_distance,
        settings.relocation_limit,
        max_ray_distance,
    );

    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism DDGI probe update"),
        timestamp_writes: None,
    });
    pass.set_pipeline(update_pipeline);
    pass.set_bind_group(0, groups.probe_update_group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(probe_count, 1, 1);
}
