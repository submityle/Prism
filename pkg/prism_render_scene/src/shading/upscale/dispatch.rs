//! `Core3d` scheduling systems recording the two temporal-upscale dispatches.
//!
//! Mirrors [`super::super::taa::dispatch`] but chains the two passes the golden
//! pipeline defines, both dispatched over the *display* grid (one workgroup per
//! 8x8 display tile; every shader invocation bounds-checks itself):
//!
//! 1. `upscale_reconstruct` resolves this frame's low-resolution `render_color`
//!    onto the display grid, reprojects the display-resolution history through
//!    the motion vectors, clips it to the current neighbourhood's `YCoCg`
//!    variance box, locks thin features and blends by the temporal
//!    accumulation, writing the resolved (pre-sharpen) colour, the new
//!    accumulation/lock metadata and this frame's depth into the write slots of
//!    the three ping-pong pairs.
//! 2. `upscale_rcas` reads that resolved colour and runs FSR's RCAS sharpen
//!    over its five-tap cross, writing the finished display image into
//!    `upscale_out`.
//!
//! Both passes share one command encoder, so wgpu inserts the storage-write
//! barrier between them: RCAS reads the reconstruction's `color_out` slot only
//! after the reconstruction pass completes. There is no copy back into
//! `scene_color`: the display extent may differ from the render (draw) extent,
//! so blitting `upscale_out` (which carries `COPY_SRC`) to the display target is
//! the `render_scale` graph wiring's job, landing with that integration.
//!
//! Gated on the [`UpscaleSettings`] resource being present (its presence is the
//! enable); a view without a resolved [`ViewUpscale`] / [`ViewUpscaleBindGroups`]
//! (no resident visibility buffer, SSR depth, or a multi-sample target) simply
//! does not match the query and is skipped.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};
use prism_render_architecture::history::InvalidationMask;

use super::abi::{GpuUpscaleRcasParams, GpuUpscaleReconstructParams, UPSCALE_WORKGROUP_SIZE};
use super::bind_groups::ViewUpscaleBindGroups;
use super::pipeline::UpscalePipeline;
use super::resources::ViewUpscale;
use super::settings::UpscaleSettings;

/// `Core3d` scheduling system recording the reconstruction + RCAS dispatch for
/// every view with resolved temporal-upscale state.
///
/// Builds this frame's [`GpuUpscaleReconstructParams`] from the render/display
/// extents, the settings' [`TemporalUpscaleSettings`] projection, the golden
/// [`UpscaleConfig`] tunables and the ping-pong history-valid flag, and the
/// [`GpuUpscaleRcasParams`] from the display extent and the settings' sharpen
/// tunables, then records the two `@workgroup_size(8,8,1)` passes over the
/// display grid.
///
/// [`TemporalUpscaleSettings`]: prism_render_architecture::temporal_upscale::TemporalUpscaleSettings
/// [`UpscaleConfig`]: prism_render_shading::upscale::UpscaleConfig
pub(crate) fn upscale_pass(
    settings: Option<Res<UpscaleSettings>>,
    view: ViewQuery<(&ViewUpscale, &ViewUpscaleBindGroups)>,
    pipeline: Res<UpscalePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    // The resource's presence is the enable: without it the feature is not
    // wired this run.
    let Some(settings) = settings else {
        return;
    };
    let (upscale, groups) = view.into_inner();

    // Both pipelines must have finished specializing before either pass runs.
    let Some(reconstruct) = cache.get_compute_pipeline(pipeline.reconstruct()) else {
        return;
    };
    let Some(rcas) = cache.get_compute_pipeline(pipeline.rcas()) else {
        return;
    };

    let display = upscale.display_size();
    if display.x == 0 || display.y == 0 {
        return;
    }

    // The reconstruction reprojects the history on the GPU through the
    // motion-vector buffer, so the params only carry the two extents, the
    // golden tunables and the history-valid flag. This self-contained slice
    // feeds an empty `invalidation_events` (no camera-cut event source is wired
    // yet); the graph integration routes the real event mask here.
    let reconstruct_params = GpuUpscaleReconstructParams::new(
        upscale.render_extent(),
        upscale.display_extent(),
        &settings.to_upscale_settings(),
        &settings.config,
        InvalidationMask::default(),
        upscale.valid(),
    );
    let rcas_params = GpuUpscaleRcasParams::new(upscale.display_extent(), &settings.rcas_params());

    // Both passes cover the display grid.
    let workgroups_x = display.x.div_ceil(UPSCALE_WORKGROUP_SIZE);
    let workgroups_y = display.y.div_ceil(UPSCALE_WORKGROUP_SIZE);

    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism upscale reconstruct"),
                timestamp_writes: None,
            });
        pass.set_pipeline(reconstruct);
        pass.set_bind_group(0, groups.reconstruct(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&reconstruct_params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism upscale rcas"),
                timestamp_writes: None,
            });
        pass.set_pipeline(rcas);
        pass.set_bind_group(0, groups.rcas(), &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&rcas_params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
