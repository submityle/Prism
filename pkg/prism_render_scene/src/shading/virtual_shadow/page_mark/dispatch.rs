//! `Core3d` node recording the virtual-shadow-map page-request (`page-mark`)
//! dispatch for every view.
//!
//! Mirrors [`super::super::dispatch`] (receiver generation) and
//! [`super::super::super::taa::dispatch`] (immediate-block upload). Runs after
//! the receiver-generation pass (which fills the receiver `storage` buffer this
//! pass reads) and before the main pass, marking, for every visible receiver,
//! the clipmap pages its soft-shadow footprint touches in a camera-snapped
//! resident-window request bitmap.
//!
//! The request bitmap is cleared to zero each frame *before* the dispatch so the
//! `atomicOr` marking accumulates only this frame's requests; overlapping
//! receiver footprints collapse to one request per slot (the golden
//! de-duplication). The receiver buffer holds one entry per screen pixel, which
//! at 4K overruns the 65535 single-dimension workgroup limit, so the 64-wide
//! workgroups are laid out in a near-square 2D grid and the shader linearises
//! back to the flat receiver index. Gated on
//! [`super::super::super::runtime::PrismShadingSettings::enable_virtual_shadow`];
//! the per-view components only exist on views that ran the prepare/bind-group
//! steps, so a disabled or light-less frame records nothing.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::super::super::runtime::PrismShadingSettings;
use super::super::abi::VSM_PAGE_MARK_WORKGROUP_SIZE;
use super::bind_groups::ViewVsmPageMarkBindGroup;
use super::pipeline::VsmPageMarkPipeline;
use super::resources::ViewVsmPageRequests;

/// `Core3d` node recording the page-request dispatch for every view.
///
/// Clears the persistent request bitmap, then marks one page footprint per
/// receiver (read from the receiver-generation pass's `storage` buffer) into it
/// with `atomicOr`.
pub(crate) fn vsm_mark_pages_pass(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(&ViewVsmPageRequests, &ViewVsmPageMarkBindGroup)>,
    pipeline: Res<VsmPageMarkPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_virtual_shadow {
        return;
    }
    let (requests, group) = view.into_inner();

    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let receiver_count = requests.receiver_count();
    if receiver_count == 0 {
        return;
    }

    // Clear the persistent request bitmap to zero before marking so the
    // `atomicOr` in the shader accumulates only this frame's requests. This runs
    // on the encoder outside the compute pass, before it begins.
    ctx.command_encoder()
        .clear_buffer(requests.requests_buffer(), 0, None);

    // One 64-wide workgroup per receiver, laid out in a near-square 2D grid so
    // the flat receiver count never overruns the 65535 single-dimension limit;
    // the shader linearises (gid.y * num_workgroups.x * 64 + gid.x) back to the
    // receiver index and bounds-checks the padding tail.
    let total_workgroups = receiver_count.div_ceil(VSM_PAGE_MARK_WORKGROUP_SIZE);
    let workgroups_x = total_workgroups.min(65_535);
    let workgroups_y = total_workgroups.div_ceil(workgroups_x);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism VSM page-mark"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, group.group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(requests.params()));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
