//! `Core3d` node recording the virtual-shadow-map receiver-generation dispatch
//! for every view.
//!
//! Mirrors [`super::super::taa::dispatch`]. Runs after the SSR geometry prepass
//! (which produces the `R32Float` `scene_depth` this pass unprojects) and before
//! the main pass, filling the per-view receiver `storage` buffer the downstream
//! page-request pass consumes. Dispatches one workgroup per 8x8 pixel tile; the
//! shader bounds-checks every invocation. Gated on
//! [`super::super::runtime::PrismShadingSettings::enable_virtual_shadow`]; the
//! per-view components only exist on views that ran the prepare/bind-group
//! steps, so a disabled or light-less frame records nothing.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::abi::VSM_RECEIVER_GEN_WORKGROUP_SIZE;
use super::bind_groups::ViewVsmReceiverGenBindGroup;
use super::pipeline::VsmReceiverGenPipeline;
use super::resources::ViewVsmReceivers;

/// `Core3d` node recording the receiver-generation dispatch for every view.
///
/// Reads the SSR prepass depth and writes one
/// [`super::abi::GpuVsmReceiver`] per pixel into the persistent receiver
/// `storage` buffer.
pub(crate) fn vsm_receiver_gen_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewVsmReceivers, &ViewVsmReceiverGenBindGroup)>,
    pipeline: Res<VsmReceiverGenPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_virtual_shadow {
        return;
    }
    let (receivers, group) = view.into_inner();

    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    let size = receivers.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let workgroups_x = size.x.div_ceil(VSM_RECEIVER_GEN_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(VSM_RECEIVER_GEN_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism VSM receiver-gen"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, group.group(), &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
