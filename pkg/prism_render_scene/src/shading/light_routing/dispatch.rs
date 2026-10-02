//! `Core3d` scheduling system recording the light-routing cull dispatch for
//! every view.
//!
//! Mirrors [`super::super::world_space_gi::dispatch`]: gate on the enable,
//! resolve the view, then run one workgroup per [`LIGHT_ROUTING_WORKGROUP_SIZE`]
//! cluster words. The entry point bounds-checks every invocation against
//! `word_count`, so a view with no lights (`word_count == 0`) dispatches
//! nothing.
//!
//! The result is an *export*: the channel-gated visibility mask the shared
//! cluster cull / resolve read back, and the NPR per-layer masks the stylized
//! composite consumes. There is no copy back over any texture.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::abi::LIGHT_ROUTING_WORKGROUP_SIZE;
use super::bind_groups::ViewLightRoutingBindGroup;
use super::pipeline::LightRoutingPipeline;
use super::resources::ViewLightRouting;
use super::settings::PrismLightRoutingSettings;

/// `Core3d` scheduling system recording the `light_routing_cull_main` dispatch
/// for every view whose light-routing buffers and bind group are resident.
pub(crate) fn light_routing_cull_pass(
    settings: Res<PrismLightRoutingSettings>,
    view: ViewQuery<(&ViewLightRouting, &ViewLightRoutingBindGroup)>,
    pipeline: Res<LightRoutingPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enabled {
        return;
    }
    let (routing, group) = view.into_inner();

    // The pipeline must be resident before the pass runs.
    let Some(cull_pipeline) = cache.get_compute_pipeline(pipeline.pipeline()) else {
        return;
    };

    // No registered lights: nothing to cull. The bounds check would no-op every
    // invocation anyway, so skip the empty dispatch.
    if routing.word_count == 0 {
        return;
    }

    let params = settings.params();
    let groups_x = routing.word_count.div_ceil(LIGHT_ROUTING_WORKGROUP_SIZE);

    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism light routing cull"),
        timestamp_writes: None,
    });
    pass.set_pipeline(cull_pipeline);
    pass.set_bind_group(0, group.group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(groups_x, 1, 1);
}
