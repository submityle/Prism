//! `PrepareBindGroups` system building the light-routing group-0 bind group per
//! view.
//!
//! Mirrors [`super::super::world_space_gi::bind_groups`]: the read-only routing
//! records and the two read-write export buffers all come from this
//! subsystem's [`ViewLightRouting`]. The group is present only when that
//! resource is resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::pipeline::LightRoutingPipeline;
use super::resources::ViewLightRouting;

/// The group-0 bind group a single view's light-routing cull records against.
/// Present only when the backing [`ViewLightRouting`] buffers are resident.
#[derive(Component)]
pub(crate) struct ViewLightRoutingBindGroup {
    /// group 0 for `light_routing_cull_main`: routing records read-only,
    /// visibility mask + per-layer masks read-write.
    group: BindGroup,
}

impl ViewLightRoutingBindGroup {
    /// group-0 bind group for the `light_routing_cull_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewLightRoutingBindGroup`] for every
/// view whose light-routing buffers are resident.
pub(crate) fn prepare_light_routing_bind_groups(
    mut commands: Commands,
    pipeline: Res<LightRoutingPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewLightRouting)>,
) {
    for (entity, routing) in &views {
        // routing records (0, read-only), visibility mask (1, read-write),
        // per-layer masks (2, read-write).
        let group = device.create_bind_group(
            "prism light routing",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                routing.routing_buffer().as_entire_binding(),
                routing.visible_buffer().as_entire_binding(),
                routing.layer_buffer().as_entire_binding(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewLightRoutingBindGroup { group });
    }
}
