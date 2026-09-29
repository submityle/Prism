//! `PrepareBindGroups` assembly of the receiver-generation pass's group-0 bind
//! group.
//!
//! Mirrors [`super::super::taa::bind_groups`]: for every view carrying both an
//! SSR depth prepass ([`super::super::ssr::ViewSsrTextures`]) and the prepared
//! [`super::resources::ViewVsmReceivers`] buffers, it assembles the three-entry
//! group-0 bind group in the byte-identical order the pipeline layout and
//! `vsm_receiver_gen.wesl` declare:
//!
//! * binding 0 -- the camera device depth (the SSR prepass's `R32Float`
//!   reverse-Z `scene_depth`, `textureLoad`ed),
//! * binding 1 -- the write-only per-pixel receiver `storage` array, and
//! * binding 2 -- the per-frame [`super::abi::GpuVsmReceiverGenParams`] uniform.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::ssr::ViewSsrTextures;
use super::pipeline::VsmReceiverGenPipeline;
use super::resources::ViewVsmReceivers;

/// The receiver-generation pass's group-0 bind group for a single view. Present
/// only when both the SSR depth prepass and the receiver buffers are resident.
#[derive(Component)]
pub(crate) struct ViewVsmReceiverGenBindGroup {
    group: BindGroup,
}

impl ViewVsmReceiverGenBindGroup {
    /// The prepared group-0 bind group bound by the receiver-generation
    /// dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewVsmReceiverGenBindGroup`] for every
/// view with a resident [`ViewSsrTextures`] depth prepass and prepared
/// [`ViewVsmReceivers`] buffers.
pub(crate) fn prepare_vsm_receiver_gen_bind_groups(
    mut commands: Commands,
    pipeline: Res<VsmReceiverGenPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewSsrTextures, &ViewVsmReceivers)>,
) {
    for (entity, ssr, receivers) in &views {
        let group = device.create_bind_group(
            "prism VSM receiver-gen",
            pipeline.layout(),
            // Order mirrors `vsm_receiver_gen.wesl`: depth(0), receivers(1),
            // params(2).
            &BindGroupEntries::sequential((
                ssr.scene_depth_view(),
                receivers.receivers_buffer().as_entire_binding(),
                receivers.params_buffer().as_entire_binding(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewVsmReceiverGenBindGroup { group });
    }
}
