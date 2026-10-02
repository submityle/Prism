//! `PrepareBindGroups` system building the world-space `ReSTIR` fill group-0
//! bind group per view.
//!
//! Mirrors [`super::super::world_space_gi::bind_groups`], scaled to the single
//! fill pass. Unlike the GI probe passes, the fill pass reads no prepass
//! textures: its only inputs are the subsystem's own resident reservoir
//! tables, so the bind group is built straight from [`ViewWorldRestir`] and is
//! present exactly when that table is resident (i.e. while the subsystem is
//! enabled). The ping-pong `src`/`dst` selection lives in [`ViewWorldRestir`],
//! so the group is rebuilt every frame to follow the flip.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::pipeline::WorldRestirPipeline;
use super::resources::ViewWorldRestir;

/// The group-0 bind group a single view's fill pass records against. Present
/// only while the view's resident reservoir table ([`ViewWorldRestir`]) is.
#[derive(Component)]
pub(crate) struct ViewWorldRestirBindGroups {
    /// group 0 for `fill_main`: previous reservoir table read-only (0) and the
    /// next reservoir table read-write (1).
    fill: BindGroup,
}

impl ViewWorldRestirBindGroups {
    /// group-0 bind group for the `fill_main` dispatch.
    pub(crate) fn fill_group(&self) -> &BindGroup {
        &self.fill
    }
}

/// `PrepareBindGroups` system building [`ViewWorldRestirBindGroups`] for every
/// view whose resident reservoir table is live.
///
/// The ping-pong flip advances in `prepare_world_restir_reservoirs`, so this
/// must run after it and rebuilds the group every frame to pick up the swapped
/// `src`/`dst` buffers.
pub(crate) fn prepare_world_restir_bind_groups(
    mut commands: Commands,
    pipeline: Res<WorldRestirPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewWorldRestir)>,
) {
    for (entity, restir) in &views {
        // fill group: previous reservoir table read-only (0), next reservoir
        // table read-write (1). Both follow this frame's ping-pong selection.
        let fill = device.create_bind_group(
            "prism world-space ReSTIR fill",
            pipeline.fill_layout(),
            &BindGroupEntries::sequential((
                restir.src_buffer().as_entire_binding(),
                restir.dst_buffer().as_entire_binding(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewWorldRestirBindGroups { fill });
    }
}
