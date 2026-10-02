//! `PrepareBindGroups` system building the `CAS` group-0 bind group.
//!
//! Mirrors [`super::super::color_grade::bind_groups`]: one group-0 bind group
//! per view binding the scene colour (read) and the sharpened output (storage
//! write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::CasPipeline;
use super::resources::ViewCas;

/// The group-0 bind group a single view's `CAS` pass records against. Present
/// only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewCasBindGroup {
    /// group 0 for `cas_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewCasBindGroup {
    /// group-0 bind group for the `cas_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewCasBindGroup`] for every view whose
/// visibility buffer and sharpen output texture are both resident.
pub(crate) fn prepare_cas_bind_groups(
    mut commands: Commands,
    pipeline: Res<CasPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewCas)>,
) {
    for (entity, visibility, cas) in &views {
        // Read the scene colour (0), write the sharpened output (1).
        let group = device.create_bind_group(
            "prism cas",
            pipeline.layout(),
            &BindGroupEntries::sequential((visibility.scene_color_view(), cas.cas_out_view())),
        );

        commands.entity(entity).insert(ViewCasBindGroup { group });
    }
}
