//! Per-view group-0 bind group for the vignette pass.
//!
//! Mirrors [`super::super::dof::bind_groups`] scaled to a single pass: one
//! `PrepareBindGroups` system builds the group a view needs from its resident
//! textures, present only when the visibility buffer (pre-exposed scene colour)
//! and the vignette output texture are both live. The group binds the shader's
//! `sequential` `{0,1}`: the scene colour read (`0`) and the output storage
//! write (`1`).

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::VignettePipeline;
use super::resources::ViewVignette;

/// The group-0 bind group a single view's vignette pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewVignetteBindGroup {
    /// group 0 for `vignette_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewVignetteBindGroup {
    /// group-0 bind group for the `vignette_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewVignetteBindGroup`] for every view
/// whose visibility buffer and vignette output texture are both resident.
pub(crate) fn prepare_vignette_bind_groups(
    mut commands: Commands,
    pipeline: Res<VignettePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewVignette)>,
) {
    for (entity, visibility, vignette) in &views {
        // Read the pre-exposed scene colour (0), write the darkened output (1).
        let group = device.create_bind_group(
            "prism vignette",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                vignette.vignette_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewVignetteBindGroup { group });
    }
}
