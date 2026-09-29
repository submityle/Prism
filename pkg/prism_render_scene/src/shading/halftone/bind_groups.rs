//! `PrepareBindGroups` system building the halftone group-0 bind group.
//!
//! Mirrors [`super::super::color_grade::bind_groups`]: one group-0 bind group
//! per view binding the resolved scene colour (read) and the filtered output
//! (storage write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::HalftonePipeline;
use super::resources::ViewHalftone;

/// The group-0 bind group a single view's halftone pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewHalftoneBindGroup {
    /// group 0 for `halftone_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewHalftoneBindGroup {
    /// group-0 bind group for the `halftone_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewHalftoneBindGroup`] for every view
/// whose visibility buffer and halftone output texture are both resident.
pub(crate) fn prepare_halftone_bind_groups(
    mut commands: Commands,
    pipeline: Res<HalftonePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewHalftone)>,
) {
    for (entity, visibility, halftone) in &views {
        // Read the resolved scene colour (0), write the filtered output (1).
        let group = device.create_bind_group(
            "prism halftone",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                halftone.halftone_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewHalftoneBindGroup { group });
    }
}
