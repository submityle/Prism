//! `PrepareBindGroups` system building the cross-hatching group-0 bind group.
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
use super::pipeline::HatchingPipeline;
use super::resources::ViewHatching;

/// The group-0 bind group a single view's cross-hatching pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewHatchingBindGroup {
    /// group 0 for `hatching_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewHatchingBindGroup {
    /// group-0 bind group for the `hatching_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewHatchingBindGroup`] for every view
/// whose visibility buffer and hatching output texture are both resident.
pub(crate) fn prepare_hatching_bind_groups(
    mut commands: Commands,
    pipeline: Res<HatchingPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewHatching)>,
) {
    for (entity, visibility, hatching) in &views {
        // Read the resolved scene colour (0), write the filtered output (1).
        let group = device.create_bind_group(
            "prism hatching",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                hatching.hatching_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewHatchingBindGroup { group });
    }
}
