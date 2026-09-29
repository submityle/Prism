//! `PrepareBindGroups` system building the Kuwahara group-0 bind group.
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
use super::pipeline::KuwaharaPipeline;
use super::resources::ViewKuwahara;

/// The group-0 bind group a single view's Kuwahara pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewKuwaharaBindGroup {
    /// group 0 for `kuwahara_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewKuwaharaBindGroup {
    /// group-0 bind group for the `kuwahara_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewKuwaharaBindGroup`] for every view
/// whose visibility buffer and Kuwahara output texture are both resident.
pub(crate) fn prepare_kuwahara_bind_groups(
    mut commands: Commands,
    pipeline: Res<KuwaharaPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewKuwahara)>,
) {
    for (entity, visibility, kuwahara) in &views {
        // Read the resolved scene colour (0), write the filtered output (1).
        let group = device.create_bind_group(
            "prism kuwahara",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                kuwahara.kuwahara_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewKuwaharaBindGroup { group });
    }
}
