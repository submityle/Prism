//! `PrepareBindGroups` system building the tone-map group-0 bind group.
//!
//! Mirrors [`super::super::gamut_map::bind_groups`]: one group-0 bind group per
//! view binding the pre-exposed scene colour (read) and the tone-mapped output
//! (storage write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::TonemapPipeline;
use super::resources::ViewTonemap;

/// The group-0 bind group a single view's tone-map pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewTonemapBindGroup {
    /// group 0 for `tonemap_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewTonemapBindGroup {
    /// group-0 bind group for the `tonemap_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewTonemapBindGroup`] for every view
/// whose visibility buffer and tone-map output texture are both resident.
pub(crate) fn prepare_tonemap_bind_groups(
    mut commands: Commands,
    pipeline: Res<TonemapPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewTonemap)>,
) {
    for (entity, visibility, tonemap) in &views {
        // Read the pre-exposed scene colour (0), write the tone-mapped output (1).
        let group = device.create_bind_group(
            "prism tonemap",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                tonemap.tonemap_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewTonemapBindGroup { group });
    }
}
