//! `PrepareBindGroups` system building the gamut-map group-0 bind group.
//!
//! Mirrors [`super::super::color_grade::bind_groups`]: one group-0 bind group
//! per view binding the pre-exposed scene colour (read) and the compressed
//! output (storage write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::GamutMapPipeline;
use super::resources::ViewGamutMap;

/// The group-0 bind group a single view's gamut-map pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewGamutMapBindGroup {
    /// group 0 for `gamut_map_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewGamutMapBindGroup {
    /// group-0 bind group for the `gamut_map_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewGamutMapBindGroup`] for every view
/// whose visibility buffer and gamut-map output texture are both resident.
pub(crate) fn prepare_gamut_map_bind_groups(
    mut commands: Commands,
    pipeline: Res<GamutMapPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewGamutMap)>,
) {
    for (entity, visibility, gamut_map) in &views {
        // Read the pre-exposed scene colour (0), write the compressed output (1).
        let group = device.create_bind_group(
            "prism gamut map",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                gamut_map.gamut_map_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewGamutMapBindGroup { group });
    }
}
