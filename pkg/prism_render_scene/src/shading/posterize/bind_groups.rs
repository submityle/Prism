//! `PrepareBindGroups` system building the posterize group-0 bind group.
//!
//! Mirrors [`super::super::color_grade::bind_groups`]: one group-0 bind group
//! per view binding the pre-exposed scene colour (read) and the posterized
//! output (storage write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::PosterizePipeline;
use super::resources::ViewPosterize;

/// The group-0 bind group a single view's posterize pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewPosterizeBindGroup {
    /// group 0 for `posterize_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewPosterizeBindGroup {
    /// group-0 bind group for the `posterize_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewPosterizeBindGroup`] for every view
/// whose visibility buffer and posterize output texture are both resident.
pub(crate) fn prepare_posterize_bind_groups(
    mut commands: Commands,
    pipeline: Res<PosterizePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewPosterize)>,
) {
    for (entity, visibility, posterize) in &views {
        // Read the pre-exposed scene colour (0), write the posterized output (1).
        let group = device.create_bind_group(
            "prism posterize",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                posterize.posterize_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewPosterizeBindGroup { group });
    }
}
