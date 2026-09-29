//! `PrepareBindGroups` system building the colour-grade group-0 bind group.
//!
//! Mirrors [`super::super::vignette::bind_groups`]: one group-0 bind group per
//! view binding the pre-exposed scene colour (read) and the graded output
//! (storage write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::ColorGradePipeline;
use super::resources::ViewColorGrade;

/// The group-0 bind group a single view's colour-grade pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewColorGradeBindGroup {
    /// group 0 for `color_grade_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewColorGradeBindGroup {
    /// group-0 bind group for the `color_grade_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewColorGradeBindGroup`] for every
/// view whose visibility buffer and grade output texture are both resident.
pub(crate) fn prepare_color_grade_bind_groups(
    mut commands: Commands,
    pipeline: Res<ColorGradePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewColorGrade)>,
) {
    for (entity, visibility, color_grade) in &views {
        // Read the pre-exposed scene colour (0), write the graded output (1).
        let group = device.create_bind_group(
            "prism color grade",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                color_grade.color_grade_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewColorGradeBindGroup { group });
    }
}
