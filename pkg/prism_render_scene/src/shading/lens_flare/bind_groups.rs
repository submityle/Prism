//! `PrepareBindGroups` system building the lens-flare group-0 bind group.
//!
//! Mirrors [`super::super::color_grade::bind_groups`]: one group-0 bind group
//! per view binding the pre-exposed scene colour (read) and the flared output
//! (storage write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::LensFlarePipeline;
use super::resources::ViewLensFlare;

/// The group-0 bind group a single view's lens-flare pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewLensFlareBindGroup {
    /// group 0 for `lens_flare_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewLensFlareBindGroup {
    /// group-0 bind group for the `lens_flare_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewLensFlareBindGroup`] for every
/// view whose visibility buffer and flare output texture are both resident.
pub(crate) fn prepare_lens_flare_bind_groups(
    mut commands: Commands,
    pipeline: Res<LensFlarePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewLensFlare)>,
) {
    for (entity, visibility, lens_flare) in &views {
        // Read the pre-exposed scene colour (0), write the flared output (1).
        let group = device.create_bind_group(
            "prism lens flare",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                lens_flare.lens_flare_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewLensFlareBindGroup { group });
    }
}
