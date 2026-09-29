//! `PrepareBindGroups` system building the ordered-dither group-0 bind group.
//!
//! Mirrors [`super::super::gamut_map::bind_groups`]: one group-0 bind group per
//! view binding the pre-exposed scene colour (read) and the dithered output
//! (storage write), present only when the backing textures are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::OrderedDitherPipeline;
use super::resources::ViewOrderedDither;

/// The group-0 bind group a single view's ordered-dither pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewOrderedDitherBindGroup {
    /// group 0 for `ordered_dither_main`: scene-colour read + output storage
    /// write.
    group: BindGroup,
}

impl ViewOrderedDitherBindGroup {
    /// group-0 bind group for the `ordered_dither_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewOrderedDitherBindGroup`] for every
/// view whose visibility buffer and ordered-dither output texture are both
/// resident.
pub(crate) fn prepare_ordered_dither_bind_groups(
    mut commands: Commands,
    pipeline: Res<OrderedDitherPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewOrderedDither)>,
) {
    for (entity, visibility, ordered_dither) in &views {
        // Read the pre-exposed scene colour (0), write the dithered output (1).
        let group = device.create_bind_group(
            "prism ordered dither",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                ordered_dither.ordered_dither_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewOrderedDitherBindGroup { group });
    }
}
