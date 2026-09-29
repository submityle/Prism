//! Per-view group-0 bind group for the film-grain pass.
//!
//! Mirrors [`super::super::vignette::bind_groups`]: one `PrepareBindGroups`
//! system builds the group a view needs from its resident textures, present
//! only when the visibility buffer (pre-exposed scene colour) and the film-grain
//! output texture are both live. The group binds the shader's `sequential`
//! `{0,1}`: the scene colour read (`0`) and the output storage write (`1`).

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::FilmGrainPipeline;
use super::resources::ViewFilmGrain;

/// The group-0 bind group a single view's film-grain pass records against.
/// Present only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewFilmGrainBindGroup {
    /// group 0 for `film_grain_main`: scene-colour read + output storage write.
    group: BindGroup,
}

impl ViewFilmGrainBindGroup {
    /// group-0 bind group for the `film_grain_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewFilmGrainBindGroup`] for every view
/// whose visibility buffer and film-grain output texture are both resident.
pub(crate) fn prepare_film_grain_bind_groups(
    mut commands: Commands,
    pipeline: Res<FilmGrainPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewFilmGrain)>,
) {
    for (entity, visibility, film_grain) in &views {
        // Read the pre-exposed scene colour (0), write the grained output (1).
        let group = device.create_bind_group(
            "prism film grain",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                film_grain.film_grain_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewFilmGrainBindGroup { group });
    }
}
