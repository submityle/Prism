//! Per-view group-0 bind group for the chromatic-aberration pass.
//!
//! Mirrors [`super::super::dof::bind_groups`]: one `PrepareBindGroups` system
//! builds the single bind group a view needs from its resident scene colour and
//! aberration output, present only when the visibility buffer (the pre-exposed
//! scene colour) and the [`ViewChromaticAberration`] output texture are both
//! live. The group binds at the shader's sequential indices `{0,1,2}`: the
//! filterable scene colour, the shared linear-clamp sampler and the write-only
//! output.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::pipeline::ChromaticAberrationPipeline;
use super::resources::ViewChromaticAberration;

/// The group-0 bind group a single view's aberration pass records against.
/// Present only when both the scene colour and the output texture are resident.
#[derive(Component)]
pub(crate) struct ViewChromaticAberrationBindGroups {
    /// group 0: filterable scene colour + linear sampler + write-only output.
    group: BindGroup,
}

impl ViewChromaticAberrationBindGroups {
    /// The group-0 bind group for the aberration dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewChromaticAberrationBindGroups`] for
/// every view whose visibility buffer and aberration output texture are both
/// resident.
pub(crate) fn prepare_chromatic_aberration_bind_groups(
    mut commands: Commands,
    pipeline: Res<ChromaticAberrationPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewChromaticAberration)>,
) {
    for (entity, visibility, chromatic) in &views {
        // Order mirrors the layout / `chromatic_aberration_main`:
        // scene_color(0), linear_sampler(1), chromatic_out(2).
        let group = device.create_bind_group(
            "prism chromatic aberration",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                pipeline.linear_sampler(),
                chromatic.chromatic_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewChromaticAberrationBindGroups { group });
    }
}
