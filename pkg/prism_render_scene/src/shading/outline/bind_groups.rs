//! Per-view group-0 bind group for the outline pass.
//!
//! Mirrors [`super::super::color_grade::bind_groups`]: one `PrepareBindGroups`
//! system builds the bind group a view needs from its resident textures,
//! present only when the visibility buffer (pre-exposed scene colour), the SSR
//! geometry prepass (device depth + view-space normal) and the outline output
//! texture are all live. The group binds the shader's `sequential` `{0, 1, 2,
//! 3}`: scene colour read, output storage write, depth read, normal read.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::OutlinePipeline;
use super::resources::ViewOutline;

/// The group-0 bind group a single view's outline pass records against. Present
/// only when the backing textures are resident.
#[derive(Component)]
pub(crate) struct ViewOutlineBindGroup {
    /// group 0 for `outline_main`: scene-colour + depth + normal reads, output
    /// storage write.
    group: BindGroup,
}

impl ViewOutlineBindGroup {
    /// group-0 bind group for the `outline_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewOutlineBindGroup`] for every view
/// whose visibility buffer, SSR textures and outline output texture are all
/// resident.
pub(crate) fn prepare_outline_bind_groups(
    mut commands: Commands,
    pipeline: Res<OutlinePipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        &ViewSsrTextures,
        &ViewOutline,
    )>,
) {
    for (entity, visibility, ssr, outline) in &views {
        // Read the pre-exposed scene colour (0), write the outlined output (1),
        // read the SSR device depth (2) and the SSR view-space normal (3).
        let group = device.create_bind_group(
            "prism outline",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                outline.outline_out_view(),
                ssr.scene_depth_sampled(),
                ssr.view_normal_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewOutlineBindGroup { group });
    }
}
