//! `PrepareBindGroups` systems building the two world-space GI group-0 bind
//! groups per view.
//!
//! Mirrors [`super::super::ssgi::trace`]'s bind-group construction: the
//! reverse-Z scene depth and packed `normal_roughness` come from the SSR
//! prepass ([`ViewSsrTextures`]), the pre-exposed scene colour from the
//! visibility buffer ([`ViewVisibilityBuffer`]), and the probe storage buffer
//! plus GI export target from this subsystem's [`ViewWorldSpaceGi`]. Both
//! groups are present only when all backing resources are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::WorldSpaceGiPipeline;
use super::resources::ViewWorldSpaceGi;

/// The two group-0 bind groups a single view's world-space GI passes record
/// against. Present only when the backing prepass, visibility and GI
/// resources are all resident.
#[derive(Component)]
pub(crate) struct ViewWorldSpaceGiBindGroups {
    /// group 0 for `probe_update_main`: depth + normal + colour reads and the
    /// probe storage buffer (read-write).
    probe_update: BindGroup,
    /// group 0 for `resolve_main`: depth + normal reads, the probe storage
    /// buffer (read-only) and the GI export storage write.
    resolve: BindGroup,
}

impl ViewWorldSpaceGiBindGroups {
    /// group-0 bind group for the `probe_update_main` dispatch.
    pub(crate) fn probe_update_group(&self) -> &BindGroup {
        &self.probe_update
    }

    /// group-0 bind group for the `resolve_main` dispatch.
    pub(crate) fn resolve_group(&self) -> &BindGroup {
        &self.resolve
    }
}

/// `PrepareBindGroups` system building [`ViewWorldSpaceGiBindGroups`] for every
/// view whose SSR prepass, visibility buffer and GI resources are all resident.
pub(crate) fn prepare_world_space_gi_bind_groups(
    mut commands: Commands,
    pipeline: Res<WorldSpaceGiPipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewSsrTextures,
        &ViewVisibilityBuffer,
        &ViewWorldSpaceGi,
    )>,
) {
    for (entity, ssr, visibility, gi) in &views {
        // probe_update group: depth (0), normal_roughness (1), pre-exposed
        // scene colour (2), probe storage buffer read-write (3).
        let probe_update = device.create_bind_group(
            "prism world-space GI probe update",
            pipeline.probe_update_layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                visibility.scene_color_view(),
                gi.probe_buffer().as_entire_binding(),
            )),
        );

        // resolve group: depth (0), normal_roughness (1), probe storage buffer
        // read-only (2), GI export storage write (3).
        let resolve = device.create_bind_group(
            "prism world-space GI resolve",
            pipeline.resolve_layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                gi.probe_buffer().as_entire_binding(),
                gi.gi_out_view(),
            )),
        );

        commands.entity(entity).insert(ViewWorldSpaceGiBindGroups {
            probe_update,
            resolve,
        });
    }
}
