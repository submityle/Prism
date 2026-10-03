//! `PrepareBindGroups` system building the DDGI sample pass group-0 bind group
//! per view.
//!
//! Mirrors [`super::super::world_space_gi::bind_groups`]: the reverse-Z scene
//! depth and packed `normal_roughness` come from the SSR prepass
//! ([`ViewSsrTextures`]); the lattice uniform, per-probe [`ProbeMeta`] storage
//! buffer, the two octahedral atlases and the GI export target come from this
//! subsystem's [`ViewDdgi`]. The group is present only when both backing
//! resources are resident.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::resources::ViewVisibilityBuffer;
use super::super::ssr::ViewSsrTextures;
use super::pipeline::DdgiPipeline;
use super::resources::ViewDdgi;

/// The group-0 bind group a single view's DDGI sample pass records against.
/// Present only when the SSR prepass and DDGI resources are both resident.
#[derive(Component)]
pub(crate) struct ViewDdgiBindGroups {
    /// group 0 for `sample_main`: depth + normal reads, the lattice uniform,
    /// the read-only [`ProbeMeta`] storage buffer, the two octahedral atlases
    /// and the GI export storage write.
    sample: BindGroup,
    /// group 0 for `probe_update_main`: depth + normal + scene-colour reads,
    /// the lattice uniform, the [`ProbeMeta`] + irradiance / depth history
    /// `read_write` storage buffers and the two octahedral atlas storage writes.
    probe_update: BindGroup,
}

impl ViewDdgiBindGroups {
    /// group-0 bind group for the `sample_main` dispatch.
    pub(crate) fn sample_group(&self) -> &BindGroup {
        &self.sample
    }

    /// group-0 bind group for the `probe_update_main` dispatch.
    pub(crate) fn probe_update_group(&self) -> &BindGroup {
        &self.probe_update
    }
}

/// `PrepareBindGroups` system building [`ViewDdgiBindGroups`] for every view
/// whose SSR prepass and DDGI resources are both resident.
pub(crate) fn prepare_ddgi_bind_groups(
    mut commands: Commands,
    pipeline: Res<DdgiPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewSsrTextures, &ViewVisibilityBuffer, &ViewDdgi)>,
) {
    for (entity, ssr, vis, gi) in &views {
        // sample group: depth (0), normal_roughness (1), lattice uniform (2),
        // probe-meta storage read-only (3), octahedral irradiance (4) + depth
        // (5) atlases, GI export storage write (6).
        let sample = device.create_bind_group(
            "prism DDGI sample",
            pipeline.sample_layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                gi.volume_uniform().as_entire_binding(),
                gi.probe_meta().as_entire_binding(),
                gi.irradiance_atlas_view(),
                gi.depth_atlas_view(),
                gi.gi_out_view(),
            )),
        );

        // probe-update group: depth (0), normal_roughness (1), scene colour
        // (2), lattice uniform (3), probe-meta (4) + irradiance history (5) +
        // depth history (6) `read_write` storage buffers, octahedral irradiance
        // (7) + depth (8) atlas storage writes.
        let probe_update = device.create_bind_group(
            "prism DDGI probe update",
            pipeline.probe_update_layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                vis.scene_color_view(),
                gi.volume_uniform().as_entire_binding(),
                gi.probe_meta().as_entire_binding(),
                gi.irradiance_history().as_entire_binding(),
                gi.depth_history().as_entire_binding(),
                gi.irradiance_atlas_view(),
                gi.depth_atlas_view(),
            )),
        );

        commands.entity(entity).insert(ViewDdgiBindGroups {
            sample,
            probe_update,
        });
    }
}
