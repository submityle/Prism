//! `PrepareBindGroups` system building the glossy-specular ReSTIR **spatial**
//! reuse pass's group-0 bind group per view.
//!
//! The spatial dispatch reads four group-0 bindings (mirroring the frozen
//! `spec_gi_spatial.wesl` contract): this frame's post-temporal reservoir table
//! read-only (0 — the very buffer the reuse pass just wrote as its `out`
//! target, now consumed as a stable snapshot behind the render-graph barrier
//! the dispatch ordering inserts), the two SSR-rebuilt G-buffer reads
//! (reverse-Z `scene_depth` 1, packed `normal_roughness` 2) and the write-only
//! resolved specular + confidence target (3 — the same target the reuse pass
//! resolved into, overwritten here with the spatially pooled estimate).
//!
//! Unlike [`super::bind_groups`] (whose reuse kernel reads its config from a
//! bound uniform), the spatial kernel carries its config in an immediate block,
//! so this system binds no uniform buffer — it only wires the reservoir
//! snapshot, the two reads and the resolved write.
//!
//! Like the reuse group, the bind group is rebuilt every frame because
//! [`ViewSpecGiReuse`]'s ping-pong `dst` selection flips each frame; it is
//! present exactly when both the view's SSR textures and its resident reuse
//! resources are live (the subsystem's gate guarantees the two appear and
//! disappear together), and only while spatial reuse is enabled.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::runtime::PrismShadingSettings;
use super::super::ssr::ViewSsrTextures;
use super::resources::ViewSpecGiReuse;
use super::spatial_pipeline::SpecGiSpatialPipeline;

/// The spatial dispatch's group-0 bind group for a single view. Present only
/// when both the view's SSR textures and its resident reuse resources are live
/// and spatial reuse is enabled.
#[derive(Component)]
pub(crate) struct ViewSpecGiSpatialBindGroup {
    /// group 0 for `spec_gi_spatial`: this frame's post-temporal reservoirs
    /// read-only (0), SSR-rebuilt depth (1) + normal/roughness (2) and the
    /// write-only resolved target (3). Rebuilt every frame to follow the
    /// reservoir ping-pong flip.
    group: BindGroup,
}

impl ViewSpecGiSpatialBindGroup {
    /// group-0 bind group the dispatch node records against.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewSpecGiSpatialBindGroup`] for every
/// view that has both resident SSR textures and resident reuse resources, while
/// spatial reuse is enabled.
///
/// Must run after `prepare_spec_gi_reuse_bind_groups` so the bound `dst`
/// reservoir buffer agrees with this frame's ping-pong flip (the reuse pass
/// writes it, then this pass reads it back read-only). Clears any stale group
/// when spatial reuse is disabled so a later re-enable rebuilds cleanly.
pub(crate) fn prepare_spec_gi_spatial_bind_groups(
    mut commands: Commands,
    pipeline: Res<SpecGiSpatialPipeline>,
    device: Res<RenderDevice>,
    settings: Res<PrismShadingSettings>,
    views: Query<(Entity, &ViewSsrTextures, &ViewSpecGiReuse)>,
) {
    for (entity, ssr, spec_gi) in &views {
        if !settings.enable_spec_gi_spatial {
            commands
                .entity(entity)
                .remove::<ViewSpecGiSpatialBindGroup>();
            continue;
        }

        // Sequential group 0: this frame's post-temporal reservoirs read-only
        // (0 — the reuse pass's `dst`, now a stable snapshot), SSR reverse-Z
        // depth (1), packed normal/roughness (2), write-only resolved (3).
        let group = device.create_bind_group(
            "prism spec_gi spatial",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                spec_gi.dst_buffer().as_entire_binding(),
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                spec_gi.resolved_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewSpecGiSpatialBindGroup { group });
    }
}
