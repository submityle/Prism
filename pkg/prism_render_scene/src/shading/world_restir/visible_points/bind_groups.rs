//! `PrepareBindGroups` system building the world-space `ReSTIR` visible-point
//! producer group-0 bind group per view.
//!
//! The producer reads the SSR prepass depth + packed `normal_roughness` and
//! writes the per-frame visible-point list, so the group binds the two SSR
//! prepass views (`textureLoad` sources) and the resident
//! [`ViewWorldRestirVisiblePoints`] buffer. It is present exactly when both the
//! SSR prepass ([`ViewSsrTextures`]) and the visible-point list are resident
//! (i.e. while the subsystem is enabled). The buffer is only reallocated on a
//! screen-size change, but the group is rebuilt every frame to track that.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::super::ssr::ViewSsrTextures;
use super::pipeline::WorldRestirVisiblePointsPipeline;
use super::resources::ViewWorldRestirVisiblePoints;

/// The group-0 bind group a single view's visible-point producer records
/// against. Present only while the view's SSR prepass and resident
/// visible-point list are.
#[derive(Component)]
pub(crate) struct ViewWorldRestirVisiblePointsBindGroup {
    /// group 0 for `visible_points_main`: SSR prepass depth (sampled, 0), SSR
    /// packed `normal_roughness` (sampled, 1) and the visible-point list
    /// (storage read-write, 2).
    group: BindGroup,
}

impl ViewWorldRestirVisiblePointsBindGroup {
    /// group-0 bind group for the `visible_points_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewWorldRestirVisiblePointsBindGroup`]
/// for every view whose SSR prepass and resident visible-point list are live.
pub(crate) fn prepare_world_restir_visible_points_bind_groups(
    mut commands: Commands,
    pipeline: Res<WorldRestirVisiblePointsPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewSsrTextures, &ViewWorldRestirVisiblePoints)>,
) {
    for (entity, ssr, points) in &views {
        let group = device.create_bind_group(
            "prism world-space ReSTIR visible points",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                points.buffer().as_entire_binding(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewWorldRestirVisiblePointsBindGroup { group });
    }
}
