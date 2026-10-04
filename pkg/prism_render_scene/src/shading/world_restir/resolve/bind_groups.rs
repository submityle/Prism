//! `PrepareBindGroups` system building the world-space `ReSTIR` resolve group-0
//! bind group per view.
//!
//! The resolve reads the SSR prepass depth + packed `normal_roughness` and the
//! resident finalised reservoir table ([`ViewWorldRestir::src_buffer`], the
//! fill pass output), and writes the direct-illumination export, so the group
//! binds the two SSR prepass views (`textureLoad` sources), the reservoir table
//! (storage read-only) and the [`ViewWorldRestirResolve`] export (storage
//! write). It is present exactly when the SSR prepass ([`ViewSsrTextures`]),
//! the reservoir table ([`ViewWorldRestir`]) and the resolve export are all
//! resident (i.e. while the subsystem is enabled). The export is only
//! reallocated on a viewport resize, but the group is rebuilt every frame to
//! track that and the reservoir ping-pong.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::super::ssr::ViewSsrTextures;
use super::super::resources::ViewWorldRestir;
use super::pipeline::WorldRestirResolvePipeline;
use super::resources::ViewWorldRestirResolve;

/// The group-0 bind group a single view's resolve dispatch records against.
/// Present only while the view's SSR prepass, reservoir table and resolve
/// export are.
#[derive(Component)]
pub(crate) struct ViewWorldRestirResolveBindGroup {
    /// group 0 for `resolve_main`: SSR prepass depth (sampled, 0), SSR packed
    /// `normal_roughness` (sampled, 1), the finalised reservoir table (storage
    /// read-only, 2) and the direct-illumination export (storage write, 3).
    group: BindGroup,
}

impl ViewWorldRestirResolveBindGroup {
    /// group-0 bind group for the `resolve_main` dispatch.
    pub(crate) fn group(&self) -> &BindGroup {
        &self.group
    }
}

/// `PrepareBindGroups` system building [`ViewWorldRestirResolveBindGroup`] for
/// every view whose SSR prepass, reservoir table and resolve export are live.
pub(crate) fn prepare_world_restir_resolve_bind_groups(
    mut commands: Commands,
    pipeline: Res<WorldRestirResolvePipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewSsrTextures,
        &ViewWorldRestir,
        &ViewWorldRestirResolve,
    )>,
) {
    for (entity, ssr, restir, resolve) in &views {
        let group = device.create_bind_group(
            "prism world-space ReSTIR resolve",
            pipeline.layout(),
            &BindGroupEntries::sequential((
                ssr.scene_depth_sampled(),
                ssr.normal_roughness_view(),
                restir.src_buffer().as_entire_binding(),
                resolve.gi_out_view(),
            )),
        );

        commands
            .entity(entity)
            .insert(ViewWorldRestirResolveBindGroup { group });
    }
}
