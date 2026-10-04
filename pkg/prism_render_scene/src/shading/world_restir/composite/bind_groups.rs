//! `PrepareBindGroups` system building the world-space `ReSTIR` composite's two
//! per-view bind groups.
//!
//! The copy group lifts the shading-resolved `scene_color` into the scratch
//! `gi_base`; the fold group reads that base, the resolve's `gi_out` export,
//! the Lambertian albedo and the shading resolve's clustered punctual direct
//! export, and writes `scene_color`. Both are present exactly when the view
//! carries a visibility buffer (`scene_color`, `ssgi_albedo`,
//! `world_restir_direct`), the resolve export ([`ViewWorldRestirResolve`]) and
//! the composite scratch ([`ViewWorldRestirComposite`]) — i.e. while the
//! subsystem is enabled. The groups are rebuilt every frame to track the
//! texture-cache views.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use super::super::super::resources::ViewVisibilityBuffer;
use super::super::resolve::ViewWorldRestirResolve;
use super::pipeline::WorldRestirCompositePipeline;
use super::resources::ViewWorldRestirComposite;

/// The composite's two per-view bind groups. Present only when the visibility
/// buffer, the resolve export and the composite scratch are all resident.
#[derive(Component)]
pub(crate) struct ViewWorldRestirCompositeBindGroups {
    /// group 0 for `wr_copy_base`: `scene_color` read + `gi_base` write.
    copy: BindGroup,
    /// group 0 for `wr_composite`: `gi_base` + `gi_out` reads, `scene_color`
    /// write, then the Lambertian albedo + clustered punctual direct exports.
    fold: BindGroup,
}

impl ViewWorldRestirCompositeBindGroups {
    /// group-0 bind group for the `wr_copy_base` dispatch.
    pub(crate) fn copy(&self) -> &BindGroup {
        &self.copy
    }

    /// group-0 bind group for the `wr_composite` dispatch.
    pub(crate) fn fold(&self) -> &BindGroup {
        &self.fold
    }
}

/// `PrepareBindGroups` system building [`ViewWorldRestirCompositeBindGroups`]
/// for every view whose visibility buffer, resolve export and composite scratch
/// are live.
pub(crate) fn prepare_world_restir_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<WorldRestirCompositePipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        &ViewWorldRestirResolve,
        &ViewWorldRestirComposite,
    )>,
) {
    for (entity, visibility, resolve, composite) in &views {
        let copy = device.create_bind_group(
            "prism world-space ReSTIR composite copy",
            pipeline.copy_layout(),
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                composite.gi_base_view(),
            )),
        );
        let fold = device.create_bind_group(
            "prism world-space ReSTIR composite fold",
            pipeline.fold_layout(),
            &BindGroupEntries::sequential((
                composite.gi_base_view(),
                resolve.gi_out_view(),
                visibility.scene_color_view(),
                visibility.ssgi_albedo_view(),
                visibility.world_restir_direct_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewWorldRestirCompositeBindGroups { copy, fold });
    }
}
