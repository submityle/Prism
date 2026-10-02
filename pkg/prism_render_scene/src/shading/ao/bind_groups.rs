//! Per-view preparation of the GTAO prepass bind groups.
//!
//! Builds the two pass-owned groups for every view that has both a resident
//! visibility buffer and the [`ViewGtaoTextures`] the prepare stage allocates:
//!
//! * **group 0** — the two visibility textures read plus the linear-depth /
//!   view-normal targets written, sourced from the view's
//!   [`ViewVisibilityBuffer`] and [`ViewGtaoTextures`].
//! * **group 1** — the shared render-world scene-instance and shading-geometry
//!   tables the surface reconstruction walks.
//!
//! The component is removed unless every upstream buffer/texture is resident,
//! so [`super::dispatch`] can treat a present [`ViewGtaoPrepassBindGroups`] as
//! "safe to record".

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
};

use crate::{GpuSceneBuffers, RenderShadingGeometryBuffers};

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;
use super::pipeline::{GtaoDenoisePipeline, GtaoKernelPipeline, GtaoPrepassPipeline};
use super::resources::ViewGtaoTextures;

/// The two pass-owned bind groups (group 0 + group 1) for one view's GTAO
/// prepass.  Present only when every backing buffer/texture is resident.
#[derive(Component)]
pub(crate) struct ViewGtaoPrepassBindGroups {
    /// group 0: visibility ids/metadata + linear-depth/view-normal outputs.
    pub(crate) view: BindGroup,
    /// group 1: scene instances + geometry tables.
    pub(crate) scene: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewGtaoPrepassBindGroups`] for every
/// view that has both a visibility buffer and GTAO textures, provided the
/// shared scene-instance and shading-geometry tables have uploaded.
pub(crate) fn prepare_gtao_prepass_bind_groups(
    mut commands: Commands,
    pipeline: Res<GtaoPrepassPipeline>,
    device: Res<RenderDevice>,
    scene: Res<GpuSceneBuffers>,
    geometry: Res<RenderShadingGeometryBuffers>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewGtaoTextures)>,
) {
    // Scene/geometry tables are shared across all views; without them there is
    // nothing to reconstruct, so drop any stale groups.
    let (Some(instances), Some((geo_headers, geo_vertices, geo_primitives))) =
        (scene.instances(), geometry.buffers())
    else {
        for (entity, _, _) in &views {
            commands
                .entity(entity)
                .remove::<ViewGtaoPrepassBindGroups>();
        }
        return;
    };

    for (entity, visibility, textures) in &views {
        let (ids, metadata) = visibility.attachments();
        let view = device.create_bind_group(
            "prism GTAO prepass view",
            &pipeline.view_layout,
            &BindGroupEntries::sequential((
                ids,
                metadata,
                textures.linear_depth_view(),
                textures.view_normal_view(),
            )),
        );
        let scene_group = device.create_bind_group(
            "prism GTAO prepass scene",
            &pipeline.scene_layout,
            &BindGroupEntries::sequential((
                instances.as_entire_binding(),
                geo_headers.as_entire_binding(),
                geo_vertices.as_entire_binding(),
                geo_primitives.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(ViewGtaoPrepassBindGroups {
            view,
            scene: scene_group,
        });
    }
}

/// The single kernel bind group (group 0) for one view's GTAO compute pass:
/// the linear-depth + view-normal inputs and the ambient-visibility output.
/// Present only when the view has resident [`ViewGtaoTextures`].
#[derive(Component)]
pub(crate) struct ViewGtaoKernelBindGroup {
    /// group 0: linear-depth + view-normal reads + ambient-visibility write.
    pub(crate) view: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewGtaoKernelBindGroup`] for every
/// view that has GTAO textures. Unlike the prepass, the kernel touches no
/// scene tables, so it depends only on the per-view textures.
pub(crate) fn prepare_gtao_kernel_bind_groups(
    mut commands: Commands,
    pipeline: Res<GtaoKernelPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewGtaoTextures)>,
) {
    for (entity, textures) in &views {
        let view = device.create_bind_group(
            "prism GTAO kernel view",
            &pipeline.view_layout,
            &BindGroupEntries::sequential((
                textures.linear_depth_view(),
                textures.view_normal_view(),
                textures.raw_ambient_occlusion_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewGtaoKernelBindGroup { view });
    }
}

/// The single denoise bind group (group 0) for one view's GTAO spatial denoise
/// pass: the raw ambient-visibility, linear-depth, and view-normal inputs and
/// the denoised ambient-visibility output. Present only when the view has
/// resident [`ViewGtaoTextures`].
#[derive(Component)]
pub(crate) struct ViewGtaoDenoiseBindGroup {
    /// group 0: raw-AO + linear-depth + view-normal reads + denoised-AO write.
    pub(crate) view: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewGtaoDenoiseBindGroup`] for every
/// view that has GTAO textures. Like the kernel it touches no scene tables, so
/// it depends only on the per-view textures.
pub(crate) fn prepare_gtao_denoise_bind_groups(
    mut commands: Commands,
    pipeline: Res<GtaoDenoisePipeline>,
    device: Res<RenderDevice>,
    settings: Res<PrismShadingSettings>,
    views: Query<(Entity, &ViewGtaoTextures)>,
) {
    for (entity, textures) in &views {
        // With temporal accumulation on, the denoise feeds the temporal pass via
        // the dedicated `denoised_ambient_occlusion` target and the temporal
        // pass writes the final `ambient_occlusion` the resolve reads. With it
        // off, the denoise writes `ambient_occlusion` directly.
        let denoise_out = if settings.enable_gtao_temporal {
            textures.denoised_ambient_occlusion_view()
        } else {
            textures.ambient_occlusion_view()
        };
        let view = device.create_bind_group(
            "prism GTAO denoise view",
            &pipeline.view_layout,
            &BindGroupEntries::sequential((
                textures.raw_ambient_occlusion_view(),
                textures.linear_depth_view(),
                textures.view_normal_view(),
                denoise_out,
            )),
        );
        commands
            .entity(entity)
            .insert(ViewGtaoDenoiseBindGroup { view });
    }
}
