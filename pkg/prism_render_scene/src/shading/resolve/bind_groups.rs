//! Per-view preparation of the shading-resolve bind groups.
//!
//! The material (group 1) and light (group 3) bind groups already live on the
//! shared [`MaterialBindGroup`]/[`LightBindGroup`] resources, so this stage only
//! builds the two pass-owned groups:
//!
//! * **group 0** — the two visibility textures, the HDR storage-texture output
//!   and the screen-space GTAO input (all sourced from the view), the two
//!   global IBL tables (the prefiltered environment cube and the DFG lookup
//!   table with their samplers), and the two write-only SSR energy-export
//!   targets (`ssr_env_specular`, `ssr_spec_weight`) the SSR composite reads.
//! * **group 2** — the per-view compacted worklist ([`ViewShadingBuffers`])
//!   spliced together with the render-world scene-instance and
//!   shading-geometry tables.
//!
//! Both groups are cleared to `None` (by dropping the component) unless every
//! upstream buffer is resident, so [`super::dispatch`] can treat a present
//! [`ViewResolveBindGroups`] as "safe to record".

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries},
    renderer::RenderDevice,
    texture::FallbackImage,
};

use crate::{GpuSceneBuffers, RenderShadingGeometryBuffers};

use super::super::ao::ViewGtaoTextures;
use super::super::ibl::{DfgLutTexture, PrefilteredEnvironmentMap};
use super::super::resources::{ViewShadingBuffers, ViewVisibilityBuffer};
use super::motion::ViewMotionUniform;
use super::pipeline::ShadingResolvePipeline;

/// The two pass-owned bind groups (group 0 + group 2) for one view.
///
/// Present only when every backing buffer/texture is resident; its absence is
/// the dispatch node's signal to skip the view this frame.
#[derive(Component)]
pub(crate) struct ViewResolveBindGroups {
    /// group 0: visibility ids/metadata + HDR storage-texture output.
    pub(crate) view: BindGroup,
    /// group 2: worklist + scene/geometry tables.
    pub(crate) scene: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewResolveBindGroups`] for every view
/// that has both a visibility buffer and a shading worklist, provided the
/// render-world scene-instance and shading-geometry tables have uploaded.
pub(crate) fn prepare_shading_resolve_bind_groups(
    mut commands: Commands,
    pipeline: Res<ShadingResolvePipeline>,
    device: Res<RenderDevice>,
    scene: Res<GpuSceneBuffers>,
    geometry: Res<RenderShadingGeometryBuffers>,
    fallback: Res<FallbackImage>,
    prefiltered_env: Res<PrefilteredEnvironmentMap>,
    dfg_lut: Res<DfgLutTexture>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        &ViewShadingBuffers,
        &ViewMotionUniform,
        Option<&ViewGtaoTextures>,
    )>,
) {
    // Scene/geometry tables are shared across all views; if either has not
    // uploaded yet there is nothing to resolve, so clear any stale groups.
    let (
        Some(instances),
        Some(current_transforms),
        Some(previous_transforms),
        Some((geo_headers, geo_vertices, geo_primitives)),
    ) = (
        scene.instances(),
        scene.current_transforms(),
        scene.previous_transforms(),
        geometry.buffers(),
    )
    else {
        for (entity, _, _, _, _) in &views {
            commands
                .entity(entity)
                .remove::<ViewResolveBindGroups>();
        }
        return;
    };

    for (entity, visibility, buffers, motion, gtao) in &views {
        let (ids, metadata) = visibility.attachments();
        // Bind the view's GTAO visibility when present, else a 1x1 white
        // texture so the shader's multiply is a no-op (the dispatch also gates
        // on the `RESOLVE_FLAG_GTAO` bit, so the fallback is never actually read).
        let ao_view = gtao.map_or(&fallback.d2.texture_view, |textures| {
            textures.ambient_occlusion_view()
        });
        let view = device.create_bind_group(
            "prism resolve view",
            &pipeline.view_layout,
            &BindGroupEntries::sequential((
                ids,
                metadata,
                visibility.scene_color_view(),
                ao_view,
                prefiltered_env.cube_view(),
                prefiltered_env.sampler(),
                dfg_lut.view(),
                dfg_lut.sampler(),
                // 8-9: SSR energy-conservation exports written every pixel.
                visibility.ssr_env_specular_view(),
                visibility.ssr_spec_weight_view(),
                // 10-11: motion-vector G-buffer + the current/previous
                // view-projection uniform that projects it.
                visibility.motion_vectors_view(),
                motion.buffer.as_entire_binding(),
                // 12-13: screen-space GI exports (pre-albedo ambient + albedo).
                visibility.ssgi_ambient_view(),
                visibility.ssgi_albedo_view(),
            )),
        );
        let scene_group = device.create_bind_group(
            "prism resolve scene",
            &pipeline.scene_layout,
            &BindGroupEntries::sequential((
                buffers.work_items.as_entire_binding(),
                buffers.class_offsets.as_entire_binding(),
                buffers.class_counts.as_entire_binding(),
                instances.as_entire_binding(),
                geo_headers.as_entire_binding(),
                geo_vertices.as_entire_binding(),
                geo_primitives.as_entire_binding(),
                // 7: per-instance current `world_from_local`, used to lift
                // local-space geometry into world space before lighting.
                current_transforms.as_entire_binding(),
                // 8: matching previous-frame transforms, used only to place the
                // surface in last frame's world space for the motion vector.
                previous_transforms.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(ViewResolveBindGroups {
            view,
            scene: scene_group,
        });
    }
}
