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
use bevy_math::Vec3;
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, Buffer, BufferInitDescriptor, BufferUsages},
    renderer::RenderDevice,
    texture::FallbackImage,
};

use prism_render_shading::ReceiverProjection;

use crate::{GpuSceneBuffers, RenderShadingGeometryBuffers};

use super::super::ao::ViewGtaoTextures;
use super::super::ibl::{DfgLutTexture, PrefilteredEnvironmentMap};
use super::super::light_routing::ViewLightRouting;
use super::super::resources::{ViewShadingBuffers, ViewVisibilityBuffer};
use super::super::runtime::PrismShadingSettings;
use super::super::virtual_shadow::{
    PrismVirtualShadowSettings, ViewVsmPhysicalAtlas, VsmPrimaryLight,
};
use super::abi::GpuVsmResolveParams;
use super::motion::ViewMotionUniform;
use super::pipeline::ShadingResolvePipeline;

/// Resolve-owned mirror of the virtual-shadow-map page table for one view.
///
/// The page table itself lives on the private `virtual_shadow::ViewVsmPageTable`
/// component, which is not re-exported outside its module, so the parent plugin
/// bridges it into this resolve-visible component after the page-table readback
/// attaches it:
///
/// ```ignore
/// for (entity, page_table) in &page_tables {
///     commands.entity(entity).insert(ViewResolveVsmPageTable {
///         buffer: page_table.buffer.clone(),
///         slot_count: page_table.slot_count,
///     });
/// }
/// ```
///
/// When absent (feature off, or the bridge/readback has not run yet) the resolve
/// binds the pipeline's fallback page table and clears the uniform's `enable`
/// bit, so the shader never samples a stale or missing table.
#[derive(Component)]
pub(crate) struct ViewResolveVsmPageTable {
    /// Flat virtual->physical page-table storage buffer
    /// (`STORAGE | COPY_DST`), one `u32` physical index per resident-window slot.
    pub(crate) buffer: Buffer,
    /// Number of `u32` slots the table covers (`levels * edge * edge`); carried
    /// for the parent's diagnostics, the shader guards the index itself.
    #[expect(
        dead_code,
        reason = "read by the parent plugin's page-table bridge/diagnostics wired in a later slice"
    )]
    pub(crate) slot_count: u32,
}

/// The pass-owned bind groups (group 0 + group 2 + group 6) for one view.
///
/// Present only when every backing buffer/texture is resident; its absence is
/// the dispatch node's signal to skip the view this frame.
#[derive(Component)]
pub(crate) struct ViewResolveBindGroups {
    /// group 0: visibility ids/metadata + HDR storage-texture output.
    pub(crate) view: BindGroup,
    /// group 2: worklist + scene/geometry tables.
    pub(crate) scene: BindGroup,
    /// group 6: virtual-shadow-map page table + physical atlas + sampler +
    /// [`GpuVsmResolveParams`] uniform. Always built with real or fallback
    /// resources so the pipeline's group 6 is bound every dispatch.
    pub(crate) vsm: BindGroup,
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
    shading_settings: Res<PrismShadingSettings>,
    vsm_settings: Option<Res<PrismVirtualShadowSettings>>,
    primary_light: Option<Res<VsmPrimaryLight>>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        &ViewShadingBuffers,
        &ViewMotionUniform,
        Option<&ViewGtaoTextures>,
        Option<&ViewVsmPhysicalAtlas>,
        Option<&ViewResolveVsmPageTable>,
        Option<&ViewLightRouting>,
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
        for (entity, _, _, _, _, _, _, _) in &views {
            commands.entity(entity).remove::<ViewResolveBindGroups>();
        }
        return;
    };

    for (entity, visibility, buffers, motion, gtao, vsm_atlas, vsm_page_table, light_routing) in
        &views
    {
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
                // 9: the light-routing channel-gated visibility mask. Bound
                // from this view's `ViewLightRouting` when the opt-in Lighting
                // Channels subsystem is on; otherwise the pipeline's all-ones
                // dummy keeps every punctual light visible.
                light_routing.map_or_else(
                    || pipeline.scene_dummy_visible_lights.as_entire_binding(),
                    |routing| routing.visible_buffer().as_entire_binding(),
                ),
            )),
        );
        // group 6: virtual-shadow-map sample bindings. The VSM branch fires
        // only when the feature is on, the settings and primary directional
        // light exist, and this view has both a resident page table and a
        // physical atlas; otherwise every binding falls back to a real, valid
        // pipeline-owned dummy and the uniform's `enable` bit stays clear so the
        // shader always takes the cascaded-shadow path.
        let light_direction = primary_light.as_ref().and_then(|light| light.direction);
        let enable = shading_settings.enable_virtual_shadow
            && vsm_settings.is_some()
            && vsm_atlas.is_some()
            && vsm_page_table.is_some()
            && light_direction.is_some();

        // Clipmap addressing + atlas geometry: real values when the settings and
        // atlas are present, else the defaults / 1x1 dummy geometry that pair
        // with the fallback bindings (never sampled because `enable` is clear).
        let clipmap = vsm_settings.as_ref().map_or_else(
            || PrismVirtualShadowSettings::default().clipmap(),
            |s| s.clipmap(),
        );
        let pcf_radius = vsm_settings.as_ref().map_or(0, |s| s.pcf_radius);
        let (physical_pages, physical_pages_per_edge) = vsm_atlas.map_or((1, 1), |atlas| {
            (atlas.physical_pages(), atlas.physical_pages_per_edge())
        });

        // Light clipmap basis (right/up for page addressing, forward for the
        // along-light reference depth), derived from the primary light exactly
        // like the receiver-generation golden. Falls back to the deterministic
        // straight-down frame when there is no directional light.
        let direction = light_direction.unwrap_or(Vec3::NEG_Y);
        let projection = ReceiverProjection::from_light_direction(direction, Vec3::ZERO, 0.0);
        let light_forward = {
            let normalized = direction.normalize_or_zero();
            if normalized == Vec3::ZERO {
                Vec3::NEG_Y
            } else {
                normalized
            }
        };

        let vsm_params = GpuVsmResolveParams::new(
            &clipmap,
            physical_pages,
            physical_pages_per_edge,
            pcf_radius,
            enable,
            projection.light_right,
            projection.light_up,
            light_forward,
        );
        let vsm_params_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism resolve vsm params"),
            contents: bytemuck::bytes_of(&vsm_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Bind the real page table + atlas when both are resident, else the
        // pipeline's format-correct fallbacks. The atlas carries its own linear
        // sampler, but the layout declares a `NonFiltering` sampler (the twin
        // compares `R32Float` depth manually and must not require the optional
        // `FLOAT32_FILTERABLE` feature), so both paths bind `pipeline.vsm_sampler`.
        let page_table_binding = vsm_page_table.map_or_else(
            || pipeline.vsm_dummy_page_table.as_entire_binding(),
            |table| table.buffer.as_entire_binding(),
        );
        let atlas_view = vsm_atlas.map_or(&pipeline.vsm_dummy_atlas, |atlas| atlas.atlas_view());
        let vsm = device.create_bind_group(
            "prism resolve vsm",
            &pipeline.vsm_layout,
            &BindGroupEntries::sequential((
                page_table_binding,
                atlas_view,
                &pipeline.vsm_sampler,
                vsm_params_buffer.as_entire_binding(),
            )),
        );

        commands.entity(entity).insert(ViewResolveBindGroups {
            view,
            scene: scene_group,
            vsm,
        });
    }
}
