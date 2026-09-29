use super::{
    bindings::MaterialBindGroup,
    buffers::MaterialGpuBuffers,
    runtime::{PrismMaterialDiagnostics, RenderMaterialRegistry},
    systems::{
        configure_material_texture_capacity, extract_standard_materials,
        prepare_material_bind_group, rebuild_material_buffers, reclaim_completed_materials,
        stage_material_uploads, write_material_buffers,
    },
    texture_upload::{prepare_material_texture_arrays, MaterialTextureArrays},
};
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    init_gpu_resource,
    renderer::{RenderGraph, RenderGraphSystems},
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};

/// Mirrors `StandardMaterial` assets into the unified Prism Material ABI.
pub struct PrismMaterialPlugin;
impl Plugin for PrismMaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "../shaders/material.wesl");
        // Variable-length surface parameter word-heap decode (ABI v4), imported by
        // every material consumer (opaque/transparent/visibility/shading_resolve/ssr_repack).
        embedded_asset!(app, "../shaders/material_unpack.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<crate::completion::GpuCompletionTracker>()
            .init_resource::<RenderMaterialRegistry>()
            .init_resource::<PrismMaterialDiagnostics>()
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<MaterialGpuBuffers>,
                    init_gpu_resource::<MaterialTextureArrays>,
                    init_gpu_resource::<MaterialBindGroup>,
                    configure_material_texture_capacity,
                    rebuild_material_buffers,
                )
                    .chain(),
            )
            .add_systems(ExtractSchedule, extract_standard_materials)
            .add_systems(
                Render,
                (
                    stage_material_uploads.in_set(RenderSystems::PrepareResources),
                    prepare_material_texture_arrays.in_set(RenderSystems::PrepareResources),
                    write_material_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                    prepare_material_bind_group
                        .after(write_material_buffers)
                        .after(prepare_material_texture_arrays)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            );
        render_app.add_systems(
            RenderGraph,
            (
                crate::completion::track_submission.in_set(RenderGraphSystems::Finish),
                reclaim_completed_materials.in_set(RenderGraphSystems::Finish),
            ),
        );
    }
}
