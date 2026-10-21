use super::{
    bindings::MaterialBindGroup,
    buffers::MaterialGpuBuffers,
    runtime::{PrismMaterialDiagnostics, RenderMaterialRegistry},
    systems::{
        configure_material_texture_capacity, extract_material_vt_sources,
        extract_standard_materials, prepare_material_bind_group, rebuild_material_buffers,
        reclaim_completed_materials, stage_material_uploads, write_material_buffers,
    },
    texture_upload::{prepare_material_texture_arrays, MaterialTextureArrays},
    virtual_texture::MaterialVirtualTextureGpuBundle,
    virtual_texture::{
        collect_material_vt_feedback, map_submitted_material_vt_feedback,
        request_material_vt_feedback_readback, MaterialVtDemandRegistry,
        MaterialVtFeedbackReadback,
    },
    vt_producer::{prepare_material_virtual_texture, MaterialVtProducer},
    vt_source::{
        MaterialVtActivePublication, MaterialVtMaterialSourceQueries, MaterialVtSourceChangeQueue,
        MaterialVtSourceRegistry,
    },
};
use crate::shading::texel_cache::ShadeDemandRegistry;
use bevy_app::{App, Plugin};
use bevy_asset::io::embedded::EmbeddedAssetRegistry;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    init_gpu_resource,
    render_asset::AssetExtractionSystems,
    renderer::{RenderGraph, RenderGraphSystems},
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use std::path::{Path, PathBuf};

fn register_material_shader_assets(app: &mut App) {
    let embedded = app.world().resource::<EmbeddedAssetRegistry>();
    let source_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../prism_render_material/src/shaders");
    embedded.insert_asset(
        source_root.join("material.wesl"),
        Path::new("prism_render_material/shaders/material.wesl"),
        prism_render_material::shaders::MATERIAL_SHADER_BYTES,
    );
    embedded.insert_asset(
        source_root.join("material_sample.wesl"),
        Path::new("prism_render_material/shaders/material_sample.wesl"),
        prism_render_material::shaders::MATERIAL_SAMPLE_SHADER_BYTES,
    );
    embedded.insert_asset(
        source_root.join("material_virtual_texture.wesl"),
        Path::new("prism_render_material/shaders/material_virtual_texture.wesl"),
        prism_render_material::shaders::MATERIAL_VIRTUAL_TEXTURE_SHADER_BYTES,
    );
    embedded.insert_asset(
        source_root.join("material_unpack.wesl"),
        Path::new("prism_render_material/shaders/material_unpack.wesl"),
        prism_render_material::shaders::MATERIAL_UNPACK_SHADER_BYTES,
    );
    embedded.insert_asset(
        source_root.join("material_classification.wesl"),
        Path::new("prism_render_material/shaders/material_classification.wesl"),
        prism_render_material::shaders::MATERIAL_CLASSIFICATION_SHADER_BYTES,
    );
}

/// Mirrors `StandardMaterial` assets into the unified Prism Material ABI.
pub struct PrismMaterialPlugin;
impl Plugin for PrismMaterialPlugin {
    fn build(&self, app: &mut App) {
        register_material_shader_assets(app);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<crate::completion::GpuCompletionTracker>()
            .init_resource::<RenderMaterialRegistry>()
            .init_resource::<MaterialVtSourceRegistry>()
            .init_resource::<MaterialVtSourceChangeQueue>()
            .init_resource::<MaterialVtActivePublication>()
            .init_resource::<MaterialVtMaterialSourceQueries>()
            .init_resource::<MaterialVtProducer>()
            .init_resource::<MaterialVtDemandRegistry>()
            .init_resource::<MaterialVtFeedbackReadback>()
            .init_resource::<ShadeDemandRegistry>()
            .init_resource::<PrismMaterialDiagnostics>()
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<MaterialGpuBuffers>,
                    init_gpu_resource::<MaterialTextureArrays>,
                    init_gpu_resource::<MaterialVirtualTextureGpuBundle>,
                    // MaterialBindGroup consumes the disabled-safe VT bundle
                    // even before a residency producer enables it.
                    init_gpu_resource::<MaterialBindGroup>,
                    configure_material_texture_capacity,
                    rebuild_material_buffers,
                )
                    .chain(),
            )
            .add_systems(
                ExtractSchedule,
                (extract_material_vt_sources, extract_standard_materials)
                    .chain()
                    .before(AssetExtractionSystems),
            )
            .add_systems(
                Render,
                (
                    collect_material_vt_feedback
                        .in_set(RenderSystems::PrepareResources)
                        .before(prepare_material_virtual_texture),
                    prepare_material_virtual_texture.in_set(RenderSystems::PrepareResources),
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
                request_material_vt_feedback_readback
                    .after(bevy_core_pipeline::schedule::camera_driver)
                    .in_set(RenderGraphSystems::Render),
                map_submitted_material_vt_feedback.in_set(RenderGraphSystems::Finish),
                crate::completion::track_submission.in_set(RenderGraphSystems::Finish),
                reclaim_completed_materials.in_set(RenderGraphSystems::Finish),
            ),
        );
    }
}
