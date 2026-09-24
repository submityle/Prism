use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_pbr::MeshPipelineSystems;
use bevy_render::{
    init_gpu_resource, GpuResourceAppExt, render_phase::AddRenderCommand, render_resource::SpecializedRenderPipelines,
    Render, RenderApp, RenderStartup, RenderSystems,
};

use super::{
    classification_gpu::{
        dispatch_material_classification, init_material_classification_pipeline,
        prepare_material_classification_bind_groups,
    },
    composite::{
        composite_shading, init_shading_composite_pipeline,
        prepare_shading_composite_bind_groups, prepare_shading_composite_pipelines,
        ShadingCompositePipeline,
    },
    resolve::{
        dispatch_shading_resolve, init_shading_resolve_pipeline,
        prepare_shading_resolve_bind_groups,
    },
    graph::shading_frame_graph,
    raster::{
        init_visibility_raster, queue_visibility_raster, visibility_raster_pass,
        DrawVisibilityRaster, Visibility3d, VisibilityRasterPipeline,
    },
    resources::{prepare_shading_buffers, prepare_visibility_buffers},
    runtime::{
        detect_shading_capabilities, prepare_shading_work, PrismShadingDiagnostics,
        PrismShadingSettings, ShadingFrameGraph,
    },
};

pub struct PrismShadingPlugin;

impl Plugin for PrismShadingPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "../shaders/visibility_raster.wesl");
        embedded_asset!(app, "../shaders/material_classification.wesl");
        embedded_asset!(app, "../shaders/brdf.wesl");
        embedded_asset!(app, "../shaders/cloth.wesl");
        embedded_asset!(app, "../shaders/subsurface.wesl");
        embedded_asset!(app, "../shaders/hair.wesl");
        embedded_asset!(app, "../shaders/water.wesl");
        embedded_asset!(app, "../shaders/clearcoat.wesl");
        embedded_asset!(app, "../shaders/material_sample.wesl");
        embedded_asset!(app, "../shaders/tangent.wesl");
        embedded_asset!(app, "../shaders/surface.wesl");
        embedded_asset!(app, "../shaders/shading_resolve.wesl");
        embedded_asset!(app, "../shaders/composite.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        let compiled_graph = shading_frame_graph()
            .compile()
            .expect("Prism shading frame graph must be valid");
        render_app
            .init_resource::<bevy_render::render_phase::ViewBinnedRenderPhases<Visibility3d>>()
            .init_resource::<bevy_render::render_phase::DrawFunctions<Visibility3d>>()
            .init_resource::<bevy_render::render_resource::SpecializedMeshPipelines<VisibilityRasterPipeline>>()
            .init_gpu_resource::<SpecializedRenderPipelines<ShadingCompositePipeline>>()
            .init_resource::<PrismShadingSettings>()
            .init_resource::<PrismShadingDiagnostics>()
            .insert_resource(ShadingFrameGraph {
                compiled: compiled_graph,
            })
            .add_render_command::<Visibility3d, DrawVisibilityRaster>()
            .add_systems(RenderStartup, detect_shading_capabilities)
            .add_systems(
                RenderStartup,
                (
                    init_visibility_raster.after(MeshPipelineSystems),
                    init_material_classification_pipeline
                        .after(init_gpu_resource::<crate::MaterialBindGroup>),
                    init_shading_resolve_pipeline
                        .after(init_gpu_resource::<crate::MaterialBindGroup>)
                        .after(init_gpu_resource::<crate::LightBindGroup>),
                    init_shading_composite_pipeline,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_shading_work
                        .after(super::super::visibility::systems::build_unified_visibility)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_visibility_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_shading_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_shading_composite_pipelines
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::Prepare),
                    prepare_material_classification_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_shading_resolve_bind_groups
                        .after(prepare_material_classification_bind_groups)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_shading_composite_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                    queue_visibility_raster.in_set(RenderSystems::QueueMeshes),
                ),
            );
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            (
                visibility_raster_pass.before(bevy_core_pipeline::Core3dSystems::MainPass),
                dispatch_material_classification
                    .after(visibility_raster_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                dispatch_shading_resolve
                    .after(dispatch_material_classification)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                composite_shading
                    .after(bevy_core_pipeline::Core3dSystems::MainPass)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
            ),
        );
    }
}
