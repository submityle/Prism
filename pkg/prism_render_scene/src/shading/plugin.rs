use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_pbr::MeshPipelineSystems;
use bevy_render::{
    init_gpu_resource, ExtractSchedule, GpuResourceAppExt, render_phase::AddRenderCommand,
    render_resource::SpecializedRenderPipelines, Render, RenderApp, RenderStartup, RenderSystems,
};

use super::{
    ao::{
        gtao_compute_pass, gtao_prepass_pass, init_gtao_kernel_pipeline,
        init_gtao_prepass_pipeline, prepare_gtao_kernel_bind_groups,
        prepare_gtao_prepass_bind_groups, prepare_gtao_textures,
    },
    ssr::{
        init_ssr_hzb_pipeline, init_ssr_prepass_pipeline, prepare_ssr_hzb_bind_groups,
        prepare_ssr_prepass_bind_groups, prepare_ssr_textures, ssr_hzb_pass, ssr_prepass_pass,
    },
    ibl::{
        dfg_lut_precompute_pass, env_prefilter_precompute_pass, extract_ibl_source,
        init_brdf_lut_pipeline, init_dfg_lut_texture, init_env_prefilter_pipeline,
        init_prefiltered_env_map, prepare_dfg_lut_bind_group,
        prepare_env_prefilter_bind_groups, EnvPrefilterBindGroups, ExtractedIblSource,
    },
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
    transparent::{
        init_oit_composite_pipeline, init_oit_forward_pipeline, oit_composite,
        prepare_oit_composite_bind_groups, prepare_oit_composite_pipelines, prepare_oit_targets,
        queue_transparent_oit, transparent_forward_pass, DrawTransparentOit, OitCompositePipeline,
        OitForwardPipeline, TransparentOit3d,
    },
    shadow::{
        ensure_shadow_atlas, extract_shadows, init_shadow_depth_pipeline, prepare_shadow_bind_group,
        prepare_shadow_depth_uniform, queue_shadow_depth, rebuild_shadow_buffers,
        register_shadow_depth_shader, shadow_depth_pass, write_shadow_buffers, ExtractedShadows,
        PrismShadowSettings, ShadowAtlas, ShadowAtlasConfig, ShadowBindGroup, ShadowDepthDrawList,
        ShadowDepthPipeline, ShadowDepthViewOffsets, ShadowDepthViewUniform,
        ShadowGpuBuffers, DEFAULT_SHADOW_ATLAS_LAYERS, DEFAULT_SHADOW_ATLAS_RESOLUTION,
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
        embedded_asset!(app, "../shaders/shadow.wesl");
        embedded_asset!(app, "../shaders/shading_resolve.wesl");
        embedded_asset!(app, "../shaders/gtao_prepass.wesl");
        embedded_asset!(app, "../shaders/gtao.wesl");
        embedded_asset!(app, "../shaders/ssr_prepass.wesl");
        embedded_asset!(app, "../shaders/ssr_hzb.wesl");
        embedded_asset!(app, "../shaders/brdf_lut.wesl");
        embedded_asset!(app, "../shaders/env_prefilter.wesl");
        embedded_asset!(app, "../shaders/composite.wesl");
        embedded_asset!(app, "../shaders/oit.wesl");
        embedded_asset!(app, "../shaders/transparent.wesl");
        register_shadow_depth_shader(app);
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
            .init_resource::<bevy_render::render_phase::ViewBinnedRenderPhases<TransparentOit3d>>()
            .init_resource::<bevy_render::render_phase::DrawFunctions<TransparentOit3d>>()
            .init_resource::<bevy_render::render_resource::SpecializedMeshPipelines<OitForwardPipeline>>()
            .init_resource::<bevy_render::render_resource::SpecializedMeshPipelines<ShadowDepthPipeline>>()
            .init_resource::<ShadowDepthDrawList>()
            .init_resource::<ShadowDepthViewOffsets>()
            .init_gpu_resource::<SpecializedRenderPipelines<ShadingCompositePipeline>>()
            .init_gpu_resource::<SpecializedRenderPipelines<OitCompositePipeline>>()
            .init_resource::<PrismShadingSettings>()
            .init_resource::<PrismShadingDiagnostics>()
            .insert_resource(ShadowAtlasConfig::new(
                DEFAULT_SHADOW_ATLAS_LAYERS,
                DEFAULT_SHADOW_ATLAS_RESOLUTION,
            ))
            .init_resource::<ExtractedShadows>()
            .init_resource::<PrismShadowSettings>()
            .init_resource::<ExtractedIblSource>()
            .init_resource::<EnvPrefilterBindGroups>()
            .insert_resource(ShadingFrameGraph {
                compiled: compiled_graph,
            })
            .add_render_command::<Visibility3d, DrawVisibilityRaster>()
            .add_render_command::<TransparentOit3d, DrawTransparentOit>()
            .add_systems(RenderStartup, detect_shading_capabilities)
            .add_systems(
                RenderStartup,
                (
                    init_visibility_raster.after(MeshPipelineSystems),
                    init_oit_forward_pipeline.after(MeshPipelineSystems),
                    init_material_classification_pipeline
                        .after(init_gpu_resource::<crate::MaterialBindGroup>),
                    init_shading_resolve_pipeline
                        .after(init_gpu_resource::<crate::MaterialBindGroup>)
                        .after(init_gpu_resource::<crate::LightBindGroup>)
                        .after(init_gpu_resource::<ShadowBindGroup>)
                        .after(init_gpu_resource::<crate::ClusterBindGroup>),
                    init_shading_composite_pipeline,
                    init_oit_composite_pipeline,
                    init_gtao_prepass_pipeline,
                    init_gtao_kernel_pipeline,
                    init_ssr_prepass_pipeline,
                    init_ssr_hzb_pipeline,
                    init_dfg_lut_texture,
                    init_brdf_lut_pipeline,
                    init_prefiltered_env_map,
                    init_env_prefilter_pipeline,
                ),
            )
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<ShadowAtlas>,
                    init_gpu_resource::<ShadowGpuBuffers>,
                    init_gpu_resource::<ShadowBindGroup>,
                )
                    .chain(),
            )
            .add_systems(
                RenderStartup,
                (
                    init_shadow_depth_pipeline
                        .after(init_gpu_resource::<crate::buffers::GpuSceneBindGroup>),
                    init_gpu_resource::<ShadowDepthViewUniform>,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_shading_work
                        .after(super::super::visibility::systems::build_unified_visibility)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_visibility_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_gtao_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_oit_targets
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_shading_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_shading_composite_pipelines
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::Prepare),
                    prepare_oit_composite_pipelines
                        .after(prepare_oit_targets)
                        .in_set(RenderSystems::Prepare),
                    prepare_material_classification_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_shading_resolve_bind_groups
                        .after(prepare_material_classification_bind_groups)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_gtao_prepass_bind_groups
                        .after(prepare_gtao_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    ensure_shadow_atlas.in_set(RenderSystems::PrepareResources),
                    rebuild_shadow_buffers.in_set(RenderSystems::PrepareResources),
                    write_shadow_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                    prepare_shadow_bind_group
                        .after(write_shadow_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_shading_composite_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_oit_composite_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                    queue_visibility_raster.in_set(RenderSystems::QueueMeshes),
                    queue_transparent_oit.in_set(RenderSystems::QueueMeshes),
                    queue_shadow_depth.in_set(RenderSystems::QueueMeshes),
                    prepare_shadow_depth_uniform
                        .after(write_shadow_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Render,
                prepare_gtao_kernel_bind_groups
                    .after(prepare_gtao_textures)
                    .in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                (
                    prepare_ssr_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_ssr_prepass_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_hzb_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_dfg_lut_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_env_prefilter_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(ExtractSchedule, (extract_shadows, extract_ibl_source));
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            (
                shadow_depth_pass.before(visibility_raster_pass),
                visibility_raster_pass.before(bevy_core_pipeline::Core3dSystems::MainPass),
                dispatch_material_classification
                    .after(visibility_raster_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                gtao_prepass_pass
                    .after(visibility_raster_pass)
                    .before(dispatch_shading_resolve),
                gtao_compute_pass
                    .after(gtao_prepass_pass)
                    .before(dispatch_shading_resolve),
                ssr_prepass_pass
                    .after(visibility_raster_pass)
                    .before(dispatch_shading_resolve),
                ssr_hzb_pass
                    .after(ssr_prepass_pass)
                    .before(dispatch_shading_resolve),
                dfg_lut_precompute_pass.before(dispatch_shading_resolve),
                env_prefilter_precompute_pass.before(dispatch_shading_resolve),
                dispatch_shading_resolve
                    .after(dispatch_material_classification)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                composite_shading
                    .after(bevy_core_pipeline::Core3dSystems::MainPass)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
                transparent_forward_pass
                    .after(dispatch_shading_resolve)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                oit_composite
                    .after(composite_shading)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
            ),
        );
    }
}
