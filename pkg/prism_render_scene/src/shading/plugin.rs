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
        gtao_compute_pass, gtao_denoise_pass, gtao_prepass_pass, gtao_temporal_pass,
        init_gtao_denoise_pipeline, init_gtao_kernel_pipeline, init_gtao_prepass_pipeline,
        init_gtao_temporal_pipeline, prepare_gtao_denoise_bind_groups,
        prepare_gtao_kernel_bind_groups, prepare_gtao_prepass_bind_groups,
        prepare_gtao_temporal_bind_groups, prepare_gtao_temporal_textures, prepare_gtao_textures,
    },
    ssr::{
        init_ssr_color_mips_pipeline, init_ssr_composite_pipeline, init_ssr_hzb_pipeline,
        init_ssr_prepass_pipeline, init_ssr_repack_pipeline, init_ssr_trace_pipeline,
        init_ssr_reconstruct_pipeline, init_ssr_temporal_pipeline,
        prepare_ssr_color_mips_bind_groups, prepare_ssr_composite_bind_groups,
        prepare_ssr_hzb_bind_groups, prepare_ssr_prepass_bind_groups,
        prepare_ssr_reconstruct_bind_groups, prepare_ssr_temporal_bind_groups,
        prepare_ssr_temporal_textures,
        prepare_ssr_repack_bind_groups, prepare_ssr_textures, prepare_ssr_trace_bind_groups,
        ssr_color_mips_pass, ssr_composite_pass, ssr_hzb_pass, ssr_prepass_pass, ssr_reconstruct_pass,
        ssr_repack_pass, ssr_temporal_pass, ssr_trace_pass,
    },
    ssgi::{
        init_ssgi_composite_pipeline, init_ssgi_denoise_pipeline, init_ssgi_trace_pipeline,
        prepare_ssgi_composite_bind_groups, prepare_ssgi_denoise_bind_groups,
        prepare_ssgi_textures, prepare_ssgi_trace_bind_groups, ssgi_composite_pass,
        ssgi_denoise_pass, ssgi_trace_pass,
    },
    taa::{
        init_taa_resolve_pipeline, prepare_taa_bind_groups, prepare_taa_jitter,
        prepare_taa_textures, taa_resolve_pass,
    },
    virtual_shadow::{
        extract_vsm_primary_light, init_vsm_receiver_gen_pipeline,
        prepare_vsm_receiver_gen_bind_groups, prepare_vsm_receiver_resources,
        vsm_receiver_gen_pass, PrismVirtualShadowSettings, VsmPrimaryLight,
        VsmReceiverBufferCache,
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
    exposure::{
        exposure_average_pass, exposure_histogram_pass, init_exposure_average_pipeline,
        init_exposure_histogram_pipeline, prepare_exposure_average_bind_groups,
        prepare_exposure_buffers, prepare_exposure_histogram_bind_groups,
    },
    bloom::{
        bloom_pass, init_bloom_pipelines, prepare_bloom_bind_groups, prepare_bloom_textures,
    },
    resolve::{
        dispatch_shading_resolve, init_shading_resolve_pipeline, prepare_resolve_motion,
        prepare_shading_resolve_bind_groups, ResolveMotionHistory,
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
        embedded_asset!(app, "../shaders/scene_transform.wesl");
        embedded_asset!(app, "../shaders/shadow.wesl");
        embedded_asset!(app, "../shaders/shading_resolve.wesl");
        embedded_asset!(app, "../shaders/gtao_prepass.wesl");
        embedded_asset!(app, "../shaders/gtao.wesl");
        embedded_asset!(app, "../shaders/gtao_denoise.wesl");
        embedded_asset!(app, "../shaders/gtao_temporal.wesl");
        embedded_asset!(app, "../shaders/ssr_prepass.wesl");
        embedded_asset!(app, "../shaders/ssr_hzb.wesl");
        embedded_asset!(app, "../shaders/ssr_repack.wesl");
        embedded_asset!(app, "../shaders/ssr_color_mips.wesl");
        embedded_asset!(app, "../shaders/ssr.wesl");
        embedded_asset!(app, "../shaders/ssr_resolve.wesl");
        embedded_asset!(app, "../shaders/ssr_temporal.wesl");
        embedded_asset!(app, "../shaders/ssr_composite.wesl");
        embedded_asset!(app, "../shaders/ssgi.wesl");
        embedded_asset!(app, "../shaders/ssgi_denoise.wesl");
        embedded_asset!(app, "../shaders/ssgi_composite.wesl");
        embedded_asset!(app, "../shaders/taa_resolve.wesl");
        embedded_asset!(app, "../shaders/vsm_receiver_gen.wesl");
        embedded_asset!(app, "../shaders/brdf_lut.wesl");
        embedded_asset!(app, "../shaders/env_prefilter.wesl");
        embedded_asset!(app, "../shaders/exposure.wesl");
        embedded_asset!(app, "../shaders/bloom.wesl");
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
            .init_resource::<ResolveMotionHistory>()
            .insert_resource(ShadowAtlasConfig::new(
                DEFAULT_SHADOW_ATLAS_LAYERS,
                DEFAULT_SHADOW_ATLAS_RESOLUTION,
            ))
            .init_resource::<ExtractedShadows>()
            .init_resource::<PrismShadowSettings>()
            .init_resource::<ExtractedIblSource>()
            .init_resource::<EnvPrefilterBindGroups>()
            .init_resource::<VsmPrimaryLight>()
            .init_resource::<PrismVirtualShadowSettings>()
            .init_resource::<VsmReceiverBufferCache>()
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
                    // Nested to keep this RenderStartup tuple within Bevy's
                    // 20-element limit: prepass + horizon kernel + denoise.
                    (
                        init_gtao_prepass_pipeline,
                        init_gtao_kernel_pipeline,
                        init_gtao_denoise_pipeline,
                        init_gtao_temporal_pipeline,
                    ),
                    init_ssr_prepass_pipeline,
                    init_ssr_hzb_pipeline,
                    init_ssr_repack_pipeline
                        .after(init_gpu_resource::<crate::MaterialBindGroup>),
                    init_ssr_color_mips_pipeline,
                    init_ssr_trace_pipeline,
                    init_ssr_reconstruct_pipeline,
                    init_ssr_temporal_pipeline,
                    // Nested to keep this RenderStartup tuple within Bevy's
                    // 20-element limit.
                    // Nested with the SSGI trace + composite initializers to
                    // keep this RenderStartup tuple within Bevy's 20-element
                    // limit.
                    (
                        init_ssr_composite_pipeline,
                        init_taa_resolve_pipeline,
                        init_ssgi_trace_pipeline,
                        init_ssgi_denoise_pipeline,
                        init_ssgi_composite_pipeline,
                        init_exposure_histogram_pipeline,
                        init_exposure_average_pipeline,
                        init_bloom_pipelines,
                        init_vsm_receiver_gen_pipeline,
                    ),
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
                    // Visibility buffers plus the motion feed nested together
                    // to keep the `add_systems` tuple within Bevy's 20-element
                    // limit. Motion matrices + G-buffer must land in
                    // PrepareResources (before PrepareBindGroups) so the resolve
                    // bind group can bind every view's motion uniform, and after
                    // the visibility buffers so the motion-vector texture exists.
                    (
                        prepare_visibility_buffers.in_set(RenderSystems::PrepareResources),
                        prepare_resolve_motion
                            .after(prepare_visibility_buffers)
                            .in_set(RenderSystems::PrepareResources),
                    ),
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
                (
                    prepare_gtao_kernel_bind_groups
                        .after(prepare_gtao_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_gtao_denoise_bind_groups
                        .after(prepare_gtao_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_gtao_temporal_textures
                        .after(prepare_gtao_textures)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_gtao_temporal_bind_groups
                        .after(prepare_gtao_temporal_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_ssr_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_ssr_temporal_textures
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_ssr_prepass_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_hzb_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_repack_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_color_mips_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_trace_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_reconstruct_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_temporal_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssr_composite_bind_groups
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ssgi_textures
                        .after(prepare_ssr_textures)
                        .in_set(RenderSystems::PrepareResources),
                    // Nested to keep this Render tuple within Bevy's 20-element
                    // limit; all three still order after `prepare_ssgi_textures`.
                    (
                        prepare_ssgi_trace_bind_groups
                            .after(prepare_ssgi_textures)
                            .in_set(RenderSystems::PrepareBindGroups),
                        prepare_ssgi_denoise_bind_groups
                            .after(prepare_ssgi_textures)
                            .in_set(RenderSystems::PrepareBindGroups),
                        prepare_ssgi_composite_bind_groups
                            .after(prepare_ssgi_textures)
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
                    prepare_taa_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_taa_bind_groups
                        .after(prepare_taa_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // Auto-exposure: the persistent state + histogram buffers
                    // land in PrepareResources (after the visibility buffers so
                    // the `scene_color` they meter exists), then both exposure
                    // bind groups build in PrepareBindGroups off those buffers.
                    prepare_exposure_buffers
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_exposure_histogram_bind_groups
                        .after(prepare_exposure_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_exposure_average_bind_groups
                        .after(prepare_exposure_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // Bloom: the pyramid allocates in PrepareResources (after
                    // the visibility buffers so `scene_color` exists), then its
                    // bind groups build in PrepareBindGroups off those targets.
                    prepare_bloom_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_bloom_bind_groups
                        .after(prepare_bloom_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // Nested to keep this Render tuple within Bevy's 20-element
                    // limit: the VSM receiver-gen params + persistent receiver
                    // buffer land in PrepareResources (after the SSR depth
                    // prepass whose `scene_depth` the pass unprojects), then its
                    // group-0 bind group builds in PrepareBindGroups.
                    (
                        prepare_vsm_receiver_resources
                            .after(prepare_ssr_textures)
                            .in_set(RenderSystems::PrepareResources),
                        prepare_vsm_receiver_gen_bind_groups
                            .after(prepare_vsm_receiver_resources)
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
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
            .add_systems(
                ExtractSchedule,
                (extract_shadows, extract_ibl_source, extract_vsm_primary_light),
            );
        // Camera jitter must land before `prepare_view_uniforms` bakes the
        // projection; `PrepareViews` is ordered ahead of that `PrepareResources`
        // system, so a standalone registration keeps the ordering explicit
        // without inflating an existing tuple past Bevy's 20-element limit.
        render_app.add_systems(
            Render,
            prepare_taa_jitter.in_set(RenderSystems::PrepareViews),
        );
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            (
                shadow_depth_pass.before(visibility_raster_pass),
                visibility_raster_pass.before(bevy_core_pipeline::Core3dSystems::MainPass),
                dispatch_material_classification
                    .after(visibility_raster_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: the three GTAO passes chain prepass -> horizon kernel
                // -> spatial denoise, all feeding the shading resolve.
                (
                    gtao_prepass_pass
                        .after(visibility_raster_pass)
                        .before(dispatch_shading_resolve),
                    gtao_compute_pass
                        .after(gtao_prepass_pass)
                        .before(dispatch_shading_resolve),
                    gtao_denoise_pass
                        .after(gtao_compute_pass)
                        .before(dispatch_shading_resolve),
                    gtao_temporal_pass
                        .after(gtao_denoise_pass)
                        .before(dispatch_shading_resolve),
                ),
                ssr_prepass_pass
                    .after(visibility_raster_pass)
                    .before(dispatch_shading_resolve),
                ssr_hzb_pass
                    .after(ssr_prepass_pass)
                    .before(dispatch_shading_resolve),
                ssr_repack_pass
                    .after(ssr_prepass_pass)
                    .before(dispatch_shading_resolve),
                dfg_lut_precompute_pass.before(dispatch_shading_resolve),
                env_prefilter_precompute_pass.before(dispatch_shading_resolve),
                dispatch_shading_resolve
                    .after(dispatch_material_classification)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ssr_color_mips_pass
                    .after(dispatch_shading_resolve)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ssr_trace_pass
                    .after(ssr_color_mips_pass)
                    .after(ssr_hzb_pass)
                    .after(ssr_repack_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ssr_reconstruct_pass
                    .after(ssr_trace_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ssr_temporal_pass
                    .after(ssr_reconstruct_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: the SSR composite folds reflections into scene_color,
                // then TAA resolves that composited buffer before the main pass.
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: the SSR composite folds reflections into scene_color,
                // then the SSGI gather traces indirect diffuse off the same
                // rebuilt inputs and its composite substitutes that gather for
                // the resolve's flat ambient (reading the SSR-composited base),
                // before TAA resolves the fully composited buffer.
                (
                    ssr_composite_pass
                        .after(ssr_temporal_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    ssgi_trace_pass
                        .after(ssr_color_mips_pass)
                        .after(ssr_hzb_pass)
                        .after(ssr_repack_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    ssgi_denoise_pass
                        .after(ssgi_trace_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    ssgi_composite_pass
                        .after(ssr_composite_pass)
                        .after(ssgi_trace_pass)
                        .after(ssgi_denoise_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    taa_resolve_pass
                        .after(ssr_composite_pass)
                        .after(ssgi_composite_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ),
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: auto-exposure meters the fully composited HDR
                // scene_color (after SSR/SSGI composite and TAA resolve), then a
                // single-invocation resolve writes the eye-adaptation multiplier
                // the composite applies, all before the main pass.
                (
                    exposure_histogram_pass
                        .after(ssgi_composite_pass)
                        .after(taa_resolve_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    exposure_average_pass
                        .after(exposure_histogram_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Bloom scatters glow from the metered HDR scene_color after
                    // the exposure resolve (so metering saw the clean scene) and
                    // before the main pass composites it.
                    bloom_pass
                        .after(exposure_average_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ),
                composite_shading
                    .after(bevy_core_pipeline::Core3dSystems::MainPass)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
                transparent_forward_pass
                    .after(dispatch_shading_resolve)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                oit_composite
                    .after(composite_shading)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
                // Receiver generation reads the SSR geometry prepass depth and
                // fills the per-view receiver buffer before the main pass; a
                // later slice consumes it for page requests + sampling.
                vsm_receiver_gen_pass
                    .after(ssr_prepass_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
            ),
        );
    }
}
