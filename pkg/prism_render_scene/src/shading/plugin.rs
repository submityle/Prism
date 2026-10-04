use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_core_pipeline::schedule::camera_driver;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_pbr::MeshPipelineSystems;
use bevy_render::{
    init_gpu_resource,
    render_phase::AddRenderCommand,
    render_resource::SpecializedRenderPipelines,
    renderer::{RenderGraph, RenderGraphSystems},
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};

use super::{
    ao::{
        gtao_compute_pass, gtao_denoise_pass, gtao_prepass_pass, gtao_temporal_pass,
        init_gtao_denoise_pipeline, init_gtao_kernel_pipeline, init_gtao_prepass_pipeline,
        init_gtao_temporal_pipeline, prepare_gtao_denoise_bind_groups,
        prepare_gtao_kernel_bind_groups, prepare_gtao_prepass_bind_groups,
        prepare_gtao_temporal_bind_groups, prepare_gtao_temporal_textures, prepare_gtao_textures,
    },
    area_light::{
        extract_area_lights, init_area_light_ltc_lut, rebuild_area_light_buffers,
        write_area_light_buffers, AreaLightGpuBuffer, ExtractedAreaLights, PrismAreaLightSettings,
    },
    bloom::{bloom_pass, init_bloom_pipelines, prepare_bloom_bind_groups, prepare_bloom_textures},
    cas::{
        cas_pass, init_cas_pipeline, prepare_cas_bind_groups, prepare_cas_textures,
        PrismCasSettings,
    },
    chromatic_aberration::{
        chromatic_aberration_pass, init_chromatic_aberration_pipeline,
        prepare_chromatic_aberration_bind_groups, prepare_chromatic_aberration_textures,
        PrismChromaticAberrationSettings,
    },
    classification_gpu::{
        dispatch_material_classification, init_material_classification_pipeline,
        prepare_material_classification_bind_groups,
    },
    classification_readback::{
        collect_classification_readback, map_submitted_classification_readback,
        request_classification_readback, ClassificationDiagnosticsReadback,
    },
    color_grade::{
        color_grade_pass, init_color_grade_pipeline, prepare_color_grade_bind_groups,
        prepare_color_grade_textures, PrismColorGradeSettings,
    },
    composite::{
        composite_shading, init_shading_composite_pipeline, prepare_shading_composite_bind_groups,
        prepare_shading_composite_pipelines, ShadingCompositePipeline,
    },
    ddgi::{
        ddgi_composite_pass, ddgi_probe_update_pass, ddgi_sample_pass,
        init_ddgi_composite_pipeline, init_ddgi_pipeline, prepare_ddgi_bind_groups,
        prepare_ddgi_composite_bind_groups, prepare_ddgi_textures, PrismDdgiSettings,
    },
    dof::{
        dof_pass, init_dof_pipeline, prepare_dof_bind_groups, prepare_dof_textures,
        PrismDofSettings,
    },
    exposure::{
        exposure_average_pass, exposure_histogram_pass, init_exposure_average_pipeline,
        init_exposure_histogram_pipeline, prepare_exposure_average_bind_groups,
        prepare_exposure_buffers, prepare_exposure_histogram_bind_groups,
    },
    film_grain::{
        film_grain_pass, init_film_grain_pipeline, prepare_film_grain_bind_groups,
        prepare_film_grain_textures, PrismFilmGrainSettings,
    },
    gamut_map::{
        gamut_map_pass, init_gamut_map_pipeline, prepare_gamut_map_bind_groups,
        prepare_gamut_map_textures, PrismGamutMapSettings,
    },
    graph::shading_frame_graph,
    halftone::{
        halftone_pass, init_halftone_pipeline, prepare_halftone_bind_groups,
        prepare_halftone_textures, PrismHalftoneSettings,
    },
    hatching::{
        hatching_pass, init_hatching_pipeline, prepare_hatching_bind_groups,
        prepare_hatching_textures, PrismHatchingSettings,
    },
    ibl::{
        dfg_lut_precompute_pass, env_prefilter_precompute_pass, extract_ibl_source,
        init_brdf_lut_pipeline, init_dfg_lut_texture, init_env_prefilter_pipeline,
        init_prefiltered_env_map, prepare_dfg_lut_bind_group, prepare_env_prefilter_bind_groups,
        EnvPrefilterBindGroups, ExtractedIblSource,
    },
    kuwahara::{
        init_kuwahara_pipeline, kuwahara_pass, prepare_kuwahara_bind_groups,
        prepare_kuwahara_textures, PrismKuwaharaSettings,
    },
    lens_flare::{
        init_lens_flare_pipeline, lens_flare_pass, prepare_lens_flare_bind_groups,
        prepare_lens_flare_textures, PrismLensFlareSettings,
    },
    light_routing::{
        init_light_routing_pipeline, light_routing_cull_pass, prepare_light_routing_bind_groups,
        prepare_light_routing_buffers, PrismLightRoutingSettings,
    },
    motion_blur::{
        init_motion_blur_pipeline, motion_blur_pass, prepare_motion_blur_bind_groups,
        prepare_motion_blur_textures, PrismMotionBlurSettings,
    },
    ordered_dither::{
        init_ordered_dither_pipeline, ordered_dither_pass, prepare_ordered_dither_bind_groups,
        prepare_ordered_dither_textures, PrismOrderedDitherSettings,
    },
    outline::{
        init_outline_pipeline, outline_pass, prepare_outline_bind_groups, prepare_outline_textures,
        PrismOutlineSettings,
    },
    posterize::{
        init_posterize_pipeline, posterize_pass, prepare_posterize_bind_groups,
        prepare_posterize_textures, PrismPosterizeSettings,
    },
    raster::{
        init_visibility_raster, queue_visibility_raster, visibility_raster_pass,
        DrawVisibilityRaster, Visibility3d, VisibilityRasterPipeline,
    },
    resolve::{
        dispatch_shading_resolve, init_shading_resolve_pipeline, prepare_resolve_motion,
        prepare_shading_resolve_bind_groups, ResolveMotionHistory,
    },
    resources::{prepare_shading_buffers, prepare_visibility_buffers},
    runtime::{
        detect_shading_capabilities, prepare_shading_work, PrismShadingDiagnostics,
        PrismShadingSettings, ShadingFrameGraph,
    },
    shadow::{
        ensure_shadow_atlas, extract_shadows, init_shadow_depth_pipeline,
        prepare_shadow_bind_group, prepare_shadow_depth_uniform, queue_shadow_depth,
        rebuild_shadow_buffers, register_shadow_depth_shader, shadow_depth_pass,
        write_shadow_buffers, ExtractedShadows, PrismShadowSettings, ShadowAtlas,
        ShadowAtlasConfig, ShadowBindGroup, ShadowDepthDrawList, ShadowDepthPipeline,
        ShadowDepthViewOffsets, ShadowDepthViewUniform, ShadowGpuBuffers,
        DEFAULT_SHADOW_ATLAS_LAYERS, DEFAULT_SHADOW_ATLAS_RESOLUTION,
    },
    sky::multiscatter::{
        init_sky_multiscatter_lut, init_sky_multiscatter_pipeline,
        prepare_sky_multiscatter_bind_group, sky_multiscatter_lut_pass,
    },
    sky::sky_view::{
        init_sky_view_lut, init_sky_view_pipeline, prepare_sky_view_bind_group, sky_view_lut_pass,
    },
    sky::transmittance::{
        init_sky_transmittance_lut, init_sky_transmittance_pipeline,
        prepare_sky_transmittance_bind_group, sky_transmittance_lut_pass,
    },
    spec_denoise::{
        init_spec_denoise_history_clamp_pipeline, init_spec_denoise_reproject_pipeline,
        init_spec_denoise_spatial_pipeline, prepare_spec_denoise_bind_groups,
        prepare_spec_denoise_history_clamp_bind_groups, prepare_spec_denoise_reproject_bind_groups,
        prepare_spec_denoise_resources, prepare_spec_denoise_temporal_resources,
        spec_denoise_history_clamp_pass, spec_denoise_reproject_pass, spec_denoise_spatial_pass,
    },
    spec_gi::{
        init_spec_gi_composite_pipeline, init_spec_gi_reuse_pipeline,
        init_spec_gi_spatial_pipeline, prepare_spec_gi_composite_bind_groups,
        prepare_spec_gi_reuse_bind_groups, prepare_spec_gi_reuse_resources,
        prepare_spec_gi_spatial_bind_groups, spec_gi_composite_pass, spec_gi_reuse_pass,
        spec_gi_spatial_pass,
    },
    ssgi::{
        init_ssgi_composite_pipeline, init_ssgi_denoise_pipeline, init_ssgi_trace_pipeline,
        prepare_ssgi_composite_bind_groups, prepare_ssgi_denoise_bind_groups,
        prepare_ssgi_textures, prepare_ssgi_trace_bind_groups, ssgi_composite_pass,
        ssgi_denoise_pass, ssgi_trace_pass,
    },
    ssr::{
        init_ssr_color_mips_pipeline, init_ssr_composite_pipeline, init_ssr_hzb_pipeline,
        init_ssr_prepass_pipeline, init_ssr_reconstruct_pipeline, init_ssr_repack_pipeline,
        init_ssr_temporal_pipeline, init_ssr_trace_pipeline, prepare_ssr_color_mips_bind_groups,
        prepare_ssr_composite_bind_groups, prepare_ssr_hzb_bind_groups,
        prepare_ssr_prepass_bind_groups, prepare_ssr_reconstruct_bind_groups,
        prepare_ssr_repack_bind_groups, prepare_ssr_temporal_bind_groups,
        prepare_ssr_temporal_textures, prepare_ssr_textures, prepare_ssr_trace_bind_groups,
        ssr_color_mips_pass, ssr_composite_pass, ssr_hzb_pass, ssr_prepass_pass,
        ssr_reconstruct_pass, ssr_repack_pass, ssr_temporal_pass, ssr_trace_pass,
    },
    surface_cache::{
        init_surface_cache_composite_pipeline, init_surface_cache_pipeline,
        prepare_surface_cache_bind_groups, prepare_surface_cache_composite_bind_groups,
        prepare_surface_cache_resources, surface_cache_composite_pass, surface_cache_pass,
        PrismSurfaceCacheSettings, SurfaceCacheBuffers,
    },
    taa::{
        init_taa_resolve_pipeline, prepare_taa_bind_groups, prepare_taa_jitter,
        prepare_taa_textures, taa_resolve_pass,
    },
    tonemap::{
        init_tonemap_pipeline, prepare_tonemap_bind_groups, prepare_tonemap_textures, tonemap_pass,
        PrismTonemapSettings,
    },
    transparent::{
        init_oit_composite_pipeline, init_oit_forward_pipeline, oit_composite,
        prepare_oit_composite_bind_groups, prepare_oit_composite_pipelines, prepare_oit_targets,
        queue_transparent_oit, transparent_forward_pass, DrawTransparentOit, OitCompositePipeline,
        OitForwardPipeline, TransparentOit3d,
    },
    upscale::{
        init_upscale_pipeline, prepare_upscale_bind_groups, prepare_upscale_textures, upscale_pass,
    },
    vignette::{
        init_vignette_pipeline, prepare_vignette_bind_groups, prepare_vignette_textures,
        vignette_pass, PrismVignetteSettings,
    },
    virtual_shadow::{
        bridge_vsm_view_resources, collect_vsm_page_readback, extract_vsm_primary_light,
        init_vsm_caster_depth_pipeline, init_vsm_page_mark_pipeline,
        init_vsm_receiver_gen_pipeline, map_submitted_vsm_page_readback,
        prepare_vsm_caster_depth_targets, prepare_vsm_caster_depth_views,
        prepare_vsm_page_mark_bind_groups, prepare_vsm_page_requests, prepare_vsm_physical_atlas,
        prepare_vsm_receiver_gen_bind_groups, prepare_vsm_receiver_resources,
        queue_vsm_caster_depth, register_vsm_caster_depth_shader, request_vsm_page_readback,
        vsm_caster_depth_pass, vsm_mark_pages_pass, vsm_receiver_gen_pass,
        PrismVirtualShadowSettings, VirtualShadowMapDriver, VsmBridgeCache, VsmCasterDepthDrawList,
        VsmCasterDepthPipeline, VsmCasterDepthTargets, VsmCasterDepthViewUniform,
        VsmPageRequestBufferCache, VsmPageRequestReadback, VsmPageTableBufferCache,
        VsmPhysicalAtlasCache, VsmPrimaryLight, VsmReceiverBufferCache,
    },
    volumetric_clouds::{
        dispatch_volumetric_clouds, init_volumetric_cloud_pipelines,
        prepare_volumetric_cloud_domain, prepare_volumetric_cloud_domain_bind_groups,
        prepare_volumetric_cloud_view_bind_groups, prepare_volumetric_cloud_views,
        PrismVolumetricCloudsSettings, VolumetricCloudViewCache,
    },
    volumetrics::{
        init_volumetrics_pipeline, prepare_volumetrics_bind_groups, prepare_volumetrics_resources,
        volumetrics_pass, PrismVolumetricsSettings, VolumetricsTextureCache,
    },
    world_restir::{
        init_world_restir_pipeline, init_world_restir_resolve_pipeline,
        init_world_restir_visible_points_pipeline, prepare_world_restir_bind_groups,
        prepare_world_restir_lights, prepare_world_restir_reservoirs, prepare_world_restir_resolve,
        prepare_world_restir_resolve_bind_groups, prepare_world_restir_visible_points,
        prepare_world_restir_visible_points_bind_groups, world_restir_fill_pass,
        world_restir_inject_pass, world_restir_resolve_pass, world_restir_seed_pass,
        world_restir_visible_points_pass, PrismWorldRestirSettings, WorldRestirLights,
    },
    world_space_gi::{
        init_world_space_gi_composite_pipeline, init_world_space_gi_pipeline,
        prepare_world_space_gi_bind_groups, prepare_world_space_gi_composite_bind_groups,
        prepare_world_space_gi_textures, world_space_gi_composite_pass, world_space_gi_pass,
        PrismWorldSpaceGiSettings,
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
        embedded_asset!(app, "../shaders/spec_gi_reservoir.wesl");
        embedded_asset!(app, "../shaders/spec_gi_reuse.wesl");
        embedded_asset!(app, "../shaders/spec_gi_spatial.wesl");
        embedded_asset!(app, "../shaders/spec_gi_composite.wesl");
        embedded_asset!(app, "../shaders/spec_denoise_spatial.wesl");
        embedded_asset!(app, "../shaders/spec_denoise_reproject.wesl");
        embedded_asset!(app, "../shaders/spec_denoise_history_clamp.wesl");
        embedded_asset!(app, "../shaders/ssgi.wesl");
        embedded_asset!(app, "../shaders/sky_multiscatter_lut.wesl");
        embedded_asset!(app, "../shaders/sky_transmittance_lut.wesl");
        embedded_asset!(app, "../shaders/sky_view_lut.wesl");
        embedded_asset!(app, "../shaders/ssgi_denoise.wesl");
        embedded_asset!(app, "../shaders/ssgi_composite.wesl");
        embedded_asset!(app, "../shaders/ddgi_sample.wesl");
        embedded_asset!(app, "../shaders/ddgi_probe_update.wesl");
        embedded_asset!(app, "../shaders/ddgi_composite.wesl");
        embedded_asset!(app, "../shaders/world_space_gi_probe_update.wesl");
        embedded_asset!(app, "../shaders/world_space_gi_resolve.wesl");
        embedded_asset!(app, "../shaders/world_space_gi_composite.wesl");
        embedded_asset!(app, "../shaders/surface_cache_alloc.wesl");
        embedded_asset!(app, "../shaders/surface_cache_update.wesl");
        embedded_asset!(app, "../shaders/surface_cache_spatial_filter.wesl");
        embedded_asset!(app, "../shaders/surface_cache_coverage.wesl");
        embedded_asset!(app, "../shaders/surface_cache_composite.wesl");
        embedded_asset!(app, "../shaders/area_light_ltc.wesl");
        embedded_asset!(app, "../shaders/light_routing.wesl");
        embedded_asset!(app, "../shaders/taa_resolve.wesl");
        embedded_asset!(app, "../shaders/vsm_receiver_gen.wesl");
        embedded_asset!(app, "../shaders/vsm_page_mark.wesl");
        embedded_asset!(app, "../shaders/motion_blur.wesl");
        embedded_asset!(app, "../shaders/volumetrics.wesl");
        embedded_asset!(app, "../shaders/volumetric_clouds.wesl");
        embedded_asset!(app, "../shaders/dof.wesl");
        embedded_asset!(app, "../shaders/chromatic_aberration.wesl");
        embedded_asset!(app, "../shaders/vignette.wesl");
        embedded_asset!(app, "../shaders/color_grade.wesl");
        embedded_asset!(app, "../shaders/film_grain.wesl");
        embedded_asset!(app, "../shaders/cas.wesl");
        embedded_asset!(app, "../shaders/posterize.wesl");
        embedded_asset!(app, "../shaders/gamut_map.wesl");
        embedded_asset!(app, "../shaders/tonemap.wesl");
        embedded_asset!(app, "../shaders/ordered_dither.wesl");
        embedded_asset!(app, "../shaders/lens_flare.wesl");
        embedded_asset!(app, "../shaders/outline.wesl");
        embedded_asset!(app, "../shaders/kuwahara.wesl");
        embedded_asset!(app, "../shaders/hatching.wesl");
        embedded_asset!(app, "../shaders/halftone.wesl");
        embedded_asset!(app, "../shaders/brdf_lut.wesl");
        embedded_asset!(app, "../shaders/env_prefilter.wesl");
        embedded_asset!(app, "../shaders/exposure.wesl");
        embedded_asset!(app, "../shaders/bloom.wesl");
        embedded_asset!(app, "../shaders/composite.wesl");
        embedded_asset!(app, "../shaders/oit.wesl");
        embedded_asset!(app, "../shaders/transparent.wesl");
        embedded_asset!(app, "../shaders/world_restir_seed.wesl");
        embedded_asset!(app, "../shaders/world_restir_fill.wesl");
        embedded_asset!(app, "../shaders/world_restir_inject.wesl");
        embedded_asset!(app, "../shaders/world_restir_visible_points.wesl");
        embedded_asset!(app, "../shaders/world_restir_resolve.wesl");
        register_shadow_depth_shader(app);
        register_vsm_caster_depth_shader(app);
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
            .init_resource::<ClassificationDiagnosticsReadback>()
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
            .init_resource::<VsmPageRequestBufferCache>()
            .init_resource::<VsmPhysicalAtlasCache>()
            .init_resource::<VirtualShadowMapDriver>()
            .init_resource::<VsmPageTableBufferCache>()
            .init_resource::<VsmPageRequestReadback>()
            .init_resource::<VsmBridgeCache>()
            .init_resource::<VsmCasterDepthDrawList>()
            .init_resource::<VsmCasterDepthTargets>()
            .init_resource::<bevy_render::render_resource::SpecializedMeshPipelines<VsmCasterDepthPipeline>>()
            .init_resource::<VsmCasterDepthViewUniform>()
            .init_resource::<PrismMotionBlurSettings>()
            .init_resource::<PrismVolumetricsSettings>()
            .init_resource::<VolumetricsTextureCache>()
            .init_resource::<PrismVolumetricCloudsSettings>()
            .init_resource::<VolumetricCloudViewCache>()
            .init_resource::<PrismDofSettings>()
            .init_resource::<PrismChromaticAberrationSettings>()
            .init_resource::<PrismVignetteSettings>()
            .init_resource::<PrismColorGradeSettings>()
            .init_resource::<PrismFilmGrainSettings>()
            .init_resource::<PrismCasSettings>()
            .init_resource::<PrismPosterizeSettings>()
            .init_resource::<PrismGamutMapSettings>()
            .init_resource::<PrismTonemapSettings>()
            .init_resource::<PrismOrderedDitherSettings>()
            .init_resource::<PrismLensFlareSettings>()
            .init_resource::<PrismOutlineSettings>()
            .init_resource::<PrismKuwaharaSettings>()
            .init_resource::<PrismHatchingSettings>()
            .init_resource::<PrismHalftoneSettings>()
            .init_resource::<PrismWorldRestirSettings>()
            .init_resource::<WorldRestirLights>()
            .init_resource::<PrismWorldSpaceGiSettings>()
            .init_resource::<PrismDdgiSettings>()
            .init_resource::<PrismSurfaceCacheSettings>()
            .init_resource::<SurfaceCacheBuffers>()
            .init_resource::<PrismLightRoutingSettings>()
            .init_resource::<PrismAreaLightSettings>()
            .init_resource::<ExtractedAreaLights>()
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
                        init_vsm_page_mark_pipeline,
                        init_vsm_caster_depth_pipeline,
                        init_motion_blur_pipeline,
                        init_volumetrics_pipeline,
                        init_dof_pipeline,
                        init_chromatic_aberration_pipeline,
                        init_vignette_pipeline,
                        init_color_grade_pipeline,
                        init_film_grain_pipeline,
                        init_world_space_gi_pipeline,
                        init_world_space_gi_composite_pipeline,
                    ),
                    // Nested to keep this RenderStartup tuple within Bevy's
                    // 20-element limit: the nine additional post-process
                    // (display-referred + stylized NPR) pipeline initializers.
                    (
                        init_cas_pipeline,
                        init_posterize_pipeline,
                        init_gamut_map_pipeline,
                        init_tonemap_pipeline,
                        init_ordered_dither_pipeline,
                        init_lens_flare_pipeline,
                        init_outline_pipeline,
                        init_kuwahara_pipeline,
                        init_hatching_pipeline,
                        init_halftone_pipeline,
                        init_upscale_pipeline,
                        init_surface_cache_pipeline,
                        init_surface_cache_composite_pipeline,
                        init_ddgi_pipeline,
                        init_ddgi_composite_pipeline,
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
            // Volumetric-cloud compute pipelines. Kept in their own
            // `add_systems` call so the eight-kernel initializer never forces the
            // already-full RenderStartup tuple past Bevy's 20-element limit.
            .add_systems(RenderStartup, (init_sky_multiscatter_lut, init_sky_multiscatter_pipeline))
            // Physical-sky LUT chain inits: transmittance and sky-view
            // allocate their LUTs + queue their compute pipelines. Own
            // `add_systems` calls so the already-full RenderStartup tuples
            // stay within Bevy's 20-element limit.
            .add_systems(RenderStartup, (init_sky_transmittance_lut, init_sky_transmittance_pipeline))
            .add_systems(RenderStartup, (init_sky_view_lut, init_sky_view_pipeline))
            .add_systems(RenderStartup, init_volumetric_cloud_pipelines)
            // Light-routing (Lighting Channels) cull pipeline. Its own
            // `add_systems` call so the already-full RenderStartup tuples stay
            // within Bevy's 20-element limit.
            .add_systems(RenderStartup, init_light_routing_pipeline)
            // Area-light `LTC` `LUT` bake + upload (opt-in, Heitz et al. 2016).
            // Own `add_systems` call so the already-full RenderStartup tuples
            // stay within Bevy's 20-element limit; gated on the settings
            // `enabled` flag, so disabled it bakes and uploads nothing.
            .add_systems(RenderStartup, init_area_light_ltc_lut)
            // Glossy-specular ReSTIR reuse compute pipeline (spec_gi).
            // Own `add_systems` call so the already-full RenderStartup
            // tuples stay within Bevy's 20-element limit; the dispatch it
            // feeds is gated on `enable_spec_gi` + the SSR/visibility
            // prerequisites in resource prep.
            .add_systems(
                RenderStartup,
                (
                    init_spec_gi_reuse_pipeline,
                    init_spec_gi_spatial_pipeline,
                    init_spec_gi_composite_pipeline,
                    init_spec_denoise_spatial_pipeline,
                    init_spec_denoise_reproject_pipeline,
                    init_spec_denoise_history_clamp_pipeline,
                ),
            )
            // Volumetric-cloud domain + per-view resource and bind-group
            // preparation. Self-contained (its own resident textures + view
            // cache, gated on the opt-in settings), so it lives in its own
            // `add_systems` call rather than crowding an already-full Render
            // tuple. The domain textures allocate first, the per-view low-res
            // targets after (they read the domain's frame parity), then the
            // domain bind groups and finally the per-view bind groups.
            .add_systems(
                Render,
                (
                    prepare_volumetric_cloud_domain.in_set(RenderSystems::PrepareResources),
                    prepare_volumetric_cloud_views
                        .after(prepare_volumetric_cloud_domain)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_volumetric_cloud_domain_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_volumetric_cloud_view_bind_groups
                        .after(prepare_volumetric_cloud_domain_bind_groups)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            // Physical-sky LUT bake chain. The sky-view LUT samples both the
            // transmittance and multiple-scattering LUTs, so it must dispatch
            // after both are baked: transmittance -> multiscatter -> sky-view,
            // all before the main pass that consumes the sky-view LUT.
            .add_systems(
                Render,
                (
                    sky_transmittance_lut_pass
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    sky_multiscatter_lut_pass
                        .after(sky_transmittance_lut_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    sky_view_lut_pass
                        .after(sky_multiscatter_lut_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
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
                    // Nested to keep this Render tuple within Bevy's 20-element
                    // limit: both shadow draw-list builders run in QueueMeshes.
                    (
                        queue_shadow_depth.in_set(RenderSystems::QueueMeshes),
                        queue_vsm_caster_depth.in_set(RenderSystems::QueueMeshes),
                    ),
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
                    // Glossy-specular ReSTIR reuse prep (spec_gi). Nested
                    // to keep this Render tuple within Bevy's 20-element
                    // limit. The resident ping-pong reservoir buffers +
                    // resolved target flip in PrepareResources (after the
                    // SSR textures that supply the depth/normal/candidate
                    // inputs); the per-view group binds them in
                    // PrepareBindGroups after that flip.
                    (
                        prepare_spec_gi_reuse_resources
                            .after(prepare_ssr_textures)
                            .in_set(RenderSystems::PrepareResources),
                        prepare_spec_gi_reuse_bind_groups
                            .after(prepare_spec_gi_reuse_resources)
                            .in_set(RenderSystems::PrepareBindGroups),
                        // The spatial reuse group binds this frame's post-temporal
                        // reservoir table (the reuse pass's `dst`) read-only plus
                        // the SSR reads and the resolved write; it orders after the
                        // reuse group so the bound `dst` agrees with this frame's
                        // ping-pong flip.
                        prepare_spec_gi_spatial_bind_groups
                            .after(prepare_spec_gi_reuse_bind_groups)
                            .in_set(RenderSystems::PrepareBindGroups),
                        // The spatial denoiser's filtered target allocates right
                        // after the reuse resolve it reads (same gate), and its
                        // per-view bind group binds after the reuse group so it
                        // sees this frame's resolve.
                        prepare_spec_denoise_resources
                            .after(prepare_spec_gi_reuse_resources)
                            .in_set(RenderSystems::PrepareResources),
                        // The temporal accumulator's persistent ping-pong planes
                        // allocate alongside the spatial target (same gate),
                        // right after the reuse resolve they clamp against; the
                        // cache is keyed by `RetainedViewEntity` so it survives
                        // across frames.
                        prepare_spec_denoise_temporal_resources
                            .after(prepare_spec_gi_reuse_resources)
                            .in_set(RenderSystems::PrepareResources),
                        // Temporal bind groups: reproject binds this frame's SSR
                        // reads + last frame's history, then history-clamp binds
                        // the reprojected planes + the reuse resolve. Both order
                        // after the temporal planes allocate; clamp orders after
                        // reproject so its reprojected inputs are this frame's.
                        prepare_spec_denoise_reproject_bind_groups
                            .after(prepare_spec_denoise_temporal_resources)
                            .after(prepare_spec_gi_reuse_bind_groups)
                            .in_set(RenderSystems::PrepareBindGroups),
                        prepare_spec_denoise_history_clamp_bind_groups
                            .after(prepare_spec_denoise_reproject_bind_groups)
                            .after(prepare_spec_gi_reuse_bind_groups)
                            .in_set(RenderSystems::PrepareBindGroups),
                        // The spatial filter binds the temporal denoised plane
                        // (when resident) instead of the raw resolve, so it must
                        // order after the temporal planes allocate as well.
                        prepare_spec_denoise_bind_groups
                            .after(prepare_spec_denoise_resources)
                            .after(prepare_spec_denoise_temporal_resources)
                            .after(prepare_spec_gi_reuse_bind_groups)
                            .in_set(RenderSystems::PrepareBindGroups),
                        // The energy-conserving composite's per-view group binds
                        // the denoised specular target (falling back to the raw
                        // reuse resolve when the denoiser is disabled) plus the
                        // colour-pyramid base copy and IBL specular; it folds the
                        // glossy reflection into scene_color in the Core3d node
                        // below under `enable_spec_gi`. Orders after the denoise
                        // resources so the filtered view it may bind exists.
                        prepare_spec_gi_composite_bind_groups
                            .after(prepare_spec_gi_reuse_resources)
                            .after(prepare_spec_denoise_resources)
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
                    // Nested to keep this Render tuple within Bevy's 20-element
                    // limit: TAA resolve targets/bind groups plus the temporal
                    // upscale (native-resolution TAAU + RCAS). The upscale
                    // reconstruction reads the resolved `scene_color` and the
                    // SSR device depth, so it orders after both the visibility
                    // buffers and the SSR textures; the per-view `ViewUpscale`
                    // it inserts in PrepareResources is visible to the composite
                    // bind group (PrepareBindGroups), which then composites
                    // `upscale_out` in place of the raw scene colour.
                    (
                        prepare_taa_textures
                            .after(prepare_visibility_buffers)
                            .in_set(RenderSystems::PrepareResources),
                        prepare_taa_bind_groups
                            .after(prepare_taa_textures)
                            .in_set(RenderSystems::PrepareBindGroups),
                        prepare_upscale_textures
                            .after(prepare_visibility_buffers)
                            .after(prepare_ssr_textures)
                            .in_set(RenderSystems::PrepareResources),
                        prepare_upscale_bind_groups
                            .after(prepare_upscale_textures)
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
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
                        // Page-mark: request-bitmap + immediate block build in
                        // PrepareResources after the receivers exist, then the
                        // group-0 bind group in PrepareBindGroups.
                        prepare_vsm_page_requests
                            .after(prepare_vsm_receiver_resources)
                            .in_set(RenderSystems::PrepareResources),
                        prepare_vsm_page_mark_bind_groups
                            .after(prepare_vsm_page_requests)
                            .in_set(RenderSystems::PrepareBindGroups),
                        // Physical page atlas: the resident-page depth texture
                        // the raster fill writes and vsm_sample.wesl reads.
                        // Needs the per-view ViewVsmReceivers marker resources
                        // produces, so it runs after them in PrepareResources.
                        prepare_vsm_physical_atlas
                            .after(prepare_vsm_receiver_resources)
                            .in_set(RenderSystems::PrepareResources),
                        // Caster-depth targets size off vsm_settings + device
                        // (no per-view input), so they build in PrepareResources;
                        // the per-view caster views then build in
                        // PrepareBindGroups off the bridge's ViewVsmCasterPages.
                        prepare_vsm_caster_depth_targets
                            .in_set(RenderSystems::PrepareResources),
                        prepare_vsm_caster_depth_views
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
            // Motion blur + froxel fog resource/bind-group preparation. Kept in
            // their own `add_systems` call so neither is forced into an already
            // full Render tuple past Bevy's 20-element limit. Motion blur reads
            // the resolved velocity G-buffer + SSR depth, so its textures order
            // after both; fog owns its per-view volumes and has no cross-input.
            .add_systems(
                Render,
                (
                    prepare_motion_blur_textures
                        .after(prepare_ssr_textures)
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_motion_blur_bind_groups
                        .after(prepare_motion_blur_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_volumetrics_resources.in_set(RenderSystems::PrepareResources),
                    prepare_volumetrics_bind_groups
                        .after(prepare_volumetrics_resources)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // DoF reads the SSR geometry-prepass depth and the resolved
                    // scene_color, exactly like motion blur, so it allocates and
                    // binds after both are resident.
                    prepare_dof_textures
                        .after(prepare_ssr_textures)
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_dof_bind_groups
                        .after(prepare_dof_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // Chromatic aberration and vignette are pure scene_color
                    // post-effects; both allocate after the resident set is
                    // known, mirroring the DoF preparation ordering.
                    prepare_chromatic_aberration_textures
                        .after(prepare_ssr_textures)
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_chromatic_aberration_bind_groups
                        .after(prepare_chromatic_aberration_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_vignette_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_vignette_bind_groups
                        .after(prepare_vignette_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // Colour grade then film grain close the pre-MainPass
                    // scene_color chain; both are visibility-only allocations.
                    prepare_color_grade_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_color_grade_bind_groups
                        .after(prepare_color_grade_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_film_grain_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_film_grain_bind_groups
                        .after(prepare_film_grain_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_sky_multiscatter_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_sky_transmittance_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_sky_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            // The nine additional post-process subsystems (display-referred
            // finishing + stylized NPR). Kept in their own `add_systems` call so
            // the eighteen prepare systems stay within Bevy's 20-element tuple
            // limit. Each allocates its scene_color-sized target after the
            // resident visibility set is known; outline additionally reads the
            // SSR geometry-prepass depth + normal G-buffer, so it also orders
            // after prepare_ssr_textures.
            .add_systems(
                Render,
                (
                    prepare_cas_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_cas_bind_groups
                        .after(prepare_cas_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_posterize_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_posterize_bind_groups
                        .after(prepare_posterize_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_gamut_map_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_gamut_map_bind_groups
                        .after(prepare_gamut_map_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_tonemap_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_tonemap_bind_groups
                        .after(prepare_tonemap_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ordered_dither_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_ordered_dither_bind_groups
                        .after(prepare_ordered_dither_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_lens_flare_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_lens_flare_bind_groups
                        .after(prepare_lens_flare_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_outline_textures
                        .after(prepare_ssr_textures)
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_outline_bind_groups
                        .after(prepare_outline_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_kuwahara_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_kuwahara_bind_groups
                        .after(prepare_kuwahara_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_hatching_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_hatching_bind_groups
                        .after(prepare_hatching_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_halftone_textures
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_halftone_bind_groups
                        .after(prepare_halftone_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_world_space_gi_textures
                        .after(prepare_ssr_textures)
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_world_space_gi_bind_groups
                        .after(prepare_world_space_gi_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_world_space_gi_composite_bind_groups
                        .after(prepare_world_space_gi_bind_groups)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // DDGI (dynamic diffuse GI irradiance volume): per-view
                    // octahedral probe atlases + GI export are allocated off
                    // the SSR prepass + visibility buffer, then the sample pass
                    // bind group wires them for the same-frame dispatch.
                    prepare_ddgi_textures
                        .after(prepare_ssr_textures)
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_ddgi_bind_groups
                        .after(prepare_ddgi_textures)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_ddgi_composite_bind_groups
                        .after(prepare_ddgi_bind_groups)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            // Surface cache (persistent surfel radiance cache): the surfel
            // atlas buffers live in a RetainedViewEntity-keyed resource that
            // survives the per-frame entity rebuild; the per-view scratch
            // textures are allocated here, then the bind groups wire the
            // persistent buffers + prepass textures for the same-frame passes.
            .add_systems(
                Render,
                (
                    prepare_surface_cache_resources
                        .after(prepare_ssr_textures)
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_surface_cache_bind_groups
                        .after(prepare_surface_cache_resources)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_surface_cache_composite_bind_groups
                        .after(prepare_surface_cache_bind_groups)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            // Light-routing (Lighting Channels) per-view buffers + cull bind
            // group. `prepare_light_routing_buffers` allocates/uploads in
            // `PrepareResources` (so it runs before the resolve's own
            // `PrepareBindGroups` bind-group build reads `ViewLightRouting`),
            // then `prepare_light_routing_bind_groups` builds the cull's group.
            // Both no-op unless the opt-in subsystem is enabled.
            .add_systems(
                Render,
                (
                    prepare_light_routing_buffers.in_set(RenderSystems::PrepareResources),
                    prepare_light_routing_bind_groups
                        .after(prepare_light_routing_buffers)
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
        // World-space ReSTIR DI fill. Its RenderStartup pipeline init, the
        // per-view resident ping-pong reservoir tables + fill bind group, and
        // the Core3d fill dispatch each live in their own `add_systems` call so
        // the already-full primary tuples stay within Bevy's 20-element limit.
        // Opt-in: a no-op unless `PrismWorldRestirSettings::enabled`, so the
        // default renderer allocates and dispatches nothing.
        render_app.add_systems(RenderStartup, init_world_restir_pipeline);
        render_app.add_systems(RenderStartup, init_world_restir_visible_points_pipeline);
        render_app.add_systems(RenderStartup, init_world_restir_resolve_pipeline);
        // Area-light `LTC` render-world systems. The `GpuAreaLight` storage
        // buffer, the per-frame extract from the main world, and the
        // rebuild/upload pair each live in their own `add_systems` call so the
        // already-full primary tuples stay within Bevy's 20-element limit. The
        // buffer is always resident (padded with a degenerate record when
        // empty) so the resolve bind group's group 7 can bind it every frame;
        // the baked `LUT` is only present when the opt-in subsystem is enabled.
        render_app.add_systems(RenderStartup, init_gpu_resource::<AreaLightGpuBuffer>);
        render_app.add_systems(ExtractSchedule, extract_area_lights);
        render_app.add_systems(
            Render,
            (
                rebuild_area_light_buffers.in_set(RenderSystems::PrepareResources),
                write_area_light_buffers.in_set(RenderSystems::PrepareResourcesFlush),
            ),
        );
        render_app.add_systems(
            Render,
            (
                prepare_world_restir_reservoirs.in_set(RenderSystems::PrepareResources),
                // The seed pass's candidate light list is a world-space table
                // producer like the reservoirs: repacked from the extracted
                // lights in PrepareResources, independent of the ping-pong flip.
                prepare_world_restir_lights.in_set(RenderSystems::PrepareResources),
                // The visible-point list is sized to the SSR prepass tile grid
                // in PrepareResources; its producer group and the inject group
                // both read it in PrepareBindGroups.
                prepare_world_restir_visible_points.in_set(RenderSystems::PrepareResources),
                prepare_world_restir_visible_points_bind_groups
                    .after(prepare_world_restir_visible_points)
                    .in_set(RenderSystems::PrepareBindGroups),
                prepare_world_restir_bind_groups
                    .after(prepare_world_restir_reservoirs)
                    .after(prepare_world_restir_visible_points)
                    .in_set(RenderSystems::PrepareBindGroups),
                // The resolve export is sized to the camera viewport in
                // PrepareResources; its screen-space consumer group reads the
                // SSR prepass views + the finalised reservoir table in
                // PrepareBindGroups.
                prepare_world_restir_resolve.in_set(RenderSystems::PrepareResources),
                prepare_world_restir_resolve_bind_groups
                    .after(prepare_world_restir_resolve)
                    .after(prepare_world_restir_reservoirs)
                    .in_set(RenderSystems::PrepareBindGroups),
            ),
        );
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            // World-space ReSTIR runs producer -> inject -> seed -> fill each
            // frame: the producer turns the SSR prepass into a visible-point
            // list, inject open-address claims one SHARC cell per point and
            // writes the slot geometry, seed streams light candidates through
            // each occupied slot's RIS reservoir, and fill streams GRIS spatial
            // reuse over the seeded cells. All four touch no screen targets, so
            // the chain only has to finish before the main pass samples the
            // resident table.
            (
                world_restir_visible_points_pass.before(world_restir_inject_pass),
                world_restir_inject_pass.before(world_restir_seed_pass),
                world_restir_seed_pass.before(world_restir_fill_pass),
                world_restir_fill_pass.before(world_restir_resolve_pass),
                // The resolve pass consumes the finalised reservoir table and
                // the SSR prepass, writing the direct-illumination export a
                // downstream composite folds into scene_color, so it runs last
                // in the chain and still finishes before the main pass.
                world_restir_resolve_pass.before(bevy_core_pipeline::Core3dSystems::MainPass),
            ),
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
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: the IBL precomputes and the light-routing cull all
                // feed the shading resolve and so must run before it.
                (
                    dfg_lut_precompute_pass.before(dispatch_shading_resolve),
                    env_prefilter_precompute_pass.before(dispatch_shading_resolve),
                    // Light-routing (Lighting Channels) cull refines the
                    // per-word punctual-light visibility mask the resolve's
                    // channel gate reads, so it must run before the resolve
                    // dispatch. No-op unless the opt-in subsystem is enabled.
                    light_routing_cull_pass.before(dispatch_shading_resolve),
                ),
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
                    // Glossy-specular ReSTIR reuse: reconstructs each
                    // pixel's glossy point off the SSR prepass depth +
                    // repacked `normal_roughness`, streams the SSR trace's
                    // raw screen-space candidate radiance into a reservoir
                    // and temporally merges the same-pixel prior, writing
                    // the resolved specular + confidence target the denoise
                    // / energy-conserving composite consume. Orders after
                    // the SSR trace + repack that fill its inputs; no-op
                    // unless `enable_spec_gi` and the SSR/visibility gate
                    // held in resource prep.
                    spec_gi_reuse_pass
                        .after(ssr_trace_pass)
                        .after(ssr_repack_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Glossy-specular ReSTIR composite (UE option C): folds
                    // the reuse pass's resolved specular back over scene_color,
                    // substituting the screen-space glossy reflection for the
                    // IBL specular under reservoir confidence (falling back to
                    // IBL on a miss). Owns the specular slot whenever
                    // `enable_spec_gi` holds; `ssr_composite_pass` early-returns
                    // under the same gate so `env_specular` is swapped exactly
                    // once. Reads the colour-pyramid base copy (built by
                    // `ssr_color_mips_pass`) and the resolved reuse target, so
                    // it orders after both; the SSR composite ordering keeps the
                    // two mutually-exclusive scene_color writers serialised.
                    // Nested with the composite to keep this Core3d tuple
                    // within Bevy's 20-element limit. The spatial denoiser runs
                    // between the reuse resolve it filters and the composite that
                    // consumes its filtered output: one screen-space dispatch
                    // (ReBLUR-style anisotropic, contact-hardened, edge-aware
                    // specular blur). No-op unless `enable_spec_gi` + the
                    // SSR/visibility gate held in resource prep.
                    (
                        // Spatial reuse: pools a frame-jittered disc of neighbour
                        // reservoirs from the completed post-temporal table onto
                        // each pixel's GGX lobe and overwrites the resolved target
                        // with the lower-variance estimate (reservoir history left
                        // pure). Runs after the reuse resolve it refines and before
                        // the denoise/composite that consume the pooled resolve.
                        spec_gi_spatial_pass
                            .after(spec_gi_reuse_pass)
                            .before(bevy_core_pipeline::Core3dSystems::MainPass),
                        // Temporal reproject: stages this frame's SSR depth for
                        // next frame and reprojects last frame's converged
                        // specular history. Runs right after the spatially pooled
                        // resolve it fuses with the history.
                        spec_denoise_reproject_pass
                            .after(spec_gi_reuse_pass)
                            .after(spec_gi_spatial_pass)
                            .before(bevy_core_pipeline::Core3dSystems::MainPass),
                        // History-clamp: fuses the reprojected history with this
                        // frame's noisy resolve under an AABB colour clamp and
                        // writes the `denoised` plane the spatial pass filters.
                        spec_denoise_history_clamp_pass
                            .after(spec_denoise_reproject_pass)
                            .before(bevy_core_pipeline::Core3dSystems::MainPass),
                        // Spatial filter: now cleans the temporally accumulated
                        // (anti-ghosted) specular rather than the raw resolve, so
                        // it orders after the history-clamp pass.
                        spec_denoise_spatial_pass
                            .after(spec_denoise_history_clamp_pass)
                            .before(bevy_core_pipeline::Core3dSystems::MainPass),
                        spec_gi_composite_pass
                            .after(spec_gi_reuse_pass)
                            .after(spec_gi_spatial_pass)
                            .after(spec_denoise_spatial_pass)
                            .after(ssr_color_mips_pass)
                            .after(ssr_composite_pass)
                            .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    ),
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
                        .after(spec_gi_composite_pass)
                        .after(ssgi_trace_pass)
                        .after(ssgi_denoise_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // World-space GI gathers screen probes off the fully
                    // SSR/SSGI-composited scene_color, then its own composite
                    // folds the indirect diffuse back in (energy-conserving
                    // ambient substitution) before TAA resolves the buffer.
                    world_space_gi_pass
                        .after(ssgi_composite_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    world_space_gi_composite_pass
                        .after(world_space_gi_pass)
                        .after(ssgi_composite_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // DDGI resolves the persistent octahedral probe field.
                    // Probe-update populates the octahedral irradiance + depth
                    // atlases this frame (one workgroup per probe, tracing the
                    // screen-space G-buffer); it must run before the sample
                    // pass reads those atlases. The sample pass reconstructs
                    // each pixel's probe gather into the GI export off the SSR
                    // prepass depth/normal, then the energy-conserving composite
                    // folds that export back over scene_color (UE option C:
                    // DDGI replaces the IBL diffuse ambient under confidence,
                    // falling back to IBL on a miss) before the surface cache
                    // gather and TAA.
                    ddgi_probe_update_pass
                        .after(world_space_gi_composite_pass)
                        .before(ddgi_sample_pass),
                    ddgi_sample_pass
                        .after(world_space_gi_composite_pass)
                        .before(ddgi_composite_pass),
                    ddgi_composite_pass
                        .after(ddgi_sample_pass)
                        .after(world_space_gi_composite_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Surface cache gathers its persistent surfels off the fully
                    // SSR/SSGI/world-space-GI/DDGI-composited scene_color, then
                    // its own composite folds the surfel diffuse back in
                    // (energy-conserving ambient substitution) before TAA.
                    surface_cache_pass
                        .after(world_space_gi_composite_pass)
                        .after(ddgi_composite_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    surface_cache_composite_pass
                        .after(surface_cache_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    taa_resolve_pass
                        .after(ssr_composite_pass)
                        .after(spec_gi_composite_pass)
                        .after(ssgi_composite_pass)
                        .after(world_space_gi_composite_pass)
                        .after(ddgi_composite_pass)
                        .after(surface_cache_composite_pass)
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
                    // Motion blur reconstructs the shutter streak from the
                    // velocity G-buffer after bloom has scattered its glow,
                    // still before the main pass composites.
                    motion_blur_pass
                        .after(bloom_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // DoF is the final pre-MainPass scene_color writer: it
                    // defocuses the fogged, motion-blurred HDR image (fog applies
                    // after motion blur, DoF after fog) before the composite
                    // reads scene_color. Serialised after volumetrics so the
                    // three scene_color copy-backs never race.
                    dof_pass
                        .after(volumetrics_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Chromatic aberration then vignette close the pre-MainPass
                    // scene_color chain: CA fringes the defocused image after
                    // DoF, vignette darkens its edges last. Serialised after the
                    // prior writer so the scene_color copy-backs never race.
                    chromatic_aberration_pass
                        .after(dof_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    vignette_pass
                        .after(chromatic_aberration_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Colour grade regrades the finished HDR image after
                    // vignette, then film grain adds sensor grain last. Both
                    // serialised after the prior writer so the scene_color
                    // copy-backs never race, before the main pass composites.
                    color_grade_pass
                        .after(vignette_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    film_grain_pass
                        .after(color_grade_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Temporal upscale is the final scene_color consumer: it
                    // reconstructs the fully post-processed HDR image onto the
                    // display grid (native-resolution TAAU) and RCAS-sharpens the
                    // result into `upscale_out`, which the composite reads in
                    // place of the raw scene_color. Ordered after the last
                    // scene_color writer (film grain) and before the composite.
                    // Opt-in: a no-op unless the `UpscaleSettings` resource is
                    // present, so the default renderer is unchanged.
                    upscale_pass
                        .after(film_grain_pass)
                        .before(composite_shading),
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
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: receiver generation reads the SSR geometry prepass
                // depth and fills the per-view receiver buffer, then page-mark
                // marks the resident-window request bitmap from those receivers,
                // both before the main pass.
                (
                    vsm_receiver_gen_pass
                        .after(ssr_prepass_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    vsm_mark_pages_pass
                        .after(vsm_receiver_gen_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Caster-depth raster fills the physical atlas pages after
                    // page-mark decides residency, before the main pass samples
                    // the VSM in the shading model.
                    vsm_caster_depth_pass
                        .after(vsm_mark_pages_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    // Froxel fog is self-contained (its own per-view volumes),
                    // so it only needs to finish before the main pass composites.
                    volumetrics_pass
                        .after(motion_blur_pass)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ),
            ),
        );
        // The nine additional post-process passes (display-referred finishing +
        // stylized NPR). The primary Core3d tuple is already at Bevy's
        // 20-element limit, so these live in their own `add_systems` call and
        // chain off `film_grain_pass` (the previous last pre-MainPass
        // scene_color writer). Every pass copies its result back into
        // scene_color, so they are serialised into a single linear chain to
        // keep the copy-backs from racing, all before the main pass composites.
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            (
                kuwahara_pass
                    .after(film_grain_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                hatching_pass
                    .after(kuwahara_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                halftone_pass
                    .after(hatching_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                outline_pass
                    .after(halftone_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                lens_flare_pass
                    .after(outline_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                tonemap_pass
                    .after(lens_flare_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                gamut_map_pass
                    .after(tonemap_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ordered_dither_pass
                    .after(gamut_map_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                cas_pass
                    .after(ordered_dither_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
                posterize_pass
                    .after(cas_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
            ),
        );

        // Volumetric-cloud compute solve. A self-contained device node: it
        // records the eight cloud kernels (five view-independent domain passes
        // once, three per-view passes per sized view) into its own compute
        // passes, writing only its resident cloud targets -- it never touches
        // scene_color, so it needs no serialisation with the post-process
        // copy-back chain and only has to finish before the main pass so the
        // resident cloud output is available to sample. No-ops while the clouds
        // are disabled or any kernel is still compiling. Its own `add_systems`
        // call keeps the primary Core3d tuple within Bevy's 20-element limit.
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            dispatch_volumetric_clouds.before(bevy_core_pipeline::Core3dSystems::MainPass),
        );

        // Virtual-shadow-map page-table upload bridge: read the GPU page-mark
        // request bitmap back one frame late, drive the golden allocator and
        // upload the resulting virtual->physical page table. Mirrors the
        // visibility parity readback's three-stage RenderGraph state machine.
        render_app.add_systems(
            RenderGraph,
            (
                request_vsm_page_readback
                    .after(camera_driver)
                    .in_set(RenderGraphSystems::Render),
                map_submitted_vsm_page_readback.in_set(RenderGraphSystems::Finish),
            ),
        );

        // Consume the previous frame's page-request readback and drive residency
        // in `PrepareResources` -- one phase *before* the resolve sampler / atlas
        // caster-depth bind groups build in `PrepareBindGroups` -- so the driven
        // page table is visible to its consumers the *same* frame it is produced.
        // `collect` records each view's outputs into `VsmBridgeCache`; the same-set
        // `bridge_vsm_view_resources` (ordered after it) re-attaches them as the
        // resolve/atlas-visible components, and the `PrepareResourcesFlush` sync
        // point applies those inserts before `PrepareBindGroups`. This keeps only
        // the readback's own unavoidable one-frame copy latency (mark on frame N,
        // resident on frame N+1) with no extra bridge frame on top.
        render_app.add_systems(
            Render,
            (
                collect_vsm_page_readback.in_set(RenderSystems::PrepareResources),
                bridge_vsm_view_resources
                    .in_set(RenderSystems::PrepareResources)
                    .after(collect_vsm_page_readback),
            ),
        );

        // Material-classification diagnostics readback: copy the per-view GPU
        // fault counters (background / stale / unsupported / overflow) back one
        // frame late and fold the aggregate into `PrismShadingDiagnostics` so
        // the classify/scatter passes' device-side accounting is CPU-observable.
        render_app.add_systems(
            RenderGraph,
            (
                collect_classification_readback.in_set(RenderGraphSystems::Begin),
                request_classification_readback
                    .after(camera_driver)
                    .in_set(RenderGraphSystems::Render),
                map_submitted_classification_readback.in_set(RenderGraphSystems::Finish),
            ),
        );
    }
}
