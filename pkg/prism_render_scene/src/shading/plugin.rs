use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_core_pipeline::schedule::camera_driver;
use bevy_ecs::{
    prelude::{Commands, Res},
    schedule::IntoScheduleConfigs,
    system::{BoxedSystem, IntoSystem},
};
use bevy_pbr::MeshPipelineSystems;
use bevy_render::{
    init_gpu_resource,
    render_phase::AddRenderCommand,
    render_resource::SpecializedRenderPipelines,
    renderer::{RenderGraph, RenderGraphSystems},
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use prism_render_architecture::render_fabric::{
    DeviceFabric, CUSTOM_FEATURE_ID, CUSTOM_VARIANT_ID, LIT_FEATURE_ID, LIT_VARIANT_ID,
    NPR_FEATURE_ID, NPR_VARIANT_ID, WATER_FEATURE_ID, WATER_VARIANT_ID,
};

use crate::gpu::fabric::SceneFabric;

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
    composition::{
        init_response_composition, prepare_response_composition_resources,
        prepare_response_composition_targets, response_composition_pass,
        ResponseCompositionSystems,
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
    execution::{
        run_shading_post_driver, run_shading_pre_resolve_driver, run_shading_resolve_driver,
        run_shading_response_driver, ExternalExecutionSite, SelectedVariantRequirement,
        ShadingExecutionPlan, ShadingExecutorBinding, ShadingExternalExecutors, ShadingPassBinding,
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
    post_graph::{validate_post_execution_bindings, PostExecutorKind, POST_EXECUTION_BINDINGS},
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
        prepare_shading_resolve_bind_groups, refresh_shading_resolve_pipeline,
        ResolveMotionHistory,
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
        init_world_restir_composite_pipeline, init_world_restir_pipeline,
        init_world_restir_resolve_pipeline, init_world_restir_visible_points_pipeline,
        prepare_world_restir_bind_groups, prepare_world_restir_composite,
        prepare_world_restir_composite_bind_groups, prepare_world_restir_lights,
        prepare_world_restir_reservoirs, prepare_world_restir_resolve,
        prepare_world_restir_resolve_bind_groups, prepare_world_restir_visible_points,
        prepare_world_restir_visible_points_bind_groups, world_restir_composite_pass,
        world_restir_fill_pass, world_restir_inject_pass, world_restir_resolve_pass,
        world_restir_seed_pass, world_restir_visible_points_pass, PrismWorldRestirSettings,
        WorldRestirLights,
    },
    world_space_gi::{
        init_world_space_gi_composite_pipeline, init_world_space_gi_pipeline,
        prepare_world_space_gi_bind_groups, prepare_world_space_gi_composite_bind_groups,
        prepare_world_space_gi_textures, world_space_gi_composite_pass, world_space_gi_pass,
        PrismWorldSpaceGiSettings,
    },
};

fn boxed_shading_system<S, M>(system: S) -> BoxedSystem
where
    S: IntoSystem<(), (), M> + 'static,
{
    Box::new(IntoSystem::into_system(system))
}

fn boxed_post_executor(kind: PostExecutorKind) -> BoxedSystem {
    match kind {
        PostExecutorKind::ExposureHistogram => boxed_shading_system(exposure_histogram_pass),
        PostExecutorKind::ExposureAverage => boxed_shading_system(exposure_average_pass),
        PostExecutorKind::Bloom => boxed_shading_system(bloom_pass),
        PostExecutorKind::Dof => boxed_shading_system(dof_pass),
        PostExecutorKind::MotionBlur => boxed_shading_system(motion_blur_pass),
        PostExecutorKind::SsrPrepass => boxed_shading_system(ssr_prepass_pass),
        PostExecutorKind::SsrHzb => boxed_shading_system(ssr_hzb_pass),
        PostExecutorKind::SsrRepack => boxed_shading_system(ssr_repack_pass),
        PostExecutorKind::SsrColorMips => boxed_shading_system(ssr_color_mips_pass),
        PostExecutorKind::SsrTrace => boxed_shading_system(ssr_trace_pass),
        PostExecutorKind::SsrReconstruct => boxed_shading_system(ssr_reconstruct_pass),
        PostExecutorKind::SsrTemporal => boxed_shading_system(ssr_temporal_pass),
        PostExecutorKind::SsrComposite => boxed_shading_system(ssr_composite_pass),
        PostExecutorKind::SsgiTrace => boxed_shading_system(ssgi_trace_pass),
        PostExecutorKind::SsgiDenoise => boxed_shading_system(ssgi_denoise_pass),
        PostExecutorKind::SsgiComposite => boxed_shading_system(ssgi_composite_pass),
        PostExecutorKind::SpecGiReuse => boxed_shading_system(spec_gi_reuse_pass),
        PostExecutorKind::SpecGiSpatial => boxed_shading_system(spec_gi_spatial_pass),
        PostExecutorKind::SpecDenoiseReproject => boxed_shading_system(spec_denoise_reproject_pass),
        PostExecutorKind::SpecDenoiseHistoryClamp => {
            boxed_shading_system(spec_denoise_history_clamp_pass)
        }
        PostExecutorKind::SpecDenoiseSpatial => boxed_shading_system(spec_denoise_spatial_pass),
        PostExecutorKind::SpecGiComposite => boxed_shading_system(spec_gi_composite_pass),
        PostExecutorKind::TaaResolve => boxed_shading_system(taa_resolve_pass),
        PostExecutorKind::Posterize => boxed_shading_system(posterize_pass),
        PostExecutorKind::Outline => boxed_shading_system(outline_pass),
        PostExecutorKind::Hatching => boxed_shading_system(hatching_pass),
        PostExecutorKind::Halftone => boxed_shading_system(halftone_pass),
        PostExecutorKind::Kuwahara => boxed_shading_system(kuwahara_pass),
        PostExecutorKind::ColorGrade => boxed_shading_system(color_grade_pass),
        PostExecutorKind::ChromaticAberration => boxed_shading_system(chromatic_aberration_pass),
        PostExecutorKind::Vignette => boxed_shading_system(vignette_pass),
        PostExecutorKind::FilmGrain => boxed_shading_system(film_grain_pass),
        PostExecutorKind::Cas => boxed_shading_system(cas_pass),
        PostExecutorKind::GamutMap => boxed_shading_system(gamut_map_pass),
        PostExecutorKind::OrderedDither => boxed_shading_system(ordered_dither_pass),
        PostExecutorKind::LensFlare => boxed_shading_system(lens_flare_pass),
        PostExecutorKind::Tonemap => boxed_shading_system(tonemap_pass),
        PostExecutorKind::Upscale => boxed_shading_system(upscale_pass),
    }
}

fn shading_execution_bindings() -> Vec<ShadingPassBinding> {
    let mut bindings = vec![
        ShadingPassBinding {
            declaration: "visibility_raster",
            requires: &[],
            executor: ShadingExecutorBinding::Systems {
                group: "visibility_raster",
                systems: vec![boxed_shading_system(visibility_raster_pass)],
            },
        },
        ShadingPassBinding {
            declaration: "material_classification_count",
            requires: &[],
            executor: ShadingExecutorBinding::Systems {
                group: "material_classification",
                systems: vec![boxed_shading_system(dispatch_material_classification)],
            },
        },
        ShadingPassBinding {
            declaration: "material_classification_prefix",
            requires: &[],
            executor: ShadingExecutorBinding::GroupMember {
                group: "material_classification",
                owner: "material_classification_count",
            },
        },
        ShadingPassBinding {
            declaration: "material_classification_scatter",
            requires: &[],
            executor: ShadingExecutorBinding::GroupMember {
                group: "material_classification",
                owner: "material_classification_count",
            },
        },
        ShadingPassBinding {
            declaration: "texel_space_shade_cache",
            requires: &[],
            executor: ShadingExecutorBinding::Systems {
                group: "texel_space_shade_cache",
                systems: vec![boxed_shading_system(shade_dispatch_pass)],
            },
        },
        ShadingPassBinding {
            declaration: "shading_resolve",
            requires: &[],
            executor: ShadingExecutorBinding::Systems {
                group: "shading_resolve",
                systems: vec![boxed_shading_system(dispatch_shading_resolve)],
            },
        },
        ShadingPassBinding {
            declaration: "water_surface_raster",
            requires: &[SelectedVariantRequirement {
                feature: WATER_FEATURE_ID,
                variant: WATER_VARIANT_ID,
            }],
            executor: ShadingExecutorBinding::ExternalSchedule {
                site: ExternalExecutionSite::WaterSurface,
            },
        },
        ShadingPassBinding {
            declaration: "response_composition",
            requires: &[],
            executor: ShadingExecutorBinding::Systems {
                group: "response_composition",
                systems: vec![boxed_shading_system(response_composition_pass)],
            },
        },
    ];

    for binding in POST_EXECUTION_BINDINGS {
        let executors = binding
            .executors
            .iter()
            .map(|kind| boxed_post_executor(*kind))
            .collect();
        bindings.push(ShadingPassBinding {
            declaration: binding.declaration,
            requires: &[],
            executor: ShadingExecutorBinding::Systems {
                group: binding.declaration,
                systems: executors,
            },
        });
    }
    bindings
}

fn validate_post_execution_registration(
    graph: &prism_render_graph::frame_graph::GpuFrameGraphBuilder,
    bindings: &[ShadingPassBinding],
) -> Result<(), String> {
    validate_post_execution_bindings(graph)?;

    let mut registered_post_bindings = 0usize;
    for expected in POST_EXECUTION_BINDINGS {
        let Some(actual) = bindings
            .iter()
            .find(|binding| binding.declaration == expected.declaration)
        else {
            return Err(format!(
                "post module {} is missing its production execution registration",
                expected.module
            ));
        };
        registered_post_bindings += 1;
        let ShadingExecutorBinding::Systems { group, systems } = &actual.executor else {
            return Err(format!(
                "post module {} requires its own Systems executor",
                expected.module
            ));
        };
        if *group != expected.declaration {
            return Err(format!(
                "post module {} uses executor group {} instead of {}",
                expected.module, group, expected.declaration
            ));
        }
        if systems.len() != expected.executors.len() {
            return Err(format!(
                "post module {} executor count {} != inventory count {}",
                expected.module,
                systems.len(),
                expected.executors.len()
            ));
        }
        for (index, (system, kind)) in systems.iter().zip(expected.executors).enumerate() {
            let expected_system = boxed_post_executor(*kind);
            // Debug names may be disabled; validate executable identity even
            // in builds where every system has the same placeholder name.
            if system.system_type() != expected_system.system_type() {
                return Err(format!(
                    "post module {} executor {index} is {} instead of {}",
                    expected.module,
                    system.name(),
                    expected_system.name(),
                ));
            }
        }
    }
    if registered_post_bindings != POST_EXECUTION_BINDINGS.len() {
        return Err(format!(
            "production post registration count {} != inventory count {}",
            registered_post_bindings,
            POST_EXECUTION_BINDINGS.len()
        ));
    }
    Ok(())
}

/// The shared resolve supports these implementations, but only responses in
/// the frozen build profile are admitted. Do not expand roots or reselect a
/// fallback when a new catalog implementation has no production executor.
fn validate_shading_response_selection(fabric: &DeviceFabric) -> Result<(), String> {
    for &feature in fabric.selection().roots().responses() {
        let expected = match feature {
            LIT_FEATURE_ID => LIT_VARIANT_ID,
            NPR_FEATURE_ID => NPR_VARIANT_ID,
            CUSTOM_FEATURE_ID => CUSTOM_VARIANT_ID,
            _ => {
                return Err(format!(
                    "no shading response executor for frozen root {feature:?}"
                ))
            }
        };
        let selected = fabric.admit(feature).map_err(|error| error.to_string())?;
        if selected.variant_id() != expected {
            return Err(format!(
                "shading response root {feature:?} requires variant {expected:?}, but frozen selection chose {:?}",
                selected.variant_id()
            ));
        }
    }
    Ok(())
}

fn build_shading_execution(
    fabric: &DeviceFabric,
    external: &ShadingExternalExecutors,
) -> Result<(ShadingFrameGraph, ShadingExecutionPlan), String> {
    validate_shading_response_selection(fabric)?;
    let graph = shading_frame_graph();
    let compiled = graph.compile().map_err(|error| error.to_string())?;
    let bindings = shading_execution_bindings();
    validate_post_execution_registration(&graph, &bindings)?;
    let execution =
        ShadingExecutionPlan::new(fabric, external, &graph, compiled.clone(), bindings)?;
    Ok((
        ShadingFrameGraph {
            compiled,
            post_execution_bindings: POST_EXECUTION_BINDINGS,
        },
        execution,
    ))
}

fn init_shading_execution(
    mut commands: Commands,
    fabric: Res<SceneFabric>,
    external: Res<ShadingExternalExecutors>,
) {
    let (graph, execution) = build_shading_execution(&fabric.0, &external)
        .expect("Prism shading executors must satisfy the device's frozen feature selection");
    commands.insert_resource(graph);
    commands.insert_resource(execution);
}

use super::texel_cache::{
    extract_shade_dispatch_inputs, prepare_shade_dispatch, shade_dispatch_pass,
    ShadeDispatchProductionGpu, ShadeDispatchProductionSettings,
};

pub struct PrismShadingPlugin;

impl Plugin for PrismShadingPlugin {
    fn build(&self, app: &mut App) {
        // Main-world page producers fill this explicit input seam. The render
        // world receives only a cloned frame snapshot during extraction.
        app.init_resource::<super::texel_cache::ShadeDispatchInputs>();
        embedded_asset!(app, "../shaders/visibility_raster.wesl");
        embedded_asset!(app, "../shaders/material_classification.wesl");
        embedded_asset!(app, "../shaders/brdf.wesl");
        embedded_asset!(app, "../shaders/cloth.wesl");
        embedded_asset!(app, "../shaders/subsurface.wesl");
        embedded_asset!(app, "../shaders/hair.wesl");
        embedded_asset!(app, "../shaders/water.wesl");
        embedded_asset!(app, "../shaders/clearcoat.wesl");
        // `material_sample.wesl` imports the VT resolver by its canonical
        // module path. Register the imported module in the production asset
        // graph as well as in off-GPU parity tests, otherwise runtime shader
        // resolution depends on a test-only registration.
        embedded_asset!(app, "../shaders/material_virtual_texture.wesl");
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
        embedded_asset!(app, "../shaders/world_restir_composite.wesl");
        register_shadow_depth_shader(app);
        register_vsm_caster_depth_shader(app);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // The pass is installed once, irrespective of scene/split contents.
        // These constraints express actual resource dependencies, independent
        // of the tuple grouping used by the older stages below.
        render_app
            .init_resource::<ShadingExternalExecutors>()
            .add_systems(
                RenderStartup,
                init_shading_execution.after(crate::gpu::fabric::init_scene_fabric),
            )
            .add_systems(RenderStartup, init_response_composition)
            .add_systems(
                Render,
                (
                    prepare_response_composition_resources
                        .after(prepare_visibility_buffers)
                        .in_set(RenderSystems::PrepareResources),
                    prepare_response_composition_targets
                        .after(prepare_response_composition_resources)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                bevy_core_pipeline::Core3d,
                run_shading_pre_resolve_driver.before(bevy_core_pipeline::Core3dSystems::MainPass),
            )
            .add_systems(
                bevy_core_pipeline::Core3d,
                run_shading_resolve_driver
                    .after(run_shading_pre_resolve_driver)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
            )
            .add_systems(
                bevy_core_pipeline::Core3d,
                run_shading_response_driver
                    .in_set(ResponseCompositionSystems::Compose)
                    .after(run_shading_resolve_driver)
                    .after(bevy_core_pipeline::Core3dSystems::MainPass)
                    .after(crate::water::WaterSurfaceSystems::Draw)
                    .before(run_shading_post_driver)
                    .before(composite_shading)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
            )
            .add_systems(
                bevy_core_pipeline::Core3d,
                run_shading_post_driver
                    .after(run_shading_response_driver)
                    .before(composite_shading)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
            );
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
            .init_resource::<super::texel_cache::ShadeDispatchInputs>()
            .init_resource::<ShadeDispatchProductionSettings>()
            .init_resource::<ShadeDispatchProductionGpu>()
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
            .add_render_command::<Visibility3d, DrawVisibilityRaster>()
            .add_render_command::<TransparentOit3d, DrawTransparentOit>()
            .add_systems(RenderStartup, detect_shading_capabilities)
            .add_systems(
                RenderStartup,
                (
                    // The writer is lazily allocated in PrepareResources when
                    // the opt-in setting is enabled; keeping this resource in
                    // the normal render lifecycle makes the no-input state
                    // explicit without allocating a fake atlas.
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
            .add_systems(ExtractSchedule, extract_shade_dispatch_inputs)
            .add_systems(
                Render,
                prepare_shade_dispatch
                    .after(crate::geometry::upload_shading_geometry_buffers)
                    .in_set(RenderSystems::PrepareResources),
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
                refresh_shading_resolve_pipeline.in_set(RenderSystems::PrepareResources),
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
        render_app.add_systems(RenderStartup, init_world_restir_composite_pipeline);
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
                // The composite scratch is sized to the resolve export in
                // PrepareResources; its copy + fold groups read the visibility
                // buffer (scene_color / ssgi_albedo / world_restir_direct) and
                // the resolve export in PrepareBindGroups, so they build after
                // the resolve group that produces gi_out.
                prepare_world_restir_composite
                    .after(prepare_world_restir_resolve)
                    .in_set(RenderSystems::PrepareResources),
                prepare_world_restir_composite_bind_groups
                    .after(prepare_world_restir_composite)
                    .after(prepare_world_restir_resolve_bind_groups)
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
                // the SSR prepass, writing the direct-illumination export the
                // composite folds into scene_color, so it runs before the
                // composite and still finishes before the main pass.
                world_restir_resolve_pass.before(world_restir_composite_pass),
                // The composite folds the resolve's direct-illumination export
                // back over scene_color (energy-conserving substitution of the
                // clustered punctual diffuse). It reads gi_out + scene_color +
                // the shading-resolve albedo / clustered-direct exports, so it
                // runs after the resolve and, being a scene_color writer, after
                // the last peer GI composite (surface cache) in the
                // scene_color-writer chain, and before the main pass.
                world_restir_composite_pass
                    .after(world_restir_resolve_pass)
                    .after(surface_cache_composite_pass)
                    .before(bevy_core_pipeline::Core3dSystems::MainPass),
            ),
        );
        render_app.add_systems(
            bevy_core_pipeline::Core3d,
            (
                shadow_depth_pass.before(run_shading_pre_resolve_driver),
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: the three GTAO passes chain prepass -> horizon kernel
                // -> spatial denoise, all feeding the shading resolve.
                (
                    gtao_prepass_pass
                        .after(run_shading_pre_resolve_driver)
                        .before(run_shading_resolve_driver),
                    gtao_compute_pass
                        .after(gtao_prepass_pass)
                        .before(run_shading_resolve_driver),
                    gtao_denoise_pass
                        .after(gtao_compute_pass)
                        .before(run_shading_resolve_driver),
                    gtao_temporal_pass
                        .after(gtao_denoise_pass)
                        .before(run_shading_resolve_driver),
                ),
                // Nested to keep the Core3d tuple within Bevy's 20-element
                // limit: the IBL precomputes and the light-routing cull all
                // feed the shading resolve and so must run before it.
                (
                    dfg_lut_precompute_pass.before(run_shading_resolve_driver),
                    env_prefilter_precompute_pass.before(run_shading_resolve_driver),
                    // Light-routing (Lighting Channels) cull refines the
                    // per-word punctual-light visibility mask the resolve's
                    // channel gate reads, so it must run before the resolve
                    // dispatch. No-op unless the opt-in subsystem is enabled.
                    light_routing_cull_pass.before(run_shading_resolve_driver),
                ),
                (
                    // Screen-space spec-GI reuse, denoising and composite are
                    // owned by the post driver, after SSR fills their current-
                    // frame inputs and before SSGI snapshots scene_color.
                    // World-space GI gathers screen probes off the fully
                    // resolve-fed scene_color, then its own composite
                    // folds the indirect diffuse back in (energy-conserving
                    // ambient substitution) before TAA resolves the buffer.
                    world_space_gi_pass
                        .after(run_shading_resolve_driver)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                    world_space_gi_composite_pass
                        .after(world_space_gi_pass)
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
                ),
                composite_shading
                    .after(bevy_core_pipeline::Core3dSystems::MainPass)
                    .after(ResponseCompositionSystems::Compose)
                    .after(run_shading_post_driver)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
                transparent_forward_pass
                    .after(run_shading_resolve_driver)
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
                        .after(run_shading_resolve_driver)
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
                        .after(run_shading_resolve_driver)
                        .before(bevy_core_pipeline::Core3dSystems::MainPass),
                ),
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::SubApp;
    use bevy_ecs::schedule::Schedule;
    use prism_render_architecture::render_fabric::{
        builtin_catalog, builtin_roots, CapabilitySet, FeatureCatalog, FeatureDefinition,
        FeatureId, FeatureVariant, FrozenRoots, RenderQualityTier, ServiceSet, VariantId,
    };

    fn build_profile_fabric(
        subsystems: &'static [FeatureId],
        responses: &'static [FeatureId],
    ) -> DeviceFabric {
        let catalog = builtin_catalog().unwrap();
        let roots = FrozenRoots::from_build_profile(&catalog, subsystems, responses).unwrap();
        DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::High,
        )
        .unwrap()
    }

    fn full_fabric() -> DeviceFabric {
        let catalog = builtin_catalog().unwrap();
        let roots = builtin_roots(&catalog).unwrap();
        DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::High,
        )
        .unwrap()
    }

    fn render_app_with_water() -> SubApp {
        let mut render_app = SubApp::new();
        crate::water::plugin::install_water_surface_executor(&mut render_app);
        render_app
    }

    #[test]
    fn production_post_registration_covers_the_repository_inventory() {
        let graph = shading_frame_graph();
        let bindings = shading_execution_bindings();
        validate_post_execution_registration(&graph, &bindings).unwrap();
    }

    fn specular_systems(bindings: &mut [ShadingPassBinding]) -> &mut Vec<BoxedSystem> {
        let binding = bindings
            .iter_mut()
            .find(|binding| binding.declaration == "post_spec_denoise")
            .unwrap();
        let ShadingExecutorBinding::Systems { systems, .. } = &mut binding.executor else {
            panic!("specular pipeline must own its executable systems");
        };
        systems
    }

    #[test]
    fn specular_pipeline_registration_uses_real_executors_once_in_order() {
        let graph = shading_frame_graph();
        let mut bindings = shading_execution_bindings();
        validate_post_execution_registration(&graph, &bindings).unwrap();
        // Construct the expected functions directly, independently of the
        // inventory and boxed_post_executor's kind-to-system mapping.
        let expected = vec![
            boxed_shading_system(spec_gi_reuse_pass),
            boxed_shading_system(spec_gi_spatial_pass),
            boxed_shading_system(spec_denoise_reproject_pass),
            boxed_shading_system(spec_denoise_history_clamp_pass),
            boxed_shading_system(spec_denoise_spatial_pass),
            boxed_shading_system(spec_gi_composite_pass),
        ];
        let actual_types = specular_systems(&mut bindings)
            .iter()
            .map(|system| system.system_type())
            .collect::<Vec<_>>();
        let expected_types = expected
            .iter()
            .map(|system| system.system_type())
            .collect::<Vec<_>>();
        assert_eq!(actual_types, expected_types);
        for expected_system in &expected {
            let count: usize = bindings
                .iter()
                .map(|binding| match &binding.executor {
                    ShadingExecutorBinding::Systems { systems, .. } => systems
                        .iter()
                        .filter(|system| system.system_type() == expected_system.system_type())
                        .count(),
                    _ => 0,
                })
                .sum();
            assert_eq!(
                count,
                1,
                "{} must execute exactly once",
                expected_system.name()
            );
        }
    }

    #[test]
    fn specular_pipeline_registration_rejects_missing_reuse() {
        let graph = shading_frame_graph();
        let mut bindings = shading_execution_bindings();
        specular_systems(&mut bindings).remove(0);
        let error = validate_post_execution_registration(&graph, &bindings).unwrap_err();
        assert!(
            error.contains("spec_denoise") && error.contains("executor count"),
            "{error}"
        );
    }

    #[test]
    fn specular_pipeline_registration_rejects_reordered_duplicate_or_wrong_executors() {
        let graph = shading_frame_graph();
        for mutation in 0..3 {
            let mut bindings = shading_execution_bindings();
            let systems = specular_systems(&mut bindings);
            match mutation {
                0 => systems.swap(0, 1),
                1 => systems[1] = boxed_shading_system(spec_gi_reuse_pass),
                _ => systems[0] = boxed_shading_system(ssr_trace_pass),
            }
            let error = validate_post_execution_registration(&graph, &bindings).unwrap_err();
            assert!(
                error.contains("spec_denoise") && error.contains("instead of"),
                "{error}"
            );
        }
    }

    #[test]
    fn production_post_registration_rejects_a_missing_module() {
        let graph = shading_frame_graph();
        let mut bindings = shading_execution_bindings();
        bindings.retain(|binding| binding.declaration != "post_bloom");
        let error = validate_post_execution_registration(&graph, &bindings).unwrap_err();
        assert!(error.contains("bloom"), "unexpected error: {error}");
    }

    #[test]
    fn production_execution_uses_frozen_selection_and_the_same_compiled_graph() {
        let fabric = full_fabric();
        let snapshot = fabric.selection().clone();
        let render_app = render_app_with_water();
        let external = render_app.world().resource::<ShadingExternalExecutors>();
        let (graph, execution) = build_shading_execution(&fabric, external).unwrap();
        assert_eq!(graph.compiled, *execution.compiled());
        assert_eq!(graph.compiled.declared_pass_count, 31);
        assert!(execution.is_alive("water_surface_raster"));
        assert!(execution.is_alive("response_composition"));
        assert_eq!(fabric.selection(), &snapshot);
        assert!(render_app
            .get_schedule(bevy_core_pipeline::Core3d)
            .is_some());
    }

    #[test]
    fn production_execution_does_not_enable_unreferenced_responses() {
        let fabric = build_profile_fabric(&[WATER_FEATURE_ID], &[LIT_FEATURE_ID]);
        let snapshot = fabric.selection().clone();
        let render_app = render_app_with_water();
        build_shading_execution(
            &fabric,
            render_app.world().resource::<ShadingExternalExecutors>(),
        )
        .unwrap();
        assert_eq!(fabric.selection().roots().responses(), &[LIT_FEATURE_ID]);
        assert!(fabric.admit(NPR_FEATURE_ID).is_err());
        assert!(fabric.admit(CUSTOM_FEATURE_ID).is_err());
        assert_eq!(fabric.selection(), &snapshot);
    }

    #[test]
    fn production_execution_rejects_uninstalled_water_executor() {
        let result = build_shading_execution(&full_fabric(), &ShadingExternalExecutors::default());
        let error = result.err().expect("missing draw must fail admission");
        assert!(
            error.contains("unregistered external executor WaterSurface"),
            "{error}"
        );
    }

    #[test]
    fn production_execution_cannot_expand_the_water_root() {
        let fabric = build_profile_fabric(&[], &[LIT_FEATURE_ID]);
        let snapshot = fabric.selection().clone();
        let render_app = render_app_with_water();
        let result = build_shading_execution(
            &fabric,
            render_app.world().resource::<ShadingExternalExecutors>(),
        );
        let error = result
            .err()
            .expect("registered topology needs explicit water root");
        assert!(error.contains("OutsideFrozenRoots"), "{error}");
        assert_eq!(fabric.selection(), &snapshot);
    }

    fn response_only_fabric(feature: FeatureId, variant: VariantId) -> DeviceFabric {
        let variants = Box::leak(Box::new([FeatureVariant {
            id: variant,
            rank: 0,
            min_quality: RenderQualityTier::Low,
            required_caps: CapabilitySet::EMPTY,
            requires_services_extra: ServiceSet::EMPTY,
            passes: &[],
            shader_permutation: None,
        }]));
        let definitions = Box::leak(Box::new([FeatureDefinition::response(
            feature,
            ServiceSet::EMPTY,
            variants,
        )]));
        let catalog = FeatureCatalog::new(definitions, &[]).unwrap();
        let roots =
            FrozenRoots::from_build_profile(&catalog, &[], Box::leak(Box::new([feature]))).unwrap();
        DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::High,
        )
        .unwrap()
    }

    #[test]
    fn production_response_executor_rejects_unsupported_selected_variant() {
        let fabric = response_only_fabric(LIT_FEATURE_ID, VariantId(0xf001));
        let snapshot = fabric.selection().clone();
        let error = validate_shading_response_selection(&fabric).unwrap_err();
        assert!(error.contains("requires variant"), "{error}");
        assert_eq!(fabric.selection(), &snapshot);
    }

    #[test]
    fn production_response_executor_rejects_unknown_response_root() {
        let fabric = response_only_fabric(FeatureId(0xf002), VariantId(0xf003));
        let error = validate_shading_response_selection(&fabric).unwrap_err();
        assert!(error.contains("no shading response executor"), "{error}");
    }

    #[test]
    fn shading_startup_consumes_deferred_fabric_and_preserves_registered_executors() {
        fn insert_fabric(mut commands: Commands) {
            commands.insert_resource(SceneFabric(full_fabric()));
        }
        let mut render_app = render_app_with_water();
        // Plugin build order must not reset a previously installed external site.
        render_app.init_resource::<ShadingExternalExecutors>();
        let mut startup = Schedule::default();
        startup.add_systems((insert_fabric, init_shading_execution.after(insert_fabric)));
        startup.run(render_app.world_mut());
        let world = render_app.world();
        assert!(world
            .resource::<ShadingExternalExecutors>()
            .contains(ExternalExecutionSite::WaterSurface));
        assert_eq!(
            world.resource::<ShadingFrameGraph>().compiled,
            *world.resource::<ShadingExecutionPlan>().compiled(),
        );
    }

    #[test]
    fn hardware_scene_startup_freezes_the_owning_device_and_admits_real_executors() {
        use bevy_render::renderer::RenderDevice;
        use prism_render_architecture::backend::capability::Capability;
        use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};

        let instance = WgpuInstance::new();
        let adapter = instance.enumerate_adapters().into_iter().find(|adapter| {
            matches!(
                adapter.get_info().device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            )
        });
        let Some(adapter) = adapter else {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but scene startup has no physical GPU"
            );
            eprintln!("SKIP hardware scene startup: no physical GPU adapter");
            return;
        };
        let info = adapter.get_info();
        let (device, _queue) = instance
            .create_device(&adapter)
            .expect("physical GPU must create the scene's owning device");
        eprintln!(
            "scene startup GPU: name={:?} backend={:?} type={:?} vendor={:#x} device={:#x}",
            info.name, info.backend, info.device_type, info.vendor, info.device
        );
        let render_device = RenderDevice::from(device.wgpu_device().clone());
        let features = render_device.features();
        let mut render_app = render_app_with_water();
        render_app.insert_resource(render_device);
        render_app.init_resource::<ShadingExternalExecutors>();
        let mut startup = Schedule::default();
        startup.add_systems((
            crate::gpu::fabric::init_scene_fabric,
            init_shading_execution.after(crate::gpu::fabric::init_scene_fabric),
        ));
        startup.run(render_app.world_mut());

        let world = render_app.world();
        let fabric = &world.resource::<SceneFabric>().0;
        let selection = fabric.selection();
        assert!(selection
            .device_caps()
            .contains_all(CapabilitySet::WEBGPU_BASELINE));
        for (feature, capability) in [
            (
                wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
                Capability::IndirectDrawCount,
            ),
            (
                wgpu::Features::EXPERIMENTAL_MESH_SHADER,
                Capability::MeshShading,
            ),
            (wgpu::Features::EXPERIMENTAL_RAY_QUERY, Capability::RayQuery),
            (
                wgpu::Features::EXPERIMENTAL_RAY_TRACING_PIPELINES,
                Capability::RayTracingPipeline,
            ),
            (
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                Capability::BindlessDescriptors,
            ),
        ] {
            assert_eq!(
                selection.device_caps().contains(capability),
                features.contains(feature),
                "scene capability {capability:?} must describe its actual device"
            );
        }
        assert_eq!(
            selection.roots(),
            &builtin_roots(&builtin_catalog().unwrap()).unwrap()
        );
        assert_eq!(selection.quality(), RenderQualityTier::High);
        assert!(world
            .resource::<ShadingExternalExecutors>()
            .contains(ExternalExecutionSite::WaterSurface));
        let graph = world.resource::<ShadingFrameGraph>();
        let execution = world.resource::<ShadingExecutionPlan>();
        assert_eq!(graph.compiled, *execution.compiled());
        assert_eq!(graph.compiled.declared_pass_count, 31);
        assert!(execution.is_alive("water_surface_raster"));
        assert!(execution.is_alive("response_composition"));
        let snapshot = selection.clone();
        build_shading_execution(fabric, world.resource::<ShadingExternalExecutors>()).unwrap();
        assert_eq!(fabric.selection(), &snapshot);
    }
}
