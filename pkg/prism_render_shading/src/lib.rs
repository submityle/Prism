//! Backend-neutral visibility-buffer and shading-work contracts.

#![forbid(unsafe_code)]
#![expect(
    missing_docs,
    reason = "The experimental shading ABI is documented as it freezes."
)]

extern crate alloc;

mod ao;
mod bloom;
mod classification;
mod cluster;
mod clearcoat;
mod cloth;
mod environment;
mod exposure;
mod face_shadow;
mod hair;
mod light_routing;
mod lighting;
mod motion_blur;
mod oit;
mod outline;
mod punctual;
mod resolve;
mod screen_space;
mod shadow;
mod subsurface;
mod stylized;
mod surface;
mod taa;
mod tangent;
mod tonemap;
mod texture_sample;
mod vecmath;
mod visibility;
mod volumetrics;
mod water;

pub use classification::{
    classify_material_header, ClassificationError, MaterialShadingClass, ShadingWorkItem,
    ShadingWorkPlan, MAX_SHADING_CLASSES,
};
pub use ao::{
    accumulate_ao, accumulate_moment, clip_history, compute_gtao, denoise_gtao, denoise_gtao_pixel,
    gtao_adaptive_history_weight, gtao_pixel, reproject_prev_uv_gtao, variance_clip_band,
    GtaoBuffers, GtaoCamera, GtaoClipResult, GtaoConfig, GtaoDenoiseConfig, GtaoTemporalParams,
};
pub use bloom::{
    combine, downsample_13tap, karis_average_weight, luminance as bloom_luminance,
    mip_blend_weights, prefilter, upsample_tent, BloomParams, BLOOM_LUMINANCE_WEIGHTS,
};
pub use motion_blur::{
    clamp_velocity, cone, cylinder, neighbor_max, sample_weight, shutter_velocity,
    soft_depth_compare, tile_max, velocity_length, MotionBlurParams,
};
pub use cluster::{
    assign_lights_to_clusters, ClusterAabb, ClusterAssignmentConfig, ClusterBoundsBuilder,
    ClusterGrid, ClusterLightAssignment,
};
pub use clearcoat::evaluate_clearcoat_direct;
pub use cloth::evaluate_cloth_direct;
pub use environment::{
    env_brdf_approx, evaluate_image_based_light, evaluate_image_based_light_specular,
    integrate_brdf, prefilter_radiance, project_cubemap_to_sh, CubemapFaces, DfgLut,
    ImageBasedLight, PrefilteredEnvMap, SpecularEnvironment, SphericalHarmonicsL2,
};
pub use exposure::{
    average_luminance_from_histogram, ev100_from_average_luminance, exposure_from_ev100,
    luminance, max_luminance_for_ev100, AutoExposureSettings, EyeAdaptation, HistogramPercentiles,
    HistogramRange, PhysicalCamera, LUMINANCE_WEIGHTS, MAX_LUMINANCE_FACTOR, METER_CALIBRATION_K,
};
pub use hair::evaluate_hair_direct;
pub use oit::{
    composite_transparency, oit_weight, OitAccumulation, OitFragment,
};
pub use lighting::{
    evaluate_principled_direct, linear_furnace_response, DirectLightSample,
    ShadingFrame, SurfaceSample,
};
pub use punctual::PunctualLight;
pub use screen_space::{
    accumulate_temporal, adaptive_history_weight, blend_specular, build_hemisphere_ray,
    build_screen_ray, clip_history_to_aabb, clip_history_to_aabb_ex, cosine_sample_direction,
    distance_fade, edge_fade, expand_bounds, facing_fade, gather_indirect_diffuse, ggx_ndf,
    hammersley, importance_sample_ggx,
    march_hierarchical, motion_vector, project_view_to_screen, project_world_to_screen,
    radical_inverse_vdc, reflect, reflection_mip,
    relax_box_for_confidence, reproject_prev_uv, reproject_prev_uv_motion,
    resolve_geometry_weight, resolve_reflection,
    reverse_z_perspective,
    roughness_fade, smith_ggx_visibility, smoothstep, trace_confidence,
    trace_indirect_ray, trace_screen_space_reflection, DepthPyramid, MotionSample, ScreenRay,
    ScreenSample, ScreenSpaceReflection, SsgiGather, SsgiParams, SsgiRaySample,
    SsrCamera, SsrConfidenceParams, SsrMarchConfig, SsrMarchResult, SsrResolveParams,
    variance_clip_box, ClipResult, SsrResolveSample, SsrTemporalParams, SsrTraceSample,
};
pub use taa::{
    halton, resolve_taa, rgb_to_ycocg, taa_jitter, tonemap_weight, ycocg_to_rgb,
    DEFAULT_TAA_JITTER_LEN, TaaParams,
};
pub use shadow::{
    allocate_shadow_atlas, apply_normal_offset, blocker_search, cascade_blend_weight,
    compute_cascade_matrices, compute_cascade_splits, cube_face_and_uv,
    cube_face_view_projections, evaluate_directional_shadow, evaluate_point_shadow,
    pcf_visibility, pcss_visibility, plan_shadow_depth_draws, select_cascade,
    evaluate_spot_shadow, invert, slope_scaled_depth_bias, spot_view_projection, transform_direction,
    transform_point, AtlasAllocation, AtlasConfig,
    AtlasSlot, BlockerSearch, CascadeMatrix, CascadeSplits, DirectionalShadowConfig,
    DirectionalShadowInput, Mat4, PcssConfig, PointShadowConfig, PointShadowInput, ShadowDepthDraw,
    ShadowDepthMode, ShadowDepthSampler, ShadowDepthView, ShadowFilter, ShadowKind, ShadowRequest,
    ShadowViewGeometry, SpotShadowConfig, SpotShadowInput, MAX_CASCADE_COUNT, POINT_LAYER_COUNT,
};
pub use resolve::{
    resolve_pixel, surface_sample_from_parameters, DirectionalLight, LightingEnvironment,
    ResolveError, ResolveInput, ResolvedPixel, IDENTITY_WORLD_FROM_LOCAL,
};
pub use subsurface::evaluate_subsurface_direct;
pub use stylized::{evaluate_stylized_direct, evaluate_toon_direct, StylizedParams};
pub use face_shadow::{
    evaluate_face_shadow, face_shadow_flip_u, face_shadow_light_cosines, FaceFrame,
    FaceShadowParams,
};
pub use light_routing::{
    cull_lights_by_channel, LightLayerMask, LightRouting, LightingChannelMask, MAX_LIGHTING_CHANNELS,
    MAX_LIGHT_LAYERS,
};
pub use outline::{
    evaluate_outline, outline_depth_edge, outline_id_edge, outline_normal_edge, OutlineGeometry,
    OutlineParams,
};
pub use water::evaluate_water_direct;
pub use tangent::{
    apply_tangent_space_normal, orthonormal_basis, resolve_tangent_basis, TangentBasis,
};
pub use tonemap::{
    apply_tonemap, tonemap_aces_fitted, tonemap_aces_narkowicz, tonemap_agx,
    tonemap_agx_with_look, tonemap_reinhard, tonemap_reinhard_extended, AgxLook, TonemapOperator,
    TonemapParams, ACES_INPUT_MATRIX, ACES_OUTPUT_MATRIX, AGX_INPUT_MATRIX, AGX_MAX_EV, AGX_MIN_EV,
    AGX_OUTPUT_MATRIX,
};
pub use texture_sample::{
    decode_tangent_normal, fold_material_texel, sample_material, sampled_material_defaults,
    srgb_channel_to_linear, srgb_to_linear, MaterialModulationParams, SampledMaterial,
    SampledTextureBinding, SEMANTIC_BASE_COLOR,
    SEMANTIC_CLEAR_COAT, SEMANTIC_CLEAR_COAT_NORMAL, SEMANTIC_CLEAR_COAT_ROUGHNESS,
    SEMANTIC_EMISSIVE, SEMANTIC_METALLIC_ROUGHNESS, SEMANTIC_NORMAL, SEMANTIC_OCCLUSION,
};
pub use surface::{
    reconstruct_surface, GpuShadingPrimitive, GpuShadingVertex, SurfaceReconstructionError,
    SurfaceReconstructionFlags, SurfaceReconstructionInput, SurfaceSampleGeometry,
};
pub use visibility::{
    encode_barycentrics, BarycentricError, VisibilityPixel, VisibilityPixelTargets,
    INVALID_VISIBILITY_ID,
    VISIBILITY_BUFFER_ABI_VERSION,
};
pub use volumetrics::{
    froxel_source, henyey_greenstein, in_scatter, integrate_froxel_column, integrate_slice,
    transmittance, Froxel, MediumSample, VolumetricIntegration,
};
