//! Backend-neutral visibility-buffer and shading-work contracts.

#![forbid(unsafe_code)]
#![expect(
    missing_docs,
    reason = "The experimental shading ABI is documented as it freezes."
)]

extern crate alloc;

mod ao;
mod bloom;
mod cas;
mod chromatic_aberration;
mod charlie_albedo_table;
mod classification;
mod clearcoat;
mod cloth;
mod cloth_advanced;
mod cluster;
mod color_grade;
mod dielectric_albedo_table;
mod dof;
mod dynamic;
mod eon_albedo_table;
mod environment;
mod exposure;
mod face_shadow;
mod film_grain;
mod gamut_map;
mod ggx_albedo_table;
mod glint;
mod hair;
mod hair_angel_ring;
mod hair_chiang;
mod hair_fiber;
mod hair_kajiya;
mod hair_matcap;
mod halftone;
mod hatching;
mod kuwahara;
mod lens_flare;
mod light_routing;
mod lighting;
mod lobe_sampling;
mod measured_brdf;
mod measured_lobe;
mod motion_blur;
mod oit;
// The conformance oracle is a *test* artifact (§4.5 constraint 1 / roadmap
// P2-f): it exists to be compared against, so it is compiled only for tests and
// adds nothing to the shipping shading binary.
#[cfg(test)]
mod openpbr_reference;
mod ordered_dither;
mod outline;
mod posterize;
mod punctual;
mod quality;
mod resolve;
mod schlick_bias_table;
mod screen_space;
mod shadow;
mod stylized;
mod stylized_hair;
mod subsurface;
mod surface;
mod taa;
mod tangent;
mod texture_lod;
mod texture_sample;
mod tonemap;
pub mod upscale;
mod vecmath;
mod vignette;
mod visibility;
mod volumetrics;
mod water;

pub use ao::{
    accumulate_ao, accumulate_moment, clip_history, compute_gtao, denoise_gtao, denoise_gtao_pixel,
    gtao_adaptive_history_weight, gtao_pixel, reproject_prev_uv_gtao, variance_clip_band,
    GtaoBuffers, GtaoCamera, GtaoClipResult, GtaoConfig, GtaoDenoiseConfig, GtaoTemporalParams,
};
pub use bloom::{
    combine, downsample_13tap, karis_average_weight, luminance as bloom_luminance,
    mip_blend_weights, prefilter, upsample_tent, BloomParams, BLOOM_LUMINANCE_WEIGHTS,
};
pub use cas::{
    apply_cas, cas_amplitude, cas_blend_channel, cas_sharpen, cas_weight, max3, min3,
    soft_max_channel, soft_min_channel, CasParams, Neighborhood,
};
pub use chromatic_aberration::{
    apply_chromatic_aberration, channel_uv, combine_channels, radial_offset, sample_offsets,
    spectral_lut, spectral_offset, ChromaticAberrationParams, CHROMATIC_ABERRATION_CHANNELS,
    CHROMATIC_ABERRATION_EPSILON,
};
pub use classification::{
    classify_material_header, ClassificationError, MaterialShadingClass, ShadingWorkItem,
    ShadingWorkPlan, MAX_SHADING_CLASSES,
};
pub use clearcoat::evaluate_clearcoat_direct;
pub use cloth::evaluate_cloth_direct;
pub use cloth_advanced::evaluate_cloth_advanced_direct;
pub use cluster::{
    assign_lights_to_clusters, ClusterAabb, ClusterAssignmentConfig, ClusterBoundsBuilder,
    ClusterGrid, ClusterLightAssignment,
};
pub use color_grade::{
    apply_color_grade, color_saturation, contrast, lift_gamma_gain, lin_to_lms, lms_to_lin, luma,
    offset, standard_illuminant_y, white_balance, ColorGradeParams, COLOR_GRADE_LUMA_WEIGHTS,
    D65_LMS,
};
pub use dof::{
    aperture_diameter, apply_dof, bokeh_weight, circle_of_confusion, coc_to_pixels,
    dof_blend_factor, far_field_coc, near_field_coc, signed_coc, DofCamera, DofParams,
};
pub use dynamic::{
    apply_dynamic_channels, dissolve_coverage, dissolve_mask, DAMAGE_BASE_DARKEN, DAMAGE_METAL_LOSS, DAMAGE_ROUGHEN,
    DISSOLVE_EDGE_GAIN, DISSOLVE_EDGE_WIDTH, DISSOLVE_MASK_CELLS, SNOW_ALBEDO,
    SNOW_DIFFUSE_ROUGHNESS, SNOW_ROUGHNESS, SNOW_SHEEN, SNOW_SPECULAR_WEIGHT, WATER_IOR,
    WET_BASE_DARKEN, WET_ROUGHNESS_SCALE,
};
pub use environment::{
    env_brdf_approx, evaluate_image_based_light, evaluate_image_based_light_specular,
    integrate_brdf, prefilter_radiance, project_cubemap_to_sh, CubemapFaces, DfgLut,
    ImageBasedLight, PrefilteredEnvMap, SpecularEnvironment, SphericalHarmonicsL2,
};
pub use exposure::{
    average_luminance_from_histogram, ev100_from_average_luminance, exposure_from_ev100, luminance,
    max_luminance_for_ev100, AutoExposureSettings, EyeAdaptation, HistogramPercentiles,
    HistogramRange, PhysicalCamera, LUMINANCE_WEIGHTS, MAX_LUMINANCE_FACTOR, METER_CALIBRATION_K,
};
pub use face_shadow::{
    evaluate_face_shadow, face_shadow_flip_u, face_shadow_light_cosines, FaceFrame,
    FaceShadowParams,
};
pub use film_grain::{
    apply_grain, grain, grain_fract, grain_lerp, grain_luminance_weight, hash12,
    luminance as film_grain_luminance, smoothstep as film_grain_smoothstep, FilmGrainParams,
    FILM_GRAIN_LUMA_WEIGHTS,
};
pub use gamut_map::{
    achromatic, apply_gamut_compress, clamp01 as gamut_clamp01, compress_distance,
    distance_from_achromatic, luminance as gamut_map_luminance, max_channel, min_channel,
    GamutMapParams, GAMUT_MAP_LUMA_WEIGHTS,
};
pub use glint::{
    fresnel_schlick_scalar as glint_fresnel_schlick, ggx_ndf as glint_ggx_ndf,
    glint_bin_solid_angle, glint_half_slope, glint_is_finite, glint_microfacet_count,
    glint_ndf, glint_ndf_blended, glint_ndf_blended_from_density, glint_ndf_from_density,
    glint_seed, glint_specular_brdf,
    hash_combine as glint_hash_combine, hash_u32 as glint_hash_u32,
    poisson_gaussian as glint_poisson_gaussian, poisson_knuth as glint_poisson_knuth,
    poisson_sample as glint_poisson_sample, quantize_axis as glint_quantize_axis,
    smith_g1 as glint_smith_g1, smith_g2_height_correlated as glint_smith_g2,
    uniform_at as glint_uniform_at, uniform_from_hash as glint_uniform_from_hash,
    GLINT_PI, GLINT_SEED_SALT, MIN_ALPHA as GLINT_MIN_ALPHA, MIN_COS as GLINT_MIN_COS,
    MIN_POSITIVE as GLINT_MIN_POSITIVE, POISSON_KNUTH_MAX_LAMBDA, POISSON_MAX_ITERS,
};
pub use hair::evaluate_hair_direct;
pub use hair_angel_ring::{accumulate_angel_rings, angel_ring_band, AngelRing, MAX_ANGEL_RINGS};
pub use hair_chiang::{evaluate_hair_chiang_direct, HairChiangParams};
pub use hair_fiber::{evaluate_hair_fiber_direct, HairFiberParams};
pub use hair_kajiya::{evaluate_hair_kajiya_direct, HairKajiyaParams};
pub use hair_matcap::{evaluate_hair_matcap, hair_matcap_uv, HairMatCapParams, MatCapTexture};
pub use halftone::{
    apply_halftone, cell_coord, dot_coverage, dot_radius_from_tone,
    luminance as halftone_luminance, rotate2d, smoothstep as halftone_smoothstep, HalftoneParams,
    HALFTONE_LUMA_WEIGHTS, HALFTONE_MAX_DOT_RADIUS,
};
pub use hatching::{
    apply_hatching, hatch_coverage, line_coverage, luminance as hatching_luminance,
    rotate2d as hatching_rotate2d, HatchingParams, HATCHING_LUMA_WEIGHTS,
};
pub use kuwahara::{
    apply_kuwahara, luminance as kuwahara_luminance, region_luma_variance, region_mean,
    select_min_variance, KuwaharaParams, KUWAHARA_LUMA_WEIGHTS,
};
pub use lens_flare::{
    accumulate_ghosts, apply_lens_flare, chromatic_ghost_offset, ghost_uv, halo_uv,
    luminance as lens_flare_luminance, mix3 as lens_flare_mix3, radial_weight,
    smoothstep as lens_flare_smoothstep, threshold_prefilter, vignette_weight, LensFlareParams,
    LENS_FLARE_CENTER, LENS_FLARE_LUMINANCE_WEIGHTS,
};
pub use light_routing::{
    cull_lights_by_channel, LightLayerMask, LightRouting, LightingChannelMask,
    MAX_LIGHTING_CHANNELS, MAX_LIGHT_LAYERS,
};
pub use lighting::{
    evaluate_principled_direct, evaluate_principled_direct_with, linear_furnace_response,
    glint_shade_seed, principled_pre, DirectLightSample, PrincipledPre, ShadingFrame,
    SurfaceSample,
};
pub use measured_lobe::evaluate_measured_direct;
pub use measured_brdf::{
    directions_to_rusinkiewicz, half_vector, is_non_negative, reciprocity_error,
    rusinkiewicz_to_directions, theta_h_to_unit, unit_to_theta_h, MeasuredBrdf, MeasuredBrdfDims,
    MeasuredBrdfError, MeasuredBrdfSample, MeasuredBrdfSampler, RusinkiewiczCoords,
    SyntheticBrdfParams, MB_DEFAULT_PHI_D, MB_DEFAULT_THETA_D, MB_DEFAULT_THETA_H,
};
pub use motion_blur::{
    clamp_velocity, cone, cylinder, neighbor_max, sample_weight, shutter_velocity,
    soft_depth_compare, tile_max, velocity_length, MotionBlurParams,
};
pub use oit::{composite_transparency, oit_weight, OitAccumulation, OitFragment};
pub use ordered_dither::{
    apply_ordered_dither, bayer4x4_threshold, quantize as ordered_dither_quantize,
    OrderedDitherParams, ORDERED_DITHER_BAYER_4X4,
};
pub use outline::{
    evaluate_outline, outline_depth_edge, outline_id_edge, outline_normal_edge, OutlineGeometry,
    OutlineParams,
};
pub use posterize::{
    apply_posterize, luminance as posterize_luminance, quantize as posterize_quantize,
    quantize_luma_preserving, quantize_rgb, PosterizeParams, POSTERIZE_LUMA_EPS,
    POSTERIZE_LUMA_WEIGHTS,
};
pub use punctual::PunctualLight;
pub use quality::{LobeSampleBudget, ShadingQuality};
pub use lobe_sampling::{
    lobe_blue_noise_1d, lobe_blue_noise_2d, lobe_clamp_unit, lobe_estimate,
    lobe_estimator_weight, lobe_full_sum, lobe_hash_combine, lobe_hash_u32, lobe_is_finite,
    lobe_make_sample, lobe_pixel_seed, lobe_r2_sample_2d, lobe_rotate_left_16,
    lobe_sample_quality_sum, lobe_sanitize_weight, lobe_select, lobe_select_blue_noise,
    lobe_temporal_stride, lobe_uniform_at, lobe_uniform_from_hash, lobe_weight_probability,
    lobe_write_probabilities, LobeSample, LobeSamplingMode, LobeSamplingPolicy,
    LOBE_SAMPLING_GOLDEN_STRIDE, LOBE_SAMPLING_MIN_WEIGHT, LOBE_SAMPLING_NO_LOBE,
    LOBE_SAMPLING_ONE_MINUS_EPSILON, LOBE_SAMPLING_R2_A1, LOBE_SAMPLING_R2_A2,
    LOBE_SAMPLING_SEED_SALT, LOBE_SAMPLING_SEQUENCE_MASK, LOBE_SAMPLING_SEQUENCE_PERIOD,
    LOBE_SAMPLING_TEMPORAL_SALT,
};

// Crate-root re-exports of the expensive-lobe `*_for_budget` adapters, so a
// host driving `ShadingQuality`/`LobeSampleBudget` can reach every tier-scaled
// entry point from one place (and the `quality` module's doc links resolve).
pub use gi::sky_lut::multiscatter_lut::bake_multiscatter_lut_for_budget;
pub use gi::subsurface::preintegrated::preintegrated_diffuse_for_budget;
pub use gi::translucency::diffusion::profile_radial_energy_for_budget as diffusion_profile_radial_energy_for_budget;
pub use resolve::{
    resolve_pixel, surface_sample_from_parameters, DirectionalLight, LightingEnvironment,
    ResolveError, ResolveInput, ResolvedPixel, IDENTITY_WORLD_FROM_LOCAL,
};
pub use screen_space::{
    accumulate_temporal, adaptive_history_weight, blend_specular, build_hemisphere_ray,
    build_screen_ray, clip_history_to_aabb, clip_history_to_aabb_ex, cosine_sample_direction,
    denoise_ssgi, denoise_ssgi_pixel, distance_fade, edge_fade, expand_bounds, facing_fade,
    gather_indirect_diffuse, ggx_ndf, hammersley, importance_sample_ggx, march_hierarchical,
    motion_vector, project_view_to_screen, project_world_to_screen, radical_inverse_vdc, reflect,
    reflection_mip, relax_box_for_confidence, reproject_prev_uv, reproject_prev_uv_motion,
    resolve_geometry_weight, resolve_reflection, reverse_z_perspective, roughness_fade,
    smith_ggx_visibility, smoothstep, trace_confidence, trace_indirect_ray,
    trace_screen_space_reflection, variance_clip_box, ClipResult, DepthPyramid, MotionSample,
    ScreenRay, ScreenSample, ScreenSpaceReflection, SsgiDenoiseBuffers, SsgiDenoiseConfig,
    SsgiGather, SsgiParams, SsgiRaySample, SsrCamera, SsrConfidenceParams, SsrMarchConfig,
    SsrMarchResult, SsrResolveParams, SsrResolveSample, SsrTemporalParams, SsrTraceSample,
};
pub use shadow::{
    allocate_shadow_atlas, apply_normal_offset, blocker_search, cascade_blend_weight,
    compute_cascade_matrices, compute_cascade_splits, cube_face_and_uv, cube_face_view_projections,
    evaluate_directional_shadow, evaluate_point_shadow, evaluate_spot_shadow, invert,
    pcf_visibility, pcss_visibility, plan_shadow_depth_draws, select_cascade,
    slope_scaled_depth_bias, spot_view_projection, transform_direction, transform_point,
    AtlasAllocation, AtlasConfig, AtlasSlot, BlockerSearch, CascadeMatrix, CascadeSplits,
    DirectionalShadowConfig, DirectionalShadowInput, Mat4, PcssConfig, PointShadowConfig,
    PointShadowInput, ShadowDepthDraw, ShadowDepthMode, ShadowDepthSampler, ShadowDepthView,
    ShadowFilter, ShadowKind, ShadowRequest, ShadowViewGeometry, SpotShadowConfig, SpotShadowInput,
    MAX_CASCADE_COUNT, POINT_LAYER_COUNT,
};
pub use shadow::{
    camera_move_invalidates_pages, decode_window_slot, filter_page_radius, generate_page_requests,
    generate_receiver, invalidate_casters, reconstruct_world_position, slot_to_page_key,
    window_slot, window_slot_count, window_slots_per_level, Allocation, AllocatorStats,
    BudgetStats, CasterMovement, ClipmapConfig, ClipmapLevel, FrameInput, FrameResult,
    Invalidation, PageRequestSet, PageTableStats, PhysicalPageAllocator, Receiver,
    ReceiverProjection, Residency, ShadowPageKey, VirtualPageTable, VirtualShadowMap,
    VirtualShadowSettings,
};
pub use stylized::{evaluate_stylized_direct, evaluate_toon_direct, StylizedParams};
pub use stylized_hair::{evaluate_stylized_hair_direct, StylizedHairParams};
pub use subsurface::evaluate_subsurface_direct;
pub use surface::{
    reconstruct_surface, GpuShadingPrimitive, GpuShadingVertex, SurfaceReconstructionError,
    SurfaceReconstructionFlags, SurfaceReconstructionInput, SurfaceSampleGeometry,
};
pub use taa::{
    halton, resolve_taa, rgb_to_ycocg, taa_jitter, tonemap_weight, ycocg_to_rgb, TaaParams,
    DEFAULT_TAA_JITTER_LEN,
};
pub use tangent::{
    apply_tangent_space_normal, orthonormal_basis, resolve_tangent_basis, TangentBasis,
};
pub use texture_lod::{
    in_finite_range as texture_lod_in_finite_range, texture_lod, triangle_lod,
    uv_gradients as texture_uv_gradients, FINITE_RANGE as TEXTURE_LOD_FINITE_RANGE,
    MIN_SCREEN_AREA,
};
pub use texture_sample::{
    decode_tangent_normal, fold_material_texel, sample_material, sampled_material_defaults,
    srgb_channel_to_linear, srgb_to_linear, MaterialModulationParams, SampledMaterial,
    SampledTextureBinding, SEMANTIC_BASE_COLOR, SEMANTIC_CLEAR_COAT, SEMANTIC_CLEAR_COAT_NORMAL,
    SEMANTIC_CLEAR_COAT_ROUGHNESS, SEMANTIC_EMISSIVE, SEMANTIC_METALLIC_ROUGHNESS, SEMANTIC_NORMAL,
    SEMANTIC_OCCLUSION,
};
pub use tonemap::{
    apply_tonemap, tonemap_aces_fitted, tonemap_aces_narkowicz, tonemap_agx, tonemap_agx_with_look,
    tonemap_reinhard, tonemap_reinhard_extended, AgxLook, TonemapOperator, TonemapParams,
    ACES_INPUT_MATRIX, ACES_OUTPUT_MATRIX, AGX_INPUT_MATRIX, AGX_MAX_EV, AGX_MIN_EV,
    AGX_OUTPUT_MATRIX,
};
pub use vignette::{
    apply_vignette, apply_vignette_params, artistic_falloff, natural_falloff, natural_vignette,
    vignette_factor, vignette_smoothstep, VignetteParams,
};
pub use visibility::{
    encode_barycentrics, BarycentricError, VisibilityPixel, VisibilityPixelTargets,
    INVALID_VISIBILITY_ID, VISIBILITY_BUFFER_ABI_VERSION,
};
pub use volumetrics::{
    froxel_source, henyey_greenstein, in_scatter, integrate_froxel_column, integrate_slice,
    transmittance, Froxel, MediumSample, VolumetricIntegration,
};
pub use water::evaluate_water_direct;

pub mod gi;
pub use gi::world_space::{
    bilinear_weights, blend_sh, cell_to_key, dir_to_oct, evaluate_irradiance,
    interpolate_irradiance, oct_to_dir, probe_bilinear_coords, probe_center_pixel, probe_coord,
    probe_count, probe_grid_dims, probe_index, probe_pixel_rect, resolve_weights,
    similarity_weight, world_to_cell, InterpolationConfig, PixelRect, ProbeNeighbor, RadianceCell,
    ShL1Rgb,
};
