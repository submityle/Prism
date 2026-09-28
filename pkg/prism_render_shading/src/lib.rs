//! Backend-neutral visibility-buffer and shading-work contracts.

#![forbid(unsafe_code)]
#![expect(
    missing_docs,
    reason = "The experimental shading ABI is documented as it freezes."
)]

extern crate alloc;

mod ao;
mod classification;
mod cluster;
mod clearcoat;
mod cloth;
mod environment;
mod hair;
mod lighting;
mod oit;
mod punctual;
mod resolve;
mod shadow;
mod subsurface;
mod surface;
mod tangent;
mod texture_sample;
mod vecmath;
mod visibility;
mod water;

pub use classification::{
    classify_material_header, ClassificationError, MaterialShadingClass, ShadingWorkItem,
    ShadingWorkPlan, MAX_SHADING_CLASSES,
};
pub use ao::{
    compute_gtao, gtao_pixel, GtaoBuffers, GtaoCamera, GtaoConfig,
};
pub use cluster::{
    assign_lights_to_clusters, ClusterAabb, ClusterAssignmentConfig, ClusterBoundsBuilder,
    ClusterGrid, ClusterLightAssignment,
};
pub use clearcoat::evaluate_clearcoat_direct;
pub use cloth::evaluate_cloth_direct;
pub use environment::{
    env_brdf_approx, evaluate_image_based_light, integrate_brdf, prefilter_radiance,
    project_cubemap_to_sh, CubemapFaces, DfgLut, ImageBasedLight, PrefilteredEnvMap,
    SphericalHarmonicsL2,
};
pub use hair::evaluate_hair_direct;
pub use oit::{
    composite_transparency, oit_weight, OitAccumulation, OitFragment,
};
pub use lighting::{
    evaluate_principled_direct, evaluate_toon_direct, linear_furnace_response, DirectLightSample,
    ShadingFrame, SurfaceSample,
};
pub use punctual::PunctualLight;
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
    ResolveError, ResolveInput, ResolvedPixel,
};
pub use subsurface::evaluate_subsurface_direct;
pub use water::evaluate_water_direct;
pub use tangent::{
    apply_tangent_space_normal, orthonormal_basis, resolve_tangent_basis, TangentBasis,
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
