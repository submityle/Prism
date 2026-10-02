//! Optional `wgpu` compute twin of Prism's volumetric-cloud scattering phase
//! functions.
//!
//! Cloud single scattering is driven by an anisotropic phase function that
//! biases light toward the forward direction (the silver-lining and glory
//! response). The `CPU` golden standard for that math lives in
//! [`prism_render_architecture::volumetric::scatter`]; this crate is the `GPU`
//! twin, validated against that reference so a passing real-device parity test
//! is direct evidence the ported kernel computes the same phase values as the
//! reference, not merely that its shader compiles.
//!
//! # Scope
//!
//! [`GpuPhaseEvaluator`] evaluates the full dual-lobe `HG`+`Draine` cloud phase
//! [`dual_lobe_draine_phase`](prism_render_architecture::volumetric::scatter::dual_lobe_draine_phase),
//! which internally composes the `Henyey-Greenstein`, `Draine` and `HG`-`Draine`
//! sub-phases, so a single kernel covers the whole phase stack the ray-march
//! and multi-scatter stages consume.
//!
//! [`GpuOctaveScatter`] evaluates the Wrenninge-style octave-scatter decay
//! [`octave_scatter`](prism_render_architecture::volumetric::scatter::octave_scatter),
//! the per-octave `(sigma_s, sigma_t, g)` geometric attenuation the same
//! multi-scatter stage sums.
//!
//! [`GpuModeling`] composes the final cloud density from the four
//! authored/weather modulators
//! ([`compose_from_modeling`](prism_render_architecture::volumetric::modeling::compose_from_modeling)):
//! the cloud-type blend, the coverage `remap`, the per-`CloudKind` `height`
//! gradient and the energy-preserving `detail erosion` `remap`, the shape half
//! of the density field the ray-march stage samples.
//!
//! # Portability
//!
//! The phase algebra uses only `sqrt`, `min`, `max` and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The phase functions contain no transcendental call, so `CPU` and `GPU`
//! evaluate the same closed-form algebra. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`), tight enough to catch a genuinely
//! wrong port yet loose enough to admit legal fma contraction. See
//! [`phase`] for the full rationale.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Henyey-Greenstein` / `Draine` (`Jendersie` and
//! `d'Eon` 2023) dual-lobe cloud phase plus `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.
#![forbid(unsafe_code)]

pub mod aabb_transform;
pub mod active_pixel;
pub mod adaptive_step;
pub mod advect_semi_lagrangian;
pub mod advect_with_wind;
pub mod aerial;
pub mod alpha_erosion;
pub mod alpha_hashed;
pub mod analytic_single_scatter;
pub mod analytic_transmittance;
pub mod anamorphic_streak;
pub mod anisotropic_footprint;
pub mod anvil_profile;
pub mod ao_sample;
pub mod aperture_blade;
pub mod apply_carve;
pub mod avsm_area;
pub mod avsm_transmittance;
pub mod backface_outline_expand;
pub mod barycentric_coord;
pub mod bicubic_sample;
pub mod billboard_atlas;
pub mod binary_search_range;
pub mod bit_pack_u32;
pub mod bit_reversal_u32;
pub mod bitonic_sort;
pub mod blend_state;
pub mod blend_with_atmosphere;
pub mod bloom_threshold;
pub mod bloom_upsample;
pub mod bounds;
pub mod camera;
pub mod capsule_capsule_closest;
pub mod capsule_sdf;
pub mod checkerboard_resolve;
pub mod chromatic_aberration;
pub mod cie_xyz;
pub mod clamp_history;
pub mod classify_precip;
pub mod closest_point_obb;
pub mod cloud_shadow_modulation;
pub mod cohen_sutherland_clip;
pub mod color_grade_lut;
pub mod color_gradient;
pub mod color_temperature;
pub mod composite_motion_vector;
pub mod conservative_raster;
pub mod contact_shadow;
pub mod context;
pub mod contrail_kernel;
pub mod contrail_spread;
pub mod convex_hull_2d;
pub mod convex_hull_3d;
pub mod counting_sort_u16;
pub mod curl;
pub mod curl_noise;
pub mod curves;
pub mod de_bruijn_log2;
pub mod decal;
pub mod deep_opacity_bake;
pub mod density_delta;
pub mod depth_downsample;
pub mod depth_linearize;
pub mod depth_of_field;
pub mod distance_field_shadow;
pub mod draw_pass_buffers;
pub mod dual_lobe_phase;
pub mod dual_quaternion;
pub mod ear_clip_triangulate;
pub mod edge_detect;
pub mod emitter_pass_buffers;
pub mod epa_penetration_3d;
pub mod event_pass_buffers;
pub mod exposure_adapt;
pub mod fibonacci_lfsr;
pub mod film_grain;
pub mod flipbook_blend;
pub mod fluid_diffusion;
pub mod fog;
pub mod forces;
pub mod fresnel_rim;
pub mod froxel_injection;
pub mod frustum_aabb_cull;
pub mod frustum_cull;
pub mod frustum_plane_extract;
pub mod fxaa;
pub mod gamut_clip;
pub mod gaussian_splat;
pub mod ggx_energy_compensation;
pub mod gjk_2d;
pub mod gjk_3d;
pub mod godray;
pub mod gpu_compact;
pub mod gpu_dispatch;
pub mod gpu_layout;
pub mod gpu_prefix_scan;
pub mod gpu_radix_histogram;
pub mod gpu_reduce;
pub mod gpu_scan_segmented;
pub mod gpu_scan_warp;
pub mod gpu_stream_append;
pub mod gravity_wave;
pub mod gray_code;
pub mod half_float_f16;
pub mod halton_sequence;
pub mod hash_rng;
pub mod heap_sort_u32;
pub mod heat_distortion;
pub mod height_fog;
pub mod hilbert_curve;
pub mod histogram_equalize;
pub mod hue_shift;
pub mod imposter_fade;
pub mod indirect_dispatch;
pub mod indirect_draw;
pub mod inertia_tensor;
pub mod integrate_segment;
pub mod interval_overlap_1d;
pub mod kawase_blur;
pub mod kawase_dual_blur;
pub mod lens_distortion;
pub mod liang_barsky_clip;
pub mod light_shaft;
pub mod line_line_closest_3d;
pub mod lod;
pub mod luminance_hist;
pub mod marching_cubes;
pub mod marching_squares;
pub mod mask;
pub mod matrix_decompose;
pub mod merge_sort_stable;
pub mod mesh_renderer;
pub mod microfacet_ggx;
pub mod minkowski_sum_2d;
pub mod modeling;
pub mod morton_code;
pub mod motion_blur;
pub mod motion_vectors;
pub mod multiscatter_lut_build;
pub mod multiscatter_lut_sample;
pub mod nd_strides_index;
pub mod noise;
pub mod noise_fbm;
pub mod normal_reconstruct;
pub mod obb_obb_sat_3d;
pub mod occlusion;
pub mod octave;
pub mod oren_nayar;
pub mod orientation_basis;
pub mod overshooting_bump;
pub mod ozone_absorption;
pub mod particle_multiscatter;
pub mod perlin;
pub mod perlin_worley;
pub mod phase;
pub mod pipeline_layout;
pub mod plane_aabb_classify;
pub mod plane_clip;
pub mod plane_line_intersect;
pub mod plucker_coord;
pub mod point_cache;
pub mod point_in_polygon;
pub mod point_in_tetrahedron;
pub mod point_triangle_closest_3d;
pub mod polygon_area_2d;
pub mod polyline_sdf_2d;
pub mod popcount_hamming;
pub mod powder;
pub mod premultiply_alpha;
pub mod probe_grid_sample;
pub mod pyrocumulus_buoyancy;
pub mod quaternion_rotate;
pub mod quickselect_u32;
pub mod radix_sort_u32;
pub mod ray_aabb;
pub mod ray_capsule;
pub mod ray_cone;
pub mod ray_cylinder;
pub mod ray_disk;
pub mod ray_obb;
pub mod ray_sphere;
pub mod ray_triangle;
pub mod rayleigh_phase;
pub mod raytrace;
pub mod reflect_refract_vec;
pub mod relax_coverage;
pub mod reservoir_sample;
pub mod rgb_ycocg;
pub mod rgbe_encode;
pub mod ribbon_geometry;
pub mod ribbon_trail;
pub mod ritter_bounding_sphere;
pub mod sat_collision_2d;
pub mod scanline_polygon_fill;
pub mod sdf;
pub mod segment_closest_point_3d;
pub mod segment_intersect_2d;
pub mod segment_obb_intersect;
pub mod segment_triangle_intersect;
pub mod select_lod;
pub mod serialization;
pub mod shading;
pub mod shadow;
pub mod sharpen_cas;
pub mod should_early_terminate;
pub mod should_fallback;
pub mod sim_pass_buffers;
pub mod sim_space;
pub mod single_scatter_reference;
pub mod sky_state_transition;
pub mod soft_particle;
pub mod sort_cull;
pub mod sort_pass_buffers;
pub mod spatial_hash;
pub mod spawn_pass_buffers;
pub mod spectral_to_rgb;
pub mod specular_aa;
pub mod sphere_aabb;
pub mod sphere_sweep;
pub mod spherical_harmonics_rotate;
pub mod spline;
pub mod sprite_stretch;
pub mod sss_wrap;
pub mod stability;
pub mod stencil_op;
pub mod storm_vertical_profile;
pub mod subframe_spawn;
pub mod sunset_inscatter_tint;
pub mod sunset_reddening;
pub mod sutherland_hodgman_2d;
pub mod sweep_aabb;
pub mod temporal_dither;
pub mod temporal_reproject;
pub mod terrain_occlusion;
pub mod tetrahedron_volume;
pub mod tiled_depth_sort;
pub mod time_control;
pub mod tonemap;
pub mod total_coverage;
pub mod tracking_transmittance;
pub mod transcendental_approx;
pub mod tri_tri_intersect;
pub mod triangle_aabb_overlap;
pub mod triangle_circumcircle;
pub mod trig_approx;
pub mod trilinear;
pub mod triplanar_blend;
pub mod unorm_snorm_pack;
pub mod uv_animation;
pub mod variance_clip;
pub mod variance_shadow;
pub mod vdb_sample;
pub mod vector_field;
pub mod velocity_at;
pub mod velocity_dilate;
pub mod vignette_mask;
pub mod virga_fade;
pub mod virga_veil;
pub mod volume_scene_shadow_cast;
pub mod volumetric_multiscatter;
pub mod vorticity_confinement;
pub mod voxel_traversal;
pub mod wind_field;
pub mod worley;
pub mod wu_line;
pub mod ycbcr_bt709;

pub use aabb_transform::{AabbTransformQuery, AabbTransformResult, GpuAabbTransform};
pub use active_pixel::{ActivePixelQuery, GpuActivePixel};
pub use adaptive_step::{AdaptiveStepQuery, GpuAdaptiveStep};
pub use advect_semi_lagrangian::{GpuAdvectSemiLagrangian, WeatherAdvectSample};
pub use advect_with_wind::GpuAdvectWithWind;
pub use aerial::{AerialQuery, GpuAerialPerspective};
pub use alpha_erosion::{AlphaErosionQuery, AlphaErosionResult, GpuAlphaErosion};
pub use alpha_hashed::{AlphaHashedQuery, AlphaHashedResult, GpuAlphaHashed};
pub use analytic_single_scatter::{AnalyticSingleScatterQuery, GpuAnalyticSingleScatter};
pub use analytic_transmittance::{AnalyticTransmittanceQuery, GpuAnalyticTransmittance};
pub use anamorphic_streak::{AnamorphicStreakQuery, AnamorphicStreakResult, GpuAnamorphicStreak};
pub use anisotropic_footprint::{FootprintQuery, FootprintResult, GpuAnisotropicFootprint};
pub use anvil_profile::{AnvilProfileQuery, GpuAnvilProfile};
pub use ao_sample::GpuAoSample;
pub use aperture_blade::{ApertureBladeQuery, ApertureBladeResult, GpuApertureBlade};
pub use apply_carve::{ApplyCarveQuery, GpuApplyCarve};
pub use avsm_area::GpuAvsmArea;
pub use avsm_transmittance::{AvsmSampleNode, GpuAvsmTransmittance};
pub use backface_outline_expand::{
    GpuBackfaceOutlineExpand, OutlineExpandResult, OutlineExpandVertex,
};
pub use barycentric_coord::{BarycentricQuery, BarycentricResult, GpuBarycentricCoord};
pub use bicubic_sample::GpuBicubicSample;
pub use billboard_atlas::{GpuBillboardAtlas, GpuBillboardAtlasQuery, GpuBillboardAtlasResult};
pub use binary_search_range::GpuBinarySearchRange;
pub use bit_pack_u32::{packed_len_words, GpuBitPackU32};
pub use bit_reversal_u32::{
    host_bit_reverse_increment, host_is_bit_reversal_palindrome, host_reverse_bits_u32,
    host_reverse_lowest_bits, GpuBitReversalU32,
};
pub use bitonic_sort::{GpuBitonicSort, GpuBitonicSortU32};
pub use blend_state::{BlendStateQuery, GpuBlendState};
pub use blend_with_atmosphere::{BlendQuery, BlendedColor, GpuBlendWithAtmosphere};
pub use bloom_threshold::{BloomThresholdQuery, GpuBloomThreshold};
pub use bloom_upsample::{BloomUpsampleQuery, GpuBloomUpsample};
pub use bounds::{GpuBounds, GpuBoundsQuery, GpuBoundsResult, AXIS_X, AXIS_Y, AXIS_Z};
pub use camera::{CameraQuery, CameraResult, GpuCamera};
pub use capsule_capsule_closest::{
    CapsuleClosestQuery, CapsuleClosestResult, GpuCapsuleCapsuleClosest,
};
pub use capsule_sdf::{CapsuleSdfQuery, CapsuleSdfResult, GpuCapsuleSdf};
pub use checkerboard_resolve::{CheckerboardResolveQuery, GpuCheckerboardResolve};
pub use chromatic_aberration::{
    ChromaticAberrationQuery, ChromaticAberrationResult, GpuChromaticAberration,
};
pub use cie_xyz::{CieXyzQuery, CieXyzResult, GpuCieXyz};
pub use clamp_history::{ClampHistoryQuery, GpuClampHistory};
pub use classify_precip::{ClassifyPrecipQuery, GpuClassifyPrecip};
pub use closest_point_obb::{ClosestPointObbQuery, ClosestPointObbResult, GpuClosestPointObb};
pub use cloud_shadow_modulation::{CloudShadowModulationQuery, GpuCloudShadowModulation};
pub use cohen_sutherland_clip::{
    ClipSegmentQuery, ClipSegmentResult, GpuCohenSutherlandClip, OUTCODE_BOTTOM, OUTCODE_INSIDE,
    OUTCODE_LEFT, OUTCODE_RIGHT, OUTCODE_TOP,
};
pub use color_grade_lut::{ColorGradeLutQuery, ColorGradeLutResult, GpuColorGradeLut};
pub use color_gradient::{ColorGradientQuery, ColorGradientSample, GpuColorGradient};
pub use color_temperature::{ColorTemperatureQuery, ColorTemperatureResult, GpuColorTemperature};
pub use composite_motion_vector::{
    CompositeMotionVectorQuery, GpuCompositeMotionVector, MotionVector,
};
pub use conservative_raster::{
    ConservativeRasterQuery, ConservativeRasterResult, GpuConservativeRaster,
};
pub use contact_shadow::{ContactShadowQuery, GpuContactShadow};
pub use context::{block_on, GpuContext};
pub use contrail_kernel::{ContrailKernelQuery, GpuContrailKernel};
pub use contrail_spread::{ContrailSpreadQuery, GpuContrailSpread};
pub use convex_hull_2d::{ConvexHull2dQuery, ConvexHull2dResult, GpuConvexHull2d};
pub use convex_hull_3d::{ConvexHull3dQuery, GpuConvexHull3d, GpuHull3d};
pub use counting_sort_u16::{CountingSortU16Query, GpuCountingSortU16};
pub use curl::{CurlQuery, GpuCurl};
pub use curl_noise::{CurlNoiseSample, GpuCurlNoise};
pub use curves::{CurveSampleQuery, GpuCurve, GpuKeyframe};
pub use de_bruijn_log2::{DeBruijnLog2Result, GpuDeBruijnLog2};
pub use decal::{DecalProjection, DecalQuery, GpuDecal};
pub use deep_opacity_bake::{march_centers, GpuDeepOpacityBake};
pub use density_delta::{CarveBrush, DensityDeltaQuery, GpuDensityDelta};
pub use depth_downsample::{DepthDownsampleQuery, GpuDepthDownsample};
pub use depth_linearize::{DepthLinearizeQuery, DepthLinearizeResult, GpuDepthLinearize};
pub use depth_of_field::{DepthOfFieldQuery, DepthOfFieldSample, GpuDepthOfField};
pub use distance_field_shadow::{GpuDistanceFieldShadow, GpuSdfGrid, SdfShadowRay};
pub use draw_pass_buffers::{GpuDrawPassBufferQuery, GpuDrawPassBufferResult, GpuDrawPassBuffers};
pub use dual_lobe_phase::{DualLobePhaseQuery, GpuDualLobePhase};
pub use dual_quaternion::{DualQuatTransformQuery, GpuDualQuaternion};
pub use ear_clip_triangulate::{EarClipAnswer, EarClipQuery, GpuEarClipTriangulate};
pub use edge_detect::{EdgeDetectOutput, EdgeDetectQuery, EdgeFrame, EdgeResponse, GpuEdgeDetect};
pub use emitter_pass_buffers::{
    GpuEmitterPassBufferQuery, GpuEmitterPassBufferResult, GpuEmitterPassBuffers,
};
pub use epa_penetration_3d::{EpaPenetration3dQuery, GpuEpa, GpuEpaPenetration3d};
pub use event_pass_buffers::{
    GpuEventPassBufferQuery, GpuEventPassBufferResult, GpuEventPassBuffers,
};
pub use exposure_adapt::{ExposureAdaptQuery, ExposureAdaptResult, GpuExposureAdapt};
pub use fibonacci_lfsr::GpuFibonacciLfsr;
pub use film_grain::{FilmGrainPixel, FilmGrainQuery, GpuFilmGrain};
pub use flipbook_blend::{FlipbookQuery, FlipbookResult, FlipbookSample, GpuFlipbookBlend};
pub use fluid_diffusion::{GpuDiffusionResult, GpuFluidDiffusion};
pub use fog::{FogQuery, GpuFogTransmittance};
pub use forces::{
    GpuForces, GpuForcesQuery, GpuForcesResult, FALLOFF_CONSTANT, FALLOFF_INVERSE_SQUARE,
    FALLOFF_LINEAR, FALLOFF_SMOOTHSTEP, FORCE_EXPLOSION, FORCE_GRAVITY_WELL, FORCE_IMPLOSION,
    FORCE_LINE_ATTRACTOR, FORCE_ORBITAL, FORCE_POINT_ATTRACTOR, FORCE_QUADRATIC_DRAG, FORCE_RADIAL,
    FORCE_SPRING_DAMPER, FORCE_TURBULENCE,
};
pub use fresnel_rim::{FresnelRimQuery, FresnelRimSample, GpuFresnelRim};
pub use froxel_injection::{FroxelInjectionQuery, GpuFroxelInjection};
pub use frustum_aabb_cull::{FrustumAabbCullPrimitive, FrustumAabbCullQuery, GpuFrustumAabbCull};
pub use frustum_cull::{FrustumCullPrimitive, FrustumCullQuery, GpuFrustumCull};
pub use frustum_plane_extract::GpuFrustumPlaneExtract;
pub use fxaa::{FxaaQuery, GpuFxaa};
pub use gamut_clip::{GamutClipMode, GamutClipQuery, GpuGamutClip};
pub use gaussian_splat::{
    GaussianSplatProjection, GpuGaussianSplat, GpuGaussianSplatQuery, SplatFootprint,
};
pub use ggx_energy_compensation::{
    GgxEnergyCompensationQuery, GgxEnergyCompensationResult, GpuGgxEnergyCompensation,
};
pub use gjk_2d::{GpuGjk2d, GpuGjk2dQuery, GpuGjk2dResult};
pub use gjk_3d::{GpuGjk3d, GpuGjk3dQuery, GpuGjk3dResult};
pub use godray::{GodRayWeightQuery, GpuGodRayWeight};
pub use gpu_compact::{GpuCompact, GpuCompactQuery, GpuCompactResult};
pub use gpu_dispatch::{GpuDispatch, GpuDispatchQuery, GpuDispatchResult};
pub use gpu_layout::{GpuLayout, GpuLayoutQuery, GpuLayoutResult};
pub use gpu_prefix_scan::{GpuPrefixScan, GpuPrefixScanQuery, GpuScanResult};
pub use gpu_radix_histogram::{GpuRadixHistogram, RadixHistogramQuery};
pub use gpu_reduce::{GpuReduce, GpuReduceInput, GpuReduceOp, GpuReduceQuery, GpuReduceResult};
pub use gpu_scan_segmented::{
    GpuSegmentedScan, GpuSegmentedScanConfig, GpuSegmentedScanQuery, GpuSegmentedScanResult,
};
pub use gpu_scan_warp::{
    GpuScanWarp, GpuWarpAggregateQuery, GpuWarpAppend, GpuWarpConfig, GpuWarpScanQuery,
    GpuWarpScanResult, DEFAULT_LANE_COUNT, WARP_LANE_LIMIT,
};
pub use gpu_stream_append::{
    GpuStreamAppend, GpuStreamAppendConfig, GpuStreamAppendCounter, GpuStreamAppendQuery,
    GpuStreamAppendResult, COUNTER_WORDS,
};
pub use gravity_wave::{GpuGravityWave, GravityWaveQuery};
pub use gray_code::GpuGrayCode;
pub use half_float_f16::{GpuHalfFloatF16, HalfFloatQuery, HalfFloatResult};
pub use halton_sequence::GpuHaltonSequence;
pub use hash_rng::{GpuHashRng, HashRngSample};
pub use heap_sort_u32::{GpuHeapSort, GpuHeapSortU32, HeapSortU32Query};
pub use heat_distortion::{GpuHeatDistortion, HeatQuery, HeatResult};
pub use height_fog::{GpuHeightFog, HeightFogQuery};
pub use hilbert_curve::{GpuHilbertCurve, MAX_ORDER};
pub use histogram_equalize::{
    GpuHistogramEqualize, HistogramEqualizeQuery, HistogramEqualizeResult,
};
pub use hue_shift::{GpuHueShift, HueShiftQuery, HueShiftResult};
pub use imposter_fade::{GpuImposterFade, ImposterFadeQuery};
pub use indirect_dispatch::{
    GpuIndirectDispatch, GpuIndirectDispatchQuery, GpuIndirectDispatchResult,
};
pub use indirect_draw::{GpuIndirectDraw, GpuIndirectDrawQuery, GpuIndirectDrawResult};
pub use inertia_tensor::{BodyOpQuery, BodyOpResult, GpuInertiaTensor};
pub use integrate_segment::{GpuIntegrateSegment, IntegrateSegmentQuery};
pub use interval_overlap_1d::{GpuIntervalOverlap1d, IntervalOverlapQuery, IntervalOverlapResult};
pub use kawase_blur::{GpuKawaseBlur, KawaseBlurQuery};
pub use kawase_dual_blur::{GpuKawaseDualBlur, GpuKawaseDualBlurQuery, GpuKawaseDualBlurResult};
pub use lens_distortion::GpuLensDistortion;
pub use liang_barsky_clip::{GpuLiangBarskyClip, LiangBarskyQuery, LiangBarskyResult, CLIP_EPS};
pub use light_shaft::{GpuLightShaft, GpuLightShaftParams, LightShaftResult};
pub use line_line_closest_3d::{GpuLineLineClosest3d, LineLineQuery, LineLineResult};
pub use lod::{GpuParticleLod, ParticleLodQuery, ParticleLodResult};
pub use luminance_hist::{GpuLuminanceHist, LuminanceHistQuery};
pub use marching_cubes::{GpuMarchingCubes, MarchingCubesCellQuery, MarchingCubesCellResult};
pub use marching_squares::{GpuMarchingSquares, MarchingSquaresQuery, MarchingSquaresResult};
pub use mask::{GpuScatteringMask, MaskQuery};
pub use matrix_decompose::{GpuMatrixDecompose, MatrixDecomposeQuery, MatrixDecomposeResult};
pub use merge_sort_stable::{GpuMergeSort, GpuMergeSortStable};
pub use mesh_renderer::{
    GpuMeshRenderer, GpuMeshRendererQuery, GpuMeshRendererResult, LOCAL_AXIS_PLUS_X,
    LOCAL_AXIS_PLUS_Y, LOCAL_AXIS_PLUS_Z, MAX_LOD_THRESHOLDS, ORIENTATION_ALIGN_TO_AXIS,
    ORIENTATION_FIXED_ROTATION, ORIENTATION_IDENTITY, ORIENTATION_VELOCITY_ALIGNED,
};
pub use microfacet_ggx::{GpuMicrofacetGgx, MicrofacetSample};
pub use minkowski_sum_2d::{GpuMinkowskiSum2d, MinkowskiSum2dQuery, MinkowskiSum2dResult};
pub use modeling::{GpuModeling, ModelingQuery};
pub use morton_code::GpuMortonCode;
pub use motion_blur::{GpuMotionBlur, MotionBlurQuery, MotionBlurResult};
pub use motion_vectors::{GpuMotionVectors, MotionVectorQuery};
pub use multiscatter_lut_build::GpuMultiScatterLutBuild;
pub use multiscatter_lut_sample::{GpuMultiScatterLutSample, MultiScatterSampleQuery};
pub use nd_strides_index::{GpuNdStridesIndex, NdStridesQuery, MAX_RANK};
pub use noise::{GpuFbmParams, GpuNoise, GpuTurbulenceParams, NoiseQuery, NoiseResult};
pub use noise_fbm::{GpuNoiseFbm, NoiseFbmQuery, NoiseFbmResult};
pub use normal_reconstruct::{GpuNormalReconstruct, NormalQuery, NormalResult};
pub use obb_obb_sat_3d::{GpuObbSat3d, ObbSat3dQuery, ObbSat3dResult};
pub use occlusion::{GpuOcclusion, GpuOcclusionQuery, GpuOcclusionResult};
pub use octave::{GpuOctaveScatter, OctaveQuery, OctaveResult};
pub use oren_nayar::{GpuOrenNayar, OrenNayarQuery, OrenNayarResult};
pub use orientation_basis::{GpuOrientationBasis, OrientationQuery};
pub use overshooting_bump::{GpuOvershootingBump, OvershootingBumpQuery};
pub use ozone_absorption::{GpuOzoneAbsorption, OzoneAbsorptionQuery};
pub use particle_multiscatter::{GpuParticleMultiScatter, MultiScatterResponse};
pub use perlin::{GpuPerlin, PerlinQuery};
pub use perlin_worley::{GpuPerlinWorley, PerlinWorleyQuery};
pub use phase::{GpuPhaseEvaluator, PhaseQuery};
pub use pipeline_layout::{GpuPipelineLayout, GpuPipelineLayoutQuery, GpuPipelineLayoutResult};
pub use plane_aabb_classify::{GpuPlaneAabbClassify, PlaneAabbClassifyQuery};
pub use plane_clip::{
    GpuPlaneClip, PlaneClipQuery, PlaneClipResult, SIDE_INSIDE, SIDE_ON, SIDE_OUTSIDE,
};
pub use plane_line_intersect::{
    GpuPlaneLineIntersect, PlaneLineQuery, PlaneLineResult, CLASS_COINCIDENT, CLASS_PARALLEL,
    CLASS_POINT,
};
pub use plucker_coord::{GpuPluckerCoord, PluckerQuery, PluckerResult};
pub use point_cache::{GpuPointCache, PointCacheQuery, PointCacheSample};
pub use point_in_polygon::{GpuPointInPolygon, PointInPolygonResult};
pub use point_in_tetrahedron::{
    GpuPointInTetrahedron, PointInTetrahedronQuery, PointInTetrahedronResult,
};
pub use point_triangle_closest_3d::{GpuPointTriangleClosest3d, PointTriangleQuery};
pub use polygon_area_2d::{GpuPolygonArea2d, GpuPolygonMetrics, MAX_POLYGON_VERTS};
pub use polyline_sdf_2d::{GpuPolylineSdf, GpuPolylineSdf2d, PolylineSdf2dQuery};
pub use popcount_hamming::GpuPopcountHamming;
pub use powder::{GpuPowder, PowderQuery};
pub use premultiply_alpha::GpuPremultiplyAlpha;
pub use probe_grid_sample::{GpuProbeGridSample, ProbeSampleQuery, PROBE_BANDS};
pub use pyrocumulus_buoyancy::{GpuPyrocumulusBuoyancy, PyrocumulusBuoyancyQuery};
pub use quaternion_rotate::{GpuQuaternionRotate, QuatRotateQuery, QuatRotateResult};
pub use quickselect_u32::{GpuQuickselect, GpuQuickselectU32, QuickselectQuery};
pub use radix_sort_u32::{GpuRadixSort, GpuRadixSortU32};
pub use ray_aabb::{GpuRayAabb, RayAabbQuery, RayAabbResult};
pub use ray_capsule::{GpuRayCapsule, RayCapsuleQuery, RayCapsuleResult};
pub use ray_cone::{GpuRayCone, RayConeQuery, RayConeResult};
pub use ray_cylinder::{GpuRayCylinder, RayCylinderQuery, RayCylinderResult};
pub use ray_disk::{GpuRayDisk, GpuRayDiskHit, RayDiskQuery, RayDiskTarget, FACE_BACK, FACE_FRONT};
pub use ray_obb::{GpuRayObb, RayObbQuery, RayObbResult};
pub use ray_sphere::{GpuRaySphere, RaySphereProbe, RaySphereResult};
pub use ray_triangle::{GpuRayTriangle, RayTriangleHit, RayTriangleQuery};
pub use rayleigh_phase::{GpuRayleighPhase, RayleighPhaseQuery};
pub use raytrace::{
    GpuRaytrace, GpuRaytraceQuery, GpuRaytraceResult, METHOD_DEPTH_BUFFER, METHOD_RAYTRACE,
    METHOD_SDF, QUALITY_HIGH, QUALITY_LOW, QUALITY_MEDIUM,
};
pub use reflect_refract_vec::{GpuReflectRefractVec, ReflectRefractQuery, ReflectRefractResult};
pub use relax_coverage::{GpuRelaxCoverage, RelaxCoverageQuery};
pub use reservoir_sample::{GpuReservoirSample, ReservoirValue};
pub use rgb_ycocg::GpuRgbYCoCg;
pub use rgbe_encode::{GpuRgbeEncode, RgbePrimQuery, RgbePrimResult};
pub use ribbon_geometry::{GpuRibbonGeometry, RibbonStripQuery};
pub use ribbon_trail::{GpuRibbonTrail, GpuRibbonTrailQuery, GpuRibbonTrailResult};
pub use ritter_bounding_sphere::{GpuRitterBoundingSphere, GpuRitterSphere, RitterQuery};
pub use sat_collision_2d::{GpuSatCollision2d, SatCollision2dQuery, SatCollision2dResult};
pub use scanline_polygon_fill::{GpuScanlineFill, GpuScanlinePolygonFill, GpuSpan};
pub use sdf::{GpuSignedDistanceField, SdfQuery, SdfResult};
pub use segment_closest_point_3d::{
    GpuSegmentClosestPoint3d, SegmentClosestQuery, SegmentClosestResult,
};
pub use segment_intersect_2d::{
    GpuSegmentIntersect2d, SegmentIntersectQuery, SegmentIntersectResult, CODE_COLLINEAR,
    CODE_NONE, CODE_POINT,
};
pub use segment_obb_intersect::{GpuSegmentObbIntersect, SegmentObbQuery, SegmentObbResult};
pub use segment_triangle_intersect::{GpuSegmentTriangleIntersect, SegmentTriangleQuery};
pub use select_lod::{GpuSelectLod, SelectLodQuery};
pub use serialization::{
    GpuSerialization, GpuSerializationQuery, GpuSerializationResult, CODE_COMPATIBLE,
    CODE_INCOMPATIBLE, CODE_NEEDS_MIGRATION,
};
pub use shading::{GpuShading, GpuShadingQuery, GpuShadingResult};
pub use shadow::{GpuShadow, ShadowRay};
pub use sharpen_cas::{GpuSharpenCas, SharpenCasQuery};
pub use should_early_terminate::{GpuShouldEarlyTerminate, ShouldEarlyTerminateQuery};
pub use should_fallback::{GpuShouldFallback, ShouldFallbackQuery};
pub use sim_pass_buffers::{GpuSimPassBuffers, GpuSimPassBuffersQuery, GpuSimPassBuffersResult};
pub use sim_space::{GpuSimSpace, GpuSimSpaceQuery, GpuSimSpaceResult};
pub use single_scatter_reference::{GpuSingleScatterReference, SingleScatterReferenceQuery};
pub use sky_state_transition::{GpuSkyStateTransition, SkyStateTransition};
pub use soft_particle::{GpuSoftParticle, LinearizeQuery, SoftParticleQuery, SoftParticleSample};
pub use sort_cull::{GpuSortCull, GpuSortCullQuery, GpuSortCullResult};
pub use sort_pass_buffers::{GpuSortPassBufferQuery, GpuSortPassBufferResult, GpuSortPassBuffers};
pub use spatial_hash::GpuSpatialHash;
pub use spawn_pass_buffers::{
    GpuSpawnPassBuffers, GpuSpawnPassBuffersQuery, GpuSpawnPassBuffersResult,
};
pub use spectral_to_rgb::{GpuSpectralToRgb, SpectralRgb};
pub use specular_aa::{
    GpuSpecularAa, SpecularAaBatchQuery, SpecularAaScalarQuery, SpecularAaScalarSample,
    SpecularAaScalars,
};
pub use sphere_aabb::{GpuSphereAabb, SphereAabbQuery};
pub use sphere_sweep::{GpuSphereSweep, SphereSweepOp, SphereSweepQuery};
pub use spherical_harmonics_rotate::{
    GpuSphericalHarmonicsRotate, ShRotationProbe, ShRotationResult, L2_COEFF_COUNT,
};
pub use spline::{GpuSpline, SplineEvalQuery, SplineEvalResult};
pub use sprite_stretch::{GpuSpriteStretch, SpriteStretchQuery, SpriteStretchResult};
pub use sss_wrap::{GpuSssWrap, SssWrapQuery, SssWrapSample};
pub use stability::{GpuStability, GpuStabilityQuery, GpuStabilityResult};
pub use stencil_op::{GpuStencilOp, StencilOpQuery, StencilOpResult};
pub use storm_vertical_profile::{GpuStormVerticalProfile, StormVerticalProfileQuery};
pub use subframe_spawn::{GpuSubframeSpawn, SubframeSpawnQuery, SubframeSpawnResult};
pub use sunset_inscatter_tint::{GpuSunsetInscatterTint, InscatterTint, SunsetInscatterTintQuery};
pub use sunset_reddening::{GpuSunsetReddening, SunsetReddeningQuery};
pub use sutherland_hodgman_2d::{
    GpuSutherlandHodgman2d, SutherlandHodgman2dQuery, SutherlandHodgman2dResult,
};
pub use sweep_aabb::{GpuSweepAabb, SweepAabbQuery};
pub use temporal_dither::{DitherPixel, DitherQuery, DitherSample, GpuTemporalDither};
pub use temporal_reproject::{
    GpuTemporalReproject, TemporalReprojectQuery, TemporalReprojectResult, NEIGHBORHOOD_TAPS,
};
pub use terrain_occlusion::{GpuTerrainOcclusion, TerrainOcclusionQuery};
pub use tetrahedron_volume::{
    GpuTetrahedronVolume, TetrahedronVolumeQuery, TetrahedronVolumeResult,
};
pub use tiled_depth_sort::{GpuTileOf, TileOfQuery, TileOfResult};
pub use time_control::{GpuTimeControl, TimeControlQuery, TimeControlResult};
pub use tonemap::{GpuTonemap, TonemapQuery, TonemapResult};
pub use total_coverage::GpuTotalCoverage;
pub use tracking_transmittance::{
    GpuTrackingTransmittance, TrackingEstimate, TrackingTransmittanceQuery,
};
pub use transcendental_approx::{GpuTranscendental, TranscendentalQuery, TranscendentalResult};
pub use tri_tri_intersect::{GpuTriTriIntersect, TriTriQuery};
pub use triangle_aabb_overlap::{GpuTriangleAabbOverlap, TriangleAabbQuery};
pub use triangle_circumcircle::{GpuTriangleCircumcircle, TriangleCircumcircleResult};
pub use trig_approx::{GpuTrigApprox, TrigApproxQuery, TrigApproxResult};
pub use trilinear::{GpuTrilinear, TrilinearQuery};
pub use triplanar_blend::{GpuTriplanarBlend, TriplanarBlendQuery, TriplanarBlendResult};
pub use unorm_snorm_pack::GpuUnormSnormPack;
pub use uv_animation::{GpuUvAnimation, UvAnimationQuery, UvAnimationResult};
pub use variance_clip::{GpuVarianceClip, VarianceClipQuery};
pub use variance_shadow::{GpuVarianceShadow, VarianceShadowQuery};
pub use vdb_sample::GpuVdbSample;
pub use vector_field::{GpuVectorField, VectorFieldQuery, VectorFieldResult};
pub use velocity_at::{GpuVelocityAt, VelocityAtQuery};
pub use velocity_dilate::GpuVelocityDilate;
pub use vignette_mask::{GpuVignetteMask, VignetteMaskQuery, VignetteMaskResult};
pub use virga_fade::{GpuVirgaFade, VirgaFadeQuery};
pub use virga_veil::{GpuVirgaVeil, VirgaVeilQuery};
pub use volume_scene_shadow_cast::{GpuVolumeShadowCast, ShadowMarch, VolumeShadowRay};
pub use volumetric_multiscatter::{
    GpuVolumetricMultiScatter, MultiScatterQuery, VolumetricMultiScatterResponse,
};
pub use vorticity_confinement::{GpuVorticityConfinement, VorticityResult};
pub use voxel_traversal::{GpuVoxelTraversal, VoxelTraversalQuery, VoxelTraversalResult};
pub use wind_field::{GpuWindField, WindFieldQuery, WindFieldResult};
pub use worley::{GpuWorley, WorleyQuery};
pub use wu_line::{GpuWuLineRasterizer, WuLineQuery, WuLineResult, WuPixel};
pub use ycbcr_bt709::{GpuYcbcrBt709, YcbcrOp, YcbcrQuery, YcbcrResult};
