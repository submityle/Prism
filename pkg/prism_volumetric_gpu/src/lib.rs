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

pub mod active_pixel;
pub mod adaptive_step;
pub mod advect_semi_lagrangian;
pub mod advect_with_wind;
pub mod aerial;
pub mod analytic_single_scatter;
pub mod analytic_transmittance;
pub mod anvil_profile;
pub mod ao_sample;
pub mod apply_carve;
pub mod avsm_area;
pub mod avsm_transmittance;
pub mod blend_state;
pub mod blend_with_atmosphere;
pub mod bloom_threshold;
pub mod bloom_upsample;
pub mod checkerboard_resolve;
pub mod clamp_history;
pub mod classify_precip;
pub mod cloud_shadow_modulation;
pub mod composite_motion_vector;
pub mod contact_shadow;
pub mod context;
pub mod contrail_kernel;
pub mod contrail_spread;
pub mod curl;
pub mod curl_noise;
pub mod deep_opacity_bake;
pub mod density_delta;
pub mod depth_downsample;
pub mod distance_field_shadow;
pub mod dual_lobe_phase;
pub mod edge_detect;
pub mod film_grain;
pub mod flipbook_blend;
pub mod fluid_diffusion;
pub mod fog;
pub mod froxel_injection;
pub mod frustum_cull;
pub mod fxaa;
pub mod gamut_clip;
pub mod gaussian_splat;
pub mod godray;
pub mod gravity_wave;
pub mod hash_rng;
pub mod height_fog;
pub mod imposter_fade;
pub mod integrate_segment;
pub mod kawase_blur;
pub mod lens_distortion;
pub mod luminance_hist;
pub mod mask;
pub mod microfacet_ggx;
pub mod modeling;
pub mod motion_blur;
pub mod motion_vectors;
pub mod multiscatter_lut_build;
pub mod multiscatter_lut_sample;
pub mod noise_fbm;
pub mod normal_reconstruct;
pub mod octave;
pub mod overshooting_bump;
pub mod ozone_absorption;
pub mod particle_multiscatter;
pub mod perlin;
pub mod perlin_worley;
pub mod phase;
pub mod powder;
pub mod premultiply_alpha;
pub mod probe_grid_sample;
pub mod pyrocumulus_buoyancy;
pub mod rayleigh_phase;
pub mod relax_coverage;
pub mod rgb_ycocg;
pub mod select_lod;
pub mod shadow;
pub mod sharpen_cas;
pub mod should_early_terminate;
pub mod should_fallback;
pub mod single_scatter_reference;
pub mod sky_state_transition;
pub mod soft_particle;
pub mod spectral_to_rgb;
pub mod spherical_harmonics_rotate;
pub mod storm_vertical_profile;
pub mod sunset_inscatter_tint;
pub mod sunset_reddening;
pub mod temporal_reproject;
pub mod terrain_occlusion;
pub mod total_coverage;
pub mod tracking_transmittance;
pub mod transcendental_approx;
pub mod trig_approx;
pub mod trilinear;
pub mod variance_clip;
pub mod variance_shadow;
pub mod vdb_sample;
pub mod velocity_at;
pub mod velocity_dilate;
pub mod virga_fade;
pub mod virga_veil;
pub mod volume_scene_shadow_cast;
pub mod vorticity_confinement;
pub mod worley;

pub use active_pixel::{ActivePixelQuery, GpuActivePixel};
pub use adaptive_step::{AdaptiveStepQuery, GpuAdaptiveStep};
pub use advect_semi_lagrangian::{GpuAdvectSemiLagrangian, WeatherAdvectSample};
pub use advect_with_wind::GpuAdvectWithWind;
pub use aerial::{AerialQuery, GpuAerialPerspective};
pub use analytic_single_scatter::{AnalyticSingleScatterQuery, GpuAnalyticSingleScatter};
pub use analytic_transmittance::{AnalyticTransmittanceQuery, GpuAnalyticTransmittance};
pub use anvil_profile::{AnvilProfileQuery, GpuAnvilProfile};
pub use ao_sample::GpuAoSample;
pub use apply_carve::{ApplyCarveQuery, GpuApplyCarve};
pub use avsm_area::GpuAvsmArea;
pub use avsm_transmittance::{AvsmSampleNode, GpuAvsmTransmittance};
pub use blend_state::{BlendStateQuery, GpuBlendState};
pub use blend_with_atmosphere::{BlendQuery, BlendedColor, GpuBlendWithAtmosphere};
pub use bloom_threshold::{BloomThresholdQuery, GpuBloomThreshold};
pub use bloom_upsample::{BloomUpsampleQuery, GpuBloomUpsample};
pub use checkerboard_resolve::{CheckerboardResolveQuery, GpuCheckerboardResolve};
pub use clamp_history::{ClampHistoryQuery, GpuClampHistory};
pub use classify_precip::{ClassifyPrecipQuery, GpuClassifyPrecip};
pub use cloud_shadow_modulation::{CloudShadowModulationQuery, GpuCloudShadowModulation};
pub use composite_motion_vector::{
    CompositeMotionVectorQuery, GpuCompositeMotionVector, MotionVector,
};
pub use contact_shadow::{ContactShadowQuery, GpuContactShadow};
pub use context::{block_on, GpuContext};
pub use contrail_kernel::{ContrailKernelQuery, GpuContrailKernel};
pub use contrail_spread::{ContrailSpreadQuery, GpuContrailSpread};
pub use curl::{CurlQuery, GpuCurl};
pub use curl_noise::{CurlNoiseSample, GpuCurlNoise};
pub use deep_opacity_bake::{march_centers, GpuDeepOpacityBake};
pub use density_delta::{CarveBrush, DensityDeltaQuery, GpuDensityDelta};
pub use depth_downsample::{DepthDownsampleQuery, GpuDepthDownsample};
pub use distance_field_shadow::{GpuDistanceFieldShadow, GpuSdfGrid, SdfShadowRay};
pub use dual_lobe_phase::{DualLobePhaseQuery, GpuDualLobePhase};
pub use edge_detect::{EdgeDetectOutput, EdgeDetectQuery, EdgeFrame, EdgeResponse, GpuEdgeDetect};
pub use film_grain::{FilmGrainPixel, FilmGrainQuery, GpuFilmGrain};
pub use flipbook_blend::{FlipbookQuery, FlipbookResult, FlipbookSample, GpuFlipbookBlend};
pub use fluid_diffusion::{GpuDiffusionResult, GpuFluidDiffusion};
pub use fog::{FogQuery, GpuFogTransmittance};
pub use froxel_injection::{FroxelInjectionQuery, GpuFroxelInjection};
pub use frustum_cull::{FrustumCullPrimitive, FrustumCullQuery, GpuFrustumCull};
pub use fxaa::{FxaaQuery, GpuFxaa};
pub use gamut_clip::{GamutClipMode, GamutClipQuery, GpuGamutClip};
pub use gaussian_splat::{
    GaussianSplatProjection, GpuGaussianSplat, GpuGaussianSplatQuery, SplatFootprint,
};
pub use godray::{GodRayWeightQuery, GpuGodRayWeight};
pub use gravity_wave::{GpuGravityWave, GravityWaveQuery};
pub use hash_rng::{GpuHashRng, HashRngSample};
pub use height_fog::{GpuHeightFog, HeightFogQuery};
pub use imposter_fade::{GpuImposterFade, ImposterFadeQuery};
pub use integrate_segment::{GpuIntegrateSegment, IntegrateSegmentQuery};
pub use kawase_blur::{GpuKawaseBlur, KawaseBlurQuery};
pub use lens_distortion::GpuLensDistortion;
pub use luminance_hist::{GpuLuminanceHist, LuminanceHistQuery};
pub use mask::{GpuScatteringMask, MaskQuery};
pub use microfacet_ggx::{GpuMicrofacetGgx, MicrofacetSample};
pub use modeling::{GpuModeling, ModelingQuery};
pub use motion_blur::{GpuMotionBlur, MotionBlurQuery, MotionBlurResult};
pub use motion_vectors::{GpuMotionVectors, MotionVectorQuery};
pub use multiscatter_lut_build::GpuMultiScatterLutBuild;
pub use multiscatter_lut_sample::{GpuMultiScatterLutSample, MultiScatterSampleQuery};
pub use noise_fbm::{GpuNoiseFbm, NoiseFbmQuery, NoiseFbmResult};
pub use normal_reconstruct::{GpuNormalReconstruct, NormalQuery, NormalResult};
pub use octave::{GpuOctaveScatter, OctaveQuery, OctaveResult};
pub use overshooting_bump::{GpuOvershootingBump, OvershootingBumpQuery};
pub use ozone_absorption::{GpuOzoneAbsorption, OzoneAbsorptionQuery};
pub use particle_multiscatter::{GpuParticleMultiScatter, MultiScatterResponse};
pub use perlin::{GpuPerlin, PerlinQuery};
pub use perlin_worley::{GpuPerlinWorley, PerlinWorleyQuery};
pub use phase::{GpuPhaseEvaluator, PhaseQuery};
pub use powder::{GpuPowder, PowderQuery};
pub use premultiply_alpha::GpuPremultiplyAlpha;
pub use probe_grid_sample::{GpuProbeGridSample, ProbeSampleQuery, PROBE_BANDS};
pub use pyrocumulus_buoyancy::{GpuPyrocumulusBuoyancy, PyrocumulusBuoyancyQuery};
pub use rayleigh_phase::{GpuRayleighPhase, RayleighPhaseQuery};
pub use relax_coverage::{GpuRelaxCoverage, RelaxCoverageQuery};
pub use rgb_ycocg::GpuRgbYCoCg;
pub use select_lod::{GpuSelectLod, SelectLodQuery};
pub use shadow::{GpuShadow, ShadowRay};
pub use sharpen_cas::{GpuSharpenCas, SharpenCasQuery};
pub use should_early_terminate::{GpuShouldEarlyTerminate, ShouldEarlyTerminateQuery};
pub use should_fallback::{GpuShouldFallback, ShouldFallbackQuery};
pub use single_scatter_reference::{GpuSingleScatterReference, SingleScatterReferenceQuery};
pub use sky_state_transition::{GpuSkyStateTransition, SkyStateTransition};
pub use soft_particle::{GpuSoftParticle, LinearizeQuery, SoftParticleQuery, SoftParticleSample};
pub use spectral_to_rgb::{GpuSpectralToRgb, SpectralRgb};
pub use spherical_harmonics_rotate::{
    GpuSphericalHarmonicsRotate, ShRotationProbe, ShRotationResult, L2_COEFF_COUNT,
};
pub use storm_vertical_profile::{GpuStormVerticalProfile, StormVerticalProfileQuery};
pub use sunset_inscatter_tint::{GpuSunsetInscatterTint, InscatterTint, SunsetInscatterTintQuery};
pub use sunset_reddening::{GpuSunsetReddening, SunsetReddeningQuery};
pub use temporal_reproject::{
    GpuTemporalReproject, TemporalReprojectQuery, TemporalReprojectResult, NEIGHBORHOOD_TAPS,
};
pub use terrain_occlusion::{GpuTerrainOcclusion, TerrainOcclusionQuery};
pub use total_coverage::GpuTotalCoverage;
pub use tracking_transmittance::{
    GpuTrackingTransmittance, TrackingEstimate, TrackingTransmittanceQuery,
};
pub use transcendental_approx::{GpuTranscendental, TranscendentalQuery, TranscendentalResult};
pub use trig_approx::{GpuTrigApprox, TrigApproxQuery, TrigApproxResult};
pub use trilinear::{GpuTrilinear, TrilinearQuery};
pub use variance_clip::{GpuVarianceClip, VarianceClipQuery};
pub use variance_shadow::{GpuVarianceShadow, VarianceShadowQuery};
pub use vdb_sample::GpuVdbSample;
pub use velocity_at::{GpuVelocityAt, VelocityAtQuery};
pub use velocity_dilate::GpuVelocityDilate;
pub use virga_fade::{GpuVirgaFade, VirgaFadeQuery};
pub use virga_veil::{GpuVirgaVeil, VirgaVeilQuery};
pub use volume_scene_shadow_cast::{GpuVolumeShadowCast, ShadowMarch, VolumeShadowRay};
pub use vorticity_confinement::{GpuVorticityConfinement, VorticityResult};
pub use worley::{GpuWorley, WorleyQuery};
