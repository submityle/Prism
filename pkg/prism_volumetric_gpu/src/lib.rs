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
pub mod aerial;
pub mod analytic_single_scatter;
pub mod analytic_transmittance;
pub mod anvil_profile;
pub mod apply_carve;
pub mod blend_state;
pub mod blend_with_atmosphere;
pub mod clamp_history;
pub mod classify_precip;
pub mod cloud_shadow_modulation;
pub mod composite_motion_vector;
pub mod context;
pub mod contrail_kernel;
pub mod contrail_spread;
pub mod curl;
pub mod density_delta;
pub mod fog;
pub mod froxel_injection;
pub mod godray;
pub mod gravity_wave;
pub mod hash_rng;
pub mod height_fog;
pub mod imposter_fade;
pub mod integrate_segment;
pub mod mask;
pub mod modeling;
pub mod octave;
pub mod overshooting_bump;
pub mod ozone_absorption;
pub mod perlin;
pub mod perlin_worley;
pub mod phase;
pub mod powder;
pub mod pyrocumulus_buoyancy;
pub mod rayleigh_phase;
pub mod relax_coverage;
pub mod select_lod;
pub mod shadow;
pub mod should_early_terminate;
pub mod should_fallback;
pub mod sky_state_transition;
pub mod spectral_to_rgb;
pub mod storm_vertical_profile;
pub mod sunset_inscatter_tint;
pub mod sunset_reddening;
pub mod terrain_occlusion;
pub mod trilinear;
pub mod variance_clip;
pub mod velocity_at;
pub mod virga_fade;
pub mod virga_veil;
pub mod worley;

pub use active_pixel::{ActivePixelQuery, GpuActivePixel};
pub use adaptive_step::{AdaptiveStepQuery, GpuAdaptiveStep};
pub use aerial::{AerialQuery, GpuAerialPerspective};
pub use analytic_single_scatter::{AnalyticSingleScatterQuery, GpuAnalyticSingleScatter};
pub use analytic_transmittance::{AnalyticTransmittanceQuery, GpuAnalyticTransmittance};
pub use anvil_profile::{AnvilProfileQuery, GpuAnvilProfile};
pub use apply_carve::{ApplyCarveQuery, GpuApplyCarve};
pub use blend_state::{BlendStateQuery, GpuBlendState};
pub use blend_with_atmosphere::{BlendQuery, BlendedColor, GpuBlendWithAtmosphere};
pub use clamp_history::{ClampHistoryQuery, GpuClampHistory};
pub use classify_precip::{ClassifyPrecipQuery, GpuClassifyPrecip};
pub use cloud_shadow_modulation::{CloudShadowModulationQuery, GpuCloudShadowModulation};
pub use composite_motion_vector::{
    CompositeMotionVectorQuery, GpuCompositeMotionVector, MotionVector,
};
pub use context::{block_on, GpuContext};
pub use contrail_kernel::{ContrailKernelQuery, GpuContrailKernel};
pub use contrail_spread::{ContrailSpreadQuery, GpuContrailSpread};
pub use curl::{CurlQuery, GpuCurl};
pub use density_delta::{CarveBrush, DensityDeltaQuery, GpuDensityDelta};
pub use fog::{FogQuery, GpuFogTransmittance};
pub use froxel_injection::{FroxelInjectionQuery, GpuFroxelInjection};
pub use godray::{GodRayWeightQuery, GpuGodRayWeight};
pub use gravity_wave::{GpuGravityWave, GravityWaveQuery};
pub use hash_rng::{GpuHashRng, HashRngSample};
pub use height_fog::{GpuHeightFog, HeightFogQuery};
pub use imposter_fade::{GpuImposterFade, ImposterFadeQuery};
pub use integrate_segment::{GpuIntegrateSegment, IntegrateSegmentQuery};
pub use mask::{GpuScatteringMask, MaskQuery};
pub use modeling::{GpuModeling, ModelingQuery};
pub use octave::{GpuOctaveScatter, OctaveQuery, OctaveResult};
pub use overshooting_bump::{GpuOvershootingBump, OvershootingBumpQuery};
pub use ozone_absorption::{GpuOzoneAbsorption, OzoneAbsorptionQuery};
pub use perlin::{GpuPerlin, PerlinQuery};
pub use perlin_worley::{GpuPerlinWorley, PerlinWorleyQuery};
pub use phase::{GpuPhaseEvaluator, PhaseQuery};
pub use powder::{GpuPowder, PowderQuery};
pub use pyrocumulus_buoyancy::{GpuPyrocumulusBuoyancy, PyrocumulusBuoyancyQuery};
pub use rayleigh_phase::{GpuRayleighPhase, RayleighPhaseQuery};
pub use relax_coverage::{GpuRelaxCoverage, RelaxCoverageQuery};
pub use select_lod::{GpuSelectLod, SelectLodQuery};
pub use shadow::{GpuShadow, ShadowRay};
pub use should_early_terminate::{GpuShouldEarlyTerminate, ShouldEarlyTerminateQuery};
pub use should_fallback::{GpuShouldFallback, ShouldFallbackQuery};
pub use sky_state_transition::{GpuSkyStateTransition, SkyStateTransition};
pub use spectral_to_rgb::{GpuSpectralToRgb, SpectralRgb};
pub use storm_vertical_profile::{GpuStormVerticalProfile, StormVerticalProfileQuery};
pub use sunset_inscatter_tint::{GpuSunsetInscatterTint, InscatterTint, SunsetInscatterTintQuery};
pub use sunset_reddening::{GpuSunsetReddening, SunsetReddeningQuery};
pub use terrain_occlusion::{GpuTerrainOcclusion, TerrainOcclusionQuery};
pub use trilinear::{GpuTrilinear, TrilinearQuery};
pub use variance_clip::{GpuVarianceClip, VarianceClipQuery};
pub use velocity_at::{GpuVelocityAt, VelocityAtQuery};
pub use virga_fade::{GpuVirgaFade, VirgaFadeQuery};
pub use virga_veil::{GpuVirgaVeil, VirgaVeilQuery};
pub use worley::{GpuWorley, WorleyQuery};
