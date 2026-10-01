//! `wgpu` compute twin of the specular anti-aliasing (`Toksvig` / `Frostbite`)
//! `CPU` gold standard for `Prism` particle shading
//! ([`specular_aa`](prism_render_architecture::particle::specular_aa), design
//! sections 16-17).
//!
//! High-gloss surfaces alias under minification: a shaded `texel` covers many
//! microscopic normals whose highlights flicker. The `CPU` golden
//! [`specular_aa`](prism_render_architecture::particle::specular_aa) suppresses
//! that flicker by *widening* the effective specular lobe rather than narrowing
//! it; [`GpuSpecularAa`] is the on-device twin that reproduces the same scalar
//! roughness response one thread per sample, so a passing real-device parity
//! test is direct evidence the ported kernels evaluate the same closed form the
//! reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Two entry points mirror the golden module:
//!
//! 1. `evaluate_main` is the headline batch evaluator. For each sample's average
//!    normal length it reproduces
//!    [`SpecularAaParams::evaluate`](prism_render_architecture::particle::specular_aa::SpecularAaParams::evaluate),
//!    returning the four
//!    [`SpecularAaResult`](prism_render_architecture::particle::specular_aa::SpecularAaResult)
//!    scalars (`linear_roughness`, `perceptual_roughness`, `toksvig_factor`,
//!    `added_variance`). It transitively exercises
//!    [`normal_length_variance`](prism_render_architecture::particle::specular_aa::normal_length_variance),
//!    [`perceptual_to_linear_roughness`](prism_render_architecture::particle::specular_aa::perceptual_to_linear_roughness),
//!    [`frostbite_specular_aa`](prism_render_architecture::particle::specular_aa::frostbite_specular_aa),
//!    [`linear_to_perceptual_roughness`](prism_render_architecture::particle::specular_aa::linear_to_perceptual_roughness)
//!    and
//!    [`toksvig_factor`](prism_render_architecture::particle::specular_aa::toksvig_factor).
//! 2. `scalars_main` directly pins every listed closed-form function per sample,
//!    each with free inputs:
//!    [`toksvig_factor`](prism_render_architecture::particle::specular_aa::toksvig_factor),
//!    [`toksvig_effective_gloss`](prism_render_architecture::particle::specular_aa::toksvig_effective_gloss),
//!    [`normal_length_variance`](prism_render_architecture::particle::specular_aa::normal_length_variance),
//!    [`perceptual_to_linear_roughness`](prism_render_architecture::particle::specular_aa::perceptual_to_linear_roughness),
//!    [`linear_to_perceptual_roughness`](prism_render_architecture::particle::specular_aa::linear_to_perceptual_roughness)
//!    and the standalone
//!    [`frostbite_specular_aa`](prism_render_architecture::particle::specular_aa::frostbite_specular_aa).
//!
//! The vector reductions
//! [`average_normal_length`](prism_render_architecture::particle::specular_aa::average_normal_length)
//! and
//! [`mip_average_normal_lengths`](prism_render_architecture::particle::specular_aa::mip_average_normal_lengths)
//! stay on the `CPU`: they are footprint reductions, not per-element closed
//! forms, and their scalar output (an average normal length) is exactly what the
//! twin's per-sample input consumes.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `sqrt` and `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow` or
//! optional device feature, so they run unmodified on `Metal`, `Vulkan` and
//! `DX12`. The only non-rational operation is the `sqrt` the reference itself
//! uses for the roughness conversions.
//!
//! # Correctness model
//!
//! Each function is a fixed, non-reorderable sequence of clamps, a divide and at
//! most one `sqrt`, so `CPU` and `GPU` evaluate the same closed-form algebra.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, and `sqrt` plus the reciprocal in the ratios carry a few
//! units in the last place. The parity test therefore asserts
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, tight enough to fail a genuinely
//! wrong port (a dropped clamp, a swapped `Toksvig` denominator, a missing
//! `kappa` cap) yet loose enough to admit a legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Toksvig` specular anti-aliasing and `Frostbite`
//! geometric specular anti-aliasing (design sections 16-17) plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::specular_aa::{SpecularAaParams, SpecularAaResult};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` batch-evaluate kernel, embedded inline. The entry
/// point `evaluate_main` mirrors the `CPU` golden
/// [`SpecularAaParams::evaluate`](prism_render_architecture::particle::specular_aa::SpecularAaParams::evaluate);
/// see the module documentation for the algorithm.
const EVALUATE_WGSL: &str = r#"
// Specular anti-aliasing batch-evaluate twin: one thread per sample reproduces
// `SpecularAaParams::evaluate` of the CPU golden `particle::specular_aa`. Uses
// only the portable core-WGSL subset (min/max/clamp, sqrt and + - * /) and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Toksvig / Frostbite specular anti-aliasing (design
// sections 16-17); no third-party engine source or derived code.

// Minimum average-normal length, matching `MIN_LEN` in the CPU module, so the
// Toksvig / variance denominators never divide by zero.
const MIN_LEN: f32 = 1.0e-6;

// Evaluate uniform: the shared `SpecularAaParams` scalars plus the sample count.
// 32-byte, 16-byte-aligned uniform struct matching `EvaluateParams` in Rust.
struct EvaluateParams {
    base_perceptual_roughness: f32,
    screen_variance: f32,
    variance_clamp: f32,
    gloss_power: f32,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One `SpecularAaResult`, matching `GpuResult` in Rust (16-byte stride).
struct AaResult {
    linear_roughness: f32,
    perceptual_roughness: f32,
    toksvig_factor: f32,
    added_variance: f32,
}

@group(0) @binding(0) var<uniform> params: EvaluateParams;
@group(0) @binding(1) var<storage, read> avg_normal_lengths: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<AaResult>;

// The Toksvig factor, mirroring the CPU golden `toksvig_factor`.
fn toksvig_factor(avg_normal_length: f32, gloss_power: f32) -> f32 {
    let r = clamp(avg_normal_length, MIN_LEN, 1.0);
    let s = max(gloss_power, 0.0);
    let denom = r + s * (1.0 - r);
    return clamp(r / denom, 0.0, 1.0);
}

// The normal-distribution variance `(1 - r) / r`, mirroring the CPU golden
// `normal_length_variance`, with `r` clamped to MIN_LEN..=1.
fn normal_length_variance(avg_normal_length: f32) -> f32 {
    let r = clamp(avg_normal_length, MIN_LEN, 1.0);
    return (1.0 - r) / r;
}

// Perceptual -> linear (NDF alpha) roughness `p * p`, mirroring the CPU golden
// `perceptual_to_linear_roughness`, with the input clamped to [0, 1].
fn perceptual_to_linear_roughness(perceptual: f32) -> f32 {
    let p = clamp(perceptual, 0.0, 1.0);
    return p * p;
}

// Linear -> perceptual roughness `sqrt(linear)`, mirroring the CPU golden
// `linear_to_perceptual_roughness`, with the input clamped to [0, 1].
fn linear_to_perceptual_roughness(linear: f32) -> f32 {
    return sqrt(clamp(linear, 0.0, 1.0));
}

// Frostbite geometric specular anti-aliasing in linear-roughness space,
// mirroring the CPU golden `frostbite_specular_aa`.
fn frostbite_specular_aa(linear_roughness: f32, variance: f32, kappa: f32) -> f32 {
    let base = clamp(linear_roughness, 0.0, 1.0);
    let base_sq = base * base;
    let kernel = min(2.0 * max(variance, 0.0), max(kappa, 0.0));
    let filtered = clamp(base_sq + kernel, 0.0, 1.0);
    return sqrt(filtered);
}

@compute @workgroup_size(64)
fn evaluate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let avg_normal_length = avg_normal_lengths[idx];
    // Reference order: map variance, add screen variance, square the base
    // roughness, run Frostbite, convert back, then the added kernel.
    let map_variance = normal_length_variance(avg_normal_length);
    let total_variance = map_variance + max(params.screen_variance, 0.0);
    let base_linear = perceptual_to_linear_roughness(params.base_perceptual_roughness);
    let linear_prime = frostbite_specular_aa(base_linear, total_variance, params.variance_clamp);
    let perceptual_prime = linear_to_perceptual_roughness(linear_prime);
    let added = min(2.0 * total_variance, max(params.variance_clamp, 0.0));

    var out: AaResult;
    out.linear_roughness = linear_prime;
    out.perceptual_roughness = perceptual_prime;
    out.toksvig_factor = toksvig_factor(avg_normal_length, params.gloss_power);
    out.added_variance = added;
    results[idx] = out;
}
"#;

/// The portable core-`WGSL` scalar-functions kernel, embedded inline. The entry
/// point `scalars_main` pins each standalone closed-form function of the `CPU`
/// golden [`specular_aa`](prism_render_architecture::particle::specular_aa).
const SCALARS_WGSL: &str = r#"
// Specular anti-aliasing scalar-functions twin: one thread per sample evaluates
// the six standalone closed-form functions of the CPU golden
// `particle::specular_aa` on free inputs. Uses only the portable core-WGSL
// subset (min/max/clamp, sqrt and + - * /) and takes no optional feature.
//
// Provenance: standard Toksvig / Frostbite specular anti-aliasing (design
// sections 16-17); no third-party engine source or derived code.

// Minimum average-normal length, matching `MIN_LEN` in the CPU module.
const MIN_LEN: f32 = 1.0e-6;

// Scalars uniform: just the sample count (every scalar input is per-element).
struct ScalarParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One free-input scalar sample, matching `GpuScalarSample` in Rust (32-byte
// stride). `perceptual` feeds `perceptual_to_linear_roughness`; `linear` feeds
// both `linear_to_perceptual_roughness` and `frostbite_specular_aa`'s roughness.
struct ScalarSample {
    avg_normal_length: f32,
    gloss_power: f32,
    perceptual: f32,
    linear: f32,
    variance: f32,
    kappa: f32,
    pad0: f32,
    pad1: f32,
}

// One scalar-function result tuple, matching `GpuScalars` in Rust (32-byte
// stride): the six closed-form outputs plus two pad words.
struct ScalarOut {
    toksvig_factor: f32,
    toksvig_effective_gloss: f32,
    normal_length_variance: f32,
    perceptual_to_linear: f32,
    linear_to_perceptual: f32,
    frostbite: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: ScalarParams;
@group(0) @binding(1) var<storage, read> samples: array<ScalarSample>;
@group(0) @binding(2) var<storage, read_write> results: array<ScalarOut>;

// The Toksvig factor, mirroring the CPU golden `toksvig_factor`.
fn toksvig_factor(avg_normal_length: f32, gloss_power: f32) -> f32 {
    let r = clamp(avg_normal_length, MIN_LEN, 1.0);
    let s = max(gloss_power, 0.0);
    let denom = r + s * (1.0 - r);
    return clamp(r / denom, 0.0, 1.0);
}

// The effective gloss `factor * s`, mirroring the CPU golden
// `toksvig_effective_gloss`.
fn toksvig_effective_gloss(avg_normal_length: f32, gloss_power: f32) -> f32 {
    return toksvig_factor(avg_normal_length, gloss_power) * max(gloss_power, 0.0);
}

// The normal-distribution variance `(1 - r) / r`, mirroring the CPU golden
// `normal_length_variance`, with `r` clamped to MIN_LEN..=1.
fn normal_length_variance(avg_normal_length: f32) -> f32 {
    let r = clamp(avg_normal_length, MIN_LEN, 1.0);
    return (1.0 - r) / r;
}

// Perceptual -> linear (NDF alpha) roughness `p * p`, mirroring the CPU golden
// `perceptual_to_linear_roughness`, with the input clamped to [0, 1].
fn perceptual_to_linear_roughness(perceptual: f32) -> f32 {
    let p = clamp(perceptual, 0.0, 1.0);
    return p * p;
}

// Linear -> perceptual roughness `sqrt(linear)`, mirroring the CPU golden
// `linear_to_perceptual_roughness`, with the input clamped to [0, 1].
fn linear_to_perceptual_roughness(linear: f32) -> f32 {
    return sqrt(clamp(linear, 0.0, 1.0));
}

// Frostbite geometric specular anti-aliasing in linear-roughness space,
// mirroring the CPU golden `frostbite_specular_aa`:
// `sqrt(clamp(roughness^2 + min(2 * variance, kappa), 0, 1))`.
fn frostbite_specular_aa(linear_roughness: f32, variance: f32, kappa: f32) -> f32 {
    let base = clamp(linear_roughness, 0.0, 1.0);
    let base_sq = base * base;
    let kernel = min(2.0 * max(variance, 0.0), max(kappa, 0.0));
    let filtered = clamp(base_sq + kernel, 0.0, 1.0);
    return sqrt(filtered);
}

@compute @workgroup_size(64)
fn scalars_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let s = samples[idx];
    var out: ScalarOut;
    out.toksvig_factor = toksvig_factor(s.avg_normal_length, s.gloss_power);
    out.toksvig_effective_gloss = toksvig_effective_gloss(s.avg_normal_length, s.gloss_power);
    out.normal_length_variance = normal_length_variance(s.avg_normal_length);
    out.perceptual_to_linear = perceptual_to_linear_roughness(s.perceptual);
    out.linear_to_perceptual = linear_to_perceptual_roughness(s.linear);
    out.frostbite = frostbite_specular_aa(s.linear, s.variance, s.kappa);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One batch-evaluate request: the shared [`SpecularAaParams`] and the batch of
/// per-footprint average normal lengths to evaluate.
///
/// Each entry's result equals
/// [`SpecularAaParams::evaluate`](prism_render_architecture::particle::specular_aa::SpecularAaParams::evaluate)
/// of that average normal length, to within the tolerance documented on this
/// module.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecularAaBatchQuery {
    /// The base roughness, screen variance, `kappa` clamp and gloss power shared
    /// by every sample.
    pub params: SpecularAaParams,
    /// The per-footprint average normal lengths; one [`SpecularAaResult`] is
    /// returned per entry in input order.
    pub avg_normal_lengths: Vec<f32>,
}

/// One free-input scalar sample for [`GpuSpecularAa::eval_scalars`].
///
/// Carries the independent inputs of the standalone closed-form functions so the
/// twin can pin each directly: `perceptual` feeds
/// [`perceptual_to_linear_roughness`](prism_render_architecture::particle::specular_aa::perceptual_to_linear_roughness),
/// while `linear` feeds both
/// [`linear_to_perceptual_roughness`](prism_render_architecture::particle::specular_aa::linear_to_perceptual_roughness)
/// and the roughness argument of
/// [`frostbite_specular_aa`](prism_render_architecture::particle::specular_aa::frostbite_specular_aa).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecularAaScalarSample {
    /// The average normal length `|Na|` fed to the `Toksvig` and variance maps.
    pub avg_normal_length: f32,
    /// The Blinn-style gloss power `s`.
    pub gloss_power: f32,
    /// The perceptual roughness fed to `perceptual_to_linear_roughness`.
    pub perceptual: f32,
    /// The linear roughness fed to `linear_to_perceptual_roughness` and as the
    /// roughness argument of `frostbite_specular_aa`.
    pub linear: f32,
    /// The normal variance fed to `frostbite_specular_aa`.
    pub variance: f32,
    /// The `kappa` clamp fed to `frostbite_specular_aa`.
    pub kappa: f32,
}

/// The six closed-form scalar results for one [`SpecularAaScalarSample`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecularAaScalars {
    /// [`toksvig_factor`](prism_render_architecture::particle::specular_aa::toksvig_factor)`(avg_normal_length, gloss_power)`.
    pub toksvig_factor: f32,
    /// [`toksvig_effective_gloss`](prism_render_architecture::particle::specular_aa::toksvig_effective_gloss)`(avg_normal_length, gloss_power)`.
    pub toksvig_effective_gloss: f32,
    /// [`normal_length_variance`](prism_render_architecture::particle::specular_aa::normal_length_variance)`(avg_normal_length)`.
    pub normal_length_variance: f32,
    /// [`perceptual_to_linear_roughness`](prism_render_architecture::particle::specular_aa::perceptual_to_linear_roughness)`(perceptual)`.
    pub perceptual_to_linear: f32,
    /// [`linear_to_perceptual_roughness`](prism_render_architecture::particle::specular_aa::linear_to_perceptual_roughness)`(linear)`.
    pub linear_to_perceptual: f32,
    /// [`frostbite_specular_aa`](prism_render_architecture::particle::specular_aa::frostbite_specular_aa)`(linear, variance, kappa)`.
    pub frostbite: f32,
}

/// One scalar-functions request: the batch of [`SpecularAaScalarSample`]s to
/// evaluate. One [`SpecularAaScalars`] tuple is returned per sample.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecularAaScalarQuery {
    /// The per-sample free inputs; one result tuple is returned per entry.
    pub samples: Vec<SpecularAaScalarSample>,
}

/// Evaluate uniform parameters for one dispatch. `32`-byte `repr(C)` matching
/// `EvaluateParams` in [`EVALUATE_WGSL`]: the shared `SpecularAaParams` scalars,
/// the sample count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EvaluateParams {
    /// Artist-facing base perceptual roughness.
    base_perceptual_roughness: f32,
    /// Extra screen-space normal variance added on top of the `mip` variance.
    screen_variance: f32,
    /// The `kappa` clamp on the maximum added kernel roughness.
    variance_clamp: f32,
    /// The Blinn-style gloss power used for the reported `Toksvig` factor.
    gloss_power: f32,
    /// Number of samples in the dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Scalars uniform parameters for one dispatch. `16`-byte `repr(C)` matching
/// `ScalarParams` in [`SCALARS_WGSL`]: the sample count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScalarParams {
    /// Number of samples in the dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One evaluate result as read back. `16`-byte `repr(C)` matching `AaResult` in
/// [`EVALUATE_WGSL`] and the field order of [`SpecularAaResult`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The anti-aliased linear (`NDF` `alpha`) roughness.
    linear_roughness: f32,
    /// The anti-aliased perceptual roughness.
    perceptual_roughness: f32,
    /// The `Toksvig` factor for the footprint.
    toksvig_factor: f32,
    /// The kernel roughness actually added in linear-roughness-squared space.
    added_variance: f32,
}

/// One free-input scalar sample as uploaded. `32`-byte `repr(C)` matching
/// `ScalarSample` in [`SCALARS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuScalarSample {
    /// The average normal length `|Na|`.
    avg_normal_length: f32,
    /// The Blinn-style gloss power `s`.
    gloss_power: f32,
    /// The perceptual roughness input.
    perceptual: f32,
    /// The linear roughness input.
    linear: f32,
    /// The normal variance input.
    variance: f32,
    /// The `kappa` clamp input.
    kappa: f32,
    /// Padding to a `32`-byte, `std430`-friendly stride.
    pad0: f32,
    /// Padding to a `32`-byte, `std430`-friendly stride.
    pad1: f32,
}

/// One scalar-functions result as read back. `32`-byte `repr(C)` matching
/// `ScalarOut` in [`SCALARS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuScalars {
    /// `toksvig_factor` output.
    toksvig_factor: f32,
    /// `toksvig_effective_gloss` output.
    toksvig_effective_gloss: f32,
    /// `normal_length_variance` output.
    normal_length_variance: f32,
    /// `perceptual_to_linear_roughness` output.
    perceptual_to_linear: f32,
    /// `linear_to_perceptual_roughness` output.
    linear_to_perceptual: f32,
    /// `frostbite_specular_aa` output.
    frostbite: f32,
    /// Padding to a `32`-byte stride.
    pad0: f32,
    /// Padding to a `32`-byte stride.
    pad1: f32,
}

/// A compiled, reusable specular anti-aliasing pipeline pair (batch evaluate and
/// free-input scalar functions).
pub struct GpuSpecularAa {
    #[expect(
        dead_code,
        reason = "kept alive so the evaluate pipeline it produced stays valid"
    )]
    module_evaluate: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the scalars pipeline it produced stays valid"
    )]
    module_scalars: ShaderModule,
    layout: BindGroupLayout,
    pipeline_evaluate: ComputePipeline,
    pipeline_scalars: ComputePipeline,
}

impl GpuSpecularAa {
    /// Compiles the evaluate and scalar-functions kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSpecularAa {
        let device = ctx.device();
        let module_evaluate = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_specular_aa_evaluate"),
            source: ShaderSource::Wgsl(EVALUATE_WGSL.into()),
        });
        let module_scalars = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_specular_aa_scalars"),
            source: ShaderSource::Wgsl(SCALARS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_specular_aa_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_specular_aa_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline_evaluate = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_specular_aa_evaluate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_evaluate,
            entry_point: Some("evaluate_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_scalars = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_specular_aa_scalars_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_scalars,
            entry_point: Some("scalars_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpecularAa {
            module_evaluate,
            module_scalars,
            layout,
            pipeline_evaluate,
            pipeline_scalars,
        }
    }

    /// Evaluates the anti-aliased roughness for every average normal length in
    /// `query.avg_normal_lengths`, returning one [`SpecularAaResult`] per entry
    /// in input order.
    ///
    /// The returned value for entry `l` equals
    /// [`SpecularAaParams::evaluate`](prism_render_architecture::particle::specular_aa::SpecularAaParams::evaluate)`(l)`
    /// to within the tolerance documented on this module. An empty batch yields
    /// an empty result with no dispatch issued — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &SpecularAaBatchQuery) -> Vec<SpecularAaResult> {
        if query.avg_normal_lengths.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = query.avg_normal_lengths.len();

        let gpu_params = EvaluateParams {
            base_perceptual_roughness: query.params.base_perceptual_roughness,
            screen_variance: query.params.screen_variance,
            variance_clamp: query.params.variance_clamp,
            gloss_power: query.params.gloss_power,
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_specular_aa_evaluate_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_specular_aa_avg_normal_lengths"),
            contents: bytemuck::cast_slice(&query.avg_normal_lengths),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let bytes = self.dispatch(
            ctx,
            &self.pipeline_evaluate,
            &params_buf,
            &input_buf,
            count,
            out_bytes,
        );
        let records = bytemuck::cast_slice::<u8, GpuResult>(&bytes);
        debug_assert_eq!(records.len(), count);
        records
            .iter()
            .map(|r| SpecularAaResult {
                linear_roughness: r.linear_roughness,
                perceptual_roughness: r.perceptual_roughness,
                toksvig_factor: r.toksvig_factor,
                added_variance: r.added_variance,
            })
            .collect()
    }

    /// Evaluates the six standalone closed-form functions for every sample in
    /// `query.samples`, returning one [`SpecularAaScalars`] tuple per entry in
    /// input order.
    ///
    /// Each field equals the matching reference function of the sample's inputs
    /// (see [`SpecularAaScalars`]) to within the tolerance documented on this
    /// module. An empty batch yields an empty result with no dispatch issued.
    #[must_use]
    pub fn eval_scalars(
        &self,
        ctx: &GpuContext,
        query: &SpecularAaScalarQuery,
    ) -> Vec<SpecularAaScalars> {
        if query.samples.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = query.samples.len();

        let gpu_samples: Vec<GpuScalarSample> = query
            .samples
            .iter()
            .map(|s| GpuScalarSample {
                avg_normal_length: s.avg_normal_length,
                gloss_power: s.gloss_power,
                perceptual: s.perceptual,
                linear: s.linear,
                variance: s.variance,
                kappa: s.kappa,
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect();
        let gpu_params = ScalarParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_specular_aa_scalar_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_specular_aa_scalar_samples"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuScalars>()) as u64;
        let bytes = self.dispatch(
            ctx,
            &self.pipeline_scalars,
            &params_buf,
            &input_buf,
            count,
            out_bytes,
        );
        let records = bytemuck::cast_slice::<u8, GpuScalars>(&bytes);
        debug_assert_eq!(records.len(), count);
        records
            .iter()
            .map(|r| SpecularAaScalars {
                toksvig_factor: r.toksvig_factor,
                toksvig_effective_gloss: r.toksvig_effective_gloss,
                normal_length_variance: r.normal_length_variance,
                perceptual_to_linear: r.perceptual_to_linear,
                linear_to_perceptual: r.linear_to_perceptual,
                frostbite: r.frostbite,
            })
            .collect()
    }

    /// Binds the uniform and input buffers, runs `pipeline` with one thread per
    /// element and reads the `out_bytes` output back as raw bytes.
    ///
    /// Shared by [`GpuSpecularAa::eval`] and [`GpuSpecularAa::eval_scalars`]:
    /// both bind a uniform, a read-only input storage buffer and a writable
    /// output storage buffer, differing only in the kernel and the record
    /// layout the caller reinterprets.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        params_buf: &wgpu::Buffer,
        input_buf: &wgpu::Buffer,
        count: usize,
        out_bytes: u64,
    ) -> Vec<u8> {
        let device = ctx.device();

        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_specular_aa_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_specular_aa_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_specular_aa_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_specular_aa_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_specular_aa_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let bytes = view.to_vec();
        drop(view);
        results_stage.unmap();
        bytes
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
