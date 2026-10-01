//! `wgpu` compute twin of the `GGX` multiple-scattering energy-compensation
//! golden
//! ([`ggx_energy_compensation`](prism_render_architecture::particle::ggx_energy_compensation),
//! particle design §17, "共享 `PBR` closure（`GGX` + 多散近似）").
//!
//! The `CPU` golden
//! [`ggx_energy_compensation`](prism_render_architecture::particle::ggx_energy_compensation)
//! owns the transcendental-free Kulla-Conty energy-compensation stack a lit
//! particle's shared `PBR` closure needs so rough metals never go
//! energy-deficient: the single-scattering directional albedo `E_ss(mu,
//! roughness)`
//! ([`single_scatter_directional_albedo`](prism_render_architecture::particle::ggx_energy_compensation::single_scatter_directional_albedo)),
//! its exact cosine-weighted hemispherical average `E_avg(roughness)`
//! ([`average_albedo`](prism_render_architecture::particle::ggx_energy_compensation::average_albedo)),
//! the analytic average `Fresnel` `F_avg`
//! ([`average_fresnel`](prism_render_architecture::particle::ggx_energy_compensation::average_fresnel)
//! and its `RGB` sibling
//! [`average_fresnel_rgb`](prism_render_architecture::particle::ggx_energy_compensation::average_fresnel_rgb)),
//! the multiple-scattering `Fresnel` scale `F_ms`
//! ([`multiscatter_fresnel_scale`](prism_render_architecture::particle::ggx_energy_compensation::multiscatter_fresnel_scale)),
//! the multiple-scattering directional albedo
//! ([`multiscatter_directional_albedo`](prism_render_architecture::particle::ggx_energy_compensation::multiscatter_directional_albedo)),
//! the bidirectional lobe
//! ([`kulla_conty_multiscatter_brdf`](prism_render_architecture::particle::ggx_energy_compensation::kulla_conty_multiscatter_brdf))
//! and the compensated specular albedo
//! ([`compensated_specular_albedo`](prism_render_architecture::particle::ggx_energy_compensation::compensated_specular_albedo)
//! with its `RGB` sibling
//! [`compensated_specular_albedo_rgb`](prism_render_architecture::particle::ggx_energy_compensation::compensated_specular_albedo_rgb)).
//! [`GpuGgxEnergyCompensation`] is the on-device twin: one thread evaluates one
//! query and emits every one of those answers, so a passing real-device parity
//! test is direct evidence the ported kernel evaluates the same closed-form
//! algebra and takes the same degenerate branch the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Every value the reference exposes is reproduced per query: the four scalar
//! pieces (`E_ss`, `E_avg`, `F_avg`, `F_ms`), the two composed scalar albedos
//! (multiple-scattering directional and compensated specular), the bidirectional
//! lobe, and the two per-channel `RGB` results (`F_avg` and compensated specular
//! albedo). The `RGB` functions are component-wise, so each lane of the `vec3`
//! outputs is exactly the scalar function of the matching lane and the parity
//! test checks them that way.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max` and `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry, no `sqrt` and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates.
//!
//! # Correctness model
//!
//! The single degenerate branch (`1 - E_avg` at or below the compare epsilon
//! collapses the bidirectional lobe to `0`) is a discrete classification from an
//! `f32` magnitude comparison, so for roughness clear of that threshold the
//! `CPU` and `GPU` take the same branch. The remaining algebra threads through
//! multiplies, adds and guarded divides, so `CPU` and `GPU` are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The parity
//! test therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <=
//! 1e-3`) on every continuous quantity, tight enough to catch a genuinely wrong
//! port (a dropped term, a swapped coefficient, a wrong clamp) yet loose enough
//! to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ggx_energy_compensation`；
//! standard Kulla-Conty 2017 energy compensation plus `wgpu` compute dispatch;
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ggx_energy_compensation::{
    average_albedo, average_fresnel, average_fresnel_rgb, compensated_specular_albedo,
    compensated_specular_albedo_rgb, kulla_conty_multiscatter_brdf,
    multiscatter_directional_albedo, multiscatter_fresnel_scale, single_scatter_directional_albedo,
};
use prism_render_architecture::particle::Vec3;
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

/// The portable core-`WGSL` energy-compensation kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`ggx_energy_compensation`](prism_render_architecture::particle::ggx_energy_compensation)
/// branch for branch; see the module documentation for the algorithm.
const GGX_ENERGY_COMPENSATION_WGSL: &str = r#"
// GGX energy-compensation twin: one thread per query reproduces the single-
// scattering directional albedo, its hemispherical average, the average Fresnel
// (scalar and per-channel), the Kulla-Conty multiple-scattering Fresnel scale,
// the multiple-scattering directional albedo, the bidirectional lobe and the
// compensated specular albedo (scalar and per-channel). It mirrors the CPU
// golden `particle::ggx_energy_compensation` branch for branch, uses only the
// portable core-WGSL subset (clamp/min/max and + - * /), needs no sqrt and no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::ggx_energy_compensation；
// 无第三方引擎源码或衍生代码。

// Head-on energy-deficit slope per unit linear roughness; matches the reference
// `DEFICIT_BASE`.
const DEFICIT_BASE: f32 = 0.35;
// Extra grazing-weighted deficit; matches the reference `DEFICIT_GRAZING`.
const DEFICIT_GRAZING: f32 = 0.25;
// Smallest denominator tolerated before a ratio is treated as degenerate; the
// compare rule used instead of an f32 == / !=. Matches the reference
// `MIN_DENOM`.
const MIN_DENOM: f32 = 1.0e-6;
// Compile-time pi, matching the reference `core::f32::consts::PI`. It appears
// only in the bidirectional lobe normalization and cancels out of every
// directional-albedo result.
const PI: f32 = 3.1415927;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // mu, mu_out, mu_in, roughness.
    mu: f32,
    mu_out: f32,
    mu_in: f32,
    roughness: f32,
    // Scalar F0, the caller's single-scattering albedo, and the standalone
    // E_avg / F_avg fed to the multiscatter Fresnel scale.
    f0: f32,
    single: f32,
    e_avg_in: f32,
    f_avg_in: f32,
    // Per-channel F0; a pad lane follows.
    f0_rgb: vec3<f32>,
    pad0: f32,
    // Per-channel single-scattering albedo; a pad lane follows.
    single_rgb: vec3<f32>,
    pad1: f32,
}

struct Result {
    // Scalar outputs: E_ss, E_avg, F_avg, F_ms, multiscatter directional albedo,
    // bidirectional lobe, compensated specular albedo and one pad lane.
    e_ss: f32,
    avg_albedo: f32,
    avg_fresnel: f32,
    fresnel_scale: f32,
    ms_dir: f32,
    brdf: f32,
    comp: f32,
    pad0: f32,
    // Per-channel average Fresnel; a pad lane follows.
    avg_fresnel_rgb: vec3<f32>,
    pad1: f32,
    // Per-channel compensated specular albedo; a pad lane follows.
    comp_rgb: vec3<f32>,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamp into [0, 1] without an f32 equality branch; mirrors the reference
// `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Single-scattering directional albedo E_ss(mu, roughness); mirrors the
// reference `single_scatter_directional_albedo`. The fixed square is an integer
// multiply, never a pow.
fn single_scatter_e(mu: f32, roughness: f32) -> f32 {
    let cos_theta = clamp01(mu);
    let perceptual = clamp01(roughness);
    let alpha = perceptual * perceptual;
    let grazing = 1.0 - cos_theta;
    let grazing_sq = grazing * grazing;
    let deficit = alpha * (DEFICIT_BASE + DEFICIT_GRAZING * grazing_sq);
    return 1.0 - deficit;
}

// Cosine-weighted hemispherical average albedo E_avg(roughness); mirrors the
// reference `average_albedo` (the exact closed form of the E_ss integral).
fn average_e(roughness: f32) -> f32 {
    let perceptual = clamp01(roughness);
    let alpha = perceptual * perceptual;
    return 1.0 - alpha * (DEFICIT_BASE + DEFICIT_GRAZING / 6.0);
}

// Analytic average Fresnel F_avg = F0 + (1 - F0)/21; mirrors the reference
// `average_fresnel`.
fn average_f(f0: f32) -> f32 {
    let base = clamp01(f0);
    return base + (1.0 - base) / 21.0;
}

// Kulla-Conty multiple-scattering Fresnel scale F_ms; mirrors the reference
// `multiscatter_fresnel_scale`. The denominator is floored at MIN_DENOM so a
// vanishing missing-energy term never divides by zero.
fn ms_fresnel_scale(e_avg_in: f32, f_avg_in: f32) -> f32 {
    let energy = clamp01(e_avg_in);
    let fresnel = clamp01(f_avg_in);
    let numerator = fresnel * fresnel * energy;
    let denominator = 1.0 - fresnel * (1.0 - energy);
    return numerator / max(denominator, MIN_DENOM);
}

// Multiple-scattering directional albedo F_ms * (1 - E_ss); mirrors the
// reference `multiscatter_directional_albedo`.
fn ms_directional(mu: f32, roughness: f32, f0: f32) -> f32 {
    let e = single_scatter_e(mu, roughness);
    let ea = average_e(roughness);
    let fa = average_f(f0);
    let scale = ms_fresnel_scale(ea, fa);
    return scale * (1.0 - e);
}

// Bidirectional Kulla-Conty lobe; mirrors the reference
// `kulla_conty_multiscatter_brdf`. A missing-energy factor at or below
// MIN_DENOM (roughness -> 0) collapses the lobe to 0 to avoid a degenerate 0/0.
fn kc_brdf(mu_out: f32, mu_in: f32, roughness: f32, f0: f32) -> f32 {
    let ea = average_e(roughness);
    let one_minus = 1.0 - ea;
    if (one_minus < MIN_DENOM) {
        return 0.0;
    }
    let e_out = single_scatter_e(mu_out, roughness);
    let e_in = single_scatter_e(mu_in, roughness);
    let fa = average_f(f0);
    let scale = ms_fresnel_scale(ea, fa);
    return scale * (1.0 - e_out) * (1.0 - e_in) / (PI * one_minus);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let e_ss = single_scatter_e(q.mu, q.roughness);
    let avg_albedo = average_e(q.roughness);
    let avg_fresnel = average_f(q.f0);
    let fresnel_scale = ms_fresnel_scale(q.e_avg_in, q.f_avg_in);
    let ms_dir = ms_directional(q.mu, q.roughness, q.f0);
    let brdf = kc_brdf(q.mu_out, q.mu_in, q.roughness, q.f0);
    // compensated_specular_albedo = single + multiscatter directional albedo.
    let comp = q.single + ms_dir;

    let avg_fresnel_rgb = vec3<f32>(
        average_f(q.f0_rgb.x),
        average_f(q.f0_rgb.y),
        average_f(q.f0_rgb.z),
    );
    let comp_rgb = vec3<f32>(
        q.single_rgb.x + ms_directional(q.mu, q.roughness, q.f0_rgb.x),
        q.single_rgb.y + ms_directional(q.mu, q.roughness, q.f0_rgb.y),
        q.single_rgb.z + ms_directional(q.mu, q.roughness, q.f0_rgb.z),
    );

    var out: Result;
    out.e_ss = e_ss;
    out.avg_albedo = avg_albedo;
    out.avg_fresnel = avg_fresnel;
    out.fresnel_scale = fresnel_scale;
    out.ms_dir = ms_dir;
    out.brdf = brdf;
    out.comp = comp;
    out.pad0 = 0.0;
    out.avg_fresnel_rgb = avg_fresnel_rgb;
    out.pad1 = 0.0;
    out.comp_rgb = comp_rgb;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// One query for the energy-compensation twin: the inputs every twinned
/// function consumes, bundled so a single query exercises the whole stack at
/// once.
///
/// The `e_avg`/`f_avg` pair feeds the standalone
/// [`multiscatter_fresnel_scale`](prism_render_architecture::particle::ggx_energy_compensation::multiscatter_fresnel_scale)
/// and is independent of the `roughness`/`f0`-derived averages the composed
/// albedos use, so both paths are checked at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxEnergyCompensationQuery {
    /// View cosine `mu` for the scalar directional albedos.
    pub mu: f32,
    /// Outgoing cosine `mu_out` for the bidirectional lobe.
    pub mu_out: f32,
    /// Incoming cosine `mu_in` for the bidirectional lobe.
    pub mu_in: f32,
    /// Perceptual roughness.
    pub roughness: f32,
    /// Scalar reflectance at normal incidence `F0`.
    pub f0: f32,
    /// Caller's single-scattering directional albedo for the compensated
    /// specular albedo.
    pub single_scatter_albedo: f32,
    /// Standalone `E_avg` fed to the multiscatter `Fresnel` scale.
    pub e_avg: f32,
    /// Standalone `F_avg` fed to the multiscatter `Fresnel` scale.
    pub f_avg: f32,
    /// Per-channel reflectance at normal incidence `F0`.
    pub f0_rgb: Vec3,
    /// Per-channel single-scattering directional albedo for the compensated
    /// specular albedo.
    pub single_scatter_albedo_rgb: Vec3,
}

/// One resolved answer, mirroring every value the reference reports across its
/// twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxEnergyCompensationResult {
    /// Single-scattering directional albedo, matching
    /// [`single_scatter_directional_albedo`](prism_render_architecture::particle::ggx_energy_compensation::single_scatter_directional_albedo).
    pub single_scatter_directional_albedo: f32,
    /// Hemispherical average albedo, matching
    /// [`average_albedo`](prism_render_architecture::particle::ggx_energy_compensation::average_albedo).
    pub average_albedo: f32,
    /// Average `Fresnel`, matching
    /// [`average_fresnel`](prism_render_architecture::particle::ggx_energy_compensation::average_fresnel).
    pub average_fresnel: f32,
    /// Multiple-scattering `Fresnel` scale, matching
    /// [`multiscatter_fresnel_scale`](prism_render_architecture::particle::ggx_energy_compensation::multiscatter_fresnel_scale).
    pub multiscatter_fresnel_scale: f32,
    /// Multiple-scattering directional albedo, matching
    /// [`multiscatter_directional_albedo`](prism_render_architecture::particle::ggx_energy_compensation::multiscatter_directional_albedo).
    pub multiscatter_directional_albedo: f32,
    /// Bidirectional lobe, matching
    /// [`kulla_conty_multiscatter_brdf`](prism_render_architecture::particle::ggx_energy_compensation::kulla_conty_multiscatter_brdf).
    pub kulla_conty_multiscatter_brdf: f32,
    /// Compensated specular albedo, matching
    /// [`compensated_specular_albedo`](prism_render_architecture::particle::ggx_energy_compensation::compensated_specular_albedo).
    pub compensated_specular_albedo: f32,
    /// Per-channel average `Fresnel`, matching
    /// [`average_fresnel_rgb`](prism_render_architecture::particle::ggx_energy_compensation::average_fresnel_rgb).
    pub average_fresnel_rgb: Vec3,
    /// Per-channel compensated specular albedo, matching
    /// [`compensated_specular_albedo_rgb`](prism_render_architecture::particle::ggx_energy_compensation::compensated_specular_albedo_rgb).
    pub compensated_specular_albedo_rgb: Vec3,
}

/// Evaluates the `CPU` golden for one query, delegating field for field to the
/// reference
/// [`ggx_energy_compensation`](prism_render_architecture::particle::ggx_energy_compensation)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ggx_energy_compensation`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &GgxEnergyCompensationQuery) -> GgxEnergyCompensationResult {
    GgxEnergyCompensationResult {
        single_scatter_directional_albedo: single_scatter_directional_albedo(
            query.mu,
            query.roughness,
        ),
        average_albedo: average_albedo(query.roughness),
        average_fresnel: average_fresnel(query.f0),
        multiscatter_fresnel_scale: multiscatter_fresnel_scale(query.e_avg, query.f_avg),
        multiscatter_directional_albedo: multiscatter_directional_albedo(
            query.mu,
            query.roughness,
            query.f0,
        ),
        kulla_conty_multiscatter_brdf: kulla_conty_multiscatter_brdf(
            query.mu_out,
            query.mu_in,
            query.roughness,
            query.f0,
        ),
        compensated_specular_albedo: compensated_specular_albedo(
            query.single_scatter_albedo,
            query.mu,
            query.roughness,
            query.f0,
        ),
        average_fresnel_rgb: average_fresnel_rgb(query.f0_rgb),
        compensated_specular_albedo_rgb: compensated_specular_albedo_rgb(
            query.single_scatter_albedo_rgb,
            query.mu,
            query.roughness,
            query.f0_rgb,
        ),
    }
}

/// `repr(C)` `std430` layout of one packed query, matching the `WGSL` `Query`
/// struct. The eight leading scalars fill two `16`-byte slots and each `vec3`
/// carries a trailing pad lane so it stays `16`-byte aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// View cosine `mu`.
    mu: f32,
    /// Outgoing cosine `mu_out`.
    mu_out: f32,
    /// Incoming cosine `mu_in`.
    mu_in: f32,
    /// Perceptual roughness.
    roughness: f32,
    /// Scalar `F0`.
    f0: f32,
    /// Caller's single-scattering albedo.
    single: f32,
    /// Standalone `E_avg`.
    e_avg_in: f32,
    /// Standalone `F_avg`.
    f_avg_in: f32,
    /// Per-channel `F0`.
    f0_rgb: [f32; 3],
    /// Pad lane after `f0_rgb`.
    pad0: f32,
    /// Per-channel single-scattering albedo.
    single_rgb: [f32; 3],
    /// Pad lane after `single_rgb`.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &GgxEnergyCompensationQuery) -> GpuQuery {
        GpuQuery {
            mu: query.mu,
            mu_out: query.mu_out,
            mu_in: query.mu_in,
            roughness: query.roughness,
            f0: query.f0,
            single: query.single_scatter_albedo,
            e_avg_in: query.e_avg,
            f_avg_in: query.f_avg,
            f0_rgb: [query.f0_rgb.x, query.f0_rgb.y, query.f0_rgb.z],
            pad0: 0.0,
            single_rgb: [
                query.single_scatter_albedo_rgb.x,
                query.single_scatter_albedo_rgb.y,
                query.single_scatter_albedo_rgb.z,
            ],
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The seven scalar outputs plus a pad lane fill two `16`-byte slots and
/// each `vec3` carries a trailing pad lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Single-scattering directional albedo `E_ss`.
    e_ss: f32,
    /// Hemispherical average albedo `E_avg`.
    avg_albedo: f32,
    /// Average `Fresnel` `F_avg`.
    avg_fresnel: f32,
    /// Multiple-scattering `Fresnel` scale `F_ms`.
    fresnel_scale: f32,
    /// Multiple-scattering directional albedo.
    ms_dir: f32,
    /// Bidirectional lobe.
    brdf: f32,
    /// Compensated specular albedo.
    comp: f32,
    /// Pad lane.
    pad0: f32,
    /// Per-channel average `Fresnel`.
    avg_fresnel_rgb: [f32; 3],
    /// Pad lane after `avg_fresnel_rgb`.
    pad1: f32,
    /// Per-channel compensated specular albedo.
    comp_rgb: [f32; 3],
    /// Pad lane after `comp_rgb`.
    pad2: f32,
}

/// Decodes one packed [`GpuResult`] into the public
/// [`GgxEnergyCompensationResult`].
fn decode_result(raw: &GpuResult) -> GgxEnergyCompensationResult {
    GgxEnergyCompensationResult {
        single_scatter_directional_albedo: raw.e_ss,
        average_albedo: raw.avg_albedo,
        average_fresnel: raw.avg_fresnel,
        multiscatter_fresnel_scale: raw.fresnel_scale,
        multiscatter_directional_albedo: raw.ms_dir,
        kulla_conty_multiscatter_brdf: raw.brdf,
        compensated_specular_albedo: raw.comp,
        average_fresnel_rgb: Vec3::new(
            raw.avg_fresnel_rgb[0],
            raw.avg_fresnel_rgb[1],
            raw.avg_fresnel_rgb[2],
        ),
        compensated_specular_albedo_rgb: Vec3::new(
            raw.comp_rgb[0],
            raw.comp_rgb[1],
            raw.comp_rgb[2],
        ),
    }
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct matching `Params` in
/// [`GGX_ENERGY_COMPENSATION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
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

/// A compiled, reusable energy-compensation compute pipeline, twinning the `CPU`
/// golden
/// [`ggx_energy_compensation`](prism_render_architecture::particle::ggx_energy_compensation).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ggx_energy_compensation`；
/// no third-party engine source or derived code.
pub struct GpuGgxEnergyCompensation {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGgxEnergyCompensation {
    /// Compiles the energy-compensation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::ggx_energy_compensation`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGgxEnergyCompensation {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation"),
            source: ShaderSource::Wgsl(GGX_ENERGY_COMPENSATION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGgxEnergyCompensation {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GgxEnergyCompensationResult`] per input, in order.
    ///
    /// Each result equals the reference answers to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::ggx_energy_compensation`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[GgxEnergyCompensationQuery],
    ) -> Vec<GgxEnergyCompensationResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ggx_energy_compensation_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ggx_energy_compensation_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
