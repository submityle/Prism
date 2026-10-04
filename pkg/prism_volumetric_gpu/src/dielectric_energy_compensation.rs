//! `wgpu` compute twin of the rough-dielectric multiple-scattering energy
//! compensation from the `CPU` golden
//! `prism_render_architecture::reference_pt::dielectric_energy`.
//!
//! A single-scatter microfacet dielectric (frosted glass, water) only transports
//! light that reflects or refracts off exactly one micro-facet; at high
//! roughness a large fraction of the energy instead bounces several times
//! between facets before escaping, and the single-scatter Smith masking term
//! silently discards it, so rough glass renders too dark. The reference supplies
//! a Turquin-style multiplicative compensation driven by a baked single-scatter
//! directional-albedo table and a smooth `Fresnel` ceiling. This module ports
//! the three stateless, no-`RNG` entry points — `single_scatter_albedo`,
//! `smooth_albedo` and `compensation_factor` — onto the device: one thread
//! solves one query.
//!
//! [`GpuDielectricEnergy`] is the on-device twin; a passing real-device parity
//! test is direct evidence the ported kernel reproduces the same trilinear
//! lookup, the same `Fresnel`-dielectric ceiling and the same clamped
//! compensation factor the reference computes, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate:
//! the trilinear [`DE_ALBEDO`]-table lookup `single_scatter_albedo(eta, alpha,
//! mu)` with its shared cell-centred axis-weight clamping; the smooth ceiling
//! `smooth_albedo(eta, mu) = R + (1 - R) / eta^2` built on an inlined
//! `fresnel_dielectric`; and the clamped multiplicative
//! `compensation_factor(eta, alpha, mu)`. There is no loop: each thread performs
//! a fixed, bounded sequence of multiplies, adds, guarded divisions and one
//! `sqrt`, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! Every continuous quantity threads through multiplies, adds, guarded
//! divisions and `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate. The parity test asserts
//! a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous channel, tight enough to catch a wrong port yet loose
//! enough to admit legal fused multiply-add contraction. The discrete `valid`
//! flag is compared exactly; it is always `1`, since the lookup clamps every
//! out-of-range input to the nearest baked node and never faults.
//!
//! # Degenerate inputs
//!
//! Out-of-range `eta`, `alpha` or `mu` are clamped by the shared axis-weight
//! logic to the nearest baked node, so no query is rejected and `valid` is
//! always `1`. The total-internal-reflection branch of the `Fresnel` term
//! (`sin2_t >= 1`) returns full reflectance; the smooth-ceiling `eta^2` divisor
//! and the compensation `MIN_ALBEDO` floor keep both ratios finite. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `select`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no float `%`, and no `u64` /
//! `i64` / `f64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::dielectric_energy`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
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

/// The portable core-`WGSL` rough-dielectric energy-compensation kernel,
/// embedded inline so the twin ships as a single source file. The single entry
/// point `solve` mirrors the `CPU` golden `single_scatter_albedo`,
/// `smooth_albedo` and `compensation_factor`; see the module documentation.
const DIELECTRIC_ENERGY_COMPENSATION_WGSL: &str = r#"
// Rough-dielectric multiple-scattering energy compensation twin: one thread per
// query reproduces the trilinear single-scatter directional albedo, the smooth
// Fresnel-dielectric ceiling, and the clamped Turquin compensation factor. It
// mirrors the CPU golden branch for branch, uses only the portable core-WGSL
// subset (min/max/clamp/select/sqrt and + - * / plus unsigned index math), takes
// no optional feature, and has no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::dielectric_energy；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Relative index of refraction eta_t / eta_i.
    eta: f32,
    // GGX width alpha.
    alpha: f32,
    // Absolute view cosine mu = |n.wo|.
    mu: f32,
    // Padding so the std430 array stride is a 16-byte multiple.
    pad0: f32,
}

struct Result {
    // Single-scatter directional albedo E(eta, alpha, mu).
    e_ss: f32,
    // Smooth ceiling E_smooth(eta, mu).
    e_smooth: f32,
    // Clamped multiplicative compensation factor.
    comp: f32,
    // Always 1: every input is clamped to a baked node, so no query faults.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Smallest relative index covered by the baked table.
const ETA_MIN: f32 = 1.1;
// Largest relative index covered by the baked table.
const ETA_MAX: f32 = 2.5;
// Upper bound on the compensation boost.
const MAX_FACTOR: f32 = 3.0;
// Floor on the albedo denominator, matching the smallest trustworthy node.
const MIN_ALBEDO: f32 = 0.05;
// Table axis extents.
const ETA_SIZE: u32 = 6u;
const ALPHA_SIZE: u32 = 8u;
const MU_SIZE: u32 = 8u;
// Guard threshold so a vanishing Fresnel denominator cannot divide by zero.
const DENOM_EPS: f32 = 1e-20;

// Baked lossless single-scatter directional albedo E(eta, alpha, mu), flattened
// in eta-major, alpha-middle, mu-minor order (flat = e*64 + a*8 + m). Copied
// verbatim from the golden ALBEDO_DATA table.
const DE_ALBEDO: array<f32, 384> = array<f32, 384>(
    0.73804, 0.72718, 0.70504, 0.68996, 0.68273, 0.67986, 0.67868, 0.67847,
    0.67819, 0.66063, 0.66459, 0.66876, 0.67154, 0.67329, 0.67530, 0.67679,
    0.63255, 0.61704, 0.62857, 0.64182, 0.65290, 0.66077, 0.66771, 0.67289,
    0.59579, 0.57771, 0.59305, 0.61321, 0.63071, 0.64638, 0.65782, 0.66684,
    0.56644, 0.54256, 0.55957, 0.58326, 0.60728, 0.62816, 0.64613, 0.66011,
    0.54392, 0.51456, 0.52964, 0.55602, 0.58240, 0.60942, 0.63210, 0.65129,
    0.52451, 0.49156, 0.50505, 0.53054, 0.55992, 0.58877, 0.61684, 0.64212,
    0.50854, 0.47271, 0.48259, 0.50765, 0.53701, 0.56815, 0.60143, 0.63192,
    0.65676, 0.61244, 0.56193, 0.52658, 0.50977, 0.49948, 0.49490, 0.49385,
    0.56366, 0.52537, 0.51175, 0.50315, 0.49764, 0.49274, 0.49068, 0.49045,
    0.51299, 0.48429, 0.47761, 0.47600, 0.47959, 0.48099, 0.48336, 0.48538,
    0.48170, 0.45173, 0.44853, 0.45234, 0.45825, 0.46646, 0.47150, 0.47668,
    0.45567, 0.42350, 0.42177, 0.42938, 0.43955, 0.44983, 0.45865, 0.46663,
    0.43544, 0.39949, 0.39830, 0.40775, 0.41927, 0.43207, 0.44479, 0.45655,
    0.42032, 0.37967, 0.37747, 0.38592, 0.39928, 0.41472, 0.43049, 0.44473,
    0.40553, 0.36254, 0.35795, 0.36687, 0.38155, 0.39722, 0.41595, 0.43329,
    0.61221, 0.54882, 0.48601, 0.44044, 0.41705, 0.40343, 0.39550, 0.39446,
    0.49421, 0.45177, 0.42977, 0.41477, 0.40395, 0.39652, 0.39374, 0.39068,
    0.44484, 0.40854, 0.39311, 0.38873, 0.38425, 0.38276, 0.38200, 0.38405,
    0.41196, 0.37761, 0.36762, 0.36440, 0.36465, 0.36773, 0.36937, 0.37303,
    0.38917, 0.35355, 0.34401, 0.34386, 0.34599, 0.34974, 0.35515, 0.36001,
    0.37090, 0.33117, 0.32275, 0.32257, 0.32640, 0.33309, 0.34144, 0.34728,
    0.35506, 0.31381, 0.30300, 0.30334, 0.30772, 0.31619, 0.32520, 0.33350,
    0.34288, 0.29831, 0.28736, 0.28733, 0.29234, 0.30100, 0.31042, 0.31970,
    0.58283, 0.50853, 0.43915, 0.39572, 0.36744, 0.35531, 0.34514, 0.34416,
    0.45867, 0.40587, 0.38242, 0.36633, 0.35174, 0.34327, 0.34277, 0.33954,
    0.40501, 0.36791, 0.34772, 0.33678, 0.33301, 0.33093, 0.32962, 0.32888,
    0.37290, 0.33664, 0.32120, 0.31619, 0.31420, 0.31193, 0.31446, 0.31496,
    0.35083, 0.31323, 0.29959, 0.29401, 0.29288, 0.29462, 0.29643, 0.29883,
    0.33221, 0.29429, 0.27987, 0.27513, 0.27477, 0.27660, 0.27941, 0.28281,
    0.31869, 0.27848, 0.26265, 0.25585, 0.25778, 0.25940, 0.26442, 0.26758,
    0.30774, 0.26178, 0.24686, 0.24017, 0.24037, 0.24277, 0.24788, 0.25407,
    0.55666, 0.48340, 0.41282, 0.37065, 0.34581, 0.33026, 0.32248, 0.32003,
    0.43423, 0.38121, 0.35683, 0.33696, 0.32854, 0.32124, 0.31570, 0.31621,
    0.38391, 0.34101, 0.32143, 0.31175, 0.30541, 0.30074, 0.29998, 0.30066,
    0.35112, 0.31141, 0.29669, 0.28986, 0.28259, 0.28219, 0.28211, 0.28364,
    0.33069, 0.29226, 0.27481, 0.26652, 0.26231, 0.26330, 0.26436, 0.26385,
    0.31504, 0.27339, 0.25533, 0.24895, 0.24314, 0.24430, 0.24539, 0.24630,
    0.30163, 0.25786, 0.23929, 0.22966, 0.22741, 0.22502, 0.22693, 0.22752,
    0.29161, 0.24463, 0.22560, 0.21425, 0.21105, 0.20930, 0.21211, 0.21282,
    0.54112, 0.46424, 0.39925, 0.35671, 0.33521, 0.32270, 0.31595, 0.31149,
    0.41664, 0.36889, 0.33929, 0.32711, 0.31922, 0.31135, 0.30705, 0.30905,
    0.36914, 0.32728, 0.30835, 0.29885, 0.29197, 0.28901, 0.28941, 0.29049,
    0.33972, 0.30283, 0.28532, 0.27335, 0.26925, 0.26807, 0.26803, 0.26827,
    0.32112, 0.28204, 0.26536, 0.25333, 0.25027, 0.24650, 0.24578, 0.24785,
    0.30642, 0.26503, 0.24605, 0.23421, 0.22988, 0.22622, 0.22448, 0.22564,
    0.29376, 0.24856, 0.22814, 0.21577, 0.21107, 0.20771, 0.20649, 0.20585,
    0.28420, 0.23535, 0.21348, 0.19920, 0.19338, 0.18936, 0.18748, 0.18784,
);

// Fractional node position for one axis of a cell-centred table.
struct Weights {
    lo: u32,
    hi: u32,
    frac: f32,
}

// Returns x squared, spelled out so the banned powi/powf are avoided.
fn sqr(x: f32) -> f32 {
    return x * x;
}

// Linear interpolation between a and b by t.
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Maps a relative index eta to the normalized table coordinate in [0, 1].
fn eta_unit(eta: f32) -> f32 {
    return clamp((eta - ETA_MIN) / (ETA_MAX - ETA_MIN), 0.0, 1.0);
}

// Shared cell-centred axis-weight clamping for a length-n table axis.
fn axis_weights(x: f32, n: u32) -> Weights {
    let nf = f32(n);
    let t = clamp(x * nf - 0.5, 0.0, nf - 1.0);
    // t >= 0, so the truncating cast matches the reference floor.
    let lo = u32(t);
    let hi = min(lo + 1u, n - 1u);
    var w: Weights;
    w.lo = lo;
    w.hi = hi;
    w.frac = t - f32(lo);
    return w;
}

// Flattened lookup into the baked albedo table.
fn fetch_albedo(e: u32, a: u32, m: u32) -> f32 {
    return DE_ALBEDO[e * 64u + a * 8u + m];
}

// Trilinearly interpolated single-scatter directional albedo, clamped to [0, 1].
fn single_scatter_albedo(eta: f32, alpha: f32, mu: f32) -> f32 {
    let ew = axis_weights(eta_unit(eta), ETA_SIZE);
    let aw = axis_weights(alpha, ALPHA_SIZE);
    let mw = axis_weights(mu, MU_SIZE);
    // Inner mu lerp at each of the four (eta, alpha) corner nodes.
    let e0a0 = lerp_f32(fetch_albedo(ew.lo, aw.lo, mw.lo), fetch_albedo(ew.lo, aw.lo, mw.hi), mw.frac);
    let e0a1 = lerp_f32(fetch_albedo(ew.lo, aw.hi, mw.lo), fetch_albedo(ew.lo, aw.hi, mw.hi), mw.frac);
    let e1a0 = lerp_f32(fetch_albedo(ew.hi, aw.lo, mw.lo), fetch_albedo(ew.hi, aw.lo, mw.hi), mw.frac);
    let e1a1 = lerp_f32(fetch_albedo(ew.hi, aw.hi, mw.lo), fetch_albedo(ew.hi, aw.hi, mw.hi), mw.frac);
    // Middle alpha lerp at each eta node, then the outer eta lerp.
    let e0 = lerp_f32(e0a0, e0a1, aw.frac);
    let e1 = lerp_f32(e1a0, e1a1, aw.frac);
    return clamp(lerp_f32(e0, e1, ew.frac), 0.0, 1.0);
}

// Exact unpolarized dielectric Fresnel reflectance (eta_i -> eta_t).
fn fresnel_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let ci = clamp(cos_i, 0.0, 1.0);
    let eta = eta_i / eta_t;
    let sin2_i = max(1.0 - ci * ci, 0.0);
    let sin2_t = eta * eta * sin2_i;
    if (sin2_t >= 1.0) {
        // Beyond the critical angle: all energy is reflected.
        return 1.0;
    }
    let ct = sqrt(max(1.0 - sin2_t, 0.0));
    let denom_p = eta_t * ci + eta_i * ct;
    let denom_s = eta_i * ci + eta_t * ct;
    // Denominators are positive here; the select is a defensive guard only.
    let r_parl = select(1.0, (eta_t * ci - eta_i * ct) / denom_p, denom_p > DENOM_EPS);
    let r_perp = select(1.0, (eta_i * ci - eta_t * ct) / denom_s, denom_s > DENOM_EPS);
    return 0.5 * (r_parl * r_parl + r_perp * r_perp);
}

// Smooth-interface ceiling E_smooth(eta, mu) = R + (1 - R) / eta^2.
fn smooth_albedo(eta: f32, mu: f32) -> f32 {
    let r = fresnel_dielectric(mu, 1.0, eta);
    return r + (1.0 - r) / sqr(eta);
}

// Clamped multiplicative multiple-scattering compensation factor.
fn compensation_factor(eta: f32, alpha: f32, mu: f32) -> f32 {
    let e = max(single_scatter_albedo(eta, alpha, mu), MIN_ALBEDO);
    let ceiling = smooth_albedo(eta, mu);
    return clamp(ceiling / e, 1.0, MAX_FACTOR);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var out: Result;
    out.e_ss = single_scatter_albedo(q.eta, q.alpha, q.mu);
    out.e_smooth = smooth_albedo(q.eta, q.mu);
    out.comp = compensation_factor(q.eta, q.alpha, q.mu);
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    eta: f32,
    alpha: f32,
    mu: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    e_ss: f32,
    e_smooth: f32,
    comp: f32,
    valid: u32,
}

/// One query for the rough-dielectric energy-compensation twin: the relative
/// index `eta`, the `GGX` width `alpha`, and the absolute view cosine `mu`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DielectricEnergyQuery {
    /// Relative index of refraction `eta_t / eta_i`.
    pub eta: f32,
    /// `GGX` width `alpha`.
    pub alpha: f32,
    /// Absolute view cosine `mu = |n.wo|`.
    pub mu: f32,
}

impl DielectricEnergyQuery {
    /// Builds a query from the relative index, the `GGX` width and the view
    /// cosine.
    #[must_use]
    pub fn new(eta: f32, alpha: f32, mu: f32) -> DielectricEnergyQuery {
        DielectricEnergyQuery { eta, alpha, mu }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `single_scatter_albedo`, `smooth_albedo` and `compensation_factor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DielectricEnergyResult {
    /// Single-scatter directional albedo `E(eta, alpha, mu)`.
    pub e_ss: f32,
    /// Smooth-interface ceiling `E_smooth(eta, mu)`.
    pub e_smooth: f32,
    /// Clamped multiplicative compensation factor.
    pub comp: f32,
    /// Always `1`: every input is clamped to a baked node, so no query faults.
    pub valid: u32,
}

/// Encodes one [`DielectricEnergyQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &DielectricEnergyQuery) -> GpuQuery {
    GpuQuery {
        eta: q.eta,
        alpha: q.alpha,
        mu: q.mu,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`DielectricEnergyResult`].
fn decode_result(raw: &GpuResult) -> DielectricEnergyResult {
    DielectricEnergyResult {
        e_ss: raw.e_ss,
        e_smooth: raw.e_smooth,
        comp: raw.comp,
        valid: raw.valid,
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

/// A compiled, reusable rough-dielectric energy-compensation compute pipeline,
/// twinning the `CPU` golden `single_scatter_albedo`, `smooth_albedo` and
/// `compensation_factor`.
pub struct GpuDielectricEnergy {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDielectricEnergy {
    /// Compiles the rough-dielectric energy-compensation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDielectricEnergy {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation"),
            source: ShaderSource::Wgsl(DIELECTRIC_ENERGY_COMPENSATION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDielectricEnergy {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`DielectricEnergyResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DielectricEnergyQuery],
    ) -> Vec<DielectricEnergyResult> {
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
            label: Some("prism_volumetric_dielectric_energy_compensation_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation_bind_group"),
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
            label: Some("prism_volumetric_dielectric_energy_compensation_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_dielectric_energy_compensation_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_dielectric_energy_compensation_pass"),
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
