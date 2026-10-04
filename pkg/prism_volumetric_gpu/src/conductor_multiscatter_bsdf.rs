//! GPU twin of the energy-conserving rough-conductor `BSDF`
//! (`MultiscatterConductor`) from
//! `prism_render_architecture::reference_pt::conductor_ms`.
//!
//! The golden model wraps the exact complex-index single-scatter `GGX`
//! conductor with the Kulla-Conty multiple-scattering compensation lobe from
//! `prism_render_architecture::reference_pt::ggx_energy`, recovering the energy
//! Smith masking drops and tinting it by the metal's average `Fresnel`
//! reflectance. This module reproduces only the two deterministic entry
//! points — `evaluate` (the combined `BRDF` `f_ss + F_ms * f_ms`) and `pdf`
//! (the one-sample `MIS` density `p_ss * ggx + (1 - p_ss) * cosine`) — on the
//! `GPU`; the stochastic `sample` path with its `RNG` is deliberately excluded.
//!
//! One compute thread handles one query. The kernel is portable core-`WGSL`
//! (only `+ - * / sqrt min max clamp select floor` and unsigned index math),
//! bakes both energy tables (`16x16` directional albedo + `16` average albedo)
//! as inline `const` arrays, and reproduces the `32`-node midpoint quadrature
//! for the hemispherical-average `Fresnel`, so it needs no optional device
//! feature and provably terminates.
//!
//! # Parity criterion
//!
//! Every channel threads through multiplies, adds, guarded divisions and
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact. Continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag matches exactly. `valid` is
//! `cos_o > 0 && cos_i > 0` only — the single-scatter term self-zeros on a
//! degenerate half vector while the compensation lobe still contributes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::conductor_ms`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` multiple-scattering rough-conductor kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `MultiscatterConductor::evaluate` and
/// `MultiscatterConductor::pdf` branch for branch; see the module documentation
/// for the algorithm.
const CONDUCTOR_MULTISCATTER_BSDF_WGSL: &str = r#"
// Energy-conserving rough-conductor twin: one thread per query reproduces the
// combined BRDF f_ss + F_ms * f_ms and the one-sample MIS density
// p_ss * ggx + (1 - p_ss) * cosine that MultiscatterConductor::evaluate and
// ::pdf derive from a complex index eta+i*k, a roughness and the (wo, wi,
// normal) directions. It mirrors the CPU golden branch for branch, uses only
// the portable core-WGSL subset (min/max/clamp/select/floor/sqrt and + - * /
// plus unsigned index math), bakes the Kulla-Conty energy tables inline and
// has a single bounded 32-node loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::conductor_ms；无第三方
// 引擎源码或衍生代码。

const PI: f32 = 3.14159265358979;
const INV_PI: f32 = 0.31830988618;
const EPS_LEN_SQ: f32 = 1e-12;
const MIN_ALPHA: f32 = 0.001;
const AVG_FRESNEL_NODES: u32 = 32u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Complex index real part eta (per channel).
    ex: f32,
    ey: f32,
    ez: f32,
    // Complex index extinction k (per channel).
    kx: f32,
    ky: f32,
    kz: f32,
    // Perceptual roughness in [0, 1].
    roughness: f32,
    // Outgoing direction (away from the surface), expected unit length.
    wox: f32,
    woy: f32,
    woz: f32,
    // Incoming direction (away from the surface), expected unit length.
    wix: f32,
    wiy: f32,
    wiz: f32,
    // Shading normal, expected unit length.
    nx: f32,
    ny: f32,
    nz: f32,
}

struct Result {
    vx: f32,
    vy: f32,
    vz: f32,
    pdf: f32,
    valid: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Baked single-scatter directional albedo E(mu, alpha) under a white Fresnel,
// flattened row-major as idx = a * 16 + c (outer alpha node, inner cos node).
const ALBEDO: array<f32, 256> = array<f32, 256>(
    0.89053, 0.94328, 0.97589, 0.98818, 0.99206, 0.99439, 0.99607, 0.99685, 0.99744, 0.99806, 0.99817, 0.99817, 0.99858, 0.99886, 0.99885, 0.99904,
    0.93451, 0.88498, 0.89344, 0.91887, 0.93761, 0.95195, 0.96259, 0.96975, 0.97565, 0.97898, 0.98269, 0.98372, 0.98662, 0.98678, 0.98818, 0.98973,
    0.94954, 0.88988, 0.87099, 0.87141, 0.88501, 0.89715, 0.91246, 0.92474, 0.93381, 0.94122, 0.94994, 0.95394, 0.95931, 0.96293, 0.96593, 0.96903,
    0.95385, 0.89693, 0.86369, 0.85375, 0.85025, 0.85593, 0.86688, 0.87682, 0.88543, 0.89518, 0.90309, 0.91140, 0.91951, 0.92397, 0.93080, 0.93376,
    0.95163, 0.89433, 0.86047, 0.83623, 0.82537, 0.82326, 0.82529, 0.83233, 0.83790, 0.84373, 0.85562, 0.86116, 0.87050, 0.87715, 0.88401, 0.88939,
    0.94932, 0.88946, 0.84891, 0.82225, 0.80592, 0.79468, 0.79105, 0.79034, 0.79559, 0.80038, 0.80365, 0.80971, 0.81667, 0.82446, 0.82961, 0.83616,
    0.94567, 0.88040, 0.83628, 0.80512, 0.78428, 0.76896, 0.76043, 0.75496, 0.75359, 0.75547, 0.75319, 0.75845, 0.76217, 0.76699, 0.77263, 0.77446,
    0.94142, 0.87175, 0.82330, 0.79018, 0.76237, 0.74273, 0.73166, 0.72252, 0.71253, 0.70897, 0.70770, 0.70686, 0.70679, 0.70787, 0.71245, 0.71734,
    0.93671, 0.86121, 0.80691, 0.76759, 0.73937, 0.72034, 0.69780, 0.68750, 0.67520, 0.66904, 0.66345, 0.66195, 0.65407, 0.65477, 0.65289, 0.65404,
    0.92915, 0.84826, 0.79412, 0.75430, 0.71678, 0.69452, 0.66872, 0.65459, 0.64058, 0.62912, 0.62176, 0.61331, 0.60612, 0.60542, 0.60027, 0.59651,
    0.92425, 0.83505, 0.77650, 0.73150, 0.69712, 0.66509, 0.64125, 0.61974, 0.60482, 0.59115, 0.57870, 0.56560, 0.56079, 0.55360, 0.54713, 0.54434,
    0.91756, 0.82414, 0.75863, 0.71164, 0.67160, 0.63993, 0.61433, 0.59389, 0.57071, 0.55353, 0.53875, 0.52792, 0.51651, 0.50474, 0.50088, 0.49233,
    0.91143, 0.81112, 0.74357, 0.68803, 0.65290, 0.61420, 0.58876, 0.55983, 0.53882, 0.51787, 0.50314, 0.48608, 0.47485, 0.46401, 0.45315, 0.44662,
    0.90468, 0.79955, 0.73071, 0.67325, 0.62805, 0.59150, 0.55480, 0.52925, 0.50862, 0.48782, 0.46760, 0.45411, 0.43806, 0.42644, 0.41378, 0.40239,
    0.89889, 0.78596, 0.71155, 0.65008, 0.60615, 0.56783, 0.53284, 0.50402, 0.47753, 0.45741, 0.43591, 0.41926, 0.39868, 0.38978, 0.37722, 0.36418,
    0.89346, 0.77409, 0.69544, 0.63218, 0.58563, 0.54142, 0.50747, 0.47595, 0.44834, 0.42990, 0.40566, 0.38615, 0.36961, 0.35676, 0.34430, 0.32918
);

// Baked cosine-weighted average albedo E_avg(alpha) for each alpha node.
const AVG_ALBEDO: array<f32, 16> = array<f32, 16>(
    0.99608, 0.97482, 0.94267, 0.90348, 0.86012, 0.81500, 0.76850, 0.72253,
    0.67764, 0.63568, 0.59390, 0.55526, 0.51834, 0.48476, 0.45266, 0.42318
);

// Exact unpolarized conductor Fresnel for one channel at incidence cosine
// cos_theta with complex index eta + i*k; byte-identical to the golden
// fresnel_conductor_channel, only sqrt/products/quotients.
fn fresnel_channel(cos_theta: f32, eta: f32, k: f32) -> f32 {
    let cos_i = clamp(cos_theta, 0.0, 1.0);
    let cos2 = cos_i * cos_i;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2b2 = sqrt(max(t0 * t0 + 4.0 * eta2 * k2, 0.0));
    let a = sqrt(max(0.5 * (a2b2 + t0), 0.0));
    let t1 = a2b2 + cos2;
    let t2 = 2.0 * a * cos_i;
    let denom_s = t1 + t2;
    let r_s = select(1.0, (t1 - t2) / denom_s, denom_s > 0.0);
    let t3 = cos2 * a2b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    let r_p = select(r_s, r_s * (t3 - t4) / denom_p, denom_p > 0.0);
    return clamp(0.5 * (r_s + r_p), 0.0, 1.0);
}

// Per-channel exact conductor Fresnel reflectance.
fn fresnel_conductor(eta: vec3<f32>, k: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return vec3<f32>(
        fresnel_channel(cos_theta, eta.x, k.x),
        fresnel_channel(cos_theta, eta.y, k.y),
        fresnel_channel(cos_theta, eta.z, k.z),
    );
}

// Hemispherical cosine-weighted average Fresnel F_avg via 32-node midpoint
// quadrature: F_avg = 2 * integral_0^1 F(mu) mu d mu.
fn average_fresnel(eta: vec3<f32>, k: vec3<f32>) -> vec3<f32> {
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < AVG_FRESNEL_NODES; i = i + 1u) {
        let mu = (f32(i) + 0.5) / f32(AVG_FRESNEL_NODES);
        let w = 2.0 * mu / f32(AVG_FRESNEL_NODES);
        acc = acc + fresnel_conductor(eta, k, mu) * w;
    }
    return acc;
}

// Isotropic GGX normal distribution D(h) from the half-vector cosine cos_h.
fn ggx_distribution(cos_h: f32, alpha: f32) -> f32 {
    if (cos_h <= 0.0) {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let c2 = cos_h * cos_h;
    let denom = c2 * (a2 - 1.0) + 1.0;
    return a2 * INV_PI / (denom * denom);
}

// Smith Lambda auxiliary for a direction whose cosine to the normal is cos_w.
fn ggx_lambda(cos_w: f32, alpha: f32) -> f32 {
    let c = abs(cos_w);
    if (c >= 1.0) {
        return 0.0;
    }
    let c2 = c * c;
    let tan2 = (1.0 - c2) / c2;
    let a2 = alpha * alpha;
    return 0.5 * (sqrt(1.0 + a2 * tan2) - 1.0);
}

fn ggx_g1(cos_w: f32, alpha: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(cos_w, alpha));
}

fn ggx_g2(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
    return 1.0 / (1.0 + ggx_lambda(cos_o, alpha) + ggx_lambda(cos_i, alpha));
}

fn ggx_reflection_pdf(cos_o: f32, cos_h: f32, alpha: f32) -> f32 {
    if (cos_o <= 0.0) {
        return 0.0;
    }
    return ggx_g1(cos_o, alpha) * ggx_distribution(cos_h, alpha) / (4.0 * cos_o);
}

// Fractional node position for a continuous axis coordinate x in [0, 1].
struct AxisW {
    lo: u32,
    hi: u32,
    frac: f32,
}

fn axis_weights(x: f32) -> AxisW {
    let t = clamp(x * 16.0 - 0.5, 0.0, 15.0);
    let lo_f = floor(t);
    let lo = u32(lo_f);
    let hi = min(lo + 1u, 15u);
    var w: AxisW;
    w.lo = lo;
    w.hi = hi;
    w.frac = t - lo_f;
    return w;
}

// Single-scatter directional albedo E(cos_theta, alpha), bilinear from the LUT.
fn directional_albedo(cos_theta: f32, alpha: f32) -> f32 {
    let aw = axis_weights(alpha);
    let cw = axis_weights(cos_theta);
    let base_lo = aw.lo * 16u;
    let base_hi = aw.hi * 16u;
    let a_lo_c_lo = ALBEDO[base_lo + cw.lo];
    let a_lo_c_hi = ALBEDO[base_lo + cw.hi];
    let a_hi_c_lo = ALBEDO[base_hi + cw.lo];
    let a_hi_c_hi = ALBEDO[base_hi + cw.hi];
    let row_lo = a_lo_c_lo + (a_lo_c_hi - a_lo_c_lo) * cw.frac;
    let row_hi = a_hi_c_lo + (a_hi_c_hi - a_hi_c_lo) * cw.frac;
    return clamp(row_lo + (row_hi - row_lo) * aw.frac, 0.0, 1.0);
}

// Cosine-weighted average albedo E_avg(alpha), linear from the LUT.
fn average_albedo(alpha: f32) -> f32 {
    let aw = axis_weights(alpha);
    let lo = AVG_ALBEDO[aw.lo];
    let hi = AVG_ALBEDO[aw.hi];
    return clamp(lo + (hi - lo) * aw.frac, 0.0, 1.0);
}

// Scalar Kulla-Conty multiple-scattering lobe.
fn multiscatter_lobe(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
    let e_avg = average_albedo(alpha);
    let denom = 1.0 - e_avg;
    if (denom <= 1.0e-4) {
        return 0.0;
    }
    let e_o = directional_albedo(cos_o, alpha);
    let e_i = directional_albedo(cos_i, alpha);
    return (1.0 - e_o) * (1.0 - e_i) / (PI * denom);
}

// Per-channel multiple-scatter Fresnel tint F_ms.
fn ms_fresnel_channel(f: f32, e_avg: f32, one_minus: f32) -> f32 {
    let denom = 1.0 - f * one_minus;
    return select(f * f * e_avg / denom, f, denom <= 1.0e-4);
}

fn multiscatter_fresnel(f_avg: vec3<f32>, alpha: f32) -> vec3<f32> {
    let e_avg = average_albedo(alpha);
    let one_minus = 1.0 - e_avg;
    return vec3<f32>(
        ms_fresnel_channel(f_avg.x, e_avg, one_minus),
        ms_fresnel_channel(f_avg.y, e_avg, one_minus),
        ms_fresnel_channel(f_avg.z, e_avg, one_minus),
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let eta = vec3<f32>(q.ex, q.ey, q.ez);
    let k = vec3<f32>(q.kx, q.ky, q.kz);
    let wo = vec3<f32>(q.wox, q.woy, q.woz);
    let wi = vec3<f32>(q.wix, q.wiy, q.wiz);
    let normal = vec3<f32>(q.nx, q.ny, q.nz);

    var out: Result;
    out.vx = 0.0;
    out.vy = 0.0;
    out.vz = 0.0;
    out.pdf = 0.0;
    out.valid = 0u;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;

    let cos_o = dot(normal, wo);
    let cos_i = dot(normal, wi);
    if (cos_o <= 0.0 || cos_i <= 0.0) {
        results[idx] = out;
        return;
    }

    let r = clamp(q.roughness, 0.0, 1.0);
    let alpha = max(r * r, MIN_ALPHA);

    // World-space half vector with a degenerate-length guard matching the
    // golden normalize_or_zero(len_sq > EPS_LEN_SQ) contract.
    let half_sum = wo + wi;
    let hls = dot(half_sum, half_sum);
    let hv = hls > EPS_LEN_SQ;
    let inv = select(0.0, 1.0 / sqrt(hls), hv);
    let half_vec = half_sum * inv;
    let cos_h = dot(normal, half_vec);

    // Single-scatter exact conductor term; self-zeros on a degenerate or
    // back-facing half vector.
    var single = vec3<f32>(0.0, 0.0, 0.0);
    if (hv && cos_h > 0.0) {
        let d = ggx_distribution(cos_h, alpha);
        let g2 = ggx_g2(cos_o, cos_i, alpha);
        let woh = max(dot(wo, half_vec), 0.0);
        let fres = fresnel_conductor(eta, k, woh);
        single = fres * (d * g2 / (4.0 * cos_o * cos_i));
    }

    // Kulla-Conty compensation lobe tinted by the average Fresnel.
    let f_avg = average_fresnel(eta, k);
    let lobe = multiscatter_lobe(cos_o, cos_i, alpha);
    let tint = multiscatter_fresnel(f_avg, alpha);
    let value = single + tint * lobe;

    // One-sample MIS density mixing the GGX and cosine strategies.
    let p_ss = clamp(average_albedo(alpha), 0.1, 0.9);
    var ggx = 0.0;
    if (hv) {
        ggx = ggx_reflection_pdf(cos_o, cos_h, alpha);
    }
    let diffuse = max(dot(normal, wi), 0.0) * INV_PI;
    let pdf = p_ss * ggx + (1.0 - p_ss) * diffuse;

    out.vx = value.x;
    out.vy = value.y;
    out.vz = value.z;
    out.pdf = pdf;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CONDUCTOR_MULTISCATTER_BSDF_WGSL`].
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
/// The `vec3` inputs are flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ex: f32,
    ey: f32,
    ez: f32,
    kx: f32,
    ky: f32,
    kz: f32,
    roughness: f32,
    wox: f32,
    woy: f32,
    woz: f32,
    wix: f32,
    wiy: f32,
    wiz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The trailing pad lanes make the host stride byte-exact with the
/// shader's `32`-byte `Result`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    vx: f32,
    vy: f32,
    vz: f32,
    pdf: f32,
    valid: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// One query for the multiple-scattering rough-conductor twin: the metal's
/// complex index `eta + i*k`, the perceptual `roughness`, and the outgoing,
/// incoming and normal directions.
///
/// Both the combined `BRDF` value and the `MIS` reflection density are derived
/// from this one tuple, so a single query exercises the whole twinned core at
/// once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConductorMultiscatterBsdfQuery {
    /// Per-channel real index of refraction `eta`.
    pub eta: [f32; 3],
    /// Per-channel extinction coefficient `k`.
    pub k: [f32; 3],
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f32,
    /// Outgoing direction (away from the surface), expected unit length.
    pub wo: [f32; 3],
    /// Incoming direction (away from the surface), expected unit length.
    pub wi: [f32; 3],
    /// Shading normal, expected unit length.
    pub normal: [f32; 3],
}

impl ConductorMultiscatterBsdfQuery {
    /// Builds a query from the complex index, the perceptual roughness and the
    /// three directions.
    #[must_use]
    pub fn new(
        eta: [f32; 3],
        k: [f32; 3],
        roughness: f32,
        wo: [f32; 3],
        wi: [f32; 3],
        normal: [f32; 3],
    ) -> ConductorMultiscatterBsdfQuery {
        ConductorMultiscatterBsdfQuery {
            eta,
            k,
            roughness,
            wo,
            wi,
            normal,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `MultiscatterConductor::evaluate` and `MultiscatterConductor::pdf` outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConductorMultiscatterBsdfResult {
    /// The combined `BRDF` value `f_ss + F_ms * f_ms` per channel; cleared to
    /// zero when invalid.
    pub value: [f32; 3],
    /// The one-sample `MIS` solid-angle reflection density; zero when invalid.
    pub pdf: f32,
    /// `1` when both directions lie above the surface, `0` for a degenerate
    /// query (either direction below the horizon).
    pub valid: u32,
}

/// Encodes one [`ConductorMultiscatterBsdfQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &ConductorMultiscatterBsdfQuery) -> GpuQuery {
    GpuQuery {
        ex: q.eta[0],
        ey: q.eta[1],
        ez: q.eta[2],
        kx: q.k[0],
        ky: q.k[1],
        kz: q.k[2],
        roughness: q.roughness,
        wox: q.wo[0],
        woy: q.wo[1],
        woz: q.wo[2],
        wix: q.wi[0],
        wiy: q.wi[1],
        wiz: q.wi[2],
        nx: q.normal[0],
        ny: q.normal[1],
        nz: q.normal[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`ConductorMultiscatterBsdfResult`].
fn decode_result(raw: &GpuResult) -> ConductorMultiscatterBsdfResult {
    ConductorMultiscatterBsdfResult {
        value: [raw.vx, raw.vy, raw.vz],
        pdf: raw.pdf,
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

/// A compiled, reusable multiple-scattering rough-conductor compute pipeline,
/// twinning the `CPU` golden `MultiscatterConductor::evaluate` and
/// `MultiscatterConductor::pdf`.
pub struct GpuConductorMultiscatterBsdf {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuConductorMultiscatterBsdf {
    /// Compiles the kernel and builds the reusable bind-group layout and compute
    /// pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConductorMultiscatterBsdf {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf"),
            source: ShaderSource::Wgsl(CONDUCTOR_MULTISCATTER_BSDF_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConductorMultiscatterBsdf {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ConductorMultiscatterBsdfResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ConductorMultiscatterBsdfQuery],
    ) -> Vec<ConductorMultiscatterBsdfResult> {
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
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_bind_group"),
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
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_conductor_multiscatter_bsdf_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_conductor_multiscatter_bsdf_pass"),
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
