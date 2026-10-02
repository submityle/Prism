//! `wgpu` compute twin of the anisotropic base plus dielectric clearcoat
//! reflectance contract
//! ([`anisotropic_clearcoat`](prism_render_architecture::particle::anisotropic_clearcoat),
//! particle design §15 "各向异性/清漆", §17 `PBR` closure).
//!
//! The `CPU` golden
//! [`anisotropic_clearcoat`](prism_render_architecture::particle::anisotropic_clearcoat)
//! owns the closed-form microfacet terms a brushed-metal, carbon-fibre or
//! car-paint particle needs: the anisotropy-to-aspect mapping
//! ([`aspect_ratio`](prism_render_architecture::particle::anisotropic_clearcoat::aspect_ratio),
//! [`anisotropic_alphas`](prism_render_architecture::particle::anisotropic_clearcoat::anisotropic_alphas)),
//! the anisotropic `GGX` base lobe
//! ([`ggx_aniso_ndf`](prism_render_architecture::particle::anisotropic_clearcoat::ggx_aniso_ndf),
//! [`ggx_aniso_visibility`](prism_render_architecture::particle::anisotropic_clearcoat::ggx_aniso_visibility)),
//! the isotropic clearcoat lobe
//! ([`clearcoat_ggx_ndf`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_ggx_ndf),
//! [`clearcoat_visibility`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_visibility)),
//! the `Schlick` clearcoat `Fresnel` and its base-layer attenuation
//! ([`clearcoat_fresnel`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_fresnel),
//! [`clearcoat_attenuation`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_attenuation)),
//! and the composite
//! [`AnisotropicClearcoatParams::evaluate`](prism_render_architecture::particle::anisotropic_clearcoat::AnisotropicClearcoatParams::evaluate)
//! that assembles them for one shading fragment. [`GpuAnisotropicClearcoat`] is
//! the on-device twin: one thread solves one query, so a passing real-device
//! parity test is direct evidence the ported kernel evaluates the same closed
//! form the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every reflectance term is reproduced through a tagged
//! [`AnisotropicClearcoatQuery`]: one variant per reference function plus the
//! composite [`AnisotropicClearcoatQuery::Evaluate`] that mirrors
//! [`AnisotropicClearcoatParams::evaluate`](prism_render_architecture::particle::anisotropic_clearcoat::AnisotropicClearcoatParams::evaluate).
//! The host-side `std430` byte-layout helpers
//! [`AnisotropicClearcoatParams::to_std430_bits`](prism_render_architecture::particle::anisotropic_clearcoat::AnisotropicClearcoatParams::to_std430_bits)
//! and
//! [`AnisotropicClearcoatParams::storage_size`](prism_render_architecture::particle::anisotropic_clearcoat::AnisotropicClearcoatParams::storage_size)
//! are deliberately *not* twinned: they are host packing utilities, not kernel
//! math, so there is nothing on device to compare them against.
//!
//! # Correctness model
//!
//! Every term is a rational function of the geometry dot products with at most
//! one `sqrt` (the aspect ratio, the `Smith` `Lambda`), so `CPU` and `GPU`
//! evaluate the same closed form in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! parity test therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every continuous lane, tight enough to catch a
//! dropped term or a swapped axis yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! Every division is guarded: the `GGX` denominators fall back to a
//! large-but-finite spike when they drop below `MIN_DENOM`, the `Smith` sum is
//! floored at `MIN_DENOM`, the alphas are floored at `MIN_ALPHA`, and the
//! composite `evaluate` normalizes each input vector with a zero-length
//! fallback, so a mirror-smooth or degenerate input yields a finite value,
//! never a `NaN`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `min`, `max`, `dot`, `sqrt` and `+ - * /` — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no inverse trigonometry and no optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The `Schlick` fifth
//! power is expanded to an explicit product `m * m * m * m * m`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::anisotropic_clearcoat`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` anisotropic-clearcoat kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` branches
/// on a per-query op code into the `CPU` golden
/// [`anisotropic_clearcoat`](prism_render_architecture::particle::anisotropic_clearcoat)
/// terms; see the module documentation for the algorithm.
const ANISOTROPIC_CLEARCOAT_WGSL: &str = r#"
// Anisotropic base plus dielectric clearcoat twin: one thread per query
// reproduces one reference function selected by `op`, or the composite evaluate
// (op 8). It mirrors the CPU golden `particle::anisotropic_clearcoat` term for
// term, uses only the portable core-WGSL subset (abs/clamp/min/max/dot/sqrt and
// + - * /), needs no transcendental call and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12. The Schlick fifth power is an
// explicit product. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::anisotropic_clearcoat；
// 无第三方引擎源码或衍生代码。

// Reciprocal of pi, matching the reference INV_PI (core::f32::consts::FRAC_1_PI).
// Written to full f32 precision so it rounds to the identical pattern.
const INV_PI: f32 = 0.31830987334251404;
// Lower clamp on a GGX alpha (linear roughness), matching the reference MIN_ALPHA.
const MIN_ALPHA: f32 = 1.0e-4;
// Generic lower bound on a denominator that could reach zero, matching MIN_DENOM.
const MIN_DENOM: f32 = 1.0e-8;
// Slope of the anisotropy-to-aspect mapping, matching ASPECT_ANISOTROPY_SCALE.
const ASPECT_ANISOTROPY_SCALE: f32 = 0.9;
// Dielectric clearcoat reflectance at normal incidence, matching CLEARCOAT_F0.
const CLEARCOAT_F0: f32 = 0.04;
// Squared-length floor for normalize_or_zero, matching the reference EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Shading frame and directions for the composite evaluate (op 8); each vec3
    // carries a trailing pad lane to stay 16-byte aligned on device.
    tangent: vec3<f32>,
    pad_t: f32,
    bitangent: vec3<f32>,
    pad_b: f32,
    normal: vec3<f32>,
    pad_n: f32,
    view: vec3<f32>,
    pad_v: f32,
    light: vec3<f32>,
    pad_l: f32,
    // Half-vector cosines for the standalone NDF ops.
    n_dot_h: f32,
    t_dot_h: f32,
    b_dot_h: f32,
    // Clamped normal cosines shared by the visibility ops.
    n_dot_v: f32,
    n_dot_l: f32,
    // Tangent-frame projections of view and light for the anisotropic visibility.
    t_dot_v: f32,
    b_dot_v: f32,
    t_dot_l: f32,
    b_dot_l: f32,
    // Material controls for the aspect / alpha / evaluate ops.
    roughness: f32,
    anisotropy: f32,
    clearcoat_roughness: f32,
    clearcoat_strength: f32,
    // Pre-supplied alphas for the standalone base ops, the clearcoat alpha for
    // the clearcoat ops, the Fresnel cosine and the attenuation input.
    alpha_t: f32,
    alpha_b: f32,
    alpha: f32,
    cos_theta: f32,
    fc: f32,
    // Op classification code (0..=8) plus a pad word to fill the slot.
    op: u32,
    pad0: f32,
}

struct Result {
    // Up to six scalar outputs; the interpretation depends on the query op.
    // For evaluate (op 8): base_ndf, base_visibility, base_attenuation,
    // clearcoat_ndf, clearcoat_visibility, clearcoat_fresnel. For the alphas op:
    // (alpha_t, alpha_b). For every single-valued op the answer is in `a`.
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Decoded composite answer before packing, mirroring the reference
// AnisotropicClearcoatSample.
struct Sample {
    base_ndf: f32,
    base_visibility: f32,
    base_attenuation: f32,
    clearcoat_ndf: f32,
    clearcoat_visibility: f32,
    clearcoat_fresnel: f32,
}

// Squares a scalar via one multiplication; mirrors the reference `square`.
fn square(x: f32) -> f32 {
    return x * x;
}

// Clamps a scalar into 0..=1; mirrors the reference `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Unit vector along v, or the zero vector when v is numerically zero; mirrors
// the reference Vec3::normalize_or_zero so normalization never yields a NaN.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// aspect = sqrt(1 - 0.9 * |clamp(anisotropy, -1, 1)|); mirrors `aspect_ratio`.
fn aspect_ratio(anisotropy: f32) -> f32 {
    let a = abs(clamp(anisotropy, -1.0, 1.0));
    return sqrt(1.0 - ASPECT_ANISOTROPY_SCALE * a);
}

// Tangent/bitangent GGX alphas from roughness and anisotropy; mirrors
// `anisotropic_alphas`. Returned as (alpha_t, alpha_b).
fn anisotropic_alphas(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let alpha = square(clamp01(roughness));
    let aspect = aspect_ratio(anisotropy);
    let alpha_t = max(alpha / aspect, MIN_ALPHA);
    let alpha_b = max(alpha * aspect, MIN_ALPHA);
    return vec2<f32>(alpha_t, alpha_b);
}

// Anisotropic GGX NDF in the Burley form; mirrors `ggx_aniso_ndf`.
fn ggx_aniso_ndf(n_dot_h: f32, t_dot_h: f32, b_dot_h: f32, alpha_t: f32, alpha_b: f32) -> f32 {
    let at = max(alpha_t, MIN_ALPHA);
    let ab = max(alpha_b, MIN_ALPHA);
    let inner = square(t_dot_h / at) + square(b_dot_h / ab) + square(n_dot_h);
    let denom = at * ab * square(inner);
    if (denom < MIN_DENOM) {
        return INV_PI / MIN_DENOM;
    }
    return INV_PI / denom;
}

// One side of the height-correlated anisotropic Smith term; mirrors
// `smith_lambda_term`.
fn smith_lambda_term(
    alpha_t: f32,
    alpha_b: f32,
    t_dot_w: f32,
    b_dot_w: f32,
    n_dot_w: f32,
    n_dot_other: f32,
) -> f32 {
    let len = sqrt(square(alpha_t * t_dot_w) + square(alpha_b * b_dot_w) + square(n_dot_w));
    return n_dot_other * len;
}

// Height-correlated anisotropic Smith visibility; mirrors
// `ggx_aniso_visibility`.
fn ggx_aniso_visibility(
    alpha_t: f32,
    alpha_b: f32,
    t_dot_v: f32,
    b_dot_v: f32,
    n_dot_v: f32,
    t_dot_l: f32,
    b_dot_l: f32,
    n_dot_l: f32,
) -> f32 {
    let lambda_v = smith_lambda_term(alpha_t, alpha_b, t_dot_v, b_dot_v, n_dot_v, n_dot_l);
    let lambda_l = smith_lambda_term(alpha_t, alpha_b, t_dot_l, b_dot_l, n_dot_l, n_dot_v);
    let denom = max(lambda_v + lambda_l, MIN_DENOM);
    return 0.5 / denom;
}

// Isotropic Trowbridge-Reitz NDF for the clearcoat lobe; mirrors
// `clearcoat_ggx_ndf`.
fn clearcoat_ggx_ndf(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = square(max(alpha, MIN_ALPHA));
    let kernel = square(clamp01(n_dot_h)) * (a2 - 1.0) + 1.0;
    let denom = square(kernel);
    if (denom < MIN_DENOM) {
        return a2 * INV_PI / MIN_DENOM;
    }
    return a2 * INV_PI / denom;
}

// Height-correlated isotropic Smith visibility for the clearcoat lobe; mirrors
// `clearcoat_visibility`.
fn clearcoat_visibility(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = square(max(alpha, MIN_ALPHA));
    let nv = clamp01(n_dot_v);
    let nl = clamp01(n_dot_l);
    let ggx_v = nl * sqrt(square(nv) * (1.0 - a2) + a2);
    let ggx_l = nv * sqrt(square(nl) * (1.0 - a2) + a2);
    let denom = max(ggx_v + ggx_l, MIN_DENOM);
    return 0.5 / denom;
}

// Schlick Fresnel at the fixed clearcoat F0; mirrors `clearcoat_fresnel`
// (fresnel_schlick with F0 = 0.04). The fifth power is an explicit product.
fn clearcoat_fresnel(cos_theta: f32) -> f32 {
    let cos = clamp01(cos_theta);
    let m = 1.0 - cos;
    return CLEARCOAT_F0 + (1.0 - CLEARCOAT_F0) * (m * m * m * m * m);
}

// Energy left for the base layer after the coat reflects fc; mirrors
// `clearcoat_attenuation`.
fn clearcoat_attenuation(fc: f32) -> f32 {
    return 1.0 - clamp01(fc);
}

// Composite base + clearcoat evaluate; mirrors
// AnisotropicClearcoatParams::evaluate.
fn evaluate(q: Query) -> Sample {
    let t = normalize_or_zero(q.tangent);
    let b = normalize_or_zero(q.bitangent);
    let n = normalize_or_zero(q.normal);
    let v = normalize_or_zero(q.view);
    let l = normalize_or_zero(q.light);
    let h = normalize_or_zero(v + l);

    let alphas = anisotropic_alphas(q.roughness, q.anisotropy);

    let n_dot_h = dot(n, h);
    let t_dot_h = dot(t, h);
    let b_dot_h = dot(b, h);
    let base_ndf = ggx_aniso_ndf(n_dot_h, t_dot_h, b_dot_h, alphas.x, alphas.y);

    let n_dot_v = clamp01(dot(n, v));
    let n_dot_l = clamp01(dot(n, l));
    let base_visibility = ggx_aniso_visibility(
        alphas.x,
        alphas.y,
        dot(t, v),
        dot(b, v),
        n_dot_v,
        dot(t, l),
        dot(b, l),
        n_dot_l,
    );

    let cc_alpha = max(square(clamp01(q.clearcoat_roughness)), MIN_ALPHA);
    let clearcoat_ndf = clearcoat_ggx_ndf(n_dot_h, cc_alpha);
    let clearcoat_vis = clearcoat_visibility(n_dot_v, n_dot_l, cc_alpha);

    let l_dot_h = clamp01(dot(l, h));
    let strength = clamp01(q.clearcoat_strength);
    let fc = clearcoat_fresnel(l_dot_h) * strength;

    var s: Sample;
    s.base_ndf = base_ndf;
    s.base_visibility = base_visibility;
    s.base_attenuation = clearcoat_attenuation(fc);
    s.clearcoat_ndf = clearcoat_ndf;
    s.clearcoat_visibility = clearcoat_vis;
    s.clearcoat_fresnel = fc;
    return s;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.a = 0.0;
    out.b = 0.0;
    out.c = 0.0;
    out.d = 0.0;
    out.e = 0.0;
    out.f = 0.0;
    out.pad0 = 0.0;
    out.pad1 = 0.0;

    if (q.op == 0u) {
        out.a = aspect_ratio(q.anisotropy);
    } else if (q.op == 1u) {
        let alphas = anisotropic_alphas(q.roughness, q.anisotropy);
        out.a = alphas.x;
        out.b = alphas.y;
    } else if (q.op == 2u) {
        out.a = ggx_aniso_ndf(q.n_dot_h, q.t_dot_h, q.b_dot_h, q.alpha_t, q.alpha_b);
    } else if (q.op == 3u) {
        out.a = ggx_aniso_visibility(
            q.alpha_t,
            q.alpha_b,
            q.t_dot_v,
            q.b_dot_v,
            q.n_dot_v,
            q.t_dot_l,
            q.b_dot_l,
            q.n_dot_l,
        );
    } else if (q.op == 4u) {
        out.a = clearcoat_ggx_ndf(q.n_dot_h, q.alpha);
    } else if (q.op == 5u) {
        out.a = clearcoat_visibility(q.n_dot_v, q.n_dot_l, q.alpha);
    } else if (q.op == 6u) {
        out.a = clearcoat_fresnel(q.cos_theta);
    } else if (q.op == 7u) {
        out.a = clearcoat_attenuation(q.fc);
    } else {
        let s = evaluate(q);
        out.a = s.base_ndf;
        out.b = s.base_visibility;
        out.c = s.base_attenuation;
        out.d = s.clearcoat_ndf;
        out.e = s.clearcoat_visibility;
        out.f = s.clearcoat_fresnel;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`ANISOTROPIC_CLEARCOAT_WGSL`].
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
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device; the scalar block that follows begins on a `16`-byte boundary.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Shading-frame tangent (`evaluate` only).
    tangent: [f32; 3],
    /// Pad lane after `tangent`.
    pad_t: f32,
    /// Shading-frame bitangent (`evaluate` only).
    bitangent: [f32; 3],
    /// Pad lane after `bitangent`.
    pad_b: f32,
    /// Shading-frame normal (`evaluate` only).
    normal: [f32; 3],
    /// Pad lane after `normal`.
    pad_n: f32,
    /// View direction (`evaluate` only).
    view: [f32; 3],
    /// Pad lane after `view`.
    pad_v: f32,
    /// Light direction (`evaluate` only).
    light: [f32; 3],
    /// Pad lane after `light`.
    pad_l: f32,
    /// Half-vector normal cosine for the standalone `NDF` ops.
    n_dot_h: f32,
    /// Half-vector tangent cosine.
    t_dot_h: f32,
    /// Half-vector bitangent cosine.
    b_dot_h: f32,
    /// Clamped view cosine for the visibility ops.
    n_dot_v: f32,
    /// Clamped light cosine for the visibility ops.
    n_dot_l: f32,
    /// View tangent projection for the anisotropic visibility.
    t_dot_v: f32,
    /// View bitangent projection.
    b_dot_v: f32,
    /// Light tangent projection.
    t_dot_l: f32,
    /// Light bitangent projection.
    b_dot_l: f32,
    /// Base-layer perceptual roughness.
    roughness: f32,
    /// Anisotropy control in `-1..=1`.
    anisotropy: f32,
    /// Clearcoat perceptual roughness.
    clearcoat_roughness: f32,
    /// Clearcoat presence in `0..=1`.
    clearcoat_strength: f32,
    /// Pre-supplied tangent alpha for the standalone base ops.
    alpha_t: f32,
    /// Pre-supplied bitangent alpha for the standalone base ops.
    alpha_b: f32,
    /// Clearcoat linear roughness for the standalone clearcoat ops.
    alpha: f32,
    /// `Fresnel` cosine for the standalone `Fresnel` op.
    cos_theta: f32,
    /// Reflected fraction for the standalone attenuation op.
    fc: f32,
    /// Op classification code (`0..=8`).
    op: u32,
    /// Padding lane filling the final `16`-byte slot.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: up to six scalar outputs plus two pad lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// First output lane (the single answer for scalar ops).
    a: f32,
    /// Second output lane.
    b: f32,
    /// Third output lane.
    c: f32,
    /// Fourth output lane.
    d: f32,
    /// Fifth output lane.
    e: f32,
    /// Sixth output lane.
    f: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
}

/// One tagged query selecting which reference term the kernel evaluates.
///
/// There is one variant per `CPU` golden function plus the composite
/// [`AnisotropicClearcoatQuery::Evaluate`]; a field a variant does not name is
/// ignored. The discriminant order matches the `u32` op codes the kernel
/// branches on.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::anisotropic_clearcoat`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnisotropicClearcoatQuery {
    /// The anisotropy-to-aspect ratio, matching
    /// [`aspect_ratio`](prism_render_architecture::particle::anisotropic_clearcoat::aspect_ratio).
    AspectRatio {
        /// Anisotropy control in `-1..=1`.
        anisotropy: f32,
    },
    /// The tangent/bitangent `GGX` alphas, matching
    /// [`anisotropic_alphas`](prism_render_architecture::particle::anisotropic_clearcoat::anisotropic_alphas).
    AnisotropicAlphas {
        /// Base-layer perceptual roughness.
        roughness: f32,
        /// Anisotropy control in `-1..=1`.
        anisotropy: f32,
    },
    /// The anisotropic `GGX` `NDF`, matching
    /// [`ggx_aniso_ndf`](prism_render_architecture::particle::anisotropic_clearcoat::ggx_aniso_ndf).
    GgxAnisoNdf {
        /// Half-vector normal cosine.
        n_dot_h: f32,
        /// Half-vector tangent cosine.
        t_dot_h: f32,
        /// Half-vector bitangent cosine.
        b_dot_h: f32,
        /// Tangent alpha.
        alpha_t: f32,
        /// Bitangent alpha.
        alpha_b: f32,
    },
    /// The height-correlated anisotropic `Smith` visibility, matching
    /// [`ggx_aniso_visibility`](prism_render_architecture::particle::anisotropic_clearcoat::ggx_aniso_visibility).
    GgxAnisoVisibility {
        /// Tangent alpha.
        alpha_t: f32,
        /// Bitangent alpha.
        alpha_b: f32,
        /// View tangent projection.
        t_dot_v: f32,
        /// View bitangent projection.
        b_dot_v: f32,
        /// View normal cosine.
        n_dot_v: f32,
        /// Light tangent projection.
        t_dot_l: f32,
        /// Light bitangent projection.
        b_dot_l: f32,
        /// Light normal cosine.
        n_dot_l: f32,
    },
    /// The isotropic clearcoat `GGX` `NDF`, matching
    /// [`clearcoat_ggx_ndf`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_ggx_ndf).
    ClearcoatGgxNdf {
        /// Half-vector normal cosine.
        n_dot_h: f32,
        /// Clearcoat linear roughness.
        alpha: f32,
    },
    /// The isotropic clearcoat `Smith` visibility, matching
    /// [`clearcoat_visibility`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_visibility).
    ClearcoatVisibility {
        /// View normal cosine.
        n_dot_v: f32,
        /// Light normal cosine.
        n_dot_l: f32,
        /// Clearcoat linear roughness.
        alpha: f32,
    },
    /// The clearcoat `Schlick` `Fresnel`, matching
    /// [`clearcoat_fresnel`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_fresnel).
    ClearcoatFresnel {
        /// Cosine between the half vector and the view (or normal and view).
        cos_theta: f32,
    },
    /// The clearcoat-to-base attenuation `(1 - Fc)`, matching
    /// [`clearcoat_attenuation`](prism_render_architecture::particle::anisotropic_clearcoat::clearcoat_attenuation).
    ClearcoatAttenuation {
        /// Reflected clearcoat fraction `Fc`.
        fc: f32,
    },
    /// The composite base + clearcoat response, matching
    /// [`AnisotropicClearcoatParams::evaluate`](prism_render_architecture::particle::anisotropic_clearcoat::AnisotropicClearcoatParams::evaluate).
    Evaluate {
        /// Base-layer perceptual roughness.
        roughness: f32,
        /// Anisotropy control in `-1..=1`.
        anisotropy: f32,
        /// Clearcoat perceptual roughness.
        clearcoat_roughness: f32,
        /// Clearcoat presence in `0..=1`.
        clearcoat_strength: f32,
        /// Shading-frame tangent `T`.
        tangent: [f32; 3],
        /// Shading-frame bitangent `B`.
        bitangent: [f32; 3],
        /// Shading-frame normal `N`.
        normal: [f32; 3],
        /// View direction `V`.
        view: [f32; 3],
        /// Light direction `L`.
        light: [f32; 3],
    },
}

impl AnisotropicClearcoatQuery {
    /// Returns the `u32` op code the kernel branches on for this variant.
    #[must_use]
    const fn code(&self) -> u32 {
        match self {
            AnisotropicClearcoatQuery::AspectRatio { .. } => 0,
            AnisotropicClearcoatQuery::AnisotropicAlphas { .. } => 1,
            AnisotropicClearcoatQuery::GgxAnisoNdf { .. } => 2,
            AnisotropicClearcoatQuery::GgxAnisoVisibility { .. } => 3,
            AnisotropicClearcoatQuery::ClearcoatGgxNdf { .. } => 4,
            AnisotropicClearcoatQuery::ClearcoatVisibility { .. } => 5,
            AnisotropicClearcoatQuery::ClearcoatFresnel { .. } => 6,
            AnisotropicClearcoatQuery::ClearcoatAttenuation { .. } => 7,
            AnisotropicClearcoatQuery::Evaluate { .. } => 8,
        }
    }
}

/// One resolved answer, tagged to match the query variant that produced it.
///
/// Each variant carries exactly the term(s) the matching
/// [`AnisotropicClearcoatQuery`] variant selects.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::anisotropic_clearcoat`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnisotropicClearcoatResult {
    /// The anisotropy-to-aspect ratio.
    AspectRatio {
        /// `aspect = sqrt(1 - 0.9 * |anisotropy|)`.
        aspect: f32,
    },
    /// The tangent/bitangent `GGX` alphas.
    AnisotropicAlphas {
        /// Tangent alpha.
        alpha_t: f32,
        /// Bitangent alpha.
        alpha_b: f32,
    },
    /// The anisotropic `GGX` `NDF`.
    GgxAnisoNdf {
        /// Distribution value.
        ndf: f32,
    },
    /// The anisotropic `Smith` visibility.
    GgxAnisoVisibility {
        /// Visibility value.
        visibility: f32,
    },
    /// The isotropic clearcoat `GGX` `NDF`.
    ClearcoatGgxNdf {
        /// Distribution value.
        ndf: f32,
    },
    /// The isotropic clearcoat `Smith` visibility.
    ClearcoatVisibility {
        /// Visibility value.
        visibility: f32,
    },
    /// The clearcoat `Schlick` `Fresnel`.
    ClearcoatFresnel {
        /// `Fresnel` reflectance.
        fresnel: f32,
    },
    /// The clearcoat-to-base attenuation `(1 - Fc)`.
    ClearcoatAttenuation {
        /// Transmitted energy fraction.
        attenuation: f32,
    },
    /// The composite base + clearcoat response.
    Evaluate {
        /// Anisotropic `GGX` `NDF` of the base layer.
        base_ndf: f32,
        /// Anisotropic `Smith` visibility of the base layer.
        base_visibility: f32,
        /// Fraction of energy the clearcoat passes to the base layer.
        base_attenuation: f32,
        /// Isotropic `GGX` `NDF` of the clearcoat lobe.
        clearcoat_ndf: f32,
        /// Isotropic `Smith` visibility of the clearcoat lobe.
        clearcoat_visibility: f32,
        /// Clearcoat `Schlick` `Fresnel` `Fc`.
        clearcoat_fresnel: f32,
    },
}

/// Encodes one [`AnisotropicClearcoatQuery`] into its `std430` [`GpuQuery`]
/// slot, zeroing the lanes the chosen op does not read.
fn encode_query(query: &AnisotropicClearcoatQuery) -> GpuQuery {
    let mut gpu = GpuQuery {
        tangent: [0.0; 3],
        pad_t: 0.0,
        bitangent: [0.0; 3],
        pad_b: 0.0,
        normal: [0.0; 3],
        pad_n: 0.0,
        view: [0.0; 3],
        pad_v: 0.0,
        light: [0.0; 3],
        pad_l: 0.0,
        n_dot_h: 0.0,
        t_dot_h: 0.0,
        b_dot_h: 0.0,
        n_dot_v: 0.0,
        n_dot_l: 0.0,
        t_dot_v: 0.0,
        b_dot_v: 0.0,
        t_dot_l: 0.0,
        b_dot_l: 0.0,
        roughness: 0.0,
        anisotropy: 0.0,
        clearcoat_roughness: 0.0,
        clearcoat_strength: 0.0,
        alpha_t: 0.0,
        alpha_b: 0.0,
        alpha: 0.0,
        cos_theta: 0.0,
        fc: 0.0,
        op: query.code(),
        pad0: 0.0,
    };
    match *query {
        AnisotropicClearcoatQuery::AspectRatio { anisotropy } => {
            gpu.anisotropy = anisotropy;
        }
        AnisotropicClearcoatQuery::AnisotropicAlphas {
            roughness,
            anisotropy,
        } => {
            gpu.roughness = roughness;
            gpu.anisotropy = anisotropy;
        }
        AnisotropicClearcoatQuery::GgxAnisoNdf {
            n_dot_h,
            t_dot_h,
            b_dot_h,
            alpha_t,
            alpha_b,
        } => {
            gpu.n_dot_h = n_dot_h;
            gpu.t_dot_h = t_dot_h;
            gpu.b_dot_h = b_dot_h;
            gpu.alpha_t = alpha_t;
            gpu.alpha_b = alpha_b;
        }
        AnisotropicClearcoatQuery::GgxAnisoVisibility {
            alpha_t,
            alpha_b,
            t_dot_v,
            b_dot_v,
            n_dot_v,
            t_dot_l,
            b_dot_l,
            n_dot_l,
        } => {
            gpu.alpha_t = alpha_t;
            gpu.alpha_b = alpha_b;
            gpu.t_dot_v = t_dot_v;
            gpu.b_dot_v = b_dot_v;
            gpu.n_dot_v = n_dot_v;
            gpu.t_dot_l = t_dot_l;
            gpu.b_dot_l = b_dot_l;
            gpu.n_dot_l = n_dot_l;
        }
        AnisotropicClearcoatQuery::ClearcoatGgxNdf { n_dot_h, alpha } => {
            gpu.n_dot_h = n_dot_h;
            gpu.alpha = alpha;
        }
        AnisotropicClearcoatQuery::ClearcoatVisibility {
            n_dot_v,
            n_dot_l,
            alpha,
        } => {
            gpu.n_dot_v = n_dot_v;
            gpu.n_dot_l = n_dot_l;
            gpu.alpha = alpha;
        }
        AnisotropicClearcoatQuery::ClearcoatFresnel { cos_theta } => {
            gpu.cos_theta = cos_theta;
        }
        AnisotropicClearcoatQuery::ClearcoatAttenuation { fc } => {
            gpu.fc = fc;
        }
        AnisotropicClearcoatQuery::Evaluate {
            roughness,
            anisotropy,
            clearcoat_roughness,
            clearcoat_strength,
            tangent,
            bitangent,
            normal,
            view,
            light,
        } => {
            gpu.roughness = roughness;
            gpu.anisotropy = anisotropy;
            gpu.clearcoat_roughness = clearcoat_roughness;
            gpu.clearcoat_strength = clearcoat_strength;
            gpu.tangent = tangent;
            gpu.bitangent = bitangent;
            gpu.normal = normal;
            gpu.view = view;
            gpu.light = light;
        }
    }
    gpu
}

/// Decodes one packed [`GpuResult`] into the public
/// [`AnisotropicClearcoatResult`], selecting the fields the originating `query`
/// variant produced.
fn decode_result(query: &AnisotropicClearcoatQuery, raw: &GpuResult) -> AnisotropicClearcoatResult {
    match query {
        AnisotropicClearcoatQuery::AspectRatio { .. } => {
            AnisotropicClearcoatResult::AspectRatio { aspect: raw.a }
        }
        AnisotropicClearcoatQuery::AnisotropicAlphas { .. } => {
            AnisotropicClearcoatResult::AnisotropicAlphas {
                alpha_t: raw.a,
                alpha_b: raw.b,
            }
        }
        AnisotropicClearcoatQuery::GgxAnisoNdf { .. } => {
            AnisotropicClearcoatResult::GgxAnisoNdf { ndf: raw.a }
        }
        AnisotropicClearcoatQuery::GgxAnisoVisibility { .. } => {
            AnisotropicClearcoatResult::GgxAnisoVisibility { visibility: raw.a }
        }
        AnisotropicClearcoatQuery::ClearcoatGgxNdf { .. } => {
            AnisotropicClearcoatResult::ClearcoatGgxNdf { ndf: raw.a }
        }
        AnisotropicClearcoatQuery::ClearcoatVisibility { .. } => {
            AnisotropicClearcoatResult::ClearcoatVisibility { visibility: raw.a }
        }
        AnisotropicClearcoatQuery::ClearcoatFresnel { .. } => {
            AnisotropicClearcoatResult::ClearcoatFresnel { fresnel: raw.a }
        }
        AnisotropicClearcoatQuery::ClearcoatAttenuation { .. } => {
            AnisotropicClearcoatResult::ClearcoatAttenuation { attenuation: raw.a }
        }
        AnisotropicClearcoatQuery::Evaluate { .. } => AnisotropicClearcoatResult::Evaluate {
            base_ndf: raw.a,
            base_visibility: raw.b,
            base_attenuation: raw.c,
            clearcoat_ndf: raw.d,
            clearcoat_visibility: raw.e,
            clearcoat_fresnel: raw.f,
        },
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

/// A compiled, reusable anisotropic-clearcoat compute pipeline, twinning the
/// `CPU` golden
/// [`anisotropic_clearcoat`](prism_render_architecture::particle::anisotropic_clearcoat).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::anisotropic_clearcoat`；无第三方引擎源码或衍生代码。
pub struct GpuAnisotropicClearcoat {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAnisotropicClearcoat {
    /// Compiles the anisotropic-clearcoat kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::anisotropic_clearcoat`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAnisotropicClearcoat {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat"),
            source: ShaderSource::Wgsl(ANISOTROPIC_CLEARCOAT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAnisotropicClearcoat {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`AnisotropicClearcoatResult`] per input, in order.
    ///
    /// Each result matches the `CPU` golden within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::anisotropic_clearcoat`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[AnisotropicClearcoatQuery],
    ) -> Vec<AnisotropicClearcoatResult> {
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
            label: Some("prism_volumetric_anisotropic_clearcoat_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat_bind_group"),
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
            label: Some("prism_volumetric_anisotropic_clearcoat_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_anisotropic_clearcoat_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_anisotropic_clearcoat_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
