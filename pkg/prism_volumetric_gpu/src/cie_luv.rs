//! `wgpu` compute twin of the `CIE` 1976 `L*u*v*` (`CIELUV`) contract
//! ([`cie_luv`](prism_render_architecture::particle::cie_luv), particle design
//! §29).
//!
//! The `CPU` golden
//! [`cie_luv`](prism_render_architecture::particle::cie_luv) owns the
//! perceptually-uniform additive-light color space the particle tint pipeline
//! shares: the `CIE` lightness transfer and its inverse
//! ([`luv_f`](prism_render_architecture::particle::cie_luv::luv_f),
//! [`luv_f_inv`](prism_render_architecture::particle::cie_luv::luv_f_inv)), the
//! affine chromaticity projection
//! ([`uv_prime`](prism_render_architecture::particle::cie_luv::uv_prime)), the
//! `XYZ` <-> `CIELUV` conversions
//! ([`xyz_to_luv`](prism_render_architecture::particle::cie_luv::xyz_to_luv),
//! [`luv_to_xyz`](prism_render_architecture::particle::cie_luv::luv_to_xyz)),
//! the chroma and saturation descriptors
//! ([`chroma`](prism_render_architecture::particle::cie_luv::chroma),
//! [`saturation`](prism_render_architecture::particle::cie_luv::saturation)),
//! the `ΔE*uv` color difference
//! ([`delta_e_uv`](prism_render_architecture::particle::cie_luv::delta_e_uv)),
//! and the component-wise blend
//! ([`luv_lerp`](prism_render_architecture::particle::cie_luv::luv_lerp)).
//! [`GpuCieLuv`] is the on-device twin: one thread per query reproduces every
//! transform branch for branch, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same algebra and classifies the
//! same degenerate cases the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries packed one per thread: the forward and inverse lightness
//! nonlinearity (including the self-contained
//! [`cbrt_newton`](prism_render_architecture::particle::cie_luv::cbrt_newton)
//! cube root they share), the `u'v'` chromaticity projection, the `XYZ` ->
//! `CIELUV` and `CIELUV` -> `XYZ` conversions, the chroma `C*uv`, the saturation
//! `s_uv`, the color difference `ΔE*uv`, and the component-wise `luv_lerp`. The
//! reference-white chromaticity
//! ([`reference_white_uv`](prism_render_architecture::particle::cie_luv::reference_white_uv))
//! and the `D65` white
//! ([`d65_white`](prism_render_architecture::particle::cie_luv::d65_white)) are
//! not per-query functions: they return fixed constants, so the kernel folds
//! those same values in as compile-time `const` inputs rather than twinning them
//! as dispatched outputs. The host-only `std430` packing helpers the reference
//! exposes (`to_std430`, `gpu_storage_bytes`) are not kernel math and are left
//! to the `CPU` reference; only the arithmetic transforms are twinned
//! on-device.
//!
//! # Correctness model
//!
//! Each transform threads through a fixed, non-reorderable sequence of
//! multiplies, adds, divides, one `sqrt` and the bounded Newton cube-root loop
//! (no transcendental call), so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every
//! continuous quantity, tight enough to catch a genuinely wrong port (a dropped
//! term, a swapped white-point offset, a wrong knot) yet loose enough to admit
//! legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The reference guards three divisions, and the kernel mirrors each with an
//! `abs(denom) < CMP_EPS` test rather than an `f32` `==`:
//! [`uv_prime`](prism_render_architecture::particle::cie_luv::uv_prime) collapses
//! both chromaticity coordinates to zero when the shared denominator
//! `X + 15Y + 3Z` is within `CMP_EPS` of zero;
//! [`luv_to_xyz`](prism_render_architecture::particle::cie_luv::luv_to_xyz)
//! short-circuits to pure black when `L*` is within `CMP_EPS` of zero and to
//! the neutral gray of that luminance when the reconstructed `v'` is within
//! `CMP_EPS` of zero; and
//! [`saturation`](prism_render_architecture::particle::cie_luv::saturation)
//! reports zero when `L*` is within `CMP_EPS` of zero. The Newton cube root
//! likewise breaks its iteration if the `3y²` denominator falls below
//! `CMP_EPS`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `cbrt`, `tan`, no inverse trigonometry and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The only loop
//! is the fixed `CBRT_ITERATIONS`-step Newton refinement with range-reduction
//! loops bounded by factors of eight, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
//! standard `CIE` 1976 `L*u*v*` color algebra plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::cie_luv::{
    cbrt_newton, chroma, delta_e_uv, luv_f, luv_f_inv, luv_lerp, luv_to_xyz, saturation, uv_prime,
    xyz_to_luv, Luv, UvPrime, Xyz,
};
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

/// The portable core-`WGSL` `CIELUV` kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden [`cie_luv`](prism_render_architecture::particle::cie_luv) branch for
/// branch; see the module documentation for the algorithm.
const CIE_LUV_WGSL: &str = r#"
// CIE 1976 L*u*v* twin: one thread per query reproduces the lightness transfer
// and its inverse (with the shared Newton cube root), the u'v' chromaticity
// projection, the XYZ <-> CIELUV conversions with their divide-by-zero guards,
// the chroma, saturation, color difference and the component-wise lerp. It
// mirrors the CPU golden particle::cie_luv branch for branch, uses only the
// portable core-WGSL subset (abs/sqrt and + - * / plus unsigned index math),
// has no transcendental call and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. The only loop is the fixed Newton
// refinement, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::cie_luv; no
// third-party engine source or derived code.

// Magnitude below which a denominator or the Newton 3y^2 slope is treated as
// zero. Matches the reference `CMP_EPS`; the compare rule used instead of an
// f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// The shared CIE knot delta = 6/29 and its derived thresholds, written as
// divisions of named integers so they are never mistaken for approximations of
// a mathematical constant. Matches the reference DELTA / DELTA_CUBED /
// THREE_DELTA_SQ / TOE_OFFSET.
const DELTA: f32 = 6.0 / 29.0;
const DELTA_CUBED: f32 = DELTA * DELTA * DELTA;
const THREE_DELTA_SQ: f32 = 3.0 * DELTA * DELTA;
const TOE_OFFSET: f32 = 4.0 / 29.0;

// Newton range-reduction window [1, 8) and the fixed refinement budget, matching
// the reference CBRT_WINDOW_LO / CBRT_WINDOW_HI / CBRT_ITERATIONS.
const CBRT_WINDOW_LO: f32 = 1.0;
const CBRT_WINDOW_HI: f32 = 8.0;
const CBRT_ITERATIONS: i32 = 16;

// CIE D65 reference white tristimulus (normalized to Y = 1), matching the
// reference XN / YN / ZN.
const XN: f32 = 0.950489;
const YN: f32 = 1.0;
const ZN: f32 = 0.888840;

// Reference-white chromaticity origin (u'n, v'n), folded in as a const input
// rather than twinned as a dispatched output. Matches the reference UN_PRIME /
// VN_PRIME derived from the D65 denominator X + 15Y + 3Z.
const D65_DENOM: f32 = XN + 15.0 * YN + 3.0 * ZN;
const UN_PRIME: f32 = 4.0 * XN / D65_DENOM;
const VN_PRIME: f32 = 9.0 * YN / D65_DENOM;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // XYZ input for uv_prime and xyz_to_luv.
    xyz: vec3<f32>,
    pad0: f32,
    // First CIELUV color: input for luv_to_xyz, chroma, saturation and the lerp
    // start, plus the first argument of delta_e_uv.
    luv_a: vec3<f32>,
    pad1: f32,
    // Second CIELUV color: the lerp end and the second argument of delta_e_uv.
    luv_b: vec3<f32>,
    pad2: f32,
    // Scalar inputs: luv_f argument, luv_f_inv argument, cbrt_newton argument and
    // the lerp parameter t.
    scalars: vec4<f32>,
}

struct Result {
    // xyz_to_luv image of the XYZ input.
    to_luv: vec3<f32>,
    pad0: f32,
    // luv_to_xyz image of luv_a.
    to_xyz: vec3<f32>,
    pad1: f32,
    // luv_lerp(luv_a, luv_b, t).
    lerp: vec3<f32>,
    pad2: f32,
    // (luv_f, luv_f_inv, cbrt_newton, chroma).
    scal0: vec4<f32>,
    // (saturation, delta_e_uv, u', v').
    scal1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Real cube root via a self-contained Newton iteration, mirroring the reference
// `cbrt_newton`. WGSL has no recursion, so the sign reflection cbrt(-x) =
// -cbrt(x) is inlined: the magnitude is range-reduced into [1, 8) by factors of
// eight and refined by y <- y - (y^3 - v) / (3y^2), then the sign is reapplied.
fn cbrt_newton(x: f32) -> f32 {
    let is_negative = x < 0.0;
    var v = abs(x);
    if (v <= 0.0) {
        return 0.0;
    }
    var scale = 1.0;
    loop {
        if (v <= CBRT_WINDOW_HI) {
            break;
        }
        v = v / 8.0;
        scale = scale * 2.0;
    }
    loop {
        if (v >= CBRT_WINDOW_LO) {
            break;
        }
        v = v * 8.0;
        scale = scale / 2.0;
    }
    var y = 1.5;
    for (var i: i32 = 0; i < CBRT_ITERATIONS; i = i + 1) {
        let denom = 3.0 * y * y;
        if (denom < CMP_EPS) {
            break;
        }
        y = y - (y * y * y - v) / denom;
    }
    let magnitude = y * scale;
    if (is_negative) {
        return -magnitude;
    }
    return magnitude;
}

// CIE lightness forward nonlinearity f(t): the cube root above the knot delta^3,
// the linear toe t / (3 delta^2) + 4/29 at or below it. Mirrors `luv_f`.
fn luv_f(t: f32) -> f32 {
    if (t > DELTA_CUBED) {
        return cbrt_newton(t);
    }
    return t / THREE_DELTA_SQ + TOE_OFFSET;
}

// Exact inverse of luv_f: cubes its argument above the knot delta, the inverse
// linear toe 3 delta^2 (t - 4/29) at or below it. Mirrors `luv_f_inv`.
fn luv_f_inv(t: f32) -> f32 {
    if (t > DELTA) {
        return t * t * t;
    }
    return THREE_DELTA_SQ * (t - TOE_OFFSET);
}

// CIELUV chromaticity (u', v') of an XYZ triple, mirroring `uv_prime`. The
// shared denominator X + 15Y + 3Z is guarded: a within-CMP_EPS magnitude
// collapses both coordinates to zero rather than dividing by ~zero.
fn uv_prime(c: vec3<f32>) -> vec2<f32> {
    let denom = c.x + 15.0 * c.y + 3.0 * c.z;
    if (abs(denom) < CMP_EPS) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(4.0 * c.x / denom, 9.0 * c.y / denom);
}

// Converts an XYZ triple (relative to the D65 white) to CIELUV, mirroring
// `xyz_to_luv`: L* = 116 f(Y/Yn) - 16, then u* = 13 L* (u' - u'n) and
// v* = 13 L* (v' - v'n).
fn xyz_to_luv(c: vec3<f32>) -> vec3<f32> {
    let l = 116.0 * luv_f(c.y / YN) - 16.0;
    let uv = uv_prime(c);
    let scale = 13.0 * l;
    let u = scale * (uv.x - UN_PRIME);
    let v = scale * (uv.y - VN_PRIME);
    return vec3<f32>(l, u, v);
}

// Inverts xyz_to_luv, mirroring `luv_to_xyz`. Y is recovered first; a within-
// CMP_EPS L* short-circuits to pure black and a within-CMP_EPS reconstructed v'
// yields the neutral gray of that luminance, guarding both divisions.
fn luv_to_xyz(c: vec3<f32>) -> vec3<f32> {
    let fy = (c.x + 16.0) / 116.0;
    let y = YN * luv_f_inv(fy);
    if (abs(c.x) < CMP_EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv = 1.0 / (13.0 * c.x);
    let u_prime = c.y * inv + UN_PRIME;
    let v_prime = c.z * inv + VN_PRIME;
    if (abs(v_prime) < CMP_EPS) {
        return vec3<f32>(0.0, y, 0.0);
    }
    let x = y * (9.0 * u_prime) / (4.0 * v_prime);
    let z = y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime);
    return vec3<f32>(x, y, z);
}

// CIELUV chroma C*uv = sqrt(u*^2 + v*^2), mirroring `chroma`.
fn chroma(c: vec3<f32>) -> f32 {
    return sqrt(c.y * c.y + c.z * c.z);
}

// CIELUV saturation s_uv = C*uv / L*, mirroring `saturation`: a within-CMP_EPS
// L* (black) reports zero rather than dividing by ~zero.
fn saturation(c: vec3<f32>) -> f32 {
    if (abs(c.x) < CMP_EPS) {
        return 0.0;
    }
    return chroma(c) / c.x;
}

// CIELUV color difference, the Euclidean distance between two colors, mirroring
// `delta_e_uv`.
fn delta_e_uv(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

// Component-wise CIELUV interpolation, mirroring `luv_lerp`: t = 0 returns a,
// t = 1 returns b.
fn luv_lerp(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let f_t = q.scalars.x;
    let f_inv_t = q.scalars.y;
    let cbrt_x = q.scalars.z;
    let t = q.scalars.w;

    let uv = uv_prime(q.xyz);

    var out: Result;
    out.to_luv = xyz_to_luv(q.xyz);
    out.pad0 = 0.0;
    out.to_xyz = luv_to_xyz(q.luv_a);
    out.pad1 = 0.0;
    out.lerp = luv_lerp(q.luv_a, q.luv_b, t);
    out.pad2 = 0.0;
    out.scal0 = vec4<f32>(luv_f(f_t), luv_f_inv(f_inv_t), cbrt_newton(cbrt_x), chroma(q.luv_a));
    out.scal1 = vec4<f32>(saturation(q.luv_a), delta_e_uv(q.luv_a, q.luv_b), uv.x, uv.y);
    results[idx] = out;
}
"#;

/// One `CIELUV` query bundling every input the reference transforms consume for
/// a single element: the `XYZ` triple, two `CIELUV` colors, and the three scalar
/// arguments plus the lerp parameter.
///
/// The transforms are independent, so a single query exercises every twinned
/// function at once.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CieLuvQuery {
    /// `XYZ` input for `uv_prime` and `xyz_to_luv`.
    pub xyz: Xyz,
    /// First `CIELUV` color: input for `luv_to_xyz`, `chroma`, `saturation`, the
    /// `luv_lerp` start and the first argument of `delta_e_uv`.
    pub luv_a: Luv,
    /// Second `CIELUV` color: the `luv_lerp` end and the second argument of
    /// `delta_e_uv`.
    pub luv_b: Luv,
    /// Argument fed to `luv_f`.
    pub f_t: f32,
    /// Argument fed to `luv_f_inv`.
    pub f_inv_t: f32,
    /// Argument fed to `cbrt_newton`.
    pub cbrt_x: f32,
    /// Interpolation parameter fed to `luv_lerp`.
    pub lerp_t: f32,
}

impl CieLuvQuery {
    /// Builds a query from the `XYZ` triple, the two `CIELUV` colors and the four
    /// scalar arguments.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        xyz: Xyz,
        luv_a: Luv,
        luv_b: Luv,
        f_t: f32,
        f_inv_t: f32,
        cbrt_x: f32,
        lerp_t: f32,
    ) -> CieLuvQuery {
        CieLuvQuery {
            xyz,
            luv_a,
            luv_b,
            f_t,
            f_inv_t,
            cbrt_x,
            lerp_t,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its twinned functions.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CieLuvResult {
    /// Forward lightness nonlinearity, matching
    /// [`luv_f`](prism_render_architecture::particle::cie_luv::luv_f).
    pub luv_f: f32,
    /// Inverse lightness nonlinearity, matching
    /// [`luv_f_inv`](prism_render_architecture::particle::cie_luv::luv_f_inv).
    pub luv_f_inv: f32,
    /// Newton cube root, matching
    /// [`cbrt_newton`](prism_render_architecture::particle::cie_luv::cbrt_newton).
    pub cbrt: f32,
    /// Chromaticity projection of the `XYZ` input, matching
    /// [`uv_prime`](prism_render_architecture::particle::cie_luv::uv_prime).
    pub uv_prime: UvPrime,
    /// `CIELUV` image of the `XYZ` input, matching
    /// [`xyz_to_luv`](prism_render_architecture::particle::cie_luv::xyz_to_luv).
    pub xyz_to_luv: Luv,
    /// `XYZ` image of `luv_a`, matching
    /// [`luv_to_xyz`](prism_render_architecture::particle::cie_luv::luv_to_xyz).
    pub luv_to_xyz: Xyz,
    /// Chroma of `luv_a`, matching
    /// [`chroma`](prism_render_architecture::particle::cie_luv::chroma).
    pub chroma: f32,
    /// Saturation of `luv_a`, matching
    /// [`saturation`](prism_render_architecture::particle::cie_luv::saturation).
    pub saturation: f32,
    /// Color difference between `luv_a` and `luv_b`, matching
    /// [`delta_e_uv`](prism_render_architecture::particle::cie_luv::delta_e_uv).
    pub delta_e_uv: f32,
    /// Interpolation of `luv_a` toward `luv_b` by `lerp_t`, matching
    /// [`luv_lerp`](prism_render_architecture::particle::cie_luv::luv_lerp).
    pub luv_lerp: Luv,
}

/// Evaluates the `CPU` golden for one query, producing every transform the
/// on-device twin reproduces.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &CieLuvQuery) -> CieLuvResult {
    CieLuvResult {
        luv_f: luv_f(query.f_t),
        luv_f_inv: luv_f_inv(query.f_inv_t),
        cbrt: cbrt_newton(query.cbrt_x),
        uv_prime: uv_prime(&query.xyz),
        xyz_to_luv: xyz_to_luv(&query.xyz),
        luv_to_xyz: luv_to_xyz(&query.luv_a),
        chroma: chroma(&query.luv_a),
        saturation: saturation(&query.luv_a),
        delta_e_uv: delta_e_uv(&query.luv_a, &query.luv_b),
        luv_lerp: luv_lerp(&query.luv_a, &query.luv_b, query.lerp_t),
    }
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots each
/// carrying a `vec3` triple on its `16`-byte-aligned slot with a padding lane,
/// plus one `vec4` of scalars, exactly as the `WGSL` `Query` struct reads it —
/// `64` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `XYZ` input.
    xyz: [f32; 3],
    /// Padding lane after the `XYZ` input.
    pad0: f32,
    /// First `CIELUV` color.
    luv_a: [f32; 3],
    /// Padding lane after the first color.
    pad1: f32,
    /// Second `CIELUV` color.
    luv_b: [f32; 3],
    /// Padding lane after the second color.
    pad2: f32,
    /// Scalar inputs: `luv_f` argument, `luv_f_inv` argument, `cbrt_newton`
    /// argument and the lerp parameter.
    scalars: [f32; 4],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &CieLuvQuery) -> GpuQuery {
        GpuQuery {
            xyz: [query.xyz.x, query.xyz.y, query.xyz.z],
            pad0: 0.0,
            luv_a: [query.luv_a.l, query.luv_a.u, query.luv_a.v],
            pad1: 0.0,
            luv_b: [query.luv_b.l, query.luv_b.u, query.luv_b.v],
            pad2: 0.0,
            scalars: [query.f_t, query.f_inv_t, query.cbrt_x, query.lerp_t],
        }
    }
}

/// `repr(C)` `std430` layout of one result: three `vec4` slots each a `vec3`
/// triple with a padding lane, plus two `vec4` scalar slots matching the `WGSL`
/// `Result` struct — `80` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `xyz_to_luv` image of the `XYZ` input.
    to_luv: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// `luv_to_xyz` image of the first color.
    to_xyz: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// `luv_lerp` of the two colors.
    lerp: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// `(luv_f, luv_f_inv, cbrt_newton, chroma)`.
    scal0: [f32; 4],
    /// `(saturation, delta_e_uv, u', v')`.
    scal1: [f32; 4],
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
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

/// A compiled, reusable `CIELUV` compute pipeline, twinning the `CPU` golden
/// [`cie_luv`](prism_render_architecture::particle::cie_luv).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
/// no third-party engine source or derived code.
pub struct GpuCieLuv {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCieLuv {
    /// Compiles the `CIELUV` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCieLuv {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cie_luv"),
            source: ShaderSource::Wgsl(CIE_LUV_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cie_luv_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cie_luv_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cie_luv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCieLuv {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`CieLuvResult`] per input,
    /// in order.
    ///
    /// Each result equals the reference transforms to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::cie_luv`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[CieLuvQuery]) -> Vec<CieLuvResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cie_luv_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cie_luv_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cie_luv_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cie_luv_bind_group"),
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
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cie_luv_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cie_luv_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cie_luv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`CieLuvResult`].
fn decode_result(raw: &GpuResult) -> CieLuvResult {
    CieLuvResult {
        luv_f: raw.scal0[0],
        luv_f_inv: raw.scal0[1],
        cbrt: raw.scal0[2],
        uv_prime: UvPrime::new(raw.scal1[2], raw.scal1[3]),
        xyz_to_luv: Luv::new(raw.to_luv[0], raw.to_luv[1], raw.to_luv[2]),
        luv_to_xyz: Xyz::new(raw.to_xyz[0], raw.to_xyz[1], raw.to_xyz[2]),
        chroma: raw.scal0[3],
        saturation: raw.scal1[0],
        delta_e_uv: raw.scal1[1],
        luv_lerp: Luv::new(raw.lerp[0], raw.lerp[1], raw.lerp[2]),
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
