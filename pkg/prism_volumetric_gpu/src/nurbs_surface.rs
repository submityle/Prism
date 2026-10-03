//! `wgpu` compute twin of the uniform **rational bicubic B-spline (`NURBS`)
//! surface** evaluator from the reference ray-scene primitive
//! (`prism_render_architecture::ray_scene::nurbs_surface`).
//!
//! The `CPU` golden `NurbsSurface` partitions an `R × C` control grid with
//! matching positive weights into `(C - 3) × (R - 3)` overlapping cubic spans.
//! A global parameter `(u, v) ∈ [0, 1]²` is scaled by the span count, the
//! integer part selects the span and the fractional part becomes the local
//! span parameter (`locate`). Each span is converted B-spline → Bézier in
//! homogeneous `[w·x, w·y, w·z, w]` space along `u` then `v` (`patch_at`), the
//! homogeneous net is projected back by one division per control point, and the
//! resulting `prism_render_architecture::ray_scene::rational_bezier_patch`
//! evaluates the position by a rational bicubic De Casteljau and the normal by
//! the normalized cross product of the quotient-rule partial-derivative
//! numerators, with a short `eps` step-search that nudges off a collapsed
//! tangent frame and falls back to `[0, 0, 1]`.
//!
//! [`GpuNurbsSurface`] is the on-device twin. One dispatch shares a single
//! row-major control-and-weight grid across the whole query batch; each
//! [`NurbsSurfaceQuery`] carries only the global `(u, v)`. The kernel runs the
//! full surface-level `locate` → `patch_at` B-spline-to-Bézier conversion →
//! rational bicubic evaluation on device, so a passing real-device parity test
//! is direct evidence the ported kernel runs the same recursion and classifies
//! the same degenerate normal, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread resolves one query against the shared grid. The kernel locates
//! the span along `u` and `v`, promotes the overlapping `4 × 4` window to
//! homogeneous coordinates, converts it B-spline → Bézier along each axis,
//! projects it back, and reproduces `NurbsSurface::point` and
//! `NurbsSurface::normal`, writing one [`NurbsSurfaceResult`] holding the
//! surface point, the unit normal and a `degenerate` flag that is set when the
//! `eps` step-search exhausts all four steps with a collapsed tangent frame and
//! the normal falls back to `[0, 0, 1]`.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: the whole evaluation is a
//! fixed, bounded sequence of `lerp`s, products, adds, comparisons, guarded
//! divisions and a single `sqrt` that runs entirely on device. The host only
//! uploads the shared grid plus the query batch into `std430` storage buffers
//! and short-circuits an empty batch, since a storage buffer cannot be
//! zero-sized. Grid validation (`cols >= 4`, `rows >= 4`, strictly-positive
//! weights) stays with the golden `NurbsSurface::new` on the host.
//!
//! # Correctness model
//!
//! The surface point and the normal are *continuous* quantities threaded
//! through nested `lerp`s, a `cross`, a `sqrt` and divisions, so the `CPU` and
//! `GPU` are not bit-exact: a device may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! compares the continuous fields with an absolute-or-relative tolerance
//! (`abs <= 1e-4 || rel <= 1e-3`, relative floor `1e-6`) while pinning the
//! integer `degenerate` flag exactly. The genuine degeneracy — a collapsed
//! tangent frame whose cross-product magnitude crosses zero — is kept off its
//! threshold by the fixtures and the rejection-sampled sweep so the two sides
//! agree on the discrete classification.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `abs`, `floor`, `sqrt`, `+ - * /` and ordered comparisons — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round`, no `%` and no `u64` / `u16` / `i64` / `f64`. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::nurbs_surface`；无第三方引擎源码或衍生代码。
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

/// Number of control points in one span's `4 × 4` homogeneous Bézier window.
const WINDOW_COUNT: usize = 16;

/// Component-wise sum of two homogeneous 4-vectors, matching the golden `add4`.
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Scales a homogeneous 4-vector by `s`, matching the golden `scale4`.
fn scale4(a: [f32; 4], s: f32) -> [f32; 4] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// Component-wise difference `a − b` over a 4-vector, matching the golden
/// `sub4`.
fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

/// Component-wise linear interpolation `a + (b − a) · t` over a 4-vector,
/// matching the golden `lerp4`.
fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Cross product `a × b`, matching the golden `cross3`.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Returns the unit vector along `v`, or `fallback` when `v` is near zero
/// length, matching the golden `normalize_or3` (`len > 1e-20`).
fn normalize_or3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt();
    if len > 1e-20 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

/// Cubic De Casteljau point at `t` over four homogeneous control points,
/// matching the golden `cubic_point4`.
fn cubic_point4(p: [[f32; 4]; 4], t: f32) -> [f32; 4] {
    let a = lerp4(p[0], p[1], t);
    let b = lerp4(p[1], p[2], t);
    let c = lerp4(p[2], p[3], t);
    let d = lerp4(a, b, t);
    let e = lerp4(b, c, t);
    lerp4(d, e, t)
}

/// Cubic De Casteljau derivative at `t`: `3·` the quadratic De Casteljau over
/// adjacent homogeneous differences, matching the golden `cubic_deriv4`.
fn cubic_deriv4(p: [[f32; 4]; 4], t: f32) -> [f32; 4] {
    let d0 = sub4(p[1], p[0]);
    let d1 = sub4(p[2], p[1]);
    let d2 = sub4(p[3], p[2]);
    let a = lerp4(d0, d1, t);
    let b = lerp4(d1, d2, t);
    let q = lerp4(a, b, t);
    [q[0] * 3.0, q[1] * 3.0, q[2] * 3.0, q[3] * 3.0]
}

/// Converts one uniform cubic B-spline span of homogeneous 4-vectors into its
/// four central-segment Bézier control points, matching the golden
/// `span_to_bezier4` (`b0 = (c0 + 4·c1 + c2)/6`, …).
fn span_to_bezier4(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], c3: [f32; 4]) -> [[f32; 4]; 4] {
    let sixth = 1.0 / 6.0;
    let third = 1.0 / 3.0;
    let b0 = scale4(add4(add4(c0, scale4(c1, 4.0)), c2), sixth);
    let b1 = scale4(add4(scale4(c1, 2.0), c2), third);
    let b2 = scale4(add4(c1, scale4(c2, 2.0)), third);
    let b3 = scale4(add4(add4(c1, scale4(c2, 4.0)), c3), sixth);
    [b0, b1, b2, b3]
}

/// Locates the span index and local parameter for a global parameter `t`,
/// matching the golden `locate`: `t` is clamped to `[0, 1]`, scaled by `spans`,
/// and split into an integer span index (clamped to the last span) and a
/// fractional local parameter.
fn locate(t: f32, spans: usize) -> (usize, f32) {
    let clamped = t.clamp(0.0, 1.0);
    let scaled = clamped * spans as f32;
    let last = spans - 1;
    let index = (scaled.floor() as usize).min(last);
    let local = scaled - index as f32;
    (index, local)
}

/// Builds the projected-then-repromoted homogeneous Bézier net for the span at
/// `(span_u, span_v)`, mirroring the golden `NurbsSurface::patch_at` followed by
/// `RationalBezierPatch::homogeneous_row`.
///
/// The overlapping `4 × 4` window of `control`/`weights` (row-major with stride
/// `cols`) is promoted to weighted homogeneous coordinates, converted B-spline →
/// Bézier along `u` then `v`, projected back by one division per point, and
/// re-promoted to `[c·w, w]`, exactly as the golden does when evaluating.
fn build_hnet(
    control: &[[f32; 3]],
    weights: &[f32],
    cols: usize,
    span_u: usize,
    span_v: usize,
) -> [[f32; 4]; WINDOW_COUNT] {
    let mut hwin = [[0.0f32; 4]; WINDOW_COUNT];
    for wr in 0..4 {
        for wc in 0..4 {
            let idx = (span_v + wr) * cols + (span_u + wc);
            let p = control[idx];
            let w = weights[idx];
            hwin[wr * 4 + wc] = [p[0] * w, p[1] * w, p[2] * w, w];
        }
    }
    let mut tmp = [[0.0f32; 4]; WINDOW_COUNT];
    for row in 0..4 {
        let base = row * 4;
        let span = span_to_bezier4(hwin[base], hwin[base + 1], hwin[base + 2], hwin[base + 3]);
        tmp[base] = span[0];
        tmp[base + 1] = span[1];
        tmp[base + 2] = span[2];
        tmp[base + 3] = span[3];
    }
    let mut out = [[0.0f32; 4]; WINDOW_COUNT];
    for col in 0..4 {
        let span = span_to_bezier4(tmp[col], tmp[4 + col], tmp[8 + col], tmp[12 + col]);
        out[col] = span[0];
        out[4 + col] = span[1];
        out[8 + col] = span[2];
        out[12 + col] = span[3];
    }
    let mut hnet = [[0.0f32; 4]; WINDOW_COUNT];
    for (slot, h) in hnet.iter_mut().zip(out.iter()) {
        let w = h[3];
        let cpt = [h[0] / w, h[1] / w, h[2] / w];
        *slot = [cpt[0] * w, cpt[1] * w, cpt[2] * w, w];
    }
    hnet
}

/// Returns the four homogeneous control points of row `i` of the span net.
fn hnet_row(hnet: &[[f32; 4]; WINDOW_COUNT], i: usize) -> [[f32; 4]; 4] {
    [
        hnet[i * 4],
        hnet[i * 4 + 1],
        hnet[i * 4 + 2],
        hnet[i * 4 + 3],
    ]
}

/// Evaluates the homogeneous surface point `H(u, v)` over the span net,
/// matching the golden `RationalBezierPatch::homogeneous_point`.
fn homo_point(hnet: &[[f32; 4]; WINDOW_COUNT], u: f32, v: f32) -> [f32; 4] {
    let column = [
        cubic_point4(hnet_row(hnet, 0), u),
        cubic_point4(hnet_row(hnet, 1), u),
        cubic_point4(hnet_row(hnet, 2), u),
        cubic_point4(hnet_row(hnet, 3), u),
    ];
    cubic_point4(column, v)
}

/// Evaluates `∂H/∂u` over the span net, matching the golden
/// `RationalBezierPatch::homogeneous_partial_u`.
fn homo_pu(hnet: &[[f32; 4]; WINDOW_COUNT], u: f32, v: f32) -> [f32; 4] {
    let column = [
        cubic_deriv4(hnet_row(hnet, 0), u),
        cubic_deriv4(hnet_row(hnet, 1), u),
        cubic_deriv4(hnet_row(hnet, 2), u),
        cubic_deriv4(hnet_row(hnet, 3), u),
    ];
    cubic_point4(column, v)
}

/// Evaluates `∂H/∂v` over the span net, matching the golden
/// `RationalBezierPatch::homogeneous_partial_v`.
fn homo_pv(hnet: &[[f32; 4]; WINDOW_COUNT], u: f32, v: f32) -> [f32; 4] {
    let column = [
        cubic_point4(hnet_row(hnet, 0), u),
        cubic_point4(hnet_row(hnet, 1), u),
        cubic_point4(hnet_row(hnet, 2), u),
        cubic_point4(hnet_row(hnet, 3), u),
    ];
    cubic_deriv4(column, v)
}

/// The quotient-rule numerator of `∂P/∂u`, matching the golden
/// `RationalBezierPatch::partial_u_numerator`.
fn pu_num(hnet: &[[f32; 4]; WINDOW_COUNT], u: f32, v: f32) -> [f32; 3] {
    let h = homo_point(hnet, u, v);
    let hu = homo_pu(hnet, u, v);
    [
        hu[0] * h[3] - h[0] * hu[3],
        hu[1] * h[3] - h[1] * hu[3],
        hu[2] * h[3] - h[2] * hu[3],
    ]
}

/// The quotient-rule numerator of `∂P/∂v`, matching the golden
/// `RationalBezierPatch::partial_v_numerator`.
fn pv_num(hnet: &[[f32; 4]; WINDOW_COUNT], u: f32, v: f32) -> [f32; 3] {
    let h = homo_point(hnet, u, v);
    let hv = homo_pv(hnet, u, v);
    [
        hv[0] * h[3] - h[0] * hv[3],
        hv[1] * h[3] - h[1] * hv[3],
        hv[2] * h[3] - h[2] * hv[3],
    ]
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::nurbs_surface`'s `NurbsSurface::point` and
/// `NurbsSurface::normal`, reproduced without importing the golden so the twin
/// stays self-contained.
///
/// `control`/`weights` are the row-major `rows × cols` grid (`col` along `u`,
/// `row` along `v`) with strictly-positive weights; `u`/`v` are the global
/// surface parameters. The global `(u, v)` are located into a span and local
/// parameters, the span is converted to its homogeneous Bézier net, the point
/// is a rational bicubic De Casteljau and the normal is `normalize(∂P/∂u ×
/// ∂P/∂v)` with the golden four-step `eps` search, falling back to `[0, 0, 1]`.
/// Returns `(point, normal, degenerate)`, where `degenerate` is `true` only
/// when every step collapses and the fallback normal is used.
#[must_use]
pub fn eval_nurbs_surface(
    control: &[[f32; 3]],
    weights: &[f32],
    rows: usize,
    cols: usize,
    u: f32,
    v: f32,
) -> ([f32; 3], [f32; 3], bool) {
    const FALLBACK: [f32; 3] = [0.0, 0.0, 1.0];
    let u_span = cols - 3;
    let v_span = rows - 3;
    let (span_u, local_u) = locate(u, u_span);
    let (span_v, local_v) = locate(v, v_span);
    let hnet = build_hnet(control, weights, cols, span_u, span_v);

    let h = homo_point(&hnet, local_u, local_v);
    let point = if h[3].abs() > 1e-20 {
        [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
    } else {
        [h[0], h[1], h[2]]
    };

    let mut normal = FALLBACK;
    let mut degenerate = true;
    for step in 0..4 {
        let eps = 1e-3 * step as f32;
        let uu = (local_u + eps).clamp(0.0, 1.0);
        let vv = (local_v + eps).clamp(0.0, 1.0);
        let du = pu_num(&hnet, uu, vv);
        let dv = pv_num(&hnet, uu, vv);
        let n = cross3(du, dv);
        let len2 = dot3(n, n);
        if len2 > 0.0 {
            normal = normalize_or3(n, FALLBACK);
            degenerate = false;
            break;
        }
    }
    (point, normal, degenerate)
}

/// The portable core-`WGSL` rational bicubic B-spline surface kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the golden `NurbsSurface::point` and `NurbsSurface::normal`;
/// see the module documentation.
const NURBS_SURFACE_WGSL: &str = r#"
// Rational bicubic B-spline (NURBS) surface twin: one thread evaluates one
// query against a shared row-major control-and-weight grid. The global (u,v) is
// located into a span and local parameters, the overlapping 4x4 window is
// promoted to homogeneous coordinates, converted B-spline to Bezier along u
// then v, projected back, and evaluated as a rational bicubic De Casteljau; the
// normal is normalize(cross(pu_num, pv_num)) with the four step eps search that
// falls back to (0,0,1). Transcendental-free: only lerps, products, adds,
// comparisons, guarded divisions and one sqrt.
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::nurbs_surface；无第三方引擎源码或衍生代码。

struct Params {
    // Control-grid dimensions (rows along v, cols along u).
    rows: u32,
    cols: u32,
    // Span counts (cols-3 along u, rows-3 along v); both >= 1.
    u_span: u32,
    v_span: u32,
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Global surface parameters in [0, 1].
    u: f32,
    v: f32,
}

struct SurfResult {
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    degenerate: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Row-major control grid as weighted-homogeneous input vec4(x, y, z, weight).
@group(0) @binding(1) var<storage, read> control_net: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<SurfResult>;

// Per-invocation homogeneous Bezier net of the located span.
var<private> hnet: array<vec4<f32>, 16>;

fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

fn normalize_or3(v: vec3<f32>, fb: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot3(v, v));
    if (len > 1e-20) {
        return v / len;
    }
    return fb;
}

fn lerp4(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    return a + (b - a) * t;
}

fn cubic_point4(p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, t: f32) -> vec4<f32> {
    let a = lerp4(p0, p1, t);
    let b = lerp4(p1, p2, t);
    let c = lerp4(p2, p3, t);
    let d = lerp4(a, b, t);
    let e = lerp4(b, c, t);
    return lerp4(d, e, t);
}

fn cubic_deriv4(p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, t: f32) -> vec4<f32> {
    let d0 = p1 - p0;
    let d1 = p2 - p1;
    let d2 = p3 - p2;
    let a = lerp4(d0, d1, t);
    let b = lerp4(d1, d2, t);
    let q = lerp4(a, b, t);
    return q * 3.0;
}

fn homo_point(u: f32, v: f32) -> vec4<f32> {
    let c0 = cubic_point4(hnet[0], hnet[1], hnet[2], hnet[3], u);
    let c1 = cubic_point4(hnet[4], hnet[5], hnet[6], hnet[7], u);
    let c2 = cubic_point4(hnet[8], hnet[9], hnet[10], hnet[11], u);
    let c3 = cubic_point4(hnet[12], hnet[13], hnet[14], hnet[15], u);
    return cubic_point4(c0, c1, c2, c3, v);
}

fn homo_pu(u: f32, v: f32) -> vec4<f32> {
    let c0 = cubic_deriv4(hnet[0], hnet[1], hnet[2], hnet[3], u);
    let c1 = cubic_deriv4(hnet[4], hnet[5], hnet[6], hnet[7], u);
    let c2 = cubic_deriv4(hnet[8], hnet[9], hnet[10], hnet[11], u);
    let c3 = cubic_deriv4(hnet[12], hnet[13], hnet[14], hnet[15], u);
    return cubic_point4(c0, c1, c2, c3, v);
}

fn homo_pv(u: f32, v: f32) -> vec4<f32> {
    let c0 = cubic_point4(hnet[0], hnet[1], hnet[2], hnet[3], u);
    let c1 = cubic_point4(hnet[4], hnet[5], hnet[6], hnet[7], u);
    let c2 = cubic_point4(hnet[8], hnet[9], hnet[10], hnet[11], u);
    let c3 = cubic_point4(hnet[12], hnet[13], hnet[14], hnet[15], u);
    return cubic_deriv4(c0, c1, c2, c3, v);
}

fn pu_num(u: f32, v: f32) -> vec3<f32> {
    let h = homo_point(u, v);
    let hu = homo_pu(u, v);
    return hu.xyz * h.w - h.xyz * hu.w;
}

fn pv_num(u: f32, v: f32) -> vec3<f32> {
    let h = homo_point(u, v);
    let hv = homo_pv(u, v);
    return hv.xyz * h.w - h.xyz * hv.w;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Locate the span and local parameter along u.
    let cu = clamp(q.u, 0.0, 1.0);
    let su = cu * f32(params.u_span);
    let last_u = params.u_span - 1u;
    var span_u = u32(floor(su));
    span_u = min(span_u, last_u);
    let local_u = su - f32(span_u);
    // Locate the span and local parameter along v.
    let cv = clamp(q.v, 0.0, 1.0);
    let sv = cv * f32(params.v_span);
    let last_v = params.v_span - 1u;
    var span_v = u32(floor(sv));
    span_v = min(span_v, last_v);
    let local_v = sv - f32(span_v);

    // Promote the overlapping 4x4 window to weighted homogeneous coordinates.
    var hwin: array<vec4<f32>, 16>;
    for (var wr: u32 = 0u; wr < 4u; wr = wr + 1u) {
        for (var wc: u32 = 0u; wc < 4u; wc = wc + 1u) {
            let gidx = (span_v + wr) * params.cols + (span_u + wc);
            let cpt = control_net[gidx];
            let w = cpt.w;
            hwin[wr * 4u + wc] = vec4<f32>(cpt.xyz * w, w);
        }
    }

    let sixth = 1.0 / 6.0;
    let third = 1.0 / 3.0;
    // Pass 1: convert each row along u.
    var tmp: array<vec4<f32>, 16>;
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let base = r * 4u;
        let c0 = hwin[base];
        let c1 = hwin[base + 1u];
        let c2 = hwin[base + 2u];
        let c3 = hwin[base + 3u];
        tmp[base] = (c0 + c1 * 4.0 + c2) * sixth;
        tmp[base + 1u] = (c1 * 2.0 + c2) * third;
        tmp[base + 2u] = (c1 + c2 * 2.0) * third;
        tmp[base + 3u] = (c1 + c2 * 4.0 + c3) * sixth;
    }
    // Pass 2: convert each column along v.
    var obez: array<vec4<f32>, 16>;
    for (var cc: u32 = 0u; cc < 4u; cc = cc + 1u) {
        let d0 = tmp[cc];
        let d1 = tmp[4u + cc];
        let d2 = tmp[8u + cc];
        let d3 = tmp[12u + cc];
        obez[cc] = (d0 + d1 * 4.0 + d2) * sixth;
        obez[4u + cc] = (d1 * 2.0 + d2) * third;
        obez[8u + cc] = (d1 + d2 * 2.0) * third;
        obez[12u + cc] = (d1 + d2 * 4.0 + d3) * sixth;
    }
    // Project back then re-promote, mirroring the golden homogeneous_row.
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let h = obez[k];
        let w = h.w;
        let cpt = h.xyz / w;
        hnet[k] = vec4<f32>(cpt * w, w);
    }

    // Surface point: rational bicubic De Casteljau with a guarded divide.
    let hp = homo_point(local_u, local_v);
    var p = hp.xyz;
    if (abs(hp.w) > 1e-20) {
        p = hp.xyz / hp.w;
    }

    // Normal: normalized cross of the quotient-rule numerators, with the
    // golden four-step eps search and fallback to (0, 0, 1).
    let fb = vec3<f32>(0.0, 0.0, 1.0);
    var normal = fb;
    var degenerate = 1u;
    for (var step: i32 = 0; step < 4; step = step + 1) {
        let eps = 1e-3 * f32(step);
        let uu = clamp(local_u + eps, 0.0, 1.0);
        let vv = clamp(local_v + eps, 0.0, 1.0);
        let du = pu_num(uu, vv);
        let dv = pv_num(uu, vv);
        let n = cross3(du, dv);
        let len2 = dot3(n, n);
        if (len2 > 0.0) {
            normal = normalize_or3(n, fb);
            degenerate = 0u;
            break;
        }
    }

    var res: SurfResult;
    res.px = p.x;
    res.py = p.y;
    res.pz = p.z;
    res.nx = normal.x;
    res.ny = normal.y;
    res.nz = normal.z;
    res.degenerate = degenerate;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: grid dimensions, span counts and the
/// query count, matching `Params` in [`NURBS_SURFACE_WGSL`]. Three pad words
/// round the struct to a `32`-byte `std430` uniform block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of control rows (along `v`).
    rows: u32,
    /// Number of control columns (along `u`).
    cols: u32,
    /// Number of cubic spans along `u` (`cols - 3`).
    u_span: u32,
    /// Number of cubic spans along `v` (`rows - 3`).
    v_span: u32,
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one control point: position plus weight packed
/// as a `vec4`, matching the `WGSL` `control_net` element.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuControl {
    /// Control-point `x`.
    x: f32,
    /// Control-point `y`.
    y: f32,
    /// Control-point `z`.
    z: f32,
    /// Strictly-positive weight.
    w: f32,
}

/// `repr(C)` `std430` layout of one query: the global `(u, v)`, matching the
/// `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Global surface parameter `u`.
    u: f32,
    /// Global surface parameter `v`.
    v: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `SurfResult`
/// struct: the surface point, the unit normal and the degenerate flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surface point `x`.
    px: f32,
    /// Surface point `y`.
    py: f32,
    /// Surface point `z`.
    pz: f32,
    /// Unit normal `x`.
    nx: f32,
    /// Unit normal `y`.
    ny: f32,
    /// Unit normal `z`.
    nz: f32,
    /// `1` when the tangent frame collapsed and the normal fell back to
    /// `[0, 0, 1]`, `0` otherwise.
    degenerate: u32,
}

/// One query: the global `(u, v)` to evaluate on the shared control grid.
///
/// The control grid and weights are supplied once per [`GpuNurbsSurface::evaluate`]
/// call and shared across the batch; the host enqueues one query per
/// evaluation, and an empty batch is short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NurbsSurfaceQuery {
    /// Global surface parameter `u` in `[0, 1]`.
    pub u: f32,
    /// Global surface parameter `v` in `[0, 1]`.
    pub v: f32,
}

impl NurbsSurfaceQuery {
    /// Builds a query from the global `(u, v)`.
    #[must_use]
    pub const fn new(u: f32, v: f32) -> NurbsSurfaceQuery {
        NurbsSurfaceQuery { u, v }
    }
}

/// One resolved query: the surface point, the unit normal and the degenerate
/// flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NurbsSurfaceResult {
    /// Surface point at the queried `(u, v)`.
    pub point: [f32; 3],
    /// Unit surface normal, or `[0, 0, 1]` when the tangent frame collapsed.
    pub normal: [f32; 3],
    /// `true` when every `eps` step collapsed and the fallback normal was used.
    pub degenerate: bool,
}

/// Decodes one packed [`GpuResult`] into the public [`NurbsSurfaceResult`],
/// unpacking the integer flag into the reference's boolean shape.
fn decode_result(raw: &GpuResult) -> NurbsSurfaceResult {
    NurbsSurfaceResult {
        point: [raw.px, raw.py, raw.pz],
        normal: [raw.nx, raw.ny, raw.nz],
        degenerate: raw.degenerate != 0,
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

/// A compiled, reusable rational bicubic B-spline surface compute pipeline,
/// twinning the golden `ray_scene::nurbs_surface`'s `NurbsSurface::point` and
/// `NurbsSurface::normal`.
pub struct GpuNurbsSurface {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuNurbsSurface {
    /// Compiles the `NURBS`-surface kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuNurbsSurface {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_nurbs_surface"),
            source: ShaderSource::Wgsl(NURBS_SURFACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_nurbs_surface_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_nurbs_surface_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_nurbs_surface_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuNurbsSurface {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` against the shared row-major
    /// `rows × cols` control grid plus matching positive `weights`, returning
    /// one [`NurbsSurfaceResult`] per input, in order.
    ///
    /// Each continuous field equals the reference within floating-point
    /// tolerance and the `degenerate` flag matches exactly. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized. The caller supplies a valid grid
    /// (`cols >= 4`, `rows >= 4`, `control.len() == weights.len() == rows * cols`,
    /// strictly-positive weights), matching the golden `NurbsSurface::new`.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        control: &[[f32; 3]],
        weights: &[f32],
        rows: u32,
        cols: u32,
        queries: &[NurbsSurfaceQuery],
    ) -> Vec<NurbsSurfaceResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            rows,
            cols,
            u_span: cols - 3,
            v_span: rows - 3,
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_nurbs_surface_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let grid: Vec<GpuControl> = control
            .iter()
            .zip(weights.iter())
            .map(|(p, &w)| GpuControl {
                x: p[0],
                y: p[1],
                z: p[2],
                w,
            })
            .collect();
        let grid_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_nurbs_surface_grid"),
            contents: bytemuck::cast_slice(&grid),
            usage: BufferUsages::STORAGE,
        });
        let encoded: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery { u: q.u, v: q.v })
            .collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_nurbs_surface_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_nurbs_surface_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_nurbs_surface_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: grid_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_nurbs_surface_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_nurbs_surface_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_nurbs_surface_pass"),
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
