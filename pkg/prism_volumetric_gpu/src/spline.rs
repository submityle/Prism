//! `wgpu` compute twin of the per-`t` local evaluation of a 3D control-point
//! spline
//! ([`spline`](prism_render_architecture::particle::spline), particle design
//! §8.3).
//!
//! The `CPU` golden
//! [`Spline`](prism_render_architecture::particle::spline::Spline) reconstructs
//! a path from a list of `Vec3` control points under one of three
//! [`SplineMode`](prism_render_architecture::particle::spline::SplineMode)
//! families (`Linear`, `Catmull-Rom`, `Bezier`), evaluated by a global
//! parameter `t` in `0..=1` spread evenly across the segments. [`GpuSpline`] is
//! the on-device twin for the *stateless, per-`t`* queries: one thread resolves
//! one `(control-point sub-slice, mode, closed, t)` query, so a passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! same positions, tangents and curvatures the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Each query reproduces, against one shared control-point buffer, the five
//! per-`t` local evaluations the reference exposes: the world-space point
//! ([`Spline::point_at`](prism_render_architecture::particle::spline::Spline::point_at)),
//! the segment-local first and second derivatives
//! ([`Spline::derivative_at`](prism_render_architecture::particle::spline::Spline::derivative_at)
//! and
//! [`Spline::second_derivative_at`](prism_render_architecture::particle::spline::Spline::second_derivative_at)),
//! the unit tangent
//! ([`Spline::tangent_at`](prism_render_architecture::particle::spline::Spline::tangent_at))
//! and the geometric curvature
//! ([`Spline::curvature_at`](prism_render_architecture::particle::spline::Spline::curvature_at)).
//! The private `segment_count`, `locate`, `cr_ctrl`, `bez_ctrl` and the
//! `eval_*` segment polynomials are twinned too and are exercised indirectly
//! through those five answers.
//!
//! The stateful, variable-length tooling of the reference is deliberately
//! **not** twinned here: the rotation-minimizing orientation frames
//! (`rmf_frames` / `rmf_step` / `orthonormal_reference`), the arc-length table
//! (`rebuild_arc_table` / `sample_by_distance` / `arc_params` / `arc_lengths`)
//! and `sample_along_spline` all carry state or grow with the input and belong
//! to a separate twin.
//!
//! # Mode codes
//!
//! The reference
//! [`SplineMode`](prism_render_architecture::particle::spline::SplineMode) is
//! carried into the kernel as a `u32`: `0` is `Linear`, `1` is `Catmull-Rom`
//! and `2` is `Bezier`. The `closed` flag is a `u32` (`0` open, `1` closed).
//! These classification codes are integer and compared with `==`.
//!
//! # Correctness model
//!
//! The segment polynomials thread through multiplies, adds, one `sqrt` and one
//! guarded division, so they are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a tolerance (`abs_diff <=
//! 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every continuous
//! quantity, tight enough to catch a dropped term or a swapped axis yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A query with too few control points for its mode has `segment_count == 0`:
//! the point then returns the first control point (or the zero vector when the
//! query names none), and the derivative, second derivative, tangent and
//! curvature all return zero, mirroring the reference's degenerate contract. A
//! (near) zero first derivative normalizes to the zero tangent when its squared
//! length is within [`EPS_LEN_SQ`] of zero, and the curvature returns `0` when
//! the speed cube is within [`EPS`] of zero, matching the reference guards bit
//! for bit. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `floor`,
//! `sqrt`, `+ - * /`, `dot`, `cross` and signed/unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. There is no loop: each thread performs a fixed, bounded sequence
//! of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`；无第三方引擎源码或衍生代码。

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
const WORKGROUP_SIZE: u32 = 64;

/// Mode code for a `Linear` polyline, matching
/// [`SplineMode::Linear`](prism_render_architecture::particle::spline::SplineMode::Linear).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
pub const SPLINE_LINEAR: u32 = 0;

/// Mode code for an interpolating `Catmull-Rom` spline, matching
/// [`SplineMode::CatmullRom`](prism_render_architecture::particle::spline::SplineMode::CatmullRom).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
pub const SPLINE_CATMULL_ROM: u32 = 1;

/// Mode code for a cubic `Bezier` chain, matching
/// [`SplineMode::Bezier`](prism_render_architecture::particle::spline::SplineMode::Bezier).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
pub const SPLINE_BEZIER: u32 = 2;

/// The portable core-`WGSL` spline kernel, embedded inline so the twin ships as
/// a single source file. The single entry point `solve` mirrors the `CPU`
/// golden
/// [`spline`](prism_render_architecture::particle::spline) branch for branch;
/// see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
const SPLINE_WGSL: &str = r#"
// Spline twin: one shared read-only control-point buffer, one thread per query.
// Each thread reproduces the per-t local point, first and second derivatives,
// unit tangent and geometric curvature. It mirrors the CPU golden
// `particle::spline` branch for branch, uses only the portable core-WGSL subset
// (clamp/floor/sqrt and + - * / plus dot/cross and index arithmetic), needs no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::spline；无第三方
// 引擎源码或衍生代码。

// Squared-length floor below which a vector normalizes to zero. Matches the
// reference `EPS_LEN_SQ`; the compare rule used instead of an f32 `==`.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Magnitude below which the speed cube is treated as zero so curvature guards
// its division. Matches the reference `EPS`.
const EPS: f32 = 1.0e-6;

// Mode codes mirroring the reference `SplineMode`.
const MODE_LINEAR: u32 = 0u;
const MODE_CATMULL: u32 = 1u;
const MODE_BEZIER: u32 = 2u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First control-point lane index of this query's sub-slice.
    points_offset: u32,
    // Number of control points in this query's sub-slice.
    points_len: u32,
    // Reconstruction mode code (MODE_LINEAR / MODE_CATMULL / MODE_BEZIER).
    mode: u32,
    // Closure flag (0 open, 1 closed).
    closed: u32,
    // Global path parameter in 0..=1.
    needle: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // point_at, with the curvature packed into w's lane.
    position: vec3<f32>,
    curvature: f32,
    // tangent_at (unit first derivative, or zero when degenerate).
    tangent: vec3<f32>,
    pad1: f32,
    // derivative_at (segment-local first derivative).
    derivative: vec3<f32>,
    pad2: f32,
    // second_derivative_at (segment-local second derivative).
    second: vec3<f32>,
    pad3: f32,
}

// A located (segment, local u) pair mirroring the reference `locate` return.
struct Located {
    seg: u32,
    u: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Reads the control point at a slice-local index.
fn ctrl(off: u32, idx: u32) -> vec3<f32> {
    return points[off + idx].xyz;
}

// Number of path segments for the mode, closure and point count; mirrors the
// reference `segment_count`.
fn segment_count(n: u32, mode: u32, closed: u32) -> u32 {
    if (mode == MODE_LINEAR || mode == MODE_CATMULL) {
        if (n < 2u) {
            return 0u;
        }
        if (closed == 1u) {
            return n;
        }
        return n - 1u;
    }
    // MODE_BEZIER.
    if (closed == 1u) {
        if (n < 3u) {
            return 0u;
        }
        return n / 3u;
    }
    if (n < 4u) {
        return 0u;
    }
    return (n - 1u) / 3u;
}

// Maps a global parameter to a (segment, local u) pair; mirrors the reference
// `locate` (clamp, spread evenly, land t == 1 on the final segment's end).
fn locate(t: f32, segments: u32) -> Located {
    let clamped = clamp(t, 0.0, 1.0);
    let scaled = clamped * f32(segments);
    let floored = floor(scaled);
    let seg = u32(floored);
    var out: Located;
    if (seg >= segments) {
        out.seg = segments - 1u;
        out.u = 1.0;
        return out;
    }
    out.seg = seg;
    out.u = scaled - floored;
    return out;
}

// Catmull-Rom control point with end clamping (open) or index wrapping
// (closed); mirrors the reference `cr_ctrl`. `i` may be -1 or segment + 1.
fn cr_ctrl(off: u32, n: u32, i: i32, closed: u32) -> vec3<f32> {
    let ni = i32(n);
    var idx: i32;
    if (closed == 1u) {
        idx = ((i % ni) + ni) % ni;
    } else {
        idx = clamp(i, 0, ni - 1);
    }
    return ctrl(off, u32(idx));
}

// Bezier control point of the segment starting at `base`, wrapping when closed;
// mirrors the reference `bez_ctrl`.
fn bez_ctrl(off: u32, n: u32, base: u32, k: u32, closed: u32) -> vec3<f32> {
    let raw = base + k;
    var idx: u32;
    if (closed == 1u) {
        idx = raw % n;
    } else {
        idx = raw;
    }
    return ctrl(off, idx);
}

// Linear segment point: straight blend between the two anchors.
fn eval_linear(off: u32, n: u32, seg: u32, u: f32) -> vec3<f32> {
    let a = ctrl(off, seg);
    let b = ctrl(off, (seg + 1u) % n);
    return a + (b - a) * u;
}

// Linear segment derivative: the constant chord vector.
fn eval_linear_deriv(off: u32, n: u32, seg: u32) -> vec3<f32> {
    return ctrl(off, (seg + 1u) % n) - ctrl(off, seg);
}

// Catmull-Rom segment point using the standard uniform 0.5 basis.
fn eval_catmull(off: u32, n: u32, seg: u32, u: f32, closed: u32) -> vec3<f32> {
    let s = i32(seg);
    let p0 = cr_ctrl(off, n, s - 1, closed);
    let p1 = cr_ctrl(off, n, s, closed);
    let p2 = cr_ctrl(off, n, s + 1, closed);
    let p3 = cr_ctrl(off, n, s + 2, closed);
    let u2 = u * u;
    let u3 = u2 * u;
    let c0 = p1 * 2.0;
    let c1 = p2 - p0;
    let c2 = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let c3 = p1 * 3.0 - p0 - p2 * 3.0 + p3;
    return (c0 + c1 * u + c2 * u2 + c3 * u3) * 0.5;
}

// Catmull-Rom segment first derivative (d/du).
fn eval_catmull_deriv(off: u32, n: u32, seg: u32, u: f32, closed: u32) -> vec3<f32> {
    let s = i32(seg);
    let p0 = cr_ctrl(off, n, s - 1, closed);
    let p1 = cr_ctrl(off, n, s, closed);
    let p2 = cr_ctrl(off, n, s + 1, closed);
    let p3 = cr_ctrl(off, n, s + 2, closed);
    let u2 = u * u;
    let c1 = p2 - p0;
    let c2 = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let c3 = p1 * 3.0 - p0 - p2 * 3.0 + p3;
    return (c1 + c2 * (2.0 * u) + c3 * (3.0 * u2)) * 0.5;
}

// Catmull-Rom segment second derivative (d^2/du^2), linear in u.
fn eval_catmull_second(off: u32, n: u32, seg: u32, u: f32, closed: u32) -> vec3<f32> {
    let s = i32(seg);
    let p0 = cr_ctrl(off, n, s - 1, closed);
    let p1 = cr_ctrl(off, n, s, closed);
    let p2 = cr_ctrl(off, n, s + 1, closed);
    let p3 = cr_ctrl(off, n, s + 2, closed);
    let c2 = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let c3 = p1 * 3.0 - p0 - p2 * 3.0 + p3;
    return c2 + c3 * (3.0 * u);
}

// Cubic Bezier segment point via the Bernstein basis.
fn eval_bezier(off: u32, n: u32, seg: u32, u: f32, closed: u32) -> vec3<f32> {
    let base = 3u * seg;
    let p0 = bez_ctrl(off, n, base, 0u, closed);
    let p1 = bez_ctrl(off, n, base, 1u, closed);
    let p2 = bez_ctrl(off, n, base, 2u, closed);
    let p3 = bez_ctrl(off, n, base, 3u, closed);
    let mu = 1.0 - u;
    let b0 = mu * mu * mu;
    let b1 = 3.0 * mu * mu * u;
    let b2 = 3.0 * mu * u * u;
    let b3 = u * u * u;
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
}

// Cubic Bezier segment first derivative (d/du).
fn eval_bezier_deriv(off: u32, n: u32, seg: u32, u: f32, closed: u32) -> vec3<f32> {
    let base = 3u * seg;
    let p0 = bez_ctrl(off, n, base, 0u, closed);
    let p1 = bez_ctrl(off, n, base, 1u, closed);
    let p2 = bez_ctrl(off, n, base, 2u, closed);
    let p3 = bez_ctrl(off, n, base, 3u, closed);
    let mu = 1.0 - u;
    return (p1 - p0) * (3.0 * mu * mu)
        + (p2 - p1) * (6.0 * mu * u)
        + (p3 - p2) * (3.0 * u * u);
}

// Cubic Bezier segment second derivative (d^2/du^2).
fn eval_bezier_second(off: u32, n: u32, seg: u32, u: f32, closed: u32) -> vec3<f32> {
    let base = 3u * seg;
    let p0 = bez_ctrl(off, n, base, 0u, closed);
    let p1 = bez_ctrl(off, n, base, 1u, closed);
    let p2 = bez_ctrl(off, n, base, 2u, closed);
    let p3 = bez_ctrl(off, n, base, 3u, closed);
    let mu = 1.0 - u;
    let term0 = (p2 - p1 * 2.0 + p0) * (6.0 * mu);
    let term1 = (p3 - p2 * 2.0 + p1) * (6.0 * u);
    return term0 + term1;
}

// normalize_or_zero: unit vector along d, or zero when d is (numerically) zero.
fn normalize_or_zero(d: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(d, d);
    if (len_sq > EPS_LEN_SQ) {
        return d * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let off = q.points_offset;
    let n = q.points_len;
    let mode = q.mode;
    let closed = q.closed;
    let segments = segment_count(n, mode, closed);

    var position = vec3<f32>(0.0, 0.0, 0.0);
    var derivative = vec3<f32>(0.0, 0.0, 0.0);
    var second = vec3<f32>(0.0, 0.0, 0.0);

    if (segments == 0u) {
        if (n > 0u) {
            position = ctrl(off, 0u);
        }
    } else {
        let loc = locate(q.needle, segments);
        let seg = loc.seg;
        let u = loc.u;
        if (mode == MODE_LINEAR) {
            position = eval_linear(off, n, seg, u);
            derivative = eval_linear_deriv(off, n, seg);
        } else if (mode == MODE_CATMULL) {
            position = eval_catmull(off, n, seg, u, closed);
            derivative = eval_catmull_deriv(off, n, seg, u, closed);
            second = eval_catmull_second(off, n, seg, u, closed);
        } else {
            position = eval_bezier(off, n, seg, u, closed);
            derivative = eval_bezier_deriv(off, n, seg, u, closed);
            second = eval_bezier_second(off, n, seg, u, closed);
        }
    }

    let tangent = normalize_or_zero(derivative);

    let speed = sqrt(dot(derivative, derivative));
    let denom = speed * speed * speed;
    var curvature = 0.0;
    if (denom > EPS) {
        let cr = cross(derivative, second);
        curvature = sqrt(dot(cr, cr)) / denom;
    }

    var out: Result;
    out.position = position;
    out.curvature = curvature;
    out.tangent = tangent;
    out.pad1 = 0.0;
    out.derivative = derivative;
    out.pad2 = 0.0;
    out.second = second;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count padded to the `std140`
/// `16`-byte alignment matching `Params` in [`SPLINE_WGSL`].
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
    /// First control-point lane index of this query's sub-slice.
    points_offset: u32,
    /// Number of control points in this query's sub-slice.
    points_len: u32,
    /// Reconstruction mode code.
    mode: u32,
    /// Closure flag (`0` open, `1` closed).
    closed: u32,
    /// Global path parameter in `0..=1`.
    needle: f32,
    /// Pad lane.
    pad0: f32,
    /// Pad lane.
    pad1: f32,
    /// Pad lane.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Each `vec3` lane carries a trailing pad word so it stays `16`-byte
/// aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `point_at` world position.
    position: [f32; 3],
    /// `curvature_at` packed into the position block's `w` lane.
    curvature: f32,
    /// `tangent_at` unit tangent.
    tangent: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// `derivative_at` segment-local first derivative.
    derivative: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// `second_derivative_at` segment-local second derivative.
    second: [f32; 3],
    /// Padding lane.
    pad3: f32,
}

/// One query for the spline twin against the shared uploaded control points: a
/// sub-slice of control points (by lane offset and length), the reconstruction
/// mode, the closure flag and the global parameter `t`.
///
/// The point, derivatives, tangent and curvature are all answered from the same
/// query, so a single query exercises every twinned evaluation at once.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplineEvalQuery {
    /// First control-point lane index of this query's sub-slice in the shared
    /// buffer.
    pub points_offset: u32,
    /// Number of control points in this query's sub-slice.
    pub points_len: u32,
    /// Mode code ([`SPLINE_LINEAR`], [`SPLINE_CATMULL_ROM`] or
    /// [`SPLINE_BEZIER`]), matching the reference
    /// [`SplineMode`](prism_render_architecture::particle::spline::SplineMode).
    pub mode: u32,
    /// Closure flag: `0` open, `1` closed, matching the reference
    /// [`Spline::is_closed`](prism_render_architecture::particle::spline::Spline::is_closed).
    pub closed: u32,
    /// Global path parameter in `0..=1`, fed to the per-`t` evaluations.
    pub t: f32,
}

/// One resolved answer for a single query, mirroring every per-`t` value the
/// reference reports.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplineEvalResult {
    /// World-space point, matching
    /// [`Spline::point_at`](prism_render_architecture::particle::spline::Spline::point_at).
    pub position: [f32; 3],
    /// Unit tangent, matching
    /// [`Spline::tangent_at`](prism_render_architecture::particle::spline::Spline::tangent_at).
    pub tangent: [f32; 3],
    /// Geometric curvature, matching
    /// [`Spline::curvature_at`](prism_render_architecture::particle::spline::Spline::curvature_at).
    pub curvature: f32,
    /// Segment-local first derivative, matching
    /// [`Spline::derivative_at`](prism_render_architecture::particle::spline::Spline::derivative_at).
    pub derivative: [f32; 3],
    /// Segment-local second derivative, matching
    /// [`Spline::second_derivative_at`](prism_render_architecture::particle::spline::Spline::second_derivative_at).
    pub second: [f32; 3],
}

/// Encodes one [`SplineEvalQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SplineEvalQuery) -> GpuQuery {
    GpuQuery {
        points_offset: q.points_offset,
        points_len: q.points_len,
        mode: q.mode,
        closed: q.closed,
        needle: q.t,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SplineEvalResult`].
fn decode_result(raw: &GpuResult) -> SplineEvalResult {
    SplineEvalResult {
        position: raw.position,
        tangent: raw.tangent,
        curvature: raw.curvature,
        derivative: raw.derivative,
        second: raw.second,
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

/// A compiled, reusable spline compute pipeline, twinning the per-`t` local
/// evaluation of the `CPU` golden
/// [`spline`](prism_render_architecture::particle::spline).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`。
pub struct GpuSpline {
    /// Logical device, cloned from the acquiring [`GpuContext`] so a dispatch
    /// needs no borrowed context.
    device: Device,
    /// Submission queue, cloned from the acquiring [`GpuContext`].
    queue: Queue,
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSpline {
    /// Compiles the spline kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSpline {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_spline"),
            source: ShaderSource::Wgsl(SPLINE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_spline_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_spline_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_spline_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpline {
            device: device.clone(),
            queue: ctx.queue().clone(),
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` against one shared control-point buffer
    /// and returns one [`SplineEvalResult`] per input, in order.
    ///
    /// `points` is the shared, flattened list of `[x, y, z]` control points
    /// every query indexes through its `points_offset` / `points_len`. The
    /// positions, tangents, derivatives and curvatures match the reference to
    /// within the tolerance documented on this module. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        queries: &[SplineEvalQuery],
        points: &[[f32; 3]],
    ) -> Vec<SplineEvalResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = &self.device;

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spline_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        // Each control point is padded to a vec4 lane so the storage array stays
        // 16-byte aligned on device. A storage buffer cannot be zero-sized, so
        // an empty point list uploads a single unused zero lane.
        let mut lanes: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        if lanes.is_empty() {
            lanes.push([0.0, 0.0, 0.0, 0.0]);
        }
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spline_points"),
            contents: bytemuck::cast_slice(&lanes),
            usage: BufferUsages::STORAGE,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spline_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spline_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_spline_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
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
            label: Some("prism_volumetric_spline_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_spline_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_spline_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        self.queue.submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("device poll should complete the submitted work");
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
