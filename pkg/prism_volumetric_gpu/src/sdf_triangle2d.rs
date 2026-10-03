//! `wgpu` compute twin of three exact 2D triangle-family signed-distance
//! fields from the reference ray-scene `SDF` primitive library
//! (`prism_render_architecture::ray_scene::sdf_primitives`).
//!
//! Three closed-form distance evaluators are twinned, each the exact field of a
//! filled 2D shape and all transcendental-free (only `abs`, `clamp`, `min`,
//! products and one `sqrt`):
//!
//! - `triangle_2d(point, a, b, c)`: the exact field of an arbitrary triangle
//!   with vertices `a`, `b` and `c`. Each of the three edges contributes a
//!   point-to-segment distance with its projection parameter clamped to the
//!   edge; the unsigned distance is the smallest of the three, and the interior
//!   sign is recovered winding-independently from the component-wise minimum of
//!   the signed edge areas scaled by the triangle orientation `s`.
//! - `isosceles_triangle_2d(point, half_base, height)`: the exact field of an
//!   isosceles triangle with apex at the origin and a horizontal base of
//!   half-width `half_base` at `y = height`. After folding `x` to its
//!   magnitude the field is the smaller of the slanted-edge and capped-base
//!   distances, signed by two edge half-plane tests.
//! - `oriented_vesica_2d(point, a, b, w)`: the exact field of a lens whose
//!   pointed tips sit at `a` and `b` with apex half-width `w`. The query is
//!   mapped into the lens-local axial / perpendicular frame and delegated to
//!   the origin-centred canonical vesica, whose nearest feature is either a
//!   cusp or one of the two arcs, selected by a single linear test.
//!
//! # What is twinned
//!
//! One thread resolves one query. Each [`SdfTriangle2dQuery`] carries the point
//! and the per-shape parameters; the kernel evaluates all three fields and
//! writes one [`SdfTriangle2dResult`] holding the three signed distances. The
//! twin spells out the same closed form with the same ordered clamps, folds and
//! sign tests as the reference, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same distance the reference does,
//! not merely that the shader compiles.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: each field is a fixed,
//! bounded sequence of clamps, products and one `sqrt` that runs entirely on
//! device. The host only flattens the query batch into a `std430` storage
//! buffer and short-circuits an empty batch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! Each distance is a *continuous* quantity threaded through `sqrt` and
//! products, so the `CPU` and `GPU` are not bit-exact: a device `sqrt` may land
//! a few units in the last place from the scalar reference. The parity test
//! compares with an absolute-or-relative tolerance (`abs <= 1e-4 || rel <=
//! 1e-3`, relative floor `1e-6`). The reference closes `triangle_2d` and
//! `isosceles_triangle_2d` with `signum`, whose Rust result is `+1` at zero
//! while `WGSL` `sign` returns `0`; the twin instead uses a positive-at-zero
//! branch on both the host oracle and the device so they agree on the interior
//! sign away from the exact edge. The sharp triangle tips are conditioning
//! hot-spots where that branch flips, so named fixtures keep clear of the exact
//! feature and the randomized sweep rejects points within a small margin of any
//! zero-crossing.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `select`, `+ - * /` and ordered comparisons — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `u64` / `u16` / `i64` / `f64`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。
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

/// Positive-at-zero sign, matching the device `signum_branch` so the host
/// oracle and the device agree on the interior sign away from the exact edge.
fn signum_branch(value: f32) -> f32 {
    if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Euclidean length of a 2D vector; one `sqrt`, which is a core arithmetic
/// primitive rather than a transcendental.
fn length2(ax: f32, ay: f32) -> f32 {
    (ax * ax + ay * ay).sqrt()
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::triangle_2d`, reproduced without importing the
/// golden so the twin stays self-contained.
///
/// Measures the clamped point-to-segment distance against each of the three
/// edges, taking the smallest, and recovers the interior sign from the signed
/// edge areas scaled by the triangle orientation.
#[must_use]
pub fn triangle_2d_sdf(point: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    let e0x = b[0] - a[0];
    let e0y = b[1] - a[1];
    let e1x = c[0] - b[0];
    let e1y = c[1] - b[1];
    let e2x = a[0] - c[0];
    let e2y = a[1] - c[1];
    let v0x = point[0] - a[0];
    let v0y = point[1] - a[1];
    let v1x = point[0] - b[0];
    let v1y = point[1] - b[1];
    let v2x = point[0] - c[0];
    let v2y = point[1] - c[1];
    let t0 = ((v0x * e0x + v0y * e0y) / (e0x * e0x + e0y * e0y)).clamp(0.0, 1.0);
    let pq0x = v0x - e0x * t0;
    let pq0y = v0y - e0y * t0;
    let t1 = ((v1x * e1x + v1y * e1y) / (e1x * e1x + e1y * e1y)).clamp(0.0, 1.0);
    let pq1x = v1x - e1x * t1;
    let pq1y = v1y - e1y * t1;
    let t2 = ((v2x * e2x + v2y * e2y) / (e2x * e2x + e2y * e2y)).clamp(0.0, 1.0);
    let pq2x = v2x - e2x * t2;
    let pq2y = v2y - e2y * t2;
    let s = signum_branch(e0x * e2y - e0y * e2x);
    let dx = (pq0x * pq0x + pq0y * pq0y)
        .min(pq1x * pq1x + pq1y * pq1y)
        .min(pq2x * pq2x + pq2y * pq2y);
    let dy = (s * (v0x * e0y - v0y * e0x))
        .min(s * (v1x * e1y - v1y * e1x))
        .min(s * (v2x * e2y - v2y * e2x));
    -dx.sqrt() * signum_branch(dy)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::isosceles_triangle_2d`, reproduced without
/// importing the golden.
///
/// Folds `x` to its magnitude, takes the smaller of the slanted-edge and
/// capped-base distances, and recovers the interior sign from two edge
/// half-plane tests.
#[must_use]
pub fn isosceles_triangle_2d_sdf(point: [f32; 2], half_base: f32, height: f32) -> f32 {
    let qx = half_base;
    let qy = height;
    let px = point[0].abs();
    let py = point[1];
    let t = ((px * qx + py * qy) / (qx * qx + qy * qy)).clamp(0.0, 1.0);
    let ax = px - qx * t;
    let ay = py - qy * t;
    let tb = (px / qx).clamp(0.0, 1.0);
    let bx = px - qx * tb;
    let by = py - qy;
    let s = -signum_branch(qy);
    let dx = (ax * ax + ay * ay).min(bx * bx + by * by);
    let dy = (s * (px * qy - py * qx)).min(s * (py - qy));
    -dx.sqrt() * signum_branch(dy)
}

/// Canonical origin-centred vesica, the exact field of a lens formed by two
/// circles of `radius` centred at `(-offset, 0)` and `(+offset, 0)`. The query
/// is folded into the first quadrant and the nearest feature is either a cusp
/// or one of the two arcs, selected by a single linear test.
fn vesica_core(point: [f32; 2], radius: f32, offset: f32) -> f32 {
    let px = point[0].abs();
    let py = point[1].abs();
    let b = (radius * radius - offset * offset).sqrt();
    if (py - b) * offset > px * b {
        length2(px, py - b) * signum_branch(offset)
    } else {
        length2(px + offset, py) - radius
    }
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::oriented_vesica_2d`, reproduced without
/// importing the golden.
///
/// Maps the query into the lens-local axial / perpendicular frame, derives the
/// canonical `radius` / `offset` from the tip separation and apex half-width,
/// and delegates to [`vesica_core`].
#[must_use]
pub fn oriented_vesica_2d_sdf(point: [f32; 2], a: [f32; 2], b: [f32; 2], w: f32) -> f32 {
    let cx = (a[0] + b[0]) * 0.5;
    let cy = (a[1] + b[1]) * 0.5;
    let bax = b[0] - a[0];
    let bay = b[1] - a[1];
    let l = length2(bax, bay);
    let vx = bax / l;
    let vy = bay / l;
    let pcx = point[0] - cx;
    let pcy = point[1] - cy;
    let axial = pcx * vx + pcy * vy;
    let perp = -pcx * vy + pcy * vx;
    let half = l * 0.5;
    let radius = (half * half + w * w) / (2.0 * w);
    let offset = (half * half - w * w) / (2.0 * w);
    vesica_core([perp, axial], radius, offset)
}

/// The portable core-`WGSL` triangle-family `SDF` kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the three golden evaluators; see the module documentation.
const SDF_TRIANGLE2D_WGSL: &str = r#"
// Triangle-family 2D SDF twin: one thread evaluates the arbitrary-triangle,
// isosceles-triangle and oriented-vesica fields for one query, mirroring the
// golden {triangle_2d, isosceles_triangle_2d, oriented_vesica_2d} with only
// clamps, folds, products and one sqrt. Transcendental-free.
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point.
    px: f32,
    py: f32,
    // Arbitrary-triangle vertices a, b, c.
    tri_ax: f32,
    tri_ay: f32,
    tri_bx: f32,
    tri_by: f32,
    tri_cx: f32,
    tri_cy: f32,
    // Isosceles triangle half-base and height.
    iso_half_base: f32,
    iso_height: f32,
    // Oriented vesica tips a, b and apex half-width.
    ves_ax: f32,
    ves_ay: f32,
    ves_bx: f32,
    ves_by: f32,
    ves_w: f32,
}

struct SdfResult {
    dist_triangle: f32,
    dist_isosceles: f32,
    dist_vesica: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<SdfResult>;

fn length2(ax: f32, ay: f32) -> f32 {
    return sqrt(ax * ax + ay * ay);
}

// Positive-at-zero sign, matching the host oracle so the CPU and GPU agree on
// the interior sign away from the exact edge.
fn signum_branch(value: f32) -> f32 {
    if (value < 0.0) {
        return -1.0;
    }
    return 1.0;
}

fn triangle_sdf(
    px: f32, py: f32,
    ax: f32, ay: f32,
    bx: f32, by: f32,
    cx: f32, cy: f32,
) -> f32 {
    let e0x = bx - ax;
    let e0y = by - ay;
    let e1x = cx - bx;
    let e1y = cy - by;
    let e2x = ax - cx;
    let e2y = ay - cy;
    let v0x = px - ax;
    let v0y = py - ay;
    let v1x = px - bx;
    let v1y = py - by;
    let v2x = px - cx;
    let v2y = py - cy;
    let t0 = clamp((v0x * e0x + v0y * e0y) / (e0x * e0x + e0y * e0y), 0.0, 1.0);
    let pq0x = v0x - e0x * t0;
    let pq0y = v0y - e0y * t0;
    let t1 = clamp((v1x * e1x + v1y * e1y) / (e1x * e1x + e1y * e1y), 0.0, 1.0);
    let pq1x = v1x - e1x * t1;
    let pq1y = v1y - e1y * t1;
    let t2 = clamp((v2x * e2x + v2y * e2y) / (e2x * e2x + e2y * e2y), 0.0, 1.0);
    let pq2x = v2x - e2x * t2;
    let pq2y = v2y - e2y * t2;
    let s = signum_branch(e0x * e2y - e0y * e2x);
    let dx = min(
        min(pq0x * pq0x + pq0y * pq0y, pq1x * pq1x + pq1y * pq1y),
        pq2x * pq2x + pq2y * pq2y,
    );
    let dy = min(
        min(s * (v0x * e0y - v0y * e0x), s * (v1x * e1y - v1y * e1x)),
        s * (v2x * e2y - v2y * e2x),
    );
    return -sqrt(dx) * signum_branch(dy);
}

fn isosceles_sdf(point_x: f32, point_y: f32, half_base: f32, height: f32) -> f32 {
    let qx = half_base;
    let qy = height;
    let px = abs(point_x);
    let py = point_y;
    let t = clamp((px * qx + py * qy) / (qx * qx + qy * qy), 0.0, 1.0);
    let ax = px - qx * t;
    let ay = py - qy * t;
    let tb = clamp(px / qx, 0.0, 1.0);
    let bx = px - qx * tb;
    let by = py - qy;
    let s = -signum_branch(qy);
    let dx = min(ax * ax + ay * ay, bx * bx + by * by);
    let dy = min(s * (px * qy - py * qx), s * (py - qy));
    return -sqrt(dx) * signum_branch(dy);
}

fn vesica_core(point_x: f32, point_y: f32, radius: f32, offset: f32) -> f32 {
    let px = abs(point_x);
    let py = abs(point_y);
    let b = sqrt(radius * radius - offset * offset);
    let cusp = length2(px, py - b) * signum_branch(offset);
    let arc = length2(px + offset, py) - radius;
    let use_cusp = (py - b) * offset > px * b;
    return select(arc, cusp, use_cusp);
}

fn oriented_vesica_sdf(
    px: f32, py: f32,
    ax: f32, ay: f32,
    bx: f32, by: f32,
    w: f32,
) -> f32 {
    let cx = (ax + bx) * 0.5;
    let cy = (ay + by) * 0.5;
    let bax = bx - ax;
    let bay = by - ay;
    let l = length2(bax, bay);
    let vx = bax / l;
    let vy = bay / l;
    let pcx = px - cx;
    let pcy = py - cy;
    let axial = pcx * vx + pcy * vy;
    let perp = -pcx * vy + pcy * vx;
    let half = l * 0.5;
    let radius = (half * half + w * w) / (2.0 * w);
    let offset = (half * half - w * w) / (2.0 * w);
    return vesica_core(perp, axial, radius, offset);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: SdfResult;
    out.dist_triangle = triangle_sdf(
        q.px, q.py,
        q.tri_ax, q.tri_ay,
        q.tri_bx, q.tri_by,
        q.tri_cx, q.tri_cy,
    );
    out.dist_isosceles = isosceles_sdf(q.px, q.py, q.iso_half_base, q.iso_height);
    out.dist_vesica = oriented_vesica_sdf(
        q.px, q.py,
        q.ves_ax, q.ves_ay,
        q.ves_bx, q.ves_by,
        q.ves_w,
    );
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`SDF_TRIANGLE2D_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the point and the per-shape
/// parameters, matching the `WGSL` `Query` struct (fifteen `f32`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Arbitrary-triangle vertex `a` `x`.
    tri_ax: f32,
    /// Arbitrary-triangle vertex `a` `y`.
    tri_ay: f32,
    /// Arbitrary-triangle vertex `b` `x`.
    tri_bx: f32,
    /// Arbitrary-triangle vertex `b` `y`.
    tri_by: f32,
    /// Arbitrary-triangle vertex `c` `x`.
    tri_cx: f32,
    /// Arbitrary-triangle vertex `c` `y`.
    tri_cy: f32,
    /// Isosceles triangle half-base.
    iso_half_base: f32,
    /// Isosceles triangle height.
    iso_height: f32,
    /// Oriented vesica tip `a` `x`.
    ves_ax: f32,
    /// Oriented vesica tip `a` `y`.
    ves_ay: f32,
    /// Oriented vesica tip `b` `x`.
    ves_bx: f32,
    /// Oriented vesica tip `b` `y`.
    ves_by: f32,
    /// Oriented vesica apex half-width.
    ves_w: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `SdfResult`
/// struct: the three signed distances.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the arbitrary triangle.
    dist_triangle: f32,
    /// Signed distance to the isosceles triangle.
    dist_isosceles: f32,
    /// Signed distance to the oriented vesica.
    dist_vesica: f32,
}

/// One query: the point plus the per-shape parameters for the three fields.
///
/// The `tri_*` sextuple carries the arbitrary triangle's vertices, the `iso_*`
/// pair the isosceles triangle, and the `ves_*` quintuple the oriented vesica.
/// The host enqueues one query per evaluation, and an empty batch is
/// short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfTriangle2dQuery {
    /// Query point `x`.
    pub px: f32,
    /// Query point `y`.
    pub py: f32,
    /// Arbitrary-triangle vertex `a` `x`.
    pub tri_ax: f32,
    /// Arbitrary-triangle vertex `a` `y`.
    pub tri_ay: f32,
    /// Arbitrary-triangle vertex `b` `x`.
    pub tri_bx: f32,
    /// Arbitrary-triangle vertex `b` `y`.
    pub tri_by: f32,
    /// Arbitrary-triangle vertex `c` `x`.
    pub tri_cx: f32,
    /// Arbitrary-triangle vertex `c` `y`.
    pub tri_cy: f32,
    /// Isosceles triangle half-base.
    pub iso_half_base: f32,
    /// Isosceles triangle height.
    pub iso_height: f32,
    /// Oriented vesica tip `a` `x`.
    pub ves_ax: f32,
    /// Oriented vesica tip `a` `y`.
    pub ves_ay: f32,
    /// Oriented vesica tip `b` `x`.
    pub ves_bx: f32,
    /// Oriented vesica tip `b` `y`.
    pub ves_by: f32,
    /// Oriented vesica apex half-width.
    pub ves_w: f32,
}

impl SdfTriangle2dQuery {
    /// Builds a query from the point and the per-shape parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query packs the point and three shapes' parameters as flat scalars for a std430 slot"
    )]
    pub const fn new(
        px: f32,
        py: f32,
        tri_ax: f32,
        tri_ay: f32,
        tri_bx: f32,
        tri_by: f32,
        tri_cx: f32,
        tri_cy: f32,
        iso_half_base: f32,
        iso_height: f32,
        ves_ax: f32,
        ves_ay: f32,
        ves_bx: f32,
        ves_by: f32,
        ves_w: f32,
    ) -> SdfTriangle2dQuery {
        SdfTriangle2dQuery {
            px,
            py,
            tri_ax,
            tri_ay,
            tri_bx,
            tri_by,
            tri_cx,
            tri_cy,
            iso_half_base,
            iso_height,
            ves_ax,
            ves_ay,
            ves_bx,
            ves_by,
            ves_w,
        }
    }
}

/// One resolved query: the three signed distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfTriangle2dResult {
    /// Signed distance to the arbitrary triangle.
    pub dist_triangle: f32,
    /// Signed distance to the isosceles triangle.
    pub dist_isosceles: f32,
    /// Signed distance to the oriented vesica.
    pub dist_vesica: f32,
}

/// Encodes one [`SdfTriangle2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfTriangle2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        tri_ax: q.tri_ax,
        tri_ay: q.tri_ay,
        tri_bx: q.tri_bx,
        tri_by: q.tri_by,
        tri_cx: q.tri_cx,
        tri_cy: q.tri_cy,
        iso_half_base: q.iso_half_base,
        iso_height: q.iso_height,
        ves_ax: q.ves_ax,
        ves_ay: q.ves_ay,
        ves_bx: q.ves_bx,
        ves_by: q.ves_by,
        ves_w: q.ves_w,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfTriangle2dResult`].
fn decode_result(raw: &GpuResult) -> SdfTriangle2dResult {
    SdfTriangle2dResult {
        dist_triangle: raw.dist_triangle,
        dist_isosceles: raw.dist_isosceles,
        dist_vesica: raw.dist_vesica,
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

/// A compiled, reusable triangle-family `SDF` compute pipeline, twinning the
/// golden `ray_scene::sdf_primitives` evaluators `triangle_2d`,
/// `isosceles_triangle_2d` and `oriented_vesica_2d`.
pub struct GpuSdfTriangle2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfTriangle2d {
    /// Compiles the triangle-family `SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfTriangle2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d"),
            source: ShaderSource::Wgsl(SDF_TRIANGLE2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfTriangle2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`SdfTriangle2dResult`]
    /// per input, in order.
    ///
    /// Each distance equals the reference within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfTriangle2dQuery],
    ) -> Vec<SdfTriangle2dResult> {
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
            label: Some("prism_volumetric_sdf_triangle2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_triangle2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_triangle2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_triangle2d_pass"),
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
