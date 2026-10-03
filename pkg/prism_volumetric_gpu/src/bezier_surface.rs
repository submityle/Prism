//! `wgpu` compute twin of the composite bicubic **Bézier surface** evaluator
//! from the reference ray-scene primitive
//! (`prism_render_architecture::ray_scene::bezier_surface`).
//!
//! The `CPU` golden `BezierSurface` tiles a `(3m + 1) × (3n + 1)` control grid
//! into `m × n` bicubic patches: a global parameter `(u, v) ∈ [0, 1]²` is
//! scaled by the patch count, the integer part selects the patch and the
//! fractional part becomes the local patch parameter (`locate`). Each patch is
//! evaluated by the purely linear De Casteljau recursion of
//! `prism_render_architecture::ray_scene::bezier_patch` — the surface position
//! is a bicubic point and the shading normal is the normalized cross product of
//! the two cubic partial derivatives, with a short `eps` step-search that nudges
//! off a collapsed tangent frame and falls back to `[0, 0, 1]` when every step
//! stays degenerate.
//!
//! [`GpuBezierSurface`] is the on-device twin. To keep one thread fully
//! self-contained it fixes the grid to the minimal single-patch case
//! `rows = 4, cols = 4`, where `locate(t, 1)` reduces exactly to
//! `clamp(t, 0, 1)` and `patch_at(0, 0)` is the identity window over the sixteen
//! control points. The kernel therefore still walks the global `(u, v)` → local
//! mapping path (a `clamp`) before running the bicubic De Casteljau, so a
//! passing real-device parity test is direct evidence the ported kernel runs the
//! same recursion and classifies the same degenerate normal, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! One thread resolves one query. Each [`BezierSurfaceQuery`] carries the
//! sixteen row-major control points of a single `4 × 4` net plus the global
//! `(u, v)`; the kernel reproduces `BezierSurface::point` and
//! `BezierSurface::normal` for the single-patch grid and writes one
//! [`BezierSurfaceResult`] holding the surface point, the unit normal and a
//! `degenerate` flag that is set when the `eps` step-search exhausts all four
//! steps with a collapsed tangent frame and the normal falls back to
//! `[0, 0, 1]`.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: the whole bicubic
//! evaluation is a fixed, bounded sequence of `lerp`s, products, adds,
//! comparisons and a single `sqrt` that runs entirely on device. The host only
//! flattens the query batch into a `std430` storage buffer and short-circuits an
//! empty batch, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The surface point and the normal are *continuous* quantities threaded through
//! nested `lerp`s, a `cross`, a `sqrt` and divisions, so the `CPU` and `GPU` are
//! not bit-exact: a device may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore
//! compares the continuous fields with an absolute-or-relative tolerance
//! (`abs <= 1e-4 || rel <= 1e-3`, relative floor `1e-6`) while pinning the
//! integer `degenerate` flag exactly. The genuine degeneracy — a collapsed
//! tangent frame whose cross product magnitude crosses zero — is kept off its
//! threshold by the fixtures and the rejection-sampled sweep so the two sides
//! agree on the discrete classification.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `abs`, `sqrt`, `+ - * /` and ordered comparisons — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round`, no
//! `%` and no `u64` / `u16` / `i64` / `f64`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bezier_surface`；无第三方引擎源码或衍生代码。
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

/// Number of control points in a single `4 × 4` bicubic net.
const CONTROL_COUNT: usize = 16;

/// Component-wise linear interpolation `a + (b - a) * t`, matching the golden
/// `lerp3`.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Component-wise difference `a − b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
/// length, matching the golden `normalize_or3`.
fn normalize_or3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len2 = dot3(v, v);
    if len2 <= 0.0 {
        return fallback;
    }
    let inv = 1.0 / len2.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Cubic Bézier point at `t` over four control points (three nested `lerp`s),
/// matching the golden `cubic_point`.
fn cubic_point(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let a = lerp3(p[0], p[1], t);
    let b = lerp3(p[1], p[2], t);
    let c = lerp3(p[2], p[3], t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    lerp3(d, e, t)
}

/// Cubic Bézier derivative at `t`: `3·` the quadratic De Casteljau over the
/// three adjacent control-point differences, matching the golden `cubic_deriv`.
fn cubic_deriv(p: [[f32; 3]; 4], t: f32) -> [f32; 3] {
    let d0 = sub3(p[1], p[0]);
    let d1 = sub3(p[2], p[1]);
    let d2 = sub3(p[3], p[2]);
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let q = lerp3(a, b, t);
    [q[0] * 3.0, q[1] * 3.0, q[2] * 3.0]
}

/// Returns the four control points of row `r` of the single `4 × 4` net.
fn net_row(control: &[[f32; 3]; CONTROL_COUNT], r: usize) -> [[f32; 3]; 4] {
    [
        control[r * 4],
        control[r * 4 + 1],
        control[r * 4 + 2],
        control[r * 4 + 3],
    ]
}

/// Evaluates the surface point of the single `4 × 4` net at local `(u, v)` by a
/// bicubic De Casteljau: collapse each row in `u`, then collapse the four
/// results in `v`.
fn net_point(control: &[[f32; 3]; CONTROL_COUNT], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_point(net_row(control, 0), u),
        cubic_point(net_row(control, 1), u),
        cubic_point(net_row(control, 2), u),
        cubic_point(net_row(control, 3), u),
    ];
    cubic_point(column, v)
}

/// Evaluates `∂P/∂u` of the single net at local `(u, v)`: each row contributes
/// its `u`-tangent, blended by a cubic De Casteljau in `v`.
fn net_partial_u(control: &[[f32; 3]; CONTROL_COUNT], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_deriv(net_row(control, 0), u),
        cubic_deriv(net_row(control, 1), u),
        cubic_deriv(net_row(control, 2), u),
        cubic_deriv(net_row(control, 3), u),
    ];
    cubic_point(column, v)
}

/// Evaluates `∂P/∂v` of the single net at local `(u, v)`: each row collapses in
/// `u`, then the four points are differentiated by a cubic derivative in `v`.
fn net_partial_v(control: &[[f32; 3]; CONTROL_COUNT], u: f32, v: f32) -> [f32; 3] {
    let column = [
        cubic_point(net_row(control, 0), u),
        cubic_point(net_row(control, 1), u),
        cubic_point(net_row(control, 2), u),
        cubic_point(net_row(control, 3), u),
    ];
    cubic_deriv(column, v)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::bezier_surface`'s `BezierSurface::point` and
/// `BezierSurface::normal` for the minimal single-patch grid
/// (`rows = 4, cols = 4`), reproduced without importing the golden so the twin
/// stays self-contained.
///
/// For the single patch `locate(t, 1)` reduces exactly to `clamp(t, 0, 1)`, so
/// the global `(u, v)` are clamped to the local patch parameters and the net is
/// evaluated directly. The surface point is a bicubic De Casteljau; the normal
/// is `normalize(∂P/∂u × ∂P/∂v)` with the golden four-step `eps` search for a
/// non-degenerate tangent frame, falling back to `[0, 0, 1]`. Returns
/// `(point, normal, degenerate)`, where `degenerate` is `true` only when every
/// step collapses and the fallback normal is used.
#[must_use]
pub fn eval_bezier_surface(
    control: &[[f32; 3]; CONTROL_COUNT],
    u: f32,
    v: f32,
) -> ([f32; 3], [f32; 3], bool) {
    const FALLBACK: [f32; 3] = [0.0, 0.0, 1.0];
    let local_u = u.clamp(0.0, 1.0);
    let local_v = v.clamp(0.0, 1.0);
    let point = net_point(control, local_u, local_v);

    let mut normal = FALLBACK;
    let mut degenerate = true;
    for step in 0..4 {
        let eps = 1e-3 * step as f32;
        let uu = (local_u + eps).clamp(0.0, 1.0);
        let vv = (local_v + eps).clamp(0.0, 1.0);
        let du = net_partial_u(control, uu, vv);
        let dv = net_partial_v(control, uu, vv);
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

/// The portable core-`WGSL` single-patch Bézier-surface kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the golden `BezierSurface::point` and `BezierSurface::normal` for the
/// minimal `4 × 4` grid; see the module documentation.
const BEZIER_SURFACE_WGSL: &str = r#"
// Single-patch bicubic Bézier-surface twin: one thread evaluates one query,
// mirroring the golden BezierSurface::point and BezierSurface::normal for the
// minimal 4x4 grid. For a single patch locate(t,1) reduces to clamp(t,0,1), so
// the global (u,v) are clamped to the local patch parameters, the point is a
// bicubic De Casteljau and the normal is normalize(cross(du,dv)) with the four
// step eps search that falls back to (0,0,1). Transcendental-free: only lerps,
// products, adds, comparisons and one sqrt.
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::bezier_surface；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Sixteen row-major control points, flattened to 48 scalars (x,y,z each).
    cp: array<f32, 48>,
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
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<SurfResult>;

// Per-invocation copy of the sixteen control points of the single net.
var<private> net: array<vec3<f32>, 16>;

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

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn cubic_point3(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let a = lerp3(p0, p1, t);
    let b = lerp3(p1, p2, t);
    let c = lerp3(p2, p3, t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    return lerp3(d, e, t);
}

fn cubic_deriv3(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let d0 = p1 - p0;
    let d1 = p2 - p1;
    let d2 = p3 - p2;
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let q = lerp3(a, b, t);
    return q * 3.0;
}

fn net_point(u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point3(net[0], net[1], net[2], net[3], u);
    let c1 = cubic_point3(net[4], net[5], net[6], net[7], u);
    let c2 = cubic_point3(net[8], net[9], net[10], net[11], u);
    let c3 = cubic_point3(net[12], net[13], net[14], net[15], u);
    return cubic_point3(c0, c1, c2, c3, v);
}

fn net_pu(u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_deriv3(net[0], net[1], net[2], net[3], u);
    let c1 = cubic_deriv3(net[4], net[5], net[6], net[7], u);
    let c2 = cubic_deriv3(net[8], net[9], net[10], net[11], u);
    let c3 = cubic_deriv3(net[12], net[13], net[14], net[15], u);
    return cubic_point3(c0, c1, c2, c3, v);
}

fn net_pv(u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point3(net[0], net[1], net[2], net[3], u);
    let c1 = cubic_point3(net[4], net[5], net[6], net[7], u);
    let c2 = cubic_point3(net[8], net[9], net[10], net[11], u);
    let c3 = cubic_point3(net[12], net[13], net[14], net[15], u);
    return cubic_deriv3(c0, c1, c2, c3, v);
}

fn normalize_or3(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len2 = dot3(v, v);
    if (len2 <= 0.0) {
        return fallback;
    }
    let inv = 1.0 / sqrt(len2);
    return v * inv;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let base = k * 3u;
        net[k] = vec3<f32>(q.cp[base], q.cp[base + 1u], q.cp[base + 2u]);
    }

    let local_u = clamp(q.u, 0.0, 1.0);
    let local_v = clamp(q.v, 0.0, 1.0);
    let fallback = vec3<f32>(0.0, 0.0, 1.0);
    let p = net_point(local_u, local_v);

    var normal = fallback;
    var degenerate = 1u;
    for (var step: i32 = 0; step < 4; step = step + 1) {
        let eps = 1e-3 * f32(step);
        let uu = clamp(local_u + eps, 0.0, 1.0);
        let vv = clamp(local_v + eps, 0.0, 1.0);
        let du = net_pu(uu, vv);
        let dv = net_pv(uu, vv);
        let n = cross3(du, dv);
        let len2 = dot3(n, n);
        if (len2 > 0.0) {
            normal = normalize_or3(n, fallback);
            degenerate = 0u;
            break;
        }
    }

    var out: SurfResult;
    out.px = p.x;
    out.py = p.y;
    out.pz = p.z;
    out.nx = normal.x;
    out.ny = normal.y;
    out.nz = normal.z;
    out.degenerate = degenerate;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`BEZIER_SURFACE_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the forty-eight flattened control
/// scalars followed by the global `(u, v)`, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Sixteen row-major control points flattened as `x, y, z` triples.
    cp: [f32; 48],
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

/// One query: the sixteen control points of a single `4 × 4` Bézier net plus
/// the global `(u, v)` to evaluate.
///
/// The `control` array is row-major (`col` along `u`, `row` along `v`); the host
/// enqueues one query per evaluation, and an empty batch is short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BezierSurfaceQuery {
    /// Row-major `4 × 4` control net.
    pub control: [[f32; 3]; CONTROL_COUNT],
    /// Global surface parameter `u` in `[0, 1]`.
    pub u: f32,
    /// Global surface parameter `v` in `[0, 1]`.
    pub v: f32,
}

impl BezierSurfaceQuery {
    /// Builds a query from the `4 × 4` control net and the global `(u, v)`.
    #[must_use]
    pub const fn new(control: [[f32; 3]; CONTROL_COUNT], u: f32, v: f32) -> BezierSurfaceQuery {
        BezierSurfaceQuery { control, u, v }
    }
}

/// One resolved query: the surface point, the unit normal and the degenerate
/// flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BezierSurfaceResult {
    /// Surface point at the queried `(u, v)`.
    pub point: [f32; 3],
    /// Unit surface normal, or `[0, 0, 1]` when the tangent frame collapsed.
    pub normal: [f32; 3],
    /// `true` when every `eps` step collapsed and the fallback normal was used.
    pub degenerate: bool,
}

/// Encodes one [`BezierSurfaceQuery`] into its `std430` [`GpuQuery`] slot,
/// flattening the sixteen control points into forty-eight scalars.
fn encode_query(q: &BezierSurfaceQuery) -> GpuQuery {
    let mut cp = [0.0f32; 48];
    for (k, point) in q.control.iter().enumerate() {
        cp[k * 3] = point[0];
        cp[k * 3 + 1] = point[1];
        cp[k * 3 + 2] = point[2];
    }
    GpuQuery { cp, u: q.u, v: q.v }
}

/// Decodes one packed [`GpuResult`] into the public [`BezierSurfaceResult`],
/// unpacking the integer flag into the reference's boolean shape.
fn decode_result(raw: &GpuResult) -> BezierSurfaceResult {
    BezierSurfaceResult {
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

/// A compiled, reusable single-patch Bézier-surface compute pipeline, twinning
/// the golden `ray_scene::bezier_surface`'s `BezierSurface::point` and
/// `BezierSurface::normal`.
pub struct GpuBezierSurface {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBezierSurface {
    /// Compiles the Bézier-surface kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBezierSurface {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bezier_surface"),
            source: ShaderSource::Wgsl(BEZIER_SURFACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bezier_surface_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bezier_surface_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bezier_surface_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBezierSurface {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`BezierSurfaceResult`]
    /// per input, in order.
    ///
    /// Each continuous field equals the reference within floating-point
    /// tolerance and the `degenerate` flag matches exactly. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BezierSurfaceQuery],
    ) -> Vec<BezierSurfaceResult> {
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
            label: Some("prism_volumetric_bezier_surface_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bezier_surface_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bezier_surface_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bezier_surface_bind_group"),
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
            label: Some("prism_volumetric_bezier_surface_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bezier_surface_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bezier_surface_pass"),
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
