//! `wgpu` compute twin of the analytic *uniform bicubic B-spline surface*
//! evaluation of the `CPU` golden path — `BsplineSurface::{point, normal}` in
//! `prism_render_architecture::ray_scene::bspline_surface`.
//!
//! A uniform bicubic B-spline surface is an `R x C` grid of control handles,
//! partitioned into `(C - 3) x (R - 3)` overlapping cubic spans that share
//! their boundary control rows and columns. The surface stays inside the
//! convex hull of its net, interpolates no control point, and is globally
//! `C²` continuous. The whole surface is parameterised by `(u, v)` in
//! `[0, 1]²`: each parameter is scaled by its span count, the integer part
//! selects the span and the fractional part is the local span parameter. The
//! selected 4x4 window is then converted from the uniform B-spline basis into
//! the equivalent cubic Bézier net — `b0 = (c0 + 4·c1 + c2)/6`,
//! `b1 = (2·c1 + c2)/3`, `b2 = (c1 + 2·c2)/3`, `b3 = (c1 + 4·c2 + c3)/6` along
//! rows then columns — a **purely linear** conversion with no transcendental
//! basis function. The surface point and its partial derivatives are then
//! evaluated with De Casteljau's algorithm (nested `lerp`s), and the normal is
//! the normalized cross product `∂P/∂u × ∂P/∂v`.
//!
//! # What is twinned
//!
//! The control net is a single shared storage buffer (`rows x cols` points,
//! flattened to three scalars each) with `rows`, `cols`, `u_spans` and
//! `v_spans` in the uniform block. Each thread reads one
//! [`BsplineSurfaceQuery`] — the global sample `(u, v)` — and writes one
//! [`BsplineSurfaceResult`] holding the surface `point`, the unit `normal`,
//! and a `degenerate` flag. The thread locates the span and local parameter
//! along each axis (clamp, scale by span count, `floor` for the integer span,
//! fractional remainder for the local parameter), extracts the 4x4 window,
//! converts it to Bézier form, evaluates the point by nested De Casteljau and
//! the normal by `cross(∂u, ∂v)`. The golden normal nudges the local sample a
//! few steps toward the interior when the tangent frame collapses; the twin
//! replicates that nudge loop exactly. When all nudges collapse (a cusp or a
//! degenerate net) the twin sets `degenerate = 1` and reports the deterministic
//! fallback normal `[0, 0, 1]`, otherwise `degenerate = 0`.
//!
//! # What stays on the host
//!
//! The surface tessellation into an `IndexedBilinearPatchMesh`, the welded
//! vertex grid, the control-net `AABB`, the construction validation and the
//! `BVH` build/traversal all stay on the host; the device sees only the
//! stateless, fixed-width per-sample evaluation against a shared control net,
//! one `(u, v)` at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The evaluation threads through nested linear interpolations, a basis
//! conversion, a cross product and one normalization `sqrt`, so the `CPU` and
//! `GPU` are not bit-exact: a fused multiply-add or a differently ordered sum
//! may land a few units in the last place from the scalar reference. The
//! parity test asserts the discrete `degenerate` flag exactly and each
//! continuous component (`point`, `normal`) within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. Fixtures and the random sweep reject-sample away from
//! the branch cliff — the cross product near zero (a degenerate tangent
//! frame) — so the `CPU` and `GPU` never pick different sides of the guard,
//! except for the axis-aligned collinear fixtures where the cross components
//! are exactly zero on both sides.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `dot`,
//! `cross`, `floor`, `clamp`, `min`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `round`/`%`, and no `f64`/`u64`/`u16`/`i64`/`i16`. The degeneracy test
//! uses an ordered `len2 > 0.0` comparison, never a bare float equality. It
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bspline_surface`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` uniform bicubic B-spline surface evaluation kernel,
/// embedded inline so the twin ships as a single source file. The single entry
/// point `solve` mirrors the `CPU` golden `BsplineSurface::{point, normal}`,
/// one `(u, v)` sample per thread against a shared control net.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bspline_surface`；无第三方引擎源码或衍生代码。
const BSPLINE_SURFACE_WGSL: &str = r#"
// Uniform bicubic B-spline surface evaluation, one (u, v) sample per thread.
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::bspline_surface。

// Squared-length guard separating a usable tangent frame from the degenerate
// cross-product branch; mirrors the golden strict `len2 > 0.0` test.
const N_EPS: f32 = 0.0;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    // Control rows (along v); always >= 4.
    rows: u32,
    // Control columns (along u); always >= 4.
    cols: u32,
    // Cubic span count along u (cols - 3); always >= 1.
    u_spans: u32,
    // Cubic span count along v (rows - 3); always >= 1.
    v_spans: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Global sample parameters in [0, 1].
    u: f32,
    v: f32,
    pad0: f32,
    pad1: f32,
}

struct Outputs {
    // Surface point P(u, v).
    px: f32,
    py: f32,
    pz: f32,
    // Unit surface normal; [0, 0, 1] fallback when degenerate.
    nx: f32,
    ny: f32,
    nz: f32,
    // Degeneracy flag: 1u when every nudge collapsed, else 0u.
    degenerate: u32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flattened control net: point i occupies control[3*i], control[3*i+1],
// control[3*i+2]; point i is net slot i = row * cols + col.
@group(0) @binding(1) var<storage, read> control: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Outputs>;

// Reads control net point at flat slot `slot`.
fn control_point(slot: u32) -> vec3<f32> {
    let base = slot * 3u;
    return vec3<f32>(control[base], control[base + 1u], control[base + 2u]);
}

// Linear interpolation a + t*(b - a).
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// Converts one uniform cubic B-spline span into the four cubic Bézier control
// points of its central segment.
fn span_to_bezier(
    c0: vec3<f32>,
    c1: vec3<f32>,
    c2: vec3<f32>,
    c3: vec3<f32>,
) -> array<vec3<f32>, 4> {
    let sixth = 1.0 / 6.0;
    let third = 1.0 / 3.0;
    let b0 = (c0 + c1 * 4.0 + c2) * sixth;
    let b1 = (c1 * 2.0 + c2) * third;
    let b2 = (c1 + c2 * 2.0) * third;
    let b3 = (c1 + c2 * 4.0 + c3) * sixth;
    return array<vec3<f32>, 4>(b0, b1, b2, b3);
}

// Cubic Bézier point over four control points: three nested lerps.
fn cubic_point(
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    p3: vec3<f32>,
    t: f32,
) -> vec3<f32> {
    let a = lerp3(p0, p1, t);
    let b = lerp3(p1, p2, t);
    let c = lerp3(p2, p3, t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    return lerp3(d, e, t);
}

// Cubic Bézier derivative: 3x the quadratic De Casteljau over the three
// adjacent control-point differences.
fn cubic_deriv(
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    p3: vec3<f32>,
    t: f32,
) -> vec3<f32> {
    let d0 = p1 - p0;
    let d1 = p2 - p1;
    let d2 = p3 - p2;
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let qd = lerp3(a, b, t);
    return qd * 3.0;
}

// Bézier surface point from a 4x4 row-major Bézier net.
fn bez_point(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_point(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_point(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_point(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_point(c0, c1, c2, c3, v);
}

// Partial derivative dP/du from a 4x4 row-major Bézier net.
fn bez_partial_u(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let d0 = cubic_deriv(bez[0], bez[1], bez[2], bez[3], u);
    let d1 = cubic_deriv(bez[4], bez[5], bez[6], bez[7], u);
    let d2 = cubic_deriv(bez[8], bez[9], bez[10], bez[11], u);
    let d3 = cubic_deriv(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_point(d0, d1, d2, d3, v);
}

// Partial derivative dP/dv from a 4x4 row-major Bézier net.
fn bez_partial_v(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_point(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_point(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_point(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_deriv(c0, c1, c2, c3, v);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Outputs;
    out.px = 0.0;
    out.py = 0.0;
    out.pz = 0.0;
    out.nx = 0.0;
    out.ny = 0.0;
    out.nz = 1.0;
    out.degenerate = 1u;
    out.pad0 = 0.0;

    let cols = params.cols;
    let u_spans = params.u_spans;
    let v_spans = params.v_spans;

    // locate(u): clamp, scale by span count, floor for span, remainder local.
    let cu = clamp(q.u, 0.0, 1.0);
    let scaled_u = cu * f32(u_spans);
    let last_u = u_spans - 1u;
    var span_u = u32(floor(scaled_u));
    span_u = min(span_u, last_u);
    let local_u = scaled_u - f32(span_u);

    // locate(v): identical construction along v.
    let cv = clamp(q.v, 0.0, 1.0);
    let scaled_v = cv * f32(v_spans);
    let last_v = v_spans - 1u;
    var span_v = u32(floor(scaled_v));
    span_v = min(span_v, last_v);
    let local_v = scaled_v - f32(span_v);

    // Extract the 4x4 B-spline window: window[wr*4+wc] =
    // control[(span_v + wr) * cols + (span_u + wc)].
    var win: array<vec3<f32>, 16>;
    for (var wr: u32 = 0u; wr < 4u; wr = wr + 1u) {
        let row = (span_v + wr) * cols + span_u;
        for (var wc: u32 = 0u; wc < 4u; wc = wc + 1u) {
            win[wr * 4u + wc] = control_point(row + wc);
        }
    }

    // Convert B-spline net to Bézier: span_to_bezier along rows, then columns.
    var tmp: array<vec3<f32>, 16>;
    for (var row: u32 = 0u; row < 4u; row = row + 1u) {
        let base = row * 4u;
        let span = span_to_bezier(win[base], win[base + 1u], win[base + 2u], win[base + 3u]);
        tmp[base] = span[0];
        tmp[base + 1u] = span[1];
        tmp[base + 2u] = span[2];
        tmp[base + 3u] = span[3];
    }
    var bez: array<vec3<f32>, 16>;
    for (var col: u32 = 0u; col < 4u; col = col + 1u) {
        let span = span_to_bezier(tmp[col], tmp[col + 4u], tmp[col + 8u], tmp[col + 12u]);
        bez[col] = span[0];
        bez[col + 4u] = span[1];
        bez[col + 8u] = span[2];
        bez[col + 12u] = span[3];
    }

    // Surface point at the local parameters.
    let point = bez_point(bez, local_u, local_v);
    out.px = point.x;
    out.py = point.y;
    out.pz = point.z;

    // Normal: nudge the local sample toward the interior until a non-degenerate
    // tangent frame is found; mirrors the golden `BezierPatch::normal` loop.
    for (var step: i32 = 0; step < 4; step = step + 1) {
        let eps = 1.0e-3 * f32(step);
        let uu = clamp(local_u + eps, 0.0, 1.0);
        let vv = clamp(local_v + eps, 0.0, 1.0);
        let du = bez_partial_u(bez, uu, vv);
        let dv = bez_partial_v(bez, uu, vv);
        let n = cross(du, dv);
        let len2 = dot(n, n);
        if (len2 > N_EPS) {
            let normal = n * (1.0 / sqrt(len2));
            out.nx = normal.x;
            out.ny = normal.y;
            out.nz = normal.z;
            out.degenerate = 0u;
            break;
        }
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count, the control-net shape
/// (`rows`, `cols`), the span counts (`u_spans`, `v_spans`) and three pad words
/// filling a `std140`-aligned uniform struct matching `Params` in
/// [`BSPLINE_SURFACE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Control rows (along `v`); always `>= 4`.
    rows: u32,
    /// Control columns (along `u`); always `>= 4`.
    cols: u32,
    /// Cubic span count along `u` (`cols - 3`); always `>= 1`.
    u_spans: u32,
    /// Cubic span count along `v` (`rows - 3`); always `>= 1`.
    v_spans: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the global sample `(u, v)` plus two pad words, `4` `f32` words at a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Sample `u` in `[0, 1]`.
    u: f32,
    /// Sample `v` in `[0, 1]`.
    v: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outputs`
/// struct: the surface point, the unit normal, the degeneracy flag and one pad
/// word, `8` words at a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surface point `x`.
    px: f32,
    /// Surface point `y`.
    py: f32,
    /// Surface point `z`.
    pz: f32,
    /// Unit normal `x` (`[0, 0, 1]` fallback when degenerate).
    nx: f32,
    /// Unit normal `y`.
    ny: f32,
    /// Unit normal `z`.
    nz: f32,
    /// Degeneracy flag (`1` = every nudge collapsed, `0` = usable frame).
    degenerate: u32,
    /// Padding word.
    pad0: f32,
}

/// One uniform bicubic B-spline surface evaluation query: a global sample.
///
/// `u` and `v` are the global surface parameters, expected in `[0, 1]`; the
/// control net is supplied once per [`GpuBsplineSurface::evaluate`] call and is
/// shared by every query in the batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsplineSurfaceQuery {
    /// Global sample `u` parameter (expected in `[0, 1]`).
    pub u: f32,
    /// Global sample `v` parameter (expected in `[0, 1]`).
    pub v: f32,
}

impl BsplineSurfaceQuery {
    /// Builds a query from a global sample `(u, v)`.
    #[must_use]
    pub const fn new(u: f32, v: f32) -> BsplineSurfaceQuery {
        BsplineSurfaceQuery { u, v }
    }
}

/// One evaluated uniform bicubic B-spline surface sample.
///
/// `point` is the surface position `P(u, v)` and `normal` is the unit surface
/// normal `cross(∂P/∂u, ∂P/∂v)` oriented along the control-net winding. When
/// the tangent frame collapses after every interior nudge, `degenerate` is `1`
/// and `normal` is the deterministic fallback `[0, 0, 1]`; otherwise
/// `degenerate` is `0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsplineSurfaceResult {
    /// Surface point `P(u, v)`.
    pub point: [f32; 3],
    /// Unit surface normal; `[0, 0, 1]` fallback when `degenerate` is `1`.
    pub normal: [f32; 3],
    /// Degeneracy flag (`1` = collapsed tangent frame, `0` = usable).
    pub degenerate: u32,
}

/// Encodes one [`BsplineSurfaceQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &BsplineSurfaceQuery) -> GpuQuery {
    GpuQuery {
        u: q.u,
        v: q.v,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`BsplineSurfaceResult`].
fn decode_result(raw: &GpuResult) -> BsplineSurfaceResult {
    BsplineSurfaceResult {
        point: [raw.px, raw.py, raw.pz],
        normal: [raw.nx, raw.ny, raw.nz],
        degenerate: raw.degenerate,
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

/// A compiled, reusable uniform bicubic B-spline surface evaluation compute
/// pipeline, twinning the `CPU` golden `BsplineSurface::{point, normal}` of
/// `prism_render_architecture::ray_scene::bspline_surface`.
pub struct GpuBsplineSurface {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBsplineSurface {
    /// Compiles the uniform bicubic B-spline surface evaluation kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBsplineSurface {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bspline_surface_module"),
            source: ShaderSource::Wgsl(BSPLINE_SURFACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bspline_surface_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bspline_surface_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bspline_surface_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBsplineSurface {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` against the shared row-major control
    /// net `control` (`rows x cols` points, `control[row * cols + col]`) and
    /// returns one [`BsplineSurfaceResult`] per input, in order.
    ///
    /// The caller guarantees `rows >= 4`, `cols >= 4` and
    /// `control.len() == rows * cols` (the same contract the golden
    /// `BsplineSurface::new` enforces). The reported sample matches the
    /// reference: the discrete `degenerate` flag exactly and the continuous
    /// fields to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        control: &[[f32; 3]],
        rows: u32,
        cols: u32,
        queries: &[BsplineSurfaceQuery],
    ) -> Vec<BsplineSurfaceResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            rows,
            cols,
            u_spans: cols - 3,
            v_spans: rows - 3,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bspline_surface_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let mut control_flat: Vec<f32> = Vec::with_capacity(control.len() * 3);
        for p in control {
            control_flat.push(p[0]);
            control_flat.push(p[1]);
            control_flat.push(p[2]);
        }
        let control_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bspline_surface_control"),
            contents: bytemuck::cast_slice(&control_flat),
            usage: BufferUsages::STORAGE,
        });

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bspline_surface_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bspline_surface_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bspline_surface_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: control_buf.as_entire_binding(),
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
            label: Some("prism_volumetric_bspline_surface_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bspline_surface_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bspline_surface_pass"),
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
