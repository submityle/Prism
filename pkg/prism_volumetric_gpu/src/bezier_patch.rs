//! `wgpu` compute twin of the analytic *bicubic Bézier patch* surface
//! evaluation of the `CPU` golden path — `BezierPatch::{point, partial_u,
//! partial_v, normal}` in `prism_render_architecture::ray_scene::bezier_patch`.
//!
//! A bicubic Bézier patch is a `4x4` net of control points whose tensor-product
//! Bernstein basis sweeps a smoothly curved surface `P(u, v)` for `u, v` in
//! `[0, 1]`. The patch is the production proxy for the Utah teapot, trimmed
//! `NURBS` cages flattened to Bézier, displacement bases, and cloth/skin
//! patches. Both the surface point and its two partial derivatives are
//! evaluated with De Casteljau's algorithm, which is **pure linear
//! interpolation** — a cubic point is three nested `lerp`s and a cubic tangent
//! is `3·` a quadratic De Casteljau over adjacent control-point differences —
//! so no transcendental basis function ever appears and the evaluation is
//! reproducible on the `GPU` using only add/sub/mul and a single `sqrt` for the
//! normal. The surface normal is the cross product `∂P/∂u × ∂P/∂v`, normalized.
//! [`GpuBezierPatch`] is the on-device twin: each thread evaluates one patch at
//! one `(u, v)` and writes the surface point, both partials, the unit normal
//! and a degeneracy flag.
//!
//! # What is twinned
//!
//! Each thread reads one [`BezierPatchQuery`] — the row-major `4x4` control net
//! (`control[row * 4 + col]`, `col` along `u`, `row` along `v`) and the sample
//! `(u, v)` — and writes one [`BezierPatchResult`] holding the surface
//! `point`, the partials `partial_u` and `partial_v`, the unit `normal`, and a
//! `degenerate` flag. The point is a cubic De Casteljau in `u` per `v`-row
//! collapsed by a cubic De Casteljau in `v`; `partial_u` differentiates each
//! row in `u` then blends in `v`; `partial_v` collapses each row in `u` then
//! differentiates in `v`. The normal is `cross(partial_u, partial_v)`: when the
//! cross product collapses below the degeneracy guard (a cusp or a flat
//! control net) the twin sets `degenerate = 1` and zeroes `normal`, otherwise
//! it reports the unit normal and `degenerate = 0`.
//!
//! # What stays on the host
//!
//! The patch tessellation into an `IndexedBilinearPatchMesh`, the welded vertex
//! grid, the control-net `AABB` and the `BVH` build/traversal all stay on the
//! host; the device sees only the stateless, fixed-width per-sample
//! evaluation, one `(u, v)` at a time, so a storage buffer is never
//! zero-sized.
//!
//! # Correctness model
//!
//! The evaluation threads through nested linear interpolations, a cross
//! product and one normalization `sqrt`, so the `CPU` and `GPU` are not
//! bit-exact: a fused multiply-add or a differently ordered sum may land a few
//! units in the last place from the scalar reference. The parity test asserts
//! the discrete `degenerate` flag exactly and each continuous component
//! (`point`, `partial_u`, `partial_v`, `normal`) within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. Fixtures and the random sweep reject-sample away from
//! the one branch cliff — the cross product near zero (a degenerate tangent
//! frame) — so the `CPU` and `GPU` never pick different sides of the guard.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `dot`,
//! `cross`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round`/`ceil`/`%`,
//! and no `f64`/`u64`/`u16`/`i64`/`i16`. The degeneracy test uses an ordered
//! `len2 <= eps` comparison, never a bare float equality. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::bezier_patch`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` bicubic-Bézier evaluation kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `BezierPatch::{point, partial_u, partial_v,
/// normal}`, one `(u, v)` sample per thread.
const BEZIER_PATCH_WGSL: &str = r#"
// Analytic bicubic Bézier patch evaluation twinned from the CPU golden path.
// Each thread evaluates one patch at one (u, v) and writes the surface point,
// both partial derivatives, the unit normal and a degeneracy flag; the patch
// tessellation, mesh weld and BVH build/traversal stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::bezier_patch；无第三方
// 引擎源码或衍生代码。

// Squared-length guard separating a usable tangent frame from the degenerate
// cross-product branch; the parity harness keeps real queries well clear of it.
const N_EPS: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Row-major 4x4 control net flattened to 48 scalars: point i occupies
    // control[3*i], control[3*i + 1], control[3*i + 2]; point i is control net
    // slot i = row * 4 + col.
    control: array<f32, 48>,
    // Sample parameters in [0, 1].
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
    // Partial derivative dP/du.
    dux: f32,
    duy: f32,
    duz: f32,
    // Partial derivative dP/dv.
    dvx: f32,
    dvy: f32,
    dvz: f32,
    // Unit surface normal; zero when degenerate.
    nx: f32,
    ny: f32,
    nz: f32,
    // Degeneracy flag: 1u when the cross product collapsed, else 0u.
    degenerate: u32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outputs>;

// Linear interpolation a + t*(b - a).
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
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
    out.dux = 0.0;
    out.duy = 0.0;
    out.duz = 0.0;
    out.dvx = 0.0;
    out.dvy = 0.0;
    out.dvz = 0.0;
    out.nx = 0.0;
    out.ny = 0.0;
    out.nz = 0.0;
    out.degenerate = 0u;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;

    // Unpack the 16 control points.
    var cp: array<vec3<f32>, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        let base = i * 3u;
        cp[i] = vec3<f32>(q.control[base], q.control[base + 1u], q.control[base + 2u]);
    }

    let u = q.u;
    let v = q.v;

    // Each v-row collapsed to a point by a cubic De Casteljau in u.
    let c0 = cubic_point(cp[0], cp[1], cp[2], cp[3], u);
    let c1 = cubic_point(cp[4], cp[5], cp[6], cp[7], u);
    let c2 = cubic_point(cp[8], cp[9], cp[10], cp[11], u);
    let c3 = cubic_point(cp[12], cp[13], cp[14], cp[15], u);
    // Surface point: collapse the four row points by a cubic De Casteljau in v.
    let point = cubic_point(c0, c1, c2, c3, v);

    // dP/du: each row's u-tangent, blended by a cubic De Casteljau in v.
    let du0 = cubic_deriv(cp[0], cp[1], cp[2], cp[3], u);
    let du1 = cubic_deriv(cp[4], cp[5], cp[6], cp[7], u);
    let du2 = cubic_deriv(cp[8], cp[9], cp[10], cp[11], u);
    let du3 = cubic_deriv(cp[12], cp[13], cp[14], cp[15], u);
    let pu = cubic_point(du0, du1, du2, du3, v);

    // dP/dv: the four row points differentiated by a cubic derivative in v.
    let pv = cubic_deriv(c0, c1, c2, c3, v);

    // Normal: cross(dP/du, dP/dv), normalized, with a degeneracy guard.
    let ncross = cross(pu, pv);
    let len2 = dot(ncross, ncross);

    out.px = point.x;
    out.py = point.y;
    out.pz = point.z;
    out.dux = pu.x;
    out.duy = pu.y;
    out.duz = pu.z;
    out.dvx = pv.x;
    out.dvy = pv.y;
    out.dvz = pv.z;

    if (len2 <= N_EPS) {
        out.degenerate = 1u;
        out.nx = 0.0;
        out.ny = 0.0;
        out.nz = 0.0;
    } else {
        let normal = ncross * (1.0 / sqrt(len2));
        out.degenerate = 0u;
        out.nx = normal.x;
        out.ny = normal.y;
        out.nz = normal.z;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BEZIER_PATCH_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the row-major `4x4` control net flattened to `48` scalars plus the sample
/// `(u, v)` and two pad words, `52` `f32` words at a `208`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Row-major `4x4` control net flattened to `48` scalars (point `i` at
    /// `[3i, 3i + 1, 3i + 2]`).
    control: [f32; 48],
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
/// struct: the surface point, both partials, the unit normal, the degeneracy
/// flag and three pad words, `16` words at a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surface point `x`.
    px: f32,
    /// Surface point `y`.
    py: f32,
    /// Surface point `z`.
    pz: f32,
    /// Partial `dP/du` `x`.
    dux: f32,
    /// Partial `dP/du` `y`.
    duy: f32,
    /// Partial `dP/du` `z`.
    duz: f32,
    /// Partial `dP/dv` `x`.
    dvx: f32,
    /// Partial `dP/dv` `y`.
    dvy: f32,
    /// Partial `dP/dv` `z`.
    dvz: f32,
    /// Unit normal `x` (zero when degenerate).
    nx: f32,
    /// Unit normal `y` (zero when degenerate).
    ny: f32,
    /// Unit normal `z` (zero when degenerate).
    nz: f32,
    /// Degeneracy flag (`1` = cross product collapsed, `0` = usable frame).
    degenerate: u32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One bicubic Bézier patch evaluation query: the control net and the sample.
///
/// `control` is the row-major `4x4` net, `control[row * 4 + col]`, with `col`
/// running along the `u` parameter and `row` along the `v` parameter; the four
/// corners interpolate the net (`P(0, 0) = control[0]`, `P(1, 0) = control[3]`,
/// `P(0, 1) = control[12]`, `P(1, 1) = control[15]`). `u` and `v` are the
/// sample parameters, expected in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BezierPatchQuery {
    /// Row-major `4x4` control net, `control[row * 4 + col]`.
    pub control: [[f32; 3]; 16],
    /// Sample `u` parameter (expected in `[0, 1]`).
    pub u: f32,
    /// Sample `v` parameter (expected in `[0, 1]`).
    pub v: f32,
}

impl BezierPatchQuery {
    /// Builds a query from a row-major `4x4` control net and a sample `(u, v)`.
    #[must_use]
    pub const fn new(control: [[f32; 3]; 16], u: f32, v: f32) -> BezierPatchQuery {
        BezierPatchQuery { control, u, v }
    }
}

/// One evaluated bicubic Bézier sample.
///
/// `point` is the surface position `P(u, v)`, `partial_u` and `partial_v` are
/// the two tensor-product partial derivatives, and `normal` is the unit surface
/// normal `cross(partial_u, partial_v)` oriented along the control-net winding.
/// When the tangent frame collapses (`cross` near zero) `degenerate` is `1` and
/// `normal` is zeroed; otherwise `degenerate` is `0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BezierPatchResult {
    /// Surface point `P(u, v)`.
    pub point: [f32; 3],
    /// Partial derivative `dP/du`.
    pub partial_u: [f32; 3],
    /// Partial derivative `dP/dv`.
    pub partial_v: [f32; 3],
    /// Unit surface normal; zero when `degenerate` is `1`.
    pub normal: [f32; 3],
    /// Degeneracy flag (`1` = collapsed tangent frame, `0` = usable).
    pub degenerate: u32,
}

/// Encodes one [`BezierPatchQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &BezierPatchQuery) -> GpuQuery {
    let mut control = [0.0f32; 48];
    for (i, p) in q.control.iter().enumerate() {
        let base = i * 3;
        control[base] = p[0];
        control[base + 1] = p[1];
        control[base + 2] = p[2];
    }
    GpuQuery {
        control,
        u: q.u,
        v: q.v,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`BezierPatchResult`].
fn decode_result(raw: &GpuResult) -> BezierPatchResult {
    BezierPatchResult {
        point: [raw.px, raw.py, raw.pz],
        partial_u: [raw.dux, raw.duy, raw.duz],
        partial_v: [raw.dvx, raw.dvy, raw.dvz],
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

/// A compiled, reusable bicubic-Bézier evaluation compute pipeline, twinning
/// the `CPU` golden `BezierPatch::{point, partial_u, partial_v, normal}` of
/// `prism_render_architecture::ray_scene::bezier_patch`.
pub struct GpuBezierPatch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBezierPatch {
    /// Compiles the bicubic-Bézier evaluation kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBezierPatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bezier_patch_module"),
            source: ShaderSource::Wgsl(BEZIER_PATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bezier_patch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bezier_patch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bezier_patch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBezierPatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`BezierPatchResult`]
    /// per input, in order.
    ///
    /// The reported sample matches the reference: the discrete `degenerate`
    /// flag exactly and the continuous fields to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BezierPatchQuery],
    ) -> Vec<BezierPatchResult> {
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
            label: Some("prism_volumetric_bezier_patch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bezier_patch_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bezier_patch_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bezier_patch_bind_group"),
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
            label: Some("prism_volumetric_bezier_patch_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bezier_patch_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bezier_patch_pass"),
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
