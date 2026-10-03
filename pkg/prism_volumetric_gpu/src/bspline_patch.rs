//! `wgpu` compute twin of the bicubic uniform B-spline surface patch
//! ([`BsplinePatch`](prism_render_architecture::ray_scene::bspline_patch)).
//!
//! The `CPU` golden
//! [`BsplinePatch`](prism_render_architecture::ray_scene::bspline_patch) owns
//! the per-sample surface math for a bicubic uniform cubic B-spline over a
//! `4x4` approximating control net. It converts the net to Bezier form by the
//! uniform-knot basis change (`b0 = (c0 + 4*c1 + c2)/6`, `b1 = (2*c1 + c2)/3`,
//! `b2 = (c1 + 2*c2)/3`, `b3 = (c1 + 4*c2 + c3)/6`) applied first along each
//! `u` row then along each `v` column, after which the surface point and
//! analytic normal come from the Bezier evaluation path (`point`, `normal`,
//! `partial_u`, `partial_v`). [`GpuBsplinePatch`] is the on-device twin: one
//! thread per query reproduces the whole pipeline, so a passing real-device
//! parity test is direct evidence the ported kernel evaluates the same surface
//! and classifies the same degenerate frames the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `4x4` control net and a parameter pair `(u, v)`. The
//! twin reproduces the surface position `point(u, v)` and the unit analytic
//! surface normal `normal(u, v)` where the normal is the normalized
//! cross-product of the two partial derivatives, with the reference's four-step
//! interior nudge over `eps = 1e-3 * step` and the `[0, 0, 1]` fallback for a
//! fully collapsed tangent frame. A `degenerate` flag reports whether the
//! sample exhausted the nudge loop and fell back, matching the reference branch
//! for branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `sqrt`,
//! `+ - * /` and manual `vec3` arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no `round`, and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. The only comparison on an `f32`
//! is the ordered `len2 > 0.0` degeneracy guard the reference itself uses; no
//! exact `==` / `!=` on an `f32` appears.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of linear interpolations,
//! multiplies, adds and divides, so `CPU` and `GPU` evaluate the same closed
//! form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields while requiring the integer `degenerate` flag to match exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 [`BsplinePatch`](prism_render_architecture::ray_scene::bspline_patch)；
//! 无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` B-spline patch kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `evaluate_patch`
/// mirrors the `CPU` golden
/// [`BsplinePatch`](prism_render_architecture::ray_scene::bspline_patch): it
/// converts the net to Bezier form by the uniform-knot basis change, then reads
/// the surface point and analytic normal from the Bezier evaluation path. See
/// the module documentation for the algorithm.
const BSPLINE_PATCH_WGSL: &str = r#"
// Bicubic uniform B-spline surface twin: one thread per query converts the 4x4
// control net to Bezier form by the uniform-knot basis change, then reads the
// surface point and the unit analytic normal from the Bezier evaluation path.
// It mirrors the CPU golden ray_scene::bspline_patch, uses only the portable
// core-WGSL subset (clamp, sqrt and + - * / over manual vec3 math) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. The only
// f32 comparison is the ordered len2 > 0.0 degeneracy guard the reference uses;
// no exact == / != on an f32 appears.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::bspline_patch;
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The 4x4 B-spline control net, row-major (col along u, row along v); each
    // vec3 point occupies its own 16-byte-aligned slot.
    control: array<vec4<f32>, 16>,
    // The surface parameters (u, v) in the first two lanes; two pad lanes.
    uv: vec4<f32>,
}

struct Result {
    // Surface position P(u, v); a pad lane follows.
    point: vec3<f32>,
    pad0: f32,
    // Unit analytic surface normal; a pad lane follows.
    normal: vec3<f32>,
    pad1: f32,
    // 1 when the nudge loop was exhausted and the fallback normal was used.
    degenerate: u32,
    pad2: u32,
    pad3: u32,
    pad4: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Linear interpolation a + (b - a) * t, matching the reference `lerp3`.
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// Cross product a x b, matching the reference `cross3` component order.
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// Converts one uniform cubic B-spline span [c0, c1, c2, c3] into the four
// cubic Bezier control points of its central segment, matching the reference
// `span_to_bezier` term and operation order.
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

// Tensor-product B-spline-to-Bezier conversion: run span_to_bezier along each
// of the four u rows, then along each of the four v columns of the
// intermediate net, matching the reference `to_bezier`.
fn to_bezier(ctrl: array<vec3<f32>, 16>) -> array<vec3<f32>, 16> {
    var tmp: array<vec3<f32>, 16>;
    for (var row: u32 = 0u; row < 4u; row = row + 1u) {
        let base = row * 4u;
        let span = span_to_bezier(
            ctrl[base],
            ctrl[base + 1u],
            ctrl[base + 2u],
            ctrl[base + 3u],
        );
        tmp[base] = span[0];
        tmp[base + 1u] = span[1];
        tmp[base + 2u] = span[2];
        tmp[base + 3u] = span[3];
    }
    var out: array<vec3<f32>, 16>;
    for (var col: u32 = 0u; col < 4u; col = col + 1u) {
        let span = span_to_bezier(
            tmp[col],
            tmp[4u + col],
            tmp[8u + col],
            tmp[12u + col],
        );
        out[col] = span[0];
        out[4u + col] = span[1];
        out[8u + col] = span[2];
        out[12u + col] = span[3];
    }
    return out;
}

// Cubic Bezier point at t over four control points (three nested lerps),
// matching the reference `cubic_point`.
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

// Cubic Bezier derivative at t: 3 times the quadratic De Casteljau over the
// three adjacent control-point differences, matching the reference
// `cubic_deriv`.
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
    let q = lerp3(a, b, t);
    return q * 3.0;
}

// Surface point P(u, v): collapse each v-row by a cubic De Casteljau in u, then
// collapse the four results by a cubic De Casteljau in v.
fn bez_point(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_point(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_point(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_point(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_point(c0, c1, c2, c3, v);
}

// Partial derivative dP/du: each v-row contributes its u-tangent, blended by a
// cubic De Casteljau in v.
fn bez_partial_u(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_deriv(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_deriv(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_deriv(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_deriv(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_point(c0, c1, c2, c3, v);
}

// Partial derivative dP/dv: collapse each v-row to a point by a cubic De
// Casteljau in u, then differentiate the four points by a cubic derivative in
// v.
fn bez_partial_v(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_point(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_point(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_point(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_deriv(c0, c1, c2, c3, v);
}

// Unit surface normal with the reference four-step interior nudge: at each step
// evaluate the two partials, cross them and accept the first non-degenerate
// frame (len2 > 0.0); failing all four steps fall back to [0, 0, 1]. The w lane
// carries the degeneracy flag (1.0 on fallback, 0.0 otherwise).
fn bez_normal(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec4<f32> {
    for (var step: u32 = 0u; step < 4u; step = step + 1u) {
        let eps = 1.0e-3 * f32(step);
        let uu = clamp(u + eps, 0.0, 1.0);
        let vv = clamp(v + eps, 0.0, 1.0);
        let du = bez_partial_u(bez, uu, vv);
        let dv = bez_partial_v(bez, uu, vv);
        let n = cross3(du, dv);
        let len2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (len2 > 0.0) {
            let inv = 1.0 / sqrt(len2);
            return vec4<f32>(n * inv, 0.0);
        }
    }
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
}

@compute @workgroup_size(64)
fn evaluate_patch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var ctrl: array<vec3<f32>, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        ctrl[i] = q.control[i].xyz;
    }
    let bez = to_bezier(ctrl);
    let u = q.uv.x;
    let v = q.uv.y;
    let p = bez_point(bez, u, v);
    let nrm = bez_normal(bez, u, v);

    var out: Result;
    out.point = p;
    out.pad0 = 0.0;
    out.normal = nrm.xyz;
    out.pad1 = 0.0;
    out.degenerate = u32(nrm.w);
    out.pad2 = 0u;
    out.pad3 = 0u;
    out.pad4 = 0u;
    results[idx] = out;
}
"#;

/// One B-spline patch query: a `4x4` row-major control net plus the surface
/// parameters `(u, v)`, the same inputs the reference `BsplinePatch::point` and
/// `BsplinePatch::normal` consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsplinePatchQuery {
    /// The `4x4` B-spline control net, row-major (`control[row * 4 + col]`, with
    /// `col` along `u` and `row` along `v`).
    pub control: [[f32; 3]; 16],
    /// The surface parameter `u`, nominally in `[0, 1]`.
    pub u: f32,
    /// The surface parameter `v`, nominally in `[0, 1]`.
    pub v: f32,
}

impl BsplinePatchQuery {
    /// Builds a query from a `4x4` row-major control net and the parameters
    /// `(u, v)`.
    #[must_use]
    pub const fn new(control: [[f32; 3]; 16], u: f32, v: f32) -> BsplinePatchQuery {
        BsplinePatchQuery { control, u, v }
    }
}

/// The resolved answer for one query, mirroring the reference surface point and
/// analytic normal plus the degeneracy classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsplinePatchResult {
    /// The surface position `P(u, v)`, matching `BsplinePatch::point`.
    pub point: [f32; 3],
    /// The unit analytic surface normal, matching `BsplinePatch::normal`.
    pub normal: [f32; 3],
    /// `1` when the reference nudge loop was exhausted and the `[0, 0, 1]`
    /// fallback normal was used, `0` otherwise.
    pub degenerate: u32,
}

/// `repr(C)` `std430` layout of one packed query: `16` `vec4` slots holding the
/// control net (`xyz` plus a pad lane each) then one `vec4` slot carrying
/// `(u, v, pad, pad)` — `272` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The `4x4` control net, each point in the `xyz` of its own slot.
    control: [[f32; 4]; 16],
    /// `(u, v, pad, pad)` in one `vec4` slot.
    uv: [f32; 4],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &BsplinePatchQuery) -> GpuQuery {
        let mut control = [[0.0f32; 4]; 16];
        for (slot, point) in control.iter_mut().zip(query.control.iter()) {
            slot[0] = point[0];
            slot[1] = point[1];
            slot[2] = point[2];
            slot[3] = 0.0;
        }
        GpuQuery {
            control,
            uv: [query.u, query.v, 0.0, 0.0],
        }
    }
}

/// `repr(C)` `std430` layout of one result: a `vec3` point slot with a pad
/// lane, a `vec3` normal slot with a pad lane, then the `degenerate` flag with
/// three pad lanes — `48` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surface position.
    point: [f32; 3],
    /// Padding lane after the point.
    pad0: f32,
    /// Unit surface normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad1: f32,
    /// Degeneracy flag (`0` or `1`).
    degenerate: u32,
    /// Padding lane.
    pad2: u32,
    /// Padding lane.
    pad3: u32,
    /// Padding lane.
    pad4: u32,
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

/// A compiled, reusable B-spline patch compute pipeline.
pub struct GpuBsplinePatch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBsplinePatch {
    /// Compiles the B-spline patch kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBsplinePatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bspline_patch"),
            source: ShaderSource::Wgsl(BSPLINE_PATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bspline_patch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bspline_patch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bspline_patch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate_patch"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBsplinePatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query on-device and returns one [`BsplinePatchResult`]
    /// per input, in order.
    ///
    /// Each result equals the reference answers (`BsplinePatch::point` and
    /// `BsplinePatch::normal`) to within the tolerance documented on this
    /// module, with the integer `degenerate` flag matching exactly. An empty
    /// input returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BsplinePatchQuery],
    ) -> Vec<BsplinePatchResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bspline_patch_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bspline_patch_output"),
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
            label: Some("prism_volumetric_bspline_patch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bspline_patch_bind_group"),
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
            label: Some("prism_volumetric_bspline_patch_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bspline_patch_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bspline_patch_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`BsplinePatchResult`].
fn decode_result(raw: &GpuResult) -> BsplinePatchResult {
    BsplinePatchResult {
        point: raw.point,
        normal: raw.normal,
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
