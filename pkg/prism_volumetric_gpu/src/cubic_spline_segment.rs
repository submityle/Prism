//! `wgpu` compute twin of the single-segment cubic-spline evaluators from the
//! `CPU` golden `prism_math::curve::spline`.
//!
//! The golden file defines six stateless, purely polynomial closed forms over
//! an interpolatable value (here instantiated at `Vec3`): the cubic Hermite
//! basis and its `t`-derivative, the uniform Catmull-Rom segment and its
//! derivative (both expressed through Hermite with half-difference tangents),
//! and the cubic Bézier (Bernstein) segment and its derivative. This module
//! ports every one of them onto the device behind an operation id, so one
//! thread resolves one query: a passing real-device parity test is direct
//! evidence the ported kernel reproduces the reference polynomial, not merely
//! that the shader compiles.
//!
//! This is distinct from the already-twinned multi-segment `spline` module
//! (which ports `prism_render_architecture::particle::spline`'s piecewise
//! `Spline::point_at`): here each query is one isolated Hermite / Catmull-Rom /
//! Bézier basis segment.
//!
//! # What is twinned
//!
//! For each query the kernel selects one evaluator by `op_id` and reproduces it
//! verbatim in operator order (with `t2 = t*t`, `t3 = t2*t`, `u = 1 - t`):
//!
//! * `0` `hermite(p0, m0, p1, m1, t)`
//! * `1` `hermite_tangent(p0, m0, p1, m1, t)`
//! * `2` `catmull_rom(p0, p1, p2, p3, t)`
//! * `3` `catmull_rom_tangent(p0, p1, p2, p3, t)`
//! * `4` `bezier_cubic(p0, p1, p2, p3, t)`
//! * `5` `bezier_cubic_tangent(p0, p1, p2, p3, t)`
//!
//! The four input vectors `a0..a3` carry the segment control data; for the
//! Hermite ops they are `(p0, m0, p1, m1)`, for the Catmull-Rom and Bézier ops
//! they are `(p0, p1, p2, p3)`. Any `op_id > 5` is rejected (`valid = 0`,
//! `value = 0`).
//!
//! # Correctness model
//!
//! Every evaluator is a finite polynomial of pure multiplies, adds and
//! subtracts, so the result threads through operators a `GPU` may contract;
//! `CPU` and `GPU` are therefore not necessarily bit-exact. Each component of
//! the valid `value` vector is compared with an `abs <= 1e-4 || rel <= 1e-3`
//! tolerance (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared
//! exactly. There is no degenerate numerical branch (no division, no
//! transcendental), so the only invalid case is an out-of-range `op_id`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - *`, `select`,
//! `vec3<f32>` construction and unsigned index arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no
//! division. Operation selection uses unsigned `u32` equality only; there is no
//! `f32` equality anywhere. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::curve::spline`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cubic-spline-segment kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the six `CPU` golden evaluators in `prism_math::curve::spline`; see the
/// module documentation for the closed forms.
const CUBIC_SPLINE_SEGMENT_WGSL: &str = r#"
// Cubic-spline-segment twin: one thread per query selects one of six cubic
// evaluators by op_id and reproduces it. It uses only the portable core-WGSL
// subset (+ - *, select, vec3<f32> construction plus unsigned index math), has
// no loop, no division and no transcendental, so it provably terminates. The
// op_id dispatch uses u32 equality only; there is no f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation id in 0..=5; any other value is rejected as invalid.
    op_id: u32,
    // First control vector: p0 for every op.
    a0x: f32,
    a0y: f32,
    a0z: f32,
    // Second control vector: m0 (Hermite) or p1 (Catmull-Rom / Bezier).
    a1x: f32,
    a1y: f32,
    a1z: f32,
    // Third control vector: p1 (Hermite) or p2 (Catmull-Rom / Bezier).
    a2x: f32,
    a2y: f32,
    a2z: f32,
    // Fourth control vector: m1 (Hermite) or p3 (Catmull-Rom / Bezier).
    a3x: f32,
    a3y: f32,
    a3z: f32,
    // Segment parameter.
    t: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Evaluated vector when valid, else (0, 0, 0).
    vx: f32,
    vy: f32,
    vz: f32,
    // 1 when op_id is in range, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Cubic Hermite basis: p0*h00 + m0*h10 + p1*h01 + m1*h11.
fn hermite(p0: vec3<f32>, m0: vec3<f32>, p1: vec3<f32>, m1: vec3<f32>, t: f32) -> vec3<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;
}

// Derivative of hermite with respect to t.
fn hermite_tangent(p0: vec3<f32>, m0: vec3<f32>, p1: vec3<f32>, m1: vec3<f32>, t: f32) -> vec3<f32> {
    let t2 = t * t;
    let h00 = 6.0 * t2 - 6.0 * t;
    let h10 = 3.0 * t2 - 4.0 * t + 1.0;
    let h01 = -6.0 * t2 + 6.0 * t;
    let h11 = 3.0 * t2 - 2.0 * t;
    return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;
}

// Uniform Catmull-Rom from p1 to p2, with half-difference tangents.
fn catmull_rom(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let m1 = (p2 - p0) * 0.5;
    let m2 = (p3 - p1) * 0.5;
    return hermite(p1, m1, p2, m2, t);
}

// Derivative of catmull_rom with respect to t.
fn catmull_rom_tangent(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let m1 = (p2 - p0) * 0.5;
    let m2 = (p3 - p1) * 0.5;
    return hermite_tangent(p1, m1, p2, m2, t);
}

// Cubic Bezier in Bernstein form.
fn bezier_cubic(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let u = 1.0 - t;
    let uu = u * u;
    let tt = t * t;
    let b0 = uu * u;
    let b1 = 3.0 * uu * t;
    let b2 = 3.0 * u * tt;
    let b3 = tt * t;
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
}

// Derivative of bezier_cubic with respect to t.
fn bezier_cubic_tangent(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let u = 1.0 - t;
    let c0 = 3.0 * u * u;
    let c1 = 6.0 * u * t;
    let c2 = 3.0 * t * t;
    return (p1 - p0) * c0 + (p2 - p1) * c1 + (p3 - p2) * c2;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a0 = vec3<f32>(q.a0x, q.a0y, q.a0z);
    let a1 = vec3<f32>(q.a1x, q.a1y, q.a1z);
    let a2 = vec3<f32>(q.a2x, q.a2y, q.a2z);
    let a3 = vec3<f32>(q.a3x, q.a3y, q.a3z);
    let t = q.t;
    let op = q.op_id;

    // Operation selection via u32 equality only; no f32 equality anywhere.
    var value = vec3<f32>(0.0, 0.0, 0.0);
    var ok = true;
    if (op == 0u) {
        value = hermite(a0, a1, a2, a3, t);
    } else if (op == 1u) {
        value = hermite_tangent(a0, a1, a2, a3, t);
    } else if (op == 2u) {
        value = catmull_rom(a0, a1, a2, a3, t);
    } else if (op == 3u) {
        value = catmull_rom_tangent(a0, a1, a2, a3, t);
    } else if (op == 4u) {
        value = bezier_cubic(a0, a1, a2, a3, t);
    } else if (op == 5u) {
        value = bezier_cubic_tangent(a0, a1, a2, a3, t);
    } else {
        ok = false;
    }

    let zero = vec3<f32>(0.0, 0.0, 0.0);
    let r = select(zero, value, ok);
    var out: Result;
    out.vx = r.x;
    out.vy = r.y;
    out.vz = r.z;
    out.valid = select(0u, 1u, ok);
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// The op id, four flattened control vectors and `t` are padded to `16` `f32`
/// words (`64` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op_id: u32,
    a0x: f32,
    a0y: f32,
    a0z: f32,
    a1x: f32,
    a1y: f32,
    a1z: f32,
    a2x: f32,
    a2y: f32,
    a2z: f32,
    a3x: f32,
    a3y: f32,
    a3z: f32,
    t: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the three evaluated components and the validity flag — `4` words
/// (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    vx: f32,
    vy: f32,
    vz: f32,
    valid: u32,
}

/// One cubic-spline-segment query: the operation id, four control vectors and
/// the segment parameter `t`.
///
/// For the Hermite ops (`0`, `1`) the vectors are `(p0, m0, p1, m1)`; for the
/// Catmull-Rom and Bézier ops (`2..=5`) they are `(p0, p1, p2, p3)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubicSplineSegmentQuery {
    /// Operation id: `0` `hermite`, `1` `hermite_tangent`, `2` `catmull_rom`,
    /// `3` `catmull_rom_tangent`, `4` `bezier_cubic`, `5` `bezier_cubic_tangent`.
    pub op_id: u32,
    /// First control vector (`p0`).
    pub a0: [f32; 3],
    /// Second control vector (`m0` for Hermite, `p1` otherwise).
    pub a1: [f32; 3],
    /// Third control vector (`p1` for Hermite, `p2` otherwise).
    pub a2: [f32; 3],
    /// Fourth control vector (`m1` for Hermite, `p3` otherwise).
    pub a3: [f32; 3],
    /// Segment parameter.
    pub t: f32,
}

impl CubicSplineSegmentQuery {
    /// Builds a query from the operation id, four control vectors and `t`.
    #[must_use]
    pub fn new(
        op_id: u32,
        a0: [f32; 3],
        a1: [f32; 3],
        a2: [f32; 3],
        a3: [f32; 3],
        t: f32,
    ) -> CubicSplineSegmentQuery {
        CubicSplineSegmentQuery {
            op_id,
            a0,
            a1,
            a2,
            a3,
            t,
        }
    }
}

/// One resolved answer for a single query, mirroring the selected golden
/// evaluator's output for that segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubicSplineSegmentResult {
    /// The evaluated vector when valid, else `[0, 0, 0]`.
    pub value: [f32; 3],
    /// `1` when `op_id` is in range (`0..=5`), else `0`.
    pub valid: u32,
}

/// Encodes one [`CubicSplineSegmentQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &CubicSplineSegmentQuery) -> GpuQuery {
    GpuQuery {
        op_id: q.op_id,
        a0x: q.a0[0],
        a0y: q.a0[1],
        a0z: q.a0[2],
        a1x: q.a1[0],
        a1y: q.a1[1],
        a1z: q.a1[2],
        a2x: q.a2[0],
        a2y: q.a2[1],
        a2z: q.a2[2],
        a3x: q.a3[0],
        a3y: q.a3[1],
        a3z: q.a3[2],
        t: q.t,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CubicSplineSegmentResult`].
fn decode_result(raw: &GpuResult) -> CubicSplineSegmentResult {
    CubicSplineSegmentResult {
        value: [raw.vx, raw.vy, raw.vz],
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

/// A compiled, reusable cubic-spline-segment compute pipeline, twinning the six
/// `CPU` golden evaluators in `prism_math::curve::spline`.
pub struct GpuCubicSplineSegment {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCubicSplineSegment {
    /// Compiles the cubic-spline-segment kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCubicSplineSegment {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment"),
            source: ShaderSource::Wgsl(CUBIC_SPLINE_SEGMENT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCubicSplineSegment {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CubicSplineSegmentResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and each `value`
    /// component to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CubicSplineSegmentQuery],
    ) -> Vec<CubicSplineSegmentResult> {
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
            label: Some("prism_volumetric_cubic_spline_segment_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment_bind_group"),
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
            label: Some("prism_volumetric_cubic_spline_segment_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cubic_spline_segment_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cubic_spline_segment_pass"),
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
