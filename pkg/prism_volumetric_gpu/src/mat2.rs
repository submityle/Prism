//! `wgpu` compute twin of the `CPU` golden `prism_math::mat::Mat2`'s 2x2
//! linear-algebra operators `determinant`, `transpose`, `mul_vec2` and
//! `inverse`, dispatched behind an integer selector so one thread resolves one
//! query.
//!
//! `Mat2` is column-major: its two columns are the vectors `x_axis` and
//! `y_axis`. This twin flattens a matrix into four `f32` in column-major order
//! `m = [x_axis.x, x_axis.y, y_axis.x, y_axis.y]`, i.e. laid out as
//!
//! ```text
//! | m0  m2 |
//! | m1  m3 |
//! ```
//!
//! A passing real-device parity test is direct evidence the ported kernel
//! computes the same algebra the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The `op_id` selects which golden operator the thread reproduces for its
//! flattened matrix `m` and (for the transform) its vector `v`:
//!
//! * `op_id == 0` mirrors `determinant`: the scalar `m0 * m3 - m2 * m1`,
//!   returned in `out[0]`.
//! * `op_id == 1` mirrors `transpose`: the columns swap rows, giving
//!   `out = [m0, m2, m1, m3]`.
//! * `op_id == 2` mirrors `mul_vec2`: the column combination
//!   `x_axis * v.x + y_axis * v.y`, giving
//!   `out = [m0 * v.x + m2 * v.y, m1 * v.x + m3 * v.y, 0, 0]`.
//! * `op_id == 3` mirrors `inverse`: with `inv = 1 / det`, the adjugate layout
//!   `out = [m3 * inv, -m1 * inv, -m2 * inv, m0 * inv]`.
//!
//! Any `op_id > 3` yields `valid = 0` with a cleared `out`. The `inverse`
//! operator additionally requires a non-singular matrix: when `abs(det)` is at
//! or below `1e-20` the result is `valid = 0` with a cleared `out`, mirroring
//! the golden's debug-assert on a singular matrix.
//!
//! # Correctness model
//!
//! The golden evaluates every operator in `f32`, and the kernel reproduces the
//! same `f32` operator order; `CPU` and `GPU` are therefore not bit-exact (a
//! `GPU` may contract a multiply-add). Each valid `out` entry is compared with
//! an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly. The parity sweep keeps `abs(det)` well
//! away from the singularity knee so the `inverse` validity decision cannot
//! flip under round-off.
//!
//! # Degenerate inputs
//!
//! An out-of-range `op_id` (greater than `3`) and a singular matrix under
//! `inverse` both yield `valid = 0` with a cleared `out`. Every branch is
//! evaluated unconditionally and combined with `select`, and the reciprocal of
//! the determinant is guarded so no un-taken branch can produce an infinity or
//! `NaN`. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `abs`,
//! `select`, ordered compares and unsigned index arithmetic — with no
//! transcendental, no `round`, no `f32` remainder and no banned wide types, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. The validity and
//! operator selection use integer `==`/`<=` on the `u32` id and the ordered
//! compare `abs(det) > 1e-20`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::mat::Mat2`；无第三方引擎源码或衍生代码。
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

/// Operator id for `determinant` (scalar result in `out[0]`).
pub const OP_DETERMINANT: u32 = 0;
/// Operator id for `transpose` (full 2x2 result).
pub const OP_TRANSPOSE: u32 = 1;
/// Operator id for `mul_vec2` (2D vector result in `out[0..2]`).
pub const OP_MUL_VEC2: u32 = 2;
/// Operator id for `inverse` (full 2x2 result; singular matrices are invalid).
pub const OP_INVERSE: u32 = 3;

/// The portable core-`WGSL` 2x2 matrix kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// four `CPU` golden `Mat2` operators selected by `op_id`; see the module
/// documentation for the closed forms.
const MAT2_WGSL: &str = r#"
// Mat2 twin: one thread per query reproduces one of four prism_math::mat::Mat2
// operators selected by op_id. It uses only the portable core-WGSL subset
// (+ - * /, abs, select, ordered compares plus unsigned index math), has no
// loop and no data-dependent control flow (every branch is computed and
// combined with select), so it provably terminates. The matrix is column-major
// flattened as m = [x_axis.x, x_axis.y, y_axis.x, y_axis.y]. The operator and
// validity selection use integer == / <= on the u32 id and the ordered compare
// abs(det) > 1e-20; there is no bare f32 equality.
// Provenance: 孪生自本仓 prism_math::mat::Mat2。

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Column-major flattened matrix: m0 = x_axis.x, m1 = x_axis.y,
    // m2 = y_axis.x, m3 = y_axis.y.
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    // Vector operand for mul_vec2.
    vx: f32,
    vy: f32,
    // Which golden operator to reproduce.
    op_id: u32,
    // Padding word to a 8-word (32-byte) stride.
    pad0: u32,
}

struct Result {
    // Flattened result: scalar in o0 for determinant, vector in o0/o1 for
    // mul_vec2, full column-major matrix in o0..o3 for transpose/inverse.
    o0: f32,
    o1: f32,
    o2: f32,
    o3: f32,
    // 1 when the operator is known and (for inverse) non-singular, else 0.
    valid: u32,
    // Padding words to a 8-word (32-byte) stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let m0 = q.m0;
    let m1 = q.m1;
    let m2 = q.m2;
    let m3 = q.m3;
    let vx = q.vx;
    let vy = q.vy;
    let op_id = q.op_id;

    // op 0: determinant.
    let det = m0 * m3 - m2 * m1;

    // op 2: mul_vec2 (column combination x_axis * vx + y_axis * vy).
    let mv0 = m0 * vx + m2 * vy;
    let mv1 = m1 * vx + m3 * vy;

    // op 3: inverse. Guard the reciprocal so a singular matrix cannot produce
    // an infinity or NaN in the un-taken branch.
    let ok_inv = abs(det) > 1e-20;
    let inv = select(0.0, 1.0 / det, ok_inv);
    let inv0 = m3 * inv;
    let inv1 = -m1 * inv;
    let inv2 = -m2 * inv;
    let inv3 = m0 * inv;

    // Combine the chosen operator's components by integer id.
    var o0 = 0.0;
    var o1 = 0.0;
    var o2 = 0.0;
    var o3 = 0.0;

    // op 0: determinant -> scalar in o0.
    o0 = select(o0, det, op_id == 0u);

    // op 1: transpose -> [m0, m2, m1, m3].
    o0 = select(o0, m0, op_id == 1u);
    o1 = select(o1, m2, op_id == 1u);
    o2 = select(o2, m1, op_id == 1u);
    o3 = select(o3, m3, op_id == 1u);

    // op 2: mul_vec2 -> [mv0, mv1, 0, 0].
    o0 = select(o0, mv0, op_id == 2u);
    o1 = select(o1, mv1, op_id == 2u);

    // op 3: inverse -> adjugate layout.
    o0 = select(o0, inv0, op_id == 3u);
    o1 = select(o1, inv1, op_id == 3u);
    o2 = select(o2, inv2, op_id == 3u);
    o3 = select(o3, inv3, op_id == 3u);

    // Validity: a known op is valid, except inverse also needs a non-singular
    // matrix. An unknown id (> 3) is invalid.
    let known = op_id <= 3u;
    let valid_bool = select(known, ok_inv, op_id == 3u);

    var out: Result;
    out.o0 = select(0.0, o0, valid_bool);
    out.o1 = select(0.0, o1, valid_bool);
    out.o2 = select(0.0, o2, valid_bool);
    out.o3 = select(0.0, o3, valid_bool);
    out.valid = select(0u, 1u, valid_bool);
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the four column-major matrix entries, the vector operand, the operator id
/// and padding to an 8-word (32-byte) stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    vx: f32,
    vy: f32,
    op_id: u32,
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the four flattened output entries, the validity flag and padding to
/// an 8-word (32-byte) stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    o0: f32,
    o1: f32,
    o2: f32,
    o3: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One 2x2 matrix query: an operator selector, the column-major flattened
/// matrix and the vector operand used by the transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat2Query {
    /// Which golden operator to reproduce (see the `OP_*` constants).
    pub op_id: u32,
    /// The column-major flattened matrix `[x_axis.x, x_axis.y, y_axis.x,
    /// y_axis.y]`.
    pub m: [f32; 4],
    /// The vector operand for `mul_vec2`; ignored by the other operators.
    pub v: [f32; 2],
}

impl Mat2Query {
    /// Builds a query selecting `op_id` for matrix `m` and vector `v`.
    #[must_use]
    pub fn new(op_id: u32, m: [f32; 4], v: [f32; 2]) -> Mat2Query {
        Mat2Query { op_id, m, v }
    }
}

/// One resolved answer for a single query, mirroring the selected golden
/// operator for that matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat2Result {
    /// The flattened result: scalar in `out[0]` for `determinant`, vector in
    /// `out[0..2]` for `mul_vec2`, full column-major matrix in `out` for
    /// `transpose`/`inverse`. Cleared to zero when invalid.
    pub out: [f32; 4],
    /// `1` when the operator is known and (for `inverse`) non-singular, else
    /// `0`.
    pub valid: u32,
}

/// Encodes one [`Mat2Query`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &Mat2Query) -> GpuQuery {
    GpuQuery {
        m0: q.m[0],
        m1: q.m[1],
        m2: q.m[2],
        m3: q.m[3],
        vx: q.v[0],
        vy: q.v[1],
        op_id: q.op_id,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`Mat2Result`].
fn decode_result(raw: &GpuResult) -> Mat2Result {
    Mat2Result {
        out: [raw.o0, raw.o1, raw.o2, raw.o3],
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

/// A compiled, reusable 2x2 matrix compute pipeline, twinning the four `CPU`
/// golden `prism_math::mat::Mat2` operators.
pub struct GpuMat2 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMat2 {
    /// Compiles the 2x2 matrix kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMat2 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mat2"),
            source: ShaderSource::Wgsl(MAT2_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mat2_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mat2_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mat2_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMat2 {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`Mat2Result`] per
    /// input, in order.
    ///
    /// The `valid` flag matches the reference exactly and each `out` entry to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[Mat2Query]) -> Vec<Mat2Result> {
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
            label: Some("prism_volumetric_mat2_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mat2_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mat2_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mat2_bind_group"),
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
            label: Some("prism_volumetric_mat2_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mat2_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mat2_pass"),
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
