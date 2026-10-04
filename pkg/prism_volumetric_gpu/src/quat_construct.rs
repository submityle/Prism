//! `wgpu` compute twin of the `prism_math` quaternion construction and basic
//! algebra closed forms, dispatched by an operation id.
//!
//! The `CPU` golden `prism_math::quat::Quat` stores a quaternion as
//! `(x, y, z, w)` with `w` the scalar part. This module ports the stateless
//! construction and algebra free functions onto the device: one thread resolves
//! one query, selecting the operation by an unsigned `op_id`, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same quaternion the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The rotation/`mul_vec3`/`nlerp`/`slerp` operations are twinned elsewhere and
//! are deliberately excluded here. The kernel reproduces, selected by `op_id`:
//!
//! * `0` `from_axis_angle(axis, angle)`: `(s, c) = sin_cos(angle * 0.5)` then
//!   `(axis.x * s, axis.y * s, axis.z * s, c)`. The golden does **not**
//!   normalize `axis`; it assumes a unit axis and simply scales, so this twin
//!   reproduces `axis * s` faithfully with no normalization.
//! * `1` `from_rotation_x(angle)`: `(s, 0, 0, c)`.
//! * `2` `from_rotation_y(angle)`: `(0, s, 0, c)`.
//! * `3` `from_rotation_z(angle)`: `(0, 0, s, c)`.
//! * `4` `conjugate(q0)`: `(-x, -y, -z, w)`.
//! * `5` `inverse(q0)`: the golden `inverse` is exactly `conjugate` (unit-
//!   quaternion convention), so this twin returns `(-x, -y, -z, w)` too — it is
//!   **not** divided by the squared length.
//! * `6` `normalize(q0)`: `inv = 1 / length` then each component scaled; a
//!   zero-length or non-finite input yields `valid = 0` with zeroed output.
//! * `7` `dot(q0, q1)`: `x0*x1 + y0*y1 + z0*z1 + w0*w1`, a scalar in `out[0]`.
//! * `8` `length(q0)`: `sqrt(dot(q0, q0))`, a scalar in `out[0]`.
//! * `9` `length_squared(q0)`: `dot(q0, q0)`, a scalar in `out[0]`.
//! * any `op_id > 9`: rejected with `valid = 0` and zeroed output.
//!
//! Quaternion-valued operations fill all four `out` lanes; scalar-valued
//! operations place the scalar in `out[0]` and leave the other lanes zero.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators (including `sin`, `cos`
//! and `sqrt`) that a `GPU` may contract or evaluate at a slightly different
//! precision, so `CPU` and `GPU` are not necessarily bit-exact; each valid
//! output scalar is compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared exactly; the
//! parity test keeps every swept operand master-valid (rejecting the zero-length
//! normalize input) so the validity decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! Only `normalize` has a degenerate branch: a zero-length or non-finite input
//! yields `valid = 0` with zeroed output. The kernel feeds the divisor through a
//! `select` guard so the un-taken (invalid) branch never divides by zero and
//! never leaks an infinity into the selected lanes. An `op_id` past the twinned
//! set is rejected the same way. An empty query batch short-circuits on the host
//! with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset — `sin`, `cos`, `sqrt`,
//! `abs`, `+ - * /`, `select` and unsigned index/branch arithmetic — with no
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no 64-bit or
//! sub-32-bit integer types, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Finiteness is tested with the ordered compare `abs(x) < 3.0e38`
//! (which rejects both infinities and `NaN`) rather than a bare `x == x`, and
//! the zero-length guard with ordered `> 1e-30`; the only `==` comparisons are
//! on the unsigned `op_id` selector.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::quat::Quat`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` quaternion-construction kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `prism_math::quat::Quat` construction and algebra
/// free functions, selected by `op_id`; see the module documentation for the
/// closed forms.
const QUAT_CONSTRUCT_WGSL: &str = r#"
// Quaternion-construction twin: one thread per query reproduces one of the
// op_id-selected quaternion closed forms. It uses only the portable core-WGSL
// subset (sin, cos, sqrt, abs, + - * /, select plus unsigned index/branch
// math) and has no loop, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) and the zero-length
// guard an ordered > 1e-30 compare, both fed to select; the only == is on the
// unsigned op_id selector.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Rotation axis for from_axis_angle (op_id 0); unused otherwise.
    axis_x: f32,
    axis_y: f32,
    axis_z: f32,
    // Rotation angle in radians for from_axis_angle / from_rotation_* (ops 0-3).
    angle: f32,
    // Primary quaternion operand q0 for ops 4-9.
    q0x: f32,
    q0y: f32,
    q0z: f32,
    q0w: f32,
    // Secondary quaternion operand q1 for dot (op 7).
    q1x: f32,
    q1y: f32,
    q1z: f32,
    q1w: f32,
    // Operation selector; see module docs.
    op_id: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Quaternion lanes, or a scalar in out0 for dot/length/length_squared.
    out0: f32,
    out1: f32,
    out2: f32,
    out3: f32,
    // 1 when the operation produced a defined result, else 0.
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const MIN_LEN_SQ: f32 = 1.0e-30;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let qu = queries[idx];
    let op = qu.op_id;

    // Half-angle sine/cosine for the construction ops; harmless when unused.
    let half_angle = qu.angle * 0.5;
    let sn = sin(half_angle);
    let cs = cos(half_angle);

    var r0 = 0.0;
    var r1 = 0.0;
    var r2 = 0.0;
    var r3 = 0.0;
    var ok = true;

    if (op == 0u) {
        // from_axis_angle: axis * s, w = c. Axis is NOT normalized (golden
        // assumes a unit axis and simply scales).
        r0 = qu.axis_x * sn;
        r1 = qu.axis_y * sn;
        r2 = qu.axis_z * sn;
        r3 = cs;
    } else if (op == 1u) {
        r0 = sn;
        r1 = 0.0;
        r2 = 0.0;
        r3 = cs;
    } else if (op == 2u) {
        r0 = 0.0;
        r1 = sn;
        r2 = 0.0;
        r3 = cs;
    } else if (op == 3u) {
        r0 = 0.0;
        r1 = 0.0;
        r2 = sn;
        r3 = cs;
    } else if (op == 4u) {
        // conjugate: (-x, -y, -z, w).
        r0 = -qu.q0x;
        r1 = -qu.q0y;
        r2 = -qu.q0z;
        r3 = qu.q0w;
    } else if (op == 5u) {
        // inverse == conjugate exactly (unit-quaternion convention).
        r0 = -qu.q0x;
        r1 = -qu.q0y;
        r2 = -qu.q0z;
        r3 = qu.q0w;
    } else if (op == 6u) {
        // normalize: inv = 1 / length, each component scaled. A zero-length or
        // non-finite input is invalid; the divisor is guarded so the un-taken
        // branch never divides by zero or leaks an infinity.
        let ls = qu.q0x * qu.q0x + qu.q0y * qu.q0y + qu.q0z * qu.q0z + qu.q0w * qu.q0w;
        let finite = abs(ls) < FINITE_LIMIT;
        let nonzero = ls > MIN_LEN_SQ;
        let nok = finite && nonzero;
        let denom = select(1.0, sqrt(ls), nok);
        let inv = 1.0 / denom;
        r0 = qu.q0x * inv;
        r1 = qu.q0y * inv;
        r2 = qu.q0z * inv;
        r3 = qu.q0w * inv;
        ok = nok;
    } else if (op == 7u) {
        // dot(q0, q1) → scalar in out0.
        r0 = qu.q0x * qu.q1x + qu.q0y * qu.q1y + qu.q0z * qu.q1z + qu.q0w * qu.q1w;
    } else if (op == 8u) {
        // length(q0) → scalar in out0.
        let ls = qu.q0x * qu.q0x + qu.q0y * qu.q0y + qu.q0z * qu.q0z + qu.q0w * qu.q0w;
        r0 = sqrt(ls);
    } else if (op == 9u) {
        // length_squared(q0) → scalar in out0.
        r0 = qu.q0x * qu.q0x + qu.q0y * qu.q0y + qu.q0z * qu.q0z + qu.q0w * qu.q0w;
    } else {
        // op_id past the twinned set: reject.
        ok = false;
    }

    var out: Result;
    out.out0 = select(0.0, r0, ok);
    out.out1 = select(0.0, r1, ok);
    out.out2 = select(0.0, r2, ok);
    out.out3 = select(0.0, r3, ok);
    out.valid = select(0u, 1u, ok);
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
/// `16` scalar words (`64` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    axis_x: f32,
    axis_y: f32,
    axis_z: f32,
    angle: f32,
    q0x: f32,
    q0y: f32,
    q0z: f32,
    q0w: f32,
    q1x: f32,
    q1y: f32,
    q1z: f32,
    q1w: f32,
    op_id: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: four output lanes, the validity flag and padding — `8` words
/// (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    out0: f32,
    out1: f32,
    out2: f32,
    out3: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One quaternion-construction query: the operation selector and its operands.
///
/// Unused operands for a given `op_id` are ignored by the kernel and may be any
/// value; the [`QuatConstructQuery::new`] constructor takes them all so the
/// caller stays explicit about the dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuatConstructQuery {
    /// Operation selector: `0` `from_axis_angle`, `1`/`2`/`3`
    /// `from_rotation_x`/`y`/`z`, `4` `conjugate`, `5` `inverse`, `6`
    /// `normalize`, `7` `dot`, `8` `length`, `9` `length_squared`.
    pub op_id: u32,
    /// Rotation axis for `from_axis_angle` (`op_id = 0`).
    pub axis: [f32; 3],
    /// Rotation angle in radians for `from_axis_angle` / `from_rotation_*`.
    pub angle: f32,
    /// Primary quaternion operand `q0` for `op_id` `4..=9`.
    pub q0: [f32; 4],
    /// Secondary quaternion operand `q1` for `dot` (`op_id = 7`).
    pub q1: [f32; 4],
}

impl QuatConstructQuery {
    /// Builds a query from the operation selector and all operands.
    #[must_use]
    pub fn new(
        op_id: u32,
        axis: [f32; 3],
        angle: f32,
        q0: [f32; 4],
        q1: [f32; 4],
    ) -> QuatConstructQuery {
        QuatConstructQuery {
            op_id,
            axis,
            angle,
            q0,
            q1,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference quaternion
/// operation output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuatConstructResult {
    /// The four output lanes: a full quaternion for quaternion-valued ops, or a
    /// scalar in `out[0]` for `dot` / `length` / `length_squared`. Zeroed when
    /// invalid.
    pub out: [f32; 4],
    /// `1` when the operation produced a defined result, else `0`.
    pub valid: u32,
}

/// Encodes one [`QuatConstructQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuatConstructQuery) -> GpuQuery {
    GpuQuery {
        axis_x: q.axis[0],
        axis_y: q.axis[1],
        axis_z: q.axis[2],
        angle: q.angle,
        q0x: q.q0[0],
        q0y: q.q0[1],
        q0z: q.q0[2],
        q0w: q.q0[3],
        q1x: q.q1[0],
        q1y: q.q1[1],
        q1z: q.q1[2],
        q1w: q.q1[3],
        op_id: q.op_id,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QuatConstructResult`].
fn decode_result(raw: &GpuResult) -> QuatConstructResult {
    QuatConstructResult {
        out: [raw.out0, raw.out1, raw.out2, raw.out3],
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

/// A compiled, reusable quaternion-construction compute pipeline, twinning the
/// `CPU` golden `prism_math::quat::Quat` construction and algebra free
/// functions.
pub struct GpuQuatConstruct {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuatConstruct {
    /// Compiles the quaternion-construction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuatConstruct {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quat_construct"),
            source: ShaderSource::Wgsl(QUAT_CONSTRUCT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quat_construct_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quat_construct_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quat_construct_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuatConstruct {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`QuatConstructResult`]
    /// per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the output scalars to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QuatConstructQuery],
    ) -> Vec<QuatConstructResult> {
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
            label: Some("prism_volumetric_quat_construct_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quat_construct_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quat_construct_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quat_construct_bind_group"),
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
            label: Some("prism_volumetric_quat_construct_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quat_construct_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quat_construct_pass"),
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
