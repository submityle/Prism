//! `wgpu` compute twin of quaternion spherical-linear interpolation, from the
//! `CPU` golden `prism_math::quat`'s `Quat::slerp`.
//!
//! Spherical linear interpolation walks the shortest great-circle arc between
//! two orientations. The golden first takes the dot product of the operands and,
//! when it is negative, negates the right operand so the interpolation follows
//! the shortest arc (a quaternion and its negation encode the same rotation).
//! When the operands are nearly colinear (`dot > 0.9995`) the arc degenerates
//! and the sine denominator approaches zero, so the golden falls back to a
//! normalized linear interpolation (`nlerp`); that fallback is part of the
//! `slerp` definition, not an independent `nlerp` twin. Otherwise it evaluates
//! the classic `sin((1-t)θ)/sinθ`, `sin(tθ)/sinθ` blend weights.
//!
//! Each query carries two quaternions `qa`, `qb` (stored `x, y, z, w`) and a
//! parameter `t`; one thread resolves one query, so a passing real-device parity
//! test is direct evidence the ported kernel reproduces the golden value, the
//! regime decision, and the shortest-arc flip.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `Quat::slerp` exactly:
//!
//! * `dot = qa · qb`; if `dot < 0` then `qb = -qb` and `dot = -dot` (shortest
//!   arc).
//! * If `dot > 0.9995` (regime `0`): `nlerp`, i.e. normalize
//!   `qa + (qb - qa) * t`.
//! * Otherwise (regime `1`): `θ = acos(clamp(dot, -1, 1))`,
//!   `s0 = sin((1-t)θ)/sinθ`, `s1 = sin(tθ)/sinθ`, result `qa*s0 + qb*s1`.
//!
//! # Correctness model
//!
//! The arithmetic threads through operators and transcendentals a `GPU` may
//! evaluate slightly differently, so `CPU` and `GPU` are not bit-exact; the four
//! output components are compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `regime` and `valid` flags are compared
//! exactly. The parity sweep keeps the (shortest-arc) dot a comfortable margin
//! away from both the `0` flip knee and the `0.9995` regime knee so the two
//! sides never straddle a branch boundary.
//!
//! # Degenerate inputs
//!
//! A zero-length operand would divide by zero inside the `nlerp` normalization,
//! so a query whose `qa` or `qb` has squared length `<= 1e-12` yields
//! `valid = 0` with the four outputs and `regime` set to zero. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the core-`WGSL` subset plus `acos`, `sin`, `sqrt`, `clamp`
//! and `dot`, with `select` and unsigned-integer flag packing. The branch
//! guards are ordered float compares (`dot < 0`, `dot > 0.9995`); there is no
//! `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::quat`；无第三方引擎源码或衍生代码。
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

/// The quaternion-slerp kernel, embedded inline so the twin ships as a single
/// source file. The single entry point `solve` mirrors the `CPU` golden
/// `Quat::slerp`, including the shortest-arc flip and the near-colinear
/// normalized-lerp fallback.
const QUAT_SLERP_WGSL: &str = r#"
// Quaternion slerp twin: one thread per query reproduces Quat::slerp. The
// shortest-arc flip and the regime decision are ordered float compares fed to
// select; there is no bare f32 equality. A zero-length operand (squared length
// <= EPS_LEN_SQ) is rejected as invalid.

const DOT_THRESHOLD: f32 = 0.9995;
const EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First quaternion (x, y, z, w).
    ax: f32,
    ay: f32,
    az: f32,
    aw: f32,
    // Second quaternion (x, y, z, w).
    bx: f32,
    by: f32,
    bz: f32,
    bw: f32,
    // Interpolation parameter.
    t: f32,
    // Padding to a 48-byte stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Interpolated quaternion (x, y, z, w), or zero when invalid.
    ox: f32,
    oy: f32,
    oz: f32,
    ow: f32,
    // 0 = near-colinear nlerp fallback, 1 = general slerp.
    regime: u32,
    // 1 when both operands are non-degenerate, else 0.
    valid: u32,
    // Padding to a 32-byte stride.
    pad0: u32,
    pad1: u32,
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
    let qa = vec4<f32>(q.ax, q.ay, q.az, q.aw);
    let qb_in = vec4<f32>(q.bx, q.by, q.bz, q.bw);
    let t = q.t;

    // Degenerate guard: a zero-length operand divides by zero in nlerp.
    let len_sq_a = dot(qa, qa);
    let len_sq_b = dot(qb_in, qb_in);
    let ok = (len_sq_a > EPS_LEN_SQ) && (len_sq_b > EPS_LEN_SQ);

    // Shortest arc: flip the right operand when the dot is negative.
    var rhs = qb_in;
    var d = dot(qa, rhs);
    if (d < 0.0) {
        rhs = -rhs;
        d = -d;
    }

    // Near-colinear (regime 0) falls back to normalized lerp.
    let near = d > DOT_THRESHOLD;

    var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (near) {
        // nlerp: normalize(qa + (rhs - qa) * t).
        let lerped = qa + (rhs - qa) * t;
        let inv = 1.0 / sqrt(dot(lerped, lerped));
        result = lerped * inv;
    } else {
        let theta = acos(clamp(d, -1.0, 1.0));
        let sin_theta = sin(theta);
        let s0 = sin((1.0 - t) * theta) / sin_theta;
        let s1 = sin(t * theta) / sin_theta;
        result = qa * s0 + rhs * s1;
    }

    let regime = select(1u, 0u, near);

    var out: Result;
    out.ox = select(0.0, result.x, ok);
    out.oy = select(0.0, result.y, ok);
    out.oz = select(0.0, result.z, ok);
    out.ow = select(0.0, result.w, ok);
    out.regime = select(0u, regime, ok);
    out.valid = select(0u, 1u, ok);
    out.pad0 = 0u;
    out.pad1 = 0u;
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
/// two quaternions and the parameter padded to `12` words (`48` bytes),
/// aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ax: f32,
    ay: f32,
    az: f32,
    aw: f32,
    bx: f32,
    by: f32,
    bz: f32,
    bw: f32,
    t: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the interpolated quaternion plus the regime and validity flags
/// padded to `8` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    ox: f32,
    oy: f32,
    oz: f32,
    ow: f32,
    regime: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
}

/// One quaternion-slerp query: both operands (stored `x, y, z, w`) and the
/// interpolation parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuatSlerpQuery {
    /// First quaternion `[x, y, z, w]`.
    pub qa: [f32; 4],
    /// Second quaternion `[x, y, z, w]`.
    pub qb: [f32; 4],
    /// Interpolation parameter, usually in `0..=1`.
    pub t: f32,
}

impl QuatSlerpQuery {
    /// Builds a query from both quaternions and the interpolation parameter.
    #[must_use]
    pub fn new(qa: [f32; 4], qb: [f32; 4], t: f32) -> QuatSlerpQuery {
        QuatSlerpQuery { qa, qb, t }
    }
}

/// One resolved answer for a single query, mirroring `Quat::slerp` for that
/// operand pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuatSlerpResult {
    /// The interpolated quaternion `[x, y, z, w]` when valid, else all zero.
    pub out: [f32; 4],
    /// `0` for the near-colinear normalized-lerp fallback, `1` for the general
    /// slerp branch; zero when invalid.
    pub regime: u32,
    /// `1` when both operands are non-degenerate, else `0`.
    pub valid: u32,
}

/// Encodes one [`QuatSlerpQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuatSlerpQuery) -> GpuQuery {
    GpuQuery {
        ax: q.qa[0],
        ay: q.qa[1],
        az: q.qa[2],
        aw: q.qa[3],
        bx: q.qb[0],
        by: q.qb[1],
        bz: q.qb[2],
        bw: q.qb[3],
        t: q.t,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QuatSlerpResult`].
fn decode_result(raw: &GpuResult) -> QuatSlerpResult {
    QuatSlerpResult {
        out: [raw.ox, raw.oy, raw.oz, raw.ow],
        regime: raw.regime,
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

/// A compiled, reusable quaternion-slerp compute pipeline, twinning the `CPU`
/// golden `Quat::slerp`.
pub struct GpuQuatSlerp {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuatSlerp {
    /// Compiles the quaternion-slerp kernel on `ctx`.
    ///
    /// The kernel uses only the core-`WGSL` subset plus standard math
    /// built-ins, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuatSlerp {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quat_slerp"),
            source: ShaderSource::Wgsl(QUAT_SLERP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quat_slerp_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quat_slerp_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quat_slerp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuatSlerp {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`QuatSlerpResult`] per
    /// input, in order.
    ///
    /// The `regime` and `valid` flags match the reference exactly and the four
    /// output components to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[QuatSlerpQuery]) -> Vec<QuatSlerpResult> {
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
            label: Some("prism_volumetric_quat_slerp_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quat_slerp_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quat_slerp_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quat_slerp_bind_group"),
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
            label: Some("prism_volumetric_quat_slerp_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quat_slerp_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quat_slerp_pass"),
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
