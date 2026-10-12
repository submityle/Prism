//! `wgpu` compute twin of the dual-quaternion screw-algebra free functions from
//! the `CPU` golden `prism_math::dual_quat::DualQuat`.
//!
//! A unit dual quaternion `q = real + e*dual` encodes a rigid motion: the real
//! part is a unit rotation quaternion `(x, y, z, w)` and the dual part carries
//! the translation. This module ports the *advanced* algebra — real-part dot,
//! conjugate, inverse, orthonormalizing normalize, negation, `nlerp`, screw
//! `sclerp` and the screw-parameter `powf` — selected by an unsigned `op_id`.
//! One thread resolves one query, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same dual quaternion the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces one golden operator (`op_id`):
//!
//! * `0` `real_dot(a, b)` — dot of the two real parts, a scalar returned in
//!   `out[0]`.
//! * `1` `conjugate(a)` — quaternion-conjugate both parts (`(x, y, z)` negated).
//! * `2` `inverse(a)` — identical to `conjugate` (unit dual-quat convention; the
//!   golden `inverse` just calls `conjugate`).
//! * `3` `normalize(a)` — scale by `1/|real|`, then subtract the residual of the
//!   dual along the real to restore `dot(real, dual) = 0`. A zero-length real is
//!   invalid (`valid = 0`).
//! * `4` `negated(a)` — negate both parts.
//! * `5` `nlerp(a, b, t)` — hemisphere-align `b`, linearly blend both parts,
//!   then `normalize`. A zero-length blended real is invalid.
//! * `6` `sclerp(a, b, t)` — screw-linear interpolation `a * (a^-1 * b)^t` with
//!   hemisphere alignment.
//! * `7` `powf(a, t)` — raise a unit dual quaternion to a real power via its
//!   screw parameters (axis, angle, pitch), with the pure-translation branch
//!   when the rotation is near zero.
//!
//! `op_id > 7` is rejected (`valid = 0`, output zeroed).
//!
//! # Correctness model
//!
//! Both sides evaluate the same pure-`f32` closed forms (Hamilton product, dual
//! product, `acos`/`sin`/`cos`/`sqrt` screw parameters), so `CPU` and `GPU` are
//! not necessarily bit-exact under reassociation; every continuous lane is
//! compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly. The parity sweep keeps inputs unit dual
//! quaternions with bounded screw angles so the validity decision and the
//! `acos` domain are never pushed to a knee.
//!
//! # Degenerate inputs
//!
//! `normalize` and the `normalize` tail of `nlerp` yield `valid = 0` when the
//! real part collapses to (near) zero length; the divisor is fed through a
//! `select` guard so the un-taken branch never divides by zero. `powf`'s
//! near-zero-rotation case (`|real.xyz|^2 < 1e-12`) is an in-domain pure
//! translation and stays `valid = 1`; its reciprocal square root is also guarded
//! so the general branch never leaks an infinity. An empty query batch
//! short-circuits on the host with no dispatch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`, `dot`,
//! `sqrt`, `acos`, `sin`, `cos`, `+ - * /`, `select` and unsigned index/branch
//! math — with no `round`, no `f32` remainder and no `f64`/`u64`/`i64` types.
//! The only `==` is on the unsigned `op_id` selector; finiteness-style guards use
//! ordered compares (`> MIN_LEN_SQ`, `< 1e-12`) rather than a bare `x == x`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::dual_quat::DualQuat`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` dual-quaternion screw kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `prism_math::dual_quat::DualQuat` operators selected by
/// `op_id`; see the module documentation for each closed form.
const DUAL_QUAT_SCREW_WGSL: &str = r#"
// Dual-quaternion screw twin: one thread per query reproduces one of the
// op_id-selected dual-quaternion closed forms. A quaternion is (x, y, z, w)
// with w the scalar part; the Hamilton product follows the golden
// `a * b = apply b then a` convention. Guards use ordered compares fed to
// select; the only == is on the unsigned op_id selector.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation selector; see module docs.
    op_id: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Dual quaternion a: real part (x, y, z, w) then dual part (x, y, z, w).
    a_rx: f32,
    a_ry: f32,
    a_rz: f32,
    a_rw: f32,
    a_dx: f32,
    a_dy: f32,
    a_dz: f32,
    a_dw: f32,
    // Dual quaternion b, same layout. Used by real_dot/nlerp/sclerp.
    b_rx: f32,
    b_ry: f32,
    b_rz: f32,
    b_rw: f32,
    b_dx: f32,
    b_dy: f32,
    b_dz: f32,
    b_dw: f32,
    // Interpolation / exponent parameter for nlerp/sclerp/powf.
    tval: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

struct Result {
    // Dual quaternion lanes: real (out0..out3) then dual (out4..out7); a scalar
    // lands in out0 for real_dot.
    out0: f32,
    out1: f32,
    out2: f32,
    out3: f32,
    out4: f32,
    out5: f32,
    out6: f32,
    out7: f32,
    // 1 when the operation produced a defined result, else 0.
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// A dual quaternion carried through the helper functions.
struct DQ {
    r: vec4<f32>,
    d: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const MIN_LEN_SQ: f32 = 1.0e-12;

// Quaternion conjugate: negate the vector part, keep the scalar.
fn q_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// Hamilton product (x, y, z, w); matches the golden quat_mul operator order.
fn q_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

// Dual-quaternion product: real = ar*br, dual = ar*bd + ad*br.
fn dq_mul(a: DQ, b: DQ) -> DQ {
    let r = q_mul(a.r, b.r);
    let d = q_mul(a.r, b.d) + q_mul(a.d, b.r);
    return DQ(r, d);
}

// Raise a unit dual quaternion to a real power via its screw parameters. The
// near-zero-rotation case is a pure translation; both branches are evaluated
// with a guarded reciprocal square root and selected so the un-taken branch
// never leaks an infinity.
fn dq_powf(a: DQ, tval: f32) -> DQ {
    let vr = a.r.xyz;
    let wr = a.r.w;
    let vd = a.d.xyz;
    let wd = a.d.w;
    let len_sq = dot(vr, vr);
    let is_pure = len_sq < MIN_LEN_SQ;
    let safe_len_sq = select(len_sq, 1.0, is_pure);
    let inv_r = 1.0 / sqrt(safe_len_sq);
    let angle0 = 2.0 * acos(clamp(wr, -1.0, 1.0));
    let pitch0 = -2.0 * wd * inv_r;
    let direction = vr * inv_r;
    let moment = (vd - direction * (pitch0 * wr * 0.5)) * inv_r;
    let angle = angle0 * tval;
    let pitch = pitch0 * tval;
    let half = angle * 0.5;
    let sin_half = sin(half);
    let cos_half = cos(half);
    let gen_rv = direction * sin_half;
    let gen_real = vec4<f32>(gen_rv.x, gen_rv.y, gen_rv.z, cos_half);
    let dual_v = moment * sin_half + direction * (pitch * 0.5 * cos_half);
    let gen_dual = vec4<f32>(dual_v.x, dual_v.y, dual_v.z, -pitch * 0.5 * sin_half);
    let sign = select(1.0, -1.0, wr < 0.0);
    let pure_real = vec4<f32>(0.0, 0.0, 0.0, sign);
    let pure_dual = a.d * tval;
    let out_real = select(gen_real, pure_real, is_pure);
    let out_dual = select(gen_dual, pure_dual, is_pure);
    return DQ(out_real, out_dual);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let qu = queries[idx];
    let op = qu.op_id;

    let ar = vec4<f32>(qu.a_rx, qu.a_ry, qu.a_rz, qu.a_rw);
    let ad = vec4<f32>(qu.a_dx, qu.a_dy, qu.a_dz, qu.a_dw);
    let br = vec4<f32>(qu.b_rx, qu.b_ry, qu.b_rz, qu.b_rw);
    let bd = vec4<f32>(qu.b_dx, qu.b_dy, qu.b_dz, qu.b_dw);
    let tval = qu.tval;

    var rr = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var rd = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var ok = true;

    if (op == 0u) {
        // real_dot(a, b): dot of the real parts, returned in out0.
        rr.x = dot(ar, br);
    } else if (op == 1u) {
        // conjugate: quaternion-conjugate both parts.
        rr = q_conj(ar);
        rd = q_conj(ad);
    } else if (op == 2u) {
        // inverse == conjugate (unit dual-quaternion convention).
        rr = q_conj(ar);
        rd = q_conj(ad);
    } else if (op == 3u) {
        // normalize: scale by 1/|real|, then orthogonalize the dual.
        let mag_sq = dot(ar, ar);
        let nok = mag_sq > MIN_LEN_SQ;
        let inv = 1.0 / select(1.0, sqrt(mag_sq), nok);
        let nr = ar * inv;
        let nd0 = ad * inv;
        let dsc = dot(nr, nd0);
        rr = nr;
        rd = nd0 - nr * dsc;
        ok = nok;
    } else if (op == 4u) {
        // negated: negate both parts.
        rr = -ar;
        rd = -ad;
    } else if (op == 5u) {
        // nlerp: hemisphere-align b, blend linearly, then normalize.
        let dp = dot(ar, br);
        let flip = dp < 0.0;
        let br2 = select(br, -br, flip);
        let bd2 = select(bd, -bd, flip);
        let real = ar * (1.0 - tval) + br2 * tval;
        let dual = ad * (1.0 - tval) + bd2 * tval;
        let mag_sq = dot(real, real);
        let nok = mag_sq > MIN_LEN_SQ;
        let inv = 1.0 / select(1.0, sqrt(mag_sq), nok);
        let nr = real * inv;
        let nd0 = dual * inv;
        let dsc = dot(nr, nd0);
        rr = nr;
        rd = nd0 - nr * dsc;
        ok = nok;
    } else if (op == 6u) {
        // sclerp: a * (a^-1 * b)^t with hemisphere alignment.
        let dp = dot(ar, br);
        let flip = dp < 0.0;
        let br2 = select(br, -br, flip);
        let bd2 = select(bd, -bd, flip);
        let diff = dq_mul(DQ(q_conj(ar), q_conj(ad)), DQ(br2, bd2));
        let powdiff = dq_powf(diff, tval);
        let res = dq_mul(DQ(ar, ad), powdiff);
        rr = res.r;
        rd = res.d;
    } else if (op == 7u) {
        // powf: screw-parameter power of a single dual quaternion.
        let res = dq_powf(DQ(ar, ad), tval);
        rr = res.r;
        rd = res.d;
    } else {
        // op_id past the twinned set: reject.
        ok = false;
    }

    var out: Result;
    out.out0 = select(0.0, rr.x, ok);
    out.out1 = select(0.0, rr.y, ok);
    out.out2 = select(0.0, rr.z, ok);
    out.out3 = select(0.0, rr.w, ok);
    out.out4 = select(0.0, rd.x, ok);
    out.out5 = select(0.0, rd.y, ok);
    out.out6 = select(0.0, rd.z, ok);
    out.out7 = select(0.0, rd.w, ok);
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
/// the `op_id` (padded to 16 bytes), the two dual quaternions `a` and `b`
/// (eight `f32` each), and the scalar parameter `t` (padded to 16 bytes). `24`
/// words / `96` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op_id: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    a_rx: f32,
    a_ry: f32,
    a_rz: f32,
    a_rw: f32,
    a_dx: f32,
    a_dy: f32,
    a_dz: f32,
    a_dw: f32,
    b_rx: f32,
    b_ry: f32,
    b_rz: f32,
    b_rw: f32,
    b_dx: f32,
    b_dy: f32,
    b_dz: f32,
    b_dw: f32,
    tval: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: eight output lanes, the validity flag and padding. `12` words /
/// `48` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    out0: f32,
    out1: f32,
    out2: f32,
    out3: f32,
    out4: f32,
    out5: f32,
    out6: f32,
    out7: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One dual-quaternion screw query: the operation selector, the two dual
/// quaternion operands (`real` lanes `0..4`, `dual` lanes `4..8`) and the
/// interpolation / exponent parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualQuatScrewQuery {
    /// Operation selector (`0..=7`); see the module documentation.
    pub op_id: u32,
    /// Operand `a`: `[real.x, real.y, real.z, real.w, dual.x, dual.y, dual.z, dual.w]`.
    pub a: [f32; 8],
    /// Operand `b`, same layout; used by `real_dot`, `nlerp` and `sclerp`.
    pub b: [f32; 8],
    /// Interpolation / exponent parameter for `nlerp`, `sclerp` and `powf`.
    pub t: f32,
}

impl DualQuatScrewQuery {
    /// Builds a query from the operation selector, two dual-quaternion operands
    /// and the scalar parameter.
    #[must_use]
    pub fn new(op_id: u32, a: [f32; 8], b: [f32; 8], t: f32) -> DualQuatScrewQuery {
        DualQuatScrewQuery { op_id, a, b, t }
    }
}

/// One resolved answer for a single query, mirroring the selected golden
/// `DualQuat` operator for that input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualQuatScrewResult {
    /// Output lanes: `real` in `0..4`, `dual` in `4..8`; a scalar result (for
    /// `real_dot`) lands in `out[0]` with the rest zero. All lanes are zero when
    /// `valid == 0`.
    pub out: [f32; 8],
    /// `1` when the operation produced a defined result, else `0`.
    pub valid: u32,
}

/// Encodes one [`DualQuatScrewQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &DualQuatScrewQuery) -> GpuQuery {
    GpuQuery {
        op_id: q.op_id,
        pad0: 0,
        pad1: 0,
        pad2: 0,
        a_rx: q.a[0],
        a_ry: q.a[1],
        a_rz: q.a[2],
        a_rw: q.a[3],
        a_dx: q.a[4],
        a_dy: q.a[5],
        a_dz: q.a[6],
        a_dw: q.a[7],
        b_rx: q.b[0],
        b_ry: q.b[1],
        b_rz: q.b[2],
        b_rw: q.b[3],
        b_dx: q.b[4],
        b_dy: q.b[5],
        b_dz: q.b[6],
        b_dw: q.b[7],
        tval: q.t,
        pad3: 0.0,
        pad4: 0.0,
        pad5: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`DualQuatScrewResult`].
fn decode_result(raw: &GpuResult) -> DualQuatScrewResult {
    DualQuatScrewResult {
        out: [
            raw.out0, raw.out1, raw.out2, raw.out3, raw.out4, raw.out5, raw.out6, raw.out7,
        ],
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

/// A compiled, reusable dual-quaternion screw compute pipeline, twinning the
/// `CPU` golden `prism_math::dual_quat::DualQuat` screw-algebra operators.
pub struct GpuDualQuatScrew {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDualQuatScrew {
    /// Compiles the dual-quaternion screw kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDualQuatScrew {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_dual_quat_screw"),
            source: ShaderSource::Wgsl(DUAL_QUAT_SCREW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_dual_quat_screw_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_dual_quat_screw_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_dual_quat_screw_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDualQuatScrew {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`DualQuatScrewResult`]
    /// per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the continuous lanes
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DualQuatScrewQuery],
    ) -> Vec<DualQuatScrewResult> {
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
            label: Some("prism_volumetric_dual_quat_screw_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_dual_quat_screw_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_dual_quat_screw_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_dual_quat_screw_bind_group"),
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
            label: Some("prism_volumetric_dual_quat_screw_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_dual_quat_screw_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_dual_quat_screw_pass"),
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
