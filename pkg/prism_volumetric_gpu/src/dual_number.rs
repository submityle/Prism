//! `wgpu` compute twin of forward-mode dual-number (first-order automatic
//! differentiation) scalar operations, from the `CPU` golden
//! `prism_math::dual`'s `Dual`.
//!
//! A dual number `Dual { re, du }` carries a value `re = f(t)` together with its
//! first derivative `du = f'(t)` with respect to one scalar parameter.
//! Evaluating any supported operation simultaneously evaluates the function and
//! its exact analytic derivative, with none of the step-size tuning of finite
//! differences. This module ports the scalar operation set onto the device:
//! each query selects one operation by an enum `op_id`, and one thread resolves
//! one query, so a passing real-device parity test is direct evidence the
//! ported kernel computes the same value/derivative pair the reference does.
//!
//! This is strictly forward mode, first order — classical numerical
//! differentiation, not machine learning. It is distinct from the dual
//! quaternion (`dual_quaternion`), which represents a rigid transform rather
//! than a differentiated scalar.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces one of the sixteen `Dual` operations
//! for the operand(s) `a = (a_re, a_du)` and, for binary operations,
//! `b = (b_re, b_du)`, plus a scalar `p` used by `powf` and `mul_scalar`:
//!
//! * `0` neg, `1` recip, `2` sqrt, `3` squared, `4` exp, `5` ln, `6` sin,
//!   `7` cos, `8` tan, `9` abs, `10` powf, `11` `mul_scalar`, `12` add, `13` sub,
//!   `14` mul (product rule), `15` div (quotient rule).
//! * An `op_id` outside `0..=15` yields `valid = 0` with `re = du = 0`.
//!
//! # Correctness model
//!
//! The arithmetic threads through operators a `GPU` may contract, so `CPU` and
//! `GPU` are not necessarily bit-exact; the `re` and `du` scalars are compared
//! with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The
//! discrete `valid` flag is compared exactly. The parity sweep keeps operands
//! in each operation's well-conditioned domain (reciprocal/division denominators
//! away from zero, `sqrt`/`ln`/`powf` bases positive) so the two sides agree.
//!
//! # Degenerate inputs
//!
//! An out-of-range `op_id` yields `valid = 0` with `re = du = 0`. The `abs`
//! derivative at exactly zero is taken as zero, matching the golden
//! `sign`-of-zero rule; the device uses the built-in `sign`, which is also zero
//! at zero. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the core-`WGSL` subset plus the standard math built-ins the
//! golden also calls (`sqrt`, `exp`, `log` for the natural logarithm, `sin`,
//! `cos`, `tan`, `pow`, `sign`, `abs`), with `select` and unsigned integer
//! `op_id` dispatch. There is no `f32` equality anywhere; validity is the
//! ordered unsigned compare `op_id <= 15`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::dual`；无第三方引擎源码或衍生代码。
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

/// The dual-number kernel, embedded inline so the twin ships as a single source
/// file. The single entry point `solve` mirrors the `CPU` golden `Dual`
/// operations selected by `op_id`; see the module documentation for the closed
/// forms.
const DUAL_NUMBER_WGSL: &str = r#"
// Dual-number twin: one thread per query reproduces one Dual operation selected
// by op_id. Validity is the ordered unsigned compare op_id <= 15, fed to
// select; there is no bare f32 equality. The abs derivative uses the built-in
// sign, which is zero at zero, matching the golden sign-of-zero rule.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation selector in 0..=15; anything else is invalid.
    op_id: u32,
    // First operand value and derivative.
    a_re: f32,
    a_du: f32,
    // Second operand value and derivative (binary operations only).
    b_re: f32,
    b_du: f32,
    // Scalar exponent (powf) or multiplier (mul_scalar); ignored otherwise.
    p: f32,
    // Padding to a 32-byte stride.
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Value f(t) of the selected operation, or 0 when invalid.
    re: f32,
    // Derivative f'(t) of the selected operation, or 0 when invalid.
    du: f32,
    // 1 when op_id is in range, else 0.
    valid: u32,
    // Padding to a 16-byte stride.
    pad0: u32,
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
    let op = q.op_id;
    let a_re = q.a_re;
    let a_du = q.a_du;
    let b_re = q.b_re;
    let b_du = q.b_du;
    let p = q.p;

    // Validity is the ordered unsigned compare; op_id > 15 is rejected.
    let ok = op <= 15u;

    var re = 0.0;
    var du = 0.0;
    if (op == 0u) {
        // neg
        re = -a_re;
        du = -a_du;
    } else if (op == 1u) {
        // recip: d(1/x) = -x'/x^2
        let inv = 1.0 / a_re;
        re = inv;
        du = -a_du * inv * inv;
    } else if (op == 2u) {
        // sqrt: d(sqrt x) = x'/(2 sqrt x)
        let r = sqrt(a_re);
        re = r;
        du = a_du / (2.0 * r);
    } else if (op == 3u) {
        // squared: d(x^2) = 2 x x'
        re = a_re * a_re;
        du = 2.0 * a_re * a_du;
    } else if (op == 4u) {
        // exp: d(e^x) = e^x x'
        let e = exp(a_re);
        re = e;
        du = e * a_du;
    } else if (op == 5u) {
        // ln (log is the natural logarithm in WGSL): d(ln x) = x'/x
        re = log(a_re);
        du = a_du / a_re;
    } else if (op == 6u) {
        // sin: d(sin x) = cos x x'
        re = sin(a_re);
        du = cos(a_re) * a_du;
    } else if (op == 7u) {
        // cos: d(cos x) = -sin x x'
        re = cos(a_re);
        du = -sin(a_re) * a_du;
    } else if (op == 8u) {
        // tan: d(tan x) = (1 + tan^2 x) x'
        let t = tan(a_re);
        re = t;
        du = (1.0 + t * t) * a_du;
    } else if (op == 9u) {
        // abs: d(|x|) = sign(x) x'; sign is zero at zero.
        re = abs(a_re);
        du = sign(a_re) * a_du;
    } else if (op == 10u) {
        // powf: d(x^p) = p x^(p-1) x'
        re = pow(a_re, p);
        du = p * pow(a_re, p - 1.0) * a_du;
    } else if (op == 11u) {
        // mul_scalar
        re = a_re * p;
        du = a_du * p;
    } else if (op == 12u) {
        // add
        re = a_re + b_re;
        du = a_du + b_du;
    } else if (op == 13u) {
        // sub
        re = a_re - b_re;
        du = a_du - b_du;
    } else if (op == 14u) {
        // mul: product rule
        re = a_re * b_re;
        du = a_du * b_re + a_re * b_du;
    } else if (op == 15u) {
        // div: quotient rule
        let inv = 1.0 / b_re;
        re = a_re * inv;
        du = (a_du * b_re - a_re * b_du) * inv * inv;
    }

    var out: Result;
    out.re = select(0.0, re, ok);
    out.du = select(0.0, du, ok);
    out.valid = select(0u, 1u, ok);
    out.pad0 = 0u;
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
/// Six operands padded to `8` words (`32` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op_id: u32,
    a_re: f32,
    a_du: f32,
    b_re: f32,
    b_du: f32,
    p: f32,
    pad0: u32,
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the value/derivative pair and the validity flag padded to `4` words
/// (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    re: f32,
    du: f32,
    valid: u32,
    pad0: u32,
}

/// One dual-number query: the operation selector, the operand(s) and the scalar
/// parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualQuery {
    /// Operation selector in `0..=15`; see the module docs for the mapping.
    pub op_id: u32,
    /// First operand value `a_re`.
    pub a_re: f32,
    /// First operand derivative `a_du`.
    pub a_du: f32,
    /// Second operand value `b_re` (binary operations only).
    pub b_re: f32,
    /// Second operand derivative `b_du` (binary operations only).
    pub b_du: f32,
    /// Scalar exponent (`powf`) or multiplier (`mul_scalar`); ignored for the
    /// other operations.
    pub p: f32,
}

impl DualQuery {
    /// Builds a query from the operation selector, both operands and the scalar
    /// parameter.
    #[must_use]
    pub fn new(op_id: u32, a_re: f32, a_du: f32, b_re: f32, b_du: f32, p: f32) -> DualQuery {
        DualQuery {
            op_id,
            a_re,
            a_du,
            b_re,
            b_du,
            p,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference `Dual`
/// output for that operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualResult {
    /// The value component `f(t)` when `op_id` is valid, else `0`.
    pub re: f32,
    /// The derivative component `f'(t)` when `op_id` is valid, else `0`.
    pub du: f32,
    /// `1` when `op_id` is in `0..=15`, else `0`.
    pub valid: u32,
}

/// Encodes one [`DualQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &DualQuery) -> GpuQuery {
    GpuQuery {
        op_id: q.op_id,
        a_re: q.a_re,
        a_du: q.a_du,
        b_re: q.b_re,
        b_du: q.b_du,
        p: q.p,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`DualResult`].
fn decode_result(raw: &GpuResult) -> DualResult {
    DualResult {
        re: raw.re,
        du: raw.du,
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

/// A compiled, reusable dual-number compute pipeline, twinning the `CPU` golden
/// `Dual` scalar operations.
pub struct GpuDual {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDual {
    /// Compiles the dual-number kernel on `ctx`.
    ///
    /// The kernel uses only the core-`WGSL` subset plus standard math
    /// built-ins, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDual {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_dual_number"),
            source: ShaderSource::Wgsl(DUAL_NUMBER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_dual_number_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_dual_number_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_dual_number_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDual {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`DualResult`] per input,
    /// in order.
    ///
    /// The `valid` flag matches the reference exactly and the `re`/`du` scalars
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[DualQuery]) -> Vec<DualResult> {
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
            label: Some("prism_volumetric_dual_number_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_dual_number_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_dual_number_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_dual_number_bind_group"),
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
            label: Some("prism_volumetric_dual_number_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_dual_number_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_dual_number_pass"),
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
