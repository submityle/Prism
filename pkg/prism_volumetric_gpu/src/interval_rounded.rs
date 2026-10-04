//! `wgpu` compute twin of the conservative, outward-rounded interval arithmetic
//! from the `CPU` golden `prism_math::interval`.
//!
//! An interval `[lo, hi]` encloses the mathematically exact result of every
//! operation: rounding error is at most half a unit in the last place (`ULP`)
//! and each operation inflates its result outward by a full `ULP` via
//! [`next_down`]/[`next_up`], so the exact value is always enclosed. This module
//! ports that single stateless family of closed forms onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same enclosures the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries an operation id (`op_id`), two intervals `a = (a_lo,
//! a_hi)` and `b = (b_lo, b_hi)`, and one scalar `v`. The `op_id` selects one
//! of the golden operations:
//!
//! * `0` `new`      — order the raw pair `(a_lo, a_hi)` into `(min, max)`.
//! * `1` `width`    — `next_up(a_hi - a_lo)` (scalar in `lo`/`hi`).
//! * `2` `midpoint` — `a_lo + (a_hi - a_lo) * 0.5` (scalar in `lo`/`hi`).
//! * `3` `contains` — `a_lo <= v && v <= a_hi` (in `flag`).
//! * `4` `overlaps` — `a_lo <= b_hi && b_lo <= a_hi` (in `flag`).
//! * `5` `hull`     — `(min(a_lo, b_lo), max(a_hi, b_hi))`.
//! * `6` `intersect`— `(max(a_lo, b_lo), min(a_hi, b_hi))`, empty → `valid = 0`.
//! * `7` `abs`      — conservative absolute value.
//! * `8` `sqrt`     — `(next_down(sqrt(max(a_lo, 0))), next_up(sqrt(max(a_hi,
//!   0))))`.
//! * `9` `neg`      — `(-a_hi, -a_lo)`.
//! * `10` `add`     — `(next_down(a_lo + b_lo), next_up(a_hi + b_hi))`.
//! * `11` `sub`     — `(next_down(a_lo - b_hi), next_up(a_hi - b_lo))`.
//! * `12` `mul`     — outward-rounded four-corner product.
//! * `13` `div`     — divisor straddling zero → unbounded `[-inf, +inf]`,
//!   else outward-rounded four-corner quotient.
//!
//! Any `op_id > 13` is rejected (`valid = 0`).
//!
//! # Correctness model
//!
//! The `ULP` stepping is pure integer bit arithmetic (`bitcast`, `+ 1`/`- 1`),
//! so `next_up`/`next_down` are bit-exact given the same input. The continuous
//! bound arithmetic (adds, subtracts, products, quotients) threads through
//! operators a `GPU` may contract, so `CPU` and `GPU` are not necessarily
//! bit-exact; the continuous bounds are compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`), while the
//! discrete `flag` and `valid` words are compared exactly.
//!
//! # Degenerate inputs
//!
//! An empty intersection yields `valid = 0`; an `op_id > 13` yields `valid = 0`.
//! A `div` with a divisor straddling zero returns the unbounded interval
//! `[-inf, +inf]`, constructed from the exact `IEEE 754` bit patterns. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `bitcast`, `abs`,
//! `min`, `max`, `sqrt`, `+ - * /`, `select` and unsigned index/bit arithmetic
//! — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `f32`
//! remainder, and no `64`-bit or `16`-bit types. `NaN`/`inf` special cases in
//! the `ULP` steppers are detected with integer exponent/mantissa masks rather
//! than any bare `f32` equality, so they survive `Metal` fast-math; interval
//! comparisons use ordered `<=`/`<`/`>=`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::interval`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` interval-arithmetic kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `prism_math::interval`; see the module documentation for the
/// per-`op_id` closed forms.
const INTERVAL_ROUNDED_WGSL: &str = r#"
// Interval-arithmetic twin: one thread per query selects a golden operation by
// op_id and computes the outward-rounded result. The ULP steppers detect NaN
// and +/-inf with integer exponent/mantissa masks (no bare f32 equality), so
// they survive Metal fast-math; interval compares are all ordered <= / < / >=.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    op_id: u32,
    a_lo: f32,
    a_hi: f32,
    b_lo: f32,
    b_hi: f32,
    sv: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    lo: f32,
    hi: f32,
    flag: u32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Smallest f32 strictly greater than x. NaN and +inf return unchanged; both
// signed zeros step to the smallest positive subnormal. Specials are detected
// by integer masks so Metal fast-math cannot fold them away.
fn next_up(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let exp = bits & 0x7f800000u;
    let mant = bits & 0x007fffffu;
    let is_nan = (exp == 0x7f800000u) && (mant != 0u);
    let is_pinf = (bits == 0x7f800000u);
    if (is_nan || is_pinf) {
        return x;
    }
    let is_zero = ((bits & 0x7fffffffu) == 0u);
    if (is_zero) {
        return bitcast<f32>(1u);
    }
    var stepped: u32;
    if ((bits >> 31u) == 0u) {
        stepped = bits + 1u;
    } else {
        stepped = bits - 1u;
    }
    return bitcast<f32>(stepped);
}

// Smallest f32 strictly less than x. NaN and -inf return unchanged; both signed
// zeros step to the smallest negative subnormal.
fn next_down(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let exp = bits & 0x7f800000u;
    let mant = bits & 0x007fffffu;
    let is_nan = (exp == 0x7f800000u) && (mant != 0u);
    let is_ninf = (bits == 0xff800000u);
    if (is_nan || is_ninf) {
        return x;
    }
    let is_zero = ((bits & 0x7fffffffu) == 0u);
    if (is_zero) {
        return bitcast<f32>(0x80000001u);
    }
    var stepped: u32;
    if ((bits >> 31u) == 0u) {
        stepped = bits - 1u;
    } else {
        stepped = bits + 1u;
    }
    return bitcast<f32>(stepped);
}

fn min4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    return min(min(min(a, b), c), d);
}

fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    return max(max(max(a, b), c), d);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let op = q.op_id;
    let alo = q.a_lo;
    let ahi = q.a_hi;
    let blo = q.b_lo;
    let bhi = q.b_hi;
    let sv = q.sv;

    var lo = 0.0;
    var hi = 0.0;
    var flag = 0u;
    var valid = 1u;

    if (op == 0u) {
        lo = min(alo, ahi);
        hi = max(alo, ahi);
    } else if (op == 1u) {
        let w = next_up(ahi - alo);
        lo = w;
        hi = w;
    } else if (op == 2u) {
        let m = alo + (ahi - alo) * 0.5;
        lo = m;
        hi = m;
    } else if (op == 3u) {
        let c = (alo <= sv) && (sv <= ahi);
        flag = select(0u, 1u, c);
    } else if (op == 4u) {
        let c = (alo <= bhi) && (blo <= ahi);
        flag = select(0u, 1u, c);
    } else if (op == 5u) {
        lo = min(alo, blo);
        hi = max(ahi, bhi);
    } else if (op == 6u) {
        let il = max(alo, blo);
        let ih = min(ahi, bhi);
        if (il <= ih) {
            lo = il;
            hi = ih;
        } else {
            valid = 0u;
        }
    } else if (op == 7u) {
        if (alo >= 0.0) {
            lo = alo;
            hi = ahi;
        } else if (ahi <= 0.0) {
            lo = -ahi;
            hi = -alo;
        } else {
            lo = 0.0;
            hi = max(abs(alo), abs(ahi));
        }
    } else if (op == 8u) {
        let sl = sqrt(max(alo, 0.0));
        let sh = sqrt(max(ahi, 0.0));
        lo = next_down(sl);
        hi = next_up(sh);
    } else if (op == 9u) {
        lo = -ahi;
        hi = -alo;
    } else if (op == 10u) {
        lo = next_down(alo + blo);
        hi = next_up(ahi + bhi);
    } else if (op == 11u) {
        lo = next_down(alo - bhi);
        hi = next_up(ahi - blo);
    } else if (op == 12u) {
        let p0 = alo * blo;
        let p1 = alo * bhi;
        let p2 = ahi * blo;
        let p3 = ahi * bhi;
        lo = next_down(min4(p0, p1, p2, p3));
        hi = next_up(max4(p0, p1, p2, p3));
    } else if (op == 13u) {
        if ((blo <= 0.0) && (bhi >= 0.0)) {
            lo = bitcast<f32>(0xff800000u);
            hi = bitcast<f32>(0x7f800000u);
        } else {
            let r0 = alo / blo;
            let r1 = alo / bhi;
            let r2 = ahi / blo;
            let r3 = ahi / bhi;
            lo = next_down(min4(r0, r1, r2, r3));
            hi = next_up(max4(r0, r1, r2, r3));
        }
    } else {
        valid = 0u;
    }

    var out: Result;
    out.lo = lo;
    out.hi = hi;
    out.flag = flag;
    out.valid = valid;
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
/// the operation id, the two intervals, one scalar, padded to `8` words
/// (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op_id: u32,
    a_lo: f32,
    a_hi: f32,
    b_lo: f32,
    b_hi: f32,
    sv: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two bounds, a boolean `flag`, and the validity flag — `4` words
/// (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    lo: f32,
    hi: f32,
    flag: u32,
    valid: u32,
}

/// One interval-arithmetic query: an operation id, two intervals and a scalar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalRoundedQuery {
    /// Operation selector in `0..=13`; any other value is rejected.
    pub op_id: u32,
    /// Lower bound of interval `a` (for `new`, the first raw bound).
    pub a_lo: f32,
    /// Upper bound of interval `a` (for `new`, the second raw bound).
    pub a_hi: f32,
    /// Lower bound of interval `b`.
    pub b_lo: f32,
    /// Upper bound of interval `b`.
    pub b_hi: f32,
    /// Scalar operand, used by `contains`.
    pub v: f32,
}

impl IntervalRoundedQuery {
    /// Builds a query from an operation id, the two intervals and the scalar.
    #[must_use]
    pub fn new(op_id: u32, a_lo: f32, a_hi: f32, b_lo: f32, b_hi: f32, v: f32) -> Self {
        IntervalRoundedQuery {
            op_id,
            a_lo,
            a_hi,
            b_lo,
            b_hi,
            v,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `prism_math::interval` output for that operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalRoundedResult {
    /// Lower bound of an interval result, or a scalar result replicated here.
    pub lo: f32,
    /// Upper bound of an interval result, or a scalar result replicated here.
    pub hi: f32,
    /// Boolean result (`contains`, `overlaps`): `1` for true, else `0`.
    pub flag: u32,
    /// `1` when the operation produced a defined result, else `0` (empty
    /// intersection or out-of-range `op_id`).
    pub valid: u32,
}

/// Encodes one [`IntervalRoundedQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &IntervalRoundedQuery) -> GpuQuery {
    GpuQuery {
        op_id: q.op_id,
        a_lo: q.a_lo,
        a_hi: q.a_hi,
        b_lo: q.b_lo,
        b_hi: q.b_hi,
        sv: q.v,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`IntervalRoundedResult`].
fn decode_result(raw: &GpuResult) -> IntervalRoundedResult {
    IntervalRoundedResult {
        lo: raw.lo,
        hi: raw.hi,
        flag: raw.flag,
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

/// A compiled, reusable interval-arithmetic compute pipeline, twinning the `CPU`
/// golden `prism_math::interval`.
pub struct GpuIntervalRounded {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuIntervalRounded {
    /// Compiles the interval-arithmetic kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIntervalRounded {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_interval_rounded"),
            source: ShaderSource::Wgsl(INTERVAL_ROUNDED_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_interval_rounded_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_interval_rounded_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_interval_rounded_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuIntervalRounded {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`IntervalRoundedResult`] per input, in order.
    ///
    /// The `flag` and `valid` words match the reference exactly and the
    /// continuous bounds to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[IntervalRoundedQuery],
    ) -> Vec<IntervalRoundedResult> {
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
            label: Some("prism_volumetric_interval_rounded_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_interval_rounded_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_interval_rounded_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_interval_rounded_bind_group"),
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
            label: Some("prism_volumetric_interval_rounded_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_interval_rounded_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_interval_rounded_pass"),
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
