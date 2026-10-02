//! `wgpu` compute twin of the signed `Q16.16` fixed-point integer core
//! ([`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16),
//! particle design §25, §29).
//!
//! The `CPU` golden
//! [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16)
//! owns a deterministic, two's-complement [`i32`] number type whose stored
//! integer `r` represents the real value `r / 65536`. Because every add,
//! subtract, shift and comparison is plain integer arithmetic, the type is the
//! bit-for-bit lock-step currency between the reference path and a future `GPU`
//! kernel. [`GpuFixedPointQ16`] is the on-device twin: one thread evaluates one
//! query, so a passing real-device parity test is direct evidence the ported
//! kernel reproduces the exact two's-complement wrap, saturation and masking
//! the reference computes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The twin mirrors the `14` purely-[`i32`] free functions line for line, routed
//! through an operation selector so one dispatch can mix every operation in a
//! single batch:
//! [`q_from_int`](prism_render_architecture::particle::fixed_point_q16::q_from_int),
//! [`q_to_int_trunc`](prism_render_architecture::particle::fixed_point_q16::q_to_int_trunc),
//! [`q_to_int_floor`](prism_render_architecture::particle::fixed_point_q16::q_to_int_floor),
//! [`q_add`](prism_render_architecture::particle::fixed_point_q16::q_add),
//! [`q_sub`](prism_render_architecture::particle::fixed_point_q16::q_sub),
//! [`q_neg`](prism_render_architecture::particle::fixed_point_q16::q_neg),
//! [`q_add_sat`](prism_render_architecture::particle::fixed_point_q16::q_add_sat),
//! [`q_sub_sat`](prism_render_architecture::particle::fixed_point_q16::q_sub_sat),
//! [`q_abs`](prism_render_architecture::particle::fixed_point_q16::q_abs),
//! [`q_min`](prism_render_architecture::particle::fixed_point_q16::q_min),
//! [`q_max`](prism_render_architecture::particle::fixed_point_q16::q_max),
//! [`q_clamp`](prism_render_architecture::particle::fixed_point_q16::q_clamp),
//! [`q_floor`](prism_render_architecture::particle::fixed_point_q16::q_floor)
//! and
//! [`q_frac`](prism_render_architecture::particle::fixed_point_q16::q_frac).
//!
//! # What is not twinned
//!
//! [`q_mul`](prism_render_architecture::particle::fixed_point_q16::q_mul),
//! [`q_div`](prism_render_architecture::particle::fixed_point_q16::q_div) and
//! [`q_lerp`](prism_render_architecture::particle::fixed_point_q16::q_lerp)
//! are intentionally excluded: each needs a `64`-bit [`i64`] intermediate to
//! hold the full product or shifted numerator, and `WGSL` has no `64`-bit
//! integer type (only `i32`, `u32`, `f32` and `bool`). They are reproduced by a
//! sibling `fixed_point_q16_muldiv` twin that simulates the `64`-bit path with a
//! `u32` pair. The `f32` boundary conversions
//! [`q_from_f32`](prism_render_architecture::particle::fixed_point_q16::q_from_f32)
//! and
//! [`q_to_f32`](prism_render_architecture::particle::fixed_point_q16::q_to_f32)
//! are likewise out of scope here because they are not part of the pure-integer
//! subset.
//!
//! # Correctness model
//!
//! Every twinned operation is exact integer arithmetic — two's-complement
//! wrapping add/subtract/negate, saturating add/subtract/absolute, min/max,
//! clamp and bit masking — so the `CPU` and `GPU` agree bit for bit. The parity
//! test therefore asserts a strict `==` on every output word with no tolerance:
//! any mismatch is a genuine port defect. The wrap operations rely on `WGSL`
//! integer arithmetic wrapping modulo `2^32` exactly as Rust's `wrapping_*`
//! helpers do; the saturating operations reproduce the reference overflow
//! detection with a sign-bit test on `~(a ^ b) & (a ^ s)` instead of calling a
//! library helper the shader does not have.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer `+ - *`,
//! integer `/`, the bitwise operators `& ^ ~`, the shifts `<<` and `>>` and a
//! `switch` on the operation code — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `sqrt` and no `64`-bit integer, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of integer operations, so the kernel
//! provably terminates.
//!
//! # Degenerate inputs
//!
//! An empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized. The caller must pass `lo <= hi` to the
//! clamp operation, matching the reference contract.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fixed_point_q16`
//! 的纯 `i32` 子集；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `Q16.16` integer kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden
/// [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16)
/// pure-`i32` functions line for line; see the module documentation for the
/// algorithm.
const FIXED_POINT_Q16_WGSL: &str = r#"
// Signed Q16.16 fixed-point integer twin: one thread per query dispatches on an
// operation code to one of the 14 pure-i32 operations and writes the resulting
// i32. It mirrors the CPU golden particle::fixed_point_q16 line for line, uses
// only the portable core-WGSL subset (integer + - * /, the bitwise operators
// & ^ ~, the shifts << and >>, and a switch on the op code), has no 64-bit
// integer and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12. Every operation is exact integer arithmetic, so CPU and GPU agree
// bit for bit. There is no loop, so the kernel provably terminates.
//
// Provenance: twinned from this repository's particle::fixed_point_q16 pure-i32
// subset; no third-party engine source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 16-byte std430 stride matching the host `GpuQuery`: the three i32
// operands and the u32 operation code. `b` is unused by the single-operand ops
// and `c` is used only by the clamp op.
struct Query {
    a: i32,
    b: i32,
    c: i32,
    op: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<i32>;

// i32::MAX, used as the positive saturation extreme.
const Q_MAX: i32 = 2147483647;
// i32::MIN, written as `-2147483647 - 1` because `-2147483648` is not a valid
// i32 literal (its magnitude is one past i32::MAX before negation).
const Q_MIN: i32 = -2147483647 - 1;

// q_from_int: `i << 16` promotes an integer to Q16.16. The shift amount is u32
// as WGSL requires; high bits that overflow are discarded, matching the
// reference `i << FRAC_BITS`.
fn q_from_int(a: i32) -> i32 {
    return a << 16u;
}

// q_to_int_trunc: integer divide by ONE rounds toward zero, matching the
// reference `q.0 / ONE`.
fn q_to_int_trunc(a: i32) -> i32 {
    return a / 65536;
}

// q_to_int_floor: an arithmetic right shift keeps the sign and rounds toward
// negative infinity, matching the reference `q.0 >> FRAC_BITS`.
fn q_to_int_floor(a: i32) -> i32 {
    return a >> 16u;
}

// q_add: two's-complement wrapping add; WGSL integer `+` wraps modulo 2^32,
// matching the reference `wrapping_add`.
fn q_add(a: i32, b: i32) -> i32 {
    return a + b;
}

// q_sub: two's-complement wrapping subtract, matching the reference
// `wrapping_sub`.
fn q_sub(a: i32, b: i32) -> i32 {
    return a - b;
}

// q_neg: two's-complement wrapping negate; `0 - a` wraps i32::MIN back to
// itself, matching the reference `wrapping_neg`.
fn q_neg(a: i32) -> i32 {
    return 0 - a;
}

// q_add_sat: saturating add. The sum overflows exactly when the operands share
// a sign and the result differs from it; `~(a ^ b)` is negative when the signs
// agree and `(a ^ s)` is negative when the result's sign flipped, so their AND
// is negative only on overflow. This reproduces the reference `saturating_add`
// without a library helper.
fn q_add_sat(a: i32, b: i32) -> i32 {
    let s = a + b;
    if (((~(a ^ b)) & (a ^ s)) < 0) {
        if (a >= 0) {
            return Q_MAX;
        }
        return Q_MIN;
    }
    return s;
}

// q_sub_sat: saturating subtract. The difference overflows exactly when the
// operands differ in sign and the result's sign flipped away from `a`,
// reproducing the reference `saturating_sub`.
fn q_sub_sat(a: i32, b: i32) -> i32 {
    let d = a - b;
    if (((a ^ b) & (a ^ d)) < 0) {
        if (a >= 0) {
            return Q_MAX;
        }
        return Q_MIN;
    }
    return d;
}

// q_abs: saturating absolute value; the sole overflow case is i32::MIN, which
// saturates to i32::MAX, matching the reference `saturating_abs`.
fn q_abs(a: i32) -> i32 {
    if (a == Q_MIN) {
        return Q_MAX;
    }
    if (a < 0) {
        return 0 - a;
    }
    return a;
}

// q_min: the smaller of two values by exact integer comparison.
fn q_min(a: i32, b: i32) -> i32 {
    if (a <= b) {
        return a;
    }
    return b;
}

// q_max: the larger of two values by exact integer comparison.
fn q_max(a: i32, b: i32) -> i32 {
    if (a >= b) {
        return a;
    }
    return b;
}

// q_clamp: min(max(v, lo), hi); the caller guarantees lo <= hi, matching the
// reference contract.
fn q_clamp(v: i32, lo: i32, hi: i32) -> i32 {
    return q_min(q_max(v, lo), hi);
}

// q_floor: clearing the low 16 bits rounds toward negative infinity. The mask
// `-65536` is `!(ONE - 1)` in two's complement, matching the reference.
fn q_floor(a: i32) -> i32 {
    return a & (-65536);
}

// q_frac: masking the low 16 bits yields the fractional part in [0, 1),
// matching the reference `q.0 & (ONE - 1)`.
fn q_frac(a: i32) -> i32 {
    return a & 65535;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    var value: i32 = 0;
    switch (q.op) {
        case 0u: { value = q_from_int(q.a); }
        case 1u: { value = q_to_int_trunc(q.a); }
        case 2u: { value = q_to_int_floor(q.a); }
        case 3u: { value = q_add(q.a, q.b); }
        case 4u: { value = q_sub(q.a, q.b); }
        case 5u: { value = q_neg(q.a); }
        case 6u: { value = q_add_sat(q.a, q.b); }
        case 7u: { value = q_sub_sat(q.a, q.b); }
        case 8u: { value = q_abs(q.a); }
        case 9u: { value = q_min(q.a, q.b); }
        case 10u: { value = q_max(q.a, q.b); }
        case 11u: { value = q_clamp(q.a, q.b, q.c); }
        case 12u: { value = q_floor(q.a); }
        case 13u: { value = q_frac(q.a); }
        default: { value = 0; }
    }
    results[idx] = value;
}
"#;

/// The `Q16.16` integer operation an individual query selects.
///
/// Each variant names one of the `14` purely-[`i32`] golden functions mirrored
/// by this twin; the stable [`GpuQ16Op::to_code`] mapping is the operation code
/// the kernel's `switch` dispatches on. The single-operand operations
/// ([`GpuQ16Op::FromInt`], [`GpuQ16Op::ToIntTrunc`], [`GpuQ16Op::ToIntFloor`],
/// [`GpuQ16Op::Neg`], [`GpuQ16Op::Abs`], [`GpuQ16Op::Floor`] and
/// [`GpuQ16Op::Frac`]) read only [`GpuQ16Query::a`]; the two-operand operations
/// also read [`GpuQ16Query::b`]; [`GpuQ16Op::Clamp`] additionally reads
/// [`GpuQ16Query::c`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fixed_point_q16`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuQ16Op {
    /// Promote an integer to `Q16.16` (`q_from_int`).
    FromInt,
    /// Truncate toward zero to an integer (`q_to_int_trunc`).
    ToIntTrunc,
    /// Floor toward negative infinity to an integer (`q_to_int_floor`).
    ToIntFloor,
    /// Wrapping add (`q_add`).
    Add,
    /// Wrapping subtract (`q_sub`).
    Sub,
    /// Wrapping negate (`q_neg`).
    Neg,
    /// Saturating add (`q_add_sat`).
    AddSat,
    /// Saturating subtract (`q_sub_sat`).
    SubSat,
    /// Saturating absolute value (`q_abs`).
    Abs,
    /// Minimum (`q_min`).
    Min,
    /// Maximum (`q_max`).
    Max,
    /// Clamp into `[lo, hi]` (`q_clamp`).
    Clamp,
    /// Floor to a whole `Q16.16` value (`q_floor`).
    Floor,
    /// Fractional part in `[0, 1)` (`q_frac`).
    Frac,
}

impl GpuQ16Op {
    /// The stable `u32` code the kernel `switch` dispatches on.
    #[must_use]
    const fn to_code(self) -> u32 {
        match self {
            GpuQ16Op::FromInt => 0,
            GpuQ16Op::ToIntTrunc => 1,
            GpuQ16Op::ToIntFloor => 2,
            GpuQ16Op::Add => 3,
            GpuQ16Op::Sub => 4,
            GpuQ16Op::Neg => 5,
            GpuQ16Op::AddSat => 6,
            GpuQ16Op::SubSat => 7,
            GpuQ16Op::Abs => 8,
            GpuQ16Op::Min => 9,
            GpuQ16Op::Max => 10,
            GpuQ16Op::Clamp => 11,
            GpuQ16Op::Floor => 12,
            GpuQ16Op::Frac => 13,
        }
    }
}

/// One `Q16.16` integer query: the operation and its raw [`i32`] operands.
///
/// The operands are raw `Q16.16` backing integers (the reference `Q16_16`
/// field), not real-valued floats. `a` is the primary operand every operation
/// reads; `b` is the second operand of the two-operand operations and the lower
/// clamp bound; `c` is the upper clamp bound, read only by [`GpuQ16Op::Clamp`].
/// Unused operands are ignored by the kernel.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fixed_point_q16`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuQ16Query {
    /// Primary operand (the value, or the clamp input `v`).
    pub a: i32,
    /// Second operand (two-operand right-hand side, or the clamp lower bound).
    pub b: i32,
    /// Third operand (the clamp upper bound); read only by [`GpuQ16Op::Clamp`].
    pub c: i32,
    /// Which golden operation to evaluate.
    pub op: GpuQ16Op,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`FIXED_POINT_Q16_WGSL`]: the query count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `16`-byte `std430` stride matching `Query` in the
/// shader: three `i32` operands and the `u32` operation code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Primary operand.
    a: i32,
    /// Second operand (or clamp lower bound).
    b: i32,
    /// Third operand (clamp upper bound).
    c: i32,
    /// Operation code from [`GpuQ16Op::to_code`].
    op: u32,
}

impl GpuQuery {
    /// Packs a [`GpuQ16Query`] into the `std430` upload layout.
    fn from_query(query: &GpuQ16Query) -> GpuQuery {
        GpuQuery {
            a: query.a,
            b: query.b,
            c: query.c,
            op: query.op.to_code(),
        }
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

/// A compiled, reusable `Q16.16` integer-operation pipeline.
pub struct GpuFixedPointQ16 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFixedPointQ16 {
    /// Compiles the `Q16.16` integer kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFixedPointQ16 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fixed_point_q16"),
            source: ShaderSource::Wgsl(FIXED_POINT_Q16_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFixedPointQ16 {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning the resulting raw
    /// `Q16.16` [`i32`] for each query in input order.
    ///
    /// Each returned word equals the corresponding golden function evaluated on
    /// the query operands bit for bit. An empty `queries` slice yields an empty
    /// vector — storage buffers cannot be zero-sized, so it is handled by an
    /// early return before any dispatch.
    #[must_use]
    pub fn run(&self, ctx: &GpuContext, queries: &[GpuQ16Query]) -> Vec<i32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<i32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fixed_point_q16_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, i32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw
    }
}
