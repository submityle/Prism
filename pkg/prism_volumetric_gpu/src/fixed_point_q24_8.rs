//! `wgpu` compute twin of the pure-`i32` subset of the signed `Q24.8`
//! fixed-point arithmetic contract
//! ([`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8),
//! particle design: deterministic simulation state).
//!
//! The `CPU` golden
//! [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8)
//! owns a deterministic, pure-integer number type whose stored [`i32`] `raw`
//! represents the real value `raw / 256` (scale `2^8 = 256`). Because every
//! add, subtract, shift and mask is exact two's-complement integer algebra, the
//! same operation runs bit-for-bit identically on a `GPU` lane.
//! [`GpuFixedPointQ24_8`] is the on-device twin: one thread evaluates one
//! operation query, so a passing real-device parity test is direct evidence the
//! ported kernel runs the same integer arithmetic the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Exactly the ten pure-`i32` free functions of the golden module are mirrored,
//! dispatched by a per-query operation selector:
//! [`from_int`](prism_render_architecture::particle::fixed_point_q24_8::from_int),
//! [`to_int_trunc`](prism_render_architecture::particle::fixed_point_q24_8::to_int_trunc),
//! [`add`](prism_render_architecture::particle::fixed_point_q24_8::add),
//! [`saturating_add`](prism_render_architecture::particle::fixed_point_q24_8::saturating_add),
//! [`sub`](prism_render_architecture::particle::fixed_point_q24_8::sub),
//! [`saturating_sub`](prism_render_architecture::particle::fixed_point_q24_8::saturating_sub),
//! [`neg`](prism_render_architecture::particle::fixed_point_q24_8::neg),
//! [`floor`](prism_render_architecture::particle::fixed_point_q24_8::floor),
//! [`fract`](prism_render_architecture::particle::fixed_point_q24_8::fract) and
//! [`abs`](prism_render_architecture::particle::fixed_point_q24_8::abs). Each
//! query carries two `i32` operands `a` and `b` (the unary operations ignore
//! `b`) and a [`GpuQ24Op`] selector; the kernel returns one `i32` word per
//! query. The host packs the selector through its stable
//! [`GpuQ24Op::to_u32`] code and a `WGSL` `switch` dispatches the ten branches
//! line for line against the reference.
//!
//! # What is deliberately not twinned
//!
//! The golden
//! [`from_ratio`](prism_render_architecture::particle::fixed_point_q24_8::from_ratio),
//! [`mul`](prism_render_architecture::particle::fixed_point_q24_8::mul) and
//! [`div`](prism_render_architecture::particle::fixed_point_q24_8::div) all use
//! an [`i64`] intermediate, and `WGSL` has no 64-bit integer type. They are
//! therefore out of scope for this twin and are instead reproduced by the
//! companion `u32`-emulation module `fixed_point_q24_8_muldiv`. This module
//! twins only the ten operations that are closed over `i32`.
//!
//! # Correctness model
//!
//! Every twinned operation is exact two's-complement `i32` algebra — wrapping
//! add/subtract/negate, arithmetic right shift, bitwise mask, and a pure-`i32`
//! overflow-detecting saturate — with no `f32` and no rounding anywhere, so the
//! `CPU` reference and the `GPU` kernel agree bit for bit. The parity test
//! therefore asserts an exact `==` on every returned word with no tolerance:
//! any mismatch is a genuine port bug. The wrapping operations
//! (`add`, `sub`, `neg`, `abs` at [`i32::MIN`]) agree because both sides reduce
//! modulo `2^32`; the saturating operations agree because the kernel reproduces
//! the reference clamp with the classic sign-bit overflow test instead of a
//! wider accumulator.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - *`, bitwise
//! `and` / `xor` / `not`, arithmetic right shift, unsigned/signed compares and a
//! `switch` on the operation code — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no `sqrt`, no 64-bit integer and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. The constant [`i32::MIN`]
//! is written `-2147483647 - 1` to avoid a literal that would overflow before
//! negation. There is no loop: each thread performs a fixed, bounded sequence of
//! integer arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fixed_point_q24_8`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `Q24.8` pure-`i32` arithmetic kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8)
/// free functions branch for branch; see the module documentation for the
/// algorithm.
const FIXED_POINT_Q24_8_WGSL: &str = r#"
// Q24.8 pure-i32 arithmetic twin: one thread per query reproduces one of the ten
// i32-closed operations (from_int, to_int_trunc, add, saturating_add, sub,
// saturating_sub, neg, floor, fract, abs). It mirrors the CPU golden
// particle::fixed_point_q24_8 free functions branch for branch, uses only the
// portable core-WGSL subset (+ - *, bitwise and/xor/not, arithmetic shift,
// compares and a switch on the op code) and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12. Every operation is exact
// two's-complement i32 algebra, so the CPU and GPU agree bit for bit. There is
// no loop, so the kernel provably terminates.
//
// Provenance: twinned from this repository's particle::fixed_point_q24_8; no
// third-party engine source or derived code.

// The Q24.8 scale factor 2^8 and its fractional bit count, matching SCALE and
// SCALE_BITS in the reference.
const SCALE: i32 = 256;
const SCALE_BITS: u32 = 8u;
// i32::MIN written as -2147483647 - 1 so no literal overflows before negation,
// and i32::MAX as its own literal.
const I32_MIN: i32 = -2147483647 - 1;
const I32_MAX: i32 = 2147483647;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The two operands; unary operations use only `a`. `op` is the GpuQ24Op
    // code (0..=9) and a pad word fills the std430 slot.
    a: i32,
    b: i32,
    op: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<i32>;

// from_int: raw = i * 256, wrapping on overflow exactly like the reference.
fn q_from_int(a: i32) -> i32 {
    return a * SCALE;
}

// to_int_trunc: arithmetic right shift by SCALE_BITS, which floors toward
// negative infinity (not truncation toward zero), matching the reference.
fn q_to_int_trunc(a: i32) -> i32 {
    return a >> SCALE_BITS;
}

// add: wrapping two's-complement add of the raw integers.
fn q_add(a: i32, b: i32) -> i32 {
    return a + b;
}

// saturating_add: wrapping sum, then clamp to the i32 extremes when the signed
// overflow test fires. Overflow happens only when a and b share a sign and the
// sum's sign differs, i.e. (~(a^b) & (a^s)) has its sign bit set.
fn q_saturating_add(a: i32, b: i32) -> i32 {
    let s = a + b;
    if (((~(a ^ b)) & (a ^ s)) < 0) {
        if (a >= 0) {
            return I32_MAX;
        }
        return I32_MIN;
    }
    return s;
}

// sub: wrapping two's-complement subtract of the raw integers.
fn q_sub(a: i32, b: i32) -> i32 {
    return a - b;
}

// saturating_sub: wrapping difference, then clamp when the signed overflow test
// fires. Overflow happens only when a and b differ in sign and the difference's
// sign differs from a, i.e. ((a^b) & (a^d)) has its sign bit set.
fn q_saturating_sub(a: i32, b: i32) -> i32 {
    let d = a - b;
    if (((a ^ b) & (a ^ d)) < 0) {
        if (a >= 0) {
            return I32_MAX;
        }
        return I32_MIN;
    }
    return d;
}

// neg: wrapping negate; 0 - i32::MIN wraps back to i32::MIN, as in the
// reference wrapping_neg.
fn q_neg(a: i32) -> i32 {
    return 0 - a;
}

// floor: clear the low SCALE_BITS fractional bits via raw & ~(SCALE-1); exact
// floor for both positive and negative values.
fn q_floor(a: i32) -> i32 {
    return a & ~(SCALE - 1);
}

// fract: the fractional part raw & (SCALE-1), always a non-negative value in
// [0, 256), consistent with floor's round-toward-negative-infinity semantics.
fn q_fract(a: i32) -> i32 {
    return a & (SCALE - 1);
}

// abs: wrapping absolute value; i32::MIN wraps back to i32::MIN (NOT saturating),
// matching the reference wrapping_abs.
fn q_abs(a: i32) -> i32 {
    if (a == I32_MIN) {
        return I32_MIN;
    }
    if (a < 0) {
        return 0 - a;
    }
    return a;
}

// Dispatch the operation selected by the stable code. Mirrors the reference free
// function set arm for arm.
fn apply_op(op: u32, a: i32, b: i32) -> i32 {
    switch (op) {
        case 0u: { return q_from_int(a); }
        case 1u: { return q_to_int_trunc(a); }
        case 2u: { return q_add(a, b); }
        case 3u: { return q_saturating_add(a, b); }
        case 4u: { return q_sub(a, b); }
        case 5u: { return q_saturating_sub(a, b); }
        case 6u: { return q_neg(a); }
        case 7u: { return q_floor(a); }
        case 8u: { return q_fract(a); }
        default: { return q_abs(a); }  // 9u
    }
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = apply_op(q.op, q.a, q.b);
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`FIXED_POINT_Q24_8_WGSL`].
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
/// four `4`-byte words (two `i32` operands, one `u32` op code and one pad).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First operand; for the unary operations this is the sole input.
    a: i32,
    /// Second operand; ignored by the unary operations.
    b: i32,
    /// Operation selector code (`0..=9`).
    op: u32,
    /// Padding word.
    pad0: u32,
}

/// The ten pure-`i32` `Q24.8` operations this twin reproduces, dispatched by a
/// stable `u32` code the host packs into each query.
///
/// Each variant names the golden free function it mirrors. The unary operations
/// ([`ToIntTrunc`](Self::ToIntTrunc), [`Neg`](Self::Neg), [`Floor`](Self::Floor),
/// [`Fract`](Self::Fract), [`Abs`](Self::Abs)) and [`FromInt`](Self::FromInt)
/// use only the `a` operand of a query.
///
/// Provenance: twinned from this repository's
/// [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuQ24Op {
    /// Mirrors
    /// [`from_int`](prism_render_architecture::particle::fixed_point_q24_8::from_int):
    /// `a * 256` (wrapping).
    FromInt,
    /// Mirrors
    /// [`to_int_trunc`](prism_render_architecture::particle::fixed_point_q24_8::to_int_trunc):
    /// `a >> 8` (arithmetic, floor).
    ToIntTrunc,
    /// Mirrors
    /// [`add`](prism_render_architecture::particle::fixed_point_q24_8::add):
    /// wrapping `a + b`.
    Add,
    /// Mirrors
    /// [`saturating_add`](prism_render_architecture::particle::fixed_point_q24_8::saturating_add):
    /// `a + b` clamped to the `i32` extremes.
    SaturatingAdd,
    /// Mirrors
    /// [`sub`](prism_render_architecture::particle::fixed_point_q24_8::sub):
    /// wrapping `a - b`.
    Sub,
    /// Mirrors
    /// [`saturating_sub`](prism_render_architecture::particle::fixed_point_q24_8::saturating_sub):
    /// `a - b` clamped to the `i32` extremes.
    SaturatingSub,
    /// Mirrors
    /// [`neg`](prism_render_architecture::particle::fixed_point_q24_8::neg):
    /// wrapping `0 - a`.
    Neg,
    /// Mirrors
    /// [`floor`](prism_render_architecture::particle::fixed_point_q24_8::floor):
    /// `a & !255`.
    Floor,
    /// Mirrors
    /// [`fract`](prism_render_architecture::particle::fixed_point_q24_8::fract):
    /// `a & 255`.
    Fract,
    /// Mirrors
    /// [`abs`](prism_render_architecture::particle::fixed_point_q24_8::abs):
    /// wrapping absolute value (`i32::MIN` maps to `i32::MIN`).
    Abs,
}

impl GpuQ24Op {
    /// Returns the stable device code (`0..=9`) the host packs into a query and
    /// the `WGSL` `switch` dispatches on.
    #[must_use]
    pub fn to_u32(self) -> u32 {
        match self {
            GpuQ24Op::FromInt => 0,
            GpuQ24Op::ToIntTrunc => 1,
            GpuQ24Op::Add => 2,
            GpuQ24Op::SaturatingAdd => 3,
            GpuQ24Op::Sub => 4,
            GpuQ24Op::SaturatingSub => 5,
            GpuQ24Op::Neg => 6,
            GpuQ24Op::Floor => 7,
            GpuQ24Op::Fract => 8,
            GpuQ24Op::Abs => 9,
        }
    }
}

/// One `Q24.8` operation query: a selector plus two `i32` operands.
///
/// For [`FromInt`](GpuQ24Op::FromInt) the operand `a` is the integer to scale;
/// for the other operations `a` (and `b` for the binary ones) are raw `Q24.8`
/// integers. The unary operations ignore `b`.
///
/// Provenance: twinned from this repository's
/// [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuQ24Query {
    /// First operand; the integer for [`FromInt`](GpuQ24Op::FromInt), otherwise
    /// a raw `Q24.8` value.
    pub a: i32,
    /// Second operand; used only by the binary operations
    /// ([`Add`](GpuQ24Op::Add), [`SaturatingAdd`](GpuQ24Op::SaturatingAdd),
    /// [`Sub`](GpuQ24Op::Sub), [`SaturatingSub`](GpuQ24Op::SaturatingSub)).
    pub b: i32,
    /// The operation to apply.
    pub op: GpuQ24Op,
}

/// Encodes one [`GpuQ24Query`] into its `std430` [`GpuQuery`] slot, packing the
/// operation enum through its stable [`GpuQ24Op::to_u32`] code.
fn encode_query(q: &GpuQ24Query) -> GpuQuery {
    GpuQuery {
        a: q.a,
        b: q.b,
        op: q.op.to_u32(),
        pad0: 0,
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

/// A compiled, reusable `Q24.8` pure-`i32` arithmetic compute pipeline, twinning
/// the `CPU` golden
/// [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8).
pub struct GpuFixedPointQ24_8 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFixedPointQ24_8 {
    /// Compiles the `Q24.8` arithmetic kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFixedPointQ24_8 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8"),
            source: ShaderSource::Wgsl(FIXED_POINT_Q24_8_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFixedPointQ24_8 {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one `i32` result per input, in
    /// order.
    ///
    /// Every output word equals the reference exactly, since every twinned
    /// operation is exact two's-complement `i32` algebra. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn run(&self, ctx: &GpuContext, queries: &[GpuQ24Query]) -> Vec<i32> {
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
            label: Some("prism_volumetric_fixed_point_q24_8_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<i32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_bind_group"),
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
            label: Some("prism_volumetric_fixed_point_q24_8_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fixed_point_q24_8_pass"),
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
        let out = bytemuck::cast_slice::<u8, i32>(&view).to_vec();
        drop(view);
        stage.unmap();

        out
    }
}
