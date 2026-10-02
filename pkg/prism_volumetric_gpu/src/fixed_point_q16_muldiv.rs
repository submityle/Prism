//! `wgpu` compute twin of the two `Q16.16` fixed-point operations that need a
//! `64`-bit intermediate — the saturating multiply
//! ([`q_mul`](prism_render_architecture::particle::fixed_point_q16::q_mul)) and
//! the lerp built on top of it
//! ([`q_lerp`](prism_render_architecture::particle::fixed_point_q16::q_lerp))
//! (particle design §25, §29 lock-step determinism).
//!
//! The `CPU` golden
//! [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16)
//! multiplies by widening both operands to [`i64`], biasing the product by half
//! a unit for round-to-nearest, arithmetic-shifting right by
//! [`Q16_16::FRAC_BITS`](prism_render_architecture::particle::fixed_point_q16::Q16_16::FRAC_BITS)
//! and clamping back into [`i32`] range
//! ([`clamp_to_i32`](prism_render_architecture::particle::fixed_point_q16)); the
//! lerp is `a + (b - a) * t` with wrapping additive helpers. `WGSL` has no
//! `i64`, so [`GpuQ16MulDiv`] rebuilds the exact `64`-bit signed path out of a
//! `(hi: u32, lo: u32)` word pair: it forms the unsigned `32 x 32 -> 64` magnitude
//! product, takes the `64`-bit two's complement when the signs differ, adds the
//! `ROUND_BIAS` of `32768` with carry propagation, arithmetic-shifts the pair
//! right by `16`, and finally reproduces the saturating `clamp_to_i32`. One
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel runs the identical `i64` arithmetic the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! - [`GpuQ16Mul`](GpuQ16MulOp::Mul) mirrors
//!   [`q_mul`](prism_render_architecture::particle::fixed_point_q16::q_mul): the
//!   full widen-multiply, round-half-up bias, arithmetic right shift and
//!   saturating clamp, reconstructed over the `(hi, lo)` word pair.
//! - [`GpuQ16Lerp`](GpuQ16MulOp::Lerp) mirrors
//!   [`q_lerp`](prism_render_architecture::particle::fixed_point_q16::q_lerp):
//!   `a + q_mul(b - a, t)`, where the `(b - a)` subtraction and the final
//!   addition use `WGSL`'s wrapping `i32` arithmetic, matching the golden
//!   `q_sub` / `q_add` two's-complement wrap.
//!
//! # The `64`-bit simulation
//!
//! `WGSL` offers only `i32`, `u32`, `f32` and `bool`, so the `i64` product is
//! simulated exactly:
//!
//! 1. Split each magnitude `|a|`, `|b|` (`i32::MIN` maps to `2^31`) into
//!    `16`-bit halves and form the four partials `ll`, `lh`, `hl`, `hh`; the
//!    cross terms accumulate into `cross`, giving the unsigned `64`-bit product
//!    `(hi, lo)`.
//! 2. When the operand signs differ, negate the pair in `64`-bit two's
//!    complement (`~lo + 1`, with a carry into `~hi`).
//! 3. Add the `ROUND_BIAS` of `32768` to `lo`, carrying into `hi`.
//! 4. Arithmetic-shift the pair right by `16`: the low word takes
//!    `(lo >> 16) | (hi << 16)` and the high word sign-extends through
//!    `bitcast<i32>(hi) >> 16`.
//! 5. Saturate: a value is in range only when its high word is the sign
//!    extension of bit `31` of the low word, otherwise it clamps to
//!    [`i32::MAX`] or [`i32::MIN`].
//!
//! This covers the full `i32 x i32` input domain, including the extreme
//! `i32::MIN * i32::MIN = 2^62` and every product that overflows `i32` and must
//! saturate at either extreme.
//!
//! # Correctness model
//!
//! Every step is exact integer bit algebra — the product, the round bias, the
//! arithmetic shift and the saturation are all computed with no rounding error,
//! so the `CPU` reference and the `GPU` kernel agree bit for bit. The parity
//! test therefore asserts an exact `==` on every `i32` output with no tolerance:
//! any mismatch is a genuine port bug (a wrong carry, a missed sign negation, a
//! shifted bit). There is no loop; each thread performs a fixed, bounded
//! sequence of integer arithmetic, so the kernel provably terminates.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — the bit operators
//! `>> << & | ~`, the arithmetic `+ - *`, unsigned and signed compares,
//! `select`, `bitcast` and a `switch` on the operation code — with no `i64`, no
//! transcendental call, no intrinsic and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. The host-side test `LCG` fixture
//! uses `u64`, exactly as the golden fixtures allow; the kernel itself stays in
//! the `32`-bit lane set.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16); no
//! third-party engine source or derived code.
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

/// The portable core-`WGSL` `Q16.16` multiply/lerp kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16)
/// line for line; see the module documentation for the `64`-bit simulation.
const FIXED_POINT_Q16_MULDIV_WGSL: &str = r#"
// Q16.16 multiply/lerp twin: one thread per query reproduces q_mul or q_lerp by
// simulating the signed 64-bit intermediate with a (hi, lo) u32 pair. It mirrors
// the CPU golden particle::fixed_point_q16 line for line, uses only the portable
// core-WGSL subset (bit ops, unsigned/signed compares, select, bitcast and a
// switch on the op code) and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. Every step is exact integer algebra, so the CPU and
// GPU agree bit for bit. There is no loop, so the kernel provably terminates.
//
// Provenance: twinned from this repository's particle::fixed_point_q16; no
// third-party engine source or derived code.

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operands: a and b feed q_mul; a, b and t (= c) feed q_lerp. op selects
    // which: 0 = q_mul, 1 = q_lerp.
    a: i32,
    b: i32,
    c: i32,
    op: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<i32>;

// Half an integer unit, the round-to-nearest bias added before the shift (2^15).
const ROUND_BIAS: u32 = 32768u;

// Magnitude |x| as a u32. i32::MIN has no positive i32 counterpart, so its
// magnitude is 2^31, produced here by the two's-complement negation of the bit
// pattern (which yields 0x80000000 unchanged).
fn u32_mag(x: i32) -> u32 {
    let u = bitcast<u32>(x);
    if (x >= 0) {
        return u;
    }
    return (~u) + 1u;
}

// q_mul: product = (i64)a * (i64)b; rounded = (product + 32768) >> 16 (arithmetic);
// clamp_to_i32(rounded). Rebuilt over a (hi, lo) word pair since WGSL has no i64.
fn q_mul(a: i32, b: i32) -> i32 {
    // 1) Unsigned 64-bit magnitude product |a| * |b|.
    let sign_neg = (a < 0) != (b < 0);
    let ua = u32_mag(a);
    let ub = u32_mag(b);
    let a0 = ua & 0xFFFFu;
    let a1 = ua >> 16u;
    let b0 = ub & 0xFFFFu;
    let b1 = ub >> 16u;
    let ll = a0 * b0;
    let lh = a0 * b1;
    let hl = a1 * b0;
    let hh = a1 * b1;
    let cross = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    var lo = (ll & 0xFFFFu) | (cross << 16u);
    var hi = hh + (lh >> 16u) + (hl >> 16u) + (cross >> 16u);
    // 2) Signs differ: negate the 64-bit value in two's complement.
    if (sign_neg) {
        lo = ~lo + 1u;
        hi = ~hi + select(0u, 1u, lo == 0u);
    }
    // 3) Add ROUND_BIAS with carry propagation into the high word.
    let nlo = lo + ROUND_BIAS;
    let carry = select(0u, 1u, nlo < lo);
    lo = nlo;
    hi = hi + carry;
    // 4) Arithmetic right shift by 16 of the (hi, lo) pair.
    let s_lo = (lo >> 16u) | (hi << 16u);
    let s_hi = bitcast<u32>(bitcast<i32>(hi) >> 16u);
    // 5) clamp_to_i32: in range only when s_hi is the sign extension of bit 31
    // of s_lo; otherwise saturate at i32::MAX or i32::MIN.
    let neg = (s_hi & 0x80000000u) != 0u;
    if (!neg) {
        if (s_hi == 0u && (s_lo & 0x80000000u) == 0u) {
            return bitcast<i32>(s_lo);
        }
        return 2147483647;
    }
    if (s_hi == 0xFFFFFFFFu && (s_lo & 0x80000000u) != 0u) {
        return bitcast<i32>(s_lo);
    }
    return -2147483647 - 1;
}

// q_lerp: a + q_mul(b - a, t). The subtraction and the final addition use WGSL's
// wrapping i32 arithmetic, matching the golden q_sub / q_add two's-complement wrap.
fn q_lerp(a: i32, b: i32, t: i32) -> i32 {
    let d = b - a;
    return a + q_mul(d, t);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var out: i32;
    switch (q.op) {
        case 0u: {
            out = q_mul(q.a, q.b);
        }
        case 1u: {
            out = q_lerp(q.a, q.b, q.c);
        }
        default: {
            out = 0;
        }
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`FIXED_POINT_Q16_MULDIV_WGSL`].
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
/// four `4`-byte words (three operands plus the operation code).
///
/// Provenance: twinned from this repository's
/// [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16); no
/// third-party engine source or derived code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First operand (`a`), the left factor for `q_mul` and the start for `q_lerp`.
    a: i32,
    /// Second operand (`b`), the right factor for `q_mul` and the end for `q_lerp`.
    b: i32,
    /// Third operand (`c`), the interpolation parameter `t` for `q_lerp` (ignored
    /// by `q_mul`).
    c: i32,
    /// Operation code: `0` selects `q_mul`, `1` selects `q_lerp`.
    op: u32,
}

/// Which `Q16.16` operation a query resolves to on the device.
///
/// The host packs this into each query's operation code; the kernel dispatches
/// on it with a `switch`.
///
/// Provenance: twinned from this repository's
/// [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuQ16MulOp {
    /// Saturating round-to-nearest multiply, mirroring
    /// [`q_mul`](prism_render_architecture::particle::fixed_point_q16::q_mul).
    Mul,
    /// Linear interpolation `a + (b - a) * t`, mirroring
    /// [`q_lerp`](prism_render_architecture::particle::fixed_point_q16::q_lerp).
    Lerp,
}

impl GpuQ16MulOp {
    /// The stable device code for this operation (`0` for `Mul`, `1` for `Lerp`).
    #[must_use]
    const fn to_u32(self) -> u32 {
        match self {
            GpuQ16MulOp::Mul => 0,
            GpuQ16MulOp::Lerp => 1,
        }
    }
}

/// One `Q16.16` multiply or lerp query: the three raw `i32` operands and the
/// selected operation.
///
/// Operands are raw `Q16.16` fixed-point integers (the `.0` of a
/// [`Q16_16`](prism_render_architecture::particle::fixed_point_q16::Q16_16)). For
/// [`GpuQ16MulOp::Mul`] only `a` and `b` are read; for [`GpuQ16MulOp::Lerp`] all
/// three are read, with `t` as the interpolation parameter.
///
/// Provenance: twinned from this repository's
/// [`fixed_point_q16`](prism_render_architecture::particle::fixed_point_q16); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuQ16MulDivQuery {
    /// First operand (left factor for `Mul`, start value for `Lerp`).
    pub a: i32,
    /// Second operand (right factor for `Mul`, end value for `Lerp`).
    pub b: i32,
    /// Interpolation parameter `t` for `Lerp`; ignored by `Mul`.
    pub t: i32,
    /// Which operation this query resolves to.
    pub op: GpuQ16MulOp,
}

/// Encodes one [`GpuQ16MulDivQuery`] into its `std430` [`GpuQuery`] slot, packing
/// the operation enum through its stable `to_u32` code.
fn encode_query(q: &GpuQ16MulDivQuery) -> GpuQuery {
    GpuQuery {
        a: q.a,
        b: q.b,
        c: q.t,
        op: q.op.to_u32(),
    }
}

/// A compiled, reusable `Q16.16` multiply/lerp kernel. The single entry point
/// `solve` resolves one query per thread.
pub struct GpuQ16MulDiv {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQ16MulDiv {
    /// Compiles the `solve` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (bit ops, compares,
    /// `select`, `bitcast` and a `switch`), so no optional device feature is
    /// required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQ16MulDiv {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv"),
            source: ShaderSource::Wgsl(FIXED_POINT_Q16_MULDIV_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQ16MulDiv {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one raw `Q16.16` `i32`
    /// result per input, in order.
    ///
    /// Every output word equals the reference exactly, since the whole contract
    /// is exact integer algebra. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn run(&self, ctx: &GpuContext, queries: &[GpuQ16MulDivQuery]) -> Vec<i32> {
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
            label: Some("prism_volumetric_fixed_point_q16_muldiv_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<i32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv_bind_group"),
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
            label: Some("prism_volumetric_fixed_point_q16_muldiv_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fixed_point_q16_muldiv_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fixed_point_q16_muldiv_pass"),
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
        let result = bytemuck::cast_slice::<u8, i32>(&view).to_vec();
        drop(view);
        stage.unmap();
        result
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
