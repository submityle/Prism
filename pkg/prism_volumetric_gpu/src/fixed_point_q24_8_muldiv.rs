//! `wgpu` compute twin of the signed `Q24.8` fixed-point multiply
//! ([`fixed_point_q24_8::mul`](prism_render_architecture::particle::fixed_point_q24_8::mul)).
//!
//! The `CPU` golden
//! [`fixed_point_q24_8`](prism_render_architecture::particle::fixed_point_q24_8)
//! is a deterministic, pure-integer fixed-point type whose `raw` [`i32`]
//! represents `raw / 256`. Its multiply
//! ([`mul`](prism_render_architecture::particle::fixed_point_q24_8::mul)) widens
//! both operands to [`i64`], forms the full `64`-bit product and arithmetic-right-
//! shifts it by `8` before a wrapping narrowing back to [`i32`]:
//! `raw = (((a.raw as i64) * (b.raw as i64)) >> 8) as i32`. There is no round
//! bias and no saturation — the shift floors toward negative infinity and the
//! `as i32` is a plain two's-complement low-`32`-bit truncation that wraps.
//!
//! [`GpuFixedPointQ24_8Muldiv`] is the on-device twin: one thread multiplies one
//! `(a, b)` pair and writes one `raw` result. A passing real-device parity test
//! is direct evidence the ported kernel reproduces the exact `64`-bit product,
//! the exact arithmetic shift and the exact wrapping narrowing the reference
//! does over the whole [`i32`] range, not merely that the shader compiles.
//!
//! # Why a `64`-bit emulation
//!
//! `WGSL` has no `i64` (only `i32`, `u32`, `f32` and `bool`), so the full
//! `64`-bit signed product cannot be formed natively. The kernel emulates it
//! with a `(hi: u32, lo: u32)` two's-complement pair:
//!
//! 1. Reduce each operand to its unsigned magnitude (`u32_mag`), tracking the
//!    product sign as the `XOR` of the two operand signs. `i32::MIN` maps to the
//!    magnitude `0x8000_0000` exactly, since `(!0x8000_0000) + 1 == 0x8000_0000`.
//! 2. Multiply the two `32`-bit magnitudes with schoolbook `16`-bit limbs
//!    (`a0`, `a1`, `b0`, `b1`), summing the partial products `ll`, `lh`, `hl`,
//!    `hh` and the carry chain into a full `64`-bit unsigned magnitude
//!    `(hi, lo)`. Each partial product and the `cross` sum stay within `u32`.
//! 3. If the sign is negative, two's-complement-negate the `64`-bit value:
//!    `lo = ~lo + 1`, `hi = ~hi + carry`, where the carry is `1` exactly when the
//!    low word wrapped to `0`.
//! 4. Arithmetic-right-shift by `8` and keep the low `32` bits. Only the low
//!    `32` bits of `value >> 8` are needed (the `as i32` narrowing discards the
//!    rest), and those bits are `(lo >> 8) | (hi << 24)` regardless of whether
//!    the shift is arithmetic or logical, so no sign extension of the discarded
//!    high bits is required. [`bitcast`] re-reads that `u32` as the signed
//!    `raw`, matching the reference's wrapping `as i32`.
//!
//! # Scope of this wave
//!
//! This module twins **only** the multiply, the one golden operation that needs
//! an [`i64`] intermediate. The sibling golden operations
//! [`div`](prism_render_architecture::particle::fixed_point_q24_8::div) and
//! [`from_ratio`](prism_render_architecture::particle::fixed_point_q24_8::from_ratio)
//! need `64`-bit **signed division**, which is a separate and considerably
//! larger emulation (a restoring or long-division loop over the emulated
//! `64`-bit pair); they are honestly left for a later wave and are not emulated
//! here. The cheaper golden operations (`add`, `sub`, `neg`, `floor`, `fract`,
//! `abs`, `to_int_trunc`, `from_int`) are plain `i32` algebra needing no `64`-bit
//! intermediate and are out of scope for this multiply-focused twin.
//!
//! # Correctness model
//!
//! Every step is exact integer bit algebra with no rounding and no floating
//! point, so `CPU` and `GPU` compute the identical `raw` bit pattern. The parity
//! test asserts an exact `==` on every [`i32`] output with no tolerance: any
//! mismatch is a genuine port bug (a wrong limb, a dropped carry, a mis-sized
//! shift, a missing sign negation).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - *`, the bit
//! operators `>> << & | ~`, unsigned index comparison, `select` and `bitcast` —
//! with no transcendental call, no intrinsic, no optional device feature and no
//! `u64`. It therefore runs unmodified on `Metal`, `Vulkan` and `DX12`. There is
//! no loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fixed_point_q24_8`
//! 的 `mul`（`u32` 对仿真 `64`-bit 有符号乘法）；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `Q24.8` multiply kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`fixed_point_q24_8::mul`](prism_render_architecture::particle::fixed_point_q24_8::mul)
/// by emulating the `64`-bit signed product with a `(hi, lo)` `u32` pair.
const FIXED_POINT_Q24_8_MULDIV_WGSL: &str = r#"
// Q24.8 fixed-point multiply twin: one thread per (a, b) pair. Mirrors the CPU
// golden `particle::fixed_point_q24_8::mul`, which computes
// `((a as i64) * (b as i64)) >> 8` and wraps-narrows to i32. WGSL has no i64, so
// the 64-bit signed product is emulated with a (hi, lo) u32 two's-complement
// pair, arithmetic-right-shifted by 8 and truncated to the low 32 bits. No round
// bias, no saturation. Pure u32 bit algebra: no transcendental, no intrinsic, no
// u64, portable on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::fixed_point_q24_8 的
// mul；无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid (a, b) pairs; threads past this short-circuit.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One multiply query: the two raw Q24.8 operands. 8-byte std430 stride matching
// the host `GpuQuery`.
struct Query {
    a: i32,
    b: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<i32>;

// Unsigned magnitude of a signed i32: |x| as u32. i32::MIN maps to 0x80000000
// exactly because (~0x80000000) + 1 == 0x80000000.
fn u32_mag(x: i32) -> u32 {
    let u = bitcast<u32>(x);
    if (x >= 0) {
        return u;
    }
    return (~u) + 1u;
}

// Emulated 64-bit signed Q24.8 multiply: forms the full 64-bit product of the
// two magnitudes with 16-bit schoolbook limbs, applies the product sign by
// two's-complement negation of the (hi, lo) pair, arithmetic-right-shifts by 8
// and keeps the low 32 bits (the reference's wrapping `as i32`).
fn q24_mul(a: i32, b: i32) -> i32 {
    let sign_neg = (a < 0) != (b < 0);
    let ua = u32_mag(a);
    let ub = u32_mag(b);
    let a0 = ua & 0xffffu;
    let a1 = ua >> 16u;
    let b0 = ub & 0xffffu;
    let b1 = ub >> 16u;
    let ll = a0 * b0;
    let lh = a0 * b1;
    let hl = a1 * b0;
    let hh = a1 * b1;
    let cross = (ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu);
    var lo = (ll & 0xffffu) | (cross << 16u);
    var hi = hh + (lh >> 16u) + (hl >> 16u) + (cross >> 16u);
    if (sign_neg) {
        // Two's-complement negate the 64-bit magnitude: ~V + 1. The high word
        // takes a carry exactly when the negated low word wrapped to 0.
        lo = ~lo + 1u;
        hi = ~hi + select(0u, 1u, lo == 0u);
    }
    // Low 32 bits of the arithmetic right shift by 8: bits [8, 40) of the 64-bit
    // value. `lo >> 8` supplies bits [8, 32); `hi << 24` supplies bits [32, 40).
    // The discarded high bits make the arithmetic/logical distinction moot here,
    // exactly matching the reference's wrapping narrowing `as i32`.
    let s_lo = (lo >> 8u) | (hi << 24u);
    return bitcast<i32>(s_lo);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx] = q24_mul(queries[idx].a, queries[idx].b);
}
"#;

/// One `Q24.8` multiply query: the two raw operands whose product in `Q24.8` is
/// computed.
///
/// Each field is the two's-complement `raw` of a
/// [`Q24_8`](prism_render_architecture::particle::fixed_point_q24_8::Q24_8),
/// representing the real value `raw / 256`. Derives [`Eq`] because it holds only
/// integer `raw` values, so queries compare exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPointQ24_8MulQuery {
    /// The raw of the left operand.
    pub a: i32,
    /// The raw of the right operand.
    pub b: i32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`FIXED_POINT_Q24_8_MULDIV_WGSL`]: the query count and three pad
/// words — `16` bytes, each field at the uniform offset the shader expects.
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

/// One query as uploaded. `8`-byte `std430` stride matching `Query` in the
/// shader: the two raw operands as plain [`i32`] with no padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The raw of the left operand.
    a: i32,
    /// The raw of the right operand.
    b: i32,
}

impl GpuQuery {
    /// Packs a [`FixedPointQ24_8MulQuery`] into the `std430` upload layout.
    fn from_query(query: &FixedPointQ24_8MulQuery) -> GpuQuery {
        GpuQuery {
            a: query.a,
            b: query.b,
        }
    }
}

/// A compiled, reusable `Q24.8` fixed-point multiply pipeline.
///
/// The name carries the `muldiv` module suffix for symmetry with the golden
/// family, but this wave twins only the multiply; `div` and `from_ratio` (which
/// need `64`-bit signed division) are left for a later wave as documented on
/// this module.
pub struct GpuFixedPointQ24_8Muldiv {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFixedPointQ24_8Muldiv {
    /// Compiles the `Q24.8` multiply kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFixedPointQ24_8Muldiv {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv"),
            source: ShaderSource::Wgsl(FIXED_POINT_Q24_8_MULDIV_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFixedPointQ24_8Muldiv {
            module,
            layout,
            pipeline,
        }
    }

    /// Multiplies every query in `queries` and returns one `raw` [`i32`] per
    /// input, in order.
    ///
    /// The returned `raw` for query `q` equals
    /// [`mul`](prism_render_architecture::particle::fixed_point_q24_8::mul)
    /// evaluated on `q.a` and `q.b` exactly, over the whole [`i32`] range. An
    /// empty `queries` slice returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn run(&self, ctx: &GpuContext, queries: &[FixedPointQ24_8MulQuery]) -> Vec<i32> {
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
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_bind_group"),
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
            label: Some("prism_volumetric_fixed_point_q24_8_muldiv_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fixed_point_q24_8_muldiv_pass"),
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
