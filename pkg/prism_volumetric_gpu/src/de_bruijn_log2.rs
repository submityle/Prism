//! `wgpu` compute twin of the pure-integer `de Bruijn` bit-scan and base-2
//! logarithm primitives
//! ([`de_bruijn_log2`](prism_render_architecture::particle::de_bruijn_log2)).
//!
//! The `CPU` golden answers "where is the top (or bottom) set bit?" and "what is
//! `floor`/`ceil` of `log2`?" with shifts, masks, a single integer multiply and
//! a small `de Bruijn` lookup table — no floating point and no transcendental
//! `log2`. This module ports only the `u32` core of that file:
//! [`floor_log2_u32`](prism_render_architecture::particle::de_bruijn_log2::floor_log2_u32),
//! [`ceil_log2_u32`](prism_render_architecture::particle::de_bruijn_log2::ceil_log2_u32),
//! [`is_power_of_two_u32`](prism_render_architecture::particle::de_bruijn_log2::is_power_of_two_u32),
//! [`next_power_of_two_u32`](prism_render_architecture::particle::de_bruijn_log2::next_power_of_two_u32)
//! and
//! [`trailing_zero_index_u32`](prism_render_architecture::particle::de_bruijn_log2::trailing_zero_index_u32).
//! It does **not** port the golden `u64` siblings (`floor_log2_u64`,
//! `ceil_log2_u64`, `is_power_of_two_u64`, `next_power_of_two_u64`): `WGSL` has
//! no `u64` type, so there is no portable 64-bit integer path to twin.
//!
//! [`GpuDeBruijnLog2`] is the on-device twin: one thread per element. Each
//! thread evaluates all five `u32` routines on its input word, so the whole
//! path is integer shifts, masks, one `wrapping` multiply and a table lookup
//! with no divide and no floating point anywhere. `CPU` and `GPU` therefore
//! compute identical bit patterns, and a passing real-device parity test is
//! direct evidence the ported kernel smears, multiplies and decodes the bits
//! exactly as the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! A single kernel reads one `u32` per element and writes a
//! [`DeBruijnLog2Result`] holding all five outputs:
//!
//! - `floor_log2` mirrors `floor_log2_u32`: smear the top bit downward into a
//!   `2^(k+1) - 1` mask, multiply by `0x07C4ACDD`, and decode the top five bits
//!   through the 32-entry `MSB` table. By convention `floor_log2(0) == 0`.
//! - `ceil_log2` mirrors `ceil_log2_u32`: `0` and `1` yield `0`, otherwise
//!   `floor_log2(x - 1) + 1`.
//! - `is_power_of_two` mirrors `is_power_of_two_u32`: `x != 0 && (x & (x - 1)) ==
//!   0`, decoded to a `bool` from the `1`/`0` word the kernel emits.
//! - `next_power_of_two` mirrors `next_power_of_two_u32`: decrement, smear every
//!   lower bit beneath the `MSB`, then increment. `next_power_of_two(0) == 1`.
//! - `trailing_zero_index` mirrors `trailing_zero_index_u32`: isolate the low
//!   bit with `x & x.wrapping_neg()`, multiply by `0x077CB531`, and decode the
//!   top five bits through the 32-entry `LSB` table. By convention
//!   `trailing_zero_index(0) == 32`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: the bit operators `>>
//! << & | ^`, unsigned `+ - *` (which wrap on overflow exactly like Rust's
//! `wrapping_mul` / `wrapping_sub` / `wrapping_neg`), unsigned comparison and a
//! `const` lookup table. There is no `log2`, `exp2`, `pow` or other
//! transcendental call, no `sqrt`, no divide, no optional device feature and no
//! `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every operation is pure `u32` bit algebra with no rounding anywhere on the
//! path, so `CPU` and `GPU` compute identical bit patterns. The parity test
//! asserts an exact `==` on every output of every element with no tolerance:
//! any mismatch is a genuine port bug (a wrong shift distance, a flipped
//! multiplier, a miscomputed mask or a transcribed table entry).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::de_bruijn_log2`；无第三方引擎源码或衍生代码。
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
/// that divides evenly on every target backend.
const WORKGROUP_SIZE: u32 = 64;

/// The `u32`-domain `de Bruijn` bit-scan kernel, mirroring the `CPU` golden
/// [`de_bruijn_log2`](prism_render_architecture::particle::de_bruijn_log2)
/// routine for routine. One entry point evaluates all five `u32` functions on
/// each input word, embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::de_bruijn_log2`。
const DE_BRUIJN_LOG2_WGSL: &str = r#"
// de Bruijn log2 twin: one thread per element reproduces the CPU golden
// `particle::de_bruijn_log2` u32 domain. Pure u32 bit algebra: shifts, masks,
// a single wrapping multiply and a de Bruijn lookup table. There is no log2 /
// exp2 / pow built-in, no sqrt, no divide and no u64, so the kernel runs
// unmodified on Metal, Vulkan and DX12. WGSL unsigned `* - +` wrap on overflow
// exactly like Rust `wrapping_mul` / `wrapping_sub` / `wrapping_neg`.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::de_bruijn_log2；无第三方
// 引擎源码或衍生代码。

// de Bruijn multiplier for the 32-bit MSB (base-2 logarithm) scan; matches the
// golden `DE_BRUIJN_MUL_LOG2_U32`.
const DE_BRUIJN_MUL_LOG2: u32 = 0x07C4ACDDu;
// de Bruijn multiplier for the 32-bit LSB (trailing-zero) scan; matches the
// golden `DE_BRUIJN_MUL_TRAILING_U32`.
const DE_BRUIJN_MUL_TRAILING: u32 = 0x077CB531u;

// Dispatch parameters. 16-byte uniform block: the valid element count plus
// three pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One result. 20-byte std430 stride of 5 scalar words, matching the host
// `GpuResult`: the two logarithms, the power-of-two flag (1u/0u), the rounded
// power of two and the trailing-zero index.
struct LogResult {
    floor_log2: u32,
    ceil_log2: u32,
    is_pow2: u32,
    next_pow2: u32,
    trailing_zero_index: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<LogResult>;

// 32-entry decode table for the MSB scan, matching the golden
// `DE_BRUIJN_LOG2_U32`. Held in a function `var` so the runtime index lands in
// an addressable array on every backend.
fn log2_lookup(index: u32) -> u32 {
    var table = array<u32, 32>(
        0u, 9u, 1u, 10u, 13u, 21u, 2u, 29u, 11u, 14u, 16u, 18u, 22u, 25u, 3u, 30u,
        8u, 12u, 20u, 28u, 15u, 17u, 24u, 7u, 19u, 27u, 23u, 6u, 26u, 5u, 4u, 31u
    );
    return table[index];
}

// 32-entry decode table for the LSB scan, matching the golden
// `DE_BRUIJN_TRAILING_U32`.
fn trailing_lookup(index: u32) -> u32 {
    var table = array<u32, 32>(
        0u, 1u, 28u, 2u, 29u, 14u, 24u, 3u, 30u, 22u, 20u, 15u, 25u, 17u, 4u, 8u,
        31u, 27u, 13u, 23u, 21u, 19u, 16u, 7u, 26u, 12u, 18u, 6u, 11u, 5u, 10u, 9u
    );
    return table[index];
}

// floor(log2(x)): the most-significant set bit. By convention x == 0 returns 0
// as a saturating stand-in. Smear the top bit into a 2^(k+1) - 1 mask, then one
// multiply lands the MSB fingerprint in the top five bits.
fn floor_log2(x: u32) -> u32 {
    if (x == 0u) {
        return 0u;
    }
    var v = x;
    v = v | (v >> 1u);
    v = v | (v >> 2u);
    v = v | (v >> 4u);
    v = v | (v >> 8u);
    v = v | (v >> 16u);
    return log2_lookup((v * DE_BRUIJN_MUL_LOG2) >> 27u);
}

// ceil(log2(x)): the smallest n with 2^n >= x. Both 0 and 1 return 0; for
// x >= 2 the result is floor_log2(x - 1) + 1.
fn ceil_log2(x: u32) -> u32 {
    if (x <= 1u) {
        return 0u;
    }
    return floor_log2(x - 1u) + 1u;
}

// 1u when x has exactly one set bit, else 0u. Zero is not a power of two:
// `x != 0 && (x & (x - 1)) == 0` clears the lowest set bit and checks nothing
// remains. The && short-circuits, so the x - 1 wrap at x == 0 is never read.
fn is_pow2(x: u32) -> u32 {
    if (x != 0u && (x & (x - 1u)) == 0u) {
        return 1u;
    }
    return 0u;
}

// Smallest power of two >= x. next_pow2(0) == 1. Decrement, smear every lower
// bit beneath the MSB, then increment. Defined while the result fits in u32.
fn next_pow2(x: u32) -> u32 {
    if (x <= 1u) {
        return 1u;
    }
    var v = x - 1u;
    v = v | (v >> 1u);
    v = v | (v >> 2u);
    v = v | (v >> 4u);
    v = v | (v >> 8u);
    v = v | (v >> 16u);
    return v + 1u;
}

// Index of the least-significant set bit (count of trailing zeros). By
// convention trailing_zero_index(0) == 32. Isolate the low bit with
// `x & (0u - x)` — the two's-complement negation matching `x.wrapping_neg()` —
// then one multiply lands the LSB fingerprint in the top five bits.
fn trailing_zero_index(x: u32) -> u32 {
    if (x == 0u) {
        return 32u;
    }
    let isolated = x & (0u - x);
    return trailing_lookup((isolated * DE_BRUIJN_MUL_TRAILING) >> 27u);
}

@compute @workgroup_size(64)
fn de_bruijn_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let x = inputs[idx];
    var out: LogResult;
    out.floor_log2 = floor_log2(x);
    out.ceil_log2 = ceil_log2(x);
    out.is_pow2 = is_pow2(x);
    out.next_pow2 = next_pow2(x);
    out.trailing_zero_index = trailing_zero_index(x);
    results[idx] = out;
}
"#;

/// One decoded `de Bruijn` result: the five `u32`-domain outputs for a single
/// input word.
///
/// The fields mirror the golden functions one for one;
/// [`is_power_of_two`](DeBruijnLog2Result::is_power_of_two) is decoded to a
/// `bool` from the `1`/`0` flag word the kernel emits, and the integer fields
/// are reported verbatim.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::de_bruijn_log2`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeBruijnLog2Result {
    /// `floor(log2(x))`, matching
    /// [`floor_log2_u32`](prism_render_architecture::particle::de_bruijn_log2::floor_log2_u32);
    /// `0` for `x == 0`.
    pub floor_log2: u32,
    /// `ceil(log2(x))`, matching
    /// [`ceil_log2_u32`](prism_render_architecture::particle::de_bruijn_log2::ceil_log2_u32);
    /// `0` for `x <= 1`.
    pub ceil_log2: u32,
    /// Whether `x` is a power of two, matching
    /// [`is_power_of_two_u32`](prism_render_architecture::particle::de_bruijn_log2::is_power_of_two_u32);
    /// `false` for `x == 0`.
    pub is_power_of_two: bool,
    /// Smallest power of two `>= x`, matching
    /// [`next_power_of_two_u32`](prism_render_architecture::particle::de_bruijn_log2::next_power_of_two_u32);
    /// `1` for `x <= 1`.
    pub next_power_of_two: u32,
    /// Index of the least-significant set bit, matching
    /// [`trailing_zero_index_u32`](prism_render_architecture::particle::de_bruijn_log2::trailing_zero_index_u32);
    /// `32` for `x == 0`.
    pub trailing_zero_index: u32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`DE_BRUIJN_LOG2_WGSL`]: the valid element count plus three pad
/// words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid elements.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One result as read back. `20`-byte `std430` stride of `5` scalar words,
/// matching `LogResult` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `floor(log2(x))`.
    floor_log2: u32,
    /// `ceil(log2(x))`.
    ceil_log2: u32,
    /// Power-of-two flag, `1` or `0`.
    is_pow2: u32,
    /// Smallest power of two `>= x`.
    next_pow2: u32,
    /// Index of the least-significant set bit.
    trailing_zero_index: u32,
}

/// A compiled, reusable `u32`-domain `de Bruijn` bit-scan kernel evaluating
/// `floor_log2`, `ceil_log2`, `is_power_of_two`, `next_power_of_two` and
/// `trailing_zero_index` per element.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::de_bruijn_log2`。
pub struct GpuDeBruijnLog2 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDeBruijnLog2 {
    /// Compiles the `de Bruijn` bit-scan kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::de_bruijn_log2`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDeBruijnLog2 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2"),
            source: ShaderSource::Wgsl(DE_BRUIJN_LOG2_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("de_bruijn_eval"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDeBruijnLog2 {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates all five `u32` `de Bruijn` routines on every element of `xs`,
    /// returning one [`DeBruijnLog2Result`] per input in order.
    ///
    /// For element `i`, every field equals the matching golden function applied
    /// to `xs[i]` exactly — the whole path is integer bit algebra, so the twin
    /// is bit-identical, not merely close. An empty `xs` slice yields an empty
    /// result with no dispatch issued, because a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::de_bruijn_log2`。
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, xs: &[u32]) -> Vec<DeBruijnLog2Result> {
        if xs.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: xs.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_inputs"),
            contents: bytemuck::cast_slice(xs),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (xs.len() as u64) * (size_of::<GpuResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_de_bruijn_log2_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_de_bruijn_log2_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
            let groups = (xs.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), xs.len());

        gpu_results
            .into_iter()
            .map(|r| DeBruijnLog2Result {
                floor_log2: r.floor_log2,
                ceil_log2: r.ceil_log2,
                // The kernel emits exactly 1 for a power of two and 0 otherwise.
                is_power_of_two: r.is_pow2 == 1,
                next_power_of_two: r.next_pow2,
                trailing_zero_index: r.trailing_zero_index,
            })
            .collect()
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
