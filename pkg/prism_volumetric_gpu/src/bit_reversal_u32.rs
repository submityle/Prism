//! `wgpu` compute twin of the pure-integer `u32` bit-reversal primitives
//! ([`bit_reversal_u32`](prism_render_architecture::particle::bit_reversal_u32)).
//!
//! Reversing the bit order of a word is the arithmetic heart of the
//! bit-reversal permutation an in-place radix-2 `FFT` applies around its
//! butterfly passes. This module ports only that permutation algebra, and only
//! the `u32` core of the golden file: it reverses the bits of a word, reverses
//! just the low `bits` lanes of an index, advances a bit-reversed counter, and
//! tests whether the low `bits` lanes form a bit-reversal palindrome. It does
//! **not** port the golden `u8`/`u16` widths, the `reverse_bits_u64` sibling
//! (`WGSL` has no `u64`), or the host-only permutation-table builder.
//!
//! [`GpuBitReversalU32`] is the on-device twin: one thread per element. Each
//! thread reverses or classifies one `u32`, so the whole path is integer shifts
//! and masks with no multiply, divide or floating point. `CPU` and `GPU`
//! therefore compute identical bit patterns, and a passing real-device parity
//! test is direct evidence the ported kernels reverse, shift and carry the bits
//! exactly as the reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Four per-element kernels mirror the four `u32` reference functions:
//!
//! - [`GpuBitReversalU32::reverse_bits`] reverses all `32` lanes of each input
//!   word, mirroring
//!   [`reverse_bits_u32`](prism_render_architecture::particle::bit_reversal_u32::reverse_bits_u32).
//!   The `WGSL` built-in `reverseBits(u32)` performs the identical lane
//!   permutation as the golden mask-and-swap network.
//! - [`GpuBitReversalU32::reverse_lowest_bits`] reverses only the low `bits`
//!   lanes of each paired `(x, bits)`, mirroring
//!   [`reverse_lowest_bits`](prism_render_architecture::particle::bit_reversal_u32::reverse_lowest_bits).
//! - [`GpuBitReversalU32::bit_reverse_increment`] advances each paired
//!   `(index, bits)` by one step of a bit-reversed counter, mirroring
//!   [`bit_reverse_increment`](prism_render_architecture::particle::bit_reversal_u32::bit_reverse_increment).
//! - [`GpuBitReversalU32::is_bit_reversal_palindrome`] reports whether the low
//!   `bits` lanes of each paired `(x, bits)` are symmetric under reversal,
//!   mirroring
//!   [`is_bit_reversal_palindrome`](prism_render_architecture::particle::bit_reversal_u32::is_bit_reversal_palindrome).
//!
//! The free host functions [`host_reverse_bits_u32`], [`host_reverse_lowest_bits`],
//! [`host_bit_reverse_increment`] and [`host_is_bit_reversal_palindrome`] are a
//! `CPU` mirror of the kernels, built from the same `WGSL`-portable algebra, so
//! callers and tests can cross-check the device output against an in-crate
//! reference as well as the golden module.
//!
//! # Domain restrictions
//!
//! The reference pins `bits` to `0..=32` for the reversal and palindrome
//! helpers and to `1..=32` for the increment. The kernels assume the same
//! ranges; feeding out-of-range `bits` is a caller error, as with the golden,
//! and is never exercised by the parity fixtures. `bits == 0` yields `0` for
//! the low-lane reversal and a vacuous `true` for the palindrome, matching the
//! reference branch for branch.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset: the bit operators
//! `>> << & | ^`, the arithmetic `- `, unsigned comparison, and the
//! `reverseBits` built-in. There is no transcendental call, no optional device
//! feature and no `u64`. Only the increment carries a loop, and that loop is
//! bounded by `bits <= 32`, so the kernels run unmodified on `Metal`, `Vulkan`
//! and `DX12`.
//!
//! # Correctness model
//!
//! Every operation is pure `u32` bit algebra with no rounding anywhere on the
//! path, so `CPU` and `GPU` compute identical bit patterns. The parity test
//! asserts an exact `==` on every element with no tolerance: any mismatch is a
//! genuine port bug (a wrong shift distance, a flipped carry direction, a
//! miscomputed mask).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32`；无第三方引擎源码或衍生代码。
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

/// `CPU` mirror of [`GpuBitReversalU32::reverse_bits`]: reverses all `32` bit
/// lanes of `x` using the standard library's `reverse_bits`, which is the
/// identical lane permutation as the golden mask-and-swap network and the
/// `WGSL` `reverseBits` built-in.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::reverse_bits_u32`。
#[must_use]
pub const fn host_reverse_bits_u32(x: u32) -> u32 {
    x.reverse_bits()
}

/// `CPU` mirror of [`GpuBitReversalU32::reverse_lowest_bits`]: reverses only the
/// low `bits` lanes of `x`, discarding the high lanes. `bits == 0` yields `0`
/// and `bits` must lie in `0..=32`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::reverse_lowest_bits`。
#[must_use]
pub const fn host_reverse_lowest_bits(x: u32, bits: u32) -> u32 {
    debug_assert!(bits <= 32);
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (32 - bits)
}

/// `CPU` mirror of [`GpuBitReversalU32::bit_reverse_increment`]: advances
/// `index` by one step of a bit-reversed counter of width `bits`, carrying from
/// the most-significant lane downward. `bits` must lie in `1..=32`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::bit_reverse_increment`。
#[must_use]
pub fn host_bit_reverse_increment(index: u32, bits: u32) -> u32 {
    debug_assert!((1..=32).contains(&bits));
    let mut idx = index;
    let mut mask = 1u32 << (bits - 1);
    loop {
        idx ^= mask;
        if (idx & mask) != 0 {
            break;
        }
        if mask == 1 {
            break;
        }
        mask >>= 1;
    }
    idx
}

/// `CPU` mirror of [`GpuBitReversalU32::is_bit_reversal_palindrome`]: reports
/// whether the low `bits` lanes of `x` read identically forward and backward.
/// `bits == 0` is vacuously `true`; `bits` must lie in `0..=32`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::is_bit_reversal_palindrome`。
#[must_use]
pub const fn host_is_bit_reversal_palindrome(x: u32, bits: u32) -> bool {
    debug_assert!(bits <= 32);
    if bits == 0 {
        return true;
    }
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let low = x & mask;
    host_reverse_lowest_bits(low, bits) == low
}

/// The `u32`-domain bit-reversal kernels, mirroring the `CPU` golden
/// [`bit_reversal_u32`](prism_render_architecture::particle::bit_reversal_u32)
/// lane for lane. One source file hosts four entry points sharing one
/// bind-group layout and one `reverse_lowest_bits` helper.
const BIT_REVERSAL_U32_WGSL: &str = r#"
// Bit-reversal u32 twin: one thread per element. Four entry points mirror the
// CPU golden `particle::bit_reversal_u32` u32 domain. Pure u32 bit algebra:
// shifts, masks, xor and the reverseBits built-in. No transcendental, no
// intrinsic beyond reverseBits, no u64, portable on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::bit_reversal_u32；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of valid elements; threads past this short-circuit.
    count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// First input: the word `x` or counter `index`.
@group(0) @binding(1) var<storage, read> in_a: array<u32>;
// Second input: the per-element `bits` width (unused by `reverse_bits`).
@group(0) @binding(2) var<storage, read> in_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

// Reverse only the low `bits` lanes of `x`, matching the reference helper:
// reverse all 32 lanes, then drop the high `32 - bits` lanes with a shift.
// `bits == 0` yields 0; `bits` is assumed in 0..=32.
fn reverse_lowest_bits(x: u32, bits: u32) -> u32 {
    if (bits == 0u) {
        return 0u;
    }
    return reverseBits(x) >> (32u - bits);
}

@compute @workgroup_size(64)
fn reverse_bits(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Reverse all 32 bit lanes of the word.
    dst[idx] = reverseBits(in_a[idx]);
}

@compute @workgroup_size(64)
fn reverse_lowest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Reverse only the low `bits` lanes of the word.
    dst[idx] = reverse_lowest_bits(in_a[idx], in_b[idx]);
}

@compute @workgroup_size(64)
fn bit_reverse_increment(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Advance a bit-reversed counter: carry from the MSB lane downward,
    // flipping lanes until one flips from 0 to 1. `bits` is assumed in 1..=32.
    let bits = in_b[idx];
    var acc = in_a[idx];
    var mask = 1u << (bits - 1u);
    loop {
        acc = acc ^ mask;
        if ((acc & mask) != 0u) {
            break;
        }
        if (mask == 1u) {
            break;
        }
        mask = mask >> 1u;
    }
    dst[idx] = acc;
}

@compute @workgroup_size(64)
fn is_palindrome(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let bits = in_b[idx];
    if (bits == 0u) {
        // Zero width is vacuously a palindrome.
        dst[idx] = 1u;
        return;
    }
    var mask = 0u;
    if (bits == 32u) {
        mask = 0xffffffffu;
    } else {
        mask = (1u << bits) - 1u;
    }
    let low = in_a[idx] & mask;
    if (reverse_lowest_bits(low, bits) == low) {
        dst[idx] = 1u;
    } else {
        dst[idx] = 0u;
    }
}
"#;

/// Uniform parameters for one dispatch: the element `count` plus padding to a
/// `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BIT_REVERSAL_U32_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable set of `u32` bit-reversal kernels (reverse-bits,
/// reverse-lowest-bits, bit-reversed increment and palindrome test).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32`。
pub struct GpuBitReversalU32 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_reverse_bits: ComputePipeline,
    pipeline_reverse_lowest: ComputePipeline,
    pipeline_increment: ComputePipeline,
    pipeline_palindrome: ComputePipeline,
}

impl GpuBitReversalU32 {
    /// Compiles the four `u32` bit-reversal kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBitReversalU32 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32"),
            source: ShaderSource::Wgsl(BIT_REVERSAL_U32_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_reverse_bits = make(
            "reverse_bits",
            "prism_volumetric_bit_reversal_u32_reverse_bits_pipeline",
        );
        let pipeline_reverse_lowest = make(
            "reverse_lowest",
            "prism_volumetric_bit_reversal_u32_reverse_lowest_pipeline",
        );
        let pipeline_increment = make(
            "bit_reverse_increment",
            "prism_volumetric_bit_reversal_u32_increment_pipeline",
        );
        let pipeline_palindrome = make(
            "is_palindrome",
            "prism_volumetric_bit_reversal_u32_palindrome_pipeline",
        );
        GpuBitReversalU32 {
            module,
            layout,
            pipeline_reverse_bits,
            pipeline_reverse_lowest,
            pipeline_increment,
            pipeline_palindrome,
        }
    }

    /// Reverses all `32` bit lanes of each input word, mirroring
    /// [`reverse_bits_u32`](prism_render_architecture::particle::bit_reversal_u32::reverse_bits_u32).
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued (a storage buffer cannot be zero-sized).
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::reverse_bits_u32`。
    #[must_use]
    pub fn reverse_bits(&self, ctx: &GpuContext, xs: &[u32]) -> Vec<u32> {
        // `reverse_bits` ignores the second input slot; feed `xs` to keep the
        // shared layout satisfied without a separate zero-sized buffer.
        self.dispatch(ctx, &self.pipeline_reverse_bits, xs, xs)
    }

    /// Reverses only the low `bits` lanes of each paired `(x, bits)`, mirroring
    /// [`reverse_lowest_bits`](prism_render_architecture::particle::bit_reversal_u32::reverse_lowest_bits).
    /// Each `bits` entry must lie in `0..=32`.
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `xs` and `bits` differ in length.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::reverse_lowest_bits`。
    #[must_use]
    pub fn reverse_lowest_bits(&self, ctx: &GpuContext, xs: &[u32], bits: &[u32]) -> Vec<u32> {
        assert_eq!(xs.len(), bits.len(), "xs and bits must be the same length");
        self.dispatch(ctx, &self.pipeline_reverse_lowest, xs, bits)
    }

    /// Advances each paired `(index, bits)` by one step of a bit-reversed
    /// counter, mirroring
    /// [`bit_reverse_increment`](prism_render_architecture::particle::bit_reversal_u32::bit_reverse_increment).
    /// Each `bits` entry must lie in `1..=32`.
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `indices` and `bits` differ in length.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::bit_reverse_increment`。
    #[must_use]
    pub fn bit_reverse_increment(
        &self,
        ctx: &GpuContext,
        indices: &[u32],
        bits: &[u32],
    ) -> Vec<u32> {
        assert_eq!(
            indices.len(),
            bits.len(),
            "indices and bits must be the same length"
        );
        self.dispatch(ctx, &self.pipeline_increment, indices, bits)
    }

    /// Reports whether the low `bits` lanes of each paired `(x, bits)` are
    /// symmetric under bit reversal, mirroring
    /// [`is_bit_reversal_palindrome`](prism_render_architecture::particle::bit_reversal_u32::is_bit_reversal_palindrome).
    /// Each `bits` entry must lie in `0..=32`; `bits == 0` is vacuously `true`.
    ///
    /// Returns one `bool` per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `xs` and `bits` differ in length.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_reversal_u32::is_bit_reversal_palindrome`。
    #[must_use]
    pub fn is_bit_reversal_palindrome(
        &self,
        ctx: &GpuContext,
        xs: &[u32],
        bits: &[u32],
    ) -> Vec<bool> {
        assert_eq!(xs.len(), bits.len(), "xs and bits must be the same length");
        // The kernel emits 1 for a palindrome and 0 otherwise; decode to bool.
        self.dispatch(ctx, &self.pipeline_palindrome, xs, bits)
            .into_iter()
            .map(|flag| flag == 1)
            .collect()
    }

    /// Issues one `1-D` dispatch of `pipeline` over the paired `in_a` and `in_b`
    /// inputs, reading the `u32` outputs back. Empty inputs short-circuit
    /// without a dispatch because a storage buffer cannot be zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        in_a: &[u32],
        in_b: &[u32],
    ) -> Vec<u32> {
        let count = in_a.len();
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
            label: Some("prism_volumetric_bit_reversal_u32_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let in_a_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_in_a"),
            contents: bytemuck::cast_slice(in_a),
            usage: BufferUsages::STORAGE,
        });
        let in_b_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_in_b"),
            contents: bytemuck::cast_slice(in_b),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = size_of_val(in_a) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: in_a_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: in_b_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bit_reversal_u32_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bit_reversal_u32_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
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
        let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
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
