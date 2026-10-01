//! `wgpu` compute twin of the fixed-width `u32` bit packer
//! ([`bit_pack_u32`](prism_render_architecture::particle::bit_pack_u32)).
//!
//! The golden standard densely packs an array of small unsigned integers into a
//! stream of `32`-bit words — every value contributes exactly `bits` bits
//! (`1..=32`), laid down `LSB`-first back-to-back with no separators, so values
//! freely straddle word boundaries — and unpacks them back. This crate is the
//! on-device twin: both the forward [`GpuBitPackU32::pack`] and the reverse
//! [`GpuBitPackU32::unpack`] run one thread per element.
//!
//! Because the whole contract is integer shifts, masks and bitwise-or with no
//! multiply-rounding and no floating point, `CPU` and `GPU` compute the
//! identical bit pattern, so a passing real-device parity test is direct
//! evidence the ported kernels place and extract every bit exactly as the
//! reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! - [`GpuBitPackU32::pack`] mirrors
//!   [`pack`](prism_render_architecture::particle::bit_pack_u32::pack): one
//!   thread per input value masks it to its low `bits` bits and `atomicOr`s the
//!   low part into its target word, spilling the high bits into the next word
//!   with a second `atomicOr` when the value crosses a `32`-bit word boundary.
//!   Disjoint low/high contributions never collide, but distinct values can
//!   share a word, so the writes must be atomic; the output buffer is
//!   zero-initialised so every `atomicOr` accumulates from `0`.
//! - [`GpuBitPackU32::unpack`] mirrors
//!   [`unpack`](prism_render_architecture::particle::bit_pack_u32::unpack): one
//!   thread per output value locates its start word and in-word offset, reads
//!   the low part with a right shift, joins the high part from the next word
//!   with a left shift when the value crosses a boundary, and masks to `bits`.
//!
//! # Boundary and degenerate cases
//!
//! `bits == 32` degenerates to a verbatim word copy: the per-element bit offset
//! is then an exact multiple of `32`, so the in-word offset is always `0`, no
//! value ever spills, and both kernels reduce to a straight copy. The shift
//! guards exploit this: the spill shift `value >> (32 - offset)` and
//! `high << (32 - offset)` only run when `offset + bits > 32`, which with
//! `bits <= 32` forces `offset >= 1`, so the shift amount stays in `1..=31` and
//! the kernels never evaluate a `>= 32` shift (which is undefined in `WGSL`).
//! The mask helper likewise avoids `1u << 32` by returning `u32::MAX` directly
//! for `bits >= 32`. Empty inputs issue no dispatch and return an empty vector,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — the bit operators
//! `>> << & |`, the arithmetic `+ - *`, unsigned index comparison and the core
//! atomics `atomicOr`, `atomicStore`, `atomicLoad` — with no transcendental
//! call, no intrinsic, no optional device feature and no `u64`. The `u64` in
//! the reference appears only in the host-side test `LCG` fixture, never in the
//! kernel, so the twin stays pure `u32` and runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every operation is pure `u32` bit algebra with no rounding anywhere, so
//! `CPU` and `GPU` compute identical bit patterns. The parity test asserts an
//! exact `==` on every `u32` output with no tolerance: any mismatch is a
//! genuine port bug (a wrong offset, a flipped shift direction, a missed
//! cross-word spill).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bit_pack_u32`；无第三方引擎源码或衍生代码。
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

/// Number of `32`-bit words needed to hold `count` values of `bits` bits each.
///
/// Equal to `(count * bits).div_ceil(32)`, matching
/// [`packed_len_words`](prism_render_architecture::particle::bit_pack_u32::packed_len_words).
#[must_use]
pub fn packed_len_words(count: usize, bits: u32) -> usize {
    (count * (bits as usize)).div_ceil(32)
}

/// The `u32`-domain fixed-width bit-packer kernels, mirroring the `CPU` golden
/// [`bit_pack_u32`](prism_render_architecture::particle::bit_pack_u32) bit for
/// bit. One source file hosts the forward `pack` and reverse `unpack` entry
/// points sharing one bind-group layout and one `low_mask` helper.
const BIT_PACK_U32_WGSL: &str = r#"
// Fixed-width u32 bit-pack twin: one thread per element. Two entry points
// mirror the CPU golden `particle::bit_pack_u32`. Packing is LSB-first: value i
// occupies bits [i*bits, i*bits + bits) of the dense word stream, spilling into
// the next word when it crosses a 32-bit boundary. Pure u32 shifts / masks / or
// plus core atomics: no transcendental, no intrinsic, no u64, portable on
// Metal, Vulkan and DX12.
//
// Shift safety: WGSL leaves shifts by >= 32 undefined. The spill shifts only
// run when `offset + bits > 32`, which with `bits <= 32` forces `offset >= 1`,
// so `32 - offset` is in 1..=31; `low_mask` returns u32::MAX directly for
// bits >= 32 instead of evaluating `1u << 32`. bits == 32 therefore has
// offset == 0 everywhere, never spills, and degenerates to a verbatim copy.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::bit_pack_u32；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of threads (input values for pack, output values for unpack);
    // threads past this short-circuit.
    count: u32,
    // Fixed width of each value in bits, in 1..=32.
    bits: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;

// Mask with the low `bits` bits set (`bits` in 1..=32). For bits >= 32 this is
// u32::MAX, so the shift is only evaluated for bits < 32 and never shifts by 32.
fn low_mask(bits: u32) -> u32 {
    if (bits >= 32u) {
        return 0xffffffffu;
    }
    return (1u << bits) - 1u;
}

@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let bits = params.bits;
    let mask = low_mask(bits);
    let value = src[idx] & mask;
    let bit_pos = idx * bits;
    let word = bit_pos >> 5u;
    let offset = bit_pos & 31u;
    // The low part lands in `word`; offset < 32 keeps this shift valid. Distinct
    // values may share a word, so accumulate with atomicOr over a zeroed buffer.
    atomicOr(&dst[word], value << offset);
    if (offset + bits > 32u) {
        // Spill the high bits into the next word. offset >= 1 here (bits <= 32),
        // so 32 - offset stays in 1..=31 and never becomes a 32-bit shift.
        atomicOr(&dst[word + 1u], value >> (32u - offset));
    }
}

@compute @workgroup_size(64)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let bits = params.bits;
    let mask = low_mask(bits);
    let bit_pos = idx * bits;
    let word = bit_pos >> 5u;
    let offset = bit_pos & 31u;
    var value = src[word] >> offset;
    if (offset + bits > 32u) {
        // Join the high bits from the next word; as in pack, offset >= 1 here so
        // 32 - offset is a valid 1..=31 shift.
        value |= src[word + 1u] << (32u - offset);
    }
    atomicStore(&dst[idx], value & mask);
}
"#;

/// Uniform parameters for one dispatch: the thread `count` and value width
/// `bits`, padded to a `16`-byte, `std140`-aligned uniform struct matching
/// `Params` in [`BIT_PACK_U32_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of threads: input values for `pack`, output values for `unpack`.
    count: u32,
    /// Fixed value width in bits, in `1..=32`.
    bits: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// A compiled, reusable pair of `u32` fixed-width bit-pack kernels (forward
/// `pack` and reverse `unpack`).
pub struct GpuBitPackU32 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_pack: ComputePipeline,
    pipeline_unpack: ComputePipeline,
}

impl GpuBitPackU32 {
    /// Compiles the `pack` and `unpack` kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset (bit ops and the
    /// core atomics `atomicOr`, `atomicStore`), so no optional device feature is
    /// required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBitPackU32 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bit_pack_u32"),
            source: ShaderSource::Wgsl(BIT_PACK_U32_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_pipeline_layout"),
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
        let pipeline_pack = make("pack", "prism_volumetric_bit_pack_u32_pack_pipeline");
        let pipeline_unpack = make("unpack", "prism_volumetric_bit_pack_u32_unpack_pipeline");
        GpuBitPackU32 {
            module,
            layout,
            pipeline_pack,
            pipeline_unpack,
        }
    }

    /// Packs the low `bits` bits of each value into a dense `u32` word stream,
    /// mirroring
    /// [`pack`](prism_render_architecture::particle::bit_pack_u32::pack).
    ///
    /// `bits` must be in `1..=32`. When `bits < 32`, each value is masked to its
    /// low `bits` bits, so out-of-range inputs are truncated; `bits == 32`
    /// copies the values verbatim. Returns `packed_len_words(values.len(), bits)`
    /// words. An empty input returns an empty vector with no dispatch issued (a
    /// storage buffer cannot be zero-sized).
    ///
    /// # Panics
    ///
    /// Panics if `bits` is not in `1..=32`.
    #[must_use]
    pub fn pack(&self, ctx: &GpuContext, values: &[u32], bits: u32) -> Vec<u32> {
        assert!((1..=32).contains(&bits), "bits must be in 1..=32");
        let out_words = packed_len_words(values.len(), bits);
        self.dispatch(
            ctx,
            &self.pipeline_pack,
            values,
            values.len(),
            out_words,
            bits,
        )
    }

    /// Unpacks `count` values of `bits` bits each from a dense `u32` word stream,
    /// mirroring
    /// [`unpack`](prism_render_architecture::particle::bit_pack_u32::unpack).
    ///
    /// `bits` must be in `1..=32` and `packed` must hold at least
    /// `packed_len_words(count, bits)` words; `bits == 32` copies the words
    /// verbatim. Returns `count` values. An empty request (`count == 0`) returns
    /// an empty vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is not in `1..=32`, or if `packed` is shorter than
    /// `packed_len_words(count, bits)`.
    #[must_use]
    pub fn unpack(&self, ctx: &GpuContext, packed: &[u32], bits: u32, count: usize) -> Vec<u32> {
        assert!((1..=32).contains(&bits), "bits must be in 1..=32");
        assert!(
            packed.len() >= packed_len_words(count, bits),
            "packed stream too short for the requested count"
        );
        self.dispatch(ctx, &self.pipeline_unpack, packed, count, count, bits)
    }

    /// Issues one `1-D` dispatch of `pipeline` over `thread_count` threads,
    /// feeding `input` on binding `1` and reading back `out_len` `u32` words from
    /// the zero-initialised output on binding `2`. Empty work (`thread_count` or
    /// `out_len` zero) short-circuits without a dispatch because a storage buffer
    /// cannot be zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        input: &[u32],
        thread_count: usize,
        out_len: usize,
        bits: u32,
    ) -> Vec<u32> {
        if thread_count == 0 || out_len == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: thread_count as u32,
            bits,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let in_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_in"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE,
        });
        // Zero-initialized explicitly so every `atomicOr` in `pack` accumulates
        // from `0` regardless of the backend's buffer-clearing policy.
        let zeros = vec![0u32; out_len];
        let out_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_out"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let out_bytes = (out_len * size_of::<u32>()) as u64;
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bit_pack_u32_encoder"),
        });
        {
            let groups = (thread_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bit_pack_u32_pass"),
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
