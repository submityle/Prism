//! `wgpu` compute twin of the normalized fixed-point quantizer
//! ([`unorm_snorm_pack`](prism_render_architecture::particle::unorm_snorm_pack)).
//!
//! The golden `CPU` reference converts author-facing `f32` values into the
//! normalized integer vertex/texture formats a `GPU` consumes (`UNORM`/`SNORM`)
//! and back, using the exact round-to-nearest and endpoint rules of the `D3D`,
//! `Vulkan` and `OpenGL` specifications. [`GpuUnormSnormPack`] is the on-device
//! twin: one thread per element, mirroring the reference quantizer bit for bit.
//!
//! # What is twinned
//!
//! Every public reference function is reproduced through six per-element
//! kernels sharing one bind-group layout. Two scalar kernels are parametrized
//! by a `mode` selector; four kernels handle the `RGBA` vector helpers:
//!
//! - [`GpuUnormSnormPack::pack_unorm8`], [`GpuUnormSnormPack::pack_unorm16`],
//!   [`GpuUnormSnormPack::pack_snorm8`] and [`GpuUnormSnormPack::pack_snorm16`]
//!   quantize an `f32` to the integer code of the matching format, mirroring the
//!   reference packers.
//! - [`GpuUnormSnormPack::unpack_unorm8`], [`GpuUnormSnormPack::unpack_unorm16`],
//!   [`GpuUnormSnormPack::unpack_snorm8`] and
//!   [`GpuUnormSnormPack::unpack_snorm16`] reconstruct the real value from an
//!   integer code.
//! - [`GpuUnormSnormPack::pack_unorm8x4`] / [`GpuUnormSnormPack::unpack_unorm8x4`]
//!   and [`GpuUnormSnormPack::pack_unorm16x2`] /
//!   [`GpuUnormSnormPack::unpack_unorm16x2`] twin the `LSB`-first `RGBA`
//!   channel packers.
//!
//! # Numeric discipline
//!
//! The reference rounds `UNORM` with round-half-up (`floor(x * MAX + 0.5)`) and
//! `SNORM` half away from zero (`floor(|y| + 0.5)` with the sign reapplied),
//! using only `floor`, `clamp`, `min`, `max` and `abs`. The kernels use the
//! identical expressions. No `round` intrinsic, no transcendental and no `u64`
//! appears, so the shaders stay in the portable core-`WGSL` subset and run
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Transport
//!
//! All data moves as `u32` words. Pack inputs carry the `f32` bit pattern
//! (`bitcast` back to `f32` in the kernel); `SNORM` pack outputs carry the `i32`
//! code bit pattern. Unpack outputs carry the reconstructed `f32` bit pattern.
//! The `CPU`-side wrappers convert the typed arguments to and from these words.
//!
//! # Correctness model
//!
//! Pack produces an integer code: the parity test asserts an exact `==` on
//! every code (integers compare exactly). Unpack produces an `f32`: parity is
//! asserted with an epsilon bound because the reference contract forbids `f32`
//! equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::unorm_snorm_pack`；无第三方引擎源码或衍生代码。
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

/// Scalar-kernel `mode` selecting the `UNORM8` format.
const MODE_UNORM8: u32 = 0;
/// Scalar-kernel `mode` selecting the `UNORM16` format.
const MODE_UNORM16: u32 = 1;
/// Scalar-kernel `mode` selecting the `SNORM8` format.
const MODE_SNORM8: u32 = 2;
/// Scalar-kernel `mode` selecting the `SNORM16` format.
const MODE_SNORM16: u32 = 3;

/// The normalized fixed-point quantizer kernels, mirroring the `CPU` golden
/// [`unorm_snorm_pack`](prism_render_architecture::particle::unorm_snorm_pack)
/// expression for expression. One source file hosts six entry points sharing
/// one bind-group layout and the reference's `quantize_unorm` /
/// `round_half_away_from_zero` / `quantize_snorm` helpers.
const UNORM_SNORM_PACK_WGSL: &str = r#"
// Normalized fixed-point quantizer twin: one thread per element. Two scalar
// kernels (mode-selected) plus four RGBA vector kernels mirror the CPU golden
// `particle::unorm_snorm_pack`. UNORM rounds half-up via floor(x*MAX+0.5);
// SNORM rounds half away from zero via floor(|y|+0.5) with the sign reapplied,
// then clamps to [-MAX, MAX] so the reserved most-negative code is never
// produced. Only floor/clamp/min/max/abs and basic arithmetic appear: no round
// intrinsic, no transcendental, no u64. Portable on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::unorm_snorm_pack；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of valid elements; threads past this short-circuit.
    count: u32,
    // Format selector for the scalar kernels (ignored by the vector kernels).
    mode: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

// Maximum integer codes, matching the reference constants.
const UNORM8_MAX: f32 = 255.0;
const UNORM16_MAX: f32 = 65535.0;
const SNORM8_MAX: f32 = 127.0;
const SNORM16_MAX: f32 = 32767.0;

// Quantize a clamped UNORM value: clamp to [0,1], scale, round half up.
fn quantize_unorm(x: f32, maxv: f32) -> f32 {
    return floor(clamp(x, 0.0, 1.0) * maxv + 0.5);
}

// Round to nearest with ties away from zero, using only floor.
fn round_half_away_from_zero(y: f32) -> f32 {
    if (y >= 0.0) {
        return floor(y + 0.5);
    }
    return -floor((-y) + 0.5);
}

// Quantize a clamped SNORM value: clamp to [-1,1], scale, round half away from
// zero, then clamp the code into [-max, max].
fn quantize_snorm(x: f32, maxv: f32) -> f32 {
    let scaled = clamp(x, -1.0, 1.0) * maxv;
    return clamp(round_half_away_from_zero(scaled), -maxv, maxv);
}

// UNORM pack code as a plain u32; SNORM pack code as the i32 bit pattern.
fn pack_scalar_code(x: f32, mode: u32) -> u32 {
    if (mode == 0u) {
        return u32(quantize_unorm(x, UNORM8_MAX));
    }
    if (mode == 1u) {
        return u32(quantize_unorm(x, UNORM16_MAX));
    }
    if (mode == 2u) {
        return bitcast<u32>(i32(quantize_snorm(x, SNORM8_MAX)));
    }
    return bitcast<u32>(i32(quantize_snorm(x, SNORM16_MAX)));
}

// Reconstruct the real value from a packed code word.
fn unpack_scalar_value(code: u32, mode: u32) -> f32 {
    if (mode == 0u) {
        return f32(code) / UNORM8_MAX;
    }
    if (mode == 1u) {
        return f32(code) / UNORM16_MAX;
    }
    if (mode == 2u) {
        return max(f32(bitcast<i32>(code)) / SNORM8_MAX, -1.0);
    }
    return max(f32(bitcast<i32>(code)) / SNORM16_MAX, -1.0);
}

@compute @workgroup_size(64)
fn pack_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let x = bitcast<f32>(src[idx]);
    dst[idx] = pack_scalar_code(x, params.mode);
}

@compute @workgroup_size(64)
fn unpack_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    dst[idx] = bitcast<u32>(unpack_scalar_value(src[idx], params.mode));
}

@compute @workgroup_size(64)
fn pack_unorm8x4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 4u;
    let r = u32(quantize_unorm(bitcast<f32>(src[base + 0u]), UNORM8_MAX));
    let g = u32(quantize_unorm(bitcast<f32>(src[base + 1u]), UNORM8_MAX));
    let b = u32(quantize_unorm(bitcast<f32>(src[base + 2u]), UNORM8_MAX));
    let a = u32(quantize_unorm(bitcast<f32>(src[base + 3u]), UNORM8_MAX));
    dst[idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(64)
fn unpack_unorm8x4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = src[idx];
    let base = idx * 4u;
    dst[base + 0u] = bitcast<u32>(f32(p & 0xFFu) / UNORM8_MAX);
    dst[base + 1u] = bitcast<u32>(f32((p >> 8u) & 0xFFu) / UNORM8_MAX);
    dst[base + 2u] = bitcast<u32>(f32((p >> 16u) & 0xFFu) / UNORM8_MAX);
    dst[base + 3u] = bitcast<u32>(f32((p >> 24u) & 0xFFu) / UNORM8_MAX);
}

@compute @workgroup_size(64)
fn pack_unorm16x2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 2u;
    let x = u32(quantize_unorm(bitcast<f32>(src[base + 0u]), UNORM16_MAX));
    let y = u32(quantize_unorm(bitcast<f32>(src[base + 1u]), UNORM16_MAX));
    dst[idx] = x | (y << 16u);
}

@compute @workgroup_size(64)
fn unpack_unorm16x2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = src[idx];
    let base = idx * 2u;
    dst[base + 0u] = bitcast<u32>(f32(p & 0xFFFFu) / UNORM16_MAX);
    dst[base + 1u] = bitcast<u32>(f32((p >> 16u) & 0xFFFFu) / UNORM16_MAX);
}
"#;

/// Uniform parameters for one dispatch: the element `count` and the scalar
/// `mode` selector plus padding to a `16`-byte, `std140`-aligned uniform struct
/// matching `Params` in [`UNORM_SNORM_PACK_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements processed by the dispatch.
    count: u32,
    /// Format selector consumed by the scalar kernels.
    mode: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// A compiled, reusable set of normalized fixed-point quantizer kernels
/// (`UNORM`/`SNORM` scalar pack and unpack plus the `RGBA` vector helpers).
pub struct GpuUnormSnormPack {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_pack_scalar: ComputePipeline,
    pipeline_unpack_scalar: ComputePipeline,
    pipeline_pack_unorm8x4: ComputePipeline,
    pipeline_unpack_unorm8x4: ComputePipeline,
    pipeline_pack_unorm16x2: ComputePipeline,
    pipeline_unpack_unorm16x2: ComputePipeline,
}

impl GpuUnormSnormPack {
    /// Compiles the six quantizer kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuUnormSnormPack {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack"),
            source: ShaderSource::Wgsl(UNORM_SNORM_PACK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_pipeline_layout"),
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
        let pipeline_pack_scalar = make(
            "pack_scalar",
            "prism_volumetric_unorm_snorm_pack_pack_scalar_pipeline",
        );
        let pipeline_unpack_scalar = make(
            "unpack_scalar",
            "prism_volumetric_unorm_snorm_pack_unpack_scalar_pipeline",
        );
        let pipeline_pack_unorm8x4 = make(
            "pack_unorm8x4",
            "prism_volumetric_unorm_snorm_pack_pack_unorm8x4_pipeline",
        );
        let pipeline_unpack_unorm8x4 = make(
            "unpack_unorm8x4",
            "prism_volumetric_unorm_snorm_pack_unpack_unorm8x4_pipeline",
        );
        let pipeline_pack_unorm16x2 = make(
            "pack_unorm16x2",
            "prism_volumetric_unorm_snorm_pack_pack_unorm16x2_pipeline",
        );
        let pipeline_unpack_unorm16x2 = make(
            "unpack_unorm16x2",
            "prism_volumetric_unorm_snorm_pack_unpack_unorm16x2_pipeline",
        );
        GpuUnormSnormPack {
            module,
            layout,
            pipeline_pack_scalar,
            pipeline_unpack_scalar,
            pipeline_pack_unorm8x4,
            pipeline_unpack_unorm8x4,
            pipeline_pack_unorm16x2,
            pipeline_unpack_unorm16x2,
        }
    }

    /// Packs each `f32` into a `UNORM8` integer code in `[0, 255]`, mirroring
    /// the reference `pack_unorm8`. Returns one code per input, in order; an
    /// empty input returns an empty vector with no dispatch.
    #[must_use]
    pub fn pack_unorm8(&self, ctx: &GpuContext, xs: &[f32]) -> Vec<u32> {
        self.pack_scalar(ctx, MODE_UNORM8, xs)
    }

    /// Packs each `f32` into a `UNORM16` integer code in `[0, 65535]`, mirroring
    /// the reference `pack_unorm16`. Returns one code per input, in order; an
    /// empty input returns an empty vector with no dispatch.
    #[must_use]
    pub fn pack_unorm16(&self, ctx: &GpuContext, xs: &[f32]) -> Vec<u32> {
        self.pack_scalar(ctx, MODE_UNORM16, xs)
    }

    /// Packs each `f32` into a `SNORM8` integer code in `[-127, 127]`, mirroring
    /// the reference `pack_snorm8`. Returns one signed code per input, in order;
    /// an empty input returns an empty vector with no dispatch.
    #[must_use]
    pub fn pack_snorm8(&self, ctx: &GpuContext, xs: &[f32]) -> Vec<i32> {
        self.pack_scalar(ctx, MODE_SNORM8, xs)
            .into_iter()
            .map(|w| w as i32)
            .collect()
    }

    /// Packs each `f32` into a `SNORM16` integer code in `[-32767, 32767]`,
    /// mirroring the reference `pack_snorm16`. Returns one signed code per
    /// input, in order; an empty input returns an empty vector with no dispatch.
    #[must_use]
    pub fn pack_snorm16(&self, ctx: &GpuContext, xs: &[f32]) -> Vec<i32> {
        self.pack_scalar(ctx, MODE_SNORM16, xs)
            .into_iter()
            .map(|w| w as i32)
            .collect()
    }

    /// Unpacks each `UNORM8` code into `[0, 1]`, mirroring the reference
    /// `unpack_unorm8`. Returns one value per input, in order; an empty input
    /// returns an empty vector with no dispatch.
    #[must_use]
    pub fn unpack_unorm8(&self, ctx: &GpuContext, codes: &[u32]) -> Vec<f32> {
        self.unpack_scalar(ctx, MODE_UNORM8, codes)
    }

    /// Unpacks each `UNORM16` code into `[0, 1]`, mirroring the reference
    /// `unpack_unorm16`. Returns one value per input, in order; an empty input
    /// returns an empty vector with no dispatch.
    #[must_use]
    pub fn unpack_unorm16(&self, ctx: &GpuContext, codes: &[u32]) -> Vec<f32> {
        self.unpack_scalar(ctx, MODE_UNORM16, codes)
    }

    /// Unpacks each `SNORM8` code into `[-1, 1]`, mirroring the reference
    /// `unpack_snorm8` (the reserved `-128` maps to `-1.0`). Returns one value
    /// per input, in order; an empty input returns an empty vector.
    #[must_use]
    pub fn unpack_snorm8(&self, ctx: &GpuContext, codes: &[i32]) -> Vec<f32> {
        let words: Vec<u32> = codes.iter().map(|&c| c as u32).collect();
        self.unpack_scalar(ctx, MODE_SNORM8, &words)
    }

    /// Unpacks each `SNORM16` code into `[-1, 1]`, mirroring the reference
    /// `unpack_snorm16` (the reserved `-32768` maps to `-1.0`). Returns one
    /// value per input, in order; an empty input returns an empty vector.
    #[must_use]
    pub fn unpack_snorm16(&self, ctx: &GpuContext, codes: &[i32]) -> Vec<f32> {
        let words: Vec<u32> = codes.iter().map(|&c| c as u32).collect();
        self.unpack_scalar(ctx, MODE_SNORM16, &words)
    }

    /// Packs four `UNORM8` channels into one `LSB`-first `u32` per element,
    /// mirroring the reference `pack_unorm8x4`. Returns one `u32` per input, in
    /// order; an empty input returns an empty vector with no dispatch.
    #[must_use]
    pub fn pack_unorm8x4(&self, ctx: &GpuContext, texels: &[[f32; 4]]) -> Vec<u32> {
        let count = texels.len();
        let mut words: Vec<u32> = Vec::with_capacity(count * 4);
        for texel in texels {
            for &channel in texel {
                words.push(channel.to_bits());
            }
        }
        self.dispatch(ctx, &self.pipeline_pack_unorm8x4, 0, count, &words, count)
    }

    /// Unpacks a `LSB`-first `RGBA8` `u32` into four `UNORM` channels in
    /// `[0, 1]` per element, mirroring the reference `unpack_unorm8x4`. Returns
    /// one `[f32; 4]` per input, in order; an empty input returns an empty
    /// vector with no dispatch.
    #[must_use]
    pub fn unpack_unorm8x4(&self, ctx: &GpuContext, packed: &[u32]) -> Vec<[f32; 4]> {
        let count = packed.len();
        let out = self.dispatch(
            ctx,
            &self.pipeline_unpack_unorm8x4,
            0,
            count,
            packed,
            count * 4,
        );
        out.chunks_exact(4)
            .map(|c| {
                [
                    f32::from_bits(c[0]),
                    f32::from_bits(c[1]),
                    f32::from_bits(c[2]),
                    f32::from_bits(c[3]),
                ]
            })
            .collect()
    }

    /// Packs two `UNORM16` channels into one `u32` per element (channel 0 low,
    /// channel 1 high), mirroring the reference `pack_unorm16x2`. Returns one
    /// `u32` per input, in order; an empty input returns an empty vector.
    #[must_use]
    pub fn pack_unorm16x2(&self, ctx: &GpuContext, pairs: &[[f32; 2]]) -> Vec<u32> {
        let count = pairs.len();
        let mut words: Vec<u32> = Vec::with_capacity(count * 2);
        for pair in pairs {
            for &channel in pair {
                words.push(channel.to_bits());
            }
        }
        self.dispatch(ctx, &self.pipeline_pack_unorm16x2, 0, count, &words, count)
    }

    /// Unpacks a packed `UNORM16x2` `u32` into two channels in `[0, 1]` per
    /// element, mirroring the reference `unpack_unorm16x2`. Returns one
    /// `[f32; 2]` per input, in order; an empty input returns an empty vector.
    #[must_use]
    pub fn unpack_unorm16x2(&self, ctx: &GpuContext, packed: &[u32]) -> Vec<[f32; 2]> {
        let count = packed.len();
        let out = self.dispatch(
            ctx,
            &self.pipeline_unpack_unorm16x2,
            0,
            count,
            packed,
            count * 2,
        );
        out.chunks_exact(2)
            .map(|c| [f32::from_bits(c[0]), f32::from_bits(c[1])])
            .collect()
    }

    /// Runs a scalar pack of `xs` under `mode`, returning the raw code words.
    fn pack_scalar(&self, ctx: &GpuContext, mode: u32, xs: &[f32]) -> Vec<u32> {
        let words: Vec<u32> = xs.iter().map(|&x| x.to_bits()).collect();
        self.dispatch(
            ctx,
            &self.pipeline_pack_scalar,
            mode,
            xs.len(),
            &words,
            xs.len(),
        )
    }

    /// Runs a scalar unpack of `codes` under `mode`, returning the `f32` values.
    fn unpack_scalar(&self, ctx: &GpuContext, mode: u32, codes: &[u32]) -> Vec<f32> {
        let out = self.dispatch(
            ctx,
            &self.pipeline_unpack_scalar,
            mode,
            codes.len(),
            codes,
            codes.len(),
        );
        out.into_iter().map(f32::from_bits).collect()
    }

    /// Issues one `1-D` dispatch of `pipeline` over `count` elements, uploading
    /// `input_words` and reading back `out_words` `u32` words. Empty inputs
    /// short-circuit without a dispatch because a storage buffer cannot be
    /// zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        mode: u32,
        count: usize,
        input_words: &[u32],
        out_words: usize,
    ) -> Vec<u32> {
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            mode,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_src"),
            contents: bytemuck::cast_slice(input_words),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (out_words * size_of::<u32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_unorm_snorm_pack_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_unorm_snorm_pack_pass"),
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
