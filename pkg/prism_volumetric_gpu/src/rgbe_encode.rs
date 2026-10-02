//! `wgpu` compute twin of the shared-exponent `HDR` packers
//! ([`rgbe_encode`](prism_render_architecture::particle::rgbe_encode), particle
//! design §27): Radiance `RGBE` (`[u8; 4]`) and Khronos `RGB9E5` (`u32`), the
//! two color formats that trade a per-pixel exponent for a large dynamic range
//! at a fraction of a full `f32x3`'s footprint.
//!
//! Emissive particles, bloom sources and light-shaft cookies routinely carry
//! radiance well above `1.0`, so a `unorm8` channel clips them and a full
//! `f32x3` wastes bandwidth. The `CPU` golden
//! [`rgbe_encode`](prism_render_architecture::particle::rgbe_encode) owns the
//! closed-form packing math; [`GpuRgbeEncode`] is the on-device twin, so a
//! passing real-device parity test is direct evidence the ported kernel packs
//! the same bits the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread per element reproduces every public reference routine:
//!
//! * [`GpuRgbeEncode::rgbe_encode`] mirrors the golden
//!   [`rgbe_encode`](prism_render_architecture::particle::rgbe_encode::rgbe_encode):
//!   a shared power-of-two exponent plus an 8-bit mantissa per channel, packed
//!   little-endian into a `u32` and read back as `[u8; 4]`.
//! * [`GpuRgbeEncode::rgbe_decode`] mirrors the golden
//!   [`rgbe_decode`](prism_render_architecture::particle::rgbe_encode::rgbe_decode).
//! * [`GpuRgbeEncode::rgb9e5_encode`] mirrors the golden
//!   [`rgb9e5_encode`](prism_render_architecture::particle::rgbe_encode::rgb9e5_encode):
//!   the Khronos `E5B9G9R9` layout with a 5-bit shared exponent and a 9-bit
//!   mantissa per channel.
//! * [`GpuRgbeEncode::rgb9e5_decode`] mirrors the golden
//!   [`rgb9e5_decode`](prism_render_architecture::particle::rgbe_encode::rgb9e5_decode).
//! * [`GpuRgbeEncode::primitives`] exposes the shared `IEEE754` primitives
//!   (`max_channel`, `channel_byte`, `floor_log2`, `pow2_i32`) directly so the
//!   parity test can assert the bit-exact integer paths on their own.
//!
//! # Step-for-step parity
//!
//! The exponent is separated straight out of the `IEEE754` layout: `floor_log2`
//! reads the biased exponent field via `bitcast<u32>` (no `log2`), and `pow2_i32`
//! rebuilds each `2^n` by writing the biased exponent field back via
//! `bitcast<f32>`, with the subnormal octaves set by a single mantissa bit and
//! the over-range case saturating to a `bitcast`-constructed infinity (no `exp2`
//! or `pow`). `channel_byte` floors a `clamp`ed product, and the `RGB9E5`
//! round-to-nearest uses `floor(x + 0.5)`. Negative and `NaN` inputs are guarded
//! to `0.0` through an integer `bitcast` `NaN` test, so the kernel never emits a
//! `NaN` or an out-of-range code.
//!
//! # Correctness model
//!
//! `floor_log2`, `pow2_i32`, the byte quantization and the final bit assembly
//! are pure integer / `bitcast` work, and `WGSL` unsigned integers wrap exactly
//! like Rust's `wrapping_*`, so the packed `RGBE` bytes and the `RGB9E5` word
//! are bit-identical and the parity test asserts an exact `==` on every code.
//! The only values that can diverge are the decoded continuous channels, and
//! only by a legal rounding of a few units in the last place, so those are
//! compared with `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`/`max`/`clamp`/
//! `floor`, `+ - * /`, unsigned bit ops and `bitcast` — with no `sin`, `cos`,
//! `exp`, `log`, `pow` or optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no `sqrt` here.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::rgbe_encode`；无第三方引擎源码或衍生代码。
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

/// Threads per workgroup; one thread encodes or decodes one element.
const WORKGROUP_SIZE: u32 = 64;

/// The shared-exponent `HDR` packer kernels, embedded inline so the twin ships
/// as one self-contained source with no external `.wgsl` asset. Five entry
/// points share one bind-group layout over flat `u32` word buffers.
const RGBE_ENCODE_WGSL: &str = r#"
// Shared-exponent HDR packer twin: one thread per color/word. Five entry
// points mirror the CPU golden `particle::rgbe_encode`: `rgbe_encode`,
// `rgbe_decode`, `rgb9e5_encode`, `rgb9e5_decode`, and a `prim_eval` probe
// exposing max_channel / channel_byte / floor_log2 / pow2_i32 so parity can
// assert the bit-exact IEEE754 primitives directly. `src` and `dst` are flat
// u32 word arrays; each kernel strides them by its own element width. Pure
// min/max/clamp/floor, + - * /, bit ops and bitcast: no sin/cos/exp/log/pow,
// no transcendental, portable on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::rgbe_encode；无第三方
// 引擎源码或衍生代码。

const RGBE_EXP_BIAS: i32 = 128;
const RGBE_MANTISSA_BITS: i32 = 8;
const RGBE_MIN: f32 = 1.0e-32;
const RGB9E5_MANTISSA_BITS: i32 = 9;
const RGB9E5_EXP_BIAS: i32 = 15;
const RGB9E5_MAX_EXP: i32 = 31;
const RGB9E5_MAX_VALUE: f32 = 65408.0;
// IEEE754 single-precision positive-infinity bit pattern, used both to build an
// infinity without a transcendental and as the NaN comparison threshold.
const INF_BITS: u32 = 0x7f800000u;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

// NaN test by magnitude bit pattern: a float is NaN iff its sign-stripped bits
// exceed the single infinity pattern. Pure unsigned compare, so no float
// equality is used.
fn is_nan_bits(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > INF_BITS;
}

// Largest of the three channels, pinning the shared exponent. Inputs are always
// sanitized/clamped before this call, so no NaN reaches it.
fn max_channel(r: f32, g: f32, b: f32) -> f32 {
    return max(max(r, g), b);
}

// Replaces negative or NaN inputs with 0, leaving valid magnitudes alone,
// mirroring the golden `sanitize`.
fn sanitize(x: f32) -> f32 {
    if (is_nan_bits(x) || x <= 0.0) {
        return 0.0;
    }
    return x;
}

// Floors a non-negative product into a color byte, clamping to 0..=255,
// mirroring the golden `channel_byte`.
fn channel_byte(v: f32) -> u32 {
    let clamped = clamp(v, 0.0, 255.0);
    return u32(clamped);
}

// Unbiased base-2 exponent of a positive f32 straight from its IEEE754 exponent
// field, i.e. floor(log2(x)); zero/subnormal report a value far below any bias.
// Mirrors the golden `floor_log2` bit for bit.
fn floor_log2(x: f32) -> i32 {
    let biased = (bitcast<u32>(x) >> 23u) & 0xffu;
    if (biased == 0u) {
        return -128;
    }
    return i32(biased) - 127;
}

// Rebuilds 2^n as an exact f32 purely by integer bit assembly, mirroring the
// golden `pow2_i32`: direct biased-exponent write for the normal range, a single
// mantissa bit for the subnormal octaves, saturation to infinity above and
// underflow to 0 below.
fn pow2_i32(n: i32) -> f32 {
    if (n >= -126 && n <= 127) {
        let biased = u32(n + 127);
        return bitcast<f32>(biased << 23u);
    }
    if (n > 127) {
        return bitcast<f32>(INF_BITS);
    }
    if (n >= -149 && n <= -127) {
        let shift = u32(n + 149);
        return bitcast<f32>(1u << shift);
    }
    return 0.0;
}

// Clamps one channel into the RGB9E5 range, mapping negatives/NaN to 0 and
// everything above MAXVAL down to it, mirroring the golden `clamp_rgb9e5`.
fn clamp_rgb9e5(x: f32) -> f32 {
    if (is_nan_bits(x)) {
        return 0.0;
    }
    return clamp(x, 0.0, RGB9E5_MAX_VALUE);
}

// Rounds value/denom to the nearest 9-bit mantissa via floor(x + 0.5),
// mirroring the golden `quantize_mantissa`.
fn quantize_mantissa(value: f32, denom: f32) -> u32 {
    let m = floor(value / denom + 0.5);
    return u32(m);
}

// Encodes one linear HDR color into a little-endian-packed RGBE word, mirroring
// the golden `rgbe_encode`.
fn rgbe_encode_one(r_in: f32, g_in: f32, b_in: f32) -> u32 {
    let r = sanitize(r_in);
    let g = sanitize(g_in);
    let b = sanitize(b_in);
    let m = max_channel(r, g, b);
    if (m < RGBE_MIN) {
        return 0u;
    }
    let frexp_exp = floor_log2(m) + 1;
    let clamped_exp = clamp(frexp_exp, -RGBE_EXP_BIAS, 255 - RGBE_EXP_BIAS);
    let scale = pow2_i32(RGBE_MANTISSA_BITS - clamped_exp);
    let rb = channel_byte(r * scale);
    let gb = channel_byte(g * scale);
    let bb = channel_byte(b * scale);
    let eb = u32(clamped_exp + RGBE_EXP_BIAS);
    return rb | (gb << 8u) | (bb << 16u) | (eb << 24u);
}

// Encodes one linear HDR color into a Khronos RGB9E5 (E5B9G9R9) word, mirroring
// the golden `rgb9e5_encode`.
fn rgb9e5_encode_one(r_in: f32, g_in: f32, b_in: f32) -> u32 {
    let rc = clamp_rgb9e5(r_in);
    let gc = clamp_rgb9e5(g_in);
    let bc = clamp_rgb9e5(b_in);
    let maxc = max_channel(rc, gc, bc);
    let raw = max(floor_log2(maxc), -RGB9E5_EXP_BIAS - 1) + 1 + RGB9E5_EXP_BIAS;
    var exp_shared = raw;
    var denom = pow2_i32(exp_shared - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS);
    let maxm = floor(maxc / denom + 0.5);
    if (maxm >= 512.0) {
        exp_shared = exp_shared + 1;
        denom = denom * 2.0;
    }
    exp_shared = clamp(exp_shared, 0, RGB9E5_MAX_EXP);
    let rm = quantize_mantissa(rc, denom);
    let gm = quantize_mantissa(gc, denom);
    let bm = quantize_mantissa(bc, denom);
    let exp_bits = u32(exp_shared) & 0x1fu;
    return (exp_bits << 27u) | (bm << 18u) | (gm << 9u) | rm;
}

@compute @workgroup_size(64)
fn rgbe_encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 3u;
    let r = bitcast<f32>(src[base]);
    let g = bitcast<f32>(src[base + 1u]);
    let b = bitcast<f32>(src[base + 2u]);
    dst[idx] = rgbe_encode_one(r, g, b);
}

@compute @workgroup_size(64)
fn rgbe_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let word = src[idx];
    let exp_byte = (word >> 24u) & 0xffu;
    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    if (exp_byte != 0u) {
        let scale = pow2_i32(i32(exp_byte) - RGBE_EXP_BIAS - RGBE_MANTISSA_BITS);
        r = f32(word & 0xffu) * scale;
        g = f32((word >> 8u) & 0xffu) * scale;
        b = f32((word >> 16u) & 0xffu) * scale;
    }
    let base = idx * 3u;
    dst[base] = bitcast<u32>(r);
    dst[base + 1u] = bitcast<u32>(g);
    dst[base + 2u] = bitcast<u32>(b);
}

@compute @workgroup_size(64)
fn rgb9e5_encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 3u;
    let r = bitcast<f32>(src[base]);
    let g = bitcast<f32>(src[base + 1u]);
    let b = bitcast<f32>(src[base + 2u]);
    dst[idx] = rgb9e5_encode_one(r, g, b);
}

@compute @workgroup_size(64)
fn rgb9e5_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let value = src[idx];
    let exp_shared = (value >> 27u) & 0x1fu;
    let rm = value & 0x1ffu;
    let gm = (value >> 9u) & 0x1ffu;
    let bm = (value >> 18u) & 0x1ffu;
    let scale = pow2_i32(i32(exp_shared) - RGB9E5_EXP_BIAS - RGB9E5_MANTISSA_BITS);
    let base = idx * 3u;
    dst[base] = bitcast<u32>(f32(rm) * scale);
    dst[base + 1u] = bitcast<u32>(f32(gm) * scale);
    dst[base + 2u] = bitcast<u32>(f32(bm) * scale);
}

@compute @workgroup_size(64)
fn prim_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 8u;
    let r = bitcast<f32>(src[base]);
    let g = bitcast<f32>(src[base + 1u]);
    let b = bitcast<f32>(src[base + 2u]);
    let v = bitcast<f32>(src[base + 3u]);
    let x = bitcast<f32>(src[base + 4u]);
    let n = bitcast<i32>(src[base + 5u]);
    let ob = idx * 4u;
    dst[ob] = bitcast<u32>(max_channel(r, g, b));
    dst[ob + 1u] = channel_byte(v);
    dst[ob + 2u] = bitcast<u32>(floor_log2(x));
    dst[ob + 3u] = bitcast<u32>(pow2_i32(n));
}
"#;

/// Uniform parameters for one dispatch: the valid element count plus padding to
/// a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RGBE_ENCODE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid elements; threads past this short-circuit.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One probe of the shared `IEEE754` primitives, exercising the four reference
/// helpers in a single thread.
///
/// `max_rgb` feeds the golden `max_channel`, `channel_value` the golden
/// `channel_byte`, `log2_input` the golden `floor_log2`, and `pow2_exponent`
/// the golden `pow2_i32`. The fields are independent, so one query can span the
/// full exponent range of `pow2_i32` while still checking the other three.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::rgbe_encode`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug)]
pub struct RgbePrimQuery {
    /// The three channels fed to `max_channel`.
    pub max_rgb: [f32; 3],
    /// The value fed to `channel_byte`.
    pub channel_value: f32,
    /// The value fed to `floor_log2`.
    pub log2_input: f32,
    /// The exponent fed to `pow2_i32`.
    pub pow2_exponent: i32,
}

/// One probe result: the four shared-primitive outputs for a [`RgbePrimQuery`].
///
/// `channel_byte`, `floor_log2` and `pow2` are bit-exact (`channel_byte` and
/// `floor_log2` are integers; `pow2` is reproduced to the bit); `max_channel`
/// is a plain selection and is compared with a continuous tolerance.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::rgbe_encode`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug)]
pub struct RgbePrimResult {
    /// `max_channel(max_rgb)`.
    pub max_channel: f32,
    /// `channel_byte(channel_value)`.
    pub channel_byte: u8,
    /// `floor_log2(log2_input)`.
    pub floor_log2: i32,
    /// `pow2_i32(pow2_exponent)`.
    pub pow2: f32,
}

/// A compiled, reusable set of shared-exponent `HDR` packer kernels, twinning
/// the `CPU` golden
/// [`rgbe_encode`](prism_render_architecture::particle::rgbe_encode).
pub struct GpuRgbeEncode {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_rgbe_encode: ComputePipeline,
    pipeline_rgbe_decode: ComputePipeline,
    pipeline_rgb9e5_encode: ComputePipeline,
    pipeline_rgb9e5_decode: ComputePipeline,
    pipeline_prim: ComputePipeline,
}

impl GpuRgbeEncode {
    /// Compiles the packer kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRgbeEncode {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rgbe_encode"),
            source: ShaderSource::Wgsl(RGBE_ENCODE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rgbe_encode_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rgbe_encode_pipeline_layout"),
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
        let pipeline_rgbe_encode = make(
            "rgbe_encode",
            "prism_volumetric_rgbe_encode_rgbe_encode_pipeline",
        );
        let pipeline_rgbe_decode = make(
            "rgbe_decode",
            "prism_volumetric_rgbe_encode_rgbe_decode_pipeline",
        );
        let pipeline_rgb9e5_encode = make(
            "rgb9e5_encode",
            "prism_volumetric_rgbe_encode_rgb9e5_encode_pipeline",
        );
        let pipeline_rgb9e5_decode = make(
            "rgb9e5_decode",
            "prism_volumetric_rgbe_encode_rgb9e5_decode_pipeline",
        );
        let pipeline_prim = make("prim_eval", "prism_volumetric_rgbe_encode_prim_pipeline");
        GpuRgbeEncode {
            module,
            layout,
            pipeline_rgbe_encode,
            pipeline_rgbe_decode,
            pipeline_rgb9e5_encode,
            pipeline_rgb9e5_decode,
            pipeline_prim,
        }
    }

    /// Encodes each linear `HDR` color into a Radiance `RGBE` quad, mirroring the
    /// golden
    /// [`rgbe_encode`](prism_render_architecture::particle::rgbe_encode::rgbe_encode).
    ///
    /// Returns one `[u8; 4]` per color, in order. An empty slice yields an empty
    /// vector with no dispatch issued (a storage buffer cannot be zero-sized).
    #[must_use]
    pub fn rgbe_encode(&self, ctx: &GpuContext, colors: &[[f32; 3]]) -> Vec<[u8; 4]> {
        let src: Vec<u32> = colors
            .iter()
            .flat_map(|c| [c[0].to_bits(), c[1].to_bits(), c[2].to_bits()])
            .collect();
        let words = self.dispatch(
            ctx,
            &self.pipeline_rgbe_encode,
            colors.len(),
            &src,
            colors.len(),
        );
        words.into_iter().map(u32::to_le_bytes).collect()
    }

    /// Decodes each Radiance `RGBE` quad into a linear `HDR` color, mirroring the
    /// golden
    /// [`rgbe_decode`](prism_render_architecture::particle::rgbe_encode::rgbe_decode).
    ///
    /// Returns one `[f32; 3]` per quad, in order. An empty slice yields an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn rgbe_decode(&self, ctx: &GpuContext, quads: &[[u8; 4]]) -> Vec<[f32; 3]> {
        let src: Vec<u32> = quads.iter().map(|q| u32::from_le_bytes(*q)).collect();
        let words = self.dispatch(
            ctx,
            &self.pipeline_rgbe_decode,
            quads.len(),
            &src,
            quads.len() * 3,
        );
        words
            .chunks_exact(3)
            .map(|c| {
                [
                    f32::from_bits(c[0]),
                    f32::from_bits(c[1]),
                    f32::from_bits(c[2]),
                ]
            })
            .collect()
    }

    /// Encodes each linear `HDR` color into a Khronos `RGB9E5` (`E5B9G9R9`)
    /// word, mirroring the golden
    /// [`rgb9e5_encode`](prism_render_architecture::particle::rgbe_encode::rgb9e5_encode).
    ///
    /// Returns one `u32` per color, in order. An empty slice yields an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn rgb9e5_encode(&self, ctx: &GpuContext, colors: &[[f32; 3]]) -> Vec<u32> {
        let src: Vec<u32> = colors
            .iter()
            .flat_map(|c| [c[0].to_bits(), c[1].to_bits(), c[2].to_bits()])
            .collect();
        self.dispatch(
            ctx,
            &self.pipeline_rgb9e5_encode,
            colors.len(),
            &src,
            colors.len(),
        )
    }

    /// Decodes each Khronos `RGB9E5` (`E5B9G9R9`) word into a linear `HDR`
    /// color, mirroring the golden
    /// [`rgb9e5_decode`](prism_render_architecture::particle::rgbe_encode::rgb9e5_decode).
    ///
    /// Returns one `[f32; 3]` per word, in order. An empty slice yields an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn rgb9e5_decode(&self, ctx: &GpuContext, words: &[u32]) -> Vec<[f32; 3]> {
        let src: Vec<u32> = words.to_vec();
        let out = self.dispatch(
            ctx,
            &self.pipeline_rgb9e5_decode,
            words.len(),
            &src,
            words.len() * 3,
        );
        out.chunks_exact(3)
            .map(|c| {
                [
                    f32::from_bits(c[0]),
                    f32::from_bits(c[1]),
                    f32::from_bits(c[2]),
                ]
            })
            .collect()
    }

    /// Evaluates the four shared `IEEE754` primitives for every query, mirroring
    /// the golden `max_channel`, `channel_byte`, `floor_log2` and `pow2_i32`.
    ///
    /// Returns one [`RgbePrimResult`] per query, in order. An empty slice yields
    /// an empty vector with no dispatch issued.
    #[must_use]
    pub fn primitives(&self, ctx: &GpuContext, queries: &[RgbePrimQuery]) -> Vec<RgbePrimResult> {
        let src: Vec<u32> = queries
            .iter()
            .flat_map(|q| {
                [
                    q.max_rgb[0].to_bits(),
                    q.max_rgb[1].to_bits(),
                    q.max_rgb[2].to_bits(),
                    q.channel_value.to_bits(),
                    q.log2_input.to_bits(),
                    q.pow2_exponent as u32,
                    0u32,
                    0u32,
                ]
            })
            .collect();
        let out = self.dispatch(
            ctx,
            &self.pipeline_prim,
            queries.len(),
            &src,
            queries.len() * 4,
        );
        out.chunks_exact(4)
            .map(|c| RgbePrimResult {
                max_channel: f32::from_bits(c[0]),
                channel_byte: c[1] as u8,
                floor_log2: c[2] as i32,
                pow2: f32::from_bits(c[3]),
            })
            .collect()
    }

    /// Issues one `1-D` dispatch of `pipeline` over `count` elements, uploading
    /// the flat `src` word buffer and reading back `dst_words` output words.
    ///
    /// Empty batches short-circuit without a dispatch because a storage buffer
    /// cannot be zero-sized. One thread handles one element in workgroups of
    /// `WORKGROUP_SIZE`, with the group count computed by `div_ceil`.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        count: usize,
        src: &[u32],
        dst_words: usize,
    ) -> Vec<u32> {
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rgbe_encode_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rgbe_encode_src"),
            contents: bytemuck::cast_slice(src),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (dst_words * size_of::<u32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rgbe_encode_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rgbe_encode_bind_group"),
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
            label: Some("prism_volumetric_rgbe_encode_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rgbe_encode_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rgbe_encode_pass"),
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
