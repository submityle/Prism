//! `wgpu` compute twin of the novel closed-form attribute-compression codecs of
//! [`compression`](prism_render_architecture::particle::compression) (design
//! §27): octahedral unit-vector encode/decode, its `snorm16` packing, and the
//! relative uniform quantizer.
//!
//! The golden module owns a family of attribute codecs. Several of them — the
//! half-float `f16` conversions (`f32_to_f16_bits` / `f16_bits_to_f32` /
//! `vec3_to_f16`) and the `unorm8` / `snorm8` / `snorm16` byte codecs
//! (`unorm8_*`, `snorm8_*`, `snorm16_*`, `unorm_rgba8_*`, `snorm_rgba8_*`) — are
//! already twinned by the sibling `half_float_f16` and `unorm_snorm_pack`
//! modules, so this twin deliberately does **not** re-twin them. It covers only
//! the remaining novel, purely algebraic surface:
//!
//! * `oct_encode` — `L1`-fold a direction onto the octahedron, mapping any
//!   nonzero vector to a pair in `-1..=1` (a numerically zero vector to
//!   `(0, 0)`).
//! * `oct_decode` — the algebraic inverse, re-folding and renormalizing with a
//!   guarded `sqrt` (`normalize_or_zero`).
//! * `oct_encode_snorm16` / `oct_decode_snorm16` — the same fold composed with
//!   the `snorm16` bit codec (round-half-away-from-zero plus a symmetric clamp).
//! * `quantize_relative` / `dequantize_relative` — uniform quantization of a
//!   scalar into `bits` bits over a closed `min..=max` range, with the
//!   degenerate-range and `bits >= 32` edges handled exactly as the golden.
//!
//! # Why op-code dispatch
//!
//! Each twinned routine is a small, independently testable numeric kernel.
//! Rather than compile one pipeline per routine, every [`CompressionQuery`]
//! carries an operation code, and the single `solve` kernel branches on it (an
//! `if` / `else if` ladder on an unsigned code, an exact integer compare). One
//! thread handles one query; the batch may freely mix operations. This keeps a
//! single shader module and bind-group layout while still surfacing every
//! primitive to the parity suite.
//!
//! # Left on the host (not twinned)
//!
//! `recommend_encoding` (an `enum` dispatch over attribute semantics),
//! `estimate_bandwidth` (a 64-bit byte-total reduction over a variable-length
//! layout plan) and `AttributeLayoutPlan::compression_ratio` (a struct method
//! folding that reduction) stay on the host: they are integer/`enum` bookkeeping
//! over variable-length data with no fixed-size per-lane device counterpart. The
//! `f16` / `unorm` / `snorm` byte codecs stay with their existing twins as noted
//! above.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `abs`, `sqrt`, `select`, integer bit shifts, `bitcast` and
//! `+ - * /` with integer/`f32` value conversions — and no transcendental call
//! or optional device feature, so it runs unmodified on Metal, Vulkan and DX12.
//! The golden `round` is reproduced as a round-half-away-from-zero built from
//! `floor`; the `bits >= 32` quantizer level count is a separate branch so the
//! shader never evaluates the undefined `1u << 32`.
//!
//! # Correctness model
//!
//! Every twinned routine is fixed closed-form algebra, so `CPU` and `GPU`
//! evaluate the same expression. The continuous octahedral and dequantized
//! outputs are not bit-exact in general — a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate — so the parity test asserts an
//! absolute-or-relative tolerance on them, while the integer `snorm16` codes and
//! quantizer codes must match exactly (their fixtures stay a half-step clear of
//! the rounding boundary so no legal `ULP` slack can flip the rounded integer).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::compression`
//! octahedral unit-vector 与 relative-quantization 闭式编解码加 `wgpu` compute
//! dispatch；纯整数/无超越数学，无需外部数学库，无第三方引擎源码或衍生代码。
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::compression::{
    dequantize_relative, oct_decode, oct_decode_snorm16, oct_encode, oct_encode_snorm16,
    quantize_relative,
};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The compute-shader workgroup size: one thread evaluates one query.
const WORKGROUP_SIZE: u32 = 64;

// Operation codes shared by the host encoder and the `solve` kernel. Each tags
// one golden routine; the kernel branches on the code with an exact integer
// compare.
const OP_OCT_ENCODE: u32 = 0;
const OP_OCT_DECODE: u32 = 1;
const OP_OCT_ENCODE_SNORM16: u32 = 2;
const OP_OCT_DECODE_SNORM16: u32 = 3;
const OP_QUANTIZE: u32 = 4;
const OP_DEQUANTIZE: u32 = 5;

/// The portable core-`WGSL` attribute-compression kernel, embedded inline so the
/// twin ships as a single source file. One thread evaluates one query, branching
/// on its operation code; see the module documentation for the op-dispatch
/// rationale.
const COMPRESSION_WGSL: &str = r#"
// Attribute-compression codec twin: one thread per query evaluates a single
// golden routine selected by an operation code. It mirrors the CPU golden
// `particle::compression` octahedral and relative-quantization codecs, uses only
// the portable core-WGSL subset (min/max/clamp/floor/abs/sqrt/select, integer
// bit shifts, bitcast and + - * / plus integer/f32 value conversions) and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: octahedral unit-vector 与 relative-quantization 闭式编解码；
// 纯整数/无超越数学，无需外部数学库，无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid lanes in the batch, one thread each.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query's packed inputs. 48-byte std430 stride matching the host GpuQuery.
struct Query {
    // Operation code selecting the golden routine.
    op: u32,
    // Quantizer bit width (quantize/dequantize ops).
    bits: u32,
    // Quantized code input (dequantize op).
    code: u32,
    ipad0: u32,
    // Float operands: packed per op (direction xyz / encoded xy / value,min,max).
    args: vec4<f32>,
    // Signed integer operands: the snorm16 code pair for the decode op.
    scodes: vec4<i32>,
}

// One query's packed outputs. 32-byte std430 stride matching the host GpuResult.
struct Res {
    // Continuous outputs: oct pair in xy, decoded vector in xyz, scalar in x.
    v: vec4<f32>,
    // Integer outputs: snorm16 code pair in xy, quantized code (bitcast) in x.
    codes: vec4<i32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

const OP_OCT_ENCODE: u32 = 0u;
const OP_OCT_DECODE: u32 = 1u;
const OP_OCT_ENCODE_SNORM16: u32 = 2u;
const OP_OCT_DECODE_SNORM16: u32 = 3u;
const OP_QUANTIZE: u32 = 4u;
const OP_DEQUANTIZE: u32 = 5u;

// The L1 length threshold below which a direction is treated as zero, matching
// the golden `EPS_LEN_SQ`; also the guarded-normalize length-squared threshold.
const EPS_LEN_SQ: f32 = 1e-12;
// The largest snorm16 magnitude, matching the golden `32767.0`.
const SNORM16_MAX: f32 = 32767.0;

// `+1.0` when `x >= 0`, otherwise `-1.0` (the octahedral fold sign), matching
// the golden `nonneg_sign`.
fn nonneg_sign(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Octahedral-encode a direction into two components in -1..=1, matching the
// golden `oct_encode`.
fn oct_encode(dir: vec3<f32>) -> vec2<f32> {
    let l1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
    if (l1 <= EPS_LEN_SQ) {
        return vec2<f32>(0.0, 0.0);
    }
    let inv = 1.0 / l1;
    let px = dir.x * inv;
    let py = dir.y * inv;
    let pz = dir.z * inv;
    if (pz >= 0.0) {
        return vec2<f32>(px, py);
    }
    let fx = (1.0 - abs(py)) * nonneg_sign(px);
    let fy = (1.0 - abs(px)) * nonneg_sign(py);
    return vec2<f32>(fx, fy);
}

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero,
// matching the golden `Vec3::normalize_or_zero` (guarded sqrt).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Decode an octahedral pair back into a unit vector, matching the golden
// `oct_decode`.
fn oct_decode(enc: vec2<f32>) -> vec3<f32> {
    var x = enc.x;
    var y = enc.y;
    let z = 1.0 - abs(x) - abs(y);
    if (z < 0.0) {
        let fx = (1.0 - abs(y)) * nonneg_sign(x);
        let fy = (1.0 - abs(x)) * nonneg_sign(y);
        x = fx;
        y = fy;
    }
    return normalize_or_zero(vec3<f32>(x, y, z));
}

// Round half away from zero, the WGSL-safe equivalent of the golden `round`.
fn round_half_away_from_zero(y: f32) -> f32 {
    if (y >= 0.0) {
        return floor(y + 0.5);
    }
    return -floor(-y + 0.5);
}

// Encode a value in -1..=1 as a snorm16 code, matching the golden
// `snorm16_encode` (clamp, round-half-away, symmetric clamp).
fn snorm16_encode(value: f32) -> i32 {
    let scaled = clamp(value, -1.0, 1.0) * SNORM16_MAX;
    let rounded = round_half_away_from_zero(scaled);
    return i32(clamp(rounded, -SNORM16_MAX, SNORM16_MAX));
}

// Decode a snorm16 code back into -1..=1, matching the golden `snorm16_decode`.
fn snorm16_decode(c: i32) -> f32 {
    return max(f32(c) / SNORM16_MAX, -1.0);
}

// The largest code a `bits`-bit unsigned quantizer can emit, matching the
// golden `max_code`. The `bits >= 32` branch avoids the undefined `1u << 32`.
fn max_code(bits: u32) -> u32 {
    if (bits >= 32u) {
        return 4294967295u;
    }
    if (bits == 0u) {
        return 0u;
    }
    return (1u << bits) - 1u;
}

// Uniformly quantize `value` into `bits` bits over the closed range
// `mn..=mx`, matching the golden `quantize_relative`.
fn quantize_relative(value: f32, mn: f32, mx: f32, bits_in: u32) -> u32 {
    if (bits_in == 0u) {
        return 0u;
    }
    let bits = min(bits_in, 32u);
    let range = mx - mn;
    if (range <= 0.0) {
        return 0u;
    }
    let levels = max_code(bits);
    let t = clamp((value - mn) / range, 0.0, 1.0);
    return u32(floor(t * f32(levels) + 0.5));
}

// Reconstruct the representative value of a relative-quantized `code`, matching
// the golden `dequantize_relative`.
fn dequantize_relative(code: u32, mn: f32, mx: f32, bits_in: u32) -> f32 {
    if (bits_in == 0u) {
        return mn;
    }
    let bits = min(bits_in, 32u);
    let levels = max_code(bits);
    if (levels == 0u) {
        return mn;
    }
    let range = mx - mn;
    let t = f32(min(code, levels)) / f32(levels);
    return mn + t * range;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    if (lane >= params.count) {
        return;
    }
    let q = queries[lane];
    var out: Res;
    out.v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.codes = vec4<i32>(0, 0, 0, 0);
    let op = q.op;
    if (op == OP_OCT_ENCODE) {
        let e = oct_encode(vec3<f32>(q.args.x, q.args.y, q.args.z));
        out.v.x = e.x;
        out.v.y = e.y;
    } else if (op == OP_OCT_DECODE) {
        let d = oct_decode(vec2<f32>(q.args.x, q.args.y));
        out.v.x = d.x;
        out.v.y = d.y;
        out.v.z = d.z;
    } else if (op == OP_OCT_ENCODE_SNORM16) {
        let e = oct_encode(vec3<f32>(q.args.x, q.args.y, q.args.z));
        out.codes.x = snorm16_encode(e.x);
        out.codes.y = snorm16_encode(e.y);
    } else if (op == OP_OCT_DECODE_SNORM16) {
        let dx = snorm16_decode(q.scodes.x);
        let dy = snorm16_decode(q.scodes.y);
        let d = oct_decode(vec2<f32>(dx, dy));
        out.v.x = d.x;
        out.v.y = d.y;
        out.v.z = d.z;
    } else if (op == OP_QUANTIZE) {
        let code = quantize_relative(q.args.x, q.args.y, q.args.z, q.bits);
        out.codes.x = bitcast<i32>(code);
    } else if (op == OP_DEQUANTIZE) {
        out.v.x = dequantize_relative(q.code, q.args.x, q.args.y, q.bits);
    }
    results[lane] = out;
}
"#;

/// One attribute-compression query: a tagged request to evaluate a single golden
/// codec routine on the device.
///
/// Each variant names one novel routine of the `CPU` golden
/// [`compression`](prism_render_architecture::particle::compression) and carries
/// just that routine's operands. Holds `f32` operands, so it derives only
/// [`Clone`], [`Copy`], [`Debug`] and [`PartialEq`] (no [`Eq`] / [`Hash`]).
/// Provenance: query tagging for the `compression` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompressionQuery {
    /// Octahedral-encode a direction, mirroring the golden `oct_encode`.
    /// `Provenance:` `oct_encode`.
    OctEncode {
        /// The direction `[x, y, z]` (any nonzero length is accepted).
        direction: [f32; 3],
    },
    /// Decode an octahedral pair into a unit vector, mirroring the golden
    /// `oct_decode`. `Provenance:` `oct_decode`.
    OctDecode {
        /// The octahedral pair `[x, y]` in `-1..=1`.
        encoded: [f32; 2],
    },
    /// Octahedral-encode a direction into two `snorm16` codes, mirroring the
    /// golden `oct_encode_snorm16`. `Provenance:` `oct_encode_snorm16`.
    OctEncodeSnorm16 {
        /// The direction `[x, y, z]` (any nonzero length is accepted).
        direction: [f32; 3],
    },
    /// Decode two `snorm16` codes into a unit vector, mirroring the golden
    /// `oct_decode_snorm16`. The codes are carried as `i32` for a `std430`
    /// layout and narrowed to `i16` on the host before the golden call.
    /// `Provenance:` `oct_decode_snorm16`.
    OctDecodeSnorm16 {
        /// The `snorm16` code pair `[c0, c1]`.
        code: [i32; 2],
    },
    /// Uniformly quantize a scalar into `bits` bits over `min..=max`, mirroring
    /// the golden `quantize_relative`. `Provenance:` `quantize_relative`.
    QuantizeRelative {
        /// The value to quantize.
        value: f32,
        /// Inclusive lower bound of the quantization range.
        min: f32,
        /// Inclusive upper bound of the quantization range.
        max: f32,
        /// Bits per component (clamped to `1..=32` when applied).
        bits: u32,
    },
    /// Reconstruct a relative-quantized `code`, mirroring the golden
    /// `dequantize_relative`. `Provenance:` `dequantize_relative`.
    DequantizeRelative {
        /// The quantized code.
        code: u32,
        /// Inclusive lower bound of the quantization range.
        min: f32,
        /// Inclusive upper bound of the quantization range.
        max: f32,
        /// Bits per component (clamped to `1..=32` when applied).
        bits: u32,
    },
}

/// The result of one attribute-compression query, mirroring the golden return of
/// the routine the query names.
///
/// Provenance: result tagging for the `compression` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompressionResult {
    /// An octahedral `(x, y)` pair.
    OctPair {
        /// The first octahedral component.
        x: f32,
        /// The second octahedral component.
        y: f32,
    },
    /// A decoded three-channel unit vector.
    Vector {
        /// The `[x, y, z]` components.
        v: [f32; 3],
    },
    /// A `snorm16` code pair.
    Snorm16Pair {
        /// The `[c0, c1]` codes.
        codes: [i32; 2],
    },
    /// A single unsigned quantized code.
    Code {
        /// The quantized code.
        code: u32,
    },
    /// A single reconstructed scalar.
    Scalar {
        /// The scalar value.
        value: f32,
    },
}

/// The `CPU` golden verdict for one query, delegating to the matching routine of
/// [`compression`](prism_render_architecture::particle::compression) so callers
/// (and the parity test) can pin the twin lane for lane.
///
/// Every twinned routine is a published `pub` function of the golden module, so
/// each arm calls it directly rather than reimplementing the formula.
///
/// Provenance: `CPU` reference for the `compression` twin, 孪生自本仓
/// `prism_render_architecture::particle::compression`.
#[must_use]
pub fn cpu_reference(query: &CompressionQuery) -> CompressionResult {
    match query {
        CompressionQuery::OctEncode { direction } => {
            let (x, y) = oct_encode(Vec3::new(direction[0], direction[1], direction[2]));
            CompressionResult::OctPair { x, y }
        }
        CompressionQuery::OctDecode { encoded } => {
            let v = oct_decode((encoded[0], encoded[1]));
            CompressionResult::Vector { v: [v.x, v.y, v.z] }
        }
        CompressionQuery::OctEncodeSnorm16 { direction } => {
            let c = oct_encode_snorm16(Vec3::new(direction[0], direction[1], direction[2]));
            CompressionResult::Snorm16Pair {
                codes: [i32::from(c[0]), i32::from(c[1])],
            }
        }
        CompressionQuery::OctDecodeSnorm16 { code } => {
            let v = oct_decode_snorm16([code[0] as i16, code[1] as i16]);
            CompressionResult::Vector { v: [v.x, v.y, v.z] }
        }
        CompressionQuery::QuantizeRelative {
            value,
            min,
            max,
            bits,
        } => CompressionResult::Code {
            code: quantize_relative(*value, *min, *max, *bits as u8),
        },
        CompressionQuery::DequantizeRelative {
            code,
            min,
            max,
            bits,
        } => CompressionResult::Scalar {
            value: dequantize_relative(*code, *min, *max, *bits as u8),
        },
    }
}

/// Uniform dispatch parameters. `repr(C)` `std430` layout matching `Params` in
/// [`COMPRESSION_WGSL`]: the lane count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader: four leading `u32` fields, one float operand `vec4` and one signed
/// integer operand `vec4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op: u32,
    bits: u32,
    code: u32,
    ipad0: u32,
    args: [f32; 4],
    scodes: [i32; 4],
}

/// One result as read back. `32`-byte `std430` stride matching `Res` in the
/// shader: a float output `vec4` and a signed integer output `vec4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    v: [f32; 4],
    codes: [i32; 4],
}

/// Encodes one query into its packed `GpuQuery`, placing each operand in the
/// slot the kernel reads for that operation code.
fn encode(query: &CompressionQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    match query {
        CompressionQuery::OctEncode { direction } => {
            g.op = OP_OCT_ENCODE;
            g.args = [direction[0], direction[1], direction[2], 0.0];
        }
        CompressionQuery::OctDecode { encoded } => {
            g.op = OP_OCT_DECODE;
            g.args = [encoded[0], encoded[1], 0.0, 0.0];
        }
        CompressionQuery::OctEncodeSnorm16 { direction } => {
            g.op = OP_OCT_ENCODE_SNORM16;
            g.args = [direction[0], direction[1], direction[2], 0.0];
        }
        CompressionQuery::OctDecodeSnorm16 { code } => {
            g.op = OP_OCT_DECODE_SNORM16;
            g.scodes = [code[0], code[1], 0, 0];
        }
        CompressionQuery::QuantizeRelative {
            value,
            min,
            max,
            bits,
        } => {
            g.op = OP_QUANTIZE;
            g.bits = *bits;
            g.args = [*value, *min, *max, 0.0];
        }
        CompressionQuery::DequantizeRelative {
            code,
            min,
            max,
            bits,
        } => {
            g.op = OP_DEQUANTIZE;
            g.bits = *bits;
            g.code = *code;
            g.args = [*min, *max, 0.0, 0.0];
        }
    }
    g
}

/// Decodes one raw `GpuResult` into the typed result the query shape implies.
fn decode(query: &CompressionQuery, r: &GpuResult) -> CompressionResult {
    match query {
        CompressionQuery::OctEncode { .. } => CompressionResult::OctPair {
            x: r.v[0],
            y: r.v[1],
        },
        CompressionQuery::OctDecode { .. } | CompressionQuery::OctDecodeSnorm16 { .. } => {
            CompressionResult::Vector {
                v: [r.v[0], r.v[1], r.v[2]],
            }
        }
        CompressionQuery::OctEncodeSnorm16 { .. } => CompressionResult::Snorm16Pair {
            codes: [r.codes[0], r.codes[1]],
        },
        CompressionQuery::QuantizeRelative { .. } => CompressionResult::Code {
            code: r.codes[0] as u32,
        },
        CompressionQuery::DequantizeRelative { .. } => CompressionResult::Scalar { value: r.v[0] },
    }
}

/// A compiled, reusable attribute-compression codec-evaluation pipeline.
///
/// Provenance: `wgpu` compute twin of
/// `prism_render_architecture::particle::compression`.
pub struct GpuCompression {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCompression {
    /// Compiles the attribute-compression codec kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required. Provenance: pipeline construction for the
    /// `compression` twin.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCompression {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_compression"),
            source: ShaderSource::Wgsl(COMPRESSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_compression_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_compression_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_compression_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCompression {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one [`CompressionResult`]
    /// per query in input order.
    ///
    /// Each lane reproduces the golden routine its query names, to within the
    /// tolerance documented on this module (exact for the integer codes). An
    /// empty `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return. Provenance: codec
    /// evaluation for the `compression` twin.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[CompressionQuery]) -> Vec<CompressionResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries.iter().map(encode).collect();
        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_compression_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_compression_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_compression_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_compression_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_compression_bind_group"),
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
            label: Some("prism_volumetric_compression_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_compression_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, in workgroups of 64 (the kernel's size).
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        queries
            .iter()
            .zip(gpu_results.iter())
            .map(|(query, raw)| decode(query, raw))
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
