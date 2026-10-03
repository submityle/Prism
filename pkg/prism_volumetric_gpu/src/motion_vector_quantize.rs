//! `wgpu` compute twin of the motion-vector fixed-point quantization primitives
//! in the compact velocity-target encoder
//! ([`encode`](prism_render_architecture::motion::encode), motion temporal
//! core).
//!
//! The `CPU` golden quantizes per-pixel motion into a signed 16-bit pair and
//! packs auxiliary masks into bytes so the temporal resolve reads one compact
//! texel per pixel. This twin reproduces the pure numeric quantization kernels —
//! the `snorm16` and `unorm8` scalar codecs and the
//! [`VelocityEncoding`](prism_render_architecture::motion::encode::VelocityEncoding)
//! velocity round-trip — on the device, so a passing real-device parity test is
//! direct evidence the ported kernel produces the same fixed-point codes and
//! dequantized values the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread performs one quantization operation, selected by an op code:
//!
//! - [`encode_snorm16`](prism_render_architecture::motion::encode::encode_snorm16):
//!   clamp to `[-1, 1]`, scale by `32767`, apply a half-step round-away-from-zero
//!   bias, then truncate toward zero. The bias-plus-truncate form is reproduced
//!   exactly (rather than a `round` builtin) so the device matches the golden
//!   bit-for-bit at half-way inputs.
//! - [`decode_snorm16`](prism_render_architecture::motion::encode::decode_snorm16):
//!   divide by `32767` and clamp to `[-1, 1]`.
//! - [`encode_unorm8`](prism_render_architecture::motion::encode::encode_unorm8):
//!   clamp to `[0, 1]`, scale by `255`, add `0.5`, truncate toward zero.
//! - [`decode_unorm8`](prism_render_architecture::motion::encode::decode_unorm8):
//!   divide by `255`.
//! - [`VelocityEncoding::encode`](prism_render_architecture::motion::encode::VelocityEncoding::encode) /
//!   [`decode`](prism_render_architecture::motion::encode::VelocityEncoding::decode):
//!   per-axis normalize by `max_velocity_pixels` then `snorm16` round-trip.
//! - [`VelocityEncoding::quantization_step_pixels`](prism_render_architecture::motion::encode::VelocityEncoding::quantization_step_pixels):
//!   `max_velocity_pixels / 32767`.
//!
//! The signed 16-bit code and the unsigned 8-bit code have no `WGSL` scalar
//! type, so they are carried as `i32` inside the kernel (clamped to their value
//! ranges by construction) and compared on the host as `i32` against the golden
//! `i16` / `u8` widened to `i32`.
//!
//! # What stays on the host
//!
//! The `NaN` resolution inside
//! [`VelocityEncoding::new`](prism_render_architecture::motion::encode::VelocityEncoding::new)
//! and `clamp_signed_unit` / `clamp01` (a `WGSL` kernel cannot test `is_nan`
//! without an `f32` equality) stays on the host: the device consumes
//! already-finite inputs and a `max_velocity_pixels` already floored to its
//! minimum scale, so the kernel needs no `NaN` detection. The mask packing
//! ([`PackedMasks`](prism_render_architecture::motion::encode::PackedMasks)) and
//! the full-sample encoder
//! ([`encode_sample`](prism_render_architecture::motion::encode::encode_sample))
//! are bit-field policy framing a batch, not per-pixel quantization arithmetic,
//! and are not twinned here.
//!
//! # Correctness model
//!
//! The encoded codes are integers built from a clamp, a multiply, an additive
//! bias, and a truncating cast, so for finite fixtures clear of a half-step tie
//! the `CPU` and `GPU` land on the same integer and the parity test asserts an
//! exact `==` on each. The decoded values thread through a divide and a clamp
//! only (no `sqrt`, no transcendental), so `CPU` and `GPU` match to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `select`,
//! `+ - * /`, truncating `i32`/`f32` conversions, and integer equality — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `smoothstep`, and no `sqrt`. Each thread performs a fixed, bounded sequence
//! of arithmetic, so the kernel provably terminates. No optional device feature
//! is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::encode`；无第三方引擎源码或衍生代码。
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

/// Op code: `snorm16` encode of a normalized `[-1, 1]` scalar.
const OP_ENCODE_SNORM16: u32 = 0;
/// Op code: `snorm16` decode back to `[-1, 1]`.
const OP_DECODE_SNORM16: u32 = 1;
/// Op code: `unorm8` encode of a `[0, 1]` scalar.
const OP_ENCODE_UNORM8: u32 = 2;
/// Op code: `unorm8` decode back to `[0, 1]`.
const OP_DECODE_UNORM8: u32 = 3;
/// Op code: velocity encode (per-axis normalize then `snorm16`).
const OP_VELOCITY_ENCODE: u32 = 4;
/// Op code: velocity decode (`snorm16` then per-axis scale).
const OP_VELOCITY_DECODE: u32 = 5;
/// Op code: quantization step in pixels.
const OP_QUANTIZATION_STEP: u32 = 6;

/// The portable core-`WGSL` motion-vector quantization kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`encode`](prism_render_architecture::motion::encode) scalar codecs; see the
/// module documentation for the algorithm.
const MOTION_VECTOR_QUANTIZE_WGSL: &str = r#"
// Per-operation motion-vector quantization twin: one thread performs one
// snorm16/unorm8 codec step or velocity round-trip, selected by an op code,
// mirroring the CPU golden `motion::encode` closed forms with only clamp,
// select, + - * /, and truncating conversions. The signed/unsigned fixed-point
// codes have no WGSL scalar type, so they are carried as i32 inside their value
// ranges. NaN resolution and mask packing stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::motion::encode；无第三方引擎
// 源码或衍生代码。

// Full-scale value of a signed 16-bit fixed-point channel (golden SNORM16_SCALE).
const SNORM16_SCALE: f32 = 32767.0;

// Op codes, matching the host-side constants.
const OP_ENCODE_SNORM16: u32 = 0u;
const OP_DECODE_SNORM16: u32 = 1u;
const OP_ENCODE_UNORM8: u32 = 2u;
const OP_DECODE_UNORM8: u32 = 3u;
const OP_VELOCITY_ENCODE: u32 = 4u;
const OP_VELOCITY_DECODE: u32 = 5u;
const OP_QUANTIZATION_STEP: u32 = 6u;

struct Params {
    // Number of operations in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Selected operation (see op constants).
    op: u32,
    pad0: u32,
    // Float inputs: normalized/unorm value, or velocity x/y components.
    in_f0: f32,
    in_f1: f32,
    // Integer inputs: encoded codes carried as i32.
    in_i0: i32,
    in_i1: i32,
    // Velocity full-scale (already floored to its minimum by the host).
    max_scale: f32,
    pad1: u32,
}

struct Result {
    // Encoded codes carried as i32 (snorm16 in [-32767, 32767], unorm8 in [0, 255]).
    out_i0: i32,
    out_i1: i32,
    // Decoded floats or the quantization step.
    out_f0: f32,
    out_f1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// snorm16 encode: clamp to [-1, 1], scale, half-step round-away-from-zero bias,
// then truncate toward zero (i32 conversion), reproducing the golden
// bias-plus-truncate rather than a round builtin so half-way inputs match
// bit-for-bit. Within range the result stays in [-32767, 32767].
fn encode_snorm16(normalized: f32) -> i32 {
    let clamped = clamp(normalized, -1.0, 1.0);
    let scaled = clamped * SNORM16_SCALE;
    let biased = select(scaled - 0.5, scaled + 0.5, scaled >= 0.0);
    return i32(biased);
}

// snorm16 decode: widen, divide by full scale, clamp back to [-1, 1].
fn decode_snorm16(encoded: i32) -> f32 {
    let v = f32(encoded) / SNORM16_SCALE;
    return clamp(v, -1.0, 1.0);
}

// unorm8 encode: clamp to [0, 1], scale by 255, add half, truncate toward zero.
// The sum is always positive, so truncation equals floor; result in [0, 255].
fn encode_unorm8(value: f32) -> i32 {
    let scaled = clamp(value, 0.0, 1.0) * 255.0 + 0.5;
    return i32(scaled);
}

// unorm8 decode: widen and divide by 255.
fn decode_unorm8(value: i32) -> f32 {
    return f32(value) / 255.0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.out_i0 = 0;
    out.out_i1 = 0;
    out.out_f0 = 0.0;
    out.out_f1 = 0.0;

    if (q.op == OP_ENCODE_SNORM16) {
        out.out_i0 = encode_snorm16(q.in_f0);
    } else if (q.op == OP_DECODE_SNORM16) {
        out.out_f0 = decode_snorm16(q.in_i0);
    } else if (q.op == OP_ENCODE_UNORM8) {
        out.out_i0 = encode_unorm8(q.in_f0);
    } else if (q.op == OP_DECODE_UNORM8) {
        out.out_f0 = decode_unorm8(q.in_i0);
    } else if (q.op == OP_VELOCITY_ENCODE) {
        out.out_i0 = encode_snorm16(q.in_f0 / q.max_scale);
        out.out_i1 = encode_snorm16(q.in_f1 / q.max_scale);
    } else if (q.op == OP_VELOCITY_DECODE) {
        out.out_f0 = decode_snorm16(q.in_i0) * q.max_scale;
        out.out_f1 = decode_snorm16(q.in_i1) * q.max_scale;
    } else if (q.op == OP_QUANTIZATION_STEP) {
        out.out_f0 = q.max_scale / SNORM16_SCALE;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the operation count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MOTION_VECTOR_QUANTIZE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid operations in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one quantization operation, matching the `WGSL`
/// `Query` struct: the op code, two float inputs, two integer (code) inputs, and
/// the velocity full-scale, padded to a `32`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Selected operation.
    op: u32,
    /// Padding word.
    pad0: u32,
    /// First float input (normalized / `unorm` value or velocity `x`).
    in_f0: f32,
    /// Second float input (velocity `y`).
    in_f1: f32,
    /// First integer (code) input, carried as `i32`.
    in_i0: i32,
    /// Second integer (code) input, carried as `i32`.
    in_i1: i32,
    /// Velocity full-scale (already floored to its minimum by the host).
    max_scale: f32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one quantization result, matching the `WGSL`
/// `Result` struct: two integer (code) outputs and two float outputs, a dense
/// `16`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// First encoded code, carried as `i32`.
    out_i0: i32,
    /// Second encoded code, carried as `i32`.
    out_i1: i32,
    /// First decoded float output (or the quantization step).
    out_f0: f32,
    /// Second decoded float output.
    out_f1: f32,
}

/// One motion-vector quantization operation to run on the device, mirroring the
/// golden [`encode`](prism_render_architecture::motion::encode) codecs.
///
/// The `max_velocity_pixels` carried by the velocity variants is expected to be
/// already sanitized by
/// [`VelocityEncoding::new`](prism_render_architecture::motion::encode::VelocityEncoding::new),
/// since the kernel performs no `NaN` detection or minimum-scale flooring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotionVectorQuantizeQuery {
    /// Encode a normalized `[-1, 1]` scalar to `snorm16`.
    EncodeSnorm16 {
        /// Normalized input value.
        normalized: f32,
    },
    /// Decode a `snorm16` code back to `[-1, 1]`.
    DecodeSnorm16 {
        /// Encoded `snorm16` code.
        encoded: i16,
    },
    /// Encode a `[0, 1]` scalar to `unorm8`.
    EncodeUnorm8 {
        /// `[0, 1]` input value.
        value: f32,
    },
    /// Decode a `unorm8` code back to `[0, 1]`.
    DecodeUnorm8 {
        /// Encoded `unorm8` code.
        value: u8,
    },
    /// Quantize a pixel-space velocity to a `snorm16` pair.
    VelocityEncode {
        /// Pixel-space velocity components.
        velocity: [f32; 2],
        /// Full-scale pixel displacement mapping to `+/-1.0`.
        max_velocity_pixels: f32,
    },
    /// Dequantize a `snorm16` pair back to a pixel-space velocity.
    VelocityDecode {
        /// Encoded `snorm16` pair.
        encoded: [i16; 2],
        /// Full-scale pixel displacement mapping to `+/-1.0`.
        max_velocity_pixels: f32,
    },
    /// The worst-case round-trip error per axis: one quantization step.
    QuantizationStep {
        /// Full-scale pixel displacement mapping to `+/-1.0`.
        max_velocity_pixels: f32,
    },
}

/// One resolved quantization result, mirroring the golden
/// [`encode`](prism_render_architecture::motion::encode) codec outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotionVectorQuantizeResult {
    /// `snorm16` code from an [`MotionVectorQuantizeQuery::EncodeSnorm16`].
    EncodeSnorm16 {
        /// Encoded `snorm16` code.
        encoded: i16,
    },
    /// `[-1, 1]` value from an [`MotionVectorQuantizeQuery::DecodeSnorm16`].
    DecodeSnorm16 {
        /// Decoded normalized value.
        normalized: f32,
    },
    /// `unorm8` code from an [`MotionVectorQuantizeQuery::EncodeUnorm8`].
    EncodeUnorm8 {
        /// Encoded `unorm8` code.
        encoded: u8,
    },
    /// `[0, 1]` value from an [`MotionVectorQuantizeQuery::DecodeUnorm8`].
    DecodeUnorm8 {
        /// Decoded `[0, 1]` value.
        value: f32,
    },
    /// `snorm16` pair from an [`MotionVectorQuantizeQuery::VelocityEncode`].
    VelocityEncode {
        /// Encoded `snorm16` pair.
        encoded: [i16; 2],
    },
    /// Pixel-space velocity from an [`MotionVectorQuantizeQuery::VelocityDecode`].
    VelocityDecode {
        /// Decoded pixel-space velocity.
        velocity: [f32; 2],
    },
    /// Quantization step from an [`MotionVectorQuantizeQuery::QuantizationStep`].
    QuantizationStep {
        /// Per-axis quantization step in pixels.
        step_pixels: f32,
    },
}

/// Encodes one [`MotionVectorQuantizeQuery`] into its `std430` [`GpuQuery`] slot,
/// widening each fixed-point code to `i32`.
fn encode_query(q: &MotionVectorQuantizeQuery) -> GpuQuery {
    let mut g = GpuQuery {
        op: OP_ENCODE_SNORM16,
        pad0: 0,
        in_f0: 0.0,
        in_f1: 0.0,
        in_i0: 0,
        in_i1: 0,
        max_scale: 1.0,
        pad1: 0,
    };
    match *q {
        MotionVectorQuantizeQuery::EncodeSnorm16 { normalized } => {
            g.op = OP_ENCODE_SNORM16;
            g.in_f0 = normalized;
        }
        MotionVectorQuantizeQuery::DecodeSnorm16 { encoded } => {
            g.op = OP_DECODE_SNORM16;
            g.in_i0 = i32::from(encoded);
        }
        MotionVectorQuantizeQuery::EncodeUnorm8 { value } => {
            g.op = OP_ENCODE_UNORM8;
            g.in_f0 = value;
        }
        MotionVectorQuantizeQuery::DecodeUnorm8 { value } => {
            g.op = OP_DECODE_UNORM8;
            g.in_i0 = i32::from(value);
        }
        MotionVectorQuantizeQuery::VelocityEncode {
            velocity,
            max_velocity_pixels,
        } => {
            g.op = OP_VELOCITY_ENCODE;
            g.in_f0 = velocity[0];
            g.in_f1 = velocity[1];
            g.max_scale = max_velocity_pixels;
        }
        MotionVectorQuantizeQuery::VelocityDecode {
            encoded,
            max_velocity_pixels,
        } => {
            g.op = OP_VELOCITY_DECODE;
            g.in_i0 = i32::from(encoded[0]);
            g.in_i1 = i32::from(encoded[1]);
            g.max_scale = max_velocity_pixels;
        }
        MotionVectorQuantizeQuery::QuantizationStep {
            max_velocity_pixels,
        } => {
            g.op = OP_QUANTIZATION_STEP;
            g.max_scale = max_velocity_pixels;
        }
    }
    g
}

/// Decodes one packed [`GpuResult`] into the public
/// [`MotionVectorQuantizeResult`], selecting the variant from the original
/// query and narrowing the `i32`-carried codes back to `i16` / `u8`.
fn decode_result(q: &MotionVectorQuantizeQuery, raw: &GpuResult) -> MotionVectorQuantizeResult {
    match *q {
        MotionVectorQuantizeQuery::EncodeSnorm16 { .. } => {
            MotionVectorQuantizeResult::EncodeSnorm16 {
                encoded: raw.out_i0 as i16,
            }
        }
        MotionVectorQuantizeQuery::DecodeSnorm16 { .. } => {
            MotionVectorQuantizeResult::DecodeSnorm16 {
                normalized: raw.out_f0,
            }
        }
        MotionVectorQuantizeQuery::EncodeUnorm8 { .. } => {
            MotionVectorQuantizeResult::EncodeUnorm8 {
                encoded: raw.out_i0 as u8,
            }
        }
        MotionVectorQuantizeQuery::DecodeUnorm8 { .. } => {
            MotionVectorQuantizeResult::DecodeUnorm8 { value: raw.out_f0 }
        }
        MotionVectorQuantizeQuery::VelocityEncode { .. } => {
            MotionVectorQuantizeResult::VelocityEncode {
                encoded: [raw.out_i0 as i16, raw.out_i1 as i16],
            }
        }
        MotionVectorQuantizeQuery::VelocityDecode { .. } => {
            MotionVectorQuantizeResult::VelocityDecode {
                velocity: [raw.out_f0, raw.out_f1],
            }
        }
        MotionVectorQuantizeQuery::QuantizationStep { .. } => {
            MotionVectorQuantizeResult::QuantizationStep {
                step_pixels: raw.out_f0,
            }
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

/// A compiled, reusable motion-vector quantization compute pipeline, twinning
/// the numeric core of the `CPU` golden
/// [`encode`](prism_render_architecture::motion::encode) scalar codecs.
pub struct GpuMotionVectorQuantize {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionVectorQuantize {
    /// Compiles the motion-vector quantization kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionVectorQuantize {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize"),
            source: ShaderSource::Wgsl(MOTION_VECTOR_QUANTIZE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionVectorQuantize {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every operation in `queries` and returns one
    /// [`MotionVectorQuantizeResult`] per input, in order.
    ///
    /// The encoded codes equal the reference exactly for finite fixtures clear
    /// of a half-step tie; the decoded values match to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MotionVectorQuantizeQuery],
    ) -> Vec<MotionVectorQuantizeResult> {
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
            label: Some("prism_volumetric_motion_vector_quantize_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize_bind_group"),
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
            label: Some("prism_volumetric_motion_vector_quantize_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_vector_quantize_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_vector_quantize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per operation, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        queries
            .iter()
            .zip(raw.iter())
            .map(|(q, r)| decode_result(q, r))
            .collect()
    }
}
