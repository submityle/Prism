//! `wgpu` compute twin of the octahedral unit-normal codec, from the `CPU`
//! golden `prism_math::octahedral` (`encode` / `decode` / `pack_snorm` /
//! `unpack_snorm`).
//!
//! Octahedral mapping stores a unit direction in two numbers by projecting the
//! sphere onto an octahedron and unfolding it to the `[-1, 1]` square. It is the
//! standard compact normal encoding for `GPU` g-buffers and vertex streams.
//! This module ports the four stateless closed forms onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same values the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query selects one of four operations with an unsigned `op_id`:
//!
//! * `0` encode: a direction `(vx, vy, vz)` to octahedral coordinates
//!   `(ox, oy)` in `[-1, 1]`. The input is normalized by its `L1` norm, so any
//!   non-zero collinear vector encodes identically. The lower hemisphere
//!   (`vz < 0`) is folded across the octahedron edges with a non-zero sign.
//! * `1` decode: octahedral coordinates `(vx, vy)` back to a unit direction
//!   `(ox, oy, oz)`.
//! * `2` pack: encode then quantize to a packed `u32` holding two 16-bit signed
//!   normalized channels (`x` low, `y` high), with round-to-nearest ties away
//!   from zero.
//! * `3` unpack: a packed `u32` back to a unit direction.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` are not necessarily bit-exact; continuous outputs are
//! compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `bits` word from the pack path and the
//! `valid` flag are compared exactly: the snorm quantization reproduces the
//! golden round-ties-away rule with `floor(x + 0.5)` / `ceil(x - 0.5)` and the
//! same two's-complement 16-bit truncation, so the packed word matches
//! bit-for-bit.
//!
//! # Degenerate inputs
//!
//! `encode` and `pack` of the zero (or non-finite) vector have no well-defined
//! `L1` normalization, so they report `valid = 0` with zeroed outputs; the
//! divisor is fed through a `select` guard so the invalid branch never produces
//! `inf`/`NaN`. `decode` and `unpack` are always valid (their folding and final
//! normalization are total), as is any valid `op_id`; an unknown `op_id`
//! reports `valid = 0`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `floor`,
//! `ceil`, `clamp`, `max`, `length`, `select`, `+ - * /` and unsigned/signed
//! integer bit arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`,
//! no `round`, no `f32` remainder, and no `u64`/`i64`/`f64`/`u16`/`i16`. The
//! snorm channels are carried in `i32`/`u32` with explicit 16-bit masking and
//! sign extension. Finiteness is an ordered `abs(x) < 3.0e38` compare and the
//! degeneracy guard an ordered `> 1e-20` compare; the operation dispatch and
//! bit tests use only `u32` equality, so there is no bare `f32` equality
//! anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::octahedral`；无第三方引擎源码或衍生代码。
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

/// `op_id` selecting the `encode` operation (direction to octahedral `(x, y)`).
pub const OP_ENCODE: u32 = 0;
/// `op_id` selecting the `decode` operation (octahedral `(x, y)` to direction).
pub const OP_DECODE: u32 = 1;
/// `op_id` selecting the `pack_snorm` operation (direction to packed `u32`).
pub const OP_PACK: u32 = 2;
/// `op_id` selecting the `unpack_snorm` operation (packed `u32` to direction).
pub const OP_UNPACK: u32 = 3;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` octahedral-codec kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `prism_math::octahedral`; see the module documentation for the
/// closed forms.
const OCTAHEDRAL_ENCODE_WGSL: &str = r#"
// Octahedral codec twin: one thread per query selects one of four operations
// (encode/decode/pack/unpack) by an unsigned op id. It uses only the portable
// core-WGSL subset; the snorm channels are kept in i32/u32 with explicit 16-bit
// masking and sign extension so no u16/i16/f64 ever appears. All comparisons are
// ordered float compares or u32 equality on ids/bit patterns; there is no bare
// f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation selector: 0 encode, 1 decode, 2 pack, 3 unpack.
    op_id: u32,
    // Packed snorm input for the unpack operation.
    bits: u32,
    // Input vector / octahedral coordinates (decode reads vx, vy only).
    vx: f32,
    vy: f32,
    vz: f32,
    // Padding words to a 32-byte stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Vector / octahedral output (encode writes ox, oy; decode/unpack all three).
    ox: f32,
    oy: f32,
    oz: f32,
    // Packed snorm output for the pack operation.
    bits: u32,
    // 1 when the operation produced a defined result, else 0.
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const TINY_L1: f32 = 1.0e-20;

// Branchless sign that returns +1.0 for v >= 0 (including +0.0) and -1.0
// otherwise, matching the golden sign_nonzero fold convention.
fn sign_nonzero(v: f32) -> f32 {
    return select(-1.0, 1.0, v >= 0.0);
}

// Encode a direction to octahedral coordinates in [-1, 1]. The divisor is
// guarded so a zero / degenerate vector never produces inf/NaN; validity is
// decided separately by the caller.
fn octa_encode(x: f32, y: f32, z: f32) -> vec2<f32> {
    let l1 = abs(x) + abs(y) + abs(z);
    let denom = select(1.0, l1, l1 > TINY_L1);
    let inv_l1 = 1.0 / denom;
    let px = x * inv_l1;
    let py = y * inv_l1;
    let fx = (1.0 - abs(py)) * sign_nonzero(px);
    let fy = (1.0 - abs(px)) * sign_nonzero(py);
    let upper = z >= 0.0;
    let ex = select(fx, px, upper);
    let ey = select(fy, py, upper);
    return vec2<f32>(ex, ey);
}

// Decode octahedral coordinates in [-1, 1] back to a unit direction.
fn octa_decode(ex: f32, ey: f32) -> vec3<f32> {
    let nz = 1.0 - abs(ex) - abs(ey);
    let t = max(-nz, 0.0);
    let nx = ex + select(t, -t, ex >= 0.0);
    let ny = ey + select(t, -t, ey >= 0.0);
    let v = vec3<f32>(nx, ny, nz);
    let inv_len = 1.0 / length(v);
    return v * inv_len;
}

// Round to nearest, ties away from zero: floor(x + 0.5) for x >= 0, else
// ceil(x - 0.5). This reproduces the golden libm roundf semantics without
// using the forbidden round builtin.
fn round_ties_away(x: f32) -> f32 {
    return select(ceil(x - 0.5), floor(x + 0.5), x >= 0.0);
}

// Quantize a value in [-1, 1] to 16-bit snorm, returned in the low 16 bits of a
// u32 as a two's-complement pattern.
fn snorm16(v: f32) -> u32 {
    let c = clamp(v, -1.0, 1.0);
    let scaled = c * 32767.0;
    let r = i32(round_ties_away(scaled));
    return u32(r) & 0xFFFFu;
}

// Dequantize a 16-bit snorm pattern (low 16 bits of b16) back to [-1, 1].
fn unsnorm16(b16: u32) -> f32 {
    var s = i32(b16);
    if ((b16 & 0x8000u) != 0u) {
        s = s - 65536;
    }
    let value = f32(s) / 32767.0;
    return clamp(value, -1.0, 1.0);
}

// Encode and quantize a direction to a packed u32 (x low 16 bits, y high 16).
fn octa_pack(x: f32, y: f32, z: f32) -> u32 {
    let e = octa_encode(x, y, z);
    let qx = snorm16(e.x);
    let qy = snorm16(e.y);
    return qx | (qy << 16u);
}

// Decode a packed u32 back to a unit direction.
fn octa_unpack(bits: u32) -> vec3<f32> {
    let x = unsnorm16(bits & 0xFFFFu);
    let y = unsnorm16((bits >> 16u) & 0xFFFFu);
    return octa_decode(x, y);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let op = q.op_id;

    // Degeneracy / finiteness decision for the encode and pack paths: all three
    // components finite (ordered abs < 3.0e38 rejects +/-inf and NaN) and the
    // L1 norm strictly above a tiny floor.
    let finite_in =
        (abs(q.vx) < FINITE_LIMIT) && (abs(q.vy) < FINITE_LIMIT) && (abs(q.vz) < FINITE_LIMIT);
    let l1 = abs(q.vx) + abs(q.vy) + abs(q.vz);
    let vec_ok = finite_in && (l1 > TINY_L1);

    let enc = octa_encode(q.vx, q.vy, q.vz);
    let dec = octa_decode(q.vx, q.vy);
    let packed = octa_pack(q.vx, q.vy, q.vz);
    let unp = octa_unpack(q.bits);

    var out: Result;
    out.ox = 0.0;
    out.oy = 0.0;
    out.oz = 0.0;
    out.bits = 0u;
    out.valid = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    if (op == 0u) {
        // encode
        out.ox = select(0.0, enc.x, vec_ok);
        out.oy = select(0.0, enc.y, vec_ok);
        out.valid = select(0u, 1u, vec_ok);
    } else if (op == 1u) {
        // decode
        out.ox = dec.x;
        out.oy = dec.y;
        out.oz = dec.z;
        out.valid = 1u;
    } else if (op == 2u) {
        // pack
        out.bits = select(0u, packed, vec_ok);
        out.valid = select(0u, 1u, vec_ok);
    } else if (op == 3u) {
        // unpack
        out.ox = unp.x;
        out.oy = unp.y;
        out.oz = unp.z;
        out.valid = 1u;
    }

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// all scalar words (two `u32`, three `f32`, three padding `u32`) for a flat
/// `32`-byte stride with no `vec3` alignment ambiguity.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op_id: u32,
    bits: u32,
    vx: f32,
    vy: f32,
    vz: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: three `f32` outputs, the packed `bits`, the `valid` flag and three
/// padding words — a flat `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    ox: f32,
    oy: f32,
    oz: f32,
    bits: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One octahedral-codec query: the operation selector plus its inputs.
///
/// The vector components `vx`/`vy`/`vz` carry the direction for `encode`/`pack`
/// and the octahedral coordinates (in `vx`/`vy`) for `decode`; `bits` carries
/// the packed word for `unpack`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctahedralEncodeQuery {
    /// Operation selector: [`OP_ENCODE`], [`OP_DECODE`], [`OP_PACK`] or
    /// [`OP_UNPACK`]; any other value reports `valid = 0`.
    pub op_id: u32,
    /// Packed snorm input for [`OP_UNPACK`].
    pub bits: u32,
    /// First vector / octahedral coordinate.
    pub vx: f32,
    /// Second vector / octahedral coordinate.
    pub vy: f32,
    /// Third vector component (ignored by `decode`/`unpack`).
    pub vz: f32,
}

impl OctahedralEncodeQuery {
    /// Builds an `encode` query for the direction `(x, y, z)`.
    #[must_use]
    pub fn encode(x: f32, y: f32, z: f32) -> OctahedralEncodeQuery {
        OctahedralEncodeQuery {
            op_id: OP_ENCODE,
            bits: 0,
            vx: x,
            vy: y,
            vz: z,
        }
    }

    /// Builds a `decode` query for the octahedral coordinates `(ex, ey)`.
    #[must_use]
    pub fn decode(ex: f32, ey: f32) -> OctahedralEncodeQuery {
        OctahedralEncodeQuery {
            op_id: OP_DECODE,
            bits: 0,
            vx: ex,
            vy: ey,
            vz: 0.0,
        }
    }

    /// Builds a `pack` query for the direction `(x, y, z)`.
    #[must_use]
    pub fn pack(x: f32, y: f32, z: f32) -> OctahedralEncodeQuery {
        OctahedralEncodeQuery {
            op_id: OP_PACK,
            bits: 0,
            vx: x,
            vy: y,
            vz: z,
        }
    }

    /// Builds an `unpack` query for the packed snorm word `bits`.
    #[must_use]
    pub fn unpack(bits: u32) -> OctahedralEncodeQuery {
        OctahedralEncodeQuery {
            op_id: OP_UNPACK,
            bits,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `prism_math::octahedral` output for the selected operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctahedralEncodeResult {
    /// First output component: `encode` x, or `decode`/`unpack` direction x.
    pub ox: f32,
    /// Second output component: `encode` y, or `decode`/`unpack` direction y.
    pub oy: f32,
    /// Third output component: `decode`/`unpack` direction z (else `0`).
    pub oz: f32,
    /// Packed snorm word produced by `pack` (else `0`).
    pub bits: u32,
    /// `1` when the operation produced a defined result, else `0`.
    pub valid: u32,
}

/// Encodes one [`OctahedralEncodeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &OctahedralEncodeQuery) -> GpuQuery {
    GpuQuery {
        op_id: q.op_id,
        bits: q.bits,
        vx: q.vx,
        vy: q.vy,
        vz: q.vz,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`OctahedralEncodeResult`].
fn decode_result(raw: &GpuResult) -> OctahedralEncodeResult {
    OctahedralEncodeResult {
        ox: raw.ox,
        oy: raw.oy,
        oz: raw.oz,
        bits: raw.bits,
        valid: raw.valid,
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

/// A compiled, reusable octahedral-codec compute pipeline, twinning the `CPU`
/// golden `prism_math::octahedral`.
pub struct GpuOctahedralEncode {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuOctahedralEncode {
    /// Compiles the octahedral-codec kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOctahedralEncode {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_octahedral_encode"),
            source: ShaderSource::Wgsl(OCTAHEDRAL_ENCODE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_octahedral_encode_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_octahedral_encode_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_octahedral_encode_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOctahedralEncode {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`OctahedralEncodeResult`] per input, in order.
    ///
    /// The `valid` flag and the packed `bits` word match the reference exactly;
    /// the continuous outputs match to the module's tolerance. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[OctahedralEncodeQuery],
    ) -> Vec<OctahedralEncodeResult> {
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
            label: Some("prism_volumetric_octahedral_encode_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_octahedral_encode_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_octahedral_encode_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_octahedral_encode_bind_group"),
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
            label: Some("prism_volumetric_octahedral_encode_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_octahedral_encode_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_octahedral_encode_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
