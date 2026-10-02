//! `wgpu` compute twin of the `BT.601` / `BT.709` `RGB` <-> `YCbCr` colour
//! transform golden
//! ([`ycbcr_bt709`](prism_render_architecture::particle::ycbcr_bt709),
//! particle design §21).
//!
//! The `CPU` golden
//! [`ycbcr_bt709`](prism_render_architecture::particle::ycbcr_bt709) owns the
//! `CPU`-verifiable maths of the video colour transform: the forward matrix
//! [`rgb_to_ycbcr`](prism_render_architecture::particle::ycbcr_bt709::rgb_to_ycbcr)
//! that splits a colour into one `luma` channel (`Y`) and two `chroma` channels
//! (`Cb`, `Cr`) under the standard-defined `Kr`/`Kg`/`Kb` weights, the inverse
//! [`ycbcr_to_rgb`](prism_render_architecture::particle::ycbcr_bt709::ycbcr_to_rgb),
//! the `8`-bit quantiser
//! [`quantize_8bit`](prism_render_architecture::particle::ycbcr_bt709::quantize_8bit)
//! and its inverse
//! [`dequantize_8bit`](prism_render_architecture::particle::ycbcr_bt709::dequantize_8bit).
//! [`GpuYcbcrBt709`] is the on-device twin: one thread evaluates one per-element
//! query, the op code selecting which of the four routines runs, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same colour maths and emits the same codewords the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Exactly the four per-element routines are reproduced, each parameterised by
//! the `luma`-weight standard
//! ([`Coefficients`](prism_render_architecture::particle::ycbcr_bt709::Coefficients))
//! and the signal range
//! ([`Range`](prism_render_architecture::particle::ycbcr_bt709::Range)): the
//! forward and inverse matrix transforms and the `8`-bit quantiser and
//! dequantiser. The variable-length `chroma` subsamplers
//! ([`subsample_420`](prism_render_architecture::particle::ycbcr_bt709::subsample_420)
//! and
//! [`subsample_422`](prism_render_architecture::particle::ycbcr_bt709::subsample_422))
//! are block reductions over a `2x2` / horizontal-pair neighbourhood rather than
//! one-thread-per-element maps, so they are out of scope for this per-element
//! twin and are left to the `CPU` golden.
//!
//! # Correctness model
//!
//! The two matrix transforms and the dequantiser thread through multiplies,
//! adds and guarded divisions, so `CPU` and `GPU` are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous channel. The
//! quantiser emits integer codewords, so there is nothing to round apart; the
//! parity test asserts an exact `==` on each byte, and the test fixtures are
//! kept clear of the half-codeword rounding boundary (by rejection sampling) so
//! a fused multiply-add in `digital = chroma * 255 + 128` cannot tip a tie to a
//! different byte on one device.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `floor`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt`, no `round` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::ycbcr_bt709::{
    dequantize_8bit, quantize_8bit, rgb_to_ycbcr, ycbcr_to_rgb, Coefficients, Range, Rgb, YCbCr,
};
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

/// The portable core-`WGSL` `YCbCr` colour kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`ycbcr_bt709`](prism_render_architecture::particle::ycbcr_bt709) branch for
/// branch; see the module documentation for the algorithm.
const YCBCR_BT709_WGSL: &str = r#"
// YCbCr BT.601/BT.709 twin: one thread per element reproduces one of four
// per-element colour routines selected by the query's op code: the forward
// RGB -> YCbCr matrix, the inverse YCbCr -> RGB matrix, the 8-bit quantiser and
// the 8-bit dequantiser, each parameterised by the luma-weight standard (coeff)
// and the signal range. It mirrors the CPU golden particle::ycbcr_bt709 branch
// for branch, uses only the portable core-WGSL subset (clamp/floor and + - * /
// plus unsigned index math), needs no sqrt and no transcendental call and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12. There
// is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::ycbcr_bt709; no
// third-party engine source or derived code.

// Y studio-swing offset (16/255) and scale (219/255) applied in the limited
// (broadcast) range, matching the reference LIMITED_Y_OFFSET / LIMITED_Y_SCALE.
const LIMITED_Y_OFFSET: f32 = 16.0 / 255.0;
const LIMITED_Y_SCALE: f32 = 219.0 / 255.0;
// Cb/Cr studio-swing offset (128/255) and scale (224/255), matching the
// reference LIMITED_C_OFFSET / LIMITED_C_SCALE.
const LIMITED_C_OFFSET: f32 = 128.0 / 255.0;
const LIMITED_C_SCALE: f32 = 224.0 / 255.0;

// Op codes selecting the per-element routine; match the host YcbcrOp order.
const OP_RGB_TO_YCBCR: u32 = 0u;
const OP_YCBCR_TO_RGB: u32 = 1u;
const OP_QUANTIZE: u32 = 2u;

// Range codes: full swing versus studio (limited) swing. Match the host Range.
const RANGE_LIMITED: u32 = 1u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Continuous input channels: RGB for the forward transform or YCbCr for the
    // inverse transform and the quantiser.
    inputs: vec3<f32>,
    // Op code selecting the routine.
    op: u32,
    // Byte codewords (0..255) consumed by the dequantiser.
    codes: vec3<u32>,
    // Luma-weight standard: 0 = BT.601, 1 = BT.709.
    coeff: u32,
    // Signal range: 0 = full, 1 = limited studio swing.
    range: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Continuous output channels (YCbCr or RGB); zero for the quantise op.
    values: vec3<f32>,
    pad0: f32,
    // Byte codewords; zero for every op but the quantiser.
    bytes: vec3<u32>,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Returns the (Kr, Kg, Kb) luma weights for the selected standard, mirroring the
// reference Coefficients::weights. The three always sum to 1.
fn luma_weights(coeff: u32) -> vec3<f32> {
    if (coeff == 1u) {
        return vec3<f32>(0.2126, 0.7152, 0.0722);
    }
    return vec3<f32>(0.299, 0.587, 0.114);
}

// Rounds a digital-scale value to the nearest [0, 255] byte without any
// transcendental call: add 0.5, floor, then clamp. Mirrors the reference
// round_clamp_u8 (floor(x + 0.5) then clamp(.., 0, 255)); the clamped value is
// integer-valued inside [0, 255] so the u32 narrowing is exact.
fn round_clamp_u8(digital: f32) -> u32 {
    let rounded = floor(digital + 0.5);
    let clamped = clamp(rounded, 0.0, 255.0);
    return u32(clamped);
}

// Forward RGB -> YCbCr under the given coeff and range, mirroring the reference
// rgb_to_ycbcr: Y is the luma-weighted sum, Cb/Cr are the blue/red differences,
// then the limited range applies the studio offset and scale.
fn forward(rgb: vec3<f32>, coeff: u32, range: u32) -> vec3<f32> {
    let k = luma_weights(coeff);
    let y = k.x * rgb.x + k.y * rgb.y + k.z * rgb.z;
    let cb = (rgb.z - y) / (2.0 * (1.0 - k.z));
    let cr = (rgb.x - y) / (2.0 * (1.0 - k.x));
    if (range == RANGE_LIMITED) {
        return vec3<f32>(
            LIMITED_Y_OFFSET + LIMITED_Y_SCALE * y,
            LIMITED_C_OFFSET + LIMITED_C_SCALE * cb,
            LIMITED_C_OFFSET + LIMITED_C_SCALE * cr,
        );
    }
    return vec3<f32>(y, cb, cr);
}

// Inverse YCbCr -> RGB under the given coeff and range, mirroring the reference
// ycbcr_to_rgb: the limited range removes the studio scale and offset first,
// then the linear matrix reconstructs RGB.
fn inverse(ycbcr: vec3<f32>, coeff: u32, range: u32) -> vec3<f32> {
    let k = luma_weights(coeff);
    var y = ycbcr.x;
    var cb = ycbcr.y;
    var cr = ycbcr.z;
    if (range == RANGE_LIMITED) {
        y = (ycbcr.x - LIMITED_Y_OFFSET) / LIMITED_Y_SCALE;
        cb = (ycbcr.y - LIMITED_C_OFFSET) / LIMITED_C_SCALE;
        cr = (ycbcr.z - LIMITED_C_OFFSET) / LIMITED_C_SCALE;
    }
    let r = y + 2.0 * (1.0 - k.x) * cr;
    let b = y + 2.0 * (1.0 - k.z) * cb;
    let g = y - (2.0 * k.x * (1.0 - k.x) / k.y) * cr - (2.0 * k.z * (1.0 - k.z) / k.y) * cb;
    return vec3<f32>(r, g, b);
}

// Quantises a YCbCr sample to three 8-bit codewords, mirroring the reference
// quantize_8bit: the full range biases chroma by 128 before rounding, while the
// limited range carries its studio offset already so each channel is simply
// scaled by 255.
fn quantize(ycbcr: vec3<f32>, range: u32) -> vec3<u32> {
    if (range == RANGE_LIMITED) {
        return vec3<u32>(
            round_clamp_u8(ycbcr.x * 255.0),
            round_clamp_u8(ycbcr.y * 255.0),
            round_clamp_u8(ycbcr.z * 255.0),
        );
    }
    return vec3<u32>(
        round_clamp_u8(ycbcr.x * 255.0),
        round_clamp_u8(ycbcr.y * 255.0 + 128.0),
        round_clamp_u8(ycbcr.z * 255.0 + 128.0),
    );
}

// Dequantises three 8-bit codewords back into a YCbCr sample, mirroring the
// reference dequantize_8bit: the full range removes the 128 chroma bias.
fn dequantize(codes: vec3<u32>, range: u32) -> vec3<f32> {
    let y = f32(codes.x) / 255.0;
    if (range == RANGE_LIMITED) {
        return vec3<f32>(y, f32(codes.y) / 255.0, f32(codes.z) / 255.0);
    }
    return vec3<f32>(
        y,
        (f32(codes.y) - 128.0) / 255.0,
        (f32(codes.z) - 128.0) / 255.0,
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var values = vec3<f32>(0.0, 0.0, 0.0);
    var bytes = vec3<u32>(0u, 0u, 0u);
    if (q.op == OP_RGB_TO_YCBCR) {
        values = forward(q.inputs, q.coeff, q.range);
    } else if (q.op == OP_YCBCR_TO_RGB) {
        values = inverse(q.inputs, q.coeff, q.range);
    } else if (q.op == OP_QUANTIZE) {
        bytes = quantize(q.inputs, q.range);
    } else {
        values = dequantize(q.codes, q.range);
    }

    var out: Result;
    out.values = values;
    out.pad0 = 0.0;
    out.bytes = bytes;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Selects which of the four per-element routines a [`YcbcrQuery`] evaluates.
///
/// The discriminant order matches the `op` codes the kernel branches on, so the
/// host encode is a plain cast.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum YcbcrOp {
    /// Forward transform `RGB` -> `YCbCr`
    /// ([`rgb_to_ycbcr`](prism_render_architecture::particle::ycbcr_bt709::rgb_to_ycbcr)).
    RgbToYcbcr,
    /// Inverse transform `YCbCr` -> `RGB`
    /// ([`ycbcr_to_rgb`](prism_render_architecture::particle::ycbcr_bt709::ycbcr_to_rgb)).
    YcbcrToRgb,
    /// `8`-bit quantiser `YCbCr` -> three codewords
    /// ([`quantize_8bit`](prism_render_architecture::particle::ycbcr_bt709::quantize_8bit)).
    Quantize,
    /// `8`-bit dequantiser three codewords -> `YCbCr`
    /// ([`dequantize_8bit`](prism_render_architecture::particle::ycbcr_bt709::dequantize_8bit)).
    Dequantize,
}

impl YcbcrOp {
    /// Returns the `u32` op code the kernel branches on for this routine.
    #[must_use]
    const fn code(self) -> u32 {
        match self {
            YcbcrOp::RgbToYcbcr => 0,
            YcbcrOp::YcbcrToRgb => 1,
            YcbcrOp::Quantize => 2,
            YcbcrOp::Dequantize => 3,
        }
    }
}

/// Returns the `u32` code for a `luma`-weight standard: `0` for
/// [`Coefficients::Bt601`](prism_render_architecture::particle::ycbcr_bt709::Coefficients::Bt601),
/// `1` for
/// [`Coefficients::Bt709`](prism_render_architecture::particle::ycbcr_bt709::Coefficients::Bt709).
fn coeff_code(coeff: Coefficients) -> u32 {
    match coeff {
        Coefficients::Bt601 => 0,
        Coefficients::Bt709 => 1,
    }
}

/// Returns the `u32` code for a signal range: `0` for
/// [`Range::Full`](prism_render_architecture::particle::ycbcr_bt709::Range::Full),
/// `1` for
/// [`Range::Limited`](prism_render_architecture::particle::ycbcr_bt709::Range::Limited).
fn range_code(range: Range) -> u32 {
    match range {
        Range::Full => 0,
        Range::Limited => 1,
    }
}

/// One per-element colour query: the routine selected by `op` under the given
/// `coeff` and `range`.
///
/// The `inputs` triple carries the continuous channels the matrix transforms
/// and the quantiser consume (`RGB` for the forward transform, `YCbCr` for the
/// inverse transform and the quantiser); the `codes` triple carries the `8`-bit
/// codewords the dequantiser consumes. The field not read by the chosen `op` is
/// ignored.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YcbcrQuery {
    /// The routine this query evaluates.
    pub op: YcbcrOp,
    /// The `luma`-weight standard selecting the transform matrix.
    pub coeff: Coefficients,
    /// The signal range (full swing versus broadcast studio swing).
    pub range: Range,
    /// Continuous input channels (`RGB` or `YCbCr`) for the matrix transforms
    /// and the quantiser.
    pub inputs: [f32; 3],
    /// `8`-bit codewords for the dequantiser.
    pub codes: [u8; 3],
}

impl YcbcrQuery {
    /// Builds a forward-transform query (`RGB` -> `YCbCr`) from an `rgb` triple.
    #[must_use]
    pub const fn forward(rgb: Rgb, coeff: Coefficients, range: Range) -> YcbcrQuery {
        YcbcrQuery {
            op: YcbcrOp::RgbToYcbcr,
            coeff,
            range,
            inputs: [rgb.r, rgb.g, rgb.b],
            codes: [0, 0, 0],
        }
    }

    /// Builds an inverse-transform query (`YCbCr` -> `RGB`) from a `ycbcr`
    /// triple.
    #[must_use]
    pub const fn inverse(ycbcr: YCbCr, coeff: Coefficients, range: Range) -> YcbcrQuery {
        YcbcrQuery {
            op: YcbcrOp::YcbcrToRgb,
            coeff,
            range,
            inputs: [ycbcr.y, ycbcr.cb, ycbcr.cr],
            codes: [0, 0, 0],
        }
    }

    /// Builds a quantiser query from a `ycbcr` triple. The `coeff` is unused by
    /// the quantiser and recorded as
    /// [`Coefficients::Bt709`](prism_render_architecture::particle::ycbcr_bt709::Coefficients::Bt709).
    #[must_use]
    pub const fn quantize(ycbcr: YCbCr, range: Range) -> YcbcrQuery {
        YcbcrQuery {
            op: YcbcrOp::Quantize,
            coeff: Coefficients::Bt709,
            range,
            inputs: [ycbcr.y, ycbcr.cb, ycbcr.cr],
            codes: [0, 0, 0],
        }
    }

    /// Builds a dequantiser query from three `8`-bit `codes`. The `coeff` is
    /// unused and recorded as
    /// [`Coefficients::Bt709`](prism_render_architecture::particle::ycbcr_bt709::Coefficients::Bt709).
    #[must_use]
    pub const fn dequantize(codes: [u8; 3], range: Range) -> YcbcrQuery {
        YcbcrQuery {
            op: YcbcrOp::Dequantize,
            coeff: Coefficients::Bt709,
            range,
            inputs: [0.0, 0.0, 0.0],
            codes,
        }
    }
}

/// The resolved answer for one query.
///
/// `values` carries the continuous output of the matrix transforms and the
/// dequantiser (`YCbCr` or `RGB`); `bytes` carries the quantiser's codewords.
/// Only the field matching the query's [`YcbcrOp`] is populated; the other is
/// left zero.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YcbcrResult {
    /// Continuous output channels, matching the reference transform or
    /// dequantiser for continuous ops.
    pub values: [f32; 3],
    /// Quantised codewords, matching the reference
    /// [`quantize_8bit`](prism_render_architecture::particle::ycbcr_bt709::quantize_8bit)
    /// for the quantise op.
    pub bytes: [u8; 3],
}

/// Evaluates the `CPU` golden for one query, delegating field for field to the
/// reference
/// [`rgb_to_ycbcr`](prism_render_architecture::particle::ycbcr_bt709::rgb_to_ycbcr),
/// [`ycbcr_to_rgb`](prism_render_architecture::particle::ycbcr_bt709::ycbcr_to_rgb),
/// [`quantize_8bit`](prism_render_architecture::particle::ycbcr_bt709::quantize_8bit)
/// and
/// [`dequantize_8bit`](prism_render_architecture::particle::ycbcr_bt709::dequantize_8bit)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &YcbcrQuery) -> YcbcrResult {
    match query.op {
        YcbcrOp::RgbToYcbcr => {
            let rgb = Rgb::new(query.inputs[0], query.inputs[1], query.inputs[2]);
            let yc = rgb_to_ycbcr(&rgb, query.coeff, query.range);
            YcbcrResult {
                values: [yc.y, yc.cb, yc.cr],
                bytes: [0, 0, 0],
            }
        }
        YcbcrOp::YcbcrToRgb => {
            let yc = YCbCr::new(query.inputs[0], query.inputs[1], query.inputs[2]);
            let rgb = ycbcr_to_rgb(&yc, query.coeff, query.range);
            YcbcrResult {
                values: [rgb.r, rgb.g, rgb.b],
                bytes: [0, 0, 0],
            }
        }
        YcbcrOp::Quantize => {
            let yc = YCbCr::new(query.inputs[0], query.inputs[1], query.inputs[2]);
            YcbcrResult {
                values: [0.0, 0.0, 0.0],
                bytes: quantize_8bit(&yc, query.range),
            }
        }
        YcbcrOp::Dequantize => {
            let yc = dequantize_8bit(query.codes, query.range);
            YcbcrResult {
                values: [yc.y, yc.cb, yc.cr],
                bytes: [0, 0, 0],
            }
        }
    }
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots holding
/// `(inputs.xyz, op)`, `(codes.xyz, coeff)` and `(range, pad, pad, pad)` — `48`
/// bytes matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Continuous input channels.
    inputs: [f32; 3],
    /// Op code, packed in the fourth lane of the first slot.
    op: u32,
    /// Byte codewords for the dequantiser.
    codes: [u32; 3],
    /// `luma`-weight standard code, packed in the fourth lane of the second
    /// slot.
    coeff: u32,
    /// Signal-range code.
    range: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &YcbcrQuery) -> GpuQuery {
        GpuQuery {
            inputs: query.inputs,
            op: query.op.code(),
            codes: [
                u32::from(query.codes[0]),
                u32::from(query.codes[1]),
                u32::from(query.codes[2]),
            ],
            coeff: coeff_code(query.coeff),
            range: range_code(query.range),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(values.xyz, pad)` and `(bytes.xyz, pad)` — `32` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Continuous output channels.
    values: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Quantised codewords.
    bytes: [u32; 3],
    /// Padding word.
    pad1: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable `YCbCr` colour-transform compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
/// no third-party engine source or derived code.
pub struct GpuYcbcrBt709 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuYcbcrBt709 {
    /// Compiles the `YCbCr` colour kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuYcbcrBt709 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709"),
            source: ShaderSource::Wgsl(YCBCR_BT709_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuYcbcrBt709 {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`YcbcrResult`] per input,
    /// in order.
    ///
    /// The continuous channels match the reference transforms and dequantiser
    /// to within the tolerance documented on this module; the quantiser's
    /// codewords match exactly. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::ycbcr_bt709`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[YcbcrQuery]) -> Vec<YcbcrResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ycbcr_bt709_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ycbcr_bt709_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`YcbcrResult`]. The byte
/// lanes are integer codewords in `[0, 255]`, so the narrowing is exact;
/// [`u8::try_from`] guards the (unreachable) out-of-domain case.
fn decode_result(raw: &GpuResult) -> YcbcrResult {
    YcbcrResult {
        values: raw.values,
        bytes: [
            u8::try_from(raw.bytes[0]).unwrap_or(0),
            u8::try_from(raw.bytes[1]).unwrap_or(0),
            u8::try_from(raw.bytes[2]).unwrap_or(0),
        ],
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
