//! `wgpu` compute twin of the `OKLab` / `OKLCh` perceptual colour contract
//! ([`oklab_color`](prism_render_architecture::particle::oklab_color), particle
//! design colour pipeline).
//!
//! The `CPU` golden
//! [`oklab_color`](prism_render_architecture::particle::oklab_color) owns the
//! perceptual colour maths several particle stages share: the forward
//! [`linear_srgb_to_oklab`](prism_render_architecture::particle::oklab_color::linear_srgb_to_oklab)
//! transform (two Ottosson matrices with a cube-root cone-response step), the
//! inverse
//! [`oklab_to_linear_srgb`](prism_render_architecture::particle::oklab_color::oklab_to_linear_srgb),
//! the cylindrical
//! [`oklab_to_oklch`](prism_render_architecture::particle::oklab_color::oklab_to_oklch)
//! and its inverse
//! [`oklch_to_oklab`](prism_render_architecture::particle::oklab_color::oklch_to_oklab),
//! the chroma-plane
//! [`rotate_hue`](prism_render_architecture::particle::oklab_color::rotate_hue),
//! and the perceptual
//! [`lerp`](prism_render_architecture::particle::oklab_color::lerp).
//! [`GpuOklabColor`] is the on-device twin: one thread evaluates one query, so a
//! passing real-device parity test is direct evidence the ported kernel computes
//! the same colours and the same hue vectors the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries: the forward `OKLab` transform, the inverse linear
//! `sRGB` transform, the forward `OKLCh` conversion, the inverse `OKLab`
//! reconstruction, the hue rotation, and the `OKLab` interpolation. The golden
//! stores hue as a unit vector `(h_cos, h_sin)` rather than an angle, so the
//! whole contract is pure multiply-add-divide-`sqrt` with no inverse
//! trigonometry anywhere: there is no `atan2` field to leave to the host, and
//! every `OKLCh` lane is twinned on device.
//!
//! # Cube root
//!
//! The forward transform needs a real cube root. The workspace bans `cbrt` and
//! `powf` for cross-backend determinism, and `WGSL` additionally bans `exp2` /
//! `log2`, so the kernel copies the reference `cbrt_newton` exactly: it strips
//! the base-2 exponent with a `bitcast` to `u32`, reconstructs the mantissa into
//! `[1, 2)`, refines its cube root with seven Newton steps
//! `y <- (2*y + m/(y*y)) / 3`, folds the residual `2^0..2` factor back with the
//! real cube roots of `2` and `4`, and re-applies the `2^q` factor by repeated
//! multiply. Only `abs`, `+ - * /`, comparisons and `bitcast` are used.
//!
//! # Correctness model
//!
//! Every output is a continuous `f32`, so `CPU` and `GPU` are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate, and the
//! Newton cube root accumulates rounding differently. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every lane, tight enough to catch a genuinely wrong
//! port (a dropped term, a swapped matrix row, a wrong Newton update) yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A near-achromatic `OKLab` colour (chroma at or below [`CHROMA_FLOOR`]) has an
//! undefined hue; the kernel canonicalises it to the hue vector `(1, 0)` exactly
//! as the reference does, instead of dividing by a near-zero chroma. A zero or
//! sub-normal cube-root argument returns `0` without any division. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `+ - * /`, integer and bitwise arithmetic, and `bitcast` — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! only loops are the fixed seven-step Newton refinement and the bounded
//! power-of-two rescale, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::oklab_color`；无第三方引擎源码或衍生代码。
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

/// Chroma magnitudes at or below this threshold are treated as achromatic, so a
/// near-grey colour maps to the canonical hue vector `(1, 0)` instead of an
/// ill-conditioned division; mirrors the reference `CHROMA_FLOOR`.
pub const CHROMA_FLOOR: f32 = 1.0e-6;

/// The portable core-`WGSL` `OKLab` / `OKLCh` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`oklab_color`](prism_render_architecture::particle::oklab_color) function by
/// function; see the module documentation for the algorithm.
const OKLAB_COLOR_WGSL: &str = r#"
// OKLab / OKLCh twin: one thread per query reproduces the forward OKLab
// transform, the inverse linear-sRGB transform, the forward OKLCh conversion,
// the inverse OKLab reconstruction, the hue rotation, and the OKLab lerp. It
// mirrors the CPU golden `particle::oklab_color` function for function, uses
// only the portable core-WGSL subset (abs/sqrt, + - * /, integer/bitwise math
// and bitcast), needs no transcendental call and takes no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12. The only loops are the fixed
// seven-step Newton cube root and the bounded power-of-two rescale, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::oklab_color；无第三方
// 引擎源码或衍生代码。

// Chroma below which the hue is undefined and canonicalised to (1, 0); matches
// the reference `CHROMA_FLOOR`.
const CHROMA_FLOOR: f32 = 1.0e-6;
// Smallest positive normal f32; a cube-root argument below this returns 0.
const MIN_POSITIVE: f32 = 1.17549435e-38;
// Real cube root of 2, folding a residual 2^1 factor back into the Newton root.
const CBRT_2: f32 = 1.2599211;
// Real cube root of 4, folding a residual 2^2 factor back into the Newton root.
const CBRT_4: f32 = 1.5874011;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Linear sRGB triple fed to the forward OKLab transform; a pad lane follows.
    lin: vec3<f32>,
    pad_lin: f32,
    // OKLab triple (l, a, b) fed to the inverse linear transform and the OKLCh
    // conversion; a pad lane follows.
    lab: vec3<f32>,
    pad_lab: f32,
    // OKLCh quad (l, c, h_cos, h_sin) fed to the inverse OKLab reconstruction and
    // the hue rotation.
    lch: vec4<f32>,
    // First OKLab endpoint of the lerp; a pad lane follows.
    lab_a: vec3<f32>,
    pad_a: f32,
    // Second OKLab endpoint of the lerp; a pad lane follows.
    lab_b: vec3<f32>,
    pad_b: f32,
    // Hue-rotation cosine and sine plus the lerp parameter, packed as
    // (cos_delta, sin_delta, t, pad).
    delta: vec4<f32>,
}

struct Result {
    // linear_srgb_to_oklab result; a pad lane follows.
    oklab_from_linear: vec3<f32>,
    pad0: f32,
    // oklab_to_linear_srgb result; a pad lane follows.
    linear_from_oklab: vec3<f32>,
    pad1: f32,
    // oklab_to_oklch result (l, c, h_cos, h_sin).
    oklch_from_oklab: vec4<f32>,
    // oklch_to_oklab result; a pad lane follows.
    oklab_from_oklch: vec3<f32>,
    pad2: f32,
    // rotate_hue result (l, c, h_cos, h_sin).
    rotated: vec4<f32>,
    // lerp result; a pad lane follows.
    lerped: vec3<f32>,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Real cube root to f32 precision via Newton's method; mirrors the reference
// `cbrt_newton`. Strips the base-2 exponent with a bitcast, refines the mantissa
// cube root, then re-applies the exponent factor split into thirds.
fn cbrt_newton(x: f32) -> f32 {
    let magnitude = abs(x);
    if (magnitude < MIN_POSITIVE) {
        return 0.0;
    }
    let bits = bitcast<u32>(magnitude);
    // Unbiased base-2 exponent of `magnitude` (guaranteed normal here).
    let exponent = i32((bits >> 23u) & 0xffu) - 127;
    // Mantissa reconstructed into the range [1, 2).
    let mantissa = bitcast<f32>((bits & 0x007fffffu) | 0x3f800000u);
    // Euclidean split exponent = 3*q + r with r in {0, 1, 2}; WGSL integer
    // division truncates toward zero, so fix up a negative remainder by hand.
    var q = exponent / 3;
    var r = exponent % 3;
    if (r < 0) {
        r = r + 3;
        q = q - 1;
    }
    // Cube root of the mantissa via Newton from y = 1; seven quadratically
    // converging steps reach full f32 precision for m in [1, 2).
    var y = 1.0;
    var iter = 0;
    loop {
        if (iter >= 7) {
            break;
        }
        let y2 = y * y;
        y = (2.0 * y + mantissa / y2) / 3.0;
        iter = iter + 1;
    }
    // Re-apply the residual 2^r factor.
    var r_scale = 1.0;
    if (r == 1) {
        r_scale = CBRT_2;
    } else if (r == 2) {
        r_scale = CBRT_4;
    }
    // Re-apply the 2^q factor by scaling by two the matching number of times.
    var two_pow_q = 1.0;
    if (q >= 0) {
        var n = 0;
        loop {
            if (n >= q) {
                break;
            }
            two_pow_q = two_pow_q * 2.0;
            n = n + 1;
        }
    } else {
        var n = 0;
        loop {
            if (n >= -q) {
                break;
            }
            two_pow_q = two_pow_q * 0.5;
            n = n + 1;
        }
    }
    let root = y * r_scale * two_pow_q;
    if (x < 0.0) {
        return -root;
    }
    return root;
}

// linear sRGB -> OKLab via Ottosson's M1 (to LMS), per-component cube root, then
// M2 (to l, a, b).
fn linear_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.41222147 * c.x + 0.53633254 * c.y + 0.051445995 * c.z;
    let m = 0.2119035 * c.x + 0.6806995 * c.y + 0.10739696 * c.z;
    let s = 0.08830246 * c.x + 0.28171884 * c.y + 0.6299787 * c.z;
    let l_ = cbrt_newton(l);
    let m_ = cbrt_newton(m);
    let s_ = cbrt_newton(s);
    return vec3<f32>(
        0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_,
        1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_,
        0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_,
    );
}

// OKLab -> linear sRGB via inverse M2, cube each term, inverse M1.
fn oklab_to_linear(c: vec3<f32>) -> vec3<f32> {
    let l_ = c.x + 0.39633778 * c.y + 0.21580376 * c.z;
    let m_ = c.x - 0.105561346 * c.y - 0.06385417 * c.z;
    let s_ = c.x - 0.08948418 * c.y - 1.2914855 * c.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
        4.0767417 * l - 3.3077116 * m + 0.23096994 * s,
        -1.268438 * l + 2.6097574 * m - 0.34131938 * s,
        -0.0041960864 * l - 0.7034186 * m + 1.7076147 * s,
    );
}

// OKLab (l, a, b) -> OKLCh (l, c, h_cos, h_sin). An achromatic colour folds to
// the canonical hue vector (1, 0).
fn oklab_to_oklch(c: vec3<f32>) -> vec4<f32> {
    let chroma = sqrt(c.y * c.y + c.z * c.z);
    if (chroma > CHROMA_FLOOR) {
        return vec4<f32>(c.x, chroma, c.y / chroma, c.z / chroma);
    }
    return vec4<f32>(c.x, 0.0, 1.0, 0.0);
}

// OKLCh (l, c, h_cos, h_sin) -> OKLab (l, a, b): project chroma onto the hue
// unit vector.
fn oklch_to_oklab(c: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(c.x, c.y * c.z, c.y * c.w);
}

// Rotate the stored hue unit vector in the chroma plane by the supplied cosine
// and sine, leaving lightness and chroma untouched.
fn rotate_hue(c: vec4<f32>, cos_delta: f32, sin_delta: f32) -> vec4<f32> {
    return vec4<f32>(
        c.x,
        c.y,
        c.z * cos_delta - c.w * sin_delta,
        c.w * cos_delta + c.z * sin_delta,
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let oklab_from_linear = linear_to_oklab(q.lin);
    let linear_from_oklab = oklab_to_linear(q.lab);
    let oklch_from_oklab = oklab_to_oklch(q.lab);
    let oklab_from_oklch = oklch_to_oklab(q.lch);
    let rotated = rotate_hue(q.lch, q.delta.x, q.delta.y);
    // Perceptual lerp: a + (b - a) * t, component-wise.
    let lerped = q.lab_a + (q.lab_b - q.lab_a) * q.delta.z;

    var out: Result;
    out.oklab_from_linear = oklab_from_linear;
    out.pad0 = 0.0;
    out.linear_from_oklab = linear_from_oklab;
    out.pad1 = 0.0;
    out.oklch_from_oklab = oklch_from_oklab;
    out.oklab_from_oklch = oklab_from_oklch;
    out.pad2 = 0.0;
    out.rotated = rotated;
    out.lerped = lerped;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`OKLAB_COLOR_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear `sRGB` triple for the forward `OKLab` transform.
    lin: [f32; 3],
    /// Pad lane after `lin`.
    pad_lin: f32,
    /// `OKLab` triple for the inverse transform and the `OKLCh` conversion.
    lab: [f32; 3],
    /// Pad lane after `lab`.
    pad_lab: f32,
    /// `OKLCh` quad for the inverse reconstruction and the hue rotation.
    lch: [f32; 4],
    /// First `OKLab` endpoint of the lerp.
    lab_a: [f32; 3],
    /// Pad lane after `lab_a`.
    pad_a: f32,
    /// Second `OKLab` endpoint of the lerp.
    lab_b: [f32; 3],
    /// Pad lane after `lab_b`.
    pad_b: f32,
    /// Hue-rotation cosine and sine plus the lerp parameter, as
    /// `(cos_delta, sin_delta, t, pad)`.
    delta: [f32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `linear_srgb_to_oklab` result.
    oklab_from_linear: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// `oklab_to_linear_srgb` result.
    linear_from_oklab: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// `oklab_to_oklch` result as `(l, c, h_cos, h_sin)`.
    oklch_from_oklab: [f32; 4],
    /// `oklch_to_oklab` result.
    oklab_from_oklch: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// `rotate_hue` result as `(l, c, h_cos, h_sin)`.
    rotated: [f32; 4],
    /// `lerp` result.
    lerped: [f32; 3],
    /// Padding lane.
    pad3: f32,
}

/// One query for the `OKLab` twin: a linear `sRGB` colour, an `OKLab` colour, an
/// `OKLCh` colour, two `OKLab` lerp endpoints, a hue-rotation cosine/sine pair,
/// and a lerp parameter.
///
/// The six twinned operations are independent, so a single query exercises every
/// one at once. The hue rotation takes its angle as a precomputed
/// `(cos_delta, sin_delta)` unit vector so the kernel never calls a
/// trigonometric function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OklabColorQuery {
    /// Linear `sRGB` triple `(r, g, b)` for `linear_srgb_to_oklab`.
    pub lin: [f32; 3],
    /// `OKLab` triple `(l, a, b)` for `oklab_to_linear_srgb` and `oklab_to_oklch`.
    pub lab: [f32; 3],
    /// `OKLCh` quad `(l, c, h_cos, h_sin)` for `oklch_to_oklab` and `rotate_hue`.
    pub lch: [f32; 4],
    /// First `OKLab` endpoint `(l, a, b)` for `lerp`.
    pub lab_a: [f32; 3],
    /// Second `OKLab` endpoint `(l, a, b)` for `lerp`.
    pub lab_b: [f32; 3],
    /// Cosine of the hue-rotation angle fed to `rotate_hue`.
    pub cos_delta: f32,
    /// Sine of the hue-rotation angle fed to `rotate_hue`.
    pub sin_delta: f32,
    /// Interpolation parameter fed to `lerp`.
    pub t: f32,
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OklabColorResult {
    /// `OKLab` colour `(l, a, b)` from `linear_srgb_to_oklab`.
    pub oklab_from_linear: [f32; 3],
    /// Linear `sRGB` colour `(r, g, b)` from `oklab_to_linear_srgb`.
    pub linear_from_oklab: [f32; 3],
    /// `OKLCh` colour `(l, c, h_cos, h_sin)` from `oklab_to_oklch`.
    pub oklch_from_oklab: [f32; 4],
    /// `OKLab` colour `(l, a, b)` from `oklch_to_oklab`.
    pub oklab_from_oklch: [f32; 3],
    /// Rotated `OKLCh` colour `(l, c, h_cos, h_sin)` from `rotate_hue`.
    pub rotated: [f32; 4],
    /// Interpolated `OKLab` colour `(l, a, b)` from `lerp`.
    pub lerped: [f32; 3],
}

/// Encodes one [`OklabColorQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &OklabColorQuery) -> GpuQuery {
    GpuQuery {
        lin: q.lin,
        pad_lin: 0.0,
        lab: q.lab,
        pad_lab: 0.0,
        lch: q.lch,
        lab_a: q.lab_a,
        pad_a: 0.0,
        lab_b: q.lab_b,
        pad_b: 0.0,
        delta: [q.cos_delta, q.sin_delta, q.t, 0.0],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`OklabColorResult`].
fn decode_result(raw: &GpuResult) -> OklabColorResult {
    OklabColorResult {
        oklab_from_linear: raw.oklab_from_linear,
        linear_from_oklab: raw.linear_from_oklab,
        oklch_from_oklab: raw.oklch_from_oklab,
        oklab_from_oklch: raw.oklab_from_oklch,
        rotated: raw.rotated,
        lerped: raw.lerped,
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

/// A compiled, reusable `OKLab` / `OKLCh` compute pipeline, twinning the `CPU`
/// golden
/// [`oklab_color`](prism_render_architecture::particle::oklab_color).
pub struct GpuOklabColor {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuOklabColor {
    /// Compiles the `OKLab` / `OKLCh` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOklabColor {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_oklab_color"),
            source: ShaderSource::Wgsl(OKLAB_COLOR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_oklab_color_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_oklab_color_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_oklab_color_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOklabColor {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`OklabColorResult`]
    /// per input, in order.
    ///
    /// Every output lane matches the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[OklabColorQuery]) -> Vec<OklabColorResult> {
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
            label: Some("prism_volumetric_oklab_color_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_oklab_color_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_oklab_color_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_oklab_color_bind_group"),
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
            label: Some("prism_volumetric_oklab_color_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_oklab_color_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_oklab_color_pass"),
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
