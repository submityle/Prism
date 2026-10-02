//! `wgpu` compute twin of the hue/saturation/value colour-adjustment contract
//! ([`hue_shift`](prism_render_architecture::particle::hue_shift), particle
//! design §16, §30).
//!
//! The `CPU` golden
//! [`hue_shift`](prism_render_architecture::particle::hue_shift) owns the
//! classic *cylindrical* colour models a `VFX` artist reaches for: `HSV`
//! (hue/saturation/value) and `HSL` (hue/saturation/lightness). It converts
//! `RGB` to and from both
//! ([`rgb_to_hsv`](prism_render_architecture::particle::hue_shift::rgb_to_hsv),
//! [`hsv_to_rgb`](prism_render_architecture::particle::hue_shift::hsv_to_rgb),
//! [`rgb_to_hsl`](prism_render_architecture::particle::hue_shift::rgb_to_hsl),
//! [`hsl_to_rgb`](prism_render_architecture::particle::hue_shift::hsl_to_rgb)),
//! rotates the hue
//! ([`shift_hue`](prism_render_architecture::particle::hue_shift::shift_hue)),
//! scales saturation, value and lightness
//! ([`adjust_saturation`](prism_render_architecture::particle::hue_shift::adjust_saturation),
//! [`adjust_value`](prism_render_architecture::particle::hue_shift::adjust_value),
//! [`adjust_lightness`](prism_render_architecture::particle::hue_shift::adjust_lightness)),
//! and rotates hue in the `YIQ` chroma plane
//! ([`rotate_hue_yiq`](prism_render_architecture::particle::hue_shift::rotate_hue_yiq)).
//! [`GpuHueShift`] is the on-device twin: one thread per colour reproduces every
//! transform branch for branch, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same six-sextant algebra and
//! classifies the same achromatic case the reference does, not merely that the
//! shader compiles.
//!
//! # No trigonometry, no transcendentals
//!
//! The hue of a colour is usually an angle on a wheel, but neither the
//! reference nor this twin touches `sin`/`cos`/`atan`. Both use the standard
//! *six-sextant* piecewise-linear parameterisation: the hue is a value in
//! `[0, 6)` derived purely from which of the red/green/blue channels is largest
//! and the linear ratio of the other two against the chroma (max minus min).
//! The only floating-point intrinsics used are `floor`, `abs`, `min` and `max`.
//! The [`rotate_hue_yiq`](prism_render_architecture::particle::hue_shift::rotate_hue_yiq)
//! helper rotates hue in the `YIQ` chroma plane instead; because a plane
//! rotation genuinely needs a cosine and sine, the host pre-multiplies and
//! *supplies* the `(cos_d, sin_d)` pair, and the kernel merely multiplies the
//! matrices — it never evaluates a trig function itself.
//!
//! # What is twinned
//!
//! Every per-colour answer the reference computes is reproduced for a batch of
//! independent colours packed into one query: the `HSV` and `HSL` images of an
//! `RGB` input, the `RGB` image of an `HSV` input and of an `HSL` input, the
//! hue-shifted `RGB`, the saturation-, value- and lightness-scaled `RGB`, and
//! the `YIQ`-plane-rotated `RGB`. The host-only `std430` packing helpers the
//! reference exposes (`to_std430`, `gpu_storage_bytes`) are not kernel math and
//! are left to the `CPU` reference; only the arithmetic transforms are twinned
//! on-device.
//!
//! # Correctness model
//!
//! Each transform threads through a fixed, non-reorderable sequence of
//! multiplies, adds, divides, `floor`, `abs`, `min` and `max` (no
//! transcendental, no `sqrt`), so `CPU` and `GPU` evaluate the same closed form
//! in the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every
//! continuous quantity, tight enough to catch a genuinely wrong port (a dropped
//! term, a swapped coefficient, a transposed branch) yet loose enough to admit
//! legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The reference guards three divisions, and the kernel mirrors all three
//! exactly with a `<= 0.0` test rather than an `f32` `==`. An achromatic colour
//! (chroma `delta <= 0`) has no defined hue and returns hue `0`; a non-positive
//! `HSV` value `max` yields saturation `0`; a non-positive `HSL` denominator
//! yields saturation `0`. Because the three channels are packed verbatim, both
//! devices take the same max/min comparison branch in the six-sextant hue, so
//! the piecewise selection cannot diverge. An empty query batch short-circuits
//! on the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `abs`,
//! `min`, `max`, `clamp`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt`
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. There is no loop: each thread performs a fixed, bounded sequence
//! of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
//! standard six-sextant `HSV`/`HSL` cylindrical colour models and the `NTSC`
//! `YIQ` chroma rotation plus `wgpu` compute dispatch; no third-party engine
//! source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::hue_shift::{
    adjust_lightness, adjust_saturation, adjust_value, hsl_to_rgb, hsv_to_rgb, rgb_to_hsl,
    rgb_to_hsv, rotate_hue_yiq, shift_hue, Hsl, Hsv, Rgb,
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

/// The portable core-`WGSL` hue/saturation/value kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`hue_shift`](prism_render_architecture::particle::hue_shift) branch for
/// branch; see the module documentation for the algorithm.
const HUE_SHIFT_WGSL: &str = r#"
// Hue/saturation/value colour twin: one thread per colour reproduces the RGB
// <-> HSV and RGB <-> HSL six-sextant conversions, the hue shift, the
// saturation/value/lightness scales and the NTSC YIQ chroma-plane rotation. It
// mirrors the CPU golden particle::hue_shift branch for branch, uses only the
// portable core-WGSL subset (floor, abs, min, max, clamp and + - * / plus
// unsigned index math), needs no sqrt and no transcendental call and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. There is
// no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::hue_shift; no
// third-party engine source or derived code.

struct Params {
    // Number of colours in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // RGB input shared by rgb_to_hsv, rgb_to_hsl, shift_hue, the three scales
    // and rotate_hue_yiq; a pad lane follows the vec3.
    rgb: vec3<f32>,
    pad0: f32,
    // HSV input for hsv_to_rgb.
    hsv: vec3<f32>,
    pad1: f32,
    // HSL input for hsl_to_rgb.
    hsl: vec3<f32>,
    pad2: f32,
    // Scalar parameters: x = delta_sextants, y = factor, z = cos_d, w = sin_d.
    knobs: vec4<f32>,
}

struct Result {
    // HSV image of the RGB input.
    to_hsv: vec3<f32>,
    pad0: f32,
    // HSL image of the RGB input.
    to_hsl: vec3<f32>,
    pad1: f32,
    // RGB image of the HSV input.
    from_hsv: vec3<f32>,
    pad2: f32,
    // RGB image of the HSL input.
    from_hsl: vec3<f32>,
    pad3: f32,
    // Hue-shifted RGB.
    shifted: vec3<f32>,
    pad4: f32,
    // Saturation-scaled RGB.
    saturated: vec3<f32>,
    pad5: f32,
    // Value-scaled RGB.
    valued: vec3<f32>,
    pad6: f32,
    // Lightness-scaled RGB.
    lightened: vec3<f32>,
    pad7: f32,
    // YIQ-chroma-plane-rotated RGB.
    yiq: vec3<f32>,
    pad8: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Wraps a raw sextant hue into the canonical [0, 6) range using only floor, so
// a negative or overflowing hue folds back onto the wheel. Mirrors the
// reference `wrap_sextant`.
fn wrap_sextant(h: f32) -> f32 {
    return h - 6.0 * floor(h / 6.0);
}

// Derives the six-sextant hue in [0, 6) from the channels and chroma (delta =
// max minus min). An achromatic colour (delta <= 0) has no defined hue and
// returns 0. Pure comparisons and ratios; no angle is ever formed. Mirrors the
// reference `sextant_hue`.
fn sextant_hue(r: f32, g: f32, b: f32, delta: f32) -> f32 {
    if (delta <= 0.0) {
        return 0.0;
    }
    var raw: f32;
    if (r >= g && r >= b) {
        raw = (g - b) / delta;
    } else if (g >= b) {
        raw = 2.0 + (b - r) / delta;
    } else {
        raw = 4.0 + (r - g) / delta;
    }
    if (raw < 0.0) {
        return raw + 6.0;
    }
    return raw;
}

// Converts a linear RGB colour into HSV (h, s, v) using the six-sextant scheme,
// mirroring the reference `rgb_to_hsv`.
fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.x, max(c.y, c.z));
    let mn = min(c.x, min(c.y, c.z));
    let delta = mx - mn;
    var s: f32;
    if (mx <= 0.0) {
        s = 0.0;
    } else {
        s = delta / mx;
    }
    let h = sextant_hue(c.x, c.y, c.z, delta);
    return vec3<f32>(h, s, mx);
}

// Converts an HSV colour (c.x, c.y, c.z) back into linear RGB, clamping s and v
// and wrapping the hue before the piecewise fill, mirroring the reference
// `hsv_to_rgb`.
fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let h = wrap_sextant(c.x);
    let s = clamp(c.y, 0.0, 1.0);
    let v = clamp(c.z, 0.0, 1.0);
    let f = h - floor(h);
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    if (h < 1.0) {
        return vec3<f32>(v, t, p);
    }
    if (h < 2.0) {
        return vec3<f32>(q, v, p);
    }
    if (h < 3.0) {
        return vec3<f32>(p, v, t);
    }
    if (h < 4.0) {
        return vec3<f32>(p, q, v);
    }
    if (h < 5.0) {
        return vec3<f32>(t, p, v);
    }
    return vec3<f32>(v, p, q);
}

// Converts a linear RGB colour into HSL (h, s, l) using the six-sextant scheme,
// mirroring the reference `rgb_to_hsl`.
fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.x, max(c.y, c.z));
    let mn = min(c.x, min(c.y, c.z));
    let delta = mx - mn;
    let l = (mx + mn) * 0.5;
    let denom = 1.0 - abs(2.0 * l - 1.0);
    var s: f32;
    if (denom <= 0.0) {
        s = 0.0;
    } else {
        s = delta / denom;
    }
    let h = sextant_hue(c.x, c.y, c.z, delta);
    return vec3<f32>(h, s, l);
}

// Converts an HSL colour (c.x, c.y, c.z) back into linear RGB, clamping s and l
// and wrapping the hue before the piecewise fill, mirroring the reference
// `hsl_to_rgb`.
fn hsl_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let h = wrap_sextant(c.x);
    let s = clamp(c.y, 0.0, 1.0);
    let l = clamp(c.z, 0.0, 1.0);
    let chroma = (1.0 - abs(2.0 * l - 1.0)) * s;
    let hmod2 = h - 2.0 * floor(h * 0.5);
    let x = chroma * (1.0 - abs(hmod2 - 1.0));
    let m = l - chroma * 0.5;
    var base: vec3<f32>;
    if (h < 1.0) {
        base = vec3<f32>(chroma, x, 0.0);
    } else if (h < 2.0) {
        base = vec3<f32>(x, chroma, 0.0);
    } else if (h < 3.0) {
        base = vec3<f32>(0.0, chroma, x);
    } else if (h < 4.0) {
        base = vec3<f32>(0.0, x, chroma);
    } else if (h < 5.0) {
        base = vec3<f32>(x, 0.0, chroma);
    } else {
        base = vec3<f32>(chroma, 0.0, x);
    }
    return base + vec3<f32>(m, m, m);
}

// Rotates the hue of a colour by delta_sextants through the HSV model, wrapping
// the result back onto [0, 6), mirroring the reference `shift_hue`.
fn shift_hue(c: vec3<f32>, delta_sextants: f32) -> vec3<f32> {
    var hsv = rgb_to_hsv(c);
    hsv.x = wrap_sextant(hsv.x + delta_sextants);
    return hsv_to_rgb(hsv);
}

// Scales HSV saturation by factor (clamped to [0, 1]), mirroring the reference
// `adjust_saturation`.
fn adjust_saturation(c: vec3<f32>, factor: f32) -> vec3<f32> {
    var hsv = rgb_to_hsv(c);
    hsv.y = clamp(hsv.y * factor, 0.0, 1.0);
    return hsv_to_rgb(hsv);
}

// Scales HSV value by factor (clamped to [0, 1]), mirroring the reference
// `adjust_value`.
fn adjust_value(c: vec3<f32>, factor: f32) -> vec3<f32> {
    var hsv = rgb_to_hsv(c);
    hsv.z = clamp(hsv.z * factor, 0.0, 1.0);
    return hsv_to_rgb(hsv);
}

// Scales HSL lightness by factor (clamped to [0, 1]), mirroring the reference
// `adjust_lightness`.
fn adjust_lightness(c: vec3<f32>, factor: f32) -> vec3<f32> {
    var hsl = rgb_to_hsl(c);
    hsl.z = clamp(hsl.z * factor, 0.0, 1.0);
    return hsl_to_rgb(hsl);
}

// Rotates hue in the YIQ chroma plane by the host-supplied (cos_d, sin_d) pair
// and clamps back to [0, 1], mirroring the reference `rotate_hue_yiq`. The
// kernel never evaluates a trig function: the cosine and sine arrive as inputs.
fn rotate_hue_yiq(c: vec3<f32>, cos_d: f32, sin_d: f32) -> vec3<f32> {
    let y = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let i = 0.596 * c.x - 0.274 * c.y - 0.322 * c.z;
    let q = 0.211 * c.x - 0.523 * c.y + 0.312 * c.z;
    let i_rot = i * cos_d - q * sin_d;
    let q_rot = i * sin_d + q * cos_d;
    let r = y + 0.956 * i_rot + 0.621 * q_rot;
    let g = y - 0.272 * i_rot - 0.647 * q_rot;
    let b = y - 1.106 * i_rot + 1.703 * q_rot;
    return vec3<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.to_hsv = rgb_to_hsv(q.rgb);
    out.to_hsl = rgb_to_hsl(q.rgb);
    out.from_hsv = hsv_to_rgb(q.hsv);
    out.from_hsl = hsl_to_rgb(q.hsl);
    out.shifted = shift_hue(q.rgb, q.knobs.x);
    out.saturated = adjust_saturation(q.rgb, q.knobs.y);
    out.valued = adjust_value(q.rgb, q.knobs.y);
    out.lightened = adjust_lightness(q.rgb, q.knobs.y);
    out.yiq = rotate_hue_yiq(q.rgb, q.knobs.z, q.knobs.w);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    out.pad3 = 0.0;
    out.pad4 = 0.0;
    out.pad5 = 0.0;
    out.pad6 = 0.0;
    out.pad7 = 0.0;
    out.pad8 = 0.0;
    results[idx] = out;
}
"#;

/// One hue-adjustment query bundling every input the reference transforms
/// consume for a single colour: the shared linear `RGB` input, an `HSV` input,
/// an `HSL` input, and the four scalar knobs (`delta_sextants`, `factor` and the
/// host-pre-multiplied `cos_d`/`sin_d` pair).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HueShiftQuery {
    /// Shared linear `RGB` input driving the `RGB` -> `HSV`/`HSL` conversions,
    /// the hue shift, the three scales and the `YIQ` rotation.
    pub rgb: Rgb,
    /// `HSV` input for the `HSV` -> `RGB` conversion.
    pub hsv: Hsv,
    /// `HSL` input for the `HSL` -> `RGB` conversion.
    pub hsl: Hsl,
    /// Hue rotation in sextants for `shift_hue`.
    pub delta_sextants: f32,
    /// Scale factor shared by the saturation, value and lightness adjustments.
    pub factor: f32,
    /// Host-pre-multiplied cosine for the `YIQ` chroma-plane rotation.
    pub cos_d: f32,
    /// Host-pre-multiplied sine for the `YIQ` chroma-plane rotation.
    pub sin_d: f32,
}

impl HueShiftQuery {
    /// Builds a query from the three colour inputs and the four scalar knobs.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        rgb: Rgb,
        hsv: Hsv,
        hsl: Hsl,
        delta_sextants: f32,
        factor: f32,
        cos_d: f32,
        sin_d: f32,
    ) -> HueShiftQuery {
        HueShiftQuery {
            rgb,
            hsv,
            hsl,
            delta_sextants,
            factor,
            cos_d,
            sin_d,
        }
    }
}

/// The resolved answer for one colour, mirroring every transform the reference
/// exposes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HueShiftResult {
    /// `HSV` image of the `RGB` input, matching
    /// [`rgb_to_hsv`](prism_render_architecture::particle::hue_shift::rgb_to_hsv).
    pub to_hsv: Hsv,
    /// `HSL` image of the `RGB` input, matching
    /// [`rgb_to_hsl`](prism_render_architecture::particle::hue_shift::rgb_to_hsl).
    pub to_hsl: Hsl,
    /// `RGB` image of the `HSV` input, matching
    /// [`hsv_to_rgb`](prism_render_architecture::particle::hue_shift::hsv_to_rgb).
    pub from_hsv: Rgb,
    /// `RGB` image of the `HSL` input, matching
    /// [`hsl_to_rgb`](prism_render_architecture::particle::hue_shift::hsl_to_rgb).
    pub from_hsl: Rgb,
    /// Hue-shifted `RGB`, matching
    /// [`shift_hue`](prism_render_architecture::particle::hue_shift::shift_hue).
    pub shifted: Rgb,
    /// Saturation-scaled `RGB`, matching
    /// [`adjust_saturation`](prism_render_architecture::particle::hue_shift::adjust_saturation).
    pub saturated: Rgb,
    /// Value-scaled `RGB`, matching
    /// [`adjust_value`](prism_render_architecture::particle::hue_shift::adjust_value).
    pub valued: Rgb,
    /// Lightness-scaled `RGB`, matching
    /// [`adjust_lightness`](prism_render_architecture::particle::hue_shift::adjust_lightness).
    pub lightened: Rgb,
    /// `YIQ`-chroma-plane-rotated `RGB`, matching
    /// [`rotate_hue_yiq`](prism_render_architecture::particle::hue_shift::rotate_hue_yiq).
    pub yiq_rotated: Rgb,
}

/// Evaluates the `CPU` golden for one query, producing every transform the
/// on-device twin reproduces.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &HueShiftQuery) -> HueShiftResult {
    HueShiftResult {
        to_hsv: rgb_to_hsv(query.rgb),
        to_hsl: rgb_to_hsl(query.rgb),
        from_hsv: hsv_to_rgb(query.hsv),
        from_hsl: hsl_to_rgb(query.hsl),
        shifted: shift_hue(query.rgb, query.delta_sextants),
        saturated: adjust_saturation(query.rgb, query.factor),
        valued: adjust_value(query.rgb, query.factor),
        lightened: adjust_lightness(query.rgb, query.factor),
        yiq_rotated: rotate_hue_yiq(query.rgb, query.cos_d, query.sin_d),
    }
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots carrying a
/// `vec3` triple plus a padding lane, then one `vec4` of scalar knobs, exactly
/// as the `WGSL` `Query` struct reads it — `64` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear `RGB` input.
    rgb: [f32; 3],
    /// Padding lane after the `RGB` input.
    pad0: f32,
    /// `HSV` input.
    hsv: [f32; 3],
    /// Padding lane after the `HSV` input.
    pad1: f32,
    /// `HSL` input.
    hsl: [f32; 3],
    /// Padding lane after the `HSL` input.
    pad2: f32,
    /// Scalar knobs: `delta_sextants`, `factor`, `cos_d`, `sin_d`.
    knobs: [f32; 4],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &HueShiftQuery) -> GpuQuery {
        GpuQuery {
            rgb: [query.rgb.r, query.rgb.g, query.rgb.b],
            pad0: 0.0,
            hsv: [query.hsv.h, query.hsv.s, query.hsv.v],
            pad1: 0.0,
            hsl: [query.hsl.h, query.hsl.s, query.hsl.l],
            pad2: 0.0,
            knobs: [query.delta_sextants, query.factor, query.cos_d, query.sin_d],
        }
    }
}

/// `repr(C)` `std430` layout of one result: nine `vec4` slots, each a `vec3`
/// triple with a padding lane matching the `WGSL` `Result` struct — `144`
/// bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `HSV` image of the `RGB` input.
    to_hsv: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// `HSL` image of the `RGB` input.
    to_hsl: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// `RGB` image of the `HSV` input.
    from_hsv: [f32; 3],
    /// Padding lane.
    pad2: f32,
    /// `RGB` image of the `HSL` input.
    from_hsl: [f32; 3],
    /// Padding lane.
    pad3: f32,
    /// Hue-shifted `RGB`.
    shifted: [f32; 3],
    /// Padding lane.
    pad4: f32,
    /// Saturation-scaled `RGB`.
    saturated: [f32; 3],
    /// Padding lane.
    pad5: f32,
    /// Value-scaled `RGB`.
    valued: [f32; 3],
    /// Padding lane.
    pad6: f32,
    /// Lightness-scaled `RGB`.
    lightened: [f32; 3],
    /// Padding lane.
    pad7: f32,
    /// `YIQ`-chroma-plane-rotated `RGB`.
    yiq: [f32; 3],
    /// Padding lane.
    pad8: f32,
}

/// Uniform parameters for one dispatch: the colour count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of colours in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable hue/saturation/value compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
/// no third-party engine source or derived code.
pub struct GpuHueShift {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHueShift {
    /// Compiles the hue-adjustment kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHueShift {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hue_shift"),
            source: ShaderSource::Wgsl(HUE_SHIFT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hue_shift_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hue_shift_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hue_shift_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHueShift {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every colour on-device and returns one [`HueShiftResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference transforms to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::hue_shift`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[HueShiftQuery]) -> Vec<HueShiftResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hue_shift_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hue_shift_output"),
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
            label: Some("prism_volumetric_hue_shift_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hue_shift_bind_group"),
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
            label: Some("prism_volumetric_hue_shift_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hue_shift_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hue_shift_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per colour, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`HueShiftResult`].
fn decode_result(raw: &GpuResult) -> HueShiftResult {
    HueShiftResult {
        to_hsv: Hsv::new(raw.to_hsv[0], raw.to_hsv[1], raw.to_hsv[2]),
        to_hsl: Hsl::new(raw.to_hsl[0], raw.to_hsl[1], raw.to_hsl[2]),
        from_hsv: Rgb::new(raw.from_hsv[0], raw.from_hsv[1], raw.from_hsv[2]),
        from_hsl: Rgb::new(raw.from_hsl[0], raw.from_hsl[1], raw.from_hsl[2]),
        shifted: Rgb::new(raw.shifted[0], raw.shifted[1], raw.shifted[2]),
        saturated: Rgb::new(raw.saturated[0], raw.saturated[1], raw.saturated[2]),
        valued: Rgb::new(raw.valued[0], raw.valued[1], raw.valued[2]),
        lightened: Rgb::new(raw.lightened[0], raw.lightened[1], raw.lightened[2]),
        yiq_rotated: Rgb::new(raw.yiq[0], raw.yiq[1], raw.yiq[2]),
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
