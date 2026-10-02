//! `wgpu` compute twin of the screen-space lens-flare layout golden
//! ([`lens_flare`](prism_render_architecture::particle::lens_flare), particle
//! design §16-§21).
//!
//! The `CPU` golden
//! [`lens_flare`](prism_render_architecture::particle::lens_flare) owns the
//! deterministic geometry a compositor draws over a bright point: it turns a
//! sampled `luminance` into a soft-`knee` bright-point weight, projects a chain
//! of *ghost* sprites along the optical axis, splits each ghost chromatically
//! along that axis, wraps a soft *halo* ring around the center, and dims every
//! ghost with a radial attenuation and an off-screen fade.
//!
//! [`GpuLensFlare`] is the on-device twin: one thread per [`LensFlareQuery`]
//! dispatches on an operation tag and reproduces the module's pure numeric
//! surface branch for branch — the `clamp01` and the manually expanded
//! `smoothstep` `t^2 (3 - 2 t)`, the euclidean `distance`, the `Rec. 709`
//! `luminance`, the `threshold_weight` soft knee, the guarded `axis_dir`
//! normalization, the `ghost_uv` reflection, the `ghost_chroma_uvs` split, the
//! `halo_weight` band, the rational `radial_attenuation`, the `screen_fade`
//! border falloff and the folded `sample_ghost` weight — so a passing
//! real-device parity test is direct evidence the ported kernel evaluates the
//! same layout the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query mirrors one golden function. The free-function primitives
//! `clamp01` / `smoothstep01` / `distance` are private to the reference module,
//! so the twin reproduces their closed forms exactly; the public
//! [`luminance`](prism_render_architecture::particle::lens_flare::luminance) and
//! the public [`LensFlareParams`](prism_render_architecture::particle::lens_flare::LensFlareParams)
//! methods (`threshold_weight`, `axis_dir`, `ghost_uv`, `ghost_chroma_uvs`,
//! `halo_weight`, `radial_attenuation`, `screen_fade`, `sample_ghost`) are
//! mirrored one op code each. The twin is one-shot per query: it lays out a
//! single ghost via `sample_ghost`, and the host supplies the index.
//!
//! # What stays on the host
//!
//! The variable-length ghost-chain walk
//! [`LensFlareParams::sample_ghosts`](prism_render_architecture::particle::lens_flare::LensFlareParams::sample_ghosts)
//! allocates a `Vec` and loops over `ghost_count`; it stays host-side and simply
//! issues one `SampleGhost` query per index. The `std430` serialization
//! [`LensFlareParams::to_std430`](prism_render_architecture::particle::lens_flare::LensFlareParams::to_std430)
//! and the `VEC4_STRIDE` packing it uses are byte tools that also stay on the
//! host; the twin consumes already-unpacked scalars.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `abs`, `+ - * /` and the two `sqrt` calls the euclidean distance and
//! the axis normalization need — with no `sin`, `cos`, `exp`, `log`, `tan`,
//! `smoothstep` builtin or optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, guarded
//! divides and at most two `sqrt` calls, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! lanes, tight enough to catch a genuinely wrong port (a dropped branch, a
//! swapped coefficient, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`；
//! standard screen-space lens-flare layout plus `wgpu` compute dispatch；无需
//! 外部数学库，无第三方引擎源码或衍生代码。

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

/// Operation tag: the private `clamp01` free function (clamp into `[0, 1]`).
const OP_CLAMP01: u32 = 0;
/// Operation tag: the private `smoothstep01` free function (`t^2 (3 - 2 t)`).
const OP_SMOOTHSTEP01: u32 = 1;
/// Operation tag: the private `distance` free function (euclidean `UV`
/// distance).
const OP_DISTANCE: u32 = 2;
/// Operation tag: [`luminance`](prism_render_architecture::particle::lens_flare::luminance).
const OP_LUMINANCE: u32 = 3;
/// Operation tag: [`LensFlareParams::threshold_weight`](prism_render_architecture::particle::lens_flare::LensFlareParams::threshold_weight).
const OP_THRESHOLD_WEIGHT: u32 = 4;
/// Operation tag: [`LensFlareParams::axis_dir`](prism_render_architecture::particle::lens_flare::LensFlareParams::axis_dir).
const OP_AXIS_DIR: u32 = 5;
/// Operation tag: [`LensFlareParams::ghost_uv`](prism_render_architecture::particle::lens_flare::LensFlareParams::ghost_uv).
const OP_GHOST_UV: u32 = 6;
/// Operation tag: [`LensFlareParams::ghost_chroma_uvs`](prism_render_architecture::particle::lens_flare::LensFlareParams::ghost_chroma_uvs).
const OP_GHOST_CHROMA_UVS: u32 = 7;
/// Operation tag: [`LensFlareParams::halo_weight`](prism_render_architecture::particle::lens_flare::LensFlareParams::halo_weight).
const OP_HALO_WEIGHT: u32 = 8;
/// Operation tag: [`LensFlareParams::radial_attenuation`](prism_render_architecture::particle::lens_flare::LensFlareParams::radial_attenuation).
const OP_RADIAL_ATTENUATION: u32 = 9;
/// Operation tag: [`LensFlareParams::screen_fade`](prism_render_architecture::particle::lens_flare::LensFlareParams::screen_fade).
const OP_SCREEN_FADE: u32 = 10;
/// Operation tag: [`LensFlareParams::sample_ghost`](prism_render_architecture::particle::lens_flare::LensFlareParams::sample_ghost).
const OP_SAMPLE_GHOST: u32 = 11;

/// The portable core-`WGSL` lens-flare numeric kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve`
/// dispatches on a per-query operation tag and mirrors the `CPU` golden
/// [`lens_flare`](prism_render_architecture::particle::lens_flare) numeric
/// surface; see the module documentation for the formulae.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`。
const LENS_FLARE_WGSL: &str = r#"
// lens_flare twin: one thread per query dispatches on an operation tag and
// reproduces the CPU golden `particle::lens_flare` numeric surface — the
// clamp01 and manually expanded smoothstep t^2(3-2t) primitives, the euclidean
// distance, the Rec. 709 luminance, the threshold_weight soft knee, the guarded
// axis_dir normalization, the ghost_uv reflection, the ghost_chroma_uvs split,
// the halo_weight band, the rational radial_attenuation, the screen_fade border
// falloff and the folded sample_ghost weight. It mirrors the reference branch
// for branch, uses only the portable core-WGSL subset (clamp/min/max/abs/sqrt
// and + - * /), takes no optional feature, uses no u64/u16, and runs unmodified
// on Metal, Vulkan and DX12. Each thread runs a fixed, bounded sequence, so the
// kernel provably terminates.
//
// Provenance: twinned from this repository's particle::lens_flare; no external
// math library, no third-party engine source or derived code.

// Denominators with magnitude at or below this are treated as (near) zero so a
// degenerate band falls back to a defined result instead of dividing by zero.
// Matches the reference `MIN_DENOM`.
const MIN_DENOM: f32 = 1e-6;

// Squared axis length at or below which the bright point is treated as sitting
// on the optical center, matching the reference `EPS_AXIS_SQ`.
const EPS_AXIS_SQ: f32 = 1e-12;

// Rec. 709 luminance channel weights, matching the reference `LUMA_*`.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

const OP_CLAMP01: u32 = 0u;
const OP_SMOOTHSTEP01: u32 = 1u;
const OP_DISTANCE: u32 = 2u;
const OP_LUMINANCE: u32 = 3u;
const OP_THRESHOLD_WEIGHT: u32 = 4u;
const OP_AXIS_DIR: u32 = 5u;
const OP_GHOST_UV: u32 = 6u;
const OP_GHOST_CHROMA_UVS: u32 = 7u;
const OP_HALO_WEIGHT: u32 = 8u;
const OP_RADIAL_ATTENUATION: u32 = 9u;
const OP_SCREEN_FADE: u32 = 10u;
const OP_SAMPLE_GHOST: u32 = 11u;

const KIND_SCALAR: u32 = 0u;
const KIND_VEC2: u32 = 1u;
const KIND_CHROMA: u32 = 2u;
const KIND_GHOST: u32 = 3u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which golden function this slot evaluates.
    op: u32,
    // Ghost index for ghost_uv / ghost_chroma_uvs / sample_ghost.
    index: u32,
    // Generic scalar input for clamp01 / smoothstep01.
    x: f32,
    // threshold_weight: luminance threshold, soft-knee half-width and sampled
    // luminance.
    threshold: f32,
    knee: f32,
    lum: f32,
    // Optical center UV.
    center_x: f32,
    center_y: f32,
    // Bright-point UV for the ghost / axis operations.
    bright_x: f32,
    bright_y: f32,
    // Sampled UV for distance (point b), halo_weight and screen_fade.
    uv_x: f32,
    uv_y: f32,
    // Ghost axis scale and per-ghost chromatic split distance.
    ghost_spacing: f32,
    chroma_offset: f32,
    // Halo ring band center, half-width and peak intensity.
    halo_radius: f32,
    halo_width: f32,
    halo_intensity: f32,
    // Radial attenuation coefficient and the radius it is evaluated at.
    radial_falloff: f32,
    r: f32,
    // Off-screen fade margin and the incoming bright-point weight.
    edge_fade: f32,
    bright_weight: f32,
    // Linear RGB triple for luminance.
    rgb_r: f32,
    rgb_g: f32,
    rgb_b: f32,
}

struct Result {
    // Result-kind tag selecting how the host decodes the lanes.
    kind: u32,
    // Scalar result (clamp01, smoothstep01, distance, luminance,
    // threshold_weight, halo_weight, radial_attenuation, screen_fade).
    scalar: f32,
    // Vector result (axis_dir, ghost_uv).
    vec_x: f32,
    vec_y: f32,
    // Chromatic per-channel UVs (ghost_chroma_uvs and sample_ghost):
    // red, green, blue.
    cr_x: f32,
    cr_y: f32,
    cg_x: f32,
    cg_y: f32,
    cb_x: f32,
    cb_y: f32,
    // Folded ghost weight (sample_ghost).
    weight: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamp into the closed unit interval, mirroring the reference `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hermite smoothstep shaper t^2(3-2t) after clamping, manually expanded so no
// smoothstep builtin is relied on. Mirrors the reference `smoothstep01`.
fn smoothstep01(t: f32) -> f32 {
    let c = clamp01(t);
    return c * c * (3.0 - 2.0 * c);
}

// Euclidean distance between two UV points, mirroring the reference `distance`.
fn dist2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    return sqrt(dx * dx + dy * dy);
}

// Unit-length optical-axis direction, mirroring the reference `axis_dir`: a
// bright point on the center degenerates to the zero vector instead of
// normalizing a (near) zero vector.
fn axis_dir(cx: f32, cy: f32, bx: f32, by: f32) -> vec2<f32> {
    let dx = bx - cx;
    let dy = by - cy;
    let len_sq = dx * dx + dy * dy;
    if (len_sq <= EPS_AXIS_SQ) {
        return vec2<f32>(0.0, 0.0);
    }
    let inv = 1.0 / sqrt(len_sq);
    return vec2<f32>(dx * inv, dy * inv);
}

// Ghost UV, the reflection of the bright UV through the center scaled by
// -spacing * index, mirroring the reference `ghost_uv`.
fn ghost_uv(cx: f32, cy: f32, spacing: f32, bx: f32, by: f32, index: u32) -> vec2<f32> {
    let factor = -spacing * f32(index);
    return vec2<f32>(cx + (bx - cx) * factor, cy + (by - cy) * factor);
}

// Rational radial attenuation 1 / (1 + falloff r^2), mirroring the reference
// `radial_attenuation`.
fn radial_attenuation(falloff: f32, r: f32) -> f32 {
    return 1.0 / (1.0 + falloff * r * r);
}

// Per-component off-screen fade, mirroring the reference private
// `edge_component`: smoothstep of the distance to the nearest border over the
// edge-fade margin.
fn edge_component(c: f32, edge_fade: f32) -> f32 {
    let border = min(c, 1.0 - c);
    return smoothstep01(border / (edge_fade + MIN_DENOM));
}

// Off-screen fade weight, mirroring the reference `screen_fade`.
fn screen_fade(uvx: f32, uvy: f32, edge_fade: f32) -> f32 {
    return edge_component(uvx, edge_fade) * edge_component(uvy, edge_fade);
}

// Halo ring weight, mirroring the reference `halo_weight`: a smoothstep band
// centered on `radius` peaking at `intensity`.
fn halo_weight(uvx: f32, uvy: f32, cx: f32, cy: f32, radius: f32, width: f32, intensity: f32) -> f32 {
    let r = dist2(uvx, uvy, cx, cy);
    let off_band = abs(r - radius);
    let t = off_band / (width + MIN_DENOM);
    return intensity * (1.0 - smoothstep01(t));
}

// Bright-point weight soft knee, mirroring the reference `threshold_weight`.
fn threshold_weight(threshold: f32, knee: f32, lum: f32) -> f32 {
    let lo = threshold - knee;
    let span = max(2.0 * knee, MIN_DENOM);
    return smoothstep01((lum - lo) / span);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.kind = KIND_SCALAR;
    out.scalar = 0.0;
    out.vec_x = 0.0;
    out.vec_y = 0.0;
    out.cr_x = 0.0;
    out.cr_y = 0.0;
    out.cg_x = 0.0;
    out.cg_y = 0.0;
    out.cb_x = 0.0;
    out.cb_y = 0.0;
    out.weight = 0.0;

    switch q.op {
        case OP_CLAMP01: {
            out.kind = KIND_SCALAR;
            out.scalar = clamp01(q.x);
        }
        case OP_SMOOTHSTEP01: {
            out.kind = KIND_SCALAR;
            out.scalar = smoothstep01(q.x);
        }
        case OP_DISTANCE: {
            out.kind = KIND_SCALAR;
            out.scalar = dist2(q.center_x, q.center_y, q.uv_x, q.uv_y);
        }
        case OP_LUMINANCE: {
            out.kind = KIND_SCALAR;
            out.scalar = q.rgb_r * LUMA_R + q.rgb_g * LUMA_G + q.rgb_b * LUMA_B;
        }
        case OP_THRESHOLD_WEIGHT: {
            out.kind = KIND_SCALAR;
            out.scalar = threshold_weight(q.threshold, q.knee, q.lum);
        }
        case OP_AXIS_DIR: {
            out.kind = KIND_VEC2;
            let d = axis_dir(q.center_x, q.center_y, q.bright_x, q.bright_y);
            out.vec_x = d.x;
            out.vec_y = d.y;
        }
        case OP_GHOST_UV: {
            out.kind = KIND_VEC2;
            let g = ghost_uv(q.center_x, q.center_y, q.ghost_spacing, q.bright_x, q.bright_y, q.index);
            out.vec_x = g.x;
            out.vec_y = g.y;
        }
        case OP_GHOST_CHROMA_UVS: {
            out.kind = KIND_CHROMA;
            let g = ghost_uv(q.center_x, q.center_y, q.ghost_spacing, q.bright_x, q.bright_y, q.index);
            let dir = axis_dir(q.center_x, q.center_y, q.bright_x, q.bright_y);
            let ox = dir.x * q.chroma_offset;
            let oy = dir.y * q.chroma_offset;
            out.cr_x = g.x + ox;
            out.cr_y = g.y + oy;
            out.cg_x = g.x;
            out.cg_y = g.y;
            out.cb_x = g.x - ox;
            out.cb_y = g.y - oy;
        }
        case OP_HALO_WEIGHT: {
            out.kind = KIND_SCALAR;
            out.scalar = halo_weight(
                q.uv_x,
                q.uv_y,
                q.center_x,
                q.center_y,
                q.halo_radius,
                q.halo_width,
                q.halo_intensity,
            );
        }
        case OP_RADIAL_ATTENUATION: {
            out.kind = KIND_SCALAR;
            out.scalar = radial_attenuation(q.radial_falloff, q.r);
        }
        case OP_SCREEN_FADE: {
            out.kind = KIND_SCALAR;
            out.scalar = screen_fade(q.uv_x, q.uv_y, q.edge_fade);
        }
        case OP_SAMPLE_GHOST: {
            out.kind = KIND_GHOST;
            let g = ghost_uv(q.center_x, q.center_y, q.ghost_spacing, q.bright_x, q.bright_y, q.index);
            let dir = axis_dir(q.center_x, q.center_y, q.bright_x, q.bright_y);
            let ox = dir.x * q.chroma_offset;
            let oy = dir.y * q.chroma_offset;
            out.cr_x = g.x + ox;
            out.cr_y = g.y + oy;
            out.cg_x = g.x;
            out.cg_y = g.y;
            out.cb_x = g.x - ox;
            out.cb_y = g.y - oy;
            let rr = dist2(g.x, g.y, q.center_x, q.center_y);
            let w = q.bright_weight * radial_attenuation(q.radial_falloff, rr) * screen_fade(g.x, g.y, q.edge_fade);
            out.weight = max(w, 0.0);
        }
        default: {
        }
    }
    results[idx] = out;
}
"#;

/// One numeric query against the lens-flare twin.
///
/// Each variant mirrors one golden function. The scalar `UV`s are plain
/// `[f32; 2]` pairs and the ghost index is a `u32`, so the twin carries no
/// reference type and depends only on already-unpacked scalars. The shaping
/// coefficients (`knee`, `halo_width`, `halo_intensity`, `radial_falloff`,
/// `edge_fade`) are expected to be the non-negative values the reference
/// [`LensFlareParams::new`](prism_render_architecture::particle::lens_flare::LensFlareParams::new)
/// clamps to, so the twin and the reference see identical inputs.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LensFlareQuery {
    /// Clamp into `[0, 1]` (the private `clamp01`).
    Clamp01 {
        /// Scalar to clamp.
        x: f32,
    },
    /// Hermite `smoothstep` `t^2 (3 - 2 t)` after clamping (the private
    /// `smoothstep01`).
    Smoothstep01 {
        /// Shaper parameter.
        t: f32,
    },
    /// Euclidean `UV` distance (the private `distance`).
    Distance {
        /// First `UV` point.
        a: [f32; 2],
        /// Second `UV` point.
        b: [f32; 2],
    },
    /// `Rec. 709` perceptual `luminance`
    /// ([`luminance`](prism_render_architecture::particle::lens_flare::luminance)).
    Luminance {
        /// Linear `RGB` triple.
        rgb: [f32; 3],
    },
    /// Soft-`knee` bright-point weight
    /// ([`LensFlareParams::threshold_weight`](prism_render_architecture::particle::lens_flare::LensFlareParams::threshold_weight)).
    ThresholdWeight {
        /// `Luminance` threshold.
        threshold: f32,
        /// Soft-`knee` half-width (already non-negative).
        knee: f32,
        /// Sampled `luminance`.
        lum: f32,
    },
    /// Unit optical-axis direction
    /// ([`LensFlareParams::axis_dir`](prism_render_architecture::particle::lens_flare::LensFlareParams::axis_dir)).
    AxisDir {
        /// Optical center `UV`.
        center: [f32; 2],
        /// Bright-point `UV`.
        bright_uv: [f32; 2],
    },
    /// Ghost `UV` reflection
    /// ([`LensFlareParams::ghost_uv`](prism_render_architecture::particle::lens_flare::LensFlareParams::ghost_uv)).
    GhostUv {
        /// Optical center `UV`.
        center: [f32; 2],
        /// Per-index axis scale.
        ghost_spacing: f32,
        /// Bright-point `UV`.
        bright_uv: [f32; 2],
        /// Ghost index.
        index: u32,
    },
    /// Per-channel chromatic split `UV`s
    /// ([`LensFlareParams::ghost_chroma_uvs`](prism_render_architecture::particle::lens_flare::LensFlareParams::ghost_chroma_uvs)).
    GhostChromaUvs {
        /// Optical center `UV`.
        center: [f32; 2],
        /// Per-index axis scale.
        ghost_spacing: f32,
        /// Per-ghost chromatic split distance.
        chroma_offset: f32,
        /// Bright-point `UV`.
        bright_uv: [f32; 2],
        /// Ghost index.
        index: u32,
    },
    /// Halo ring weight
    /// ([`LensFlareParams::halo_weight`](prism_render_architecture::particle::lens_flare::LensFlareParams::halo_weight)).
    HaloWeight {
        /// Optical center `UV`.
        center: [f32; 2],
        /// Halo ring band center radius.
        halo_radius: f32,
        /// Halo ring band half-width (already non-negative).
        halo_width: f32,
        /// Halo ring peak intensity (already non-negative).
        halo_intensity: f32,
        /// Sampled `UV`.
        uv: [f32; 2],
    },
    /// Rational radial attenuation
    /// ([`LensFlareParams::radial_attenuation`](prism_render_architecture::particle::lens_flare::LensFlareParams::radial_attenuation)).
    RadialAttenuation {
        /// Attenuation coefficient (already non-negative).
        radial_falloff: f32,
        /// Radius from the center.
        r: f32,
    },
    /// Off-screen fade weight
    /// ([`LensFlareParams::screen_fade`](prism_render_architecture::particle::lens_flare::LensFlareParams::screen_fade)).
    ScreenFade {
        /// Off-screen fade margin (already non-negative).
        edge_fade: f32,
        /// Sampled `UV`.
        uv: [f32; 2],
    },
    /// A single laid-out ghost sample
    /// ([`LensFlareParams::sample_ghost`](prism_render_architecture::particle::lens_flare::LensFlareParams::sample_ghost)).
    SampleGhost {
        /// Optical center `UV`.
        center: [f32; 2],
        /// Per-index axis scale.
        ghost_spacing: f32,
        /// Per-ghost chromatic split distance.
        chroma_offset: f32,
        /// Attenuation coefficient (already non-negative).
        radial_falloff: f32,
        /// Off-screen fade margin (already non-negative).
        edge_fade: f32,
        /// Bright-point `UV`.
        bright_uv: [f32; 2],
        /// Incoming bright-point weight.
        bright_weight: f32,
        /// Ghost index.
        index: u32,
    },
}

/// One resolved answer, mirroring whichever golden function the query selected.
///
/// `Scalar` carries the single-value results; `Vec2` the axis / ghost `UV`
/// pairs; `ChromaUvs` the three per-channel split `UV`s; `Ghost` the full
/// laid-out ghost (three `UV`s plus the folded weight).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LensFlareResult {
    /// A single scalar (`clamp01`, `smoothstep01`, `distance`, `luminance`,
    /// `threshold_weight`, `halo_weight`, `radial_attenuation`, `screen_fade`).
    Scalar(f32),
    /// A single `UV` pair (`axis_dir`, `ghost_uv`).
    Vec2([f32; 2]),
    /// The three per-channel split `UV`s `[red, green, blue]`
    /// (`ghost_chroma_uvs`).
    ChromaUvs([[f32; 2]; 3]),
    /// A laid-out ghost (`sample_ghost`).
    Ghost {
        /// Per-channel sampled `UV`s `[red, green, blue]`.
        uv_rgb: [[f32; 2]; 3],
        /// Folded ghost weight.
        weight: f32,
    },
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// All members are `4`-byte scalars, so the struct packs with no interior
/// padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting which golden function this slot evaluates.
    op: u32,
    /// Ghost index.
    index: u32,
    /// Generic scalar input (`clamp01` / `smoothstep01`).
    x: f32,
    /// `Luminance` threshold.
    threshold: f32,
    /// Soft-`knee` half-width.
    knee: f32,
    /// Sampled `luminance`.
    lum: f32,
    /// Optical center `UV` x.
    center_x: f32,
    /// Optical center `UV` y.
    center_y: f32,
    /// Bright-point `UV` x.
    bright_x: f32,
    /// Bright-point `UV` y.
    bright_y: f32,
    /// Sampled `UV` x (distance point b / `halo_weight` / `screen_fade`).
    uv_x: f32,
    /// Sampled `UV` y.
    uv_y: f32,
    /// Per-index axis scale.
    ghost_spacing: f32,
    /// Per-ghost chromatic split distance.
    chroma_offset: f32,
    /// Halo ring band center radius.
    halo_radius: f32,
    /// Halo ring band half-width.
    halo_width: f32,
    /// Halo ring peak intensity.
    halo_intensity: f32,
    /// Radial attenuation coefficient.
    radial_falloff: f32,
    /// Radius the attenuation is evaluated at.
    r: f32,
    /// Off-screen fade margin.
    edge_fade: f32,
    /// Incoming bright-point weight.
    bright_weight: f32,
    /// Linear `RGB` red channel.
    rgb_r: f32,
    /// Linear `RGB` green channel.
    rgb_g: f32,
    /// Linear `RGB` blue channel.
    rgb_b: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Result-kind tag.
    kind: u32,
    /// Scalar result lane.
    scalar: f32,
    /// Vector result `UV` x.
    vec_x: f32,
    /// Vector result `UV` y.
    vec_y: f32,
    /// Red channel `UV` x.
    cr_x: f32,
    /// Red channel `UV` y.
    cr_y: f32,
    /// Green channel `UV` x.
    cg_x: f32,
    /// Green channel `UV` y.
    cg_y: f32,
    /// Blue channel `UV` x.
    cb_x: f32,
    /// Blue channel `UV` y.
    cb_y: f32,
    /// Folded ghost weight.
    weight: f32,
}

/// A fully zeroed query slot, filled per variant by [`encode_query`].
fn empty_query() -> GpuQuery {
    GpuQuery {
        op: 0,
        index: 0,
        x: 0.0,
        threshold: 0.0,
        knee: 0.0,
        lum: 0.0,
        center_x: 0.0,
        center_y: 0.0,
        bright_x: 0.0,
        bright_y: 0.0,
        uv_x: 0.0,
        uv_y: 0.0,
        ghost_spacing: 0.0,
        chroma_offset: 0.0,
        halo_radius: 0.0,
        halo_width: 0.0,
        halo_intensity: 0.0,
        radial_falloff: 0.0,
        r: 0.0,
        edge_fade: 0.0,
        bright_weight: 0.0,
        rgb_r: 0.0,
        rgb_g: 0.0,
        rgb_b: 0.0,
    }
}

/// Encodes one [`LensFlareQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(query: &LensFlareQuery) -> GpuQuery {
    let mut q = empty_query();
    match query {
        LensFlareQuery::Clamp01 { x } => {
            q.op = OP_CLAMP01;
            q.x = *x;
        }
        LensFlareQuery::Smoothstep01 { t } => {
            q.op = OP_SMOOTHSTEP01;
            q.x = *t;
        }
        LensFlareQuery::Distance { a, b } => {
            q.op = OP_DISTANCE;
            q.center_x = a[0];
            q.center_y = a[1];
            q.uv_x = b[0];
            q.uv_y = b[1];
        }
        LensFlareQuery::Luminance { rgb } => {
            q.op = OP_LUMINANCE;
            q.rgb_r = rgb[0];
            q.rgb_g = rgb[1];
            q.rgb_b = rgb[2];
        }
        LensFlareQuery::ThresholdWeight {
            threshold,
            knee,
            lum,
        } => {
            q.op = OP_THRESHOLD_WEIGHT;
            q.threshold = *threshold;
            q.knee = *knee;
            q.lum = *lum;
        }
        LensFlareQuery::AxisDir { center, bright_uv } => {
            q.op = OP_AXIS_DIR;
            q.center_x = center[0];
            q.center_y = center[1];
            q.bright_x = bright_uv[0];
            q.bright_y = bright_uv[1];
        }
        LensFlareQuery::GhostUv {
            center,
            ghost_spacing,
            bright_uv,
            index,
        } => {
            q.op = OP_GHOST_UV;
            q.center_x = center[0];
            q.center_y = center[1];
            q.ghost_spacing = *ghost_spacing;
            q.bright_x = bright_uv[0];
            q.bright_y = bright_uv[1];
            q.index = *index;
        }
        LensFlareQuery::GhostChromaUvs {
            center,
            ghost_spacing,
            chroma_offset,
            bright_uv,
            index,
        } => {
            q.op = OP_GHOST_CHROMA_UVS;
            q.center_x = center[0];
            q.center_y = center[1];
            q.ghost_spacing = *ghost_spacing;
            q.chroma_offset = *chroma_offset;
            q.bright_x = bright_uv[0];
            q.bright_y = bright_uv[1];
            q.index = *index;
        }
        LensFlareQuery::HaloWeight {
            center,
            halo_radius,
            halo_width,
            halo_intensity,
            uv,
        } => {
            q.op = OP_HALO_WEIGHT;
            q.center_x = center[0];
            q.center_y = center[1];
            q.halo_radius = *halo_radius;
            q.halo_width = *halo_width;
            q.halo_intensity = *halo_intensity;
            q.uv_x = uv[0];
            q.uv_y = uv[1];
        }
        LensFlareQuery::RadialAttenuation { radial_falloff, r } => {
            q.op = OP_RADIAL_ATTENUATION;
            q.radial_falloff = *radial_falloff;
            q.r = *r;
        }
        LensFlareQuery::ScreenFade { edge_fade, uv } => {
            q.op = OP_SCREEN_FADE;
            q.edge_fade = *edge_fade;
            q.uv_x = uv[0];
            q.uv_y = uv[1];
        }
        LensFlareQuery::SampleGhost {
            center,
            ghost_spacing,
            chroma_offset,
            radial_falloff,
            edge_fade,
            bright_uv,
            bright_weight,
            index,
        } => {
            q.op = OP_SAMPLE_GHOST;
            q.center_x = center[0];
            q.center_y = center[1];
            q.ghost_spacing = *ghost_spacing;
            q.chroma_offset = *chroma_offset;
            q.radial_falloff = *radial_falloff;
            q.edge_fade = *edge_fade;
            q.bright_x = bright_uv[0];
            q.bright_y = bright_uv[1];
            q.bright_weight = *bright_weight;
            q.index = *index;
        }
    }
    q
}

/// Decodes one packed [`GpuResult`] into the public [`LensFlareResult`], using
/// the originating `query` to select the result shape.
fn decode_result(query: &LensFlareQuery, raw: &GpuResult) -> LensFlareResult {
    match query {
        LensFlareQuery::Clamp01 { .. }
        | LensFlareQuery::Smoothstep01 { .. }
        | LensFlareQuery::Distance { .. }
        | LensFlareQuery::Luminance { .. }
        | LensFlareQuery::ThresholdWeight { .. }
        | LensFlareQuery::HaloWeight { .. }
        | LensFlareQuery::RadialAttenuation { .. }
        | LensFlareQuery::ScreenFade { .. } => LensFlareResult::Scalar(raw.scalar),
        LensFlareQuery::AxisDir { .. } | LensFlareQuery::GhostUv { .. } => {
            LensFlareResult::Vec2([raw.vec_x, raw.vec_y])
        }
        LensFlareQuery::GhostChromaUvs { .. } => LensFlareResult::ChromaUvs([
            [raw.cr_x, raw.cr_y],
            [raw.cg_x, raw.cg_y],
            [raw.cb_x, raw.cb_y],
        ]),
        LensFlareQuery::SampleGhost { .. } => LensFlareResult::Ghost {
            uv_rgb: [
                [raw.cr_x, raw.cr_y],
                [raw.cg_x, raw.cg_y],
                [raw.cb_x, raw.cb_y],
            ],
            weight: raw.weight,
        },
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

/// A compiled, reusable lens-flare compute pipeline, twinning the `CPU` golden
/// [`lens_flare`](prism_render_architecture::particle::lens_flare).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`。
pub struct GpuLensFlare {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLensFlare {
    /// Compiles the lens-flare kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLensFlare {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_lens_flare"),
            source: ShaderSource::Wgsl(LENS_FLARE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_lens_flare_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_lens_flare_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_lens_flare_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLensFlare {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`LensFlareResult`]
    /// per input, in order.
    ///
    /// The results match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::lens_flare`。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[LensFlareQuery]) -> Vec<LensFlareResult> {
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
            label: Some("prism_volumetric_lens_flare_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_lens_flare_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_lens_flare_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_lens_flare_bind_group"),
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
            label: Some("prism_volumetric_lens_flare_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_lens_flare_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_lens_flare_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
