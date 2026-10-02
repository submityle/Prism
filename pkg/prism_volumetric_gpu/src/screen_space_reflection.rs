//! `wgpu` compute twin of the screen-space-reflection (`SSR`) numeric kernel
//! ([`screen_space_reflection`](prism_render_architecture::particle::screen_space_reflection),
//! particle design §16-§21).
//!
//! The `CPU` golden
//! [`screen_space_reflection`](prism_render_architecture::particle::screen_space_reflection)
//! owns the view/screen-space reflected-ray march that mirrors nearby geometry
//! into glossy particle surfaces. This twin reproduces, one thread per query,
//! the module's pure numeric surface as a tagged batch so one dispatch can mix
//! clamp, interpolation, vector, projection and confidence-fade queries:
//!
//! - `clamp01`, the branch-free `0..=1` clamp;
//! - `smoothstep(edge0, edge1, x)`, the hand-written Hermite polynomial
//!   `t * t * (3 - 2 * t)` with the degenerate-span hard step (never the `WGSL`
//!   builtin `smoothstep`);
//! - the [`Vec3`](prism_render_architecture::particle::screen_space_reflection::Vec3)
//!   primitives `plus`, `minus`, `scale`, `dot`, `length` and `normalize`;
//! - [`reflect`](prism_render_architecture::particle::screen_space_reflection::reflect),
//!   the mirror identity `incident - 2 * dot(incident, normal) * normal`;
//! - [`Projection::project`](prism_render_architecture::particle::screen_space_reflection::Projection::project),
//!   the pinhole `view`-to-`UV` map whose `Option` is remapped to an
//!   `on_screen` flag plus the `UV` lanes;
//! - [`screen_edge_fade`](prism_render_architecture::particle::screen_space_reflection::screen_edge_fade),
//!   [`grazing_fade`](prism_render_architecture::particle::screen_space_reflection::grazing_fade)
//!   and [`distance_fade`](prism_render_architecture::particle::screen_space_reflection::distance_fade),
//!   the three confidence terms;
//! - the numeric part of `finalize_hit`, folding the three fades into a clamped
//!   confidence and packing the recovered
//!   [`SsrHit`](prism_render_architecture::particle::screen_space_reflection::SsrHit).
//!
//! [`GpuScreenSpaceReflection`] is the on-device twin: a passing real-device
//! parity test is direct evidence the ported kernel evaluates the same closed
//! forms and classifies the same degenerate spans the reference does, not merely
//! that the shader compiles.
//!
//! # What stays on the host
//!
//! The reference closure-driven ray-march
//! [`trace_reflection`](prism_render_architecture::particle::screen_space_reflection::trace_reflection)
//! and its `refine_crossing` bisection are **not** twinned: both drive a
//! `sample_depth` closure in a loop, which has no place in the
//! one-thread-per-element, no-aliasing kernel contract. The `march_schedule`
//! `Vec` allocation and the `to_std430` / `*_buffer_bytes` byte utilities also
//! stay host-side. The twin instead exposes the pure per-step numeric pieces the
//! march composes; a host that needs the full march pre-samples depth into a
//! fixed-length array and folds the crossing itself, then hands the twin the
//! `FinalizeHit` inputs.
//!
//! # Degenerate regimes
//!
//! `smoothstep` collapses a near-equal span (`edge1 - edge0 < MIN_EDGE`) to a
//! hard step rather than dividing by zero; `normalize` returns its input
//! unchanged below `MIN_NORM`; `project` reports off-screen when the `view`
//! depth is below `NEAR_MIN`; and `distance_fade` yields `0` for a non-positive
//! search radius. The parity fixtures stay clear of these thresholds by
//! rejection sampling so the comparison exercises the live solve.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `dot`, `sqrt` (reached only through the guarded `normalize` and
//! `length`, exactly as the reference does) and `+ - * /` — with no
//! transcendental call, no builtin `smoothstep`, no `u64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Each thread
//! performs a fixed, bounded sequence with no loop, so it provably terminates.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! guarded divides, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields and an exact
//! match on the discrete `on_screen` and `hit` flags.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::screen_space_reflection`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::screen_space_reflection::{SsrHit, Vec2, Vec3};
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

/// Operation tag: `clamp01`.
const OP_CLAMP01: u32 = 0;
/// Operation tag: `smoothstep`.
const OP_SMOOTHSTEP: u32 = 1;
/// Operation tag: [`Vec3::plus`](prism_render_architecture::particle::screen_space_reflection::Vec3::plus).
const OP_VEC3_PLUS: u32 = 2;
/// Operation tag: [`Vec3::minus`](prism_render_architecture::particle::screen_space_reflection::Vec3::minus).
const OP_VEC3_MINUS: u32 = 3;
/// Operation tag: [`Vec3::scale`](prism_render_architecture::particle::screen_space_reflection::Vec3::scale).
const OP_VEC3_SCALE: u32 = 4;
/// Operation tag: [`Vec3::dot`](prism_render_architecture::particle::screen_space_reflection::Vec3::dot).
const OP_VEC3_DOT: u32 = 5;
/// Operation tag: [`Vec3::length`](prism_render_architecture::particle::screen_space_reflection::Vec3::length).
const OP_VEC3_LENGTH: u32 = 6;
/// Operation tag: [`Vec3::normalize`](prism_render_architecture::particle::screen_space_reflection::Vec3::normalize).
const OP_VEC3_NORMALIZE: u32 = 7;
/// Operation tag: [`reflect`](prism_render_architecture::particle::screen_space_reflection::reflect).
const OP_REFLECT: u32 = 8;
/// Operation tag: [`Projection::project`](prism_render_architecture::particle::screen_space_reflection::Projection::project).
const OP_PROJECT: u32 = 9;
/// Operation tag: [`screen_edge_fade`](prism_render_architecture::particle::screen_space_reflection::screen_edge_fade).
const OP_SCREEN_EDGE_FADE: u32 = 10;
/// Operation tag: [`grazing_fade`](prism_render_architecture::particle::screen_space_reflection::grazing_fade).
const OP_GRAZING_FADE: u32 = 11;
/// Operation tag: [`distance_fade`](prism_render_architecture::particle::screen_space_reflection::distance_fade).
const OP_DISTANCE_FADE: u32 = 12;
/// Operation tag: the numeric part of `finalize_hit`.
const OP_FINALIZE_HIT: u32 = 13;

/// Discrete flag code written for an on-screen projection or a valid hit;
/// decoded with `== 1` so no `f32` equality is used.
const CODE_FLAG: u32 = 1;

/// The portable core-`WGSL` screen-space-reflection numeric kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` dispatches on a per-query operation tag and mirrors the `CPU` golden
/// [`screen_space_reflection`](prism_render_architecture::particle::screen_space_reflection)
/// numeric surface; see the module documentation for the formulae.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::screen_space_reflection`。
const SCREEN_SPACE_REFLECTION_WGSL: &str = r#"
// screen_space_reflection twin: one thread per query dispatches on an operation
// tag and reproduces the CPU golden `particle::screen_space_reflection` numeric
// surface — clamp01, the hand-written smoothstep polynomial, the Vec3 plus /
// minus / scale / dot / length / normalize primitives, the reflect mirror
// identity, the pinhole projection (Option remapped to an on_screen flag), the
// three confidence fades, and the numeric fold of finalize_hit. It mirrors the
// reference branch for branch, uses only the portable core-WGSL subset
// (clamp/min/max/dot/sqrt and + - * /), never the builtin smoothstep, takes no
// optional feature, and runs unmodified on Metal, Vulkan and DX12. There is no
// loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::screen_space_reflection;
// 无第三方引擎源码或衍生代码。

// Soft-edge / denominator guard below which a smoothstep interval collapses to a
// hard step. Matches the reference `MIN_EDGE`.
const MIN_EDGE: f32 = 1.0e-6;
// Minimum vector length treated as non-degenerate when normalizing. Matches the
// reference `MIN_NORM`.
const MIN_NORM: f32 = 1.0e-6;
// Minimum positive view-space depth in front of the pinhole. Matches the
// reference `NEAR_MIN`.
const NEAR_MIN: f32 = 1.0e-4;
// Minimum march step / search radius treated as making progress. Matches the
// reference `MIN_STEP`.
const MIN_STEP: f32 = 1.0e-6;

// Operation tags, mirroring the host-side OP_* constants.
const OP_CLAMP01: u32 = 0u;
const OP_SMOOTHSTEP: u32 = 1u;
const OP_VEC3_PLUS: u32 = 2u;
const OP_VEC3_MINUS: u32 = 3u;
const OP_VEC3_SCALE: u32 = 4u;
const OP_VEC3_DOT: u32 = 5u;
const OP_VEC3_LENGTH: u32 = 6u;
const OP_VEC3_NORMALIZE: u32 = 7u;
const OP_REFLECT: u32 = 8u;
const OP_PROJECT: u32 = 9u;
const OP_SCREEN_EDGE_FADE: u32 = 10u;
const OP_GRAZING_FADE: u32 = 11u;
const OP_DISTANCE_FADE: u32 = 12u;
const OP_FINALIZE_HIT: u32 = 13u;

// Result-kind tags.
const KIND_SCALAR: u32 = 0u;
const KIND_VECTOR: u32 = 1u;
const KIND_PROJECTION: u32 = 2u;
const KIND_HIT: u32 = 3u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which golden function this slot evaluates.
    op: u32,
    pad_op0: u32,
    pad_op1: u32,
    pad_op2: u32,
    // Scalar block: op-specific scalar arguments (edges / x / scale / focal /
    // edge width / bias / distances), packed into one vec4 slot.
    args: vec4<f32>,
    // Primary vector slot: vec operand a / incident / reflect_dir / projected
    // point.
    vec_a: vec3<f32>,
    pad_a: f32,
    // Secondary vector slot: vec operand b / surface normal / view_dir.
    vec_b: vec3<f32>,
    pad_b: f32,
    // Screen-UV slot (uv.x, uv.y, 0) for the edge-fade and finalize ops.
    uv: vec3<f32>,
    pad_uv: f32,
}

struct Result {
    // Result-kind tag selecting how the host decodes the lanes.
    kind: u32,
    // Discrete flag: on_screen for a projection, hit for a finalize.
    flag: u32,
    pad0: u32,
    pad1: u32,
    // Scalar result (clamp / smoothstep / dot / length / fades / confidence).
    scalar: f32,
    // Hit distance carried through a finalize result.
    hit_distance: f32,
    pad2: f32,
    pad3: f32,
    // Vector result, or the projected / finalized UV in its xy lanes.
    vec_r: vec3<f32>,
    pad_r: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Branch-free 0..=1 clamp, mirroring the reference `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hand-written Hermite smoothstep, mirroring the reference `smoothstep` branch
// for branch: a near-equal span collapses to a hard step at `edge1`, otherwise
// the clamped parameter is shaped by `t * t * (3 - 2 * t)`. Named `smooth_fade`
// so it never collides with the forbidden WGSL builtin `smoothstep`.
fn smooth_fade(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (span < MIN_EDGE) {
        if (x < edge1) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp01((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Guarded normalize, mirroring the reference `Vec3::normalize`: a vector shorter
// than `MIN_NORM` is returned unchanged so a degenerate direction never divides
// by zero.
fn normalize_or_self(v: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(v, v));
    if (len < MIN_NORM) {
        return v;
    }
    return v * (1.0 / len);
}

// Screen-border confidence fade, mirroring the reference `screen_edge_fade`.
fn screen_edge_fade(uv: vec2<f32>, edge: f32) -> f32 {
    if (edge < MIN_EDGE) {
        let inside = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
        if (inside) {
            return 1.0;
        }
        return 0.0;
    }
    let fx = smooth_fade(0.0, edge, uv.x) * smooth_fade(0.0, edge, 1.0 - uv.x);
    let fy = smooth_fade(0.0, edge, uv.y) * smooth_fade(0.0, edge, 1.0 - uv.y);
    return clamp01(fx * fy);
}

// Backward-ray confidence fade, mirroring the reference `grazing_fade`.
fn grazing_fade(reflect_dir: vec3<f32>, view_dir: vec3<f32>, bias: f32) -> f32 {
    let facing = dot(reflect_dir, view_dir);
    return smooth_fade(-bias, bias, facing);
}

// March-budget confidence fade, mirroring the reference `distance_fade`.
fn distance_fade(hit_distance: f32, max_distance: f32) -> f32 {
    if (max_distance < MIN_STEP) {
        return 0.0;
    }
    let frac = hit_distance / max_distance;
    return 1.0 - smooth_fade(0.0, 1.0, frac);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let vec_a = q.vec_a;
    let vec_b = q.vec_b;
    let uv = vec2<f32>(q.uv.x, q.uv.y);

    var out: Result;
    out.kind = KIND_SCALAR;
    out.flag = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.scalar = 0.0;
    out.hit_distance = 0.0;
    out.pad2 = 0.0;
    out.pad3 = 0.0;
    out.vec_r = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_r = 0.0;

    switch q.op {
        case OP_CLAMP01: {
            out.kind = KIND_SCALAR;
            out.scalar = clamp01(q.args.x);
        }
        case OP_SMOOTHSTEP: {
            out.kind = KIND_SCALAR;
            out.scalar = smooth_fade(q.args.x, q.args.y, q.args.z);
        }
        case OP_VEC3_PLUS: {
            out.kind = KIND_VECTOR;
            out.vec_r = vec_a + vec_b;
        }
        case OP_VEC3_MINUS: {
            out.kind = KIND_VECTOR;
            out.vec_r = vec_a - vec_b;
        }
        case OP_VEC3_SCALE: {
            out.kind = KIND_VECTOR;
            out.vec_r = vec_a * q.args.x;
        }
        case OP_VEC3_DOT: {
            out.kind = KIND_SCALAR;
            out.scalar = dot(vec_a, vec_b);
        }
        case OP_VEC3_LENGTH: {
            out.kind = KIND_SCALAR;
            out.scalar = sqrt(dot(vec_a, vec_a));
        }
        case OP_VEC3_NORMALIZE: {
            out.kind = KIND_VECTOR;
            out.vec_r = normalize_or_self(vec_a);
        }
        case OP_REFLECT: {
            out.kind = KIND_VECTOR;
            // reflect(incident, normal) = incident - 2 * dot(incident, normal) * normal.
            out.vec_r = vec_a - vec_b * (2.0 * dot(vec_a, vec_b));
        }
        case OP_PROJECT: {
            out.kind = KIND_PROJECTION;
            if (vec_a.z < NEAR_MIN) {
                out.flag = 0u;
                out.vec_r = vec3<f32>(0.0, 0.0, 0.0);
            } else {
                let ndc_x = q.args.x * vec_a.x / vec_a.z;
                let ndc_y = q.args.y * vec_a.y / vec_a.z;
                out.flag = 1u;
                out.vec_r = vec3<f32>(0.5 + 0.5 * ndc_x, 0.5 + 0.5 * ndc_y, 0.0);
            }
        }
        case OP_SCREEN_EDGE_FADE: {
            out.kind = KIND_SCALAR;
            out.scalar = screen_edge_fade(uv, q.args.x);
        }
        case OP_GRAZING_FADE: {
            out.kind = KIND_SCALAR;
            out.scalar = grazing_fade(vec_a, vec_b, q.args.x);
        }
        case OP_DISTANCE_FADE: {
            out.kind = KIND_SCALAR;
            out.scalar = distance_fade(q.args.x, q.args.y);
        }
        case OP_FINALIZE_HIT: {
            out.kind = KIND_HIT;
            // args.x = edge_fade, args.y = grazing_bias, args.z = max_distance,
            // args.w = hit_distance; vec_a = reflect_dir, vec_b = view_dir.
            let edge = screen_edge_fade(uv, q.args.x);
            let graze = grazing_fade(vec_a, vec_b, q.args.y);
            let dist = distance_fade(q.args.w, q.args.z);
            out.flag = 1u;
            out.scalar = clamp01(edge * graze * dist);
            out.hit_distance = q.args.w;
            out.vec_r = vec3<f32>(uv.x, uv.y, 0.0);
        }
        default: {
        }
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`SCREEN_SPACE_REFLECTION_WGSL`].
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
    /// Operation tag selecting which golden function this slot evaluates.
    op: u32,
    /// Padding word.
    pad_op0: u32,
    /// Padding word.
    pad_op1: u32,
    /// Padding word.
    pad_op2: u32,
    /// Op-specific scalar arguments.
    args: [f32; 4],
    /// Primary vector slot.
    vec_a: [f32; 3],
    /// Pad lane after `vec_a`.
    pad_a: f32,
    /// Secondary vector slot.
    vec_b: [f32; 3],
    /// Pad lane after `vec_b`.
    pad_b: f32,
    /// Screen-`UV` slot (`uv.x`, `uv.y`, `0`).
    uv: [f32; 3],
    /// Pad lane after `uv`.
    pad_uv: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Result-kind tag.
    kind: u32,
    /// Discrete flag (`on_screen` for a projection, `hit` for a finalize).
    flag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Scalar result.
    scalar: f32,
    /// Hit distance carried through a finalize result.
    hit_distance: f32,
    /// Padding word.
    pad2: f32,
    /// Padding word.
    pad3: f32,
    /// Vector result, or the projected / finalized `UV` in its `xy` lanes.
    vec_r: [f32; 3],
    /// Pad lane after `vec_r`.
    pad_r: f32,
}

/// One numeric query against the screen-space-reflection twin.
///
/// Each variant mirrors one golden function. `Project` returns the pinhole
/// `UV` with an `on_screen` flag standing in for the reference `Option`;
/// `FinalizeHit` folds the three confidence fades into the clamped confidence of
/// a recovered
/// [`SsrHit`](prism_render_architecture::particle::screen_space_reflection::SsrHit).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::screen_space_reflection`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScreenSpaceReflectionQuery {
    /// Branch-free `0..=1` clamp (`clamp01`).
    Clamp01 {
        /// Scalar to clamp.
        x: f32,
    },
    /// Hand-written Hermite `smoothstep(edge0, edge1, x)`.
    Smoothstep {
        /// Lower edge.
        edge0: f32,
        /// Upper edge.
        edge1: f32,
        /// Sample point.
        x: f32,
    },
    /// Component-wise sum
    /// ([`Vec3::plus`](prism_render_architecture::particle::screen_space_reflection::Vec3::plus)).
    Vec3Plus {
        /// Left operand.
        a: Vec3,
        /// Right operand.
        b: Vec3,
    },
    /// Component-wise difference
    /// ([`Vec3::minus`](prism_render_architecture::particle::screen_space_reflection::Vec3::minus)).
    Vec3Minus {
        /// Left operand.
        a: Vec3,
        /// Right operand.
        b: Vec3,
    },
    /// Scalar multiple
    /// ([`Vec3::scale`](prism_render_architecture::particle::screen_space_reflection::Vec3::scale)).
    Vec3Scale {
        /// Vector operand.
        a: Vec3,
        /// Scalar factor.
        scalar: f32,
    },
    /// Dot product
    /// ([`Vec3::dot`](prism_render_architecture::particle::screen_space_reflection::Vec3::dot)).
    Vec3Dot {
        /// Left operand.
        a: Vec3,
        /// Right operand.
        b: Vec3,
    },
    /// Euclidean length
    /// ([`Vec3::length`](prism_render_architecture::particle::screen_space_reflection::Vec3::length)).
    Vec3Length {
        /// Vector operand.
        a: Vec3,
    },
    /// Guarded unit direction
    /// ([`Vec3::normalize`](prism_render_architecture::particle::screen_space_reflection::Vec3::normalize)).
    Vec3Normalize {
        /// Vector operand.
        a: Vec3,
    },
    /// Mirror of an incident direction about a normal
    /// ([`reflect`](prism_render_architecture::particle::screen_space_reflection::reflect)).
    Reflect {
        /// Incident direction.
        incident: Vec3,
        /// Surface normal.
        normal: Vec3,
    },
    /// Pinhole projection of a `view`-space point to screen `UV`
    /// ([`Projection::project`](prism_render_architecture::particle::screen_space_reflection::Projection::project)).
    Project {
        /// Horizontal `NDC`-per-`view`-unit scale.
        focal_x: f32,
        /// Vertical `NDC`-per-`view`-unit scale.
        focal_y: f32,
        /// `view`-space point to project.
        point: Vec3,
    },
    /// Screen-border confidence fade
    /// ([`screen_edge_fade`](prism_render_architecture::particle::screen_space_reflection::screen_edge_fade)).
    ScreenEdgeFade {
        /// Screen `UV` to fade.
        uv: Vec2,
        /// Ramp width at each border.
        edge: f32,
    },
    /// Backward-ray confidence fade
    /// ([`grazing_fade`](prism_render_architecture::particle::screen_space_reflection::grazing_fade)).
    GrazingFade {
        /// Reflected ray direction.
        reflect_dir: Vec3,
        /// View direction.
        view_dir: Vec3,
        /// `smoothstep` half-width.
        bias: f32,
    },
    /// March-budget confidence fade
    /// ([`distance_fade`](prism_render_architecture::particle::screen_space_reflection::distance_fade)).
    DistanceFade {
        /// Distance along the ray at which the hit landed.
        hit_distance: f32,
        /// Search radius / fade denominator.
        max_distance: f32,
    },
    /// Numeric fold of `finalize_hit`: the clamped product of the three fades
    /// packed into a recovered
    /// [`SsrHit`](prism_render_architecture::particle::screen_space_reflection::SsrHit).
    FinalizeHit {
        /// Recovered reflection `UV`.
        uv: Vec2,
        /// Distance along the ray at which the hit landed.
        hit_distance: f32,
        /// Reflected ray direction.
        reflect_dir: Vec3,
        /// View direction.
        view_dir: Vec3,
        /// Screen-border ramp width.
        edge_fade: f32,
        /// Grazing `smoothstep` half-width.
        grazing_bias: f32,
        /// Search radius / distance-fade denominator.
        max_distance: f32,
    },
}

/// One resolved answer, mirroring whichever golden function the query selected.
///
/// `Projection` carries the `on_screen` flag that stands in for the reference
/// `Option<Vec2>`; `Hit` carries the recovered
/// [`SsrHit`](prism_render_architecture::particle::screen_space_reflection::SsrHit).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::screen_space_reflection`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScreenSpaceReflectionResult {
    /// A single scalar (clamp, smoothstep, dot, length, any fade).
    Scalar(f32),
    /// A single vector (plus, minus, scale, normalize, reflect).
    Vector(Vec3),
    /// A pinhole projection: the `UV` and whether the point was on screen.
    Projection {
        /// Whether the point projected to a finite on-screen `UV`.
        on_screen: bool,
        /// The projected `UV` (meaningful only when `on_screen`).
        uv: Vec2,
    },
    /// A recovered reflection sample from the numeric `finalize_hit` fold.
    Hit(SsrHit),
}

/// Converts a [`Vec3`] into its padded `std430` lane.
fn lane(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Reads a padded `std430` lane back into a [`Vec3`].
fn unlane(l: [f32; 3]) -> Vec3 {
    Vec3::new(l[0], l[1], l[2])
}

/// A fully zeroed query slot, filled per variant by [`encode_query`].
fn empty_query() -> GpuQuery {
    GpuQuery {
        op: 0,
        pad_op0: 0,
        pad_op1: 0,
        pad_op2: 0,
        args: [0.0; 4],
        vec_a: [0.0; 3],
        pad_a: 0.0,
        vec_b: [0.0; 3],
        pad_b: 0.0,
        uv: [0.0; 3],
        pad_uv: 0.0,
    }
}

/// Encodes one [`ScreenSpaceReflectionQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(query: &ScreenSpaceReflectionQuery) -> GpuQuery {
    let mut q = empty_query();
    match query {
        ScreenSpaceReflectionQuery::Clamp01 { x } => {
            q.op = OP_CLAMP01;
            q.args[0] = *x;
        }
        ScreenSpaceReflectionQuery::Smoothstep { edge0, edge1, x } => {
            q.op = OP_SMOOTHSTEP;
            q.args[0] = *edge0;
            q.args[1] = *edge1;
            q.args[2] = *x;
        }
        ScreenSpaceReflectionQuery::Vec3Plus { a, b } => {
            q.op = OP_VEC3_PLUS;
            q.vec_a = lane(*a);
            q.vec_b = lane(*b);
        }
        ScreenSpaceReflectionQuery::Vec3Minus { a, b } => {
            q.op = OP_VEC3_MINUS;
            q.vec_a = lane(*a);
            q.vec_b = lane(*b);
        }
        ScreenSpaceReflectionQuery::Vec3Scale { a, scalar } => {
            q.op = OP_VEC3_SCALE;
            q.vec_a = lane(*a);
            q.args[0] = *scalar;
        }
        ScreenSpaceReflectionQuery::Vec3Dot { a, b } => {
            q.op = OP_VEC3_DOT;
            q.vec_a = lane(*a);
            q.vec_b = lane(*b);
        }
        ScreenSpaceReflectionQuery::Vec3Length { a } => {
            q.op = OP_VEC3_LENGTH;
            q.vec_a = lane(*a);
        }
        ScreenSpaceReflectionQuery::Vec3Normalize { a } => {
            q.op = OP_VEC3_NORMALIZE;
            q.vec_a = lane(*a);
        }
        ScreenSpaceReflectionQuery::Reflect { incident, normal } => {
            q.op = OP_REFLECT;
            q.vec_a = lane(*incident);
            q.vec_b = lane(*normal);
        }
        ScreenSpaceReflectionQuery::Project {
            focal_x,
            focal_y,
            point,
        } => {
            q.op = OP_PROJECT;
            q.args[0] = *focal_x;
            q.args[1] = *focal_y;
            q.vec_a = lane(*point);
        }
        ScreenSpaceReflectionQuery::ScreenEdgeFade { uv, edge } => {
            q.op = OP_SCREEN_EDGE_FADE;
            q.uv = [uv.x, uv.y, 0.0];
            q.args[0] = *edge;
        }
        ScreenSpaceReflectionQuery::GrazingFade {
            reflect_dir,
            view_dir,
            bias,
        } => {
            q.op = OP_GRAZING_FADE;
            q.vec_a = lane(*reflect_dir);
            q.vec_b = lane(*view_dir);
            q.args[0] = *bias;
        }
        ScreenSpaceReflectionQuery::DistanceFade {
            hit_distance,
            max_distance,
        } => {
            q.op = OP_DISTANCE_FADE;
            q.args[0] = *hit_distance;
            q.args[1] = *max_distance;
        }
        ScreenSpaceReflectionQuery::FinalizeHit {
            uv,
            hit_distance,
            reflect_dir,
            view_dir,
            edge_fade,
            grazing_bias,
            max_distance,
        } => {
            q.op = OP_FINALIZE_HIT;
            q.uv = [uv.x, uv.y, 0.0];
            q.vec_a = lane(*reflect_dir);
            q.vec_b = lane(*view_dir);
            q.args[0] = *edge_fade;
            q.args[1] = *grazing_bias;
            q.args[2] = *max_distance;
            q.args[3] = *hit_distance;
        }
    }
    q
}

/// Decodes one packed [`GpuResult`] into the public
/// [`ScreenSpaceReflectionResult`], using the originating `query` to select the
/// result shape.
fn decode_result(
    query: &ScreenSpaceReflectionQuery,
    raw: &GpuResult,
) -> ScreenSpaceReflectionResult {
    match query {
        ScreenSpaceReflectionQuery::Clamp01 { .. }
        | ScreenSpaceReflectionQuery::Smoothstep { .. }
        | ScreenSpaceReflectionQuery::Vec3Dot { .. }
        | ScreenSpaceReflectionQuery::Vec3Length { .. }
        | ScreenSpaceReflectionQuery::ScreenEdgeFade { .. }
        | ScreenSpaceReflectionQuery::GrazingFade { .. }
        | ScreenSpaceReflectionQuery::DistanceFade { .. } => {
            ScreenSpaceReflectionResult::Scalar(raw.scalar)
        }
        ScreenSpaceReflectionQuery::Vec3Plus { .. }
        | ScreenSpaceReflectionQuery::Vec3Minus { .. }
        | ScreenSpaceReflectionQuery::Vec3Scale { .. }
        | ScreenSpaceReflectionQuery::Vec3Normalize { .. }
        | ScreenSpaceReflectionQuery::Reflect { .. } => {
            ScreenSpaceReflectionResult::Vector(unlane(raw.vec_r))
        }
        ScreenSpaceReflectionQuery::Project { .. } => ScreenSpaceReflectionResult::Projection {
            on_screen: raw.flag == CODE_FLAG,
            uv: Vec2::new(raw.vec_r[0], raw.vec_r[1]),
        },
        ScreenSpaceReflectionQuery::FinalizeHit { .. } => {
            ScreenSpaceReflectionResult::Hit(SsrHit {
                uv: Vec2::new(raw.vec_r[0], raw.vec_r[1]),
                confidence: raw.scalar,
                hit_distance: raw.hit_distance,
                hit: raw.flag == CODE_FLAG,
            })
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

/// A compiled, reusable screen-space-reflection compute pipeline, twinning the
/// `CPU` golden
/// [`screen_space_reflection`](prism_render_architecture::particle::screen_space_reflection).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::screen_space_reflection`。
pub struct GpuScreenSpaceReflection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuScreenSpaceReflection {
    /// Compiles the screen-space-reflection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuScreenSpaceReflection {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_screen_space_reflection"),
            source: ShaderSource::Wgsl(SCREEN_SPACE_REFLECTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_screen_space_reflection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_screen_space_reflection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_screen_space_reflection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuScreenSpaceReflection {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`ScreenSpaceReflectionResult`] per input, in order.
    ///
    /// The results match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ScreenSpaceReflectionQuery],
    ) -> Vec<ScreenSpaceReflectionResult> {
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
            label: Some("prism_volumetric_screen_space_reflection_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_screen_space_reflection_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_screen_space_reflection_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_screen_space_reflection_bind_group"),
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
            label: Some("prism_volumetric_screen_space_reflection_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_screen_space_reflection_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_screen_space_reflection_pass"),
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
