//! `wgpu` compute twin of the particle temporal-reprojection primitives
//! ([`temporal_reprojection`](prism_render_architecture::particle::temporal_reprojection),
//! design section 21), exposed one pure-numeric function at a time.
//!
//! Where the sibling
//! [`temporal_reproject`](crate::temporal_reproject) twin runs the whole
//! history-resolve chain end to end (one thread resolves one pixel), this module
//! takes the complementary *unit-test* view: a single compute kernel with an
//! **operation-code dispatch** evaluates exactly one golden function per lane, so
//! a parity test can pin each building block —
//! [`Vec2`](prism_render_architecture::particle::temporal_reprojection::Vec2)
//! algebra, the cubic
//! [`smoothstep`](prism_render_architecture::particle::temporal_reprojection::smoothstep)
//! polynomial,
//! [`reproject_uv`](prism_render_architecture::particle::temporal_reprojection::reproject_uv),
//! [`is_on_screen`](prism_render_architecture::particle::temporal_reprojection::is_on_screen),
//! [`depth_confidence`](prism_render_architecture::particle::temporal_reprojection::depth_confidence),
//! [`velocity_confidence`](prism_render_architecture::particle::temporal_reprojection::velocity_confidence),
//! [`history_valid`](prism_render_architecture::particle::temporal_reprojection::history_valid),
//! [`neighborhood_box`](prism_render_architecture::particle::temporal_reprojection::neighborhood_box),
//! [`AabbRgba::widened`](prism_render_architecture::particle::temporal_reprojection::AabbRgba::widened),
//! [`neighborhood_clamp`](prism_render_architecture::particle::temporal_reprojection::neighborhood_clamp),
//! [`rgb_to_ycocg`](prism_render_architecture::particle::temporal_reprojection::rgb_to_ycocg),
//! [`ycocg_to_rgb`](prism_render_architecture::particle::temporal_reprojection::ycocg_to_rgb),
//! [`clip_history_ycocg`](prism_render_architecture::particle::temporal_reprojection::clip_history_ycocg),
//! [`history_weight`](prism_render_architecture::particle::temporal_reprojection::history_weight)
//! and
//! [`blend_history`](prism_render_architecture::particle::temporal_reprojection::blend_history)
//! — in isolation on a real device, not merely the fused final colour.
//!
//! # Why op-code dispatch
//!
//! The golden module is a flat library of small, independently testable numeric
//! functions. Rather than compile one pipeline per function, every
//! [`TemporalReprojectionQuery`] carries an operation code, and the single
//! `solve` kernel branches on it (`if` / `else if` ladder on an unsigned code,
//! an exact integer compare). One thread handles one query; the batch may freely
//! mix operations. This keeps a single shader module and bind-group layout while
//! still surfacing each primitive to the parity suite.
//!
//! # What is twinned
//!
//! All of the pure-numeric functions listed above, including the private
//! `clip_toward` axis clip reached through
//! [`clip_history_ycocg`](prism_render_architecture::particle::temporal_reprojection::clip_history_ycocg).
//! The fixed `9`-sample `3x3` neighbourhood is uploaded inline with each query.
//!
//! # Left on the host (not twinned)
//!
//! The `std430` byte packing —
//! [`ReprojectionParams::to_std430`](prism_render_architecture::particle::temporal_reprojection::ReprojectionParams::to_std430)
//! and
//! [`pack_params_std430`](prism_render_architecture::particle::temporal_reprojection::pack_params_std430)
//! — is pure host serialization with no on-device counterpart, so it stays on
//! the host; this twin consumes [`ReprojectionParams`] scalars directly. The
//! generation of the `3x3` neighbourhood samples is likewise an upstream host
//! responsibility; the kernel receives the nine samples ready-made.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `+ - * /` and one `sqrt` for the vector length — with no
//! `exp`, `pow`, `sin` / `cos`, the built-in `smoothstep`, or any optional
//! device feature, so it runs unmodified on Metal, Vulkan and DX12. The golden
//! `smoothstep` is hand-expanded as the cubic `x*x*(3 - 2x)` (named
//! `smoothstep_poly` to avoid the forbidden built-in), exactly as the reference.
//!
//! # Correctness model
//!
//! Every twinned function is fixed closed-form algebra, so `CPU` and `GPU`
//! evaluate the same expression. They are not bit-exact in general: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate (the `YCoCg`
//! combinations, the clip crossing fraction, the `smoothstep` cubic, the lerp),
//! perturbing the low mantissa bits by a few units in the last place. The parity
//! test therefore asserts an absolute-or-relative tolerance on continuous values
//! and an *exact* equality on the discrete on-screen flag.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::temporal_reprojection`;
//! standard `TAA` / temporal-accumulation primitives (reproject, disocclusion /
//! velocity confidence, `YCoCg` neighbourhood clip, confidence blend) plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::temporal_reprojection::{
    blend_history, clip_history_ycocg, depth_confidence, history_valid, history_weight,
    is_on_screen, neighborhood_box, neighborhood_clamp, reproject_uv, rgb_to_ycocg, smoothstep,
    velocity_confidence, ycocg_to_rgb, AabbRgba, ReprojectionParams, Rgba, Vec2,
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

/// Number of colour taps in the current-frame neighbourhood window: a `3x3`
/// box, centre at index four (row-major), matching the reference `[Rgba; 9]`.
const TAPS: usize = 9;

// Operation codes shared by the host encoder and the `solve` kernel. Each tags
// one golden function; the kernel branches on the code with an exact integer
// compare.
const OP_VEC2_MINUS: u32 = 0;
const OP_VEC2_SCALE: u32 = 1;
const OP_VEC2_LENGTH_SQUARED: u32 = 2;
const OP_VEC2_LENGTH: u32 = 3;
const OP_VEC2_NORMALIZE: u32 = 4;
const OP_REPROJECT: u32 = 5;
const OP_ON_SCREEN: u32 = 6;
const OP_SMOOTHSTEP: u32 = 7;
const OP_DEPTH_CONFIDENCE: u32 = 8;
const OP_VELOCITY_CONFIDENCE: u32 = 9;
const OP_HISTORY_VALID: u32 = 10;
const OP_NEIGHBORHOOD_BOX: u32 = 11;
const OP_WIDENED: u32 = 12;
const OP_NEIGHBORHOOD_CLAMP: u32 = 13;
const OP_RGB_TO_YCOCG: u32 = 14;
const OP_YCOCG_TO_RGB: u32 = 15;
const OP_CLIP_HISTORY: u32 = 16;
const OP_HISTORY_WEIGHT: u32 = 17;
const OP_BLEND: u32 = 18;

/// The portable core-`WGSL` temporal-reprojection primitive kernel, embedded
/// inline so the twin ships as a single source file. One thread evaluates one
/// query, branching on its operation code; see the module documentation for the
/// op-dispatch rationale.
const TEMPORAL_REPROJECTION_WGSL: &str = r#"
// Temporal-reprojection primitive twin: one thread per query evaluates a single
// golden function selected by an operation code. It mirrors the CPU golden
// `particle::temporal_reprojection`, uses only the portable core-WGSL subset
// (min/max/clamp/abs and + - * / plus one sqrt for the vector length) and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard TAA / temporal-accumulation primitives; no third-party
// engine source or derived code.

struct Params {
    // Number of valid lanes in the batch, one thread each.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query's packed inputs. 224-byte std430 stride matching the host GpuQuery.
struct Query {
    // Operation code selecting the golden function.
    op: u32,
    qpad0: u32,
    qpad1: u32,
    qpad2: u32,
    // Reprojection parameters: x=max_history_weight, y=depth_reject_relative,
    // z=velocity_reject_uv, w=clamp_widen (zero when unused).
    params: vec4<f32>,
    // General operand slots; meaning depends on the operation code.
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    // The current frame's 3x3 colour neighbourhood (row-major, centre at four).
    samples: array<vec4<f32>, 9>,
}

// One query's packed outputs. 48-byte std430 stride matching the host GpuResult.
struct Res {
    // Primary vector/scalar/colour result (scalar in v0.x).
    v0: vec4<f32>,
    // Secondary result (the upper AABB bound for box ops).
    v1: vec4<f32>,
    // Discrete result code (the on-screen flag: 1 inside, 0 outside).
    code: u32,
    rpad0: u32,
    rpad1: u32,
    rpad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// Floating-point comparison tolerance, matching the reference CMP_EPS.
const CMP_EPS: f32 = 1e-6;
// Floor on the depth denominator, matching the reference DEPTH_EPS.
const DEPTH_EPS: f32 = 1e-6;
// Smallest length treated as non-zero when normalizing, matching EPS_LEN.
const EPS_LEN: f32 = 1e-12;
// Number of neighbourhood taps per query (3x3), matching the reference window.
const TAPS: u32 = 9u;

const OP_VEC2_MINUS: u32 = 0u;
const OP_VEC2_SCALE: u32 = 1u;
const OP_VEC2_LENGTH_SQUARED: u32 = 2u;
const OP_VEC2_LENGTH: u32 = 3u;
const OP_VEC2_NORMALIZE: u32 = 4u;
const OP_REPROJECT: u32 = 5u;
const OP_ON_SCREEN: u32 = 6u;
const OP_SMOOTHSTEP: u32 = 7u;
const OP_DEPTH_CONFIDENCE: u32 = 8u;
const OP_VELOCITY_CONFIDENCE: u32 = 9u;
const OP_HISTORY_VALID: u32 = 10u;
const OP_NEIGHBORHOOD_BOX: u32 = 11u;
const OP_WIDENED: u32 = 12u;
const OP_NEIGHBORHOOD_CLAMP: u32 = 13u;
const OP_RGB_TO_YCOCG: u32 = 14u;
const OP_YCOCG_TO_RGB: u32 = 15u;
const OP_CLIP_HISTORY: u32 = 16u;
const OP_HISTORY_WEIGHT: u32 = 17u;
const OP_BLEND: u32 = 18u;

// Cubic smoothstep t*t*(3 - 2t) clamped to [0, 1], the hand-expanded form of the
// reference `smoothstep` (the WGSL built-in `smoothstep` is forbidden). Inputs
// outside [0, 1] saturate to the endpoints.
fn smoothstep_poly(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

// Whether a reprojected UV lands inside the inclusive [0, 1] screen rectangle,
// matching the reference `is_on_screen`. A NaN component fails every compare and
// is treated as off-screen, exactly as the reference range `contains` does.
fn is_on_screen(uv: vec2<f32>) -> bool {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

// Depth-based history confidence in [0, 1], matching the reference
// `depth_confidence`.
fn depth_confidence(current_depth: f32, history_depth: f32, reject_relative: f32) -> f32 {
    let denom = max(abs(current_depth), DEPTH_EPS);
    let rel = abs(current_depth - history_depth) / denom;
    let thr = max(reject_relative, CMP_EPS);
    if (rel >= thr) {
        return 0.0;
    }
    return 1.0 - smoothstep_poly(rel / thr);
}

// Velocity-based history confidence in [0, 1], matching the reference
// `velocity_confidence`. A non-positive reject limit disables the penalty.
fn velocity_confidence(motion: vec2<f32>, reject_uv: f32) -> f32 {
    let limit = max(reject_uv, 0.0);
    if (limit <= CMP_EPS) {
        return 1.0;
    }
    let speed = sqrt(motion.x * motion.x + motion.y * motion.y);
    return clamp(1.0 - smoothstep_poly(speed / limit), 0.0, 1.0);
}

// Combined history confidence in [0, 1], matching the reference `history_valid`:
// zero off-screen, else depth confidence times velocity confidence, clamped.
fn history_valid(
    history_uv: vec2<f32>,
    current_depth: f32,
    history_depth: f32,
    motion: vec2<f32>,
    depth_reject: f32,
    vel_reject: f32,
) -> f32 {
    if (!is_on_screen(history_uv)) {
        return 0.0;
    }
    let depth = depth_confidence(current_depth, history_depth, depth_reject);
    let velocity = velocity_confidence(motion, vel_reject);
    return clamp(depth * velocity, 0.0, 1.0);
}

// Linear RGB to YCoCg (alpha ignored), matching the reference `rgb_to_ycocg`.
fn rgb_to_ycocg(color: vec4<f32>) -> vec3<f32> {
    let r = color.x;
    let g = color.y;
    let b = color.z;
    let y = r * 0.25 + g * 0.5 + b * 0.25;
    let co = r * 0.5 - b * 0.5;
    let cg = -r * 0.25 + g * 0.5 - b * 0.25;
    return vec3<f32>(y, co, cg);
}

// Inverse of `rgb_to_ycocg`, carrying alpha through unchanged, matching the
// reference `ycocg_to_rgb`.
fn ycocg_to_rgb(ycocg: vec3<f32>, alpha: f32) -> vec4<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    return vec4<f32>(r, g, b, alpha);
}

// Tightens the retained fraction `t` for one axis of the YCoCg clip, matching
// one iteration of the reference `clip_toward`. An axis whose component barely
// moves (within CMP_EPS) imposes no constraint, so a vanishing extent never
// divides by zero.
fn clip_axis(t: f32, lo: f32, hi: f32, qa: f32, pa: f32) -> f32 {
    let delta = pa - qa;
    if (delta > CMP_EPS) {
        return min(t, (hi - qa) / delta);
    } else if (delta < -CMP_EPS) {
        return min(t, (lo - qa) / delta);
    }
    return t;
}

// Moves point `p` toward point `q` (assumed inside the box) until it lies on or
// inside the [box_min, box_max] AABB, matching the reference `clip_toward`.
fn clip_toward(box_min: vec3<f32>, box_max: vec3<f32>, q: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    var t = 1.0;
    t = clip_axis(t, box_min.x, box_max.x, q.x, p.x);
    t = clip_axis(t, box_min.y, box_max.y, q.y, p.y);
    t = clip_axis(t, box_min.z, box_max.z, q.z, p.z);
    t = clamp(t, 0.0, 1.0);
    return vec3<f32>(
        q.x + t * (p.x - q.x),
        q.y + t * (p.y - q.y),
        q.z + t * (p.z - q.z),
    );
}

// History blend weight in [0, 1], matching the reference `history_weight`.
fn history_weight(confidence: f32, max_hw: f32) -> f32 {
    return clamp(clamp(confidence, 0.0, 1.0) * max_hw, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let q = queries[i];

    var out: Res;
    out.v0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.v1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.code = 0u;
    out.rpad0 = 0u;
    out.rpad1 = 0u;
    out.rpad2 = 0u;

    let op = q.op;
    if (op == OP_VEC2_MINUS) {
        out.v0 = vec4<f32>(q.a.x - q.b.x, q.a.y - q.b.y, 0.0, 0.0);
    } else if (op == OP_VEC2_SCALE) {
        out.v0 = vec4<f32>(q.a.x * q.c.x, q.a.y * q.c.x, 0.0, 0.0);
    } else if (op == OP_VEC2_LENGTH_SQUARED) {
        out.v0 = vec4<f32>(q.a.x * q.a.x + q.a.y * q.a.y, 0.0, 0.0, 0.0);
    } else if (op == OP_VEC2_LENGTH) {
        out.v0 = vec4<f32>(sqrt(q.a.x * q.a.x + q.a.y * q.a.y), 0.0, 0.0, 0.0);
    } else if (op == OP_VEC2_NORMALIZE) {
        let len = sqrt(q.a.x * q.a.x + q.a.y * q.a.y);
        if (len <= EPS_LEN) {
            out.v0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        } else {
            let inv = 1.0 / len;
            out.v0 = vec4<f32>(q.a.x * inv, q.a.y * inv, 0.0, 0.0);
        }
    } else if (op == OP_REPROJECT) {
        out.v0 = vec4<f32>(q.a.x - q.b.x, q.a.y - q.b.y, 0.0, 0.0);
    } else if (op == OP_ON_SCREEN) {
        if (is_on_screen(vec2<f32>(q.a.x, q.a.y))) {
            out.code = 1u;
        } else {
            out.code = 0u;
        }
    } else if (op == OP_SMOOTHSTEP) {
        out.v0 = vec4<f32>(smoothstep_poly(q.c.x), 0.0, 0.0, 0.0);
    } else if (op == OP_DEPTH_CONFIDENCE) {
        out.v0 = vec4<f32>(depth_confidence(q.c.x, q.c.y, q.c.z), 0.0, 0.0, 0.0);
    } else if (op == OP_VELOCITY_CONFIDENCE) {
        out.v0 = vec4<f32>(velocity_confidence(vec2<f32>(q.a.x, q.a.y), q.c.x), 0.0, 0.0, 0.0);
    } else if (op == OP_HISTORY_VALID) {
        let hv = history_valid(
            vec2<f32>(q.a.x, q.a.y),
            q.c.x,
            q.c.y,
            vec2<f32>(q.b.x, q.b.y),
            q.params.y,
            q.params.z,
        );
        out.v0 = vec4<f32>(hv, 0.0, 0.0, 0.0);
    } else if (op == OP_NEIGHBORHOOD_BOX) {
        var lo = q.samples[0];
        var hi = q.samples[0];
        for (var k = 1u; k < TAPS; k = k + 1u) {
            lo = min(lo, q.samples[k]);
            hi = max(hi, q.samples[k]);
        }
        out.v0 = lo;
        out.v1 = hi;
    } else if (op == OP_WIDENED) {
        let factor = 1.0 + max(q.c.x, 0.0);
        let centre = (q.a + q.b) * 0.5;
        let half_extent = (q.b - q.a) * 0.5 * factor;
        out.v0 = centre - half_extent;
        out.v1 = centre + half_extent;
    } else if (op == OP_NEIGHBORHOOD_CLAMP) {
        var lo = q.samples[0];
        var hi = q.samples[0];
        for (var k = 1u; k < TAPS; k = k + 1u) {
            lo = min(lo, q.samples[k]);
            hi = max(hi, q.samples[k]);
        }
        let factor = 1.0 + max(q.params.w, 0.0);
        let centre = (lo + hi) * 0.5;
        let half_extent = (hi - lo) * 0.5 * factor;
        lo = centre - half_extent;
        hi = centre + half_extent;
        out.v0 = clamp(q.a, lo, hi);
    } else if (op == OP_RGB_TO_YCOCG) {
        let yc = rgb_to_ycocg(q.a);
        out.v0 = vec4<f32>(yc.x, yc.y, yc.z, 0.0);
    } else if (op == OP_YCOCG_TO_RGB) {
        out.v0 = ycocg_to_rgb(vec3<f32>(q.a.x, q.a.y, q.a.z), q.c.x);
    } else if (op == OP_CLIP_HISTORY) {
        let qy = rgb_to_ycocg(q.a);
        let py = rgb_to_ycocg(q.b);
        var lo = rgb_to_ycocg(q.samples[0]);
        var hi = lo;
        for (var k = 1u; k < TAPS; k = k + 1u) {
            let value = rgb_to_ycocg(q.samples[k]);
            lo = min(lo, value);
            hi = max(hi, value);
        }
        let factor = 1.0 + max(q.params.w, 0.0);
        let centre = (lo + hi) * 0.5;
        let half_extent = (hi - lo) * 0.5 * factor;
        lo = centre - half_extent;
        hi = centre + half_extent;
        let clipped = clip_toward(lo, hi, qy, py);
        out.v0 = ycocg_to_rgb(clipped, q.b.w);
    } else if (op == OP_HISTORY_WEIGHT) {
        out.v0 = vec4<f32>(history_weight(q.c.x, q.params.x), 0.0, 0.0, 0.0);
    } else if (op == OP_BLEND) {
        let w = history_weight(q.c.x, q.params.x);
        out.v0 = q.a * (1.0 - w) + q.b * w;
    }

    results[i] = out;
}
"#;

/// One temporal-reprojection primitive query: a tagged request to evaluate a
/// single golden function on the device.
///
/// Each variant names one function of the `CPU` golden
/// [`temporal_reprojection`](prism_render_architecture::particle::temporal_reprojection)
/// and carries just that function's operands; the `9`-sample `3x3` neighbourhood
/// is supplied inline where needed. Holds `f32` operands, so it derives only
/// [`Clone`], [`Copy`], [`Debug`] and [`PartialEq`] (no [`Eq`] / [`Hash`]).
/// Provenance: query tagging for the `temporal_reprojection` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemporalReprojectionQuery {
    /// Component-wise subtraction `a - b`, mirroring
    /// [`Vec2::minus`](prism_render_architecture::particle::temporal_reprojection::Vec2::minus).
    Vec2Minus {
        /// The minuend.
        a: Vec2,
        /// The subtrahend.
        b: Vec2,
    },
    /// Uniform scale `v * s`, mirroring
    /// [`Vec2::scale`](prism_render_architecture::particle::temporal_reprojection::Vec2::scale).
    Vec2Scale {
        /// The vector scaled.
        v: Vec2,
        /// The scalar factor.
        s: f32,
    },
    /// Squared length, mirroring
    /// [`Vec2::length_squared`](prism_render_architecture::particle::temporal_reprojection::Vec2::length_squared).
    Vec2LengthSquared {
        /// The vector measured.
        v: Vec2,
    },
    /// Euclidean length, mirroring
    /// [`Vec2::length`](prism_render_architecture::particle::temporal_reprojection::Vec2::length).
    Vec2Length {
        /// The vector measured.
        v: Vec2,
    },
    /// Unit direction or zero, mirroring
    /// [`Vec2::normalize_or_zero`](prism_render_architecture::particle::temporal_reprojection::Vec2::normalize_or_zero).
    Vec2Normalize {
        /// The vector normalized.
        v: Vec2,
    },
    /// History `UV` reprojection, mirroring
    /// [`reproject_uv`](prism_render_architecture::particle::temporal_reprojection::reproject_uv).
    Reproject {
        /// The current pixel `UV`.
        current_uv: Vec2,
        /// The screen-space motion vector (`current - previous`).
        motion_uv: Vec2,
    },
    /// On-screen test, mirroring
    /// [`is_on_screen`](prism_render_architecture::particle::temporal_reprojection::is_on_screen).
    OnScreen {
        /// The `UV` tested against the inclusive `[0, 1]` rectangle.
        uv: Vec2,
    },
    /// Cubic `smoothstep`, mirroring
    /// [`smoothstep`](prism_render_architecture::particle::temporal_reprojection::smoothstep).
    Smoothstep {
        /// The interpolation parameter (saturated to `[0, 1]`).
        t: f32,
    },
    /// Depth-disocclusion confidence, mirroring
    /// [`depth_confidence`](prism_render_architecture::particle::temporal_reprojection::depth_confidence).
    DepthConfidence {
        /// The current surface depth.
        current_depth: f32,
        /// The history surface depth.
        history_depth: f32,
        /// The relative-difference rejection threshold.
        reject_relative: f32,
    },
    /// Velocity confidence, mirroring
    /// [`velocity_confidence`](prism_render_architecture::particle::temporal_reprojection::velocity_confidence).
    VelocityConfidence {
        /// The screen-space motion vector.
        motion_uv: Vec2,
        /// The motion magnitude at which confidence fully fades.
        reject_uv: f32,
    },
    /// Combined history confidence, mirroring
    /// [`history_valid`](prism_render_architecture::particle::temporal_reprojection::history_valid).
    HistoryValid {
        /// The reprojected history `UV`.
        history_uv: Vec2,
        /// The current surface depth.
        current_depth: f32,
        /// The history surface depth.
        history_depth: f32,
        /// The screen-space motion vector.
        motion_uv: Vec2,
        /// The validity-clamped reprojection parameters.
        params: ReprojectionParams,
    },
    /// Neighbourhood `AABB`, mirroring
    /// [`neighborhood_box`](prism_render_architecture::particle::temporal_reprojection::neighborhood_box).
    NeighborhoodBox {
        /// The `3x3` colour window (row-major, centre at index four).
        samples: [Rgba; TAPS],
    },
    /// Symmetric box widening, mirroring
    /// [`AabbRgba::widened`](prism_render_architecture::particle::temporal_reprojection::AabbRgba::widened).
    Widened {
        /// The box lower bound.
        min: Rgba,
        /// The box upper bound.
        max: Rgba,
        /// The non-negative fractional expansion.
        extra: f32,
    },
    /// Neighbourhood min/max clamp, mirroring
    /// [`neighborhood_clamp`](prism_render_architecture::particle::temporal_reprojection::neighborhood_clamp).
    NeighborhoodClamp {
        /// The reprojected history colour clamped into the box.
        history: Rgba,
        /// The `3x3` colour window.
        samples: [Rgba; TAPS],
        /// The validity-clamped reprojection parameters (uses `clamp_widen`).
        params: ReprojectionParams,
    },
    /// Linear `RGB` to `YCoCg`, mirroring
    /// [`rgb_to_ycocg`](prism_render_architecture::particle::temporal_reprojection::rgb_to_ycocg).
    RgbToYcocg {
        /// The linear `RGBA` colour (alpha ignored).
        color: Rgba,
    },
    /// `YCoCg` to linear `RGB`, mirroring
    /// [`ycocg_to_rgb`](prism_render_architecture::particle::temporal_reprojection::ycocg_to_rgb).
    YcocgToRgb {
        /// The `[Y, Co, Cg]` triple.
        ycocg: [f32; 3],
        /// The alpha carried through unchanged.
        alpha: f32,
    },
    /// `YCoCg`-space line clip, mirroring
    /// [`clip_history_ycocg`](prism_render_architecture::particle::temporal_reprojection::clip_history_ycocg).
    ClipHistory {
        /// The current sample the clip moves toward.
        current: Rgba,
        /// The reprojected history clipped into the box.
        history: Rgba,
        /// The `3x3` colour window defining the box.
        samples: [Rgba; TAPS],
        /// The validity-clamped reprojection parameters (uses `clamp_widen`).
        params: ReprojectionParams,
    },
    /// History blend weight, mirroring
    /// [`history_weight`](prism_render_architecture::particle::temporal_reprojection::history_weight).
    HistoryWeight {
        /// The confidence in `[0, 1]`.
        confidence: f32,
        /// The validity-clamped reprojection parameters (uses `max_history_weight`).
        params: ReprojectionParams,
    },
    /// Confidence blend, mirroring
    /// [`blend_history`](prism_render_architecture::particle::temporal_reprojection::blend_history).
    Blend {
        /// The current sample.
        current: Rgba,
        /// The already-constrained history.
        clamped_history: Rgba,
        /// The confidence in `[0, 1]`.
        confidence: f32,
        /// The validity-clamped reprojection parameters.
        params: ReprojectionParams,
    },
}

/// The resolved verdict for one [`TemporalReprojectionQuery`], the host-side
/// mirror of a kernel result lane.
///
/// Carries `f32` payloads, so it derives [`Clone`], [`Copy`], [`Debug`] and
/// [`PartialEq`] (no [`Eq`] / [`Hash`]). Provenance: result decoding for the
/// `temporal_reprojection` twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemporalReprojectionResult {
    /// A two-component vector result (`Vec2` ops and the reprojection).
    Vec2 {
        /// The `[x, y]` components.
        v: [f32; 2],
    },
    /// A scalar result (lengths, confidences, `smoothstep`, blend weight).
    Scalar {
        /// The scalar value.
        value: f32,
    },
    /// The on-screen boolean verdict.
    OnScreen {
        /// Whether the `UV` lies inside the inclusive `[0, 1]` rectangle.
        on: bool,
    },
    /// A `[Y, Co, Cg]` triple (the `RGB`-to-`YCoCg` transform).
    Ycocg {
        /// The `[Y, Co, Cg]` components.
        ycocg: [f32; 3],
    },
    /// A linear `RGBA` colour result (clamp, inverse transform, clip, blend).
    Color {
        /// The `RGBA` channels.
        color: [f32; 4],
    },
    /// A per-channel `RGBA` `AABB` (the neighbourhood box and its widening).
    Aabb {
        /// The per-channel lower bound.
        min: [f32; 4],
        /// The per-channel upper bound.
        max: [f32; 4],
    },
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// Every arm calls the matching public golden function of
/// [`temporal_reprojection`](prism_render_architecture::particle::temporal_reprojection)
/// directly; the private `clip_toward` axis clip is reached through
/// [`clip_history_ycocg`](prism_render_architecture::particle::temporal_reprojection::clip_history_ycocg).
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::temporal_reprojection`.
#[must_use]
pub fn cpu_reference(query: &TemporalReprojectionQuery) -> TemporalReprojectionResult {
    match query {
        TemporalReprojectionQuery::Vec2Minus { a, b } => {
            let r = a.minus(*b);
            TemporalReprojectionResult::Vec2 { v: [r.x, r.y] }
        }
        TemporalReprojectionQuery::Vec2Scale { v, s } => {
            let r = v.scale(*s);
            TemporalReprojectionResult::Vec2 { v: [r.x, r.y] }
        }
        TemporalReprojectionQuery::Vec2LengthSquared { v } => TemporalReprojectionResult::Scalar {
            value: v.length_squared(),
        },
        TemporalReprojectionQuery::Vec2Length { v } => {
            TemporalReprojectionResult::Scalar { value: v.length() }
        }
        TemporalReprojectionQuery::Vec2Normalize { v } => {
            let r = v.normalize_or_zero();
            TemporalReprojectionResult::Vec2 { v: [r.x, r.y] }
        }
        TemporalReprojectionQuery::Reproject {
            current_uv,
            motion_uv,
        } => {
            let r = reproject_uv(*current_uv, *motion_uv);
            TemporalReprojectionResult::Vec2 { v: [r.x, r.y] }
        }
        TemporalReprojectionQuery::OnScreen { uv } => TemporalReprojectionResult::OnScreen {
            on: is_on_screen(*uv),
        },
        TemporalReprojectionQuery::Smoothstep { t } => TemporalReprojectionResult::Scalar {
            value: smoothstep(*t),
        },
        TemporalReprojectionQuery::DepthConfidence {
            current_depth,
            history_depth,
            reject_relative,
        } => TemporalReprojectionResult::Scalar {
            value: depth_confidence(*current_depth, *history_depth, *reject_relative),
        },
        TemporalReprojectionQuery::VelocityConfidence {
            motion_uv,
            reject_uv,
        } => TemporalReprojectionResult::Scalar {
            value: velocity_confidence(*motion_uv, *reject_uv),
        },
        TemporalReprojectionQuery::HistoryValid {
            history_uv,
            current_depth,
            history_depth,
            motion_uv,
            params,
        } => TemporalReprojectionResult::Scalar {
            value: history_valid(
                *history_uv,
                *current_depth,
                *history_depth,
                *motion_uv,
                *params,
            ),
        },
        TemporalReprojectionQuery::NeighborhoodBox { samples } => {
            let bounds = neighborhood_box(samples);
            TemporalReprojectionResult::Aabb {
                min: bounds.min,
                max: bounds.max,
            }
        }
        TemporalReprojectionQuery::Widened { min, max, extra } => {
            let bounds = AabbRgba {
                min: *min,
                max: *max,
            }
            .widened(*extra);
            TemporalReprojectionResult::Aabb {
                min: bounds.min,
                max: bounds.max,
            }
        }
        TemporalReprojectionQuery::NeighborhoodClamp {
            history,
            samples,
            params,
        } => TemporalReprojectionResult::Color {
            color: neighborhood_clamp(*history, samples, *params),
        },
        TemporalReprojectionQuery::RgbToYcocg { color } => TemporalReprojectionResult::Ycocg {
            ycocg: rgb_to_ycocg(*color),
        },
        TemporalReprojectionQuery::YcocgToRgb { ycocg, alpha } => {
            TemporalReprojectionResult::Color {
                color: ycocg_to_rgb(*ycocg, *alpha),
            }
        }
        TemporalReprojectionQuery::ClipHistory {
            current,
            history,
            samples,
            params,
        } => TemporalReprojectionResult::Color {
            color: clip_history_ycocg(*current, *history, samples, *params),
        },
        TemporalReprojectionQuery::HistoryWeight { confidence, params } => {
            TemporalReprojectionResult::Scalar {
                value: history_weight(*confidence, *params),
            }
        }
        TemporalReprojectionQuery::Blend {
            current,
            clamped_history,
            confidence,
            params,
        } => TemporalReprojectionResult::Color {
            color: blend_history(*current, *clamped_history, *confidence, *params),
        },
    }
}

/// Uniform dispatch parameters. `repr(C)` `std430` layout matching `Params` in
/// [`TEMPORAL_REPROJECTION_WGSL`]: the lane count and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One query as uploaded. `224`-byte `std430` stride matching `Query` in the
/// shader: the op code and three pad words, the parameter `vec4`, three operand
/// `vec4`s and the nine inline neighbourhood samples.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    params: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    samples: [[f32; 4]; TAPS],
}

/// One result as read back. `48`-byte `std430` stride matching `Res` in the
/// shader: the primary and secondary vectors, the discrete code and three pad
/// words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    v0: [f32; 4],
    v1: [f32; 4],
    code: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Flattens a validity-clamped parameter block into the shader `vec4` order:
/// `max_history_weight`, `depth_reject_relative`, `velocity_reject_uv`,
/// `clamp_widen`.
fn params_array(params: &ReprojectionParams) -> [f32; 4] {
    [
        params.max_history_weight,
        params.depth_reject_relative,
        params.velocity_reject_uv,
        params.clamp_widen,
    ]
}

/// Encodes one query into its packed `GpuQuery`, placing each operand in the
/// slot the kernel reads for that operation code.
fn encode(query: &TemporalReprojectionQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    match query {
        TemporalReprojectionQuery::Vec2Minus { a, b } => {
            g.op = OP_VEC2_MINUS;
            g.a = [a.x, a.y, 0.0, 0.0];
            g.b = [b.x, b.y, 0.0, 0.0];
        }
        TemporalReprojectionQuery::Vec2Scale { v, s } => {
            g.op = OP_VEC2_SCALE;
            g.a = [v.x, v.y, 0.0, 0.0];
            g.c = [*s, 0.0, 0.0, 0.0];
        }
        TemporalReprojectionQuery::Vec2LengthSquared { v } => {
            g.op = OP_VEC2_LENGTH_SQUARED;
            g.a = [v.x, v.y, 0.0, 0.0];
        }
        TemporalReprojectionQuery::Vec2Length { v } => {
            g.op = OP_VEC2_LENGTH;
            g.a = [v.x, v.y, 0.0, 0.0];
        }
        TemporalReprojectionQuery::Vec2Normalize { v } => {
            g.op = OP_VEC2_NORMALIZE;
            g.a = [v.x, v.y, 0.0, 0.0];
        }
        TemporalReprojectionQuery::Reproject {
            current_uv,
            motion_uv,
        } => {
            g.op = OP_REPROJECT;
            g.a = [current_uv.x, current_uv.y, 0.0, 0.0];
            g.b = [motion_uv.x, motion_uv.y, 0.0, 0.0];
        }
        TemporalReprojectionQuery::OnScreen { uv } => {
            g.op = OP_ON_SCREEN;
            g.a = [uv.x, uv.y, 0.0, 0.0];
        }
        TemporalReprojectionQuery::Smoothstep { t } => {
            g.op = OP_SMOOTHSTEP;
            g.c = [*t, 0.0, 0.0, 0.0];
        }
        TemporalReprojectionQuery::DepthConfidence {
            current_depth,
            history_depth,
            reject_relative,
        } => {
            g.op = OP_DEPTH_CONFIDENCE;
            g.c = [*current_depth, *history_depth, *reject_relative, 0.0];
        }
        TemporalReprojectionQuery::VelocityConfidence {
            motion_uv,
            reject_uv,
        } => {
            g.op = OP_VELOCITY_CONFIDENCE;
            g.a = [motion_uv.x, motion_uv.y, 0.0, 0.0];
            g.c = [*reject_uv, 0.0, 0.0, 0.0];
        }
        TemporalReprojectionQuery::HistoryValid {
            history_uv,
            current_depth,
            history_depth,
            motion_uv,
            params,
        } => {
            g.op = OP_HISTORY_VALID;
            g.a = [history_uv.x, history_uv.y, 0.0, 0.0];
            g.b = [motion_uv.x, motion_uv.y, 0.0, 0.0];
            g.c = [*current_depth, *history_depth, 0.0, 0.0];
            g.params = params_array(params);
        }
        TemporalReprojectionQuery::NeighborhoodBox { samples } => {
            g.op = OP_NEIGHBORHOOD_BOX;
            g.samples = *samples;
        }
        TemporalReprojectionQuery::Widened { min, max, extra } => {
            g.op = OP_WIDENED;
            g.a = *min;
            g.b = *max;
            g.c = [*extra, 0.0, 0.0, 0.0];
        }
        TemporalReprojectionQuery::NeighborhoodClamp {
            history,
            samples,
            params,
        } => {
            g.op = OP_NEIGHBORHOOD_CLAMP;
            g.a = *history;
            g.samples = *samples;
            g.params = params_array(params);
        }
        TemporalReprojectionQuery::RgbToYcocg { color } => {
            g.op = OP_RGB_TO_YCOCG;
            g.a = *color;
        }
        TemporalReprojectionQuery::YcocgToRgb { ycocg, alpha } => {
            g.op = OP_YCOCG_TO_RGB;
            g.a = [ycocg[0], ycocg[1], ycocg[2], 0.0];
            g.c = [*alpha, 0.0, 0.0, 0.0];
        }
        TemporalReprojectionQuery::ClipHistory {
            current,
            history,
            samples,
            params,
        } => {
            g.op = OP_CLIP_HISTORY;
            g.a = *current;
            g.b = *history;
            g.samples = *samples;
            g.params = params_array(params);
        }
        TemporalReprojectionQuery::HistoryWeight { confidence, params } => {
            g.op = OP_HISTORY_WEIGHT;
            g.c = [*confidence, 0.0, 0.0, 0.0];
            g.params = params_array(params);
        }
        TemporalReprojectionQuery::Blend {
            current,
            clamped_history,
            confidence,
            params,
        } => {
            g.op = OP_BLEND;
            g.a = *current;
            g.b = *clamped_history;
            g.c = [*confidence, 0.0, 0.0, 0.0];
            g.params = params_array(params);
        }
    }
    g
}

/// Decodes one packed `GpuResult` back into the typed verdict for its originating
/// query variant.
fn decode(query: &TemporalReprojectionQuery, r: &GpuResult) -> TemporalReprojectionResult {
    match query {
        TemporalReprojectionQuery::Vec2Minus { .. }
        | TemporalReprojectionQuery::Vec2Scale { .. }
        | TemporalReprojectionQuery::Vec2Normalize { .. }
        | TemporalReprojectionQuery::Reproject { .. } => TemporalReprojectionResult::Vec2 {
            v: [r.v0[0], r.v0[1]],
        },
        TemporalReprojectionQuery::Vec2LengthSquared { .. }
        | TemporalReprojectionQuery::Vec2Length { .. }
        | TemporalReprojectionQuery::Smoothstep { .. }
        | TemporalReprojectionQuery::DepthConfidence { .. }
        | TemporalReprojectionQuery::VelocityConfidence { .. }
        | TemporalReprojectionQuery::HistoryValid { .. }
        | TemporalReprojectionQuery::HistoryWeight { .. } => {
            TemporalReprojectionResult::Scalar { value: r.v0[0] }
        }
        TemporalReprojectionQuery::OnScreen { .. } => {
            TemporalReprojectionResult::OnScreen { on: r.code == 1 }
        }
        TemporalReprojectionQuery::NeighborhoodBox { .. }
        | TemporalReprojectionQuery::Widened { .. } => TemporalReprojectionResult::Aabb {
            min: r.v0,
            max: r.v1,
        },
        TemporalReprojectionQuery::NeighborhoodClamp { .. }
        | TemporalReprojectionQuery::YcocgToRgb { .. }
        | TemporalReprojectionQuery::ClipHistory { .. }
        | TemporalReprojectionQuery::Blend { .. } => {
            TemporalReprojectionResult::Color { color: r.v0 }
        }
        TemporalReprojectionQuery::RgbToYcocg { .. } => TemporalReprojectionResult::Ycocg {
            ycocg: [r.v0[0], r.v0[1], r.v0[2]],
        },
    }
}

/// A compiled, reusable temporal-reprojection primitive-evaluation pipeline.
///
/// Provenance: `wgpu` compute twin of
/// `prism_render_architecture::particle::temporal_reprojection`.
pub struct GpuTemporalReprojection {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTemporalReprojection {
    /// Compiles the temporal-reprojection primitive kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required. Provenance: pipeline construction for the
    /// `temporal_reprojection` twin.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTemporalReprojection {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_temporal_reprojection"),
            source: ShaderSource::Wgsl(TEMPORAL_REPROJECTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_temporal_reprojection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_temporal_reprojection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_temporal_reprojection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTemporalReprojection {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one
    /// [`TemporalReprojectionResult`] per query in input order.
    ///
    /// Each lane reproduces the golden function its query names, to within the
    /// tolerance documented on this module (the on-screen flag is exact). An
    /// empty `queries` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return. Provenance: primitive
    /// evaluation for the `temporal_reprojection` twin.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[TemporalReprojectionQuery],
    ) -> Vec<TemporalReprojectionResult> {
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
            label: Some("prism_volumetric_temporal_reprojection_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_temporal_reprojection_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_temporal_reprojection_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_temporal_reprojection_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_temporal_reprojection_bind_group"),
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
            label: Some("prism_volumetric_temporal_reprojection_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_temporal_reprojection_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, in workgroups of 64 (the kernel's size).
            let groups = (queries.len() as u32).div_ceil(64);
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
