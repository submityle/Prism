//! `wgpu` compute twin of the parallax-occlusion `UV`-offset golden
//! ([`parallax_offset`](prism_render_architecture::particle::parallax_offset),
//! particle design §16-§21, the height-field ray-march to a sampled `UV`).
//!
//! The `CPU` golden
//! ([`parallax_offset`](prism_render_architecture::particle::parallax_offset))
//! turns a tangent-space view ray plus a height field into the `UV` a shader
//! should sample: a view-dependent layer count, a full-depth tangent-space
//! offset vector, a steep-parallax march, and a one-step secant refinement, plus
//! an optional contact self-shadow. Every shaped curve in that path is a `lerp`,
//! a rational `1 / max(cos, eps)` truncation, or the multiply-only `smoothstep`
//! `t^2 (3 - 2 t)`; the only floating primitive beyond `+ - * /` is one guarded
//! `sqrt` for view-ray normalization. That makes the per-element arithmetic port
//! cleanly to a portable core-`WGSL` kernel with no transcendental call.
//!
//! [`GpuParallaxOffset`] is the on-device twin: one thread per
//! [`ParallaxOffsetQuery`] reproduces the matching golden answer, so a passing
//! real-device parity test is direct evidence the ported kernel evaluates the
//! same polynomials and rationals the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Per query, the twin reproduces the golden's closed-form numeric cores: the
//! unit-interval clamp `clamp01`, the multiply-only `smoothstep01`
//! (`t^2 (3 - 2 t)`), a plain `lerp`, the tangent-space normalization
//! `normalize3` (`Option` mapped to a validity flag), the full-depth offset
//! vector `full_offset` (`height_scale / max(cos, eps)`),
//! [`ParallaxConfig::layer_count`], [`secant_refine`], one steep-parallax march
//! step (`cur_uv -= delta_uv`, `cur_layer_depth += layer_step`,
//! `surface_depth = 1 - clamp01(height)`, plus the penetration comparison), the
//! [`SteepHit`] bracket resolution ([`SteepHit::before`], [`SteepHit::after`],
//! [`SteepHit::refined_uv`]), one self-shadow march step (overlap running
//! maximum), and the final self-shadow resolve (the softness-scaled
//! `smoothstep01`).
//!
//! # What is not twinned
//!
//! The host-only, state-and-loop parts of the golden are deliberately left on
//! the `CPU`: the `u16` `min_layers` / `max_layers` bounds (the host widens them
//! to `u32` before downfeeding, since the kernel has no `u16`), the `u16`
//! integer hash `hash_u32` and the `u16`-to-unit-interval `lattice_height` that
//! seed the procedural relief, the `HeightField` `usize` indexing and
//! clamp-to-edge bilinear reconstruction (`cell_coord` and `sample`), the
//! variable-length steep and self-shadow march loops (the twin runs a single
//! step; the host owns the iteration and feeds each step's sampled height), the
//! `std430` packing `to_std430`, and the `Vec` batch `parallax_uv_batch`. Those
//! are list bookkeeping, integer addressing and a bounded loop, not the
//! per-element arithmetic a one-thread-per-element kernel is for.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `select`, `+ - * /` and one guarded `sqrt` — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no inverse trigonometry, no `smoothstep` builtin
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. The golden's `smoothstep` is expanded by hand as the Hermite
//! polynomial `t^2 (3 - 2 t)`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! guarded divide / `sqrt`, so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields, while the normalization validity and penetration flags are integer
//! codes compared for exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`parallax_offset`](prism_render_architecture::particle::parallax_offset); no
//! third-party engine source or derived code.

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

/// Op code for the unit-interval clamp `clamp01`.
const OP_CLAMP01: u32 = 0;
/// Op code for the multiply-only `smoothstep01`.
const OP_SMOOTHSTEP01: u32 = 1;
/// Op code for the plain linear interpolation `lerp`.
const OP_LERP: u32 = 2;
/// Op code for the tangent-space normalization `normalize3`.
const OP_NORMALIZE3: u32 = 3;
/// Op code for the full-depth tangent-space offset vector `full_offset`.
const OP_FULL_OFFSET: u32 = 4;
/// Op code for the view-dependent layer count `layer_count`.
const OP_LAYER_COUNT: u32 = 5;
/// Op code for the one-step secant refinement `secant_refine`.
const OP_SECANT_REFINE: u32 = 6;
/// Op code for one steep-parallax march step.
const OP_STEEP_STEP: u32 = 7;
/// Op code for the steep-hit bracket resolution.
const OP_STEEP_RESOLVE: u32 = 8;
/// Op code for one self-shadow march step.
const OP_SELF_SHADOW_STEP: u32 = 9;
/// Op code for the final self-shadow resolve.
const OP_SELF_SHADOW_RESOLVE: u32 = 10;

/// The portable core-`WGSL` parallax-occlusion kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve`
/// dispatches on an op code to mirror each golden
/// [`parallax_offset`](prism_render_architecture::particle::parallax_offset)
/// numeric core; see the module documentation for the algorithm.
const PARALLAX_OFFSET_WGSL: &str = r#"
// Parallax-offset twin: one thread per query reproduces the clamp, smoothstep,
// lerp, normalization, full-depth offset vector, view-dependent layer count,
// secant refinement, one steep march step, the steep-hit bracket resolution,
// one self-shadow march step and the final self-shadow resolve. It mirrors the
// CPU golden parallax_offset numeric cores routine for routine.
//
// Portability: only the core subset (abs, min, max, clamp, select, + - * / and
// one guarded sqrt) is used; no sin/cos/tan/exp/log/pow, no inverse
// trigonometry, no smoothstep builtin and no optional device feature, so it
// runs unmodified on Metal, Vulkan and DX12. The golden smoothstep is expanded
// by hand as t^2 (3 - 2 t).
//
// Provenance: twinned from this repository's particle::parallax_offset; no
// third-party engine source or derived code.

// Shared epsilon guarding divisions and the f32 magnitude comparisons that
// stand in for forbidden equality, matching the reference CMP_EPS.
const CMP_EPS: f32 = 1.0e-6;

const OP_CLAMP01: u32 = 0u;
const OP_SMOOTHSTEP01: u32 = 1u;
const OP_LERP: u32 = 2u;
const OP_NORMALIZE3: u32 = 3u;
const OP_FULL_OFFSET: u32 = 4u;
const OP_LAYER_COUNT: u32 = 5u;
const OP_SECANT_REFINE: u32 = 6u;
const OP_STEEP_STEP: u32 = 7u;
const OP_STEEP_RESOLVE: u32 = 8u;
const OP_SELF_SHADOW_STEP: u32 = 9u;
const OP_SELF_SHADOW_RESOLVE: u32 = 10u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Op code selecting the routine.
    op: u32,
    // Layer-count bounds, widened to u32 on the host (the kernel has no u16).
    min_layers: u32,
    max_layers: u32,
    pad0: u32,
    // Scalar parameters, meaning per op.
    s: array<f32, 8>,
    // Vector input A (view/light unit vector xyz, or prev/cur/ray uv in xy).
    a: array<f32, 4>,
    // Vector input B (hit uv or per-layer delta uv in xy).
    b: array<f32, 4>,
}

struct Result {
    // Validity / penetration flag.
    flag: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Scalar outputs (primary, secondary, tertiary) plus a pad lane.
    scalar: f32,
    scalar2: f32,
    scalar3: f32,
    pad3: f32,
    // Vector output (xy for a uv or offset, xyz for a normalized vector).
    vx: f32,
    vy: f32,
    vz: f32,
    pad4: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps a scalar to the closed unit interval, mirroring the reference clamp01.
fn clamp01(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

// Hermite smoothstep t^2 (3 - 2 t) on a clamped t, mirroring the reference
// smoothstep01; written out so no smoothstep builtin is used.
fn smoothstep01(t_in: f32) -> f32 {
    let t = clamp01(t_in);
    return t * t * (3.0 - 2.0 * t);
}

// Linear interpolation a + (b - a) * t, the plain lerp the reference uses.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var r: Result;
    r.flag = 0u;
    r.pad0 = 0u;
    r.pad1 = 0u;
    r.pad2 = 0u;
    r.scalar = 0.0;
    r.scalar2 = 0.0;
    r.scalar3 = 0.0;
    r.pad3 = 0.0;
    r.vx = 0.0;
    r.vy = 0.0;
    r.vz = 0.0;
    r.pad4 = 0.0;

    let op = q.op;

    if (op == OP_CLAMP01) {
        r.scalar = clamp01(q.s[0]);
    } else if (op == OP_SMOOTHSTEP01) {
        r.scalar = smoothstep01(q.s[0]);
    } else if (op == OP_LERP) {
        r.scalar = lerp(q.s[0], q.s[1], q.s[2]);
    } else if (op == OP_NORMALIZE3) {
        let vx = q.a[0];
        let vy = q.a[1];
        let vz = q.a[2];
        let len_sq = vx * vx + vy * vy + vz * vz;
        if (len_sq < CMP_EPS * CMP_EPS) {
            r.flag = 0u;
        } else {
            let inv = 1.0 / sqrt(len_sq);
            r.flag = 1u;
            r.vx = vx * inv;
            r.vy = vy * inv;
            r.vz = vz * inv;
        }
    } else if (op == OP_FULL_OFFSET) {
        // height_scale in s[0], view_unit in a.xyz; the 1 / max(z, eps)
        // truncation stands in for tan without a transcendental.
        let denom = max(q.a[2], CMP_EPS);
        let k = q.s[0] / denom;
        r.vx = q.a[0] * k;
        r.vy = q.a[1] * k;
    } else if (op == OP_LAYER_COUNT) {
        let min_f = f32(q.min_layers);
        let max_f = f32(q.max_layers);
        let grazing = clamp01(1.0 - clamp01(q.s[0]));
        let n = min_f + grazing * (max_f - min_f);
        r.scalar = max(n, 1.0);
    } else if (op == OP_SECANT_REFINE) {
        // prev_uv in a.xy, hit_uv in b.xy, before/after in s[0]/s[1].
        let before = q.s[0];
        let after = q.s[1];
        let denom = before - after;
        if (abs(denom) < CMP_EPS) {
            r.vx = q.b[0];
            r.vy = q.b[1];
        } else {
            let weight = clamp01(before / denom);
            r.vx = q.a[0] + (q.b[0] - q.a[0]) * weight;
            r.vy = q.a[1] + (q.b[1] - q.a[1]) * weight;
        }
    } else if (op == OP_STEEP_STEP) {
        // cur_uv in a.xy, delta_uv in b.xy, cur_layer_depth/layer_step/height
        // in s[0]/s[1]/s[2]. One step of the steep-parallax march.
        let next_uv_x = q.a[0] - q.b[0];
        let next_uv_y = q.a[1] - q.b[1];
        let next_layer_depth = q.s[0] + q.s[1];
        let next_surface_depth = 1.0 - clamp01(q.s[2]);
        r.vx = next_uv_x;
        r.vy = next_uv_y;
        r.scalar = next_layer_depth;
        r.scalar2 = next_surface_depth;
        r.flag = select(0u, 1u, next_layer_depth >= next_surface_depth);
    } else if (op == OP_STEEP_RESOLVE) {
        // prev_uv in a.xy, hit_uv in b.xy; prev_layer/prev_surface/hit_layer/
        // hit_surface in s[0..4]. SteepHit before/after plus the secant refine.
        let before = q.s[1] - q.s[0];
        let after = q.s[3] - q.s[2];
        r.scalar = before;
        r.scalar2 = after;
        let denom = before - after;
        if (abs(denom) < CMP_EPS) {
            r.vx = q.b[0];
            r.vy = q.b[1];
        } else {
            let weight = clamp01(before / denom);
            r.vx = q.a[0] + (q.b[0] - q.a[0]) * weight;
            r.vy = q.a[1] + (q.b[1] - q.a[1]) * weight;
        }
    } else if (op == OP_SELF_SHADOW_STEP) {
        // ray_uv in a.xy, delta_uv in b.xy; ray_depth/depth_step/height/
        // max_overlap in s[0..4]. One step of the self-shadow march.
        let new_ray_uv_x = q.a[0] + q.b[0];
        let new_ray_uv_y = q.a[1] + q.b[1];
        let new_ray_depth = q.s[0] - q.s[1];
        let surface_depth = 1.0 - clamp01(q.s[2]);
        let overlap = new_ray_depth - surface_depth;
        let new_max_overlap = max(q.s[3], overlap);
        r.vx = new_ray_uv_x;
        r.vy = new_ray_uv_y;
        r.scalar = new_ray_depth;
        r.scalar2 = surface_depth;
        r.scalar3 = new_max_overlap;
    } else {
        // OP_SELF_SHADOW_RESOLVE: max_overlap in s[0], softness in s[1].
        let max_overlap = q.s[0];
        let softness = q.s[1];
        if (max_overlap <= 0.0) {
            r.scalar = 1.0;
        } else {
            var shadow = 1.0;
            if (softness >= CMP_EPS) {
                shadow = smoothstep01(max_overlap / softness);
            }
            r.scalar = clamp01(1.0 - shadow);
        }
    }

    results[idx] = r;
}
"#;

/// One parallax-occlusion query: the numeric routine and its inputs.
///
/// Each variant mirrors one golden
/// [`parallax_offset`](prism_render_architecture::particle::parallax_offset)
/// numeric core. The march-step variants ([`ParallaxOffsetQuery::SteepStep`],
/// [`ParallaxOffsetQuery::SelfShadowStep`]) take the height already sampled by
/// the host for the stepped `UV`, since the host owns the variable-length march
/// loop and the twin runs only a single step's arithmetic.
///
/// Provenance: twinned from this repository's
/// [`parallax_offset`](prism_render_architecture::particle::parallax_offset); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParallaxOffsetQuery {
    /// Clamp a scalar into the closed unit interval `[0, 1]` (mirrors the
    /// reference private `clamp01`).
    Clamp01 {
        /// The scalar to clamp.
        x: f32,
    },
    /// Multiply-only `smoothstep01` `t^2 (3 - 2 t)` on a clamped `t` (mirrors
    /// the reference private `smoothstep01`).
    Smoothstep01 {
        /// The ramp parameter, clamped into `[0, 1]` first.
        t: f32,
    },
    /// Plain linear interpolation `a + (b - a) * t` (the reference `lerp`).
    Lerp {
        /// The value at `t = 0`.
        a: f32,
        /// The value at `t = 1`.
        b: f32,
        /// The interpolation parameter.
        t: f32,
    },
    /// Normalize a tangent-space vector, returning `None` for a near-zero input
    /// (mirrors the reference private `normalize3`).
    Normalize3 {
        /// The vector to normalize.
        v: [f32; 3],
    },
    /// The full-depth tangent-space offset vector `height_scale / max(z, eps)`
    /// along the view direction (mirrors the reference private `full_offset`).
    FullOffset {
        /// The normalized tangent-space view direction (`z` along the normal).
        view_unit: [f32; 3],
        /// The height-field depth in `UV` units.
        height_scale: f32,
    },
    /// The view-dependent layer count, a `lerp` from `min_layers` toward
    /// `max_layers` by the grazing fraction
    /// ([`ParallaxConfig::layer_count`](prism_render_architecture::particle::parallax_offset::ParallaxConfig::layer_count)).
    LayerCount {
        /// The view cosine (`z` of the normalized view direction).
        cos_view: f32,
        /// Head-on layer count (host-widened from the golden `u16`).
        min_layers: u32,
        /// Grazing layer count (host-widened from the golden `u16`).
        max_layers: u32,
    },
    /// The one-step secant refinement of a bracketed crossing
    /// ([`secant_refine`](prism_render_architecture::particle::parallax_offset::secant_refine)).
    SecantRefine {
        /// `UV` at the last outside sample.
        prev_uv: [f32; 2],
        /// `UV` at the first inside sample.
        hit_uv: [f32; 2],
        /// Ray-minus-surface value at the outside sample (`>= 0`).
        before: f32,
        /// Ray-minus-surface value at the inside sample (`<= 0`).
        after: f32,
    },
    /// One steep-parallax march step: slide the `UV` by `delta_uv`, advance the
    /// ray depth by `layer_step`, re-evaluate the surface depth and report
    /// whether the ray reached or passed below the surface.
    SteepStep {
        /// Current `UV` before the step.
        cur_uv: [f32; 2],
        /// Per-layer `UV` slide (`full_offset * layer_step`).
        delta_uv: [f32; 2],
        /// Current ray depth before the step.
        cur_layer_depth: f32,
        /// Depth advance per layer (`1 / layers`).
        layer_step: f32,
        /// Height sampled by the host at the stepped `UV`, in `[0, 1]`.
        height_sample: f32,
    },
    /// The steep-hit bracket resolution: the `before` / `after` values and the
    /// secant-refined `UV` ([`SteepHit::before`](prism_render_architecture::particle::parallax_offset::SteepHit::before),
    /// [`SteepHit::after`](prism_render_architecture::particle::parallax_offset::SteepHit::after),
    /// [`SteepHit::refined_uv`](prism_render_architecture::particle::parallax_offset::SteepHit::refined_uv)).
    SteepResolve {
        /// `UV` at the last outside sample.
        prev_uv: [f32; 2],
        /// `UV` at the first inside sample.
        hit_uv: [f32; 2],
        /// Ray depth at the outside sample.
        prev_layer_depth: f32,
        /// Surface depth at the outside sample.
        prev_surface_depth: f32,
        /// Ray depth at the inside sample.
        hit_layer_depth: f32,
        /// Surface depth at the inside sample.
        hit_surface_depth: f32,
    },
    /// One self-shadow march step: slide the shadow ray toward the light, drop
    /// its depth, re-evaluate the surface depth and update the running maximum
    /// overlap.
    SelfShadowStep {
        /// Current shadow-ray `UV` before the step.
        ray_uv: [f32; 2],
        /// Per-step `UV` slide (`full_offset * depth_step`).
        delta_uv: [f32; 2],
        /// Current shadow-ray depth before the step.
        ray_depth: f32,
        /// Depth drop per step.
        depth_step: f32,
        /// Height sampled by the host at the stepped `UV`, in `[0, 1]`.
        height_sample: f32,
        /// Running maximum overlap before this step.
        max_overlap: f32,
    },
    /// The final self-shadow resolve: convert the largest overlap into a lit
    /// fraction through the softness-scaled `smoothstep01`.
    SelfShadowResolve {
        /// The largest ray-minus-surface overlap found along the shadow ray.
        max_overlap: f32,
        /// The self-shadow contact softness, in depth units.
        softness: f32,
    },
}

/// The resolved answer for one [`ParallaxOffsetQuery`], one variant per query
/// kind.
///
/// Provenance: twinned from this repository's
/// [`parallax_offset`](prism_render_architecture::particle::parallax_offset); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParallaxOffsetResult {
    /// The clamped scalar, matching the reference `clamp01`.
    Clamp01(f32),
    /// The smoothstep value, matching the reference `smoothstep01`.
    Smoothstep01(f32),
    /// The interpolated value, matching the reference `lerp`.
    Lerp(f32),
    /// The normalized vector, matching the reference `normalize3` (`None` for a
    /// near-zero input).
    Normalize3(Option<[f32; 3]>),
    /// The full-depth offset vector, matching the reference `full_offset`.
    FullOffset([f32; 2]),
    /// The view-dependent layer count, matching `layer_count`.
    LayerCount(f32),
    /// The secant-refined `UV`, matching `secant_refine`.
    SecantRefine([f32; 2]),
    /// One steep-parallax march step, matching the golden loop body.
    SteepStep {
        /// `UV` after the step.
        next_uv: [f32; 2],
        /// Ray depth after the step.
        next_layer_depth: f32,
        /// Surface depth at the stepped `UV`.
        next_surface_depth: f32,
        /// Whether the ray reached or passed below the surface.
        penetrated: bool,
    },
    /// The steep-hit bracket resolution, matching `SteepHit`.
    SteepResolve {
        /// The `before` value `prev_surface_depth - prev_layer_depth`.
        before: f32,
        /// The `after` value `hit_surface_depth - hit_layer_depth`.
        after: f32,
        /// The secant-refined crossing `UV`.
        refined_uv: [f32; 2],
    },
    /// One self-shadow march step, matching the golden loop body.
    SelfShadowStep {
        /// Shadow-ray `UV` after the step.
        ray_uv: [f32; 2],
        /// Shadow-ray depth after the step.
        ray_depth: f32,
        /// Surface depth at the stepped `UV`.
        surface_depth: f32,
        /// Running maximum overlap after this step.
        max_overlap: f32,
    },
    /// The final self-shadow lit fraction, matching the golden resolve.
    SelfShadowResolve(f32),
}

/// `repr(C)` `std430` layout of one packed query: three `u32` control words
/// (`op` plus the two widened layer bounds) and a pad, an eight-lane `f32`
/// scalar block, then two four-lane vector payloads — `80` bytes, matching the
/// `WGSL` `Query` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Op code selecting the routine.
    op: u32,
    /// Head-on layer count (host-widened from the golden `u16`).
    min_layers: u32,
    /// Grazing layer count (host-widened from the golden `u16`).
    max_layers: u32,
    /// Padding word.
    pad0: u32,
    /// Scalar parameters, meaning per op.
    s: [f32; 8],
    /// Vector input A (`xyz` for a direction, `xy` for a `UV`).
    a: [f32; 4],
    /// Vector input B (`xy` for a `UV` or per-layer delta).
    b: [f32; 4],
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &ParallaxOffsetQuery) -> GpuQuery {
        let mut g = GpuQuery::zeroed();
        match query {
            ParallaxOffsetQuery::Clamp01 { x } => {
                g.op = OP_CLAMP01;
                g.s[0] = *x;
            }
            ParallaxOffsetQuery::Smoothstep01 { t } => {
                g.op = OP_SMOOTHSTEP01;
                g.s[0] = *t;
            }
            ParallaxOffsetQuery::Lerp { a, b, t } => {
                g.op = OP_LERP;
                g.s[0] = *a;
                g.s[1] = *b;
                g.s[2] = *t;
            }
            ParallaxOffsetQuery::Normalize3 { v } => {
                g.op = OP_NORMALIZE3;
                set_vec3(&mut g.a, *v);
            }
            ParallaxOffsetQuery::FullOffset {
                view_unit,
                height_scale,
            } => {
                g.op = OP_FULL_OFFSET;
                g.s[0] = *height_scale;
                set_vec3(&mut g.a, *view_unit);
            }
            ParallaxOffsetQuery::LayerCount {
                cos_view,
                min_layers,
                max_layers,
            } => {
                g.op = OP_LAYER_COUNT;
                g.s[0] = *cos_view;
                g.min_layers = *min_layers;
                g.max_layers = *max_layers;
            }
            ParallaxOffsetQuery::SecantRefine {
                prev_uv,
                hit_uv,
                before,
                after,
            } => {
                g.op = OP_SECANT_REFINE;
                g.s[0] = *before;
                g.s[1] = *after;
                set_vec2(&mut g.a, *prev_uv);
                set_vec2(&mut g.b, *hit_uv);
            }
            ParallaxOffsetQuery::SteepStep {
                cur_uv,
                delta_uv,
                cur_layer_depth,
                layer_step,
                height_sample,
            } => {
                g.op = OP_STEEP_STEP;
                g.s[0] = *cur_layer_depth;
                g.s[1] = *layer_step;
                g.s[2] = *height_sample;
                set_vec2(&mut g.a, *cur_uv);
                set_vec2(&mut g.b, *delta_uv);
            }
            ParallaxOffsetQuery::SteepResolve {
                prev_uv,
                hit_uv,
                prev_layer_depth,
                prev_surface_depth,
                hit_layer_depth,
                hit_surface_depth,
            } => {
                g.op = OP_STEEP_RESOLVE;
                g.s[0] = *prev_layer_depth;
                g.s[1] = *prev_surface_depth;
                g.s[2] = *hit_layer_depth;
                g.s[3] = *hit_surface_depth;
                set_vec2(&mut g.a, *prev_uv);
                set_vec2(&mut g.b, *hit_uv);
            }
            ParallaxOffsetQuery::SelfShadowStep {
                ray_uv,
                delta_uv,
                ray_depth,
                depth_step,
                height_sample,
                max_overlap,
            } => {
                g.op = OP_SELF_SHADOW_STEP;
                g.s[0] = *ray_depth;
                g.s[1] = *depth_step;
                g.s[2] = *height_sample;
                g.s[3] = *max_overlap;
                set_vec2(&mut g.a, *ray_uv);
                set_vec2(&mut g.b, *delta_uv);
            }
            ParallaxOffsetQuery::SelfShadowResolve {
                max_overlap,
                softness,
            } => {
                g.op = OP_SELF_SHADOW_RESOLVE;
                g.s[0] = *max_overlap;
                g.s[1] = *softness;
            }
        }
        g
    }
}

/// Writes a two-lane `UV` into a packed four-lane slot (`zw` stay zero).
fn set_vec2(slot: &mut [f32; 4], uv: [f32; 2]) {
    slot[0] = uv[0];
    slot[1] = uv[1];
}

/// Writes a three-lane vector into a packed four-lane slot (`w` stays zero).
fn set_vec3(slot: &mut [f32; 4], v: [f32; 3]) {
    slot[0] = v[0];
    slot[1] = v[1];
    slot[2] = v[2];
}

/// `repr(C)` `std430` layout of one result: four `u32` control words (a flag
/// plus three pads), a scalar triple plus a pad, then a vector slot plus a pad —
/// `48` bytes, matching the `WGSL` `Result` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Validity / penetration flag.
    flag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Primary scalar output.
    scalar: f32,
    /// Secondary scalar output.
    scalar2: f32,
    /// Tertiary scalar output.
    scalar3: f32,
    /// Padding lane.
    pad3: f32,
    /// Vector output `x`.
    vx: f32,
    /// Vector output `y`.
    vy: f32,
    /// Vector output `z`.
    vz: f32,
    /// Padding lane.
    pad4: f32,
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

/// Decodes one packed [`GpuResult`] into the public [`ParallaxOffsetResult`]
/// matching the originating `query`'s variant.
fn decode_result(query: &ParallaxOffsetQuery, raw: &GpuResult) -> ParallaxOffsetResult {
    match query {
        ParallaxOffsetQuery::Clamp01 { .. } => ParallaxOffsetResult::Clamp01(raw.scalar),
        ParallaxOffsetQuery::Smoothstep01 { .. } => ParallaxOffsetResult::Smoothstep01(raw.scalar),
        ParallaxOffsetQuery::Lerp { .. } => ParallaxOffsetResult::Lerp(raw.scalar),
        ParallaxOffsetQuery::Normalize3 { .. } => {
            if raw.flag == 1 {
                ParallaxOffsetResult::Normalize3(Some([raw.vx, raw.vy, raw.vz]))
            } else {
                ParallaxOffsetResult::Normalize3(None)
            }
        }
        ParallaxOffsetQuery::FullOffset { .. } => {
            ParallaxOffsetResult::FullOffset([raw.vx, raw.vy])
        }
        ParallaxOffsetQuery::LayerCount { .. } => ParallaxOffsetResult::LayerCount(raw.scalar),
        ParallaxOffsetQuery::SecantRefine { .. } => {
            ParallaxOffsetResult::SecantRefine([raw.vx, raw.vy])
        }
        ParallaxOffsetQuery::SteepStep { .. } => ParallaxOffsetResult::SteepStep {
            next_uv: [raw.vx, raw.vy],
            next_layer_depth: raw.scalar,
            next_surface_depth: raw.scalar2,
            penetrated: raw.flag == 1,
        },
        ParallaxOffsetQuery::SteepResolve { .. } => ParallaxOffsetResult::SteepResolve {
            before: raw.scalar,
            after: raw.scalar2,
            refined_uv: [raw.vx, raw.vy],
        },
        ParallaxOffsetQuery::SelfShadowStep { .. } => ParallaxOffsetResult::SelfShadowStep {
            ray_uv: [raw.vx, raw.vy],
            ray_depth: raw.scalar,
            surface_depth: raw.scalar2,
            max_overlap: raw.scalar3,
        },
        ParallaxOffsetQuery::SelfShadowResolve { .. } => {
            ParallaxOffsetResult::SelfShadowResolve(raw.scalar)
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

/// A compiled, reusable parallax-occlusion compute pipeline.
///
/// Provenance: twinned from this repository's
/// [`parallax_offset`](prism_render_architecture::particle::parallax_offset); no
/// third-party engine source or derived code.
pub struct GpuParallaxOffset {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuParallaxOffset {
    /// Compiles the parallax-occlusion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuParallaxOffset {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_parallax_offset"),
            source: ShaderSource::Wgsl(PARALLAX_OFFSET_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_parallax_offset_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_parallax_offset_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_parallax_offset_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuParallaxOffset {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`ParallaxOffsetResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answer for the query's variant to within
    /// the tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[ParallaxOffsetQuery],
    ) -> Vec<ParallaxOffsetResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_parallax_offset_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_parallax_offset_output"),
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
            label: Some("prism_volumetric_parallax_offset_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_parallax_offset_bind_group"),
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
            label: Some("prism_volumetric_parallax_offset_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_parallax_offset_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_parallax_offset_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
