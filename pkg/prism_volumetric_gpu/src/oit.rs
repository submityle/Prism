//! `wgpu` compute twin of the order-independent-transparency golden
//! ([`oit`](prism_render_architecture::particle::oit), particle design §12,
//! the composite / reconstruction half of the transparency path).
//!
//! The `CPU` golden [`oit`](prism_render_architecture::particle::oit) owns the
//! *composite / reconstruction math* a translucent resolve runs once the
//! fragments of a draw have been gathered: the strategy-to-method routing
//! ([`composite_method`](prism_render_architecture::particle::oit::composite_method)),
//! the weighted-blended `OIT` depth weight
//! ([`wboit_weight`](prism_render_architecture::particle::oit::wboit_weight)) and
//! its single-pass resolve / `over` compositing, the multiply-only physical
//! transforms
//! ([`approx_absorbance`](prism_render_architecture::particle::oit::approx_absorbance)
//! and
//! [`approx_transmittance`](prism_render_architecture::particle::oit::approx_transmittance)),
//! and the moment-based `OIT` power-moment reconstruction
//! ([`moment_occlusion`](prism_render_architecture::particle::oit::moment_occlusion),
//! [`reconstruct_optical_depth`](prism_render_architecture::particle::oit::reconstruct_optical_depth),
//! [`moment_transmittance`](prism_render_architecture::particle::oit::moment_transmittance))
//! plus the soft-particle depth fade
//! ([`soft_particle_fade`](prism_render_architecture::particle::oit::soft_particle_fade)).
//! Every one of those answers is a closed-form sequence of `+ - * /`, `abs`,
//! `min`, `max`, `clamp` and one guarded `sqrt`, so the whole interface ports to
//! a portable core-`WGSL` kernel with no transcendental call.
//!
//! [`GpuOit`] is the on-device twin: one thread per [`OitQuery`] reproduces the
//! matching golden answer, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same polynomials, folds the same
//! moment reconstruction and classifies the same composite method the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The twin reproduces, per query, exactly the golden routines:
//! [`composite_method`](prism_render_architecture::particle::oit::composite_method)
//! (the strategy / blend / tier classification code),
//! [`wboit_weight`](prism_render_architecture::particle::oit::wboit_weight),
//! [`ResolvedTransparency::over`](prism_render_architecture::particle::oit::ResolvedTransparency::over),
//! the final-division resolve and the revealage of a
//! [`WboitAccumulator`](prism_render_architecture::particle::oit::WboitAccumulator)
//! (the host pre-accumulates the running sums and the twin runs only the
//! resolve divide / `over`),
//! [`approx_absorbance`](prism_render_architecture::particle::oit::approx_absorbance),
//! [`approx_transmittance`](prism_render_architecture::particle::oit::approx_transmittance),
//! the normalization of a
//! [`MomentAccumulator`](prism_render_architecture::particle::oit::MomentAccumulator)
//! into [`PowerMoments`](prism_render_architecture::particle::oit::PowerMoments),
//! [`moment_occlusion`](prism_render_architecture::particle::oit::moment_occlusion),
//! [`reconstruct_optical_depth`](prism_render_architecture::particle::oit::reconstruct_optical_depth),
//! [`moment_transmittance`](prism_render_architecture::particle::oit::moment_transmittance),
//! and
//! [`soft_particle_fade`](prism_render_architecture::particle::oit::soft_particle_fade).
//!
//! # What is not twinned
//!
//! The host-only, list-and-state parts of the golden are deliberately left on
//! the `CPU`: the
//! [`FragmentList`](prism_render_architecture::particle::oit::FragmentList)
//! insert / overflow policy, its `Vec` capacity, and the depth-sorted
//! back-to-front traversal order of the exact resolve. Those are per-pixel list
//! bookkeeping and a sort, not the per-element arithmetic a
//! one-thread-per-element kernel is for. The host pre-sorts and pre-accumulates,
//! then feeds the twin the already-reduced per-point inputs.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `select`, `+ - * /` and one guarded `sqrt` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`
//! builtin and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The golden's `smoothstep` is expanded by hand as the
//! Hermite polynomial `3t^2 - 2t^3`, and the physical `exp` / `log` transforms
//! are the golden's own multiply-only truncated series, copied verbatim.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! guarded divide / `sqrt`, so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields, while the composite-method classification is an integer code compared
//! for exact equality.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`oit`](prism_render_architecture::particle::oit); no third-party engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::oit::{OitMethod, OitQuality, PowerMoments};
use prism_render_architecture::particle::renderers::ParticleBlend;
use prism_render_architecture::particle::SortStrategy;
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

/// Op code for the composite-method classification.
const OP_COMPOSITE_METHOD: u32 = 0;
/// Op code for the weighted-blended `OIT` depth weight.
const OP_WBOIT_WEIGHT: u32 = 1;
/// Op code for the `over` composite of a resolved transparency.
const OP_OVER: u32 = 2;
/// Op code for the weighted-blended resolve division.
const OP_WBOIT_RESOLVE: u32 = 3;
/// Op code for the weighted-blended revealage read-out.
const OP_WBOIT_REVEALAGE: u32 = 4;
/// Op code for the multiply-only absorbance approximation.
const OP_APPROX_ABSORBANCE: u32 = 5;
/// Op code for the multiply-only transmittance approximation.
const OP_APPROX_TRANSMITTANCE: u32 = 6;
/// Op code for the moment-accumulator normalization.
const OP_MOMENTS_NORMALIZED: u32 = 7;
/// Op code for the four-power-moment occlusion reconstruction.
const OP_MOMENT_OCCLUSION: u32 = 8;
/// Op code for the reconstructed optical depth.
const OP_RECONSTRUCT_OPTICAL_DEPTH: u32 = 9;
/// Op code for the reconstructed moment transmittance.
const OP_MOMENT_TRANSMITTANCE: u32 = 10;
/// Op code for the soft-particle depth fade.
const OP_SOFT_PARTICLE_FADE: u32 = 11;

/// Classification code for [`SortStrategy::None`].
const STRAT_NONE: u32 = 0;
/// Classification code for [`SortStrategy::SharedOit`].
const STRAT_SHARED_OIT: u32 = 1;
/// Classification code for [`SortStrategy::ViewDepthRadix`].
const STRAT_RADIX: u32 = 2;
/// Classification code for [`SortStrategy::ViewDepthBitonic`].
const STRAT_BITONIC: u32 = 3;

/// Classification code for [`ParticleBlend::Opaque`].
const BLEND_OPAQUE: u32 = 0;
/// Classification code for [`ParticleBlend::AlphaMask`].
const BLEND_ALPHA_MASK: u32 = 1;
/// Classification code for [`ParticleBlend::Additive`].
const BLEND_ADDITIVE: u32 = 2;
/// Classification code for [`ParticleBlend::Premultiplied`].
const BLEND_PREMULTIPLIED: u32 = 3;
/// Classification code for [`ParticleBlend::AlphaBlend`].
const BLEND_ALPHA_BLEND: u32 = 4;

/// Classification code for [`OitQuality::Fast`].
const TIER_FAST: u32 = 0;
/// Classification code for [`OitQuality::Balanced`].
const TIER_BALANCED: u32 = 1;
/// Classification code for [`OitQuality::Reference`].
const TIER_REFERENCE: u32 = 2;

/// Classification code for [`OitMethod::Additive`].
const METHOD_ADDITIVE: u32 = 1;
/// Classification code for [`OitMethod::WeightedBlended`].
const METHOD_WEIGHTED_BLENDED: u32 = 2;
/// Classification code for [`OitMethod::MomentBased`].
const METHOD_MOMENT_BASED: u32 = 3;
/// Classification code for [`OitMethod::PerPixelLinkedList`].
const METHOD_PER_PIXEL_LINKED_LIST: u32 = 4;

/// The portable core-`WGSL` order-independent-transparency kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` dispatches on an op code to mirror each golden
/// [`oit`](prism_render_architecture::particle::oit) routine for routine; see
/// the module documentation for the algorithm.
const OIT_WGSL: &str = r#"
// OIT twin: one thread per query reproduces the composite-method routing, the
// weighted-blended depth weight and resolve, the multiply-only absorbance /
// transmittance series, the four-power-moment reconstruction and the
// soft-particle depth fade. It mirrors the CPU golden oit routines for routine.
//
// Portability: only the core subset (abs, min, max, clamp, select, + - * / and
// one guarded sqrt) is used; no sin/cos/exp/log/pow/tan, no inverse
// trigonometry, no smoothstep builtin and no optional device feature, so it
// runs unmodified on Metal, Vulkan and DX12. The golden smoothstep is expanded
// by hand as 3t^2 - 2t^3; the exp/log transforms are the golden's own
// multiply-only truncated series.
//
// Provenance: twinned from this repository's particle::oit; no third-party
// engine source or derived code.

// Shared epsilon guarding divisions and f32 equality bands, matching the
// reference OIT_EPS.
const OIT_EPS: f32 = 1.0e-6;

const OP_COMPOSITE_METHOD: u32 = 0u;
const OP_WBOIT_WEIGHT: u32 = 1u;
const OP_OVER: u32 = 2u;
const OP_WBOIT_RESOLVE: u32 = 3u;
const OP_WBOIT_REVEALAGE: u32 = 4u;
const OP_APPROX_ABSORBANCE: u32 = 5u;
const OP_APPROX_TRANSMITTANCE: u32 = 6u;
const OP_MOMENTS_NORMALIZED: u32 = 7u;
const OP_MOMENT_OCCLUSION: u32 = 8u;
const OP_RECONSTRUCT_OPTICAL_DEPTH: u32 = 9u;
const OP_MOMENT_TRANSMITTANCE: u32 = 10u;
const OP_SOFT_PARTICLE_FADE: u32 = 11u;

const STRAT_NONE: u32 = 0u;
const STRAT_SHARED_OIT: u32 = 1u;
const STRAT_RADIX: u32 = 2u;
const STRAT_BITONIC: u32 = 3u;

const BLEND_ADDITIVE: u32 = 2u;
const BLEND_PREMULTIPLIED: u32 = 3u;

const TIER_FAST: u32 = 0u;
const TIER_BALANCED: u32 = 1u;

const METHOD_NONE: u32 = 0u;
const METHOD_ADDITIVE: u32 = 1u;
const METHOD_WEIGHTED_BLENDED: u32 = 2u;
const METHOD_MOMENT_BASED: u32 = 3u;
const METHOD_PER_PIXEL_LINKED_LIST: u32 = 4u;

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
    // Strategy / blend / tier classification codes for the composite-method op.
    strategy: u32,
    blend: u32,
    tier: u32,
    // Scalar parameters, meaning per op (view_depth/alpha/near/far, coverage,
    // alpha_weight/revealage, optical_depth, raw moment sums, depth/bias,
    // scene_depth/particle_depth/contrast).
    s: array<f32, 8>,
    // Colour / colour-weight input (xyz, w pad).
    color: array<f32, 4>,
    // Background input for the over composite (xyz, w pad).
    background: array<f32, 4>,
    // Normalized power moments for the reconstruction ops.
    moments: array<f32, 4>,
}

struct Result {
    // Composite-method classification code.
    method: u32,
    // Whether the moment normalization produced a value (1) or not (0).
    present: u32,
    pad0: u32,
    pad1: u32,
    // Single-scalar output (weight, absorbance, transmittance, occlusion,
    // optical depth, revealage, fade).
    scalar: f32,
    // Resolve coverage output.
    coverage: f32,
    // Normalized total output.
    total: f32,
    pad2: f32,
    // Vector output (over / resolve colour).
    vx: f32,
    vy: f32,
    vz: f32,
    pad3: f32,
    // Normalized power-moment output.
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps a scalar to the closed unit interval, mirroring the reference clamp01.
fn clamp01(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

// Linear interpolation (1 - t) * a + t * b, mirroring the reference lerp.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Hermite smoothstep 3t^2 - 2t^3 on an already-normalized t, mirroring the
// reference smoothstep01; written out so no smoothstep builtin is used.
fn smoothstep01(t_in: f32) -> f32 {
    let t = clamp01(t_in);
    return t * t * (3.0 - 2.0 * t);
}

// Composite-method routing, mirroring the reference composite_method.
fn composite_method(strategy: u32, blend: u32, tier: u32) -> u32 {
    if (strategy == STRAT_NONE) {
        if (blend == BLEND_ADDITIVE || blend == BLEND_PREMULTIPLIED) {
            return METHOD_ADDITIVE;
        }
        return METHOD_NONE;
    }
    if (strategy == STRAT_RADIX || strategy == STRAT_BITONIC) {
        return METHOD_NONE;
    }
    if (tier == TIER_FAST) {
        return METHOD_WEIGHTED_BLENDED;
    }
    if (tier == TIER_BALANCED) {
        return METHOD_MOMENT_BASED;
    }
    return METHOD_PER_PIXEL_LINKED_LIST;
}

// Weighted-blended OIT depth/alpha weight, mirroring the reference wboit_weight.
fn wboit_weight(view_depth: f32, alpha: f32, near: f32, far_in: f32) -> f32 {
    let far = max(far_in, max(near, OIT_EPS));
    let d = max(view_depth, max(near, 0.0));
    let u = d / far;
    let u2 = u * u;
    let u4 = u2 * u2;
    let raw = 0.03 / (1.0e-5 + u4);
    return clamp01(alpha) * clamp(raw, 1.0e-2, 3.0e3);
}

// Multiply-only absorbance series alpha + alpha^2/2 + ... + alpha^8/8, copied
// verbatim from the reference approx_absorbance.
fn approx_absorbance(alpha: f32) -> f32 {
    let a = min(clamp01(alpha), 0.999);
    let a2 = a * a;
    let a3 = a2 * a;
    let a4 = a3 * a;
    let a5 = a4 * a;
    let a6 = a5 * a;
    let a7 = a6 * a;
    let a8 = a7 * a;
    return a + a2 / 2.0 + a3 / 3.0 + a4 / 4.0 + a5 / 5.0 + a6 / 6.0 + a7 / 7.0 + a8 / 8.0;
}

// Reciprocal truncated exponential series 1 / (1 + x + x^2/2 + x^3/6 + x^4/24),
// copied verbatim from the reference approx_transmittance.
fn approx_transmittance(optical_depth: f32) -> f32 {
    let x = max(optical_depth, 0.0);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let denom = 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0;
    return clamp01(1.0 / denom);
}

// Four-power-moment MSM occlusion reconstruction, mirroring the reference
// moment_occlusion exactly (Peters & Klein, as adopted by MBOIT).
fn moment_occlusion(m0: f32, m1: f32, m2: f32, m3: f32, depth: f32, bias_in: f32) -> f32 {
    let bias = clamp01(bias_in);
    let b0 = lerp(m0, 0.0, bias);
    let b1 = lerp(m1, 0.375, bias);
    let b2 = lerp(m2, 0.0, bias);
    let b3 = lerp(m3, 0.375, bias);

    let l32d22 = b2 - b0 * b1;
    let d22 = b1 - b0 * b0;
    let sq_depth_var = b3 - b1 * b1;
    let d33d22 = sq_depth_var * d22 - l32d22 * l32d22;

    if (abs(d22) < OIT_EPS || abs(d33d22) < OIT_EPS) {
        return select(1.0, 0.0, depth <= b0);
    }

    let inv_d22 = 1.0 / d22;
    let l32 = l32d22 * inv_d22;

    var c0 = 1.0;
    var c1 = depth - b0;
    var c2 = depth * depth - b1 - l32 * c1;
    c1 = c1 * inv_d22;
    c2 = c2 * (d22 / d33d22);
    c1 = c1 - l32 * c2;
    c0 = c0 - (c1 * b0 + c2 * b1);

    if (abs(c2) < OIT_EPS) {
        return select(1.0, 0.0, depth <= b0);
    }
    let p = c1 / c2;
    let q = c0 / c2;
    let disc = max(p * p * 0.25 - q, 0.0);
    let r = sqrt(disc);
    let z1 = -p * 0.5 - r;
    let z2 = -p * 0.5 + r;

    var sx = 0.0;
    var sy = 0.0;
    var sz = 0.0;
    var sw = 0.0;
    if (z2 < depth) {
        sx = z1;
        sy = depth;
        sz = 1.0;
        sw = 1.0;
    } else if (z1 < depth) {
        sx = depth;
        sy = z1;
        sz = 0.0;
        sw = 1.0;
    } else {
        sx = 0.0;
        sy = 0.0;
        sz = 0.0;
        sw = 0.0;
    }
    let denom = (z2 - sy) * (depth - z1);
    var quotient = 0.0;
    if (abs(denom) >= OIT_EPS) {
        quotient = (sx * z2 - b0 * (sx + z2) + b1) / denom;
    }
    return clamp01(sz + sw * quotient);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let qd = queries[idx];
    var r: Result;
    r.method = 0u;
    r.present = 0u;
    r.pad0 = 0u;
    r.pad1 = 0u;
    r.scalar = 0.0;
    r.coverage = 0.0;
    r.total = 0.0;
    r.pad2 = 0.0;
    r.vx = 0.0;
    r.vy = 0.0;
    r.vz = 0.0;
    r.pad3 = 0.0;
    r.b0 = 0.0;
    r.b1 = 0.0;
    r.b2 = 0.0;
    r.b3 = 0.0;

    let op = qd.op;

    if (op == OP_COMPOSITE_METHOD) {
        r.method = composite_method(qd.strategy, qd.blend, qd.tier);
    } else if (op == OP_WBOIT_WEIGHT) {
        r.scalar = wboit_weight(qd.s[0], qd.s[1], qd.s[2], qd.s[3]);
    } else if (op == OP_OVER) {
        let col = vec3<f32>(qd.color[0], qd.color[1], qd.color[2]);
        let bg = vec3<f32>(qd.background[0], qd.background[1], qd.background[2]);
        let revealage = 1.0 - clamp01(qd.s[0]);
        let outc = col + bg * revealage;
        r.vx = outc.x;
        r.vy = outc.y;
        r.vz = outc.z;
    } else if (op == OP_WBOIT_RESOLVE) {
        let cw = vec3<f32>(qd.color[0], qd.color[1], qd.color[2]);
        let average = cw * (1.0 / max(qd.s[0], OIT_EPS));
        let coverage = 1.0 - clamp01(qd.s[1]);
        let outc = average * coverage;
        r.vx = outc.x;
        r.vy = outc.y;
        r.vz = outc.z;
        r.coverage = coverage;
    } else if (op == OP_WBOIT_REVEALAGE) {
        r.scalar = clamp01(qd.s[0]);
    } else if (op == OP_APPROX_ABSORBANCE) {
        r.scalar = approx_absorbance(qd.s[0]);
    } else if (op == OP_APPROX_TRANSMITTANCE) {
        r.scalar = approx_transmittance(qd.s[0]);
    } else if (op == OP_MOMENTS_NORMALIZED) {
        let total = qd.s[0];
        if (total < OIT_EPS) {
            r.present = 0u;
        } else {
            let inv = 1.0 / total;
            r.present = 1u;
            r.total = total;
            r.b0 = qd.s[1] * inv;
            r.b1 = qd.s[2] * inv;
            r.b2 = qd.s[3] * inv;
            r.b3 = qd.s[4] * inv;
        }
    } else if (op == OP_MOMENT_OCCLUSION) {
        r.scalar = moment_occlusion(
            qd.moments[0],
            qd.moments[1],
            qd.moments[2],
            qd.moments[3],
            qd.s[0],
            qd.s[1],
        );
    } else if (op == OP_RECONSTRUCT_OPTICAL_DEPTH) {
        let occ = moment_occlusion(
            qd.moments[0],
            qd.moments[1],
            qd.moments[2],
            qd.moments[3],
            qd.s[1],
            qd.s[2],
        );
        r.scalar = qd.s[0] * occ;
    } else if (op == OP_MOMENT_TRANSMITTANCE) {
        let occ = moment_occlusion(
            qd.moments[0],
            qd.moments[1],
            qd.moments[2],
            qd.moments[3],
            qd.s[1],
            qd.s[2],
        );
        r.scalar = approx_transmittance(qd.s[0] * occ);
    } else {
        let contrast = max(qd.s[2], OIT_EPS);
        let delta = qd.s[0] - qd.s[1];
        r.scalar = smoothstep01(delta / contrast);
    }

    results[idx] = r;
}
"#;

/// One order-independent-transparency query: the routine and its inputs.
///
/// Each variant mirrors one golden
/// [`oit`](prism_render_architecture::particle::oit) routine. The accumulation
/// variants ([`OitQuery::WboitResolve`], [`OitQuery::MomentsNormalized`]) take
/// the already-reduced running sums the host computed, since the twin runs only
/// the final resolve / normalization arithmetic.
///
/// Provenance: twinned from this repository's
/// [`oit`](prism_render_architecture::particle::oit); no third-party engine
/// source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OitQuery {
    /// Classify the composite path for a translucent draw
    /// ([`composite_method`](prism_render_architecture::particle::oit::composite_method)).
    CompositeMethod {
        /// The upstream sort strategy.
        strategy: SortStrategy,
        /// The particle blend mode.
        blend: ParticleBlend,
        /// The shared-`OIT` quality tier.
        tier: OitQuality,
    },
    /// Evaluate the weighted-blended `OIT` depth weight
    /// ([`wboit_weight`](prism_render_architecture::particle::oit::wboit_weight)).
    WboitWeight {
        /// View-space depth of the fragment.
        view_depth: f32,
        /// Straight alpha (coverage) of the fragment.
        alpha: f32,
        /// Near plane distance.
        near: f32,
        /// Far plane distance.
        far: f32,
    },
    /// Composite a resolved transparency over an opaque background
    /// ([`ResolvedTransparency::over`](prism_render_architecture::particle::oit::ResolvedTransparency::over)).
    Over {
        /// The coverage-weighted foreground colour.
        color: [f32; 3],
        /// Total coverage `1 - revealage` in `[0, 1]`.
        coverage: f32,
        /// The opaque background colour.
        background: [f32; 3],
    },
    /// Resolve the pre-accumulated weighted-blended sums into a coverage-weighted
    /// colour
    /// ([`WboitAccumulator::resolve`](prism_render_architecture::particle::oit::WboitAccumulator::resolve)).
    WboitResolve {
        /// The accumulated weighted premultiplied colour `sum(color * alpha * weight)`.
        color_weight: [f32; 3],
        /// The accumulated weighted alpha `sum(alpha * weight)`.
        alpha_weight: f32,
        /// The accumulated revealage product `product(1 - alpha)`.
        revealage: f32,
    },
    /// Clamp the accumulated revealage product into `[0, 1]`
    /// ([`WboitAccumulator::revealage`](prism_render_architecture::particle::oit::WboitAccumulator::revealage)).
    WboitRevealage {
        /// The accumulated revealage product `product(1 - alpha)`.
        revealage: f32,
    },
    /// Approximate the optical depth `-ln(1 - alpha)` of one fragment
    /// ([`approx_absorbance`](prism_render_architecture::particle::oit::approx_absorbance)).
    ApproxAbsorbance {
        /// Straight alpha of the fragment.
        alpha: f32,
    },
    /// Approximate the transmittance `exp(-optical_depth)` behind a depth
    /// ([`approx_transmittance`](prism_render_architecture::particle::oit::approx_transmittance)).
    ApproxTransmittance {
        /// The optical depth in front of the query point.
        optical_depth: f32,
    },
    /// Normalize the raw moment sums into
    /// [`PowerMoments`](prism_render_architecture::particle::oit::PowerMoments),
    /// or `None` when no absorbance was accumulated
    /// ([`MomentAccumulator::normalized`](prism_render_architecture::particle::oit::MomentAccumulator::normalized)).
    MomentsNormalized {
        /// Total accumulated absorbance (the zeroth moment).
        total: f32,
        /// Raw first power-moment sum `sum(a * d)`.
        m1: f32,
        /// Raw second power-moment sum `sum(a * d^2)`.
        m2: f32,
        /// Raw third power-moment sum `sum(a * d^3)`.
        m3: f32,
        /// Raw fourth power-moment sum `sum(a * d^4)`.
        m4: f32,
    },
    /// Reconstruct the occluded absorbance fraction in front of `depth`
    /// ([`moment_occlusion`](prism_render_architecture::particle::oit::moment_occlusion)).
    MomentOcclusion {
        /// Normalized power moments `[E[d], E[d^2], E[d^3], E[d^4]]`.
        moments: [f32; 4],
        /// Query depth in `[0, 1]`.
        depth: f32,
        /// Conditioning bias in `[0, 1]`.
        bias: f32,
    },
    /// Reconstruct the optical depth in front of `depth` from power moments
    /// ([`reconstruct_optical_depth`](prism_render_architecture::particle::oit::reconstruct_optical_depth)).
    ReconstructOpticalDepth {
        /// Total accumulated absorbance (the zeroth moment).
        total: f32,
        /// Normalized power moments `[E[d], E[d^2], E[d^3], E[d^4]]`.
        moments: [f32; 4],
        /// Query depth in `[0, 1]`.
        depth: f32,
        /// Conditioning bias in `[0, 1]`.
        bias: f32,
    },
    /// Reconstruct the transmittance a fragment at `depth` sees from power
    /// moments
    /// ([`moment_transmittance`](prism_render_architecture::particle::oit::moment_transmittance)).
    MomentTransmittance {
        /// Total accumulated absorbance (the zeroth moment).
        total: f32,
        /// Normalized power moments `[E[d], E[d^2], E[d^3], E[d^4]]`.
        moments: [f32; 4],
        /// Query depth in `[0, 1]`.
        depth: f32,
        /// Conditioning bias in `[0, 1]`.
        bias: f32,
    },
    /// Fade a translucent fragment approaching the opaque surface
    /// ([`soft_particle_fade`](prism_render_architecture::particle::oit::soft_particle_fade)).
    SoftParticleFade {
        /// View-space depth of the opaque scene behind the fragment.
        scene_depth: f32,
        /// View-space depth of the fragment itself.
        particle_depth: f32,
        /// Fade contrast (the ramp width in front of the surface).
        contrast: f32,
    },
}

/// The resolved answer for one [`OitQuery`], one variant per query kind.
///
/// Provenance: twinned from this repository's
/// [`oit`](prism_render_architecture::particle::oit); no third-party engine
/// source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OitResult {
    /// The classified composite method, matching `composite_method`.
    CompositeMethod(OitMethod),
    /// The weighted-blended depth weight, matching `wboit_weight`.
    WboitWeight(f32),
    /// The composited frame colour, matching `ResolvedTransparency::over`.
    Over([f32; 3]),
    /// The resolved coverage-weighted colour and coverage, matching
    /// `WboitAccumulator::resolve`.
    WboitResolve {
        /// The coverage-weighted resolved colour.
        color: [f32; 3],
        /// Total coverage `1 - revealage` in `[0, 1]`.
        coverage: f32,
    },
    /// The clamped revealage, matching `WboitAccumulator::revealage`.
    WboitRevealage(f32),
    /// The approximate absorbance, matching `approx_absorbance`.
    ApproxAbsorbance(f32),
    /// The approximate transmittance, matching `approx_transmittance`.
    ApproxTransmittance(f32),
    /// The normalized power moments, matching
    /// `MomentAccumulator::normalized` (`None` for an empty pixel).
    MomentsNormalized(Option<PowerMoments>),
    /// The occluded absorbance fraction, matching `moment_occlusion`.
    MomentOcclusion(f32),
    /// The reconstructed optical depth, matching `reconstruct_optical_depth`.
    ReconstructOpticalDepth(f32),
    /// The reconstructed moment transmittance, matching `moment_transmittance`.
    MomentTransmittance(f32),
    /// The soft-particle fade factor, matching `soft_particle_fade`.
    SoftParticleFade(f32),
}

/// Maps a [`SortStrategy`] to its kernel classification code.
fn strategy_code(strategy: SortStrategy) -> u32 {
    match strategy {
        SortStrategy::None => STRAT_NONE,
        SortStrategy::SharedOit => STRAT_SHARED_OIT,
        SortStrategy::ViewDepthRadix => STRAT_RADIX,
        SortStrategy::ViewDepthBitonic => STRAT_BITONIC,
    }
}

/// Maps a [`ParticleBlend`] to its kernel classification code.
fn blend_code(blend: ParticleBlend) -> u32 {
    match blend {
        ParticleBlend::Opaque => BLEND_OPAQUE,
        ParticleBlend::AlphaMask => BLEND_ALPHA_MASK,
        ParticleBlend::Additive => BLEND_ADDITIVE,
        ParticleBlend::Premultiplied => BLEND_PREMULTIPLIED,
        ParticleBlend::AlphaBlend => BLEND_ALPHA_BLEND,
    }
}

/// Maps an [`OitQuality`] tier to its kernel classification code.
fn tier_code(tier: OitQuality) -> u32 {
    match tier {
        OitQuality::Fast => TIER_FAST,
        OitQuality::Balanced => TIER_BALANCED,
        OitQuality::Reference => TIER_REFERENCE,
    }
}

/// Maps a kernel classification code back to an [`OitMethod`]; every code
/// outside `1..=4` is the pass-through [`OitMethod::None`].
fn method_from_code(code: u32) -> OitMethod {
    match code {
        METHOD_ADDITIVE => OitMethod::Additive,
        METHOD_WEIGHTED_BLENDED => OitMethod::WeightedBlended,
        METHOD_MOMENT_BASED => OitMethod::MomentBased,
        METHOD_PER_PIXEL_LINKED_LIST => OitMethod::PerPixelLinkedList,
        _ => OitMethod::None,
    }
}

/// `repr(C)` `std430` layout of one packed query: four `u32` control words
/// (`op` plus three classification codes), an eight-lane `f32` scalar block,
/// then colour, background and moment payloads of four `f32` each — `96` bytes,
/// matching the `WGSL` `Query` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Op code selecting the routine.
    op: u32,
    /// Sort-strategy classification code.
    strategy: u32,
    /// Blend-mode classification code.
    blend: u32,
    /// Quality-tier classification code.
    tier: u32,
    /// Scalar parameters, meaning per op.
    s: [f32; 8],
    /// Colour / colour-weight input (`xyz`, `w` pad).
    color: [f32; 4],
    /// Background input (`xyz`, `w` pad).
    background: [f32; 4],
    /// Normalized power moments.
    moments: [f32; 4],
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &OitQuery) -> GpuQuery {
        let mut g = GpuQuery::zeroed();
        match query {
            OitQuery::CompositeMethod {
                strategy,
                blend,
                tier,
            } => {
                g.op = OP_COMPOSITE_METHOD;
                g.strategy = strategy_code(*strategy);
                g.blend = blend_code(*blend);
                g.tier = tier_code(*tier);
            }
            OitQuery::WboitWeight {
                view_depth,
                alpha,
                near,
                far,
            } => {
                g.op = OP_WBOIT_WEIGHT;
                g.s[0] = *view_depth;
                g.s[1] = *alpha;
                g.s[2] = *near;
                g.s[3] = *far;
            }
            OitQuery::Over {
                color,
                coverage,
                background,
            } => {
                g.op = OP_OVER;
                g.s[0] = *coverage;
                set_color(&mut g.color, *color);
                set_color(&mut g.background, *background);
            }
            OitQuery::WboitResolve {
                color_weight,
                alpha_weight,
                revealage,
            } => {
                g.op = OP_WBOIT_RESOLVE;
                g.s[0] = *alpha_weight;
                g.s[1] = *revealage;
                set_color(&mut g.color, *color_weight);
            }
            OitQuery::WboitRevealage { revealage } => {
                g.op = OP_WBOIT_REVEALAGE;
                g.s[0] = *revealage;
            }
            OitQuery::ApproxAbsorbance { alpha } => {
                g.op = OP_APPROX_ABSORBANCE;
                g.s[0] = *alpha;
            }
            OitQuery::ApproxTransmittance { optical_depth } => {
                g.op = OP_APPROX_TRANSMITTANCE;
                g.s[0] = *optical_depth;
            }
            OitQuery::MomentsNormalized {
                total,
                m1,
                m2,
                m3,
                m4,
            } => {
                g.op = OP_MOMENTS_NORMALIZED;
                g.s[0] = *total;
                g.s[1] = *m1;
                g.s[2] = *m2;
                g.s[3] = *m3;
                g.s[4] = *m4;
            }
            OitQuery::MomentOcclusion {
                moments,
                depth,
                bias,
            } => {
                g.op = OP_MOMENT_OCCLUSION;
                g.s[0] = *depth;
                g.s[1] = *bias;
                g.moments = *moments;
            }
            OitQuery::ReconstructOpticalDepth {
                total,
                moments,
                depth,
                bias,
            } => {
                g.op = OP_RECONSTRUCT_OPTICAL_DEPTH;
                g.s[0] = *total;
                g.s[1] = *depth;
                g.s[2] = *bias;
                g.moments = *moments;
            }
            OitQuery::MomentTransmittance {
                total,
                moments,
                depth,
                bias,
            } => {
                g.op = OP_MOMENT_TRANSMITTANCE;
                g.s[0] = *total;
                g.s[1] = *depth;
                g.s[2] = *bias;
                g.moments = *moments;
            }
            OitQuery::SoftParticleFade {
                scene_depth,
                particle_depth,
                contrast,
            } => {
                g.op = OP_SOFT_PARTICLE_FADE;
                g.s[0] = *scene_depth;
                g.s[1] = *particle_depth;
                g.s[2] = *contrast;
            }
        }
        g
    }
}

/// Writes an `RGB` triple into a packed four-lane colour slot (`w` stays zero).
fn set_color(slot: &mut [f32; 4], rgb: [f32; 3]) {
    slot[0] = rgb[0];
    slot[1] = rgb[1];
    slot[2] = rgb[2];
}

/// `repr(C)` `std430` layout of one result: two `u32` control words plus two
/// pads, a scalar / coverage / total triple plus a pad, a vector slot plus a
/// pad and the four normalized moments — `64` bytes, matching the `WGSL`
/// `Result` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Composite-method classification code.
    method: u32,
    /// Whether the moment normalization produced a value.
    present: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Single-scalar output.
    scalar: f32,
    /// Resolve coverage output.
    coverage: f32,
    /// Normalized total output.
    total: f32,
    /// Padding lane.
    pad2: f32,
    /// Vector output `x`.
    vx: f32,
    /// Vector output `y`.
    vy: f32,
    /// Vector output `z`.
    vz: f32,
    /// Padding lane.
    pad3: f32,
    /// Normalized power moment `0`.
    b0: f32,
    /// Normalized power moment `1`.
    b1: f32,
    /// Normalized power moment `2`.
    b2: f32,
    /// Normalized power moment `3`.
    b3: f32,
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

/// Decodes one packed [`GpuResult`] into the public [`OitResult`] matching the
/// originating `query`'s variant.
fn decode_result(query: &OitQuery, raw: &GpuResult) -> OitResult {
    match query {
        OitQuery::CompositeMethod { .. } => {
            OitResult::CompositeMethod(method_from_code(raw.method))
        }
        OitQuery::WboitWeight { .. } => OitResult::WboitWeight(raw.scalar),
        OitQuery::Over { .. } => OitResult::Over([raw.vx, raw.vy, raw.vz]),
        OitQuery::WboitResolve { .. } => OitResult::WboitResolve {
            color: [raw.vx, raw.vy, raw.vz],
            coverage: raw.coverage,
        },
        OitQuery::WboitRevealage { .. } => OitResult::WboitRevealage(raw.scalar),
        OitQuery::ApproxAbsorbance { .. } => OitResult::ApproxAbsorbance(raw.scalar),
        OitQuery::ApproxTransmittance { .. } => OitResult::ApproxTransmittance(raw.scalar),
        OitQuery::MomentsNormalized { .. } => {
            if raw.present == 1 {
                OitResult::MomentsNormalized(Some(PowerMoments {
                    total: raw.total,
                    b: [raw.b0, raw.b1, raw.b2, raw.b3],
                }))
            } else {
                OitResult::MomentsNormalized(None)
            }
        }
        OitQuery::MomentOcclusion { .. } => OitResult::MomentOcclusion(raw.scalar),
        OitQuery::ReconstructOpticalDepth { .. } => OitResult::ReconstructOpticalDepth(raw.scalar),
        OitQuery::MomentTransmittance { .. } => OitResult::MomentTransmittance(raw.scalar),
        OitQuery::SoftParticleFade { .. } => OitResult::SoftParticleFade(raw.scalar),
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

/// A compiled, reusable order-independent-transparency compute pipeline.
///
/// Provenance: twinned from this repository's
/// [`oit`](prism_render_architecture::particle::oit); no third-party engine
/// source or derived code.
pub struct GpuOit {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuOit {
    /// Compiles the order-independent-transparency kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOit {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_oit"),
            source: ShaderSource::Wgsl(OIT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_oit_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_oit_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_oit_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOit {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`OitResult`] per input, in
    /// order.
    ///
    /// Each result equals the reference answer for the query's variant to within
    /// the tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[OitQuery]) -> Vec<OitResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_oit_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_oit_output"),
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
            label: Some("prism_volumetric_oit_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_oit_bind_group"),
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
            label: Some("prism_volumetric_oit_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_oit_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_oit_pass"),
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
