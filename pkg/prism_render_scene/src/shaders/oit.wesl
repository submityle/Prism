//! Weighted-blended order-independent transparency (WBOIT).
//!
//! GPU twin of `prism_render_shading::oit`.  Transparent geometry cannot live
//! in the visibility buffer (one surface per pixel), so it is drawn in a
//! forward pass that accumulates into two render targets with McGuire & Bavoil
//! blending (JCGT 2013):
//!
//! * accumulation target (RGBA16F, additive blend): weighted premultiplied
//!   colour in `rgb`, summed weighted alpha in `a`;
//! * revealage target (R16F, multiplicative blend): running product of
//!   `1 - alpha`.
//!
//! A fullscreen `composite` pass then normalises the accumulation and blends it
//! over the resolved opaque scene colour.  Every constant and expression here
//! matches the CPU golden bit-for-bit within tolerance so the two agree.

// Lower/upper clamps keep the depth weight finite in an f16 target; the epsilon
// guards the average-colour division. These mirror the golden's WEIGHT_MIN,
// WEIGHT_MAX and ACCUM_EPSILON.
const OIT_WEIGHT_MIN: f32 = 1.0e-2;
const OIT_WEIGHT_MAX: f32 = 3.0e3;
const OIT_ACCUM_EPSILON: f32 = 1.0e-5;

// Per-fragment MRT contribution: what the transparent pass writes before the
// hardware blend combines it with the targets.
struct OitTargets {
    // location(0): weighted premultiplied colour (rgb) + weighted alpha (a).
    accum: vec4<f32>,
    // location(1): the (1 - alpha) factor multiplied into the revealage target.
    revealage: f32,
}

// Depth-based fragment weight (McGuire & Bavoil 2013, eq. 10). Nearer fragments
// weigh more; the result is scaled by alpha and clamped to the f16-safe range.
fn oit_weight(view_depth: f32, alpha: f32) -> f32 {
    let depth = max(view_depth, 0.0);
    let scaled = depth * (1.0 / 200.0);
    let falloff = scaled * scaled * scaled * scaled;
    let weight = clamp(0.03 / (OIT_ACCUM_EPSILON + falloff), OIT_WEIGHT_MIN, OIT_WEIGHT_MAX);
    return clamp(alpha, 0.0, 1.0) * weight;
}

// Builds the MRT contribution for one transparent fragment.
fn oit_accumulate(color: vec3<f32>, alpha: f32, view_depth: f32) -> OitTargets {
    let a = clamp(alpha, 0.0, 1.0);
    let weight = oit_weight(view_depth, a);
    var targets: OitTargets;
    targets.accum = vec4<f32>(color * (a * weight), a * weight);
    targets.revealage = 1.0 - a;
    return targets;
}

// Normalises accumulated transparency and composites it over the opaque
// background. Mirrors OitAccumulation::resolve on the CPU.
fn oit_resolve(accum: vec4<f32>, revealage: f32, background: vec3<f32>) -> vec3<f32> {
    let normaliser = max(accum.a, OIT_ACCUM_EPSILON);
    let average = accum.rgb / normaliser;
    let reveal = clamp(revealage, 0.0, 1.0);
    let coverage = 1.0 - reveal;
    return average * coverage + background * reveal;
}

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle, matching composite.wesl. No vertex buffer is bound.
@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    var out: FullscreenVertexOutput;
    out.uv = uv;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    return out;
}

// The source colour the composite pass emits so that fixed-function
// SrcAlpha/OneMinusSrcAlpha blending over the view target reproduces
// oit_resolve exactly, without the pass sampling the opaque background.
// Returns rgb = normalised average colour, a = coverage (= 1 - revealage).
// Mirrors OitAccumulation::composite_source on the CPU.
fn oit_composite_source(accum: vec4<f32>, revealage: f32) -> vec4<f32> {
    let normaliser = max(accum.a, OIT_ACCUM_EPSILON);
    let average = accum.rgb / normaliser;
    let reveal = clamp(revealage, 0.0, 1.0);
    return vec4<f32>(average, 1.0 - reveal);
}

// group 0: the two WBOIT targets the transparent forward pass wrote, read by
// integer textureLoad for a 1:1 pixel composite. The opaque background is the
// bound view target and is combined by fixed-function blending, not sampled.
@group(0) @binding(0) var oit_accum: texture_2d<f32>;
@group(0) @binding(1) var oit_revealage: texture_2d<f32>;

// Fullscreen composite: emit the blend source (average colour + coverage) so
// the pipeline SrcAlpha/OneMinusSrcAlpha blend lays transparency over the
// already-composited opaque view target. Pixels with no transparent coverage
// emit coverage 0 and leave the target untouched.
@fragment
fn composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(in.position.xy);
    let accum = textureLoad(oit_accum, coord, 0);
    let revealage = textureLoad(oit_revealage, coord, 0).r;
    return oit_composite_source(accum, revealage);
}
