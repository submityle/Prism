//! GPU stylized (NPR) multi-ring "angel ring" highlight stack for anime hair.
//!
//! Numerical twin of the CPU golden reference
//! `prism_render_shading::hair_angel_ring` (`accumulate_angel_rings` /
//! `angel_ring_band`). Where `stylized_hair.wesl` ships a fixed two-ring anime
//! response (a tight white primary band plus a broad tinted secondary band),
//! this kernel owns the N-ring extension: an ordered, fixed-capacity stack of
//! independently tuned `AngelRing` bands whose additive contribution composes
//! on top of the diffuse cel ramp produced elsewhere.
//!
//! Each ring is a shifted, thresholded `Kajiya-Kay` anisotropic highlight: the
//! `sin` of the angle between a (shifted) strand tangent and the half vector is
//! raised to a sharpness exponent and pushed through a `smoothstep` threshold
//! so it reads as a crisp ink-edged band. At `tangent_decouple == 0` the band
//! math is value-for-value identical to `stylized_hair.wesl`'s
//! `stylized_hair_highlight_band`, so the two front ends stay coherent.
//!
//! **Tangent decoupling** is the angel-ring signature: `tangent_decouple` in
//! `[0, 1]` biases the strand tangent toward a fixed `reference_axis` (typically
//! the head-up direction in the shading frame) so the ring sits as a stable
//! halo across the head regardless of strand curl. `0` follows the strand,
//! matching the base module; `1` pins the band to the reference axis.
//!
//! `dot`, `max`, `min`, `clamp`, `sqrt` and `pow` are WESL builtins;
//! `brdf_normalize_or` is shared with the principled lobe. The custom
//! `ar_smoothstep`/`ar_mix` reproduce the CPU helpers' exact collapsed-edge and
//! lerp behaviour rather than relying on the builtins' edge cases.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};

// Roughness floor shared with the principled/hair lobes. Mirrors
// `MIN_ROUGHNESS`.
const ANGEL_RING_MIN_ROUGHNESS: f32 = 0.045;

// Maximum rings the GPU-facing stack sizes to. Mirrors `MAX_ANGEL_RINGS`.
const MAX_ANGEL_RINGS: u32 = 8u;

// One stylized anisotropic highlight band. Field-for-field twin of the CPU
// `AngelRing` (scalars grouped ahead of the `vec3` tint to keep std140-friendly
// packing predictable).
struct AngelRing {
    // Tangent shift along the normal that places the band.
    shift: f32,
    // Sharpness exponent (larger => tighter band); effectively `>= 1`.
    exponent: f32,
    // Band intensity; 0 disables this ring.
    intensity: f32,
    // `smoothstep` center in [0, 1] where the band flips on.
    threshold: f32,
    // Half-width of the band edge; 0 yields a hard ink edge.
    softness: f32,
    // Blend of the strand tangent toward the reference axis in [0, 1].
    tangent_decouple: f32,
    // Linear tint of the band.
    color: vec3<f32>,
}

// A fixed-capacity ordered ring stack plus the count of active rings.
struct AngelRingStack {
    // Number of active rings in `rings` (clamped to `MAX_ANGEL_RINGS`).
    ring_count: u32,
    rings: array<AngelRing, 8>,
}

// Scalar lerp matching the CPU `mix` helper (`a + (b - a) * t`) exactly.
fn ar_mix(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Hermite smoothstep matching the CPU helper: a collapsed or inverted edge pair
// degrades to a hard step at `edge0` rather than the builtin's undefined
// behaviour.
fn ar_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        if x < edge0 {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Evaluates a single ring's scalar band weight in [0, 1], before tint and
// intensity. Mirrors `angel_ring_band`: a zero-length `reference_axis` falls
// back to the strand tangent, so any decouple factor collapses to strand-flow.
fn angel_ring_band(
    ring: AngelRing,
    tangent: vec3<f32>,
    reference_axis: vec3<f32>,
    normal: vec3<f32>,
    half_vec: vec3<f32>,
    gloss: f32,
) -> f32 {
    let decouple = clamp(ring.tangent_decouple, 0.0, 1.0);
    let axis = brdf_normalize_or(reference_axis, tangent);
    // Bias the strand tangent toward the reference axis; `decouple == 0` returns
    // the strand tangent exactly, keeping parity with the base two-ring module.
    let biased = brdf_normalize_or(tangent + (axis - tangent) * decouple, tangent);
    let shifted = brdf_normalize_or(biased + normal * ring.shift, biased);
    let t_dot_h = clamp(dot(shifted, half_vec), -1.0, 1.0);
    let sin_t_h = sqrt(max(1.0 - t_dot_h * t_dot_h, 0.0));
    let exponent = max(ring.exponent * ar_mix(0.5, 1.0, clamp(gloss, 0.0, 1.0)), 1.0);
    let raw = pow(sin_t_h, exponent);
    let edge = clamp(ring.softness * 0.5, 0.0, 0.5);
    return ar_smoothstep(ring.threshold - edge, ring.threshold + edge, raw);
}

// Accumulates the additive highlight radiance of an angel-ring stack for one
// analytic light. Mirrors `accumulate_angel_rings`: the whole stack is gated to
// the lit hemisphere and attenuated by the clamped visibility (a fully occluded
// or back-lit light yields black), rings are summed in order and disabled rings
// are skipped. Returns only the summed ring contribution; the caller adds it on
// top of the diffuse cel ramp and emissive.
fn accumulate_angel_rings(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    reference_axis: vec3<f32>,
    stack: AngelRingStack,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let t = brdf_normalize_or(frame.tangent, vec3<f32>(1.0, 0.0, 0.0));

    let visibility = clamp(light.visibility, 0.0, 1.0);
    let lit_gate = max(dot(n, l), 0.0);
    var color = vec3<f32>(0.0, 0.0, 0.0);
    if lit_gate <= 0.0 || visibility <= 0.0 {
        return color;
    }

    let h = brdf_normalize_or(v + l, n);
    let roughness = clamp(surface.perceptual_roughness, ANGEL_RING_MIN_ROUGHNESS, 1.0);
    let gloss = 1.0 - roughness;

    let count = min(stack.ring_count, MAX_ANGEL_RINGS);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let ring = stack.rings[i];
        if ring.intensity <= 0.0 {
            continue;
        }
        let band = angel_ring_band(ring, t, reference_axis, n, h, gloss);
        let weight = max(ring.intensity, 0.0) * band * lit_gate * visibility;
        if weight <= 0.0 {
            continue;
        }
        // Tint the band weight and fold in the analytic light color, then
        // accumulate additively across the stack.
        let tinted = max(ring.color, vec3<f32>(0.0, 0.0, 0.0)) * weight * light.illuminance;
        color = color + tinted;
    }
    return color;
}
