//! GPU stylized (NPR) hair shading front end - the anime "angel-ring" look.
//!
//! Numerical twin of the CPU golden reference
//! `prism_render_shading::stylized_hair::evaluate_stylized_hair_direct`. Where
//! `hair.wesl` is the physically based Marschner R/TT/TRT strand BSDF, this is
//! its non-photoreal sibling: the cel-shaded hair whose signature is the sharp,
//! banded anisotropic highlight ring (the "angel ring") that slides around the
//! head as the view moves.
//!
//! It layers three responses, each mapped one-to-one onto the CPU reference:
//!
//! * a wrapped (half-Lambert) diffuse cosine quantized into `bands` cel steps
//!   with softenable edges, modulated by the base color;
//! * two shifted, thresholded `Kajiya-Kay` anisotropic highlight bands (a tight
//!   white primary ring shifted root-ward and a broad tinted secondary ring
//!   shifted tip-ward, Scheuermann's two-tone hair specular), each pushed
//!   through a `smoothstep` so it reads as a crisp ink-edged ring; and
//! * a view Fresnel rim gated to the lit hemisphere that deliberately survives
//!   shadow so back-lit silhouettes stay readable.
//!
//! `dot`, `max`, `min`, `clamp`, `sqrt`, `pow` and `floor` are WESL builtins;
//! `brdf_normalize_or` is shared with the principled lobe. The custom
//! `sh_smoothstep`/`sh_mix` reproduce the CPU helpers' exact collapsed-edge and
//! lerp behaviour rather than relying on the builtins' edge cases.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};

// Roughness floor shared with the principled lobe. Mirrors `MIN_ROUGHNESS`.
const STYLIZED_HAIR_MIN_ROUGHNESS: f32 = 0.045;

// Tunable controls for the stylized hair front end. Field-for-field twin of the
// CPU `StylizedHairParams`.
struct StylizedHairParams {
    // Number of cel quantization bands for the diffuse term (`>= 1`).
    bands: u32,
    // Cosine wrap in [0, 1]; 0 is pure Lambert, 0.5 is half-Lambert.
    wrap: f32,
    // Softness of the diffuse band transitions in [0, 1].
    ramp_softness: f32,
    // Tangent shift of the primary highlight along the normal (root-ward).
    primary_shift: f32,
    // Sharpness exponent of the primary highlight.
    primary_exponent: f32,
    // Intensity of the primary highlight; 0 disables it.
    primary_intensity: f32,
    // Threshold in [0, 1] where the primary ring flips on.
    primary_threshold: f32,
    // Half-width of the primary ring edge.
    primary_softness: f32,
    // Linear tint of the primary highlight (classically near-white).
    primary_color: vec3<f32>,
    // Tangent shift of the secondary highlight along the normal (tip-ward).
    secondary_shift: f32,
    // Sharpness exponent of the secondary highlight.
    secondary_exponent: f32,
    // Intensity of the secondary highlight; 0 disables it.
    secondary_intensity: f32,
    // Threshold in [0, 1] where the secondary ring flips on.
    secondary_threshold: f32,
    // Half-width of the secondary ring edge.
    secondary_softness: f32,
    // Linear tint of the secondary highlight (classically hair-toned).
    secondary_color: vec3<f32>,
    // Rim (edge) light intensity; 0 disables the rim.
    rim_intensity: f32,
    // Fresnel exponent controlling how tightly the rim hugs the silhouette.
    rim_power: f32,
    // Linear tint of the rim light.
    rim_color: vec3<f32>,
}

// Scalar lerp matching the CPU `mix` helper (`a + (b - a) * t`) exactly.
fn sh_mix(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Hermite smoothstep matching the CPU helper: a collapsed or inverted edge
// pair degrades to a hard step at `edge0` rather than the builtin's undefined
// behaviour.
fn sh_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        if x < edge0 {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Re-shapes the cosine term into an optionally wrapped, quantized cel ramp.
// Mirrors `hair_ramp`.
fn stylized_hair_ramp(n_dot_l: f32, bands: u32, wrap: f32, softness: f32) -> f32 {
    let steps = f32(max(bands, 1u));
    let w = clamp(wrap, 0.0, 1.0);
    var lit: f32;
    if w > 0.0 {
        lit = clamp((n_dot_l + w) / (1.0 + w), 0.0, 1.0);
    } else {
        lit = max(n_dot_l, 0.0);
    }
    let scaled = lit * steps;
    let level = floor(scaled);
    let lower = level / steps;
    if softness <= 0.0 {
        return lower;
    }
    let frac = scaled - level;
    let upper = min((level + 1.0) / steps, 1.0);
    let edge = clamp(softness * 0.5, 0.0, 0.5);
    let t = sh_smoothstep(0.5 - edge, 0.5 + edge, frac);
    return sh_mix(lower, upper, t);
}

// One shifted, thresholded `Kajiya-Kay` anisotropic highlight band. Mirrors
// `hair_highlight_band`.
fn stylized_hair_highlight_band(
    tangent: vec3<f32>,
    normal: vec3<f32>,
    half_vec: vec3<f32>,
    shift: f32,
    exponent: f32,
    threshold: f32,
    softness: f32,
) -> f32 {
    let shifted = brdf_normalize_or(tangent + normal * shift, tangent);
    let t_dot_h = clamp(dot(shifted, half_vec), -1.0, 1.0);
    let sin_t_h = sqrt(max(1.0 - t_dot_h * t_dot_h, 0.0));
    let raw = pow(sin_t_h, max(exponent, 1.0));
    let edge = clamp(softness * 0.5, 0.0, 0.5);
    return sh_smoothstep(threshold - edge, threshold + edge, raw);
}

// Evaluates the stylized hair front end for a single analytic light. Mirrors
// `evaluate_stylized_hair_direct`: the diffuse ramp and highlight rings are
// gated to the lit hemisphere and attenuated by the clamped visibility, the rim
// survives shadow, and emissive is folded in exactly once.
fn stylized_hair_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: StylizedHairParams,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let t = brdf_normalize_or(frame.tangent, vec3<f32>(1.0, 0.0, 0.0));

    let visibility = clamp(light.visibility, 0.0, 1.0);
    let n_dot_l = dot(n, l);
    let lit_gate = max(n_dot_l, 0.0);

    // Diffuse cel ramp modulated by base color, illuminance and visibility.
    let ramp = stylized_hair_ramp(n_dot_l, params.bands, params.wrap, params.ramp_softness);
    var color = surface.base_color * light.illuminance * (ramp * visibility);

    if lit_gate > 0.0 {
        let h = brdf_normalize_or(v + l, n);
        // A glossier strand tightens both rings, matching the PBR sibling.
        let roughness = clamp(surface.perceptual_roughness, STYLIZED_HAIR_MIN_ROUGHNESS, 1.0);
        let gloss = 1.0 - roughness;

        // Primary white ring, shifted toward the root.
        if params.primary_intensity > 0.0 {
            let exponent = params.primary_exponent * sh_mix(0.5, 1.0, gloss);
            let band = stylized_hair_highlight_band(
                t,
                n,
                h,
                params.primary_shift,
                exponent,
                params.primary_threshold,
                params.primary_softness,
            );
            let weight = params.primary_intensity * band * lit_gate * visibility;
            color = color + params.primary_color * weight * light.illuminance;
        }

        // Secondary tinted ring, shifted toward the tip.
        if params.secondary_intensity > 0.0 {
            let exponent = params.secondary_exponent * sh_mix(0.5, 1.0, gloss);
            let band = stylized_hair_highlight_band(
                t,
                n,
                h,
                params.secondary_shift,
                exponent,
                params.secondary_threshold,
                params.secondary_softness,
            );
            let weight = params.secondary_intensity * band * lit_gate * visibility;
            let tint = params.secondary_color * surface.base_color;
            color = color + tint * weight * light.illuminance;
        }
    }

    // Rim / edge light: a view Fresnel gated to the lit hemisphere, unshadowed.
    if params.rim_intensity > 0.0 && lit_gate > 0.0 {
        let n_dot_v = max(dot(n, v), 0.0);
        let rim = pow(max(1.0 - n_dot_v, 0.0), max(params.rim_power, 0.0));
        let weight = params.rim_intensity * rim * lit_gate;
        color = color + params.rim_color * weight * light.illuminance;
    }

    return color + surface.emissive;
}
