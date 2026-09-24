//! GPU hair (strand) BSDF lobe.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::hair::evaluate_hair_direct`.  The closure follows the
//! real-time strand model used across production shaders (and Unreal's hair
//! path): a Kajiya-Kay translucent diffuse plus two shifted Scheuermann strand
//! highlights.
//!
//! * diffuse = Kajiya-Kay `sin(T, L)` softened by a half-Lambert wrap so the
//!   fibre still lifts under backward grazing light, tinted by the base color,
//! * specular = a sharp white *primary* (R) glint shifted toward the root plus
//!   a broad *secondary* (TRT) glint shifted the other way and tinted by the
//!   fibre color (that energy transmitted through the strand). Each is
//!   `smoothstep`-attenuated `pow(sin(T', H), exponent)`.
//!
//! `dot`, `max`, `clamp`, `mix`, `smoothstep`, `sqrt` and `pow` are WESL
//! builtins; `brdf_normalize_or` is shared with the principled lobe.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};
import prism_render_scene::shaders::lighting::{
    INV_PI,
};

// Floors / weights mirroring `hair.rs`.
const HAIR_MIN_ROUGHNESS: f32 = 0.045;
const HAIR_PRIMARY_SHIFT: f32 = -0.08;
const HAIR_SECONDARY_SHIFT: f32 = 0.10;
const HAIR_PRIMARY_EXP_MIN: f32 = 16.0;
const HAIR_PRIMARY_EXP_MAX: f32 = 160.0;
const HAIR_SECONDARY_EXP_MIN: f32 = 8.0;
const HAIR_SECONDARY_EXP_MAX: f32 = 80.0;
const HAIR_PRIMARY_WEIGHT: f32 = 0.20;
const HAIR_SECONDARY_WEIGHT: f32 = 0.35;

// Shifts the strand tangent along the normal and re-normalizes. Mirrors
// `shift_tangent`.
fn hair_shift_tangent(tangent: vec3<f32>, normal: vec3<f32>, shift: f32) -> vec3<f32> {
    return brdf_normalize_or(tangent + normal * shift, tangent);
}

// Scheuermann shifted strand specular. Mirrors `strand_specular`.
fn hair_strand_specular(tangent: vec3<f32>, half_vector: vec3<f32>, exponent: f32) -> f32 {
    let dot_th = dot(tangent, half_vector);
    let sin_th = sqrt(max(1.0 - dot_th * dot_th, 0.0));
    let directional = smoothstep(-1.0, 0.0, dot_th);
    return directional * pow(sin_th, exponent);
}

// Hair BSDF for a single analytic light. Mirrors `evaluate_hair_direct`.
fn hair_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let visibility = clamp(light.visibility, 0.0, 1.0);
    if visibility <= 0.0 {
        return surface.emissive;
    }

    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let t = brdf_normalize_or(frame.tangent, vec3<f32>(1.0, 0.0, 0.0));
    let h = brdf_normalize_or(v + l, n);

    let roughness = clamp(surface.perceptual_roughness, HAIR_MIN_ROUGHNESS, 1.0);
    let gloss = 1.0 - roughness;
    let primary_exp = mix(HAIR_PRIMARY_EXP_MIN, HAIR_PRIMARY_EXP_MAX, gloss);
    let secondary_exp = mix(HAIR_SECONDARY_EXP_MIN, HAIR_SECONDARY_EXP_MAX, gloss);

    // Kajiya-Kay translucent diffuse.
    let dot_nl = dot(n, l);
    let wrap = clamp(dot_nl * 0.5 + 0.5, 0.0, 1.0);
    let dot_tl = dot(t, l);
    let sin_tl = sqrt(max(1.0 - dot_tl * dot_tl, 0.0));
    let diffuse = surface.base_color * (INV_PI * wrap * sin_tl);

    // Two shifted strand highlights: white primary + fibre-tinted secondary.
    let t_primary = hair_shift_tangent(t, n, HAIR_PRIMARY_SHIFT);
    let t_secondary = hair_shift_tangent(t, n, HAIR_SECONDARY_SHIFT);
    let spec_primary = hair_strand_specular(t_primary, h, primary_exp);
    let spec_secondary = hair_strand_specular(t_secondary, h, secondary_exp);
    let primary = vec3<f32>(spec_primary * HAIR_PRIMARY_WEIGHT);
    let secondary = surface.base_color * (spec_secondary * HAIR_SECONDARY_WEIGHT);
    let specular = primary + secondary;

    let radiance = (diffuse + specular) * light.illuminance * visibility;
    return radiance + surface.emissive;
}
