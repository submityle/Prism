//! GPU Kajiya-Kay strand highlight (cheap real-time fallback tier).
//!
//! Numerical twin of the CPU golden reference
//! `prism_render_shading::hair_kajiya::evaluate_hair_kajiya_direct`. Where the
//! Marschner (`hair.wesl`), Chiang (`hair_chiang.wesl`) and Yan fibre-level
//! (`hair_fiber.wesl`) front ends model the physical R/TT/TRT lobes and medulla
//! scattering, this cheapest member follows Kajiya & Kay "Rendering Fur with
//! Three Dimensional Textures" (1989): a single anisotropic diffuse term plus
//! one shifted anisotropic specular term about the strand tangent. No
//! absorption, no transmission, no medulla - the affordable rung of the
//! degradation matrix for weak platforms and distant strands.
//!
//! * Diffuse = `sin(T, L)` cylinder diffuse, tinted by the base color.
//! * Specular = `(T'.L)(T'.V) + sin(T',L) sin(T',V)` raised to the shininess
//!   exponent about a tangent `T'` tilted toward the normal by `specular_shift`
//!   (the cheap stand-in for cuticle tilt), tinted white->base by
//!   `specular_tint`.
//!
//! `dot`, `max`, `clamp`, `sqrt`, `pow` and `mix` are WESL builtins;
//! `brdf_normalize_or` is shared with the principled lobe.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};

// Shininess floor so the specular power stays well-defined (mirrors the CPU
// `MIN_SPECULAR_EXPONENT`).
const HAIR_KAJIYA_MIN_SPECULAR_EXPONENT: f32 = 1.0;

// Tunable controls, mirroring the CPU `HairKajiyaParams`.
struct HairKajiyaParams {
    // Diffuse strength `Kd` scaling the `sin(T, L)` cylinder diffuse.
    diffuse_strength: f32,
    // Specular strength `Ks` scaling the anisotropic highlight.
    specular_strength: f32,
    // Specular shininess exponent (`>= 1`); larger is a tighter highlight.
    specular_exponent: f32,
    // Tangent tilt toward the surface normal for the specular lobe.
    specular_shift: f32,
    // Specular tint in [0, 1]: 0 is a white glint, 1 tints fully by base color.
    specular_tint: f32,
}

// `sin` of the angle between two unit-ish vectors from their clamped cosine.
fn hair_kajiya_sin_from_cos(cos_theta: f32) -> f32 {
    return sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
}

// Evaluates the Kajiya-Kay strand highlight for a single analytic light.
// Returns linear radiance plus emissive; only emissive when fully occluded.
fn hair_kajiya_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: HairKajiyaParams,
) -> vec3<f32> {
    let visibility = clamp(light.visibility, 0.0, 1.0);
    if visibility <= 0.0 {
        return surface.emissive;
    }

    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let t = brdf_normalize_or(frame.tangent, vec3<f32>(1.0, 0.0, 0.0));
    let base = surface.base_color;

    // Diffuse: `sin(T, L)` about the raw strand tangent, tinted by the base.
    let t_dot_l = clamp(dot(t, l), -1.0, 1.0);
    let diffuse_term = max(params.diffuse_strength, 0.0) * hair_kajiya_sin_from_cos(t_dot_l);
    let diffuse = base * diffuse_term;

    // Specular: shift the tangent toward the normal (cheap cuticle-tilt stand-in),
    // then evaluate the Kajiya-Kay anisotropic highlight about the shifted axis.
    let t_shifted = brdf_normalize_or(t + n * params.specular_shift, t);
    let ts_dot_l = clamp(dot(t_shifted, l), -1.0, 1.0);
    let ts_dot_v = clamp(dot(t_shifted, v), -1.0, 1.0);
    let spec_cos = clamp(
        ts_dot_l * ts_dot_v
            + hair_kajiya_sin_from_cos(ts_dot_l) * hair_kajiya_sin_from_cos(ts_dot_v),
        0.0,
        1.0,
    );
    let exponent = max(params.specular_exponent, HAIR_KAJIYA_MIN_SPECULAR_EXPONENT);
    let spec_term = max(params.specular_strength, 0.0) * pow(spec_cos, exponent);
    let spec_tint = clamp(params.specular_tint, 0.0, 1.0);
    let spec = mix(vec3<f32>(1.0, 1.0, 1.0), base, spec_tint) * spec_term;

    let scatter = diffuse + spec;
    let radiance = scatter * light.illuminance * visibility;
    return radiance + surface.emissive;
}
