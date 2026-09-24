//! GPU single-layer water BSDF lobe.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::water::evaluate_water_direct`. The closure follows
//! Unreal's `SingleLayerWater` shading model reduced to its analytic
//! direct-lighting terms: a smooth dielectric sheet over a participating
//! medium whose response splits by Fresnel into two energy-conserving paths.
//!
//! * surface specular = the principled dielectric GGX glint (Schlick Fresnel +
//!   Trowbridge-Reitz distribution + height-correlated Smith visibility) with a
//!   water `F0` derived from `reflectance`, sharing the principled helpers so
//!   the highlight matches exactly,
//! * volume scattering = the refracted body gated by `(1 - F)` at the light and
//!   view angles, scattering the authored `base_color` and attenuated by a
//!   wavelength-dependent Beer-Lambert transmittance driven by optical
//!   `thickness`, so deep water darkens and shifts toward its least-absorbed
//!   channel.
//!
//! `dot`, `max`, `clamp` and `exp` are WESL builtins; the GGX/Fresnel helpers
//! and `brdf_normalize_or` are shared with the principled lobe.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
    fresnel_schlick,
    distribution_ggx,
    vis_smith,
};
import prism_render_scene::shaders::lighting::{
    INV_PI,
};

// Floors / weights mirroring `water.rs`.
const WATER_MIN_ROUGHNESS: f32 = 0.045;
const WATER_MIN_N_DOT: f32 = 1.0e-5;
const WATER_DEPTH_SCALE: f32 = 4.0;

// Scalar Schlick Fresnel for the dielectric interface. Mirrors
// `fresnel_scalar` and splits energy between the surface reflection and the
// refracted body.
fn water_fresnel_scalar(f0: f32, cos_theta: f32) -> f32 {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m2 = m * m;
    return f0 + (1.0 - f0) * m2 * m2 * m;
}

// Per-channel Beer-Lambert transmittance `exp(-(1 - albedo) * depth)`. Channels
// the medium absorbs (low albedo) attenuate fastest. Mirrors `transmittance`.
fn water_transmittance(albedo: vec3<f32>, depth: f32) -> vec3<f32> {
    let extinction = max(vec3<f32>(1.0) - albedo, vec3<f32>(0.0));
    return exp(-extinction * depth);
}

// Single-layer water BSDF for a single analytic light. Mirrors
// `evaluate_water_direct`.
fn water_direct(
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
    let h = brdf_normalize_or(v + l, n);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), WATER_MIN_N_DOT);
    let n_dot_h = max(dot(n, h), WATER_MIN_N_DOT);
    let v_dot_h = max(dot(v, h), 0.0);

    let roughness = clamp(surface.perceptual_roughness, WATER_MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    // Dielectric water `F0`; the principled `0.16 * reflectance^2` mapping puts
    // the default reflectance near water's ~0.02 normal-incidence reflectance.
    let f0 = 0.16 * reflectance * reflectance;

    // Surface glint: coloured Schlick Fresnel feeds the shared GGX highlight.
    let f_spec = fresnel_schlick(vec3<f32>(f0), v_dot_h);
    let d = distribution_ggx(n_dot_h, alpha);
    let g = vis_smith(n_dot_v, n_dot_l, alpha);
    let specular = f_spec * (d * g) * n_dot_l;

    // Refracted body: energy entering/leaving the interface is gated by the
    // scalar Fresnel at the light and view angles; the medium scatters the
    // authored albedo and the path is absorbed along the optical depth.
    let f_light = water_fresnel_scalar(f0, n_dot_l);
    let f_view = water_fresnel_scalar(f0, n_dot_v);
    let refracted = (1.0 - f_light) * (1.0 - f_view);
    let depth = clamp(surface.thickness, 0.0, 1.0) * WATER_DEPTH_SCALE;
    let tau = water_transmittance(surface.base_color, depth);
    let scattered = surface.base_color * (refracted * n_dot_l * INV_PI) * tau;

    let lit = specular + scattered;
    let radiance = lit * light.illuminance * visibility;
    return radiance + surface.emissive;
}
