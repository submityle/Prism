//! Runtime dynamic channels: the analytic modulation of the *frozen* über
//! surface (§17.1, roadmap P2-g). GPU twin of
//! `prism_render_shading::dynamic`.
//!
//! The bake freezes a material's static über fields; a runtime state change
//! (wetness / snow / damage / dissolve) then **remaps those fields analytically**
//! instead of adding lobes, bakes or a texture blend. No new permutation, no
//! per-pixel budget gamble, and — because every remap is closed-form — the CPU
//! golden and this twin stay comparable bit for bit within 1 ULP.
//!
//! Contract (identical on both lines):
//! * `0` per channel is a bit-for-bit identity, and the bypass branch is taken
//!   per material, not per pixel, so a wavefront never diverges on it.
//! * Channels apply in a fixed order — `wetness → snow → damage → dissolve` — so
//!   "wet snow" cannot drift apart between the two implementations.
//! * No new author parameters: the remaps use fixed, physically-motivated
//!   targets (water's index, snow's albedo, a chip's roughness).
//! * `dyn_mix` is spelled `a*(1-t) + b*t`, *not* the built-in `mix`, so both
//!   endpoints are exact (full coverage lands on the target value bit for bit).
//!
//! The dissolve *coverage* half lives in `dynamic_mask.wesl`, because alpha is
//! consumed by the raster alpha-test / blend passes, not by this radiance path.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
};
import prism_render_scene::shaders::dynamic_mask::{
    dyn_dissolve_coverage,
};

/// Relative darkening of the base colour under a full water film (§17.1):
/// water fills the pores, so diffuse scattering becomes surface reflection.
const WET_BASE_DARKEN: f32 = 0.55;
/// A film is smoother than the substrate it covers, so highlights tighten.
const WET_ROUGHNESS_SCALE: f32 = 0.35;
/// Refractive index of the water film (20 °C, 589 nm).
const WATER_IOR: f32 = 1.333;

/// Diffuse albedo of fresh snow (slightly cool, as snow is).
const SNOW_ALBEDO: vec3<f32> = vec3<f32>(0.95, 0.97, 1.0);
const SNOW_ROUGHNESS: f32 = 0.9;
const SNOW_DIFFUSE_ROUGHNESS: f32 = 0.9;
/// Snow is a dielectric powder, not a lacquer.
const SNOW_SPECULAR_WEIGHT: f32 = 0.35;
/// Sheen a snow cover retains from its granular top layer.
const SNOW_SHEEN: f32 = 0.15;

/// Relative darkening of the base colour under full wear (oxidation, grime).
const DAMAGE_BASE_DARKEN: f32 = 0.45;
/// Fraction of the metal interface destroyed by full wear.
const DAMAGE_METAL_LOSS: f32 = 0.6;
/// How far full wear pushes the specular roughness (`d * this` towards 1).
const DAMAGE_ROUGHEN: f32 = 0.6;

/// Peak emissive gain of the cut line, relative to the material's own emission.
const DISSOLVE_EDGE_GAIN: f32 = 4.0;

/// `a*(1 - t) + b*t`: exact at both endpoints, unlike the built-in `mix`.
fn dyn_mix(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

/// Apply the four runtime dynamic channels to `surface` (§17.1).
///
/// `surface_cell` is the same shade-time cell the glint seed quantises (the
/// surface UV or a tangent-frame projection of the position); it drives the
/// procedural dissolve mask through `dyn_dissolve_coverage`.
fn apply_dynamic_channels(
    surface: SurfaceSample,
    channels: vec4<f32>,
    surface_cell: vec2<f32>,
) -> SurfaceSample {
    var s = surface;

    let wetness = clamp(channels.x, 0.0, 1.0);
    if (wetness > 0.0) {
        let darken = dyn_mix(1.0, WET_BASE_DARKEN, wetness);
        s.base_color = vec3<f32>(
            s.base_color.x * darken,
            s.base_color.y * darken,
            s.base_color.z * darken,
        );
        s.diffuse_roughness = dyn_mix(s.diffuse_roughness, 0.0, wetness);
        s.perceptual_roughness =
            dyn_mix(s.perceptual_roughness, s.perceptual_roughness * WET_ROUGHNESS_SCALE, wetness);
        s.specular_weight = dyn_mix(s.specular_weight, 1.0, wetness);
        s.specular_ior = dyn_mix(s.specular_ior, WATER_IOR, wetness);
        s.sheen = s.sheen * (1.0 - wetness);
    }

    let snow = clamp(channels.y, 0.0, 1.0);
    if (snow > 0.0) {
        s.base_color = vec3<f32>(
            dyn_mix(s.base_color.x, SNOW_ALBEDO.x, snow),
            dyn_mix(s.base_color.y, SNOW_ALBEDO.y, snow),
            dyn_mix(s.base_color.z, SNOW_ALBEDO.z, snow),
        );
        s.metallic = dyn_mix(s.metallic, 0.0, snow);
        s.perceptual_roughness = dyn_mix(s.perceptual_roughness, SNOW_ROUGHNESS, snow);
        s.diffuse_roughness = dyn_mix(s.diffuse_roughness, SNOW_DIFFUSE_ROUGHNESS, snow);
        s.specular_weight = dyn_mix(s.specular_weight, SNOW_SPECULAR_WEIGHT, snow);
        s.sheen = dyn_mix(s.sheen, SNOW_SHEEN, snow);
    }

    let damage = clamp(channels.z, 0.0, 1.0);
    if (damage > 0.0) {
        let darken = dyn_mix(1.0, DAMAGE_BASE_DARKEN, damage);
        s.base_color = vec3<f32>(
            s.base_color.x * darken,
            s.base_color.y * darken,
            s.base_color.z * darken,
        );
        s.metallic = s.metallic * (1.0 - DAMAGE_METAL_LOSS * damage);
        s.perceptual_roughness = dyn_mix(s.perceptual_roughness, 1.0, damage * DAMAGE_ROUGHEN);
        // Chipped wear breaks the coat open.
        s.clearcoat = s.clearcoat * (1.0 - damage);
    }

    let dissolve = clamp(channels.w, 0.0, 1.0);
    if (dissolve > 0.0) {
        // The *coverage* half of the cut lives where alpha is consumed (the
        // raster alpha-test / blend passes call `dyn_dissolve_coverage`); this
        // struct has no opacity field, because the shading resolve produces
        // radiance, not coverage. What belongs here is the cut *line*: the band
        // where coverage is between 0 and 1 is lit by the material's own
        // emission, so the rim takes the colour the author already chose (and a
        // non-emissive material gets a clean cut with no rim at all).
        let coverage = dyn_dissolve_coverage(dissolve, surface_cell);
        let rim = DISSOLVE_EDGE_GAIN * coverage * (1.0 - coverage);
        s.emissive = s.emissive * (1.0 + rim);
    }

    return s;
}
