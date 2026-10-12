//! GPU subsurface scattering (SSS) BSDF lobe.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::subsurface::evaluate_subsurface_direct`.  The closure
//! follows the real-time subsurface model used by Unreal's *Subsurface*
//! shading model:
//!
//! * specular = principled dielectric GGX lobe (shared `fresnel_schlick`,
//!   `distribution_ggx`, `vis_smith` helpers), so the highlight is identical to
//!   the principled lobe,
//! * diffuse = energy-conserving wrap whose width is the `subsurface` weight
//!   (collapses to a plain Lambert cosine at `subsurface == 0`),
//! * back transmission = Frostbite/DICE "fast subsurface scattering" fit: a
//!   view-dependent lobe around the inverted, normal-distorted light vector,
//!   scaled by `subsurface` and attenuated by optical `thickness`.
//!
//! There is deliberately no back-facing early-out so radiance can transmit
//! through thin geometry from lights behind the surface.
//!
//! `pow`, `dot`, `max`, `min`, `clamp`, `mix` and `inverseSqrt` are WESL
//! builtins.

import prism_render_scene::shaders::lighting::{
    INV_PI,
};
import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
    dielectric_f0,
    fresnel_schlick,
    distribution_ggx,
    vis_smith,
};

// Floors + transmission constants mirroring `subsurface.rs`.
const SUBSURFACE_MIN_ROUGHNESS: f32 = 0.045;
const SUBSURFACE_MIN_N_DOT: f32 = 1.0e-5;
const SUBSURFACE_DISTORTION: f32 = 0.2;
const SUBSURFACE_POWER: f32 = 12.0;

// Energy-conserving diffuse wrap. Mirrors `fd_wrap`.
fn subsurface_fd_wrap(x: f32, w: f32) -> f32 {
    return clamp((x + w) / ((1.0 + w) * (1.0 + w)), 0.0, 1.0);
}

// Subsurface BSDF for a single analytic light. Mirrors
// `evaluate_subsurface_direct`.
fn subsurface_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);

    let raw_n_dot_l = dot(n, l);
    let n_dot_l = max(raw_n_dot_l, 0.0);
    let n_dot_v = max(dot(n, v), SUBSURFACE_MIN_N_DOT);
    let n_dot_h = max(dot(n, h), SUBSURFACE_MIN_N_DOT);
    let v_dot_h = max(dot(v, h), 0.0);

    let subsurface = clamp(surface.subsurface, 0.0, 1.0);
    let thickness = clamp(surface.thickness, 0.0, 1.0);
    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, SUBSURFACE_MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let f0 = mix(dielectric_f0(surface), surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);

    // Front-face GGX specular; reuses the principled helpers for exact parity.
    let d = distribution_ggx(n_dot_h, alpha);
    let g = vis_smith(n_dot_v, n_dot_l, alpha);
    let specular = f * (d * g) * n_dot_l;

    // Wrapped diffuse: the wrap width is the subsurface weight scaled by the
    // per-channel `subsurface_radius` (coloured terminator bleed), tinted by
    // `subsurface_color`. Both default to the identity.
    let scat_color = surface.subsurface_color;
    let diffuse_weight = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    var diffuse = vec3<f32>(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {
        // `subsurface_radius` (scalar) times `subsurface_radius_scale` (color3)
        // gives the per-channel scattering extent.
        let wrap = clamp(
            subsurface * surface.subsurface_radius * surface.subsurface_radius_scale[c],
            0.0,
            1.0,
        );
        // Normalise the wrap lobe to unit albedo: its analytic directional albedo is
        // `E(w) = (1 + 2w)/(1 + w)²` (1 at `w = 0`, 0.75 at `w = 1`), so an
        // unnormalised wrap silently loses energy per channel. Mirrors
        // `subsurface.rs` and the specification's `subsurface_albedo_constraint`.
        let one_plus_w = 1.0 + wrap;
        let e_wrap = max((1.0 + 2.0 * wrap) / (one_plus_w * one_plus_w), 1.0e-3);
        let diffuse_cos = subsurface_fd_wrap(raw_n_dot_l, wrap) / e_wrap;
        diffuse[c] = surface.base_color[c]
            * scat_color[c]
            * INV_PI
            * diffuse_cos
            * diffuse_weight[c];
    }

    // Back transmission through thin geometry.
    let h_transmission = brdf_normalize_or(l + n * SUBSURFACE_DISTORTION, n);
    // `OpenPBR` `scatter_anisotropy` shapes the transmitted lobe: forward (+)
    // sharpens, backward (-) broadens. `0.0` is the historical constant.
    let sharpness = max(
        SUBSURFACE_POWER * (1.0 + clamp(surface.scatter_anisotropy, -1.0, 1.0)),
        SUBSURFACE_MIN_N_DOT,
    );
    let back = pow(clamp(dot(v, -h_transmission), 0.0, 1.0), sharpness);
    let thickness_attenuation = 1.0 - thickness;
    let transmitted = (surface.base_color * scat_color)
        * (subsurface * back * thickness_attenuation);

    let lit = diffuse + specular + transmitted;
    let radiance = lit * light.illuminance * clamp(light.visibility, 0.0, 1.0);
    return radiance + surface.emissive;
}
