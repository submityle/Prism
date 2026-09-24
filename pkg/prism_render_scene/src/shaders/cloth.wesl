//! GPU cloth (fabric) BSDF lobe.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::cloth::evaluate_cloth_direct`.  The closure follows
//! the physically based cloth model popularised by Filament / Unreal:
//!
//! * sheen specular = Estevez-Kulla *Charlie* distribution * Ashikhmin/Neubelt
//!   visibility, tinted by the scalar sheen weight, with **no Fresnel** term,
//! * diffuse = Lambertian base, optionally softened by an energy-conserving
//!   wrap term and tinted toward the base color when subsurface > 0.
//!
//! `pow`, `dot`, `max`, `clamp` and `inverseSqrt` are WESL builtins.

import prism_render_scene::shaders::lighting::{
    PI,
    INV_PI,
};
import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};

// Floors mirroring `cloth.rs` (`MIN_ROUGHNESS`, `MIN_N_DOT`, `MIN_SIN2`).
const CLOTH_MIN_ROUGHNESS: f32 = 0.045;
const CLOTH_MIN_N_DOT: f32 = 1.0e-5;
const CLOTH_MIN_SIN2: f32 = 0.0078125;
const CLOTH_WRAP_WIDTH: f32 = 0.5;

// Estevez-Kulla *Charlie* sheen distribution. Mirrors `distribution_charlie`.
fn cloth_distribution_charlie(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / roughness;
    let sin2h = max(1.0 - n_dot_h * n_dot_h, CLOTH_MIN_SIN2);
    return (2.0 + inv_r) * pow(sin2h, inv_r * 0.5) * (1.0 / (2.0 * PI));
}

// Ashikhmin/Neubelt visibility term. Mirrors `visibility_ashikhmin`.
fn cloth_visibility_ashikhmin(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), CLOTH_MIN_N_DOT);
}

// Energy-conserving diffuse wrap. Mirrors `fd_wrap`.
fn cloth_fd_wrap(x: f32, w: f32) -> f32 {
    return clamp((x + w) / ((1.0 + w) * (1.0 + w)), 0.0, 1.0);
}

// Cloth BSDF for a single analytic light. Mirrors `evaluate_cloth_direct`.
fn cloth_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);

    let n_dot_l = max(dot(n, l), 0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }
    let n_dot_v = max(dot(n, v), CLOTH_MIN_N_DOT);
    let n_dot_h = max(dot(n, h), CLOTH_MIN_N_DOT);
    let roughness = clamp(surface.perceptual_roughness, CLOTH_MIN_ROUGHNESS, 1.0);

    // Sheen specular lobe (no Fresnel); sheen_color = vec3(sheen).
    let sheen = clamp(surface.sheen, 0.0, 1.0);
    let d = cloth_distribution_charlie(roughness, n_dot_h);
    let vis = cloth_visibility_ashikhmin(n_dot_v, n_dot_l);
    let sheen_specular = vec3<f32>(sheen * d * vis);

    // Diffuse lobe, optionally softened + tinted by the subsurface weight.
    let subsurface = clamp(surface.subsurface, 0.0, 1.0);
    var diffuse = surface.base_color * INV_PI;
    if subsurface > 0.0 {
        diffuse = diffuse * cloth_fd_wrap(n_dot_l, CLOTH_WRAP_WIDTH);
        let tint = clamp(
            surface.base_color * subsurface + vec3<f32>(n_dot_l),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        diffuse = diffuse * tint;
    }

    let radiance = (diffuse + sheen_specular) * light.illuminance
        * (n_dot_l * clamp(light.visibility, 0.0, 1.0));
    return radiance + surface.emissive;
}
