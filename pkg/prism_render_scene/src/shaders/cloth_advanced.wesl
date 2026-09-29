//! GPU advanced cloth (fabric) BSDF closure.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::cloth_advanced::evaluate_cloth_advanced_direct`.
//! Highest-fidelity sibling of `cloth.wesl`: on top of the bare `Charlie`
//! sheen it layers
//!
//! * energy-conserving `Estevez`-`Kulla` `Charlie` sheen (albedo compensation +
//!   `Filament` sheen-scaling diffuse darkening),
//! * `Ashikhmin`-`Shirley` woven warp/weft anisotropy about a rotated tangent
//!   basis,
//! * `DICE`-style thin double-sided transmission modulated by thickness,
//! * `Turquin` 2019 style multiple-scattering compensation,
//! * per-wavelength thin-film interference `Fresnel` driven by thickness,
//! * a tension-driven wrinkle half-vector blend.
//!
//! `pow`, `dot`, `cross`, `max`, `min`, `clamp`, `sqrt`, `sin`, `cos`, `abs`
//! and `mix` are WESL builtins; `brdf_normalize_or` is shared with the
//! principled lobe.

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

// Floors / physical constants mirroring `cloth_advanced.rs`.
const CLOTH_ADV_TWO_PI: f32 = 6.2831853071795864;
const CLOTH_ADV_MIN_ROUGHNESS: f32 = 0.045;
const CLOTH_ADV_MIN_N_DOT: f32 = 1.0e-5;
const CLOTH_ADV_MIN_SIN2: f32 = 0.0078125;
const CLOTH_ADV_FILM_IOR: f32 = 1.35;
const CLOTH_ADV_FILM_MAX_NM: f32 = 600.0;
const CLOTH_ADV_LAMBDA_R: f32 = 650.0;
const CLOTH_ADV_LAMBDA_G: f32 = 550.0;
const CLOTH_ADV_LAMBDA_B: f32 = 450.0;
const CLOTH_ADV_FILM_F0: f32 = 0.04;
const CLOTH_ADV_TRANS_DISTORTION: f32 = 0.2;
const CLOTH_ADV_TRANS_POWER: f32 = 4.0;

// Estevez-Kulla energy-conserving Charlie sheen distribution. Mirrors
// `distribution_charlie_ec`.
fn cloth_advanced_distribution_charlie_ec(roughness: f32, n_dot_h: f32) -> f32 {
    let inv_r = 1.0 / roughness;
    let sin2h = max(1.0 - n_dot_h * n_dot_h, CLOTH_ADV_MIN_SIN2);
    let charlie = (2.0 + inv_r) * pow(sin2h, inv_r * 0.5) * (1.0 / CLOTH_ADV_TWO_PI);
    let comp = 1.0 + roughness * (0.309 - 0.288 * roughness);
    return charlie * comp;
}

// Ashikhmin/Neubelt visibility term. Mirrors `visibility_ashikhmin`.
fn cloth_advanced_visibility_ashikhmin(n_dot_v: f32, n_dot_l: f32) -> f32 {
    return 1.0 / max(4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v), CLOTH_ADV_MIN_N_DOT);
}

// Sheen directional albedo approximation. Mirrors `sheen_albedo`.
fn cloth_advanced_sheen_albedo(n_dot_v: f32, roughness: f32) -> f32 {
    let grazing = clamp(1.0 - n_dot_v, 0.0, 1.0);
    let grazing3 = grazing * grazing * grazing;
    let fresnel_like = CLOTH_ADV_FILM_F0 + (1.0 - CLOTH_ADV_FILM_F0) * grazing3;
    return clamp(fresnel_like * (0.25 + 0.75 * roughness), 0.0, 1.0);
}

// Turquin 2019 style multiple-scattering energy compensation. Mirrors
// `multiscatter_comp`.
fn cloth_advanced_multiscatter_comp(e: f32, sheen: f32) -> f32 {
    let ec = clamp(e, 0.0, 1.0);
    return 1.0 + sheen * (1.0 - ec);
}

// Re-orthonormalized, rotated warp/weft tangent basis. Mirrors `woven_axes`.
struct ClothAdvancedAxes {
    warp: vec3<f32>,
    weft: vec3<f32>,
}

fn cloth_advanced_woven_axes(
    normal: vec3<f32>,
    basis_tangent: vec3<f32>,
    basis_bitangent: vec3<f32>,
    rotation: f32,
) -> ClothAdvancedAxes {
    let projected = basis_tangent - normal * dot(normal, basis_tangent);
    let tangent0 = brdf_normalize_or(projected, basis_tangent);
    let handedness = select(1.0, -1.0, dot(cross(normal, tangent0), basis_bitangent) < 0.0);
    let bitangent0 = cross(normal, tangent0) * handedness;
    let cos_r = cos(rotation);
    let sin_r = sin(rotation);
    var axes: ClothAdvancedAxes;
    axes.warp = tangent0 * cos_r + bitangent0 * sin_r;
    axes.weft = bitangent0 * cos_r - tangent0 * sin_r;
    return axes;
}

// Ashikhmin-Shirley woven warp/weft anisotropic highlight (scalar). Mirrors
// `anisotropic_woven`.
fn cloth_advanced_anisotropic_woven(
    surface: SurfaceSample,
    frame: ShadingFrame,
    n: vec3<f32>,
    v: vec3<f32>,
    h: vec3<f32>,
    n_dot_h: f32,
    n_dot_v: f32,
    n_dot_l: f32,
) -> f32 {
    let axes = cloth_advanced_woven_axes(
        n,
        frame.tangent,
        frame.bitangent,
        surface.anisotropy_rotation,
    );
    let roughness = clamp(surface.perceptual_roughness, CLOTH_ADV_MIN_ROUGHNESS, 1.0);
    let anisotropy = clamp(surface.anisotropy, -1.0, 1.0);
    let aspect = sqrt(1.0 - 0.9 * anisotropy);
    let r_warp = clamp(roughness / aspect, CLOTH_ADV_MIN_ROUGHNESS, 1.0);
    let r_weft = clamp(roughness * aspect, CLOTH_ADV_MIN_ROUGHNESS, 1.0);
    let nu = max(2.0 / (r_warp * r_warp) - 2.0, 0.0);
    let nv = max(2.0 / (r_weft * r_weft) - 2.0, 0.0);
    let h_dot_t = dot(h, axes.warp);
    let h_dot_b = dot(h, axes.weft);
    let sin2h = max(1.0 - n_dot_h * n_dot_h, CLOTH_ADV_MIN_SIN2);
    let exponent = (nu * h_dot_t * h_dot_t + nv * h_dot_b * h_dot_b) / sin2h;
    let norm = sqrt((nu + 1.0) * (nv + 1.0)) * (1.0 / CLOTH_ADV_TWO_PI);
    let d = norm * pow(n_dot_h, exponent);
    let v_dot_h = max(dot(v, h), CLOTH_ADV_MIN_N_DOT);
    let denom = 4.0 * v_dot_h * max(n_dot_v, n_dot_l);
    return d / max(denom, CLOTH_ADV_MIN_N_DOT);
}

// DICE-style thin double-sided transmission (back-light). Mirrors
// `thin_transmission`.
fn cloth_advanced_thin_transmission(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, thickness: f32) -> f32 {
    let lt = l + n * CLOTH_ADV_TRANS_DISTORTION;
    let forward = clamp(dot(v, -lt), 0.0, 1.0);
    let lobe = pow(forward, CLOTH_ADV_TRANS_POWER);
    let thin = 1.0 - clamp(thickness, 0.0, 1.0);
    return lobe * thin;
}

// Thin-film interference Fresnel tint (per channel). Mirrors
// `thin_film_fresnel`.
fn cloth_advanced_thin_film_fresnel(n_dot_v: f32, thickness: f32) -> vec3<f32> {
    let t = clamp(thickness, 0.0, 1.0);
    let cos_t = clamp(n_dot_v, 0.0, 1.0);
    let one_minus = 1.0 - cos_t;
    let sq = one_minus * one_minus;
    let fresnel = CLOTH_ADV_FILM_F0 + (1.0 - CLOTH_ADV_FILM_F0) * sq * sq * one_minus;
    let optical = 2.0 * CLOTH_ADV_FILM_IOR * (CLOTH_ADV_FILM_MAX_NM * t) * cos_t;
    let ir = 0.5 + 0.5 * cos(CLOTH_ADV_TWO_PI * optical / CLOTH_ADV_LAMBDA_R);
    let ig = 0.5 + 0.5 * cos(CLOTH_ADV_TWO_PI * optical / CLOTH_ADV_LAMBDA_G);
    let ib = 0.5 + 0.5 * cos(CLOTH_ADV_TWO_PI * optical / CLOTH_ADV_LAMBDA_B);
    return vec3<f32>(
        (1.0 + (ir - 1.0) * t) * fresnel,
        (1.0 + (ig - 1.0) * t) * fresnel,
        (1.0 + (ib - 1.0) * t) * fresnel,
    );
}

// Tension-driven wrinkle half-vector blend. Mirrors `wrinkle_normal_blend`.
fn cloth_advanced_wrinkle_normal_blend(n_dot_h: f32, anisotropy: f32) -> f32 {
    let tension = clamp(abs(anisotropy), 0.0, 1.0);
    let sharpened = pow(n_dot_h, 1.0 + tension);
    let blended = n_dot_h + (sharpened - n_dot_h) * tension;
    return clamp(blended, CLOTH_ADV_MIN_N_DOT, 1.0);
}

// Advanced cloth BSDF for a single analytic light. Mirrors
// `evaluate_cloth_advanced_direct`.
fn cloth_advanced_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), CLOTH_ADV_MIN_N_DOT);
    let n_dot_h_raw = max(dot(n, h), CLOTH_ADV_MIN_N_DOT);
    let roughness = clamp(surface.perceptual_roughness, CLOTH_ADV_MIN_ROUGHNESS, 1.0);
    let sheen = clamp(surface.sheen, 0.0, 1.0);
    let visibility = clamp(light.visibility, 0.0, 1.0);

    // Tension-driven wrinkle blend on the half-vector cosine.
    let n_dot_h = cloth_advanced_wrinkle_normal_blend(n_dot_h_raw, surface.anisotropy);

    // Energy-conserving Charlie sheen (white fuzz).
    let d = cloth_advanced_distribution_charlie_ec(roughness, n_dot_h);
    let vis = cloth_advanced_visibility_ashikhmin(n_dot_v, n_dot_l);
    let e = cloth_advanced_sheen_albedo(n_dot_v, roughness);
    let ms = cloth_advanced_multiscatter_comp(e, sheen);
    let sheen_lobe = sheen * d * vis * ms;

    // Woven warp/weft anisotropic threads, tinted by the thin-film interference.
    let woven = cloth_advanced_anisotropic_woven(surface, frame, n, v, h, n_dot_h, n_dot_v, n_dot_l);
    let film = cloth_advanced_thin_film_fresnel(n_dot_v, surface.thickness);
    let specular = vec3<f32>(sheen_lobe) + woven * film;

    // Diffuse: Lambertian base darkened by the sheen directional albedo.
    let diff_scale = clamp(1.0 - sheen * e, 0.0, 1.0);
    let diffuse = surface.base_color * (INV_PI * diff_scale);

    // Front-facing radiance (naturally zero when the surface faces away).
    let front = (diffuse + specular) * light.illuminance * (n_dot_l * visibility);

    // Thin double-sided transmission (back-light), tinted by the base color.
    let trans = cloth_advanced_thin_transmission(n, v, l, surface.thickness);
    let back = surface.base_color * light.illuminance * (trans * visibility);

    return front + back + surface.emissive;
}
