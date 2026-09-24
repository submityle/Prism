//! GPU BSDF closures mirroring the CPU golden reference in
//! `prism_render_shading` (`lighting.rs`, `environment.rs`).
//!
//! These functions are the byte-for-byte numerical twin of the backend-neutral
//! reference so the compute resolve pass agrees with the CPU golden tests
//! within tolerance. The analytic light ABI, spherical-harmonic probe helpers
//! and the environment-BRDF fit are imported from `lighting.wesl`; this module
//! adds the microfacet direct-lighting lobes (GGX principled + clearcoat), the
//! toon lobe and the split-sum image-based term.
//!
//! `pow`, `exp2`, `reflect`, `normalize` and `inverseSqrt` are shader builtins
//! and are used directly; there is no CPU `bevy_math::ops` restriction here.

import prism_render_scene::shaders::lighting::{
    PI,
    INV_PI,
    LightEnvironment,
    sh_radiance,
    sh_irradiance,
    env_brdf_approx,
};

// Clamp floors mirroring `lighting.rs` (`MIN_ROUGHNESS`, `MIN_N_DOT`).
const MIN_ROUGHNESS: f32 = 0.045;
const MIN_N_DOT: f32 = 1.0e-5;

// Physically based surface parameters consumed by the analytic lobes. Mirrors
// `SurfaceSample` in `lighting.rs`; emissive is carried separately so the
// resolve pass can add it exactly once.
struct SurfaceSample {
    base_color: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    reflectance: f32,
    ambient_occlusion: f32,
    emissive: vec3<f32>,
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen: f32,
    subsurface: f32,
    thickness: f32,
}

// World-space shading basis. Mirrors `ShadingFrame` in `lighting.rs`.
struct ShadingFrame {
    normal: vec3<f32>,
    view: vec3<f32>,
}

// A single analytic light already reduced to a surface-relative sample.
// Mirrors `DirectLightSample` in `lighting.rs`.
struct DirectLightSample {
    direction: vec3<f32>,
    illuminance: vec3<f32>,
    visibility: f32,
}

// Normalizes `value`, falling back to `fallback` for degenerate inputs.
// Mirrors `normalize_or` in the CPU reference.
fn brdf_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if length_squared > 1.0e-12 {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

// Schlick's Fresnel approximation. Mirrors `fresnel_schlick`.
fn fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    let one_minus = 1.0 - clamp(v_dot_h, 0.0, 1.0);
    let squared = one_minus * one_minus;
    let factor = squared * squared * one_minus;
    return f0 + (vec3<f32>(1.0) - f0) * factor;
}

// GGX/Trowbridge-Reitz normal distribution. Mirrors `distribution_ggx`.
fn distribution_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared / max(PI * denominator * denominator, MIN_N_DOT);
}

// Height-correlated Smith visibility term. Mirrors
// `visibility_smith_ggx_correlated`.
fn vis_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let alpha_squared = alpha * alpha;
    let gv = n_dot_l * sqrt((n_dot_v - n_dot_v * alpha_squared) * n_dot_v + alpha_squared);
    let gl = n_dot_v * sqrt((n_dot_l - n_dot_l * alpha_squared) * n_dot_l + alpha_squared);
    return 0.5 / max(gv + gl, MIN_N_DOT);
}

// Physically based direct lighting for one analytic light, including a
// Kulla-Conty energy-compensation term and a clearcoat lobe. Mirrors
// `evaluate_principled_direct`.
fn principled_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), MIN_N_DOT);
    let n_dot_h = max(dot(n, h), MIN_N_DOT);
    let v_dot_h = max(dot(v, h), 0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(f0_dielectric), surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);
    let d = distribution_ggx(n_dot_h, alpha);
    let g = vis_smith(n_dot_v, n_dot_l, alpha);
    let single_scatter = f * (d * g);

    // Kulla-Conty-style bounded compensation returning energy lost to
    // unresolved multiple scattering while preserving the primary lobe.
    let energy = 1.0 - 0.28 * roughness * roughness;
    let compensation = f0 * min((1.0 / max(energy, 0.25)) - 1.0, 1.0);
    let specular = single_scatter + compensation;

    let diffuse_weight = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = (surface.base_color * INV_PI) * diffuse_weight;
    let base = (diffuse + specular) * n_dot_l;

    // Clearcoat: a second, always-dielectric GGX lobe that attenuates the base
    // response by its own Fresnel reflectance.
    let coat = clamp(surface.clearcoat, 0.0, 1.0);
    let coat_roughness = clamp(surface.clearcoat_roughness, MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    let coat_f = fresnel_schlick(vec3<f32>(0.04), v_dot_h);
    let coat_specular = coat_f
        * (coat
            * distribution_ggx(n_dot_h, coat_alpha)
            * vis_smith(n_dot_v, n_dot_l, coat_alpha)
            * n_dot_l);
    let coat_attenuation = 1.0 - coat * clamp(coat_f.x, 0.0, 1.0);
    let reflected = base * coat_attenuation + coat_specular;

    let radiance = reflected * light.illuminance;
    return radiance * clamp(light.visibility, 0.0, 1.0) + surface.emissive;
}

// Non-photoreal quantized diffuse lobe. Mirrors `evaluate_toon_direct`.
fn toon_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    bands: u32,
) -> vec3<f32> {
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let l = brdf_normalize_or(light.direction, n);
    let steps = f32(max(bands, 1u));
    let diffuse = floor(max(dot(n, l), 0.0) * steps) / steps;
    return (surface.base_color * light.illuminance) * (diffuse * clamp(light.visibility, 0.0, 1.0))
        + surface.emissive;
}

// Split-sum image-based indirect term. Mirrors `evaluate_image_based_light`.
// The GPU probe carries no independent intensity, so intensity is 1.0. Probe
// reconstruction can ring negative, so both halves clamp to non-negative before
// weighting, matching the CPU `.max(0.0)` guards.
fn image_based(
    surface: SurfaceSample,
    frame: ShadingFrame,
    env: LightEnvironment,
) -> vec3<f32> {
    let normal = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let view = brdf_normalize_or(frame.view, normal);
    let n_dot_v = max(dot(normal, view), 1.0e-4);

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, 0.0, 1.0);
    let occlusion = clamp(surface.ambient_occlusion, 0.0, 1.0);
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(f0_dielectric), surface.base_color, metallic);

    // Diffuse: Lambertian albedo lit by cosine-convolved irradiance.
    let irradiance = max(sh_irradiance(env, normal), vec3<f32>(0.0));
    let diffuse_weight = 1.0 - metallic;
    let diffuse = surface.base_color * diffuse_weight * irradiance * INV_PI;

    // Specular: prefiltered radiance along the reflection vector weighted by
    // the analytic environment BRDF.
    let reflection = reflect(-view, normal);
    let prefiltered = max(sh_radiance(env, reflection), vec3<f32>(0.0));
    let dfg = env_brdf_approx(n_dot_v, roughness);
    let specular = prefiltered * (f0 * dfg.x + dfg.y);

    return (diffuse + specular) * occlusion;
}
