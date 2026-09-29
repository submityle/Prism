//! GPU Chiang near-field hair BSDF (energy-conserving, absorption/medulla).
//!
//! Numerical twin of the CPU golden reference
//! `prism_render_shading::hair_chiang::evaluate_hair_chiang_direct`. Where the
//! baseline `hair.wesl` strand lobe uses Karis' real-time Marschner form with a
//! `pow`-tinted absorption, this close-up sibling follows Chiang et al. "A
//! Practical and Controllable Hair and Fur Model for Production Path Tracing"
//! (2016): the fibre absorption coefficient is derived from the base color and
//! the azimuthal roughness so the transmission lobes stay energy-consistent as
//! the fibre darkens, plus an optional hollow-core medulla forward-scatter fill.
//!
//! * R = white dielectric cuticle glint (base-color independent),
//! * TT = forward transmission, `(1 - F)^2` times one Beer-Lambert traversal,
//! * TRT = coloured secondary highlight, `(1 - F)^2 * F` times two traversals,
//! * medulla = optional isotropic hollow-core forward scatter fill.
//!
//! `dot`, `max`, `min`, `abs`, `clamp`, `sqrt`, `cos`, `asin`, `exp`, `log`
//! (natural log) and `pow` are WESL builtins; `brdf_normalize_or` is shared
//! with the principled lobe.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};
import prism_render_scene::shaders::lighting::{
    INV_PI,
};

// Floors / physical constants mirroring `hair_chiang.rs`.
const HAIR_CHIANG_MIN_ROUGHNESS: f32 = 0.045;
const HAIR_CHIANG_MIN_AZIMUTHAL_ROUGHNESS: f32 = 0.02;
const HAIR_CHIANG_MIN_ABSORPTION_COLOR: f32 = 1.0e-3;
const HAIR_CHIANG_CUTICLE_SHIFT: f32 = 0.035;
const HAIR_CHIANG_TWO_PI: f32 = 6.2831853071795864;

// Tunable controls, mirroring the CPU `HairChiangParams`.
struct HairChiangParams {
    // Azimuthal (`N`) roughness in [0, 1]; feeds the color-to-absorption fit.
    azimuthal_roughness: f32,
    // Longitudinal (`M`) roughness in [0, 1]; `> 0` overrides surface roughness.
    longitudinal_roughness: f32,
    // Hollow-core medulla volume fraction in [0, 1]; 0 is solid keratin.
    medulla_ratio: f32,
    // Isotropic scatter albedo of the medulla core in [0, 1].
    medulla_scatter: f32,
    // Fibre index of refraction (keratin is about 1.55).
    ior: f32,
}

// Normalized longitudinal Gaussian `Mp`. Mirrors `hair_gaussian`.
fn hair_chiang_gaussian(beta: f32, x: f32) -> f32 {
    return exp(-0.5 * x * x / (beta * beta)) / (sqrt(HAIR_CHIANG_TWO_PI) * beta);
}

// Schlick Fresnel for the fibre dielectric with the given IOR. Mirrors
// `hair_fresnel`.
fn hair_chiang_fresnel(cos_theta: f32, ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    let f0 = r * r;
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(m, 5.0);
}

// Per-channel absorption coefficient `sigma_a` from the base color and
// azimuthal roughness (Chiang et al. 2016 eq. 9). Mirrors `sigma_a_from_color`.
fn hair_chiang_sigma_a(color: vec3<f32>, azimuthal_roughness: f32) -> vec3<f32> {
    let b = clamp(azimuthal_roughness, HAIR_CHIANG_MIN_AZIMUTHAL_ROUGHNESS, 1.0);
    let denom = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * b * b * b
        + 5.574 * b * b * b * b
        + 0.245 * b * b * b * b * b;
    let c = clamp(color, vec3<f32>(HAIR_CHIANG_MIN_ABSORPTION_COLOR), vec3<f32>(1.0));
    let t = log(c) / denom;
    return t * t;
}

// Chiang near-field hair BSDF for a single analytic light. Mirrors
// `evaluate_hair_chiang_direct`.
fn hair_chiang_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: HairChiangParams,
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

    // Longitudinal sines about the strand tangent.
    let sin_theta_l = clamp(dot(t, l), -1.0, 1.0);
    let sin_theta_v = clamp(dot(t, v), -1.0, 1.0);

    // Half difference angle drives Fresnel and the transmission widths.
    let cos_theta_d = max(cos(0.5 * abs(asin(sin_theta_v) - asin(sin_theta_l))), 1.0e-3);

    // Azimuthal cosine from the tangent-plane projections of L and V.
    let lp = l - t * sin_theta_l;
    let vp = v - t * sin_theta_v;
    let cos_phi = clamp(dot(lp, vp) / sqrt(dot(lp, lp) * dot(vp, vp) + 1.0e-4), -1.0, 1.0);
    let cos_half_phi = sqrt(clamp(0.5 + 0.5 * cos_phi, 0.0, 1.0));

    // Modified (Bravais) refractive index for the projected azimuth.
    let eta = max(params.ior, 1.001);
    let n_prime = max(sqrt(max(eta * eta - 1.0, 0.0)) / cos_theta_d + cos_theta_d, 1.001);

    // Absorption coefficient from the fibre color (the Chiang contribution).
    let sigma_a = hair_chiang_sigma_a(base, params.azimuthal_roughness);

    // Internal refraction azimuth and one-traversal optical path length, then
    // the per-channel Beer-Lambert transmittance through the cortex.
    let sin_gamma_i = clamp(cos_half_phi, 0.0, 1.0);
    let cos_gamma_i = sqrt(max(1.0 - sin_gamma_i * sin_gamma_i, 0.0));
    let sin_gamma_t = clamp(sin_gamma_i / n_prime, 0.0, 1.0);
    let cos_gamma_t = sqrt(max(1.0 - sin_gamma_t * sin_gamma_t, 0.0));
    let seg = 2.0 * cos_gamma_t / cos_theta_d;
    let transmittance = exp(-sigma_a * seg);

    // Interface Fresnel at the refracted-adjusted incidence.
    let fresnel = clamp(hair_chiang_fresnel(cos_theta_d * cos_gamma_i, eta), 0.0, 1.0);
    let one_minus_f = 1.0 - fresnel;

    // Per-lobe longitudinal widths and cuticle shifts. A positive
    // `longitudinal_roughness` decouples the highlight tightness from the base
    // material; otherwise the surface roughness drives it.
    var roughness = surface.perceptual_roughness;
    if params.longitudinal_roughness > 0.0 {
        roughness = params.longitudinal_roughness;
    }
    roughness = clamp(roughness, HAIR_CHIANG_MIN_ROUGHNESS, 1.0);
    let r2 = roughness * roughness;
    let beta = vec3<f32>(r2, r2 * 0.5, r2 * 2.0);
    let alpha = vec3<f32>(
        -2.0 * HAIR_CHIANG_CUTICLE_SHIFT,
        HAIR_CHIANG_CUTICLE_SHIFT,
        4.0 * HAIR_CHIANG_CUTICLE_SHIFT,
    );
    let long_arg = sin_theta_l + sin_theta_v;
    let mp = vec3<f32>(
        hair_chiang_gaussian(beta.x, long_arg - alpha.x),
        hair_chiang_gaussian(beta.y, long_arg - alpha.y),
        hair_chiang_gaussian(beta.z, long_arg - alpha.z),
    );

    // R: white primary cuticle reflection (base-color independent).
    let r_s = mp.x * (0.25 * cos_half_phi) * fresnel;
    let r_lobe = vec3<f32>(r_s);

    // TT: forward transmission, `(1 - F)^2` times one cortex traversal.
    let tt_np = exp(-3.65 * cos_phi - 3.98);
    let tt_weight = mp.y * tt_np * one_minus_f * one_minus_f;
    let tt_lobe = transmittance * tt_weight;

    // TRT: coloured secondary highlight, `(1 - F)^2 * F` times two traversals.
    let trt_np = exp(17.0 * cos_phi - 16.78);
    let trt_weight = mp.z * trt_np * one_minus_f * one_minus_f * fresnel;
    let trt_lobe = (transmittance * transmittance) * trt_weight;

    // Medulla: optional isotropic hollow-core scatter, tinted by the cortex
    // half-path and gated by a wrapped cosine so it stays bounded.
    let ratio = clamp(params.medulla_ratio, 0.0, 1.0);
    let albedo = clamp(params.medulla_scatter, 0.0, 1.0);
    var medulla = vec3<f32>(0.0);
    if ratio > 0.0 && albedo > 0.0 {
        let wrap = clamp(dot(n, l) * 0.5 + 0.5, 0.0, 1.0);
        let half_path = sqrt(max(transmittance, vec3<f32>(0.0)));
        medulla = half_path * (ratio * albedo * wrap * INV_PI);
    }

    let scatter = r_lobe + tt_lobe + trt_lobe + medulla;
    let radiance = scatter * light.illuminance * visibility;
    return radiance + surface.emissive;
}
