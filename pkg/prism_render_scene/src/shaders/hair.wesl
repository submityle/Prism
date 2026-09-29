//! GPU hair (strand) BSDF lobe.
//!
//! Numerical twin of the CPU golden reference
//! `prism_render_shading::hair::evaluate_hair_direct`.  The closure implements
//! a physically based Marschner R/TT/TRT strand BSDF (Karis' energy-conserving
//! real-time form from "Physically Based Hair Shading in Unreal") plus a
//! Zinke-style dual-scattering multiple-scattering fill:
//!
//! * R = sharp white cuticle reflection (primary highlight),
//! * TT = forward transmission tinted by absorption (back-lit rim),
//! * TRT = coloured secondary highlight (far-wall reflection),
//! * dual-scattering = translucent forward/backward fill so light-coloured
//!   hair stays luminous.
//!
//! `dot`, `max`, `abs`, `clamp`, `mix`, `sqrt`, `sin`, `cos`, `asin`, `exp` and
//! `pow` are WESL builtins; `brdf_normalize_or` is shared with the principled
//! lobe.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};
import prism_render_scene::shaders::lighting::{
    INV_PI,
};

// Floors / physical constants mirroring `hair.rs`.
const HAIR_MIN_ROUGHNESS: f32 = 0.045;
const HAIR_IOR: f32 = 1.55;
const HAIR_CUTICLE_SHIFT: f32 = 0.035;
const HAIR_SPECULAR: f32 = 0.5;
const HAIR_BACKLIT: f32 = 1.0;
const HAIR_MS_WEIGHT: f32 = 0.25;
const HAIR_TWO_PI: f32 = 6.2831853071795864;

// Normalized longitudinal Gaussian `Mp`. Mirrors `hair_gaussian`.
fn hair_gaussian(beta: f32, x: f32) -> f32 {
    return exp(-0.5 * x * x / (beta * beta)) / (sqrt(HAIR_TWO_PI) * beta);
}

// Schlick Fresnel for the fibre dielectric. Mirrors `hair_fresnel`.
fn hair_fresnel(cos_theta: f32) -> f32 {
    let r = (1.0 - HAIR_IOR) / (1.0 + HAIR_IOR);
    let f0 = r * r;
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(m, 5.0);
}

// Per-channel absorption tint `pow(base, e)`. Mirrors `pow3`.
fn hair_pow3(base: vec3<f32>, e: f32) -> vec3<f32> {
    return vec3<f32>(pow(base.x, e), pow(base.y, e), pow(base.z, e));
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
    let base = surface.base_color;

    // Longitudinal sines about the strand tangent, and the raw V.L.
    let sin_theta_l = clamp(dot(t, l), -1.0, 1.0);
    let sin_theta_v = clamp(dot(t, v), -1.0, 1.0);
    let v_dot_l = dot(v, l);

    // Half difference angle drives Fresnel and the transmission widths.
    let cos_theta_d = cos(0.5 * abs(asin(sin_theta_v) - asin(sin_theta_l)));

    // Azimuthal cosine from the tangent-plane projections of L and V.
    let lp = l - t * sin_theta_l;
    let vp = v - t * sin_theta_v;
    let cos_phi = dot(lp, vp) / sqrt(dot(lp, lp) * dot(vp, vp) + 1.0e-4);
    let cos_half_phi = sqrt(clamp(0.5 + 0.5 * cos_phi, 0.0, 1.0));

    // Modified (Bravais) refractive index for the projected azimuth.
    let n_prime = 1.19 / cos_theta_d + 0.36 * cos_theta_d;

    // Per-lobe longitudinal widths (roughness^2 scaled) and cuticle shifts.
    let roughness = clamp(surface.perceptual_roughness, HAIR_MIN_ROUGHNESS, 1.0);
    let r2 = roughness * roughness;
    let beta = vec3<f32>(r2, r2 * 0.5, r2 * 2.0);
    let alpha = vec3<f32>(-2.0 * HAIR_CUTICLE_SHIFT, HAIR_CUTICLE_SHIFT, 4.0 * HAIR_CUTICLE_SHIFT);
    let long_arg = sin_theta_l + sin_theta_v;
    let mp = vec3<f32>(
        hair_gaussian(beta.x, long_arg - alpha.x),
        hair_gaussian(beta.y, long_arg - alpha.y),
        hair_gaussian(beta.z, long_arg - alpha.z),
    );

    // R: white primary cuticle reflection (base-color independent).
    let r_np = 0.25 * cos_half_phi;
    let r_fp = hair_fresnel(sqrt(clamp(0.5 + 0.5 * v_dot_l, 0.0, 1.0)));
    let r_backlit = mix(1.0, HAIR_BACKLIT, clamp(-v_dot_l, 0.0, 1.0));
    let r_s = mp.x * r_np * r_fp * (HAIR_SPECULAR * 2.0) * r_backlit;
    let r_lobe = vec3<f32>(r_s);

    // TT: forward transmission tinted by absorption through the fibre.
    let tt_a = 1.0 / n_prime;
    let tt_h = cos_half_phi * (1.0 + tt_a * (0.6 - 0.8 * cos_phi));
    let tt_f = hair_fresnel(cos_theta_d * sqrt(clamp(1.0 - tt_h * tt_h, 0.0, 1.0)));
    let tt_fp = (1.0 - tt_f) * (1.0 - tt_f);
    let tt_ha = tt_h * tt_a;
    let tt_tp = hair_pow3(base, 0.5 * sqrt(clamp(1.0 - tt_ha * tt_ha, 0.0, 1.0)) / cos_theta_d);
    let tt_np = exp(-3.65 * cos_phi - 3.98);
    let tt_lobe = tt_tp * (mp.y * tt_np * tt_fp * HAIR_BACKLIT);

    // TRT: coloured secondary highlight (double transmission + reflection).
    let trt_f = hair_fresnel(cos_theta_d * 0.5);
    let trt_fp = (1.0 - trt_f) * (1.0 - trt_f) * trt_f;
    let trt_tp = hair_pow3(base, 0.8 / cos_theta_d);
    let trt_np = exp(17.0 * cos_phi - 16.78);
    let trt_lobe = trt_tp * (mp.z * trt_np * trt_fp);

    let spec = r_lobe + tt_lobe + trt_lobe;

    // Dual-scattering multiple-scattering fill (Zinke): translucent
    // forward/backward wrap so light-coloured hair stays luminous.
    let wrap = clamp(dot(n, l) * 0.5 + 0.5, 0.0, 1.0);
    let ms = base * (INV_PI * wrap * HAIR_MS_WEIGHT);

    let radiance = (spec + ms) * light.illuminance * visibility;
    return radiance + surface.emissive;
}
