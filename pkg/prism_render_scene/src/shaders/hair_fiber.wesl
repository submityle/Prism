//! GPU fibre-level fur/hair BSDF (Yan et al. dual-cylinder, near-field high-config).
//!
//! Numerical twin of the CPU golden reference
//! `prism_render_shading::hair_fiber::evaluate_hair_fiber_direct`. Where the
//! Chiang sibling (`hair_chiang.wesl`) models one absorbing cortex cylinder with
//! an isotropic medulla *fill*, this highest-fidelity close-up front end follows
//! Yan et al. "Physically Accurate Fur Reflectance: Modeling, Measurement and
//! Rendering" (2015): the fibre is a *double cylinder* (outer cortex + inner
//! scattering medulla) and the core contributes two extra *scattered* lobes on
//! top of the three cuticle lobes:
//!
//! * R = white dielectric cuticle glint (base-color independent),
//! * TT = forward cortex transmission, `(1 - F)^2` times one Beer-Lambert pass,
//! * TRT = coloured secondary highlight, `(1 - F)^2 * F` times two passes,
//! * `TTs` = scattered forward transmission through the medulla core, broadened
//!   longitudinally and shaped azimuthally by a Henyey-Greenstein phase lobe,
//! * `TRTs` = the same core-scatter path with one extra internal reflection.
//!
//! With `medulla_ratio = 0` the two scattered lobes vanish and the model
//! degrades gracefully to the three cuticle lobes. Absorption is derived from
//! the base color (Chiang et al. 2016 eq. 9) so the transmission lobes stay
//! energy-consistent as the fibre darkens.
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

// Floors / physical constants mirroring `hair_fiber.rs`.
const HAIR_FIBER_MIN_ROUGHNESS: f32 = 0.045;
const HAIR_FIBER_MIN_AZIMUTHAL_ROUGHNESS: f32 = 0.02;
const HAIR_FIBER_MIN_ABSORPTION_COLOR: f32 = 1.0e-3;
const HAIR_FIBER_CUTICLE_SHIFT: f32 = 0.035;
const HAIR_FIBER_TWO_PI: f32 = 6.2831853071795864;
// Reciprocal of four-pi, the isotropic phase-function normaliser (`INV_FOUR_PI`).
const HAIR_FIBER_INV_FOUR_PI: f32 = 0.07957747154594767;

// Tunable controls, mirroring the CPU `HairFiberParams`.
struct HairFiberParams {
    // Azimuthal (`N`) roughness in [0, 1]; feeds the color-to-absorption fit.
    azimuthal_roughness: f32,
    // Longitudinal (`M`) roughness in [0, 1]; `> 0` overrides surface roughness.
    longitudinal_roughness: f32,
    // Medulla radius fraction `kappa` in [0, 1]; 0 is a solid cortex fibre.
    medulla_ratio: f32,
    // Medulla single-scatter albedo in [0, 1].
    medulla_scatter: f32,
    // Medulla absorption coefficient (`>= 0`); attenuates the scattered lobes.
    medulla_absorption: f32,
    // Henyey-Greenstein anisotropy `g` in (-1, 1); `> 0` is forward scattering.
    scatter_anisotropy: f32,
    // Fibre index of refraction (keratin is about 1.55).
    ior: f32,
}

// Normalized longitudinal Gaussian `Mp`. Mirrors `hair_gaussian`.
fn hair_fiber_gaussian(beta: f32, x: f32) -> f32 {
    return exp(-0.5 * x * x / (beta * beta)) / (sqrt(HAIR_FIBER_TWO_PI) * beta);
}

// Schlick Fresnel for the fibre dielectric with the given IOR. Mirrors
// `hair_fresnel`.
fn hair_fiber_fresnel(cos_theta: f32, ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    let f0 = r * r;
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(m, 5.0);
}

// Henyey-Greenstein phase function for scatter cosine `cos_theta` and
// anisotropy `g`. Normalised so the isotropic (`g = 0`) case is `INV_FOUR_PI`.
// Mirrors `henyey_greenstein`.
fn hair_fiber_henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let gc = clamp(g, -0.95, 0.95);
    let denom = max(1.0 + gc * gc - 2.0 * gc * cos_theta, 1.0e-4);
    return HAIR_FIBER_INV_FOUR_PI * (1.0 - gc * gc) / (denom * sqrt(denom));
}

// Per-channel absorption coefficient `sigma_a` from the base color and
// azimuthal roughness (Chiang et al. 2016 eq. 9). Mirrors `sigma_a_from_color`.
fn hair_fiber_sigma_a(color: vec3<f32>, azimuthal_roughness: f32) -> vec3<f32> {
    let b = clamp(azimuthal_roughness, HAIR_FIBER_MIN_AZIMUTHAL_ROUGHNESS, 1.0);
    let denom = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * b * b * b
        + 5.574 * b * b * b * b
        + 0.245 * b * b * b * b * b;
    let c = clamp(color, vec3<f32>(HAIR_FIBER_MIN_ABSORPTION_COLOR), vec3<f32>(1.0));
    let t = log(c) / denom;
    return t * t;
}

// Fibre-level fur/hair BSDF for a single analytic light. Mirrors
// `evaluate_hair_fiber_direct`.
fn hair_fiber_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: HairFiberParams,
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

    // Cortex absorption coefficient from the fibre color.
    let sigma_a = hair_fiber_sigma_a(base, params.azimuthal_roughness);

    // Internal refraction azimuth and one-traversal optical path length, then
    // the per-channel Beer-Lambert transmittance through the cortex.
    let sin_gamma_i = clamp(cos_half_phi, 0.0, 1.0);
    let cos_gamma_i = sqrt(max(1.0 - sin_gamma_i * sin_gamma_i, 0.0));
    let sin_gamma_t = clamp(sin_gamma_i / n_prime, 0.0, 1.0);
    let cos_gamma_t = sqrt(max(1.0 - sin_gamma_t * sin_gamma_t, 0.0));
    let seg = 2.0 * cos_gamma_t / cos_theta_d;
    let transmittance = exp(-sigma_a * seg);
    let half_transmittance = sqrt(max(transmittance, vec3<f32>(0.0)));

    // Interface Fresnel at the refracted-adjusted incidence.
    let fresnel = clamp(hair_fiber_fresnel(cos_theta_d * cos_gamma_i, eta), 0.0, 1.0);
    let one_minus_f = 1.0 - fresnel;

    // Per-lobe longitudinal widths and cuticle shifts. A positive
    // `longitudinal_roughness` decouples the highlight tightness from the base
    // material; otherwise the surface roughness drives it.
    var roughness = surface.perceptual_roughness;
    if params.longitudinal_roughness > 0.0 {
        roughness = params.longitudinal_roughness;
    }
    roughness = clamp(roughness, HAIR_FIBER_MIN_ROUGHNESS, 1.0);
    let r2 = roughness * roughness;
    let beta = vec3<f32>(r2, r2 * 0.5, r2 * 2.0);
    let alpha = vec3<f32>(
        -2.0 * HAIR_FIBER_CUTICLE_SHIFT,
        HAIR_FIBER_CUTICLE_SHIFT,
        4.0 * HAIR_FIBER_CUTICLE_SHIFT,
    );
    let long_arg = sin_theta_l + sin_theta_v;
    let mp = vec3<f32>(
        hair_fiber_gaussian(beta.x, long_arg - alpha.x),
        hair_fiber_gaussian(beta.y, long_arg - alpha.y),
        hair_fiber_gaussian(beta.z, long_arg - alpha.z),
    );

    // ---- Cuticle lobes (shared with the Chiang front end) ----------------
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

    // ---- Medulla scattered lobes (the fibre-level contribution) ----------
    let kappa = clamp(params.medulla_ratio, 0.0, 1.0);
    let albedo = clamp(params.medulla_scatter, 0.0, 1.0);
    var tts_lobe = vec3<f32>(0.0);
    var trts_lobe = vec3<f32>(0.0);
    if kappa > 0.0 && albedo > 0.0 {
        // Broadened longitudinal profile: the core randomises the exit
        // elevation, so widen `beta` with the core fraction.
        let beta_s = clamp(
            r2 * (1.0 + 3.0 * kappa),
            HAIR_FIBER_MIN_ROUGHNESS * HAIR_FIBER_MIN_ROUGHNESS,
            4.0,
        );
        let ms = hair_fiber_gaussian(beta_s, long_arg);
        // `cos_phi = -1` is straight-through forward scatter.
        let phase = hair_fiber_henyey_greenstein(-cos_phi, params.scatter_anisotropy);
        // Core attenuation along a `kappa`-scaled path, then the albedo weight.
        let core_atten = exp(-max(params.medulla_absorption, 0.0) * (0.5 + kappa));
        let core_weight = kappa * albedo * core_atten * ms * phase;

        // TTs: (1 - F)^2 in, one cortex half-pass each way, forward scatter.
        let tts_weight = core_weight * one_minus_f * one_minus_f;
        tts_lobe = half_transmittance * base * tts_weight;

        // TRTs: one extra internal reflection (F weighted), a full cortex pass
        // of tint for the longer detour.
        let trts_weight = core_weight * one_minus_f * one_minus_f * fresnel;
        trts_lobe = transmittance * base * trts_weight;
    }

    let scatter = r_lobe + tt_lobe + trt_lobe + tts_lobe + trts_lobe;
    let radiance = scatter * light.illuminance * visibility;
    return radiance + surface.emissive;
}
