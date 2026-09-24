//! GPU two-layer clear-coat BSDF lobe.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::clearcoat::evaluate_clearcoat_direct`. The closure
//! follows Unreal's `ClearCoatBxDF`: a thin, smooth dielectric coat
//! (`F0 = 0.04`) over a fully featured isotropic principled base.
//!
//! Two effects distinguish it from the shared principled fallback:
//!
//! * a dedicated coat highlight driven by `clearcoat_roughness`, independent of
//!   the base roughness, so a rough painted base still carries a tight varnish
//!   glint,
//! * double-pass coat absorption: light refracts through the coat on entry and
//!   again on exit, so the base is gated by the coat Fresnel twice
//!   (`(1 - clearcoat * Fc)^2`), darkening the base under the coat more strongly
//!   at grazing angles than the principled single-pass `(1 - clearcoat * Fc)`.
//!
//! The base is treated as isotropic, so with `clearcoat == 0` and
//! `anisotropy == 0` it reduces to `principled_direct`. `dot`, `max`, `min`,
//! `clamp` and `mix` are WESL builtins; the GGX/Fresnel helpers and
//! `brdf_normalize_or` are shared with the principled lobe.

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

// Floors / weights mirroring `clearcoat.rs`.
const CLEARCOAT_MIN_ROUGHNESS: f32 = 0.045;
const CLEARCOAT_MIN_N_DOT: f32 = 1.0e-5;
const CLEARCOAT_F0: f32 = 0.04;

// Two-layer clear-coat BSDF for a single analytic light. Mirrors
// `evaluate_clearcoat_direct`.
fn clearcoat_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> vec3<f32> {
    let visibility = clamp(light.visibility, 0.0, 1.0);

    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let l = brdf_normalize_or(light.direction, n);
    let h = brdf_normalize_or(v + l, n);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), CLEARCOAT_MIN_N_DOT);
    let n_dot_h = max(dot(n, h), CLEARCOAT_MIN_N_DOT);
    let v_dot_h = max(dot(v, h), 0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }

    // Base layer: isotropic principled dielectric/metal.
    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let roughness = clamp(surface.perceptual_roughness, CLEARCOAT_MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix(vec3<f32>(f0_dielectric), surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);
    let d = distribution_ggx(n_dot_h, alpha);
    let g = vis_smith(n_dot_v, n_dot_l, alpha);
    let single_scatter = f * (d * g);

    // Kulla-Conty-style bounded compensation, identical to the principled lobe
    // so the base matches it exactly when the coat is absent.
    let energy = 1.0 - 0.28 * roughness * roughness;
    let compensation = f0 * min((1.0 / max(energy, 0.25)) - 1.0, 1.0);
    let specular = single_scatter + compensation;
    let diffuse_weight = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = (surface.base_color * INV_PI) * diffuse_weight;
    let base = (diffuse + specular) * n_dot_l;

    // Coat layer: a second smooth dielectric GGX lobe. The coat is an
    // index-matched white dielectric, so its Fresnel is achromatic; the scalar
    // `Fc` drives both the coat highlight and the base attenuation.
    let coat = clamp(surface.clearcoat, 0.0, 1.0);
    let coat_roughness = clamp(surface.clearcoat_roughness, CLEARCOAT_MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    let coat_f = clamp(fresnel_schlick(vec3<f32>(CLEARCOAT_F0), v_dot_h).x, 0.0, 1.0);
    let coat_d = distribution_ggx(n_dot_h, coat_alpha);
    let coat_g = vis_smith(n_dot_v, n_dot_l, coat_alpha);
    let coat_specular = vec3<f32>(coat * coat_f * coat_d * coat_g * n_dot_l);

    // Double-pass transmission: the base is gated by the coat Fresnel twice.
    let transmission = 1.0 - coat * coat_f;
    let throughput = transmission * transmission;

    let reflected = base * throughput + coat_specular;
    let radiance = reflected * light.illuminance;
    return radiance * visibility + surface.emissive;
}
