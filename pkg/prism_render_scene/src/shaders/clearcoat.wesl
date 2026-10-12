//! GPU two-layer clear-coat BSDF lobe.
//!
//! Byte-for-byte numerical twin of the CPU golden reference
//! `prism_render_shading::clearcoat::evaluate_clearcoat_direct`. The closure
//! follows Unreal's `ClearCoatBxDF`, upgraded to the `OpenPBR` `coat` group: a
//! thin, smooth dielectric coat of relative index `coat_ior` (default `1.5`,
//! i.e. the historical `F0 = 0.04`) over a fully featured isotropic principled
//! base.
//!
//! Three effects distinguish it from the shared principled fallback:
//!
//! * a dedicated coat highlight driven by `clearcoat_roughness`, independent of
//!   the base roughness, so a rough painted base still carries a tight varnish
//!   glint,
//! * a per-channel coat tint: `coat_color` scales the highlight, so a tinted
//!   varnish colours the glint (white is the identity),
//! * double-pass coat absorption at the view and light angles
//!   (`(1 - clearcoat*F(n·v)) * (1 - clearcoat*F(n·l))`), darkening the base
//!   under the coat more strongly at grazing angles than a single-pass term.
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
    dielectric_f0,
    fresnel_schlick,
    distribution_ggx,
    distribution_ggx_anisotropic,
    anisotropic_axes,
    oren_nayar_factor,
    fresnel_dielectric,
    BRDF_MIN_DIELECTRIC_IOR,
    vis_smith,
    vis_smith_anisotropic,
    MIN_ANISOTROPIC_ALPHA,
};
import prism_render_scene::shaders::lighting::{
    INV_PI,
};

// Floors / weights mirroring `clearcoat.rs`.
const CLEARCOAT_MIN_ROUGHNESS: f32 = 0.045;
const CLEARCOAT_MIN_N_DOT: f32 = 1.0e-5;
// The coat interface now shares `brdf.wesl::fresnel_dielectric` with the
// transmission lobe (single dielectric-Fresnel implementation).

// The coat interface normal: `OpenPBR` `coat_normal` when authored, else the
// base shading normal. Twin of `clearcoat.rs::coat_interface_normal`; the zero
// vector is the identity sentinel.
fn clearcoat_interface_normal(coat_normal: vec3<f32>, base_normal: vec3<f32>) -> vec3<f32> {
    if dot(coat_normal, coat_normal) > 0.0 {
        return brdf_normalize_or(coat_normal, base_normal);
    }
    return base_normal;
}

// `OpenPBR` §Coat *Darkening*: hemispherical (average) albedo of the coat
// Fresnel via the specification's closed-form fit for `η ≥ 1`, continued to
// `η < 1` by `E_F(η) = 1 − η²(1 − E_F(1/η))`. Byte-for-byte twin of
// `clearcoat.rs::coat_hemispherical_fresnel`.
fn cc_hemispherical_fresnel_ge1(e: f32) -> f32 {
    let num = 10893.0 * e - 1438.2;
    let den = -774.4 * e * e + 10212.0 * e + 1.0;
    if num <= 0.0 || den <= 0.0 {
        return 0.0;
    }
    return clamp(log(num / den), 0.0, 1.0);
}
fn cc_hemispherical_fresnel(eta: f32) -> f32 {
    let e = max(eta, BRDF_MIN_DIELECTRIC_IOR);
    if e >= 1.0 {
        return cc_hemispherical_fresnel_ge1(e);
    }
    let up = cc_hemispherical_fresnel_ge1(1.0 / e);
    return clamp(1.0 - e * e * (1.0 - up), 0.0, 1.0);
}

// `openpbr` equation `internal_diffuse_reflection_coefficient_for_rough_base`:
// `K_r = 1 − (1 − E_F(η_c))/η_c²`. Mirrors `coat_internal_reflection_rough`.
fn cc_internal_reflection_rough(eta_c: f32) -> f32 {
    let e = cc_hemispherical_fresnel(eta_c);
    return clamp(1.0 - (1.0 - e) / (eta_c * eta_c), 0.0, 1.0);
}

// `openpbr` equations `internal_diffuse_reflection_coefficient_for_general_base`
// / `dielectric_roughness_estimate` / `base_roughness_estimate`: the effective
// internal-reflection coefficient blend. Mirrors
// `clearcoat.rs::coat_base_roughness_estimate` + `_coefficient`.
fn cc_base_roughness_estimate(
    specular_roughness: f32,
    specular_weight: f32,
    specular_ior: f32,
    base_metalness: f32,
) -> f32 {
    let r = clamp(specular_roughness, 0.0, 1.0);
    let ior = clamp(specular_ior, 1.0, 3.0);
    let t = (1.0 - ior) / (1.0 + ior);
    let w = clamp(clamp(specular_weight, 0.0, 1.0) * (t * t), 0.0, 1.0);
    let r_d = 1.0 + (r - 1.0) * w;
    let m = clamp(base_metalness, 0.0, 1.0);
    return r_d + (r - r_d) * m;
}

fn cc_internal_reflection_coefficient(eta_c: f32, mu_o: f32, r_b: f32) -> f32 {
    let k_s = fresnel_dielectric(mu_o, eta_c);
    let k_r = cc_internal_reflection_rough(eta_c);
    return clamp(k_s + (k_r - k_s) * clamp(r_b, 0.0, 1.0), 0.0, 1.0);
}

// `openpbr` equation `general_darkening_formula`: `Δ = (1 − K)/(1 − E_b·K)`.
fn cc_darkening_delta(base_albedo: f32, k: f32) -> f32 {
    let eb = clamp(base_albedo, 0.0, 1.0);
    let kk = clamp(k, 0.0, 1.0);
    let den = 1.0 - eb * kk;
    if den <= 1.0e-30 {
        return 1.0;
    }
    return clamp((1.0 - kk) / den, 0.0, 1.0);
}

// `openpbr` equation `modulated_darkening_factor`: `lerp(1, Δ, C·δ)`.
fn cc_modulated_darkening(coat_weight: f32, coat_darkening: f32, delta: f32) -> f32 {
    let c = clamp(coat_weight, 0.0, 1.0);
    let d = clamp(coat_darkening, 0.0, 1.0);
    return 1.0 + (clamp(delta, 0.0, 1.0) - 1.0) * (c * d);
}

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
    let f0 = mix(dielectric_f0(surface), surface.base_color, metallic);
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
    // `OpenPBR` base group, shared with the principled lobe: `diffuse_roughness`
    // turns the base diffuse into Oren-Nayar (0 = exactly Lambert) and
    // `base_weight` scales the whole base layer (1.0 = exactly the identity).
    let diffuse_factor =
        oren_nayar_factor(n_dot_v, n_dot_l, dot(v, l), surface.diffuse_roughness);
    let diffuse = (surface.base_color * (INV_PI * diffuse_factor)) * diffuse_weight;
    let base = (diffuse + specular) * (n_dot_l * clamp(surface.base_weight, 0.0, 1.0));

    // Coat layer: a second smooth dielectric GGX lobe. The coat is an
    // index-matched white dielectric, so its Fresnel is achromatic; the scalar
    // `Fc` drives both the coat highlight and the base attenuation.
    let coat = clamp(surface.clearcoat, 0.0, 1.0);
    let coat_roughness = clamp(surface.clearcoat_roughness, CLEARCOAT_MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    // The coat's Fresnel is achromatic (driven solely by `coat_ior`), so the
    // scalar `Fc` drives both the highlight and the base attenuation.
    let coat_ior = max(surface.coat_ior, BRDF_MIN_DIELECTRIC_IOR);
    let coat_f = fresnel_dielectric(v_dot_h, coat_ior);
    // `OpenPBR` `coat_normal`: the coat layer's own interface normal when a
    // clearcoat normal map is authored; it drives the coat microfacet geometry
    // only, while the base keeps the base normal.
    let coat_n = clearcoat_interface_normal(surface.coat_normal, n);
    let coat_n_dot_h = dot(coat_n, h);
    let coat_n_dot_v = dot(coat_n, v);
    let coat_n_dot_l = dot(coat_n, l);
    var coat_d = 0.0;
    var coat_g = 0.0;
    if coat_n_dot_l > 0.0 && coat_n_dot_v > 0.0 && coat_n_dot_h > 0.0 {
        let coat_aniso = clamp(surface.coat_roughness_anisotropy, 0.0, 1.0);
        if coat_aniso > 0.0 {
            // `OpenPBR` `coat_roughness_anisotropy`: same shared tangent frame
            // and `openpbr-anisotropy-formula` split as the base specular lobe
            // (α_t = α·sqrt(2/(1+(1−a)²)), α_b = (1−a)·α_t).
            let axes = anisotropic_axes(
                coat_n,
                frame.tangent,
                frame.bitangent,
                surface.anisotropy_rotation,
            );
            let one_minus = 1.0 - coat_aniso;
            let stretch = sqrt(2.0 / (1.0 + one_minus * one_minus));
            let ax = max(coat_alpha * stretch, MIN_ANISOTROPIC_ALPHA);
            let ay = max(one_minus * coat_alpha * stretch, MIN_ANISOTROPIC_ALPHA);
            coat_d = distribution_ggx_anisotropic(
                dot(axes.tangent, h),
                dot(axes.bitangent, h),
                coat_n_dot_h,
                ax,
                ay,
            );
            coat_g = vis_smith_anisotropic(
                dot(axes.tangent, v),
                dot(axes.bitangent, v),
                dot(axes.tangent, l),
                dot(axes.bitangent, l),
                max(coat_n_dot_v, CLEARCOAT_MIN_N_DOT),
                coat_n_dot_l,
                ax,
                ay,
            );
        } else {
            // Isotropic path: bit-for-bit the historical coat lobe.
            coat_d = distribution_ggx(coat_n_dot_h, coat_alpha);
            coat_g = vis_smith(max(coat_n_dot_v, CLEARCOAT_MIN_N_DOT), coat_n_dot_l, coat_alpha);
        }
    }
    // `OpenPBR` `coat_color` tints the highlight; white is the exact identity.
    let coat_specular = surface.coat_color * (coat * coat_f * coat_d * coat_g * coat_n_dot_l);

    // Double-pass transmission at the view and light angles (the reference PT
    // `CoatLayer` form): the base is gated entering and leaving the coat.
    let throughput = (1.0 - coat * fresnel_dielectric(n_dot_v, coat_ior))
        * (1.0 - coat * fresnel_dielectric(n_dot_l, coat_ior));
    // `OpenPBR` §Coat *Darkening*: internal reflections at the coat/air
    // interface make light strike the base repeatedly (darker, more saturated).
    // The specification's `modulated_darkening_factor` `lerp(1, Δ, C·δ)`
    // multiplies the whole base BSDF, with `Δ = (1 − K)/(1 − E_b·K)` and
    // `K = lerp(K_s, K_r, r_b)`. Byte-identical to `clearcoat.rs`.
    let r_b = cc_base_roughness_estimate(
        surface.perceptual_roughness,
        surface.specular_weight,
        surface.specular_ior,
        metallic,
    );
    let k = cc_internal_reflection_coefficient(coat_ior, n_dot_v, r_b);
    let e_b = clamp(
        (surface.base_color.r + surface.base_color.g + surface.base_color.b) / 3.0,
        0.0,
        1.0,
    );
    let delta = cc_darkening_delta(e_b, k);
    let darkening = cc_modulated_darkening(coat, clamp(surface.coat_darkening, 0.0, 1.0), delta);

    let reflected = base * (throughput * darkening) + coat_specular;
    let radiance = reflected * light.illuminance;
    return radiance * visibility + surface.emissive;
}
