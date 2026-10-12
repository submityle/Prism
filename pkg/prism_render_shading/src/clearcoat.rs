//! Two-layer clear-coat BSDF direct-lighting lobe.
//!
//! Backend-neutral golden reference for the `ClearCoat` shading class and the
//! byte-for-byte numerical twin of `clearcoat.wesl` (`clearcoat_direct`).
//!
//! The closure follows Unreal's *`ClearCoatBxDF`*, upgraded to the `OpenPBR`
//! `coat` group (§4.4): a thin, smooth dielectric coat of relative index
//! `coat_ior` (default `1.5`, i.e. the historical `F0 = 0.04`) sits above a
//! fully featured principled base layer. Three effects distinguish it from the
//! shared principled fallback:
//!
//! * **A dedicated coat highlight.** The coat is a second dielectric GGX lobe
//!   driven by `clearcoat_roughness` (independent of the base roughness), so a
//!   rough painted base can still carry a tight, glossy varnish glint.
//! * **A per-channel coat tint.** `coat_color` scales the coat highlight, so a
//!   tinted varnish (white for clear lacquer) colours the glint.
//! * **Double-pass coat absorption.** Light refracts *through* the coat on the
//!   way in and again on the way out, so the base layer is attenuated by the
//!   coat Fresnel at the **view and light angles**: `(1 - clearcoat * Fc(n·v)) *
//!   (1 - clearcoat * Fc(n·l))`. The principled fallback applies a comparable
//!   two-pass term, so this lobe darkens the base under the coat at grazing
//!   angles - the physically correct behaviour Unreal models for the coat
//!   interface.
//!
//! The specialized coat treats the base as **isotropic** (real clear-coat
//! materials - car paint, lacquered wood - are overwhelmingly isotropic under
//! the varnish), so with `clearcoat == 0` and `anisotropy == 0` the base term
//! reduces exactly to [`evaluate_principled_direct`](crate::evaluate_principled_direct);
//! the `zero_clearcoat_matches_principled_base` test pins that equivalence.
//!
//! Emissive is folded in exactly once, matching the other lobes so the resolve
//! integrators accumulate over many lights without double-counting emission.

use core::f32::consts::PI;

use crate::lighting::{
    anisotropic_axes, dielectric_f0, distribution_ggx, distribution_ggx_anisotropic,
    fresnel_dielectric, fresnel_schlick, openpbr_anisotropy_to_alpha, oren_nayar_factor,
    visibility_smith_ggx_anisotropic, visibility_smith_ggx_correlated, MIN_DIELECTRIC_IOR,
};
use crate::vecmath::{add, cross, dot, mix3, mul, mul_scalar, normalize_or, sub};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// `OpenPBR` §Coat *Darkening*: the hemispherical (average) albedo of the coat
/// Fresnel, `E_F(η_c)`, via the specification's closed-form fit
/// `ln((10893η − 1438.2)/(−774.4η² + 10212η + 1))` for `η ≥ 1`, continued to
/// `η < 1` by the reciprocity identity `E_F(η) = 1 − η²(1 − E_F(1/η))`.
#[inline]
pub(crate) fn coat_hemispherical_fresnel(eta: f32) -> f32 {
    let eta = eta.max(MIN_DIELECTRIC_IOR);
    if eta >= 1.0 {
        coat_hemispherical_fresnel_ge1(eta)
    } else {
        let up = coat_hemispherical_fresnel_ge1(1.0 / eta);
        (1.0 - eta * eta * (1.0 - up)).clamp(0.0, 1.0)
    }
}

/// The `η ≥ 1` branch of [`coat_hemispherical_fresnel`], split out so the WESL
/// twin needs no recursion (WESL forbids it).
#[inline]
fn coat_hemispherical_fresnel_ge1(eta: f32) -> f32 {
    let num = 10893.0 * eta - 1438.2;
    let den = -774.4 * eta * eta + 10212.0 * eta + 1.0;
    if num <= 0.0 || den <= 0.0 {
        return 0.0;
    }
    bevy_math::ops::ln(num / den).clamp(0.0, 1.0)
}

/// `OpenPBR` equation `internal_diffuse_reflection_coefficient_for_rough_base`:
/// `K_r = 1 − (1 − E_F(η_c))/η_c²` — the internal-reflection fraction for a
/// fully rough (Lambertian-like) base.
#[inline]
pub(crate) fn coat_internal_reflection_rough(eta_c: f32) -> f32 {
    let e = coat_hemispherical_fresnel(eta_c);
    (1.0 - (1.0 - e) / (eta_c * eta_c)).clamp(0.0, 1.0)
}

/// `OpenPBR` equations `internal_diffuse_reflection_coefficient_for_general_base`
/// / `dielectric_roughness_estimate` / `base_roughness_estimate`: the effective
/// internal-reflection coefficient `K = lerp(K_s, K_r, r_b)`, where the
/// smooth-base limit is the coat Fresnel `K_s = F(μ_o, η_c)` and `r_b` blends
/// the dielectric (`lerp(1, r, ξ_s·F_s)`) and metal (`r`) roughness estimates
/// by `base_metalness`.
#[inline]
pub(crate) fn coat_base_roughness_estimate(
    specular_roughness: f32,
    specular_weight: f32,
    specular_ior: f32,
    base_metalness: f32,
) -> f32 {
    let r = specular_roughness.clamp(0.0, 1.0);
    let f_s = {
        let ior = specular_ior.clamp(1.0, 3.0);
        let t = (1.0 - ior) / (1.0 + ior);
        t * t
    };
    // `ξ_s·F_s` must be clamped into [0, 1] before it is used as a lerp weight.
    let w = (specular_weight.clamp(0.0, 1.0) * f_s).clamp(0.0, 1.0);
    let r_d = 1.0 + (r - 1.0) * w;
    let m = base_metalness.clamp(0.0, 1.0);
    r_d + (r - r_d) * m
}

/// `OpenPBR` equation `internal_diffuse_reflection_coefficient_for_general_base`.
#[inline]
pub(crate) fn coat_internal_reflection_coefficient(eta_c: f32, mu_o: f32, r_b: f32) -> f32 {
    let k_s = fresnel_dielectric(mu_o, eta_c);
    let k_r = coat_internal_reflection_rough(eta_c);
    (k_s + (k_r - k_s) * r_b.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// `OpenPBR` equation `general_darkening_formula`: `Δ = (1 − K)/(1 − E_b·K)`.
#[inline]
pub(crate) fn coat_darkening_delta(base_albedo: f32, k: f32) -> f32 {
    let eb = base_albedo.clamp(0.0, 1.0);
    let k = k.clamp(0.0, 1.0);
    let den = 1.0 - eb * k;
    if den <= f32::MIN_POSITIVE {
        return 1.0;
    }
    ((1.0 - k) / den).clamp(0.0, 1.0)
}

/// `OpenPBR` equation `modulated_darkening_factor`:
/// `lerp(1, Δ, C·δ)`, the uniform multiplier the specification applies to the
/// base BSDF beneath the coat.
#[inline]
pub(crate) fn coat_modulated_darkening(coat_weight: f32, coat_darkening: f32, delta: f32) -> f32 {
    let c = coat_weight.clamp(0.0, 1.0);
    let d = coat_darkening.clamp(0.0, 1.0);
    1.0 + (delta.clamp(0.0, 1.0) - 1.0) * (c * d)
}

/// Normal-incidence albedo of the base beneath the coat, reduced to the scalar
/// `E_b` that `general_darkening_formula` is written in terms of. The
/// specification leaves the reduction of the color-valued base albedo to the
/// implementation; Prism uses the arithmetic mean of the linear RGB channels,
/// which is exactly mirrorable in WESL.
#[inline]
fn base_normal_albedo(base_color: [f32; 3]) -> f32 {
    ((base_color[0] + base_color[1] + base_color[2]) / 3.0).clamp(0.0, 1.0)
}

/// `OpenPBR` §Coat *Roughening*: a rough coat convolves the base NDF, so the
/// base's effective specular roughness becomes
/// `r'_B = lerp(r_B, min(1, r_B⁴ + 2·r_C⁴)^(1/4), C)`. `C == 0` is the exact
/// identity and `r_C == 0` leaves the base untouched.
#[inline]
pub(crate) fn coat_roughened_base_roughness(r_base: f32, r_coat: f32, coat_weight: f32) -> f32 {
    let rb = r_base.clamp(0.0, 1.0);
    let rc = r_coat.clamp(0.0, 1.0);
    let c = coat_weight.clamp(0.0, 1.0);
    let rb2 = rb * rb;
    let rc2 = rc * rc;
    let convolved = (rb2 * rb2 + 2.0 * rc2 * rc2).min(1.0).sqrt().sqrt();
    rb + (convolved - rb) * c
}

/// `OpenPBR` §Coat *View-dependent absorption*: `coat_color` is the **two-way**
/// normal-incidence transmittance through the coating, so the base beneath is
/// tinted by `coat_color^((1/μᵗᵢ + 1/μᵗₒ)/2)` with the ray cosines refracted
/// into the coat, `μᵗ = sqrt(1 − (1 − μ²)/η²_ca)`. A totally internally
/// reflected ray travels an unbounded path and contributes no transmittance.
#[inline]
pub(crate) fn coat_absorption_rgb(
    coat_color: [f32; 3],
    mu_i: f32,
    mu_o: f32,
    eta_ca: f32,
) -> [f32; 3] {
    let mu_t = |mu: f32| -> f32 {
        let m = mu.clamp(0.0, 1.0);
        let s = 1.0 - (1.0 - m * m) / (eta_ca * eta_ca);
        if s <= 0.0 {
            0.0
        } else {
            s.sqrt()
        }
    };
    let ti = mu_t(mu_i);
    let to = mu_t(mu_o);
    if ti <= 0.0 || to <= 0.0 {
        return [0.0; 3];
    }
    let exponent = 0.5 * (1.0 / ti + 1.0 / to);
    [
        bevy_math::ops::powf(coat_color[0].clamp(0.0, 1.0), exponent),
        bevy_math::ops::powf(coat_color[1].clamp(0.0, 1.0), exponent),
        bevy_math::ops::powf(coat_color[2].clamp(0.0, 1.0), exponent),
    ]
}

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Dot-product floor shared with the principled lobe (`MIN_N_DOT`).
const MIN_N_DOT: f32 = 1.0e-5;
/// The coat interface normal: `OpenPBR` `coat_normal` when authored, else the
/// base shading normal.
///
/// The all-zero [`SurfaceSample::coat_normal`] is the identity sentinel (see
/// `lighting.rs`), so a surface without a clearcoat normal map evaluates the
/// coat lobe against the base normal exactly as it did before the field existed.
#[inline]
fn coat_interface_normal(coat_normal: [f32; 3], base_normal: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = coat_normal;
    if x * x + y * y + z * z > 0.0 {
        normalize_or(coat_normal, base_normal)
    } else {
        base_normal
    }
}

/// `OpenPBR` `geometry_coat_tangent`, falling back to the base frame tangent
/// when unauthored (the all-zero identity sentinel).
#[inline]
fn coat_tangent_or(
    coat_tangent: [f32; 3],
    base_tangent: [f32; 3],
) -> [f32; 3] {
    let [x, y, z] = coat_tangent;
    if x * x + y * y + z * z > 0.0 {
        normalize_or(coat_tangent, base_tangent)
    } else {
        base_tangent
    }
}

/// The bitangent matching [`coat_tangent_or`]: the coat's own (or the base
/// frame's) handedness axis.
#[inline]
fn coat_bitangent_or(
    coat_tangent: [f32; 3],
    coat_normal: [f32; 3],
    base_tangent: [f32; 3],
    base_bitangent: [f32; 3],
) -> [f32; 3] {
    let [x, y, z] = coat_tangent;
    if x * x + y * y + z * z > 0.0 {
        cross(coat_normal, normalize_or(coat_tangent, base_tangent))
    } else {
        base_bitangent
    }
}

/// Evaluates the two-layer clear-coat BSDF for a single analytic light.
///
/// Returns linear radiance for the light plus the surface emissive term. When
/// the light is fully occluded or below the horizon only the emissive term
/// survives, matching the principled lobe.
pub fn evaluate_clearcoat_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> [f32; 3] {
    let visibility = light.visibility.clamp(0.0, 1.0);

    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let h = normalize_or(add(v, l), n);

    let n_dot_l = dot(n, l).max(0.0);
    let n_dot_v = dot(n, v).max(MIN_N_DOT);
    let n_dot_h = dot(n, h).max(MIN_N_DOT);
    let v_dot_h = dot(v, h).max(0.0);
    if n_dot_l <= 0.0 {
        return surface.emissive;
    }

    // --- Base layer: isotropic principled dielectric/metal ---
    let metallic = surface.metallic.clamp(0.0, 1.0);
    // `OpenPBR` §Coat *Roughening*: the coat's own NDF convolves the base's, so
    // the base specular lobe below uses `r'_B` rather than the authored
    // `specular_roughness`. `coat_weight == 0` reduces it to the exact identity,
    // which is what keeps the no-coat path bit-identical to the principled lobe.
    let coat = surface.clearcoat.clamp(0.0, 1.0);
    let coat_roughness = surface.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let roughness = coat_roughened_base_roughness(
        surface.perceptual_roughness,
        surface.clearcoat_roughness,
        coat,
    )
    .clamp(MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let f0 = mix3(dielectric_f0(&surface), surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);
    let d = distribution_ggx(n_dot_h, alpha);
    let g = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let single_scatter = mul_scalar(f, d * g);

    // Multiple-scattering energy compensation (Kulla-Conty 2017 + Turquin 2019),
    // byte-identical to `evaluate_principled_direct` so the base reduces to the
    // principled lobe exactly when the coat is absent (pinned by
    // `zero_clearcoat_matches_principled_base`). This replaces the former crude
    // `1 - 0.28*rough^2` scalar, which drifted out of sync when the principled
    // base adopted the analytic multiscatter model in P0-a.
    let compensation = crate::gi::env_brdf::multiscatter::multiscatter_direct(
        bevy_math::Vec3::from_array(f0),
        n_dot_v,
        n_dot_l,
        roughness,
    )
    .to_array();
    let specular = add(single_scatter, compensation);
    let diffuse_weight = mul_scalar(sub([1.0; 3], f), 1.0 - metallic);
    // `OpenPBR` base group, shared with the principled lobe: `diffuse_roughness`
    // turns the base diffuse into Oren-Nayar (`0` = exactly Lambert) and
    // `base_weight` scales the whole base layer (`1.0` = exactly the identity).
    // Both are identity at their defaults, so a material that never authors them
    // keeps the historical coat response bit-for-bit.
    let diffuse_factor =
        oren_nayar_factor(n_dot_v, n_dot_l, dot(v, l), surface.diffuse_roughness);
    let diffuse = mul(
        mul_scalar(surface.base_color, INV_PI * diffuse_factor),
        diffuse_weight,
    );
    let base = mul_scalar(
        add(diffuse, specular),
        n_dot_l * surface.base_weight.clamp(0.0, 1.0),
    );

    // --- Coat layer: a second smooth dielectric GGX lobe ---
    let coat_alpha = coat_roughness * coat_roughness;
    // The coat's Fresnel is achromatic (driven solely by `coat_ior`), so the
    // scalar `Fc` drives both the highlight and the base attenuation.
    let coat_ior = surface.coat_ior.max(MIN_DIELECTRIC_IOR);
    let coat_f = fresnel_dielectric(v_dot_h, coat_ior);
    // OpenPBR `coat_normal`: the coat layer has its own interface normal when a
    // clearcoat normal map is authored. It drives the coat's microfacet
    // geometry only; the base layer and its attenuation keep the base normal.
    let coat_n = coat_interface_normal(surface.coat_normal, n);
    let coat_n_dot_h = dot(coat_n, h);
    let coat_n_dot_v = dot(coat_n, v);
    let coat_n_dot_l = dot(coat_n, l);
    // A coat facet that faces away from the light reflects nothing into it, but
    // must not gate the base (the base is evaluated against its own normal).
    let (coat_d, coat_g) = if coat_n_dot_l > 0.0 && coat_n_dot_v > 0.0 && coat_n_dot_h > 0.0 {
        let coat_aniso = surface.coat_roughness_anisotropy.clamp(0.0, 1.0);
        if coat_aniso > 0.0 {
            // `OpenPBR` `coat_roughness_anisotropy`: elongate the coat highlight
            // along the shared surface tangent frame, exactly as the base
            // specular lobe does (same axes, same specification
            // `openpbr-anisotropy-formula` split).
            let (coat_t, coat_b) = anisotropic_axes(
                coat_n,
                coat_tangent_or(surface.coat_tangent, frame.tangent),
                coat_bitangent_or(
                    surface.coat_tangent,
                    coat_n,
                    frame.tangent,
                    frame.bitangent,
                ),
                surface.anisotropy_rotation,
            );
            let (ax, ay) = openpbr_anisotropy_to_alpha(coat_alpha, coat_aniso);
            (
                distribution_ggx_anisotropic(
                    dot(coat_t, h),
                    dot(coat_b, h),
                    coat_n_dot_h,
                    ax,
                    ay,
                ),
                visibility_smith_ggx_anisotropic(
                    dot(coat_t, v),
                    dot(coat_b, v),
                    dot(coat_t, l),
                    dot(coat_b, l),
                    coat_n_dot_v.max(MIN_N_DOT),
                    coat_n_dot_l,
                    ax,
                    ay,
                ),
            )
        } else {
            // Isotropic path: bit-for-bit the historical coat lobe.
            (
                distribution_ggx(coat_n_dot_h, coat_alpha),
                visibility_smith_ggx_correlated(
                    coat_n_dot_v.max(MIN_N_DOT),
                    coat_n_dot_l,
                    coat_alpha,
                ),
            )
        }
    } else {
        (0.0, 0.0)
    };
    // The coat's own specular reflection is *not* tinted by `coat_color`: the
    // specification reserves `coat_color` for the absorbing medium, which tints
    // the base instead (see the transmission factor below).
    let coat_specular = [coat * coat_f * coat_d * coat_g * coat_n_dot_l; 3];

    // Double-pass transmission: light is refracted through the coat on entry
    // and again on exit, so the base is gated by the coat Fresnel at the view
    // and light angles (the reference PT `CoatLayer` form). This is the physical
    // departure from the principled single-pass fallback.
    let throughput = (1.0 - coat * fresnel_dielectric(n_dot_v, coat_ior))
        * (1.0 - coat * fresnel_dielectric(n_dot_l, coat_ior));
    // `OpenPBR` §Coat *Darkening*: internal reflections at the coat/air
    // interface make light strike the base repeatedly (more absorption, so a
    // darker and more saturated base). The specification's
    // `modulated_darkening_factor` `lerp(1, Δ, C·δ)` multiplies the whole base
    // BSDF, with `Δ = (1 − K)/(1 − E_b·K)` from `general_darkening_formula` and
    // the internal-reflection coefficient `K = lerp(K_s, K_r, r_b)` blended by
    // the estimated base roughness. `δ == 0` restores the un-darkened base
    // exactly (`lerp(1, Δ, 0) = 1`).
    let coat_darkening = surface.coat_darkening.clamp(0.0, 1.0);
    let r_b = coat_base_roughness_estimate(
        surface.perceptual_roughness,
        surface.specular_weight,
        surface.specular_ior,
        metallic,
    );
    let k = coat_internal_reflection_coefficient(coat_ior, n_dot_v, r_b);
    let e_b = base_normal_albedo(surface.base_color);
    let delta = coat_darkening_delta(e_b, k);
    let darkening = coat_modulated_darkening(coat, coat_darkening, delta);
    // `OpenPBR` §Coat *View-dependent absorption*: the absorbing medium tints the
    // light that reaches the base and returns, so the base — and only the base —
    // is multiplied by the two-way transmittance. `lerp(1, ·, C)` keeps
    // `coat_weight == 0` an exact identity.
    let absorption = coat_absorption_rgb(surface.coat_color, n_dot_v, n_dot_l, coat_ior);
    let base = mul(
        base,
        [
            1.0 + (absorption[0] - 1.0) * coat,
            1.0 + (absorption[1] - 1.0) * coat,
            1.0 + (absorption[2] - 1.0) * coat,
        ],
    );

    let reflected = add(mul_scalar(base, throughput * darkening), coat_specular);
    let radiance = mul(reflected, light.illuminance);
    add(mul_scalar(radiance, visibility), surface.emissive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate_principled_direct;

    fn frame() -> ShadingFrame {
        ShadingFrame {
            normal: [0.0, 1.0, 0.0],
            view: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent: [0.0, 0.0, -1.0],
        }
    }

    fn light() -> DirectLightSample {
        DirectLightSample {
            direction: [0.0, 1.0, 0.0],
            illuminance: [1.0; 3],
            visibility: 1.0,
        }
    }

    fn base_surface() -> SurfaceSample {
        SurfaceSample {
            base_color: [0.2, 0.4, 0.6],
            metallic: 0.0,
            perceptual_roughness: 0.4,
            clearcoat: 0.8,
            clearcoat_roughness: 0.1,
            ..Default::default()
        }
    }

    fn sum(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn clearcoat_is_finite_and_non_negative_across_parameters() {
        let dirs = [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, -0.5, 0.86],
            [-0.7, 0.2, 0.68],
        ];
        for roughness in [0.045, 0.3, 0.7, 1.0] {
            for clearcoat in [0.0, 0.5, 1.0] {
                for coat_roughness in [0.045, 0.3, 1.0] {
                    for specular_weight in [0.0, 0.5, 1.0] {
                        for dir in dirs {
                            let value = evaluate_clearcoat_direct(
                                SurfaceSample {
                                    perceptual_roughness: roughness,
                                    clearcoat,
                                    clearcoat_roughness: coat_roughness,
                                    specular_weight,
                                    ..base_surface()
                                },
                                frame(),
                                DirectLightSample {
                                    direction: dir,
                                    ..light()
                                },
                            );
                            assert!(
                                value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                                "r={roughness} cc={clearcoat} ccr={coat_roughness} w={specular_weight} dir={dir:?} -> {value:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_visibility_returns_only_emissive() {
        let surface = SurfaceSample {
            emissive: [0.05, 0.06, 0.07],
            ..base_surface()
        };
        let shadowed = DirectLightSample {
            visibility: 0.0,
            ..light()
        };
        assert_eq!(
            evaluate_clearcoat_direct(surface, frame(), shadowed),
            surface.emissive
        );
    }

    #[test]
    fn zero_clearcoat_matches_principled_base() {
        // With no coat and no anisotropy the specialized base must reduce to the
        // principled lobe: this is the cross-check that the isotropic base layer
        // is not a divergent re-implementation.
        let surface = SurfaceSample {
            clearcoat: 0.0,
            anisotropy: 0.0,
            emissive: [0.02, 0.03, 0.04],
            ..base_surface()
        };
        let dirs = [[0.0, 1.0, 0.0], [0.6, 0.8, 0.0], [-0.4, 0.7, 0.59]];
        for dir in dirs {
            let sample = DirectLightSample {
                direction: dir,
                ..light()
            };
            let coated = evaluate_clearcoat_direct(surface, frame(), sample);
            let principled = evaluate_principled_direct(surface, frame(), sample);
            for c in 0..3 {
                let diff = (coated[c] - principled[c]).abs();
                let tol = 1.0e-3 * principled[c].abs().max(1.0);
                assert!(
                    diff <= tol,
                    "dir={dir:?} channel {c}: coated={coated:?} principled={principled:?}"
                );
            }
        }
    }

    #[test]
    fn coat_adds_glint_at_peak() {
        // On the specular peak the coat lobe injects a bright highlight; even
        // after the double-pass base attenuation the net radiance rises.
        let coated = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                // Isolate the highlight: with the specification's darkening at
                // its default (`coat_darkening == 1`) the coat also removes the
                // internally-reflected energy from the base, which can exceed the
                // added glint. `coat_darkening == 0` is the specification's own
                // "un-darkened base" escape hatch.
                coat_darkening: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        let uncoated = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 0.0,
                coat_darkening: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        assert!(
            sum(coated) > sum(uncoated),
            "coat should add a peak glint: coated={coated:?} uncoated={uncoated:?}"
        );
    }

    #[test]
    fn coat_attenuates_base_at_grazing() {
        // At a grazing view the coat Fresnel approaches 1, so the double-pass
        // transmission drives the base toward zero; with a rough coat the added
        // highlight is spread thin, so the coated result is dimmer than the
        // uncoated base.
        let grazing = ShadingFrame {
            view: normalize_or([0.985, 0.174, 0.0], [0.0, 1.0, 0.0]),
            ..frame()
        };
        let coated = evaluate_clearcoat_direct(
            SurfaceSample {
                perceptual_roughness: 1.0,
                clearcoat: 1.0,
                clearcoat_roughness: 1.0,
                ..base_surface()
            },
            grazing,
            light(),
        );
        let uncoated = evaluate_clearcoat_direct(
            SurfaceSample {
                perceptual_roughness: 1.0,
                clearcoat: 0.0,
                clearcoat_roughness: 1.0,
                ..base_surface()
            },
            grazing,
            light(),
        );
        assert!(
            sum(coated) < sum(uncoated),
            "grazing coat must attenuate the base: coated={coated:?} uncoated={uncoated:?}"
        );
    }

    #[test]
    fn glossier_coat_narrows_off_peak_glint() {
        // Off the specular peak a tighter (glossy) coat sheds energy faster than
        // a rough coat; the base and its attenuation are identical, so only the
        // coat lobe width drives the difference.
        let off_peak = DirectLightSample {
            direction: normalize_or([0.6, 0.8, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let glossy = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                clearcoat_roughness: 0.045,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        let rough = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                clearcoat_roughness: 1.0,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        assert!(
            sum(glossy) < sum(rough),
            "glossy off-peak coat {glossy:?} should be dimmer than rough {rough:?}"
        );
    }

    #[test]
    fn coat_ior_default_reproduces_the_legacy_normal_incidence_f0() {
        // The exact dielectric Fresnel must reduce to the historical fixed
        // `F0 = 0.04` at normal incidence and index 1.5, so a default material
        // keeps the same interface reflectance it had before `coat_ior` existed.
        assert!((fresnel_dielectric(1.0, 1.5) - 0.04).abs() < 1.0e-6);
        // Fresnel grows toward grazing and never leaves [0, 1].
        let grazing = fresnel_dielectric(0.05, 1.5);
        let normal = fresnel_dielectric(1.0, 1.5);
        assert!(grazing > normal);
        for cos in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for ior in [1.001, 1.3, 1.5, 2.0, 3.0] {
                let f = fresnel_dielectric(cos, ior);
                assert!(f.is_finite() && (0.0..=1.0).contains(&f), "cos={cos} ior={ior} -> {f}");
            }
        }
    }

    #[test]
    fn denser_coat_ior_strengthens_the_interface() {
        // A denser coat reflects more at the interface, which both brightens the
        // peak highlight and deepens the base attenuation.
        let light_coat = evaluate_clearcoat_direct(
            SurfaceSample {
                coat_ior: 1.1,
                ..base_surface()
            },
            frame(),
            light(),
        );
        let dense_coat = evaluate_clearcoat_direct(
            SurfaceSample {
                coat_ior: 2.4,
                ..base_surface()
            },
            frame(),
            light(),
        );
        assert!(
            sum(dense_coat) > sum(light_coat),
            "denser coat {dense_coat:?} should out-reflect lighter {light_coat:?} at the peak"
        );
    }

    #[test]
    fn coat_color_tints_the_base_not_the_highlight() {
        // `OpenPBR` §Coat *View-dependent absorption*: `coat_color` describes the
        // **absorbing medium** of the coat, so it tints the light that reaches
        // the base. The specification is explicit that the specular reflection
        // from the coat itself is *not* tinted.
        let base = base_surface();
        let white = evaluate_clearcoat_direct(base, frame(), light());
        let red = evaluate_clearcoat_direct(
            SurfaceSample {
                coat_color: [1.0, 0.0, 0.0],
                ..base
            },
            frame(),
            light(),
        );
        // The base beneath the red coat loses green and blue.
        assert!(red[1] < white[1], "red coat must absorb green: {red:?} vs {white:?}");
        assert!(red[2] < white[2], "red coat must absorb blue: {red:?} vs {white:?}");
        // Isolating the coat with `base_weight = 0` proves the highlight is
        // untouched: white and red `coat_color` give byte-identical results.
        let coat_only = |color: [f32; 3]| {
            evaluate_clearcoat_direct(
                SurfaceSample {
                    base_weight: 0.0,
                    coat_color: color,
                    ..base
                },
                frame(),
                light(),
            )
        };
        assert_eq!(
            coat_only([1.0; 3]),
            coat_only([1.0, 0.0, 0.0]),
            "coat_color must not tint the coat's own highlight"
        );
    }

    #[test]
    fn coat_interface_normal_is_the_identity_when_unset() {
        let base = normalize_or([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);
        // The all-zero sentinel falls back to the base normal exactly.
        assert_eq!(coat_interface_normal([0.0; 3], base), base);
        // A non-zero coat normal is normalized against the base as a fallback.
        let tilted = coat_interface_normal([0.3, 1.0, 0.0], base);
        assert!((tilted[0] * tilted[0] + tilted[1] * tilted[1] + tilted[2] * tilted[2] - 1.0).abs() < 1.0e-5);
        // Degenerate (non-normalizable) input falls back rather than dividing by zero.
        let tiny = coat_interface_normal([1.0e-30, 1.0e-30, 0.0], base);
        assert!(tiny.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn tilted_coat_normal_perturbs_only_the_coat_highlight() {
        // Isolate the coat contribution as (with coat) - (without coat): the
        // uncoated result ignores `coat_normal` entirely, so any difference in
        // the delta is the coat lobe reacting to its own interface normal.
        let zero_normal_glint = {
            let coated = evaluate_clearcoat_direct(
                SurfaceSample {
                    clearcoat: 1.0,
                    ..base_surface()
                },
                frame(),
                light(),
            );
            let uncoated = evaluate_clearcoat_direct(
                SurfaceSample {
                    clearcoat: 0.0,
                    ..base_surface()
                },
                frame(),
                light(),
            );
            sum(coated) - sum(uncoated)
        };
        let tilted_normal_glint = {
            let coated = evaluate_clearcoat_direct(
                SurfaceSample {
                    clearcoat: 1.0,
                    coat_normal: normalize_or([0.6, 0.8, 0.0], [0.0, 1.0, 0.0]),
                    ..base_surface()
                },
                frame(),
                light(),
            );
            let uncoated = evaluate_clearcoat_direct(
                SurfaceSample {
                    clearcoat: 0.0,
                    ..base_surface()
                },
                frame(),
                light(),
            );
            sum(coated) - sum(uncoated)
        };
        assert!(
            (tilted_normal_glint - zero_normal_glint).abs() > 1.0e-4,
            "a tilted coat normal must change the coat glint: tilted={tilted_normal_glint} zero={zero_normal_glint}"
        );
    }

    #[test]
    fn coat_normal_facing_away_zeroes_only_the_highlight() {
        // A coat normal pointing away from the light reflects nothing into it.
        // `base_weight = 0` removes the whole base layer, leaving the coat's own
        // reflection: facing away it must be exactly zero, facing the light it
        // must be positive. (`coat_color` is the *absorbing medium's* tint and
        // does not touch the highlight, so it cannot be used to zero it here.)
        let coat_only = |normal: [f32; 3]| {
            evaluate_clearcoat_direct(
                SurfaceSample {
                    clearcoat: 1.0,
                    coat_normal: normal,
                    base_weight: 0.0,
                    ..base_surface()
                },
                frame(),
                light(),
            )
        };
        let toward = coat_only([0.0; 3]);
        let away = coat_only([0.0, -1.0, 0.0]);
        assert!(sum(toward) > 0.0, "coat highlight missing: {toward:?}");
        assert_eq!(away, [0.0; 3], "facing-away coat still reflects: {away:?}");

        // And the orientation changes *only* the highlight: the difference
        // between the full surfaces equals the difference between the
        // coat-only surfaces, i.e. the base contribution cancels exactly.
        let full = |normal: [f32; 3]| {
            evaluate_clearcoat_direct(
                SurfaceSample {
                    clearcoat: 1.0,
                    coat_normal: normal,
                    ..base_surface()
                },
                frame(),
                light(),
            )
        };
        let (full_toward, full_away) = (full([0.0; 3]), full([0.0, -1.0, 0.0]));
        for c in 0..3 {
            let base_delta = full_away[c] - full_toward[c];
            let coat_delta = away[c] - toward[c];
            assert!(
                (base_delta - coat_delta).abs() < 1.0e-6,
                "channel {c}: base leaked through the coat normal, \
                 full={full_toward:?}/{full_away:?} coat={toward:?}/{away:?}"
            );
        }
    }

    #[test]
    fn coat_roughness_anisotropy_changes_the_off_peak_glint() {
        // Off the specular peak the coat highlight samples the microfacet
        // distribution away from the normal, where the aspect split matters;
        // at the exact peak `ax * ay == alpha^2` leaves it untouched.
        let off_peak = DirectLightSample {
            direction: normalize_or([0.6, 0.8, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let isotropic = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                clearcoat_roughness: 0.5,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        let anisotropic = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                clearcoat_roughness: 0.5,
                coat_roughness_anisotropy: 1.0,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        assert!(
            (sum(anisotropic) - sum(isotropic)).abs() > 1.0e-4,
            "anisotropic coat must change the off-peak glint: {anisotropic:?} vs {isotropic:?}"
        );
    }

    #[test]
    fn coat_anisotropy_rotation_rotates_the_elongation() {
        // The elongation axis is the shared surface tangent frame rotated by
        // `anisotropy_rotation`, so swapping the axis by 90 degrees must move
        // the off-peak highlight.
        let off_peak = DirectLightSample {
            direction: normalize_or([0.6, 0.8, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let along_tangent = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                clearcoat_roughness: 0.5,
                coat_roughness_anisotropy: 1.0,
                anisotropy_rotation: 0.0,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        let along_bitangent = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                clearcoat_roughness: 0.5,
                coat_roughness_anisotropy: 1.0,
                anisotropy_rotation: core::f32::consts::FRAC_PI_2,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        assert!(
            (sum(along_tangent) - sum(along_bitangent)).abs() > 1.0e-4,
            "rotating the coat anisotropy axis must move the glint: {along_tangent:?} vs {along_bitangent:?}"
        );
    }

    #[test]
    fn base_weight_zero_removes_the_coated_base_layer() {
        // Shared `OpenPBR` base group: a zero `base_weight` drops the base layer
        // under the coat. With no coat either, only emission is left.
        let surface = SurfaceSample {
            clearcoat: 0.0,
            base_weight: 0.0,
            emissive: [0.02, 0.03, 0.04],
            ..base_surface()
        };
        assert_eq!(evaluate_clearcoat_direct(surface, frame(), light()), surface.emissive);

        // The coat highlight itself is unaffected by `base_weight`.
        let coat_only = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                base_weight: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        assert!(sum(coat_only) > 0.0, "the coat glint must survive: {coat_only:?}");
    }

    #[test]
    fn diffuse_roughness_roughs_the_base_under_the_coat() {
        // Grezing geometry where Oren-Nayar's cavity inter-reflection shows up;
        // the coat is left on so the base is evaluated through its attenuation.
        let grazing = ShadingFrame {
            view: normalize_or([0.99, 0.14, 0.0], [0.0, 1.0, 0.0]),
            ..frame()
        };
        let grazing_light = DirectLightSample {
            direction: normalize_or([0.99, 0.14, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let smooth = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                metallic: 0.0,
                diffuse_roughness: 0.0,
                ..base_surface()
            },
            grazing,
            grazing_light,
        );
        let rough = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                metallic: 0.0,
                diffuse_roughness: 1.0,
                ..base_surface()
            },
            grazing,
            grazing_light,
        );
        assert!(
            rough[0] > smooth[0],
            "a rough coated base must retro-reflect more: {rough:?} vs {smooth:?}"
        );
    }

    #[test]
    fn coat_darkening_darkens_the_base_under_the_coat() {
        // `coat_darkening = 0` is the historical identity; raising it traps more
        // internally reflected light and darkens the base.
        let none = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                coat_darkening: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        let full = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 1.0,
                coat_darkening: 1.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        assert!(sum(full) < sum(none), "darkened={full:?} plain={none:?}");
        // A weightless coat cannot darken anything.
        let no_coat = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 0.0,
                coat_darkening: 1.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        let no_coat_plain = evaluate_clearcoat_direct(
            SurfaceSample {
                clearcoat: 0.0,
                coat_darkening: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        for c in 0..3 {
            assert!((no_coat[c] - no_coat_plain[c]).abs() < 1.0e-6);
        }
    }
}
