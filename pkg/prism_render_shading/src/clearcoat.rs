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

use crate::lighting::{distribution_ggx, fresnel_schlick, visibility_smith_ggx_correlated};
use crate::vecmath::{add, dot, mix3, mul, mul_scalar, normalize_or, sub};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Dot-product floor shared with the principled lobe (`MIN_N_DOT`).
const MIN_N_DOT: f32 = 1.0e-5;
/// Smallest relative coat index accepted, keeping the coat a genuine dielectric
/// interface. Mirrors `reference_pt::coat::MIN_COAT_IOR`.
const MIN_COAT_IOR: f32 = 1.0 + 1.0e-3;

/// Unpolarized dielectric Fresnel reflectance of the coat interface for a
/// direction whose cosine with the normal is `cos_i`, with the relative index
/// `eta_t / eta_i = coat_ior` and the exterior medium at index 1.
///
/// This is the exact Fresnel reflectance shared with the path-traced reference
/// (`reference_pt::dielectric::fresnel_dielectric(cos_i, 1.0, coat_ior)`),
/// including the total-internal-reflection branch. At `cos_i == 1` and
/// `coat_ior == 1.5` it returns exactly `0.04`, reproducing the legacy fixed
/// `F0`. Only `sqrt`, a divide and two squares are used, so it stays inside the
/// transcendental budget of the shading path.
#[inline]
fn fresnel_coat_dielectric(cos_i: f32, coat_ior: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let ior = coat_ior.max(MIN_COAT_IOR);
    // Snell's law in squared-sine form with eta_i = 1, eta_t = ior.
    let eta = 1.0 / ior;
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = eta * eta * sin2_i;
    if sin2_t >= 1.0 {
        // Beyond the critical angle: all energy is reflected.
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    // Parallel- and perpendicular-polarized amplitude reflection coefficients
    // for eta_i = 1 and eta_t = ior.
    let r_parl = (ior * cos_i - cos_t) / (ior * cos_i + cos_t);
    let r_perp = (cos_i - ior * cos_t) / (cos_i + ior * cos_t);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
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
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = surface.reflectance.clamp(0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix3([f0_dielectric; 3], surface.base_color, metallic);
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
    let diffuse = mul(mul_scalar(surface.base_color, INV_PI), diffuse_weight);
    let base = mul_scalar(add(diffuse, specular), n_dot_l);

    // --- Coat layer: a second smooth dielectric GGX lobe ---
    let coat = surface.clearcoat.clamp(0.0, 1.0);
    let coat_roughness = surface.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    // The coat's Fresnel is achromatic (driven solely by `coat_ior`), so the
    // scalar `Fc` drives both the highlight and the base attenuation.
    let coat_ior = surface.coat_ior.max(MIN_COAT_IOR);
    let coat_f = fresnel_coat_dielectric(v_dot_h, coat_ior);
    let coat_d = distribution_ggx(n_dot_h, coat_alpha);
    let coat_g = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, coat_alpha);
    // OpenPBR `coat_color` tints the highlight; white is the exact identity.
    let coat_specular = mul(
        surface.coat_color,
        [coat * coat_f * coat_d * coat_g * n_dot_l; 3],
    );

    // Double-pass transmission: light is refracted through the coat on entry
    // and again on exit, so the base is gated by the coat Fresnel at the view
    // and light angles (the reference PT `CoatLayer` form). This is the physical
    // departure from the principled single-pass fallback.
    let throughput = (1.0 - coat * fresnel_coat_dielectric(n_dot_v, coat_ior))
        * (1.0 - coat * fresnel_coat_dielectric(n_dot_l, coat_ior));

    let reflected = add(mul_scalar(base, throughput), coat_specular);
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
            reflectance: 0.5,
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
                    for reflectance in [0.0, 0.5, 1.0] {
                        for dir in dirs {
                            let value = evaluate_clearcoat_direct(
                                SurfaceSample {
                                    perceptual_roughness: roughness,
                                    clearcoat,
                                    clearcoat_roughness: coat_roughness,
                                    reflectance,
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
                                "r={roughness} cc={clearcoat} ccr={coat_roughness} refl={reflectance} dir={dir:?} -> {value:?}"
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
        assert!((fresnel_coat_dielectric(1.0, 1.5) - 0.04).abs() < 1.0e-6);
        // Fresnel grows toward grazing and never leaves [0, 1].
        let grazing = fresnel_coat_dielectric(0.05, 1.5);
        let normal = fresnel_coat_dielectric(1.0, 1.5);
        assert!(grazing > normal);
        for cos in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for ior in [1.001, 1.3, 1.5, 2.0, 3.0] {
                let f = fresnel_coat_dielectric(cos, ior);
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
    fn coat_color_tints_only_the_coat_highlight() {
        // A red coat keeps the red channel of the highlight and drops green and
        // blue; the base and its attenuation are untouched by the tint.
        let white = evaluate_clearcoat_direct(base_surface(), frame(), light());
        let red = evaluate_clearcoat_direct(
            SurfaceSample {
                coat_color: [1.0, 0.0, 0.0],
                ..base_surface()
            },
            frame(),
            light(),
        );
        assert!(red[0] >= white[0], "red coat must keep the red highlight: {red:?} vs {white:?}");
        assert!(red[1] < white[1], "red coat must drop green: {red:?} vs {white:?}");
        assert!(red[2] < white[2], "red coat must drop blue: {red:?} vs {white:?}");
    }
}
