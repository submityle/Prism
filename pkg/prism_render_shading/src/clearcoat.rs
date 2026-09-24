//! Two-layer clear-coat BSDF direct-lighting lobe.
//!
//! Backend-neutral golden reference for the `ClearCoat` shading class and the
//! byte-for-byte numerical twin of `clearcoat.wesl` (`clearcoat_direct`).
//!
//! The closure follows Unreal's *`ClearCoatBxDF`*: a thin, smooth dielectric
//! coat (index-matched `F0 = 0.04`) sits above a fully featured principled base
//! layer. Two effects distinguish it from the shared principled fallback:
//!
//! * **A dedicated coat highlight.** The coat is a second dielectric GGX lobe
//!   driven by `clearcoat_roughness` (independent of the base roughness), so a
//!   rough painted base can still carry a tight, glossy varnish glint.
//! * **Double-pass coat absorption.** Light refracts *through* the coat on the
//!   way in and again on the way out, so the base layer is attenuated by the
//!   coat Fresnel **twice**: `(1 - clearcoat * Fc)^2`. The principled fallback
//!   only applies a single `(1 - clearcoat * Fc)` pass, so this lobe darkens the
//!   base under the coat more strongly at grazing angles - the physically
//!   correct behaviour Unreal models for the coat interface.
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
/// Normal-incidence reflectance of the dielectric coat interface. `0.04`
/// matches an index of refraction of ~1.5 (varnish/lacquer). Mirrors
/// `CLEARCOAT_F0`.
const CLEARCOAT_F0: f32 = 0.04;

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

    // Kulla-Conty-style bounded compensation, identical to the principled lobe
    // so the base matches it exactly when the coat is absent.
    let energy = 1.0 - 0.28 * roughness * roughness;
    let compensation = mul_scalar(f0, ((1.0 / energy.max(0.25)) - 1.0).min(1.0));
    let specular = add(single_scatter, compensation);
    let diffuse_weight = mul_scalar(sub([1.0; 3], f), 1.0 - metallic);
    let diffuse = mul(mul_scalar(surface.base_color, INV_PI), diffuse_weight);
    let base = mul_scalar(add(diffuse, specular), n_dot_l);

    // --- Coat layer: a second smooth dielectric GGX lobe ---
    let coat = surface.clearcoat.clamp(0.0, 1.0);
    let coat_roughness = surface.clearcoat_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let coat_alpha = coat_roughness * coat_roughness;
    // The coat is index-matched white dielectric, so its Fresnel is achromatic;
    // the scalar `Fc` drives both the coat highlight and the base attenuation.
    let coat_f = fresnel_schlick([CLEARCOAT_F0; 3], v_dot_h)[0].clamp(0.0, 1.0);
    let coat_d = distribution_ggx(n_dot_h, coat_alpha);
    let coat_g = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, coat_alpha);
    let coat_specular = [coat * coat_f * coat_d * coat_g * n_dot_l; 3];

    // Double-pass transmission: light is refracted through the coat on entry
    // and again on exit, so the base is gated by the coat Fresnel twice. This
    // is the physical departure from the principled single-pass fallback.
    let transmission = 1.0 - coat * coat_f;
    let throughput = transmission * transmission;

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
        let dirs = [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [-0.4, 0.7, 0.59],
        ];
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
}
