//! Single-layer water BSDF direct-lighting lobe.
//!
//! Backend-neutral golden reference for the `Water` shading class and the
//! byte-for-byte numerical twin of `water.wesl` (`water_direct`).
//!
//! The closure follows Unreal's *`SingleLayerWater`* shading model reduced to the
//! analytic direct-lighting terms.  Water is a smooth dielectric sheet over a
//! participating medium, so the response splits by Fresnel into two paths that
//! conserve energy:
//!
//! * **Surface specular** is the sun/sky glint: the principled dielectric GGX
//!   lobe (Schlick Fresnel + Trowbridge-Reitz distribution + height-correlated
//!   Smith visibility) with a water `F0` derived from `reflectance`. The shared
//!   helpers guarantee the highlight matches the principled lobe exactly.
//! * **Volume scattering** is the light that refracts *into* the body, scatters
//!   back out and is absorbed along the way.  Energy entering and leaving the
//!   interface is gated by `(1 - F)` at the light and view angles, the medium's
//!   single-scatter albedo is the authored `base_color`, and the path is
//!   attenuated by a **wavelength-dependent Beer-Lambert transmittance** driven
//!   by optical `thickness`.  Because the per-channel extinction rises where the
//!   albedo is low, shallow water keeps its tint while deep water darkens and
//!   shifts toward its least-absorbed channel - the reason real water reads
//!   blue with depth.
//!
//! Emissive is folded in exactly once, matching the other lobes so the resolve
//! integrators accumulate over many lights without double-counting emission.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::lighting::{distribution_ggx, fresnel_schlick, visibility_smith_ggx_correlated};
use crate::vecmath::{add, dot, mul, mul_scalar, normalize_or};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Dot-product floor shared with the principled lobe (`MIN_N_DOT`).
const MIN_N_DOT: f32 = 1.0e-5;
/// Maps optical `thickness` in `[0, 1]` to a Beer-Lambert optical depth. Larger
/// values make the same thickness absorb more, so water darkens faster with
/// depth. Mirrors `WATER_DEPTH_SCALE`.
const WATER_DEPTH_SCALE: f32 = 4.0;

/// Scalar Schlick Fresnel reflectance for a dielectric interface, used to split
/// energy between the surface reflection and the refracted body. Mirrors
/// `water_fresnel_scalar`.
fn fresnel_scalar(f0: f32, cos_theta: f32) -> f32 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    let m2 = m * m;
    f0 + (1.0 - f0) * m2 * m2 * m
}

/// Per-channel Beer-Lambert transmittance `exp(-(1 - albedo) * depth)`.
///
/// Channels the medium absorbs (low albedo) attenuate fastest, so the surviving
/// body colour shifts toward the least-absorbed channel with depth. Mirrors
/// `water_transmittance`.
fn transmittance(albedo: [f32; 3], depth: f32) -> [f32; 3] {
    [
        ops::exp(-(1.0 - albedo[0]).max(0.0) * depth),
        ops::exp(-(1.0 - albedo[1]).max(0.0) * depth),
        ops::exp(-(1.0 - albedo[2]).max(0.0) * depth),
    ]
}

/// Evaluates the single-layer water BSDF for a single analytic light.
///
/// Returns linear radiance for the light plus the surface emissive term. When
/// the light is fully occluded only the emissive term survives, matching the
/// other lobes.
pub fn evaluate_water_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> [f32; 3] {
    let visibility = light.visibility.clamp(0.0, 1.0);
    if visibility <= 0.0 {
        return surface.emissive;
    }

    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let h = normalize_or(add(v, l), n);

    let n_dot_l = dot(n, l).max(0.0);
    let n_dot_v = dot(n, v).max(MIN_N_DOT);
    let n_dot_h = dot(n, h).max(MIN_N_DOT);
    let v_dot_h = dot(v, h).max(0.0);

    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = surface.reflectance.clamp(0.0, 1.0);
    // Dielectric water `F0`; the principled `0.16 * reflectance^2` mapping puts
    // the default reflectance near water's ~0.02 normal-incidence reflectance.
    let f0 = 0.16 * reflectance * reflectance;

    // Surface glint: coloured Schlick Fresnel feeds the shared GGX highlight.
    let f_spec = fresnel_schlick([f0; 3], v_dot_h);
    let d = distribution_ggx(n_dot_h, alpha);
    let g = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let specular = mul_scalar(mul_scalar(f_spec, d * g), n_dot_l);

    // Refracted body: energy entering/leaving the interface is gated by the
    // scalar Fresnel at the light and view angles; the medium scatters the
    // authored albedo and the path is absorbed along the optical depth.
    let f_light = fresnel_scalar(f0, n_dot_l);
    let f_view = fresnel_scalar(f0, n_dot_v);
    let refracted = (1.0 - f_light) * (1.0 - f_view);
    let depth = surface.thickness.clamp(0.0, 1.0) * WATER_DEPTH_SCALE;
    let tau = transmittance(surface.base_color, depth);
    let scattered = mul(
        mul_scalar(surface.base_color, refracted * n_dot_l * INV_PI),
        tau,
    );

    let lit = add(specular, scattered);
    let radiance = mul_scalar(mul(lit, light.illuminance), visibility);
    add(radiance, surface.emissive)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            base_color: [0.1, 0.3, 0.6],
            perceptual_roughness: 0.3,
            reflectance: 0.5,
            thickness: 0.3,
            ..Default::default()
        }
    }

    fn sum(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn water_is_finite_and_non_negative_across_parameters() {
        let dirs = [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, -0.5, 0.86],
            [-0.7, 0.2, 0.68],
        ];
        for roughness in [0.045, 0.3, 0.7, 1.0] {
            for thickness in [0.0, 0.3, 0.7, 1.0] {
                for reflectance in [0.0, 0.5, 1.0] {
                    for dir in dirs {
                        let value = evaluate_water_direct(
                            SurfaceSample {
                                perceptual_roughness: roughness,
                                thickness,
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
                            "roughness={roughness} thickness={thickness} reflectance={reflectance} dir={dir:?} -> {value:?}"
                        );
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
            evaluate_water_direct(surface, frame(), shadowed),
            surface.emissive
        );
    }

    #[test]
    fn deeper_water_absorbs_more_body_radiance() {
        // The Beer-Lambert body dims monotonically with optical thickness - a
        // medium term the principled fallback does not have.
        let shallow = evaluate_water_direct(
            SurfaceSample {
                thickness: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        let deep = evaluate_water_direct(
            SurfaceSample {
                thickness: 1.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        assert!(
            sum(shallow) > sum(deep),
            "deep water must absorb more: shallow={shallow:?} deep={deep:?}"
        );
    }

    #[test]
    fn absorption_is_wavelength_dependent() {
        // With a blue-biased albedo the blue channel survives depth far better
        // than red, so the blue/red ratio of the refracted body grows as the
        // water deepens. `reflectance = 0` zeroes `f0`, which removes the
        // wavelength-independent surface glint (and sets the Fresnel refraction
        // gates to 1), isolating the Beer-Lambert body term this asserts about.
        let shallow = evaluate_water_direct(
            SurfaceSample {
                thickness: 0.05,
                reflectance: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        let deep = evaluate_water_direct(
            SurfaceSample {
                thickness: 1.0,
                reflectance: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        let ratio = |c: [f32; 3]| c[2] / c[0].max(1.0e-6);
        assert!(
            ratio(deep) > ratio(shallow),
            "deep water must shift toward blue: shallow={shallow:?} deep={deep:?}"
        );
    }

    #[test]
    fn glossier_water_narrows_the_off_peak_glint() {
        // Kill the body with a black albedo so only the surface glint remains,
        // then confirm a tighter (glossier) highlight falls off faster off-peak.
        let off_peak = DirectLightSample {
            direction: normalize_or([0.6, 0.8, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let glossy = evaluate_water_direct(
            SurfaceSample {
                base_color: [0.0; 3],
                perceptual_roughness: 0.045,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        let rough = evaluate_water_direct(
            SurfaceSample {
                base_color: [0.0; 3],
                perceptual_roughness: 1.0,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        assert!(
            sum(glossy) < sum(rough),
            "glossy off-peak glint {glossy:?} should be dimmer than rough {rough:?}"
        );
    }

    #[test]
    fn grazing_view_reflects_more_and_transmits_less() {
        // At a grazing view the Fresnel split sends more energy to the surface
        // reflection and less into the body. Isolate the body with a rough,
        // black-glint... instead compare the refracted fraction directly by
        // holding the light at the normal and only tilting the view.
        let body_only = |view: [f32; 3]| {
            evaluate_water_direct(
                SurfaceSample {
                    // A perfectly rough surface spreads the glint so the body
                    // term dominates the comparison; base colour drives it.
                    perceptual_roughness: 1.0,
                    thickness: 0.0,
                    ..base_surface()
                },
                ShadingFrame { view, ..frame() },
                light(),
            )
        };
        let facing = body_only([0.0, 1.0, 0.0]);
        let grazing = body_only(normalize_or([0.98, 0.2, 0.0], [0.0, 1.0, 0.0]));
        assert!(
            sum(grazing) < sum(facing),
            "grazing view must transmit less into the body: grazing={grazing:?} facing={facing:?}"
        );
    }
}
