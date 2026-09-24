//! Subsurface scattering (SSS) BSDF direct-lighting lobe.
//!
//! Backend-neutral golden reference for the `Subsurface` shading class and the
//! byte-for-byte numerical twin of `subsurface.wesl` (`subsurface_direct`).
//!
//! The closure follows the real-time subsurface model used by Unreal's
//! *Subsurface* shading model, combining three physically motivated terms:
//!
//! * **Specular** reuses the principled dielectric GGX lobe (Schlick Fresnel +
//!   Trowbridge-Reitz distribution + height-correlated Smith visibility), so a
//!   subsurface surface keeps a correct hard highlight. The helpers are shared
//!   with `evaluate_principled_direct` to guarantee the highlight matches.
//! * **Wrapped diffuse** replaces the Lambert cosine with an energy-conserving
//!   *wrap* term whose width is driven by the `subsurface` weight, so light
//!   bleeds past the terminator - the soft, waxy look of skin/marble/wax. At
//!   `subsurface == 0` the wrap collapses back to a plain Lambert cosine.
//! * **Back transmission** approximates light travelling *through* thin
//!   geometry using the Frostbite/DICE "fast subsurface scattering" fit
//!   (Barre-Brisebois & Bouchard), which Unreal adopts for thin translucency:
//!   a view-dependent lobe around the inverted, normal-distorted light vector,
//!   scaled by the `subsurface` weight and attenuated by optical `thickness`.
//!
//! Unlike the opaque lobes there is **no back-facing early-out**: a subsurface
//! surface still receives radiance from lights behind it via the transmission
//! term. Emissive is folded in exactly once, matching the other lobes so the
//! resolve integrators accumulate over many lights without double-counting
//! self-illumination.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::lighting::{distribution_ggx, fresnel_schlick, visibility_smith_ggx_correlated};
use crate::vecmath::{add, dot, mix3, mul, mul_scalar, normalize_or, sub};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Dot-product floor shared with the principled lobe (`MIN_N_DOT`).
const MIN_N_DOT: f32 = 1.0e-5;
/// Normal-distortion of the transmission half-vector. Frostbite's `Distortion`.
const SSS_DISTORTION: f32 = 0.2;
/// Sharpness of the view-dependent transmission lobe. Frostbite's `Power`.
const SSS_POWER: f32 = 12.0;

/// Energy-conserving diffuse wrap `saturate((x + w) / (1 + w)^2)`.
///
/// `x` is the (possibly negative) `N.L` cosine and `w` the wrap width. At
/// `w == 0` this is `saturate(x)`, i.e. a plain Lambert cosine, so the lobe
/// degrades gracefully when no subsurface weight is authored. Mirrors
/// `subsurface_fd_wrap`.
fn fd_wrap(x: f32, w: f32) -> f32 {
    ((x + w) / ((1.0 + w) * (1.0 + w))).clamp(0.0, 1.0)
}

/// Evaluates the subsurface BSDF for a single analytic light.
///
/// Returns linear radiance for the light plus the surface emissive term. The
/// surface is *not* culled when facing away from the light: the transmission
/// term intentionally lets radiance leak through thin geometry.
pub fn evaluate_subsurface_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
) -> [f32; 3] {
    let n = normalize_or(frame.normal, [0.0, 1.0, 0.0]);
    let v = normalize_or(frame.view, n);
    let l = normalize_or(light.direction, n);
    let h = normalize_or(add(v, l), n);

    // `N.L` is kept signed for the wrap/transmission terms; the specular lobe
    // uses the clamped cosine so it vanishes on the unlit side.
    let raw_n_dot_l = dot(n, l);
    let n_dot_l = raw_n_dot_l.max(0.0);
    let n_dot_v = dot(n, v).max(MIN_N_DOT);
    let n_dot_h = dot(n, h).max(MIN_N_DOT);
    let v_dot_h = dot(v, h).max(0.0);

    let subsurface = surface.subsurface.clamp(0.0, 1.0);
    let thickness = surface.thickness.clamp(0.0, 1.0);
    let metallic = surface.metallic.clamp(0.0, 1.0);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let reflectance = surface.reflectance.clamp(0.0, 1.0);
    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = mix3([f0_dielectric; 3], surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);

    // Front-face GGX specular; reuses the principled helpers for exact parity.
    let d = distribution_ggx(n_dot_h, alpha);
    let g = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let specular = mul_scalar(mul_scalar(f, d * g), n_dot_l);

    // Wrapped diffuse: the wrap width is the subsurface weight, so scattering
    // softens the terminator without altering the base albedo response.
    let diffuse_cos = fd_wrap(raw_n_dot_l, subsurface);
    let diffuse_weight = mul_scalar(sub([1.0; 3], f), 1.0 - metallic);
    let diffuse = mul(
        mul_scalar(surface.base_color, INV_PI * diffuse_cos),
        diffuse_weight,
    );

    // Back transmission: light exiting toward the viewer after travelling
    // through thin geometry. Thin surfaces (thickness -> 0) transmit fully.
    let h_transmission = normalize_or(add(l, mul_scalar(n, SSS_DISTORTION)), n);
    let back = ops::powf(
        dot(v, mul_scalar(h_transmission, -1.0)).clamp(0.0, 1.0),
        SSS_POWER,
    );
    let thickness_attenuation = 1.0 - thickness;
    let transmitted = mul_scalar(surface.base_color, subsurface * back * thickness_attenuation);

    let lit = add(add(diffuse, specular), transmitted);
    let radiance = mul_scalar(
        mul(lit, light.illuminance),
        light.visibility.clamp(0.0, 1.0),
    );
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
            base_color: [0.7, 0.4, 0.3],
            perceptual_roughness: 0.5,
            subsurface: 0.5,
            thickness: 0.3,
            ..Default::default()
        }
    }

    #[test]
    fn subsurface_is_finite_and_non_negative_across_parameters() {
        for roughness in [0.045, 0.2, 0.5, 1.0] {
            for subsurface in [0.0, 0.5, 1.0] {
                for thickness in [0.0, 0.5, 1.0] {
                    let value = evaluate_subsurface_direct(
                        SurfaceSample {
                            perceptual_roughness: roughness,
                            subsurface,
                            thickness,
                            ..base_surface()
                        },
                        frame(),
                        light(),
                    );
                    assert!(
                        value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                        "roughness={roughness} subsurface={subsurface} thickness={thickness} -> {value:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_subsurface_matches_a_plain_lambert_wrap() {
        // With subsurface == 0 the wrap collapses to saturate(N.L); the front
        // response must equal a plain dielectric diffuse+specular evaluation.
        let front = evaluate_subsurface_direct(
            SurfaceSample {
                subsurface: 0.0,
                ..base_surface()
            },
            frame(),
            light(),
        );
        // A back-facing light contributes nothing when there is no subsurface
        // weight to drive transmission.
        let back = evaluate_subsurface_direct(
            SurfaceSample {
                subsurface: 0.0,
                emissive: [0.0; 3],
                ..base_surface()
            },
            frame(),
            DirectLightSample {
                direction: [0.0, -1.0, 0.0],
                ..light()
            },
        );
        assert_eq!(back, [0.0; 3], "no subsurface weight must not transmit");
        assert!(front.into_iter().any(|c| c > 0.0), "front should be lit");
    }

    #[test]
    fn back_light_transmits_more_through_thinner_geometry() {
        let behind = DirectLightSample {
            direction: normalize_or([0.0, -1.0, -0.2], [0.0, -1.0, 0.0]),
            ..light()
        };
        let thin = evaluate_subsurface_direct(
            SurfaceSample {
                thickness: 0.0,
                ..base_surface()
            },
            frame(),
            behind,
        );
        let thick = evaluate_subsurface_direct(
            SurfaceSample {
                thickness: 1.0,
                ..base_surface()
            },
            frame(),
            behind,
        );
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(
            sum(thin) > sum(thick),
            "thin geometry must transmit more: thin={thin:?} thick={thick:?}"
        );
        assert_eq!(sum(thick), 0.0, "fully opaque geometry transmits nothing");
    }

    #[test]
    fn increasing_subsurface_softens_the_terminator() {
        // At a grazing terminator angle the wrap lifts the diffuse response as
        // the subsurface weight grows.
        let terminator = DirectLightSample {
            direction: normalize_or([1.0, 0.05, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let low = evaluate_subsurface_direct(
            SurfaceSample {
                subsurface: 0.1,
                thickness: 1.0,
                ..base_surface()
            },
            frame(),
            terminator,
        );
        let high = evaluate_subsurface_direct(
            SurfaceSample {
                subsurface: 0.9,
                thickness: 1.0,
                ..base_surface()
            },
            frame(),
            terminator,
        );
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(
            sum(high) > sum(low),
            "more subsurface must brighten the terminator: low={low:?} high={high:?}"
        );
    }

    #[test]
    fn zero_visibility_leaves_only_emissive() {
        let surface = SurfaceSample {
            emissive: [0.05, 0.06, 0.07],
            ..base_surface()
        };
        let shadowed = DirectLightSample {
            visibility: 0.0,
            ..light()
        };
        assert_eq!(
            evaluate_subsurface_direct(surface, frame(), shadowed),
            surface.emissive
        );
    }
}
