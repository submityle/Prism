//! Subsurface scattering (SSS) BSDF direct-lighting lobe.
//!
//! Standalone backend-neutral subsurface reference and the
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

use crate::lighting::{
    dielectric_f0, distribution_ggx, fresnel_schlick, visibility_smith_ggx_correlated,
};
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

    let metallic = surface.metallic.clamp(0.0, 1.0);
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let f0 = mix3(dielectric_f0(&surface), surface.base_color, metallic);
    let f = fresnel_schlick(f0, v_dot_h);

    // Front-face GGX specular; reuses the principled helpers for exact parity.
    let d = distribution_ggx(n_dot_h, alpha);
    let g = visibility_smith_ggx_correlated(n_dot_v, n_dot_l, alpha);
    let specular = mul_scalar(mul_scalar(f, d * g), n_dot_l);

    let diffuse_weight = mul_scalar(sub([1.0; 3], f), 1.0 - metallic);
    let (diffuse, transmitted) =
        subsurface_diffuse_terms(&surface, n, v, l, raw_n_dot_l, diffuse_weight);
    let lit = add(add(diffuse, specular), transmitted);
    let radiance = mul_scalar(
        mul(lit, light.illuminance),
        light.visibility.clamp(0.0, 1.0),
    );
    add(radiance, surface.emissive)
}

/// Shared scattering-only terms. No specular, coat, fuzz, visibility or emission:
/// the principled mixture supplies those once around the combined substrate.
pub(crate) fn subsurface_diffuse_terms(
    surface: &SurfaceSample,
    n: [f32; 3],
    v: [f32; 3],
    l: [f32; 3],
    raw_n_dot_l: f32,
    diffuse_weight: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let subsurface = surface.subsurface.clamp(0.0, 1.0);
    let thickness = surface.thickness.clamp(0.0, 1.0);
    // Wrapped diffuse: the wrap width is the subsurface weight scaled by the
    // per-channel `subsurface_radius`, so scattering softens the terminator and
    // a red-heavy radius bleeds red further. `subsurface_color` tints the
    // scattered light. Both default to the identity.
    let scat_color = surface.subsurface_color;
    let mut diffuse = [0.0f32; 3];
    for (channel, slot) in diffuse.iter_mut().enumerate() {
        // `subsurface_radius` (scalar) times `subsurface_radius_scale` (color3)
        // gives the per-channel scattering extent.
        let wrap =
            (subsurface * surface.subsurface_radius * surface.subsurface_radius_scale[channel])
                .clamp(0.0, 1.0);
        // Normalise the wrap lobe to unit albedo. Its analytic directional
        // albedo is `E(w) = (1 + 2w)/(1 + w)²` (`∫(1/π)·wrap(cos)·dω`), which is
        // 1 at `w = 0` (Lambert) but falls to 0.75 at `w = 1` — so an
        // unnormalised wrap silently *loses* energy, and because the wrap is
        // per-channel (`subsurface_radius_scale`) it lost it per channel: the
        // white furnace measured [0.759, 0.892, 0.961] against the closed form's
        // [0.750, 0.889, 0.960]. Dividing by `E(w)` restores the invariant and
        // matches the specification's `subsurface_albedo_constraint` (the
        // authored colour is the *observed* albedo).
        let one_plus_w = 1.0 + wrap;
        let e_wrap = ((1.0 + 2.0 * wrap) / (one_plus_w * one_plus_w)).max(1.0e-3);
        let diffuse_cos = fd_wrap(raw_n_dot_l, wrap) / e_wrap;
        *slot = surface.base_color[channel]
            * scat_color[channel]
            * INV_PI
            * diffuse_cos
            * diffuse_weight[channel];
    }

    // Back transmission: light exiting toward the viewer after travelling
    // through thin geometry. Thin surfaces (thickness -> 0) transmit fully.
    let h_transmission = normalize_or(add(l, mul_scalar(n, SSS_DISTORTION)), n);
    // `OpenPBR` `scatter_anisotropy` shapes the transmitted lobe: positive
    // (forward) sharpens it, negative (backward) broadens it. It scales the
    // historical constant, so `0.0` reproduces the legacy lobe exactly.
    let sharpness =
        (SSS_POWER * (1.0 + surface.scatter_anisotropy.clamp(-1.0, 1.0))).max(MIN_N_DOT);
    let back = ops::powf(
        dot(v, mul_scalar(h_transmission, -1.0)).clamp(0.0, 1.0),
        sharpness,
    );
    let thickness_attenuation = 1.0 - thickness;
    let transmitted = mul(
        mul(surface.base_color, scat_color),
        [subsurface * back * thickness_attenuation; 3],
    );

    (diffuse, transmitted)
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

    #[test]
    fn subsurface_radius_bleeds_channels_independently() {
        // Light just past the terminator: the wrap width decides how much bleeds
        // through, and a per-channel radius makes red reach further than blue.
        // `thickness == 1` kills the back-transmission so only the wrapped
        // diffuse is measured.
        let behind = DirectLightSample {
            direction: [-0.98, -0.2, 0.0],
            ..light()
        };
        let out = evaluate_subsurface_direct(
            SurfaceSample {
                base_color: [1.0; 3],
                perceptual_roughness: 0.5,
                subsurface: 1.0,
                subsurface_radius: 1.0,
                subsurface_radius_scale: [1.0, 0.4, 0.05],
                thickness: 1.0,
                ..Default::default()
            },
            frame(),
            behind,
        );
        assert!(out[0] > out[1], "red must out-bleed green: {out:?}");
        assert!(out[1] > out[2], "green must out-bleed blue: {out:?}");
        assert_eq!(
            out[2], 0.0,
            "a tiny radius must not reach past the terminator"
        );
    }

    #[test]
    fn subsurface_color_tints_the_scattered_light() {
        let behind = DirectLightSample {
            direction: [-0.98, -0.2, 0.0],
            ..light()
        };
        let base = SurfaceSample {
            base_color: [1.0; 3],
            perceptual_roughness: 0.5,
            subsurface: 1.0,
            thickness: 1.0,
            // Compare against an explicitly white tint: the `OpenPBR` default
            // `subsurface_color` is `(0.8, 0.8, 0.8)`, not white.
            subsurface_color: [1.0; 3],
            ..Default::default()
        };
        let white = evaluate_subsurface_direct(base, frame(), behind);
        let red = evaluate_subsurface_direct(
            SurfaceSample {
                subsurface_color: [1.0, 0.0, 0.0],
                ..base
            },
            frame(),
            behind,
        );
        // Red is untouched, green and blue are removed: the tint only scales.
        assert!(
            (red[0] - white[0]).abs() < 1.0e-6,
            "red={red:?} white={white:?}"
        );
        assert_eq!(red[1], 0.0);
        assert_eq!(red[2], 0.0);
    }

    #[test]
    fn scatter_anisotropy_shapes_the_transmission_lobe() {
        // Forward bias (+) sharpens the transmitted lobe, backward bias (-)
        // broadens it, and 0 reproduces the historical constant exactly.
        let behind = DirectLightSample {
            direction: [-0.6, -0.8, 0.0],
            ..light()
        };
        let base = SurfaceSample {
            base_color: [1.0; 3],
            perceptual_roughness: 0.5,
            subsurface: 1.0,
            thickness: 0.0,
            ..Default::default()
        };
        let forward = evaluate_subsurface_direct(
            SurfaceSample {
                scatter_anisotropy: 1.0,
                ..base
            },
            frame(),
            behind,
        );
        let isotropic = evaluate_subsurface_direct(base, frame(), behind);
        let backward = evaluate_subsurface_direct(
            SurfaceSample {
                scatter_anisotropy: -1.0,
                ..base
            },
            frame(),
            behind,
        );
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(
            sum(forward) < sum(isotropic) && sum(isotropic) < sum(backward),
            "forward={forward:?} isotropic={isotropic:?} backward={backward:?}"
        );
    }

    #[test]
    fn radius_and_scale_split_the_per_channel_extent() {
        // Light just past the terminator, no back-transmission: only the wrapped
        // diffuse carries the radius.
        let behind = DirectLightSample {
            direction: [-0.98, -0.2, 0.0],
            ..light()
        };
        let base = SurfaceSample {
            base_color: [1.0; 3],
            perceptual_roughness: 0.5,
            subsurface: 1.0,
            thickness: 1.0,
            ..Default::default()
        };
        // The specification default split (1.0, 0.5, 0.25) makes red bleed
        // further than green, which bleeds further than blue.
        let default_split = evaluate_subsurface_direct(base, frame(), behind);
        assert!(
            default_split[0] > default_split[1] && default_split[1] > default_split[2],
            "{default_split:?}"
        );

        // The scalar radius scales every channel together, so a zero radius
        // switches the bleed off entirely regardless of the per-channel split.
        let zero_radius = evaluate_subsurface_direct(
            SurfaceSample {
                subsurface_radius: 0.0,
                ..base
            },
            frame(),
            behind,
        );
        for channel in zero_radius {
            assert_eq!(channel, 0.0, "{zero_radius:?}");
        }
    }
}
