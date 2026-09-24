//! Hair (strand) BSDF direct-lighting lobe.
//!
//! Backend-neutral golden reference for the `Hair` shading class and the
//! byte-for-byte numerical twin of `hair.wesl` (`hair_direct`).  The closure
//! follows the real-time strand model that Unreal and countless production
//! shaders build on - Kajiya-Kay for the translucent diffuse term and
//! Scheuermann's *shifted strand specular* for the anisotropic highlights:
//!
//! * **Diffuse** is a Kajiya-Kay `sin(T, L)` lobe modulated by a half-Lambert
//!   *wrap* so a fibre still catches light that grazes from behind (hair is
//!   forward/back-scattering, not a hard `N.L` cutoff).  It is tinted by the
//!   base color.
//! * **Specular** is the sum of two shifted strand highlights.  The *primary*
//!   (R) highlight is a sharp, white dielectric glint shifted toward the root;
//!   the *secondary* (TRT) highlight is broader, shifted the other way and
//!   tinted by the fibre color because that energy has transmitted through the
//!   strand.  Each lobe uses `pow(sin(T', H), exponent)` with a `smoothstep`
//!   backward-direction attenuation, exactly the ATI/Scheuermann formulation.
//!
//! The strand direction is the shading frame's `tangent`; roughness maps to the
//! two highlight exponents (glossier hair => tighter highlights).  Emissive is
//! folded in exactly once, matching `evaluate_principled_direct` so the resolve
//! integrator can accumulate over many lights without multiplying emission.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::vecmath::{add, dot, mul, mul_scalar, normalize_or};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Tangent shift (along the normal) of the sharp primary (R) highlight; a
/// small negative shift moves it toward the strand root, matching Scheuermann.
const PRIMARY_SHIFT: f32 = -0.08;
/// Tangent shift of the broad secondary (TRT) highlight, shifted the opposite
/// way so the two glints separate into the characteristic hair double band.
const SECONDARY_SHIFT: f32 = 0.10;
/// Primary highlight `pow` exponent at maximum roughness (broadest glint).
const PRIMARY_EXP_MIN: f32 = 16.0;
/// Primary highlight `pow` exponent at zero roughness (tightest glint).
const PRIMARY_EXP_MAX: f32 = 160.0;
/// Secondary highlight exponent at maximum roughness.
const SECONDARY_EXP_MIN: f32 = 8.0;
/// Secondary highlight exponent at zero roughness.
const SECONDARY_EXP_MAX: f32 = 80.0;
/// Scalar weight of the (white) primary highlight.
const PRIMARY_WEIGHT: f32 = 0.20;
/// Scalar weight of the (fibre-tinted) secondary highlight.
const SECONDARY_WEIGHT: f32 = 0.35;

/// Linear interpolation `a + (b - a) * t`, matching the GPU `mix` builtin.
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite `smoothstep(edge0, edge1, x)`, matching the GPU `smoothstep`
/// builtin used to fade the strand highlight in the backward hemisphere.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Shifts the strand tangent along the surface normal, re-normalizing, so the
/// primary/secondary highlights peak at slightly different angles. Mirrors
/// `hair_shift_tangent`.
fn shift_tangent(tangent: [f32; 3], normal: [f32; 3], shift: f32) -> [f32; 3] {
    normalize_or(add(tangent, mul_scalar(normal, shift)), tangent)
}

/// Scheuermann shifted strand specular: `smoothstep`-attenuated
/// `pow(sin(T', H), exponent)`. Mirrors `hair_strand_specular`.
fn strand_specular(tangent: [f32; 3], half: [f32; 3], exponent: f32) -> f32 {
    let dot_th = dot(tangent, half);
    let sin_th = (1.0 - dot_th * dot_th).max(0.0).sqrt();
    let directional = smoothstep(-1.0, 0.0, dot_th);
    directional * ops::powf(sin_th, exponent)
}

/// Evaluates the hair BSDF for a single analytic light.
///
/// Returns linear radiance plus the surface emissive term. When the light is
/// fully occluded the emissive term is returned on its own, matching the other
/// lobes; unlike opaque lobes the diffuse survives backward grazing light so
/// the strand keeps its translucent rim.
pub fn evaluate_hair_direct(
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
    let t = normalize_or(frame.tangent, [1.0, 0.0, 0.0]);
    let h = normalize_or(add(v, l), n);

    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let gloss = 1.0 - roughness;
    let primary_exp = mix(PRIMARY_EXP_MIN, PRIMARY_EXP_MAX, gloss);
    let secondary_exp = mix(SECONDARY_EXP_MIN, SECONDARY_EXP_MAX, gloss);

    // Kajiya-Kay translucent diffuse: `sin(T, L)` softened by a half-Lambert
    // wrap so light grazing from behind still lifts the fibre.
    let dot_nl = dot(n, l);
    let wrap = (dot_nl * 0.5 + 0.5).clamp(0.0, 1.0);
    let dot_tl = dot(t, l);
    let sin_tl = (1.0 - dot_tl * dot_tl).max(0.0).sqrt();
    let diffuse = mul_scalar(surface.base_color, INV_PI * wrap * sin_tl);

    // Two shifted strand highlights: sharp white primary + broad tinted
    // secondary (transmitted through the fibre, so it carries the base color).
    let t_primary = shift_tangent(t, n, PRIMARY_SHIFT);
    let t_secondary = shift_tangent(t, n, SECONDARY_SHIFT);
    let spec_primary = strand_specular(t_primary, h, primary_exp);
    let spec_secondary = strand_specular(t_secondary, h, secondary_exp);
    let primary = [spec_primary * PRIMARY_WEIGHT; 3];
    let secondary = mul_scalar(surface.base_color, spec_secondary * SECONDARY_WEIGHT);
    let specular = add(primary, secondary);

    let radiance = mul_scalar(
        mul(add(diffuse, specular), light.illuminance),
        visibility,
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
            base_color: [0.4, 0.25, 0.1],
            perceptual_roughness: 0.4,
            ..Default::default()
        }
    }

    fn sum(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn hair_is_finite_and_non_negative_across_parameters() {
        let tangents = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.0, 0.7]];
        let dirs = [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, -0.5, 0.86],
            [-0.7, 0.2, 0.68],
        ];
        for roughness in [0.045, 0.3, 0.7, 1.0] {
            for tangent in tangents {
                for dir in dirs {
                    let value = evaluate_hair_direct(
                        SurfaceSample {
                            perceptual_roughness: roughness,
                            ..base_surface()
                        },
                        ShadingFrame {
                            tangent,
                            ..frame()
                        },
                        DirectLightSample {
                            direction: dir,
                            ..light()
                        },
                    );
                    assert!(
                        value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                        "roughness={roughness} tangent={tangent:?} dir={dir:?} -> {value:?}"
                    );
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
            evaluate_hair_direct(surface, frame(), shadowed),
            surface.emissive
        );
    }

    #[test]
    fn black_fibre_still_shows_the_white_primary_highlight() {
        // With a black base color the diffuse and the (base-tinted) secondary
        // highlight vanish, so any positive radiance must be the white primary
        // strand specular - proving the lobe is not the principled fallback's
        // base-color-driven response.
        let surface = SurfaceSample {
            base_color: [0.0, 0.0, 0.0],
            perceptual_roughness: 0.2,
            ..base_surface()
        };
        // v straight up, light tilted so the half-vector is perpendicular to
        // the strand => strong `sin(T, H)`.
        let tilted = DirectLightSample {
            direction: normalize_or([0.0, 0.6, 0.8], [0.0, 1.0, 0.0]),
            ..light()
        };
        let value = evaluate_hair_direct(surface, frame(), tilted);
        assert!(sum(value) > 0.0, "expected a pure specular glint, got {value:?}");
    }

    #[test]
    fn glossier_hair_narrows_the_off_peak_highlight() {
        // At an off-peak angle a tighter (glossier) highlight must fall off
        // faster than a broad (rough) one. Use a black fibre so only the white
        // primary lobe contributes.
        let off_peak = DirectLightSample {
            direction: normalize_or([0.6, 0.8, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let glossy = evaluate_hair_direct(
            SurfaceSample {
                base_color: [0.0; 3],
                perceptual_roughness: 0.045,
                ..base_surface()
            },
            frame(),
            off_peak,
        );
        let rough = evaluate_hair_direct(
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
            "glossy off-peak {glossy:?} should be dimmer than rough {rough:?}"
        );
    }

    #[test]
    fn rotating_the_strand_changes_the_response() {
        let along_x = evaluate_hair_direct(
            base_surface(),
            ShadingFrame {
                tangent: [1.0, 0.0, 0.0],
                ..frame()
            },
            DirectLightSample {
                direction: normalize_or([0.7, 0.7, 0.0], [0.0, 1.0, 0.0]),
                ..light()
            },
        );
        let along_z = evaluate_hair_direct(
            base_surface(),
            ShadingFrame {
                tangent: [0.0, 0.0, 1.0],
                ..frame()
            },
            DirectLightSample {
                direction: normalize_or([0.7, 0.7, 0.0], [0.0, 1.0, 0.0]),
                ..light()
            },
        );
        assert_ne!(along_x, along_z, "hair must be anisotropic about the strand");
    }

    #[test]
    fn backward_grazing_light_keeps_a_translucent_rim() {
        // A light coming from slightly behind still lifts the fibre via the
        // half-Lambert wrap - hair does not hard-cut at `N.L = 0`.
        let surface = base_surface();
        let behind = DirectLightSample {
            direction: normalize_or([0.9, -0.2, 0.0], [0.0, 1.0, 0.0]),
            ..light()
        };
        let value = evaluate_hair_direct(surface, frame(), behind);
        assert!(
            sum(value) > 0.0,
            "translucent rim expected for backward light, got {value:?}"
        );
    }
}
