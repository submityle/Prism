//! Hair (strand) BSDF direct-lighting lobe.
//!
//! Backend-neutral golden reference for the `Hair` shading class and the
//! numerical twin of `hair.wesl` (`hair_direct`). The closure implements a
//! physically based Marschner R/TT/TRT strand BSDF (Karis' energy-conserving
//! real-time form from "Physically Based Hair Shading in Unreal") plus a
//! Zinke-style dual-scattering multiple-scattering fill:
//!
//! * **R** (reflection) is a sharp white dielectric glint off the outer
//!   cuticle - the classic primary hair highlight.
//! * **TT** (transmission-transmission) is forward-transmitted light tinted by
//!   absorption through the fibre - the bright rim on back-lit hair.
//! * **TRT** (transmission-reflection-transmission) is the coloured secondary
//!   highlight from light that reflects off the far cuticle wall.
//! * **dual-scattering** approximates global multiple scattering (Zinke), the
//!   translucent forward/backward fill that keeps light-coloured hair from
//!   reading as dead black.
//!
//! Each specular lobe is a longitudinal Gaussian `Mp` times an azimuthal term
//! `Np` and a Fresnel/absorption weight, with per-lobe roughness widths and
//! cuticle-tilt shifts. The strand direction is the shading frame `tangent`;
//! `perceptual_roughness` drives the longitudinal widths (glossier => tighter
//! highlights). Emissive is folded in exactly once, matching
//! `evaluate_principled_direct` so the resolve integrator can accumulate over
//! many lights without multiplying emission.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::vecmath::{add, dot, mul, mul_scalar, normalize_or, sub};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Index of refraction of the hair fibre (keratin), driving Fresnel `F0`.
const HAIR_IOR: f32 = 1.55;
/// Cuticle tilt (radians) separating the R/TT/TRT lobes longitudinally; the
/// three lobes are shifted by `-2a`, `a` and `4a` about it (Marschner tilt).
const CUTICLE_SHIFT: f32 = 0.035;
/// Scalar weight of the specular (R) reflection lobe.
const SPECULAR: f32 = 0.5;
/// Backward-scatter boost for light coming from behind the fibre.
const BACKLIT: f32 = 1.0;
/// Weight of the dual-scattering multiple-scattering fill term.
const MS_WEIGHT: f32 = 0.25;

/// Linear interpolation `a + (b - a) * t`, matching the GPU `mix` builtin.
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Normalized longitudinal Gaussian `Mp` of width `beta` evaluated at `x`
/// (the sin-theta half-angle offset). Mirrors `hair_gaussian`.
fn hair_gaussian(beta: f32, x: f32) -> f32 {
    ops::exp(-0.5 * x * x / (beta * beta)) / (ops::sqrt(2.0 * PI) * beta)
}

/// Schlick Fresnel for the fibre dielectric interface. Mirrors `hair_fresnel`.
fn hair_fresnel(cos_theta: f32) -> f32 {
    let r = (1.0 - HAIR_IOR) / (1.0 + HAIR_IOR);
    let f0 = r * r;
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    f0 + (1.0 - f0) * ops::powf(m, 5.0)
}

/// Per-channel `pow(base, exponent)` for the absorption tint. Mirrors the
/// component-wise `pow(base_color, e)` in the WESL twin.
fn pow3(base: [f32; 3], exponent: f32) -> [f32; 3] {
    [
        ops::powf(base[0], exponent),
        ops::powf(base[1], exponent),
        ops::powf(base[2], exponent),
    ]
}

/// Evaluates the hair BSDF for a single analytic light.
///
/// Returns linear radiance plus the surface emissive term. When the light is
/// fully occluded the emissive term is returned on its own, matching the other
/// lobes; unlike opaque lobes the dual-scattering fill survives backward
/// grazing light so the strand keeps its translucent rim.
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
    let base = surface.base_color;

    // Longitudinal sines about the strand tangent, and the raw V.L.
    let sin_theta_l = dot(t, l).clamp(-1.0, 1.0);
    let sin_theta_v = dot(t, v).clamp(-1.0, 1.0);
    let v_dot_l = dot(v, l);

    // Half difference angle drives Fresnel and the transmission widths.
    let cos_theta_d = ops::cos(0.5 * (ops::asin(sin_theta_v) - ops::asin(sin_theta_l)).abs());

    // Azimuthal cosine from the tangent-plane projections of L and V.
    let lp = sub(l, mul_scalar(t, sin_theta_l));
    let vp = sub(v, mul_scalar(t, sin_theta_v));
    let cos_phi = dot(lp, vp) / (dot(lp, lp) * dot(vp, vp) + 1.0e-4).sqrt();
    let cos_half_phi = (0.5 + 0.5 * cos_phi).clamp(0.0, 1.0).sqrt();

    // Modified (Bravais) refractive index for the projected azimuth.
    let n_prime = 1.19 / cos_theta_d + 0.36 * cos_theta_d;

    // Per-lobe longitudinal widths (roughness^2 scaled) and cuticle shifts.
    let roughness = surface.perceptual_roughness.clamp(MIN_ROUGHNESS, 1.0);
    let r2 = roughness * roughness;
    let beta = [r2, r2 * 0.5, r2 * 2.0];
    let alpha = [-2.0 * CUTICLE_SHIFT, CUTICLE_SHIFT, 4.0 * CUTICLE_SHIFT];
    let long_arg = sin_theta_l + sin_theta_v;
    let mp = [
        hair_gaussian(beta[0], long_arg - alpha[0]),
        hair_gaussian(beta[1], long_arg - alpha[1]),
        hair_gaussian(beta[2], long_arg - alpha[2]),
    ];

    // R: white primary cuticle reflection (base-color independent).
    let r_lobe = {
        let np = 0.25 * cos_half_phi;
        let fp = hair_fresnel((0.5 + 0.5 * v_dot_l).clamp(0.0, 1.0).sqrt());
        let backlit = mix(1.0, BACKLIT, (-v_dot_l).clamp(0.0, 1.0));
        let s = mp[0] * np * fp * (SPECULAR * 2.0) * backlit;
        [s; 3]
    };

    // TT: forward transmission tinted by absorption through the fibre.
    let tt_lobe = {
        let a = 1.0 / n_prime;
        let h = cos_half_phi * (1.0 + a * (0.6 - 0.8 * cos_phi));
        let f = hair_fresnel(cos_theta_d * (1.0 - h * h).clamp(0.0, 1.0).sqrt());
        let fp = (1.0 - f) * (1.0 - f);
        let ha = h * a;
        let tp = pow3(
            base,
            0.5 * (1.0 - ha * ha).clamp(0.0, 1.0).sqrt() / cos_theta_d,
        );
        let np = ops::exp(-3.65 * cos_phi - 3.98);
        mul_scalar(tp, mp[1] * np * fp * BACKLIT)
    };

    // TRT: coloured secondary highlight (double transmission + reflection).
    let trt_lobe = {
        let f = hair_fresnel(cos_theta_d * 0.5);
        let fp = (1.0 - f) * (1.0 - f) * f;
        let tp = pow3(base, 0.8 / cos_theta_d);
        let np = ops::exp(17.0 * cos_phi - 16.78);
        mul_scalar(tp, mp[2] * np * fp)
    };

    let spec = add(add(r_lobe, tt_lobe), trt_lobe);

    // Dual-scattering multiple-scattering fill (Zinke): a translucent
    // forward/backward wrap so light-coloured hair stays luminous and keeps a
    // rim even when the light grazes from behind.
    let wrap = (dot(n, l) * 0.5 + 0.5).clamp(0.0, 1.0);
    let ms = mul_scalar(base, INV_PI * wrap * MS_WEIGHT);

    let radiance = mul_scalar(mul(add(spec, ms), light.illuminance), visibility);
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
                        ShadingFrame { tangent, ..frame() },
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
        // With a black base color the TT/TRT tint and the dual-scattering
        // fill vanish, so any positive radiance must be the white primary R
        // reflection - proving the lobe is not the principled fallback's
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
        assert!(
            sum(value) > 0.0,
            "expected a pure specular glint, got {value:?}"
        );
    }

    #[test]
    fn glossier_hair_narrows_the_off_peak_highlight() {
        // At an off-peak angle a tighter (glossier) longitudinal lobe must
        // fall off faster than a broad (rough) one. Use a black fibre so only
        // the white primary R reflection contributes.
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
        assert_ne!(
            along_x, along_z,
            "hair must be anisotropic about the strand"
        );
    }

    #[test]
    fn backward_grazing_light_keeps_a_translucent_rim() {
        // A light coming from slightly behind still lifts the fibre via the
        // dual-scattering wrap - hair does not hard-cut at `N.L = 0`.
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
