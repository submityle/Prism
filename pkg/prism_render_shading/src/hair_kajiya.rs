//! Kajiya-Kay anisotropic strand highlight (cheap real-time fallback tier).
//!
//! Cheapest member of the hair shading family and the fallback rung of the
//! degradation matrix (§8): where [`crate::evaluate_hair_direct`] (Marschner),
//! [`crate::evaluate_hair_chiang_direct`] (Chiang near-field) and
//! [`crate::evaluate_hair_fiber_direct`] (Yan fibre-level) model the physical
//! R/TT/TRT lobes and medulla scattering, this front end is the classic
//! Kajiya-Kay strand model from Kajiya & Kay "Rendering Fur with Three
//! Dimensional Textures" (1989): a single anisotropic diffuse term plus one
//! shifted anisotropic specular term about the strand tangent. No absorption,
//! no transmission, no medulla - just the two closed-form trig terms that make
//! it the affordable tier for weak platforms and distant strands.
//!
//! * **Diffuse** - `sin(T, L)`, the classic cylinder diffuse that peaks when the
//!   light is perpendicular to the strand tangent, tinted by the base color.
//! * **Specular** - `(T' . L)(T' . V) + sin(T', L) sin(T', V)` raised to the
//!   shininess exponent, evaluated about a tangent `T'` optionally tilted toward
//!   the surface normal by `specular_shift` (the cheap stand-in for the cuticle
//!   tilt of the physical lobes), tinted between white and the base color by
//!   `specular_tint`.
//!
//! The result is deterministic, array-in / array-out and is the numerical twin
//! of `hair_kajiya.wesl` (`hair_kajiya_direct`). Emissive is folded in once.

use bevy_math::ops;

use crate::vecmath::{add, dot, mix3, mul, mul_scalar, normalize_or};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Shininess floor so the specular power stays well-defined.
const MIN_SPECULAR_EXPONENT: f32 = 1.0;

/// Tunable controls for the Kajiya-Kay fallback front end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairKajiyaParams {
    /// Diffuse strength `Kd` scaling the `sin(T, L)` cylinder diffuse.
    pub diffuse_strength: f32,
    /// Specular strength `Ks` scaling the anisotropic highlight.
    pub specular_strength: f32,
    /// Specular shininess exponent (`>= 1`); larger is a tighter highlight.
    pub specular_exponent: f32,
    /// Tangent tilt toward the surface normal for the specular lobe; the cheap
    /// stand-in for the cuticle tilt of the physical models. `0` leaves the
    /// highlight centred on the raw strand tangent.
    pub specular_shift: f32,
    /// Specular tint in `[0, 1]`: `0` is a white dielectric glint, `1` tints the
    /// highlight fully by the base color.
    pub specular_tint: f32,
}

impl Default for HairKajiyaParams {
    /// A neutral strand: mostly diffuse with a modest white shifted highlight.
    fn default() -> Self {
        Self {
            diffuse_strength: 0.75,
            specular_strength: 0.2,
            specular_exponent: 40.0,
            specular_shift: 0.0,
            specular_tint: 0.0,
        }
    }
}

/// `sin` of the angle between two unit-ish vectors from their clamped cosine.
fn sin_from_cos(cos_theta: f32) -> f32 {
    (1.0 - cos_theta * cos_theta).max(0.0).sqrt()
}

/// Evaluates the Kajiya-Kay strand highlight for a single analytic light.
///
/// Returns linear radiance plus the surface emissive term. When the light is
/// fully occluded only the emissive term is returned, matching the other lobes.
#[must_use]
pub fn evaluate_hair_kajiya_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: HairKajiyaParams,
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

    // Diffuse: `sin(T, L)` about the raw strand tangent, tinted by the base.
    let t_dot_l = dot(t, l).clamp(-1.0, 1.0);
    let diffuse_term = params.diffuse_strength.max(0.0) * sin_from_cos(t_dot_l);
    let diffuse = mul_scalar(base, diffuse_term);

    // Specular: shift the tangent toward the normal (cheap cuticle-tilt stand-in),
    // then evaluate the Kajiya-Kay anisotropic highlight about the shifted axis.
    let t_shifted = normalize_or(add(t, mul_scalar(n, params.specular_shift)), t);
    let ts_dot_l = dot(t_shifted, l).clamp(-1.0, 1.0);
    let ts_dot_v = dot(t_shifted, v).clamp(-1.0, 1.0);
    let spec_cos =
        (ts_dot_l * ts_dot_v + sin_from_cos(ts_dot_l) * sin_from_cos(ts_dot_v)).clamp(0.0, 1.0);
    let exponent = params.specular_exponent.max(MIN_SPECULAR_EXPONENT);
    let spec_term = params.specular_strength.max(0.0) * ops::powf(spec_cos, exponent);
    let spec_tint = params.specular_tint.clamp(0.0, 1.0);
    let spec = mul_scalar(mix3([1.0, 1.0, 1.0], base, spec_tint), spec_term);

    let scatter = add(diffuse, spec);
    let radiance = mul_scalar(mul(scatter, light.illuminance), visibility);
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
            base_color: [0.5, 0.4, 0.3],
            perceptual_roughness: 0.4,
            ..Default::default()
        }
    }

    fn sum(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn kajiya_is_finite_and_non_negative_across_parameters() {
        let tangents = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.0, 0.7]];
        let dirs = [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, -0.5, 0.86],
            [-0.7, 0.2, 0.68],
        ];
        for exponent in [1.0, 20.0, 80.0, 200.0] {
            for shift in [-0.2, 0.0, 0.3] {
                for tint in [0.0, 0.5, 1.0] {
                    for tangent in tangents {
                        for dir in dirs {
                            let value = evaluate_hair_kajiya_direct(
                                base_surface(),
                                ShadingFrame { tangent, ..frame() },
                                DirectLightSample {
                                    direction: dir,
                                    ..light()
                                },
                                HairKajiyaParams {
                                    specular_exponent: exponent,
                                    specular_shift: shift,
                                    specular_tint: tint,
                                    ..Default::default()
                                },
                            );
                            assert!(
                                value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                                "exponent={exponent} shift={shift} tint={tint} \
                                 tangent={tangent:?} dir={dir:?} -> {value:?}"
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
            evaluate_hair_kajiya_direct(surface, frame(), shadowed, HairKajiyaParams::default()),
            surface.emissive
        );
    }

    #[test]
    fn diffuse_peaks_perpendicular_to_tangent() {
        // Kajiya-Kay diffuse follows `sin(T, L)`: a light perpendicular to the
        // strand tangent must out-light one nearly parallel to it.
        let diffuse_only = HairKajiyaParams {
            specular_strength: 0.0,
            ..Default::default()
        };
        // Tangent along +x. Perpendicular light (+y) => sin(T,L)=1.
        let perpendicular = evaluate_hair_kajiya_direct(
            base_surface(),
            frame(),
            DirectLightSample {
                direction: [0.0, 1.0, 0.0],
                ..light()
            },
            diffuse_only,
        );
        // Near-parallel light => sin(T,L) ~ 0.
        let parallel = evaluate_hair_kajiya_direct(
            base_surface(),
            frame(),
            DirectLightSample {
                direction: normalize_or([0.98, 0.2, 0.0], [1.0, 0.0, 0.0]),
                ..light()
            },
            diffuse_only,
        );
        assert!(
            sum(perpendicular) > sum(parallel),
            "perpendicular diffuse {perpendicular:?} should beat parallel {parallel:?}"
        );
    }

    #[test]
    fn specular_adds_energy_over_pure_diffuse() {
        let diffuse_only = evaluate_hair_kajiya_direct(
            base_surface(),
            frame(),
            light(),
            HairKajiyaParams {
                specular_strength: 0.0,
                ..Default::default()
            },
        );
        let with_spec = evaluate_hair_kajiya_direct(
            base_surface(),
            frame(),
            light(),
            HairKajiyaParams {
                specular_strength: 0.6,
                ..Default::default()
            },
        );
        assert!(
            sum(with_spec) > sum(diffuse_only),
            "specular {with_spec:?} should add energy over diffuse {diffuse_only:?}"
        );
    }

    #[test]
    fn sharper_exponent_narrows_the_highlight() {
        // On the specular cone a larger exponent must not brighten an
        // off-peak sample; the tighter lobe falls off faster.
        let off_peak_dir = normalize_or([0.5, 0.6, 0.62], [0.0, 1.0, 0.0]);
        let make = |exponent: f32| {
            evaluate_hair_kajiya_direct(
                SurfaceSample {
                    base_color: [0.0, 0.0, 0.0],
                    ..base_surface()
                },
                frame(),
                DirectLightSample {
                    direction: off_peak_dir,
                    ..light()
                },
                HairKajiyaParams {
                    diffuse_strength: 0.0,
                    specular_strength: 1.0,
                    specular_exponent: exponent,
                    ..Default::default()
                },
            )
        };
        let broad = make(10.0);
        let sharp = make(120.0);
        assert!(
            sum(sharp) < sum(broad),
            "sharp off-peak {sharp:?} should be dimmer than broad {broad:?}"
        );
    }

    #[test]
    fn evaluation_is_deterministic() {
        let a = evaluate_hair_kajiya_direct(
            base_surface(),
            frame(),
            light(),
            HairKajiyaParams::default(),
        );
        let b = evaluate_hair_kajiya_direct(
            base_surface(),
            frame(),
            light(),
            HairKajiyaParams::default(),
        );
        assert_eq!(a, b);
    }
}
