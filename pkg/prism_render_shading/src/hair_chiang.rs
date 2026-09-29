//! Chiang near-field hair BSDF (energy-conserving, absorption/medulla).
//!
//! High-fidelity close-up sibling of [`crate::evaluate_hair_direct`]. Where the
//! baseline strand lobe uses Karis' real-time Marschner form with a `pow`-tinted
//! absorption, this reference follows Chiang et al. "A Practical and
//! Controllable Hair and Fur Model for Production Path Tracing" (2016): the
//! fibre absorption coefficient is derived from the base color and the
//! azimuthal roughness so the transmission lobes stay energy-consistent as the
//! fibre darkens, and an optional hollow-core medulla adds the desaturated
//! forward scatter that gray and animal hair need at close range.
//!
//! The three cuticle lobes keep their physical meaning:
//!
//! * **R** - the white dielectric glint off the outer cuticle (base-color
//!   independent), weighted by the interface Fresnel.
//! * **TT** - forward transmission attenuated by `(1 - F)^2` and one internal
//!   traversal of the absorbing cortex, tinted by the Beer-Lambert transmittance
//!   `T = exp(-sigma_a * path)`.
//! * **TRT** - the coloured secondary highlight from a double transmission plus
//!   an internal reflection, attenuated by `(1 - F)^2 * F` and `T^2`.
//! * **medulla** - an optional isotropic core-scatter fill (Chiang's fur
//!   double-cylinder extension), tinted by the cortex half-path `sqrt(T)` and
//!   gated by a wrapped cosine, so it survives grazing light without blowing up.
//!
//! The absorption derivation is the distinguishing physical content: for a
//! near-black fibre `ln(color)` is large and negative, so `sigma_a` grows and
//! the TT/TRT lobes are driven down, whereas the baseline lobe would keep a
//! `pow`-tinted residue. The result is deterministic, array-in / array-out and
//! is the numerical twin of `hair_chiang.wesl` (`hair_chiang_direct`). Emissive
//! is folded in exactly once, matching the other direct lobes.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::vecmath::{add, dot, mul_scalar, normalize_or, sub};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of pi, matching the GPU `INV_PI` constant.
const INV_PI: f32 = 1.0 / PI;
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Azimuthal-roughness floor so the color-to-`sigma_a` denominator stays finite.
const MIN_AZIMUTHAL_ROUGHNESS: f32 = 0.02;
/// Color floor fed to `ln` so a pure-black fibre yields a large but finite
/// absorption instead of a non-finite one.
const MIN_ABSORPTION_COLOR: f32 = 1.0e-3;
/// Cuticle tilt (radians) separating the R/TT/TRT lobes longitudinally; the
/// three lobes are shifted by `-2a`, `a` and `4a` about it (Marschner tilt).
const CUTICLE_SHIFT: f32 = 0.035;

/// Tunable controls for the Chiang near-field hair front end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairChiangParams {
    /// Azimuthal (`N`) roughness in `[0, 1]`; widens the azimuthal lobes and
    /// feeds the color-to-absorption fit so rougher fibres absorb differently.
    pub azimuthal_roughness: f32,
    /// Longitudinal (`M`) roughness in `[0, 1]`. When `> 0` it overrides the
    /// surface `perceptual_roughness` for the longitudinal lobe widths, letting
    /// a close-up fibre decouple its highlight tightness from the base material.
    pub longitudinal_roughness: f32,
    /// Hollow-core medulla volume fraction in `[0, 1]`; `0` is a solid keratin
    /// fibre (human hair), higher values model gray/animal fur.
    pub medulla_ratio: f32,
    /// Isotropic scatter albedo of the medulla core in `[0, 1]`.
    pub medulla_scatter: f32,
    /// Fibre index of refraction (keratin is about `1.55`).
    pub ior: f32,
}

impl Default for HairChiangParams {
    /// A solid dark-human-hair fibre: moderate azimuthal roughness, longitudinal
    /// width taken from the surface, no medulla.
    fn default() -> Self {
        Self {
            azimuthal_roughness: 0.3,
            longitudinal_roughness: 0.0,
            medulla_ratio: 0.0,
            medulla_scatter: 0.5,
            ior: 1.55,
        }
    }
}

/// Normalized longitudinal Gaussian `Mp` of width `beta` evaluated at `x`.
/// Mirrors `hair_gaussian`.
fn hair_gaussian(beta: f32, x: f32) -> f32 {
    ops::exp(-0.5 * x * x / (beta * beta)) / (ops::sqrt(2.0 * PI) * beta)
}

/// Schlick Fresnel for the fibre dielectric interface at `cos_theta` with the
/// given index of refraction. Mirrors `hair_chiang_fresnel`.
fn hair_fresnel(cos_theta: f32, ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    let f0 = r * r;
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    f0 + (1.0 - f0) * ops::powf(m, 5.0)
}

/// Per-channel `exp(base)` for the Beer-Lambert transmittance.
fn exp3(base: [f32; 3]) -> [f32; 3] {
    [ops::exp(base[0]), ops::exp(base[1]), ops::exp(base[2])]
}

/// Per-channel `sqrt(base)` (clamped non-negative) for the cortex half-path.
fn sqrt3(base: [f32; 3]) -> [f32; 3] {
    [
        ops::sqrt(base[0].max(0.0)),
        ops::sqrt(base[1].max(0.0)),
        ops::sqrt(base[2].max(0.0)),
    ]
}

/// Component-wise product `a * b`.
fn hadamard(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// Derives the per-channel absorption coefficient `sigma_a` from the base color
/// and azimuthal roughness, following Chiang et al. (2016) eq. 9. A darker
/// channel maps to a larger `sigma_a`, so the transmission lobes lose energy as
/// the fibre darkens (the physically correct, energy-conserving behaviour).
fn sigma_a_from_color(color: [f32; 3], azimuthal_roughness: f32) -> [f32; 3] {
    let b = azimuthal_roughness.clamp(MIN_AZIMUTHAL_ROUGHNESS, 1.0);
    let denom = 5.969 - 0.215 * b + 2.532 * b * b - 10.73 * b * b * b
        + 5.574 * b * b * b * b
        + 0.245 * b * b * b * b * b;
    let mut out = [0.0_f32; 3];
    let mut i = 0;
    while i < 3 {
        let c = color[i].clamp(MIN_ABSORPTION_COLOR, 1.0);
        let t = ops::ln(c) / denom;
        out[i] = t * t;
        i += 1;
    }
    out
}

/// Evaluates the Chiang near-field hair BSDF for a single analytic light.
///
/// Returns linear radiance plus the surface emissive term. When the light is
/// fully occluded only the emissive term is returned, matching the other lobes.
#[must_use]
pub fn evaluate_hair_chiang_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: HairChiangParams,
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

    // Longitudinal sines about the strand tangent.
    let sin_theta_l = dot(t, l).clamp(-1.0, 1.0);
    let sin_theta_v = dot(t, v).clamp(-1.0, 1.0);

    // Half difference angle drives Fresnel and the transmission widths.
    let cos_theta_d = ops::cos(0.5 * (ops::asin(sin_theta_v) - ops::asin(sin_theta_l)).abs());
    let cos_theta_d = cos_theta_d.max(1.0e-3);

    // Azimuthal cosine from the tangent-plane projections of L and V.
    let lp = sub(l, mul_scalar(t, sin_theta_l));
    let vp = sub(v, mul_scalar(t, sin_theta_v));
    let cos_phi = dot(lp, vp) / (dot(lp, lp) * dot(vp, vp) + 1.0e-4).sqrt();
    let cos_phi = cos_phi.clamp(-1.0, 1.0);
    let cos_half_phi = (0.5 + 0.5 * cos_phi).clamp(0.0, 1.0).sqrt();

    // Modified (Bravais) refractive index for the projected azimuth.
    let eta = params.ior.max(1.001);
    let n_prime = (eta * eta - 1.0).max(0.0).sqrt() / cos_theta_d + cos_theta_d;
    let n_prime = n_prime.max(1.001);

    // Absorption coefficient from the fibre color (the Chiang contribution).
    let sigma_a = sigma_a_from_color(base, params.azimuthal_roughness);

    // Internal refraction azimuth and one-traversal optical path length, then
    // the per-channel Beer-Lambert transmittance through the cortex.
    let sin_gamma_i = cos_half_phi.clamp(0.0, 1.0);
    let cos_gamma_i = (1.0 - sin_gamma_i * sin_gamma_i).max(0.0).sqrt();
    let sin_gamma_t = (sin_gamma_i / n_prime).clamp(0.0, 1.0);
    let cos_gamma_t = (1.0 - sin_gamma_t * sin_gamma_t).max(0.0).sqrt();
    let seg = 2.0 * cos_gamma_t / cos_theta_d;
    let transmittance = exp3([-sigma_a[0] * seg, -sigma_a[1] * seg, -sigma_a[2] * seg]);

    // Interface Fresnel at the refracted-adjusted incidence.
    let fresnel = hair_fresnel(cos_theta_d * cos_gamma_i, eta).clamp(0.0, 1.0);
    let one_minus_f = 1.0 - fresnel;

    // Per-lobe longitudinal widths and cuticle shifts. A positive
    // `longitudinal_roughness` decouples the highlight tightness from the base
    // material; otherwise the surface roughness drives it.
    let roughness = if params.longitudinal_roughness > 0.0 {
        params.longitudinal_roughness
    } else {
        surface.perceptual_roughness
    }
    .clamp(MIN_ROUGHNESS, 1.0);
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
        let s = mp[0] * np * fresnel;
        [s; 3]
    };

    // TT: forward transmission, `(1 - F)^2` times one cortex traversal.
    let tt_lobe = {
        let np = ops::exp(-3.65 * cos_phi - 3.98);
        let weight = mp[1] * np * one_minus_f * one_minus_f;
        mul_scalar(transmittance, weight)
    };

    // TRT: coloured secondary highlight, `(1 - F)^2 * F` times two traversals.
    let trt_lobe = {
        let np = ops::exp(17.0 * cos_phi - 16.78);
        let weight = mp[2] * np * one_minus_f * one_minus_f * fresnel;
        let t2 = hadamard(transmittance, transmittance);
        mul_scalar(t2, weight)
    };

    // Medulla: optional isotropic hollow-core scatter, tinted by the cortex
    // half-path and gated by a wrapped cosine so it stays bounded.
    let medulla = {
        let ratio = params.medulla_ratio.clamp(0.0, 1.0);
        let albedo = params.medulla_scatter.clamp(0.0, 1.0);
        if ratio > 0.0 && albedo > 0.0 {
            let wrap = (dot(n, l) * 0.5 + 0.5).clamp(0.0, 1.0);
            let half_path = sqrt3(transmittance);
            mul_scalar(half_path, ratio * albedo * wrap * INV_PI)
        } else {
            [0.0; 3]
        }
    };

    let scatter = add(add(add(r_lobe, tt_lobe), trt_lobe), medulla);
    let radiance = mul_scalar(hadamard(scatter, light.illuminance), visibility);
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
    fn chiang_is_finite_and_non_negative_across_parameters() {
        let tangents = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.0, 0.7]];
        let dirs = [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, -0.5, 0.86],
            [-0.7, 0.2, 0.68],
        ];
        for roughness in [0.045, 0.3, 0.7, 1.0] {
            for medulla in [0.0, 0.5, 1.0] {
                for tangent in tangents {
                    for dir in dirs {
                        let value = evaluate_hair_chiang_direct(
                            SurfaceSample {
                                perceptual_roughness: roughness,
                                ..base_surface()
                            },
                            ShadingFrame { tangent, ..frame() },
                            DirectLightSample {
                                direction: dir,
                                ..light()
                            },
                            HairChiangParams {
                                medulla_ratio: medulla,
                                ..Default::default()
                            },
                        );
                        assert!(
                            value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                            "roughness={roughness} medulla={medulla} tangent={tangent:?} \
                             dir={dir:?} -> {value:?}"
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
            evaluate_hair_chiang_direct(surface, frame(), shadowed, HairChiangParams::default()),
            surface.emissive
        );
    }

    #[test]
    fn darker_fibre_absorbs_more_transmission_energy() {
        // Off-peak azimuth so TT/TRT dominate over the white R glint. A darker
        // fibre must map to a larger `sigma_a` and therefore a dimmer coloured
        // response - the energy-conserving absorption the baseline lobe lacks.
        let dir = normalize_or([0.6, 0.2, 0.77], [0.0, 1.0, 0.0]);
        let bright = evaluate_hair_chiang_direct(
            SurfaceSample {
                base_color: [0.8, 0.6, 0.4],
                ..base_surface()
            },
            frame(),
            DirectLightSample {
                direction: dir,
                ..light()
            },
            HairChiangParams::default(),
        );
        let dark = evaluate_hair_chiang_direct(
            SurfaceSample {
                base_color: [0.1, 0.07, 0.04],
                ..base_surface()
            },
            frame(),
            DirectLightSample {
                direction: dir,
                ..light()
            },
            HairChiangParams::default(),
        );
        assert!(
            sum(dark) < sum(bright),
            "dark fibre {dark:?} should absorb more than bright {bright:?}"
        );
    }

    #[test]
    fn medulla_adds_forward_scatter_fill() {
        // A hollow-core medulla must add luminous fill on top of the solid
        // fibre, never remove it.
        let solid = evaluate_hair_chiang_direct(
            base_surface(),
            frame(),
            light(),
            HairChiangParams {
                medulla_ratio: 0.0,
                ..Default::default()
            },
        );
        let cored = evaluate_hair_chiang_direct(
            base_surface(),
            frame(),
            light(),
            HairChiangParams {
                medulla_ratio: 0.8,
                medulla_scatter: 0.7,
                ..Default::default()
            },
        );
        assert!(
            sum(cored) > sum(solid),
            "medulla {cored:?} should add fill over solid {solid:?}"
        );
    }

    #[test]
    fn black_fibre_still_shows_the_white_primary_highlight() {
        // With a black base color the TT/TRT transmittance and medulla vanish,
        // so any positive radiance must be the white primary R reflection.
        let surface = SurfaceSample {
            base_color: [0.0, 0.0, 0.0],
            perceptual_roughness: 0.2,
            ..base_surface()
        };
        let tilted = DirectLightSample {
            direction: normalize_or([0.0, 0.6, 0.8], [0.0, 1.0, 0.0]),
            ..light()
        };
        let value =
            evaluate_hair_chiang_direct(surface, frame(), tilted, HairChiangParams::default());
        assert!(
            sum(value) > 0.0,
            "expected a pure specular glint, got {value:?}"
        );
    }

    #[test]
    fn evaluation_is_deterministic() {
        let a = evaluate_hair_chiang_direct(
            base_surface(),
            frame(),
            light(),
            HairChiangParams::default(),
        );
        let b = evaluate_hair_chiang_direct(
            base_surface(),
            frame(),
            light(),
            HairChiangParams::default(),
        );
        assert_eq!(a, b);
    }
}
