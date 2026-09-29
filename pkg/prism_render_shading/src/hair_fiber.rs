//! Fibre-level fur/hair BSDF (Yan et al. dual-cylinder, near-field high-config).
//!
//! Highest-fidelity close-up sibling of [`crate::evaluate_hair_chiang_direct`].
//! Where the Chiang front end models a single absorbing cortex cylinder with an
//! isotropic medulla *fill*, this reference follows Yan et al. "Physically
//! Accurate Fur Reflectance: Modeling, Measurement and Rendering" (2015) and its
//! production-friendly successor (2017): the fibre is a *double cylinder* — an
//! outer cortex and an inner scattering medulla — and the medulla contributes
//! two extra *scattered* lobes on top of the three cuticle lobes:
//!
//! * **R** - white dielectric cuticle glint (base-color independent),
//! * **TT** - forward cortex transmission, `(1 - F)^2` and one Beer-Lambert pass,
//! * **TRT** - coloured secondary highlight, `(1 - F)^2 * F` and two passes,
//! * **`TTs`** - *scattered* transmission: light enters the cortex, scatters
//!   inside the medulla core and exits forward. Broadened longitudinally (the
//!   core randomises the exit elevation) and shaped azimuthally by a
//!   Henyey-Greenstein phase lobe, attenuated by the cortex transmittance and
//!   the medulla single-scatter albedo. This is the desaturated forward glow
//!   that gives grey hair and animal fur their soft, luminous look.
//! * **`TRTs`** - *scattered* internal reflection: the same core-scatter path
//!   with one extra internal bounce (`F` weighted, two cortex passes).
//!
//! The medulla is parameterised by its radius fraction `kappa`
//! (`medulla_ratio`), its single-scatter albedo (`medulla_scatter`), its
//! absorption (`medulla_absorption`) and the forward-scatter anisotropy
//! (`scatter_anisotropy`, the Henyey-Greenstein `g`). With `kappa = 0` the two
//! scattered lobes vanish and the model degrades gracefully to the three
//! cuticle lobes. As with the other front ends the absorption is derived from
//! the base color (Chiang et al. 2016 eq. 9) so the transmission lobes stay
//! energy-consistent as the fibre darkens.
//!
//! The result is deterministic, array-in / array-out and is the numerical twin
//! of `hair_fiber.wesl` (`hair_fiber_direct`). Emissive is folded in once.

use core::f32::consts::PI;

use bevy_math::ops;

use crate::vecmath::{add, dot, mul_scalar, normalize_or, sub};
use crate::{DirectLightSample, ShadingFrame, SurfaceSample};

/// Reciprocal of four-pi, the isotropic phase-function normaliser.
const INV_FOUR_PI: f32 = 1.0 / (4.0 * PI);
/// Roughness floor shared with the principled lobe (`MIN_ROUGHNESS`).
const MIN_ROUGHNESS: f32 = 0.045;
/// Azimuthal-roughness floor so the color-to-`sigma_a` denominator stays finite.
const MIN_AZIMUTHAL_ROUGHNESS: f32 = 0.02;
/// Color floor fed to `ln` so a pure-black fibre yields a large but finite
/// absorption instead of a non-finite one.
const MIN_ABSORPTION_COLOR: f32 = 1.0e-3;
/// Cuticle tilt (radians) separating the R/TT/TRT lobes longitudinally.
const CUTICLE_SHIFT: f32 = 0.035;

/// Tunable controls for the fibre-level fur/hair front end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFiberParams {
    /// Azimuthal (`N`) roughness in `[0, 1]`; widens the azimuthal lobes and
    /// feeds the color-to-absorption fit.
    pub azimuthal_roughness: f32,
    /// Longitudinal (`M`) roughness in `[0, 1]`. When `> 0` it overrides the
    /// surface `perceptual_roughness` for the longitudinal lobe widths.
    pub longitudinal_roughness: f32,
    /// Medulla radius fraction `kappa` in `[0, 1]`; `0` is a solid cortex fibre
    /// (fine human hair), higher values model thick grey hair / animal fur.
    pub medulla_ratio: f32,
    /// Medulla single-scatter albedo in `[0, 1]` (`sigma_s / sigma_t`).
    pub medulla_scatter: f32,
    /// Medulla absorption coefficient (`>= 0`); attenuates the scattered lobes
    /// as the core path lengthens.
    pub medulla_absorption: f32,
    /// Henyey-Greenstein anisotropy `g` in `(-1, 1)`; `> 0` is forward
    /// scattering (the physical fur regime), `0` isotropic.
    pub scatter_anisotropy: f32,
    /// Fibre index of refraction (keratin is about `1.55`).
    pub ior: f32,
}

impl Default for HairFiberParams {
    /// A medium grey-fur fibre: moderate azimuthal roughness, a scattering
    /// medulla with mild forward anisotropy.
    fn default() -> Self {
        Self {
            azimuthal_roughness: 0.3,
            longitudinal_roughness: 0.0,
            medulla_ratio: 0.4,
            medulla_scatter: 0.6,
            medulla_absorption: 0.2,
            scatter_anisotropy: 0.4,
            ior: 1.55,
        }
    }
}

/// Normalized longitudinal Gaussian `Mp` of width `beta` evaluated at `x`.
fn hair_gaussian(beta: f32, x: f32) -> f32 {
    ops::exp(-0.5 * x * x / (beta * beta)) / (ops::sqrt(2.0 * PI) * beta)
}

/// Schlick Fresnel for the fibre dielectric interface at `cos_theta`.
fn hair_fresnel(cos_theta: f32, ior: f32) -> f32 {
    let r = (1.0 - ior) / (1.0 + ior);
    let f0 = r * r;
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    f0 + (1.0 - f0) * ops::powf(m, 5.0)
}

/// Henyey-Greenstein phase function for scatter cosine `cos_theta` and
/// anisotropy `g`. Normalised so the isotropic (`g = 0`) case is `INV_FOUR_PI`.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g = g.clamp(-0.95, 0.95);
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    INV_FOUR_PI * (1.0 - g * g) / (denom.max(1.0e-4) * ops::sqrt(denom.max(1.0e-4)))
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
/// and azimuthal roughness, following Chiang et al. (2016) eq. 9.
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

/// Evaluates the fibre-level fur/hair BSDF for a single analytic light.
///
/// Returns linear radiance plus the surface emissive term. When the light is
/// fully occluded only the emissive term is returned, matching the other lobes.
#[must_use]
pub fn evaluate_hair_fiber_direct(
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    params: HairFiberParams,
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

    // Cortex absorption coefficient from the fibre color.
    let sigma_a = sigma_a_from_color(base, params.azimuthal_roughness);

    // Internal refraction azimuth and one-traversal optical path, then the
    // per-channel Beer-Lambert cortex transmittance.
    let sin_gamma_i = cos_half_phi.clamp(0.0, 1.0);
    let cos_gamma_i = (1.0 - sin_gamma_i * sin_gamma_i).max(0.0).sqrt();
    let sin_gamma_t = (sin_gamma_i / n_prime).clamp(0.0, 1.0);
    let cos_gamma_t = (1.0 - sin_gamma_t * sin_gamma_t).max(0.0).sqrt();
    let seg = 2.0 * cos_gamma_t / cos_theta_d;
    let transmittance = exp3([-sigma_a[0] * seg, -sigma_a[1] * seg, -sigma_a[2] * seg]);
    let half_transmittance = sqrt3(transmittance);

    // Interface Fresnel at the refracted-adjusted incidence.
    let fresnel = hair_fresnel(cos_theta_d * cos_gamma_i, eta).clamp(0.0, 1.0);
    let one_minus_f = 1.0 - fresnel;

    // Per-lobe longitudinal widths and cuticle shifts.
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

    // ---- Cuticle lobes (shared with the Chiang front end) ----------------
    // R: white primary cuticle reflection (base-color independent).
    let r_lobe = {
        let np = 0.25 * cos_half_phi;
        [mp[0] * np * fresnel; 3]
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
        mul_scalar(hadamard(transmittance, transmittance), weight)
    };

    // ---- Medulla scattered lobes (the fibre-level contribution) ----------
    // The core scatters light over a broad range, so both scattered lobes use a
    // widened longitudinal Gaussian and an azimuthal Henyey-Greenstein phase
    // lobe. Their strength scales with the medulla radius fraction `kappa`, the
    // single-scatter albedo and the core absorption along a `kappa`-scaled path.
    let kappa = params.medulla_ratio.clamp(0.0, 1.0);
    let albedo = params.medulla_scatter.clamp(0.0, 1.0);
    let (tts_lobe, trts_lobe) = if kappa > 0.0 && albedo > 0.0 {
        // Broadened longitudinal profile: the core randomises the exit
        // elevation, so widen `beta` with the core fraction.
        let beta_s = (r2 * (1.0 + 3.0 * kappa)).clamp(MIN_ROUGHNESS * MIN_ROUGHNESS, 4.0);
        let ms = hair_gaussian(beta_s, long_arg);
        // Forward-scatter azimuthal profile. `cos_phi = -1` is straight-through
        // forward scatter, so evaluate the phase lobe about the forward cosine.
        let phase = henyey_greenstein(-cos_phi, params.scatter_anisotropy);
        // Core attenuation along a `kappa`-scaled path (Beer-Lambert in the
        // medulla), then the single-scatter albedo weight.
        let core_atten = ops::exp(-params.medulla_absorption.max(0.0) * (0.5 + kappa));
        let core_weight = kappa * albedo * core_atten * ms * phase;

        // TTs: (1 - F)^2 in, one cortex half-pass each way, forward scatter.
        let tts_weight = core_weight * one_minus_f * one_minus_f;
        let tts = mul_scalar(hadamard(half_transmittance, base), tts_weight);

        // TRTs: one extra internal reflection (F weighted), a full cortex pass
        // of tint (T) for the longer detour.
        let trts_weight = core_weight * one_minus_f * one_minus_f * fresnel;
        let trts = mul_scalar(hadamard(transmittance, base), trts_weight);
        (tts, trts)
    } else {
        ([0.0; 3], [0.0; 3])
    };

    let scatter = add(
        add(add(r_lobe, tt_lobe), trt_lobe),
        add(tts_lobe, trts_lobe),
    );
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
            base_color: [0.5, 0.4, 0.3],
            perceptual_roughness: 0.4,
            ..Default::default()
        }
    }

    fn sum(c: [f32; 3]) -> f32 {
        c[0] + c[1] + c[2]
    }

    #[test]
    fn fiber_is_finite_and_non_negative_across_parameters() {
        let tangents = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.7, 0.0, 0.7]];
        let dirs = [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, -0.5, 0.86],
            [-0.7, 0.2, 0.68],
        ];
        for roughness in [0.045, 0.3, 0.7, 1.0] {
            for kappa in [0.0, 0.5, 1.0] {
                for g in [-0.6, 0.0, 0.6] {
                    for tangent in tangents {
                        for dir in dirs {
                            let value = evaluate_hair_fiber_direct(
                                SurfaceSample {
                                    perceptual_roughness: roughness,
                                    ..base_surface()
                                },
                                ShadingFrame { tangent, ..frame() },
                                DirectLightSample {
                                    direction: dir,
                                    ..light()
                                },
                                HairFiberParams {
                                    medulla_ratio: kappa,
                                    scatter_anisotropy: g,
                                    ..Default::default()
                                },
                            );
                            assert!(
                                value.into_iter().all(|c| c.is_finite() && c >= 0.0),
                                "roughness={roughness} kappa={kappa} g={g} \
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
            evaluate_hair_fiber_direct(surface, frame(), shadowed, HairFiberParams::default()),
            surface.emissive
        );
    }

    #[test]
    fn medulla_adds_scattered_fill_over_solid_cortex() {
        // A scattering medulla core must add luminous fill on top of the solid
        // cortex fibre, never remove it.
        let solid = evaluate_hair_fiber_direct(
            base_surface(),
            frame(),
            light(),
            HairFiberParams {
                medulla_ratio: 0.0,
                ..Default::default()
            },
        );
        let cored = evaluate_hair_fiber_direct(
            base_surface(),
            frame(),
            light(),
            HairFiberParams {
                medulla_ratio: 0.8,
                ..Default::default()
            },
        );
        assert!(
            sum(cored) > sum(solid),
            "medulla {cored:?} should add fill over solid cortex {solid:?}"
        );
    }

    #[test]
    fn forward_anisotropy_favours_forward_scatter() {
        // With a forward tangent geometry the light scatters forward through the
        // core; a forward-biased phase (`g > 0`) must send more energy that way
        // than a backward-biased one (`g < 0`).
        // Backlit forward-transmission geometry: L and V sit on opposite
        // azimuthal sides of the strand (`cos_phi = -1`), so the medulla
        // forward-scatter lobe peaks straight through the core.
        let tangent = [0.0, 0.0, 1.0];
        let dir = normalize_or([0.0, -0.7, -0.72], [0.0, 1.0, 0.0]);
        let make = |g: f32| {
            evaluate_hair_fiber_direct(
                SurfaceSample {
                    base_color: [0.8, 0.8, 0.8],
                    ..base_surface()
                },
                ShadingFrame { tangent, ..frame() },
                DirectLightSample {
                    direction: dir,
                    ..light()
                },
                HairFiberParams {
                    medulla_ratio: 0.9,
                    medulla_scatter: 0.9,
                    medulla_absorption: 0.0,
                    scatter_anisotropy: g,
                    ..Default::default()
                },
            )
        };
        let forward = make(0.7);
        let backward = make(-0.7);
        assert!(
            sum(forward) > sum(backward),
            "forward phase {forward:?} should beat backward {backward:?}"
        );
    }

    #[test]
    fn darker_fibre_absorbs_more_transmission_energy() {
        let dir = normalize_or([0.6, 0.2, 0.77], [0.0, 1.0, 0.0]);
        let bright = evaluate_hair_fiber_direct(
            SurfaceSample {
                base_color: [0.8, 0.6, 0.4],
                ..base_surface()
            },
            frame(),
            DirectLightSample {
                direction: dir,
                ..light()
            },
            HairFiberParams::default(),
        );
        let dark = evaluate_hair_fiber_direct(
            SurfaceSample {
                base_color: [0.1, 0.07, 0.04],
                ..base_surface()
            },
            frame(),
            DirectLightSample {
                direction: dir,
                ..light()
            },
            HairFiberParams::default(),
        );
        assert!(
            sum(dark) < sum(bright),
            "dark fibre {dark:?} should absorb more than bright {bright:?}"
        );
    }

    #[test]
    fn evaluation_is_deterministic() {
        let a = evaluate_hair_fiber_direct(
            base_surface(),
            frame(),
            light(),
            HairFiberParams::default(),
        );
        let b = evaluate_hair_fiber_direct(
            base_surface(),
            frame(),
            light(),
            HairFiberParams::default(),
        );
        assert_eq!(a, b);
    }
}
