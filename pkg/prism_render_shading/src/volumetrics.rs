//! Backend-neutral CPU golden for froxel volumetric fog (single scattering).
//!
//! A froxel volumetric integrates *participating media* — fog, dust, god rays —
//! into a view-frustum-aligned 3D grid ("froxels"). Each froxel stores the
//! medium's scattering/absorption coefficients and the light in-scattered into
//! it; a final pass marches front-to-back along each froxel column, folding the
//! per-slice in-scattering under the accumulated transmittance so nearer media
//! correctly occlude farther media. This is the Frostbite / UE "Volumetric Fog"
//! scattering integration, expressed as pure math so the CPU golden and the
//! `shaders/volumetrics.wesl` GPU twin evaluate the *same* reduction.
//!
//! The physical model is single scattering with a Henyey-Greenstein phase
//! function:
//!
//! * **Extinction** `sigma_t = sigma_s + sigma_a` drives Beer-Lambert
//!   transmittance `T = exp(-sigma_t * d)` per colour channel.
//! * **In-scattering** from a light is `sigma_s * phase(cos) * L * vis`, the
//!   radiance the medium redirects toward the eye.
//! * **Energy-conserving slice integration** folds the constant per-slice
//!   source `S` through the slice's own attenuation analytically:
//!   `S_int = (S - S * T) / sigma_t` (with the `sigma_t -> 0` limit `S * d`),
//!   avoiding the banding a midpoint sample would leave.
//!
//! Everything is per-channel `[f32; 3]` so coloured extinction (e.g. a warm
//! dusty haze that eats blue faster than red) integrates correctly.

use bevy_math::ops;

/// A participating-medium sample inside one froxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediumSample {
    /// Scattering coefficient `sigma_s` per channel (how much light the medium
    /// redirects). Larger values give a denser, brighter fog.
    pub scattering: [f32; 3],
    /// Absorption coefficient `sigma_a` per channel (how much light the medium
    /// swallows). Extinction is `sigma_s + sigma_a`.
    pub absorption: [f32; 3],
    /// Emissive radiance density added as an in-slice source (self-lit media
    /// such as fire/magical fog), independent of any external light.
    pub emissive: [f32; 3],
}

impl Default for MediumSample {
    /// Vacuum: no scattering, no absorption, no emission — a froxel that leaves
    /// the background radiance untouched (transmittance 1, in-scattering 0).
    fn default() -> Self {
        Self {
            scattering: [0.0; 3],
            absorption: [0.0; 3],
            emissive: [0.0; 3],
        }
    }
}

impl MediumSample {
    /// Per-channel extinction `sigma_t = sigma_s + sigma_a`.
    #[must_use]
    pub fn extinction(&self) -> [f32; 3] {
        [
            self.scattering[0] + self.absorption[0],
            self.scattering[1] + self.absorption[1],
            self.scattering[2] + self.absorption[2],
        ]
    }
}

/// One froxel ready for the column integration: its medium, the light already
/// in-scattered toward the eye (summed over every light that reaches it, via
/// [`in_scatter`]) and the slice's view-space thickness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Froxel {
    /// The participating medium in this slice.
    pub medium: MediumSample,
    /// Radiance in-scattered toward the eye from all lights, *before* the
    /// slice's own attenuation and the accumulated transmittance are applied.
    /// The medium emissive is folded in here as well by [`froxel_source`].
    pub in_scattering: [f32; 3],
    /// View-space thickness of this froxel slice (the near-to-far depth span).
    pub thickness: f32,
}

/// The result of integrating one froxel column front-to-back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumetricIntegration {
    /// Accumulated in-scattered radiance reaching the eye from the whole column.
    pub in_scattering: [f32; 3],
    /// Remaining per-channel transmittance after the column (the weight the
    /// background/opaque surface radiance is multiplied by before adding the
    /// fog). `1` = clear, `0` = fully occluded.
    pub transmittance: [f32; 3],
}

/// Numerical floor below which extinction is treated as zero (the `S * d`
/// limit of the analytic slice integral).
const EXTINCTION_EPSILON: f32 = 1.0e-5;

/// Henyey-Greenstein phase function.
///
/// `cos_theta` is the cosine between the light direction and the view
/// direction; `g` in `(-1, 1)` is the anisotropy (`0` isotropic, `>0` forward
/// scattering / haze glow around the sun, `<0` backward). Normalised so its
/// integral over the sphere is `1`; `g == 0` returns exactly `1 / (4*pi)`.
#[must_use]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g = g.clamp(-0.99, 0.99);
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    // denom is strictly positive for |g| < 1; guard the pow against a tiny
    // negative from rounding.
    let denom = denom.max(1.0e-8);
    let inv_4pi = 1.0 / (4.0 * core::f32::consts::PI);
    inv_4pi * (1.0 - g2) / (denom * denom.sqrt())
}

/// Per-channel Beer-Lambert transmittance `exp(-sigma_t * distance)`.
#[must_use]
pub fn transmittance(extinction: [f32; 3], distance: f32) -> [f32; 3] {
    let d = distance.max(0.0);
    [
        ops::exp(-extinction[0].max(0.0) * d),
        ops::exp(-extinction[1].max(0.0) * d),
        ops::exp(-extinction[2].max(0.0) * d),
    ]
}

/// Radiance a light in-scatters toward the eye inside a froxel.
///
/// `cos_theta` is the light/view cosine feeding the phase function,
/// `light_radiance` is the (already attenuated + shadowed off-medium) incident
/// radiance and `visibility` is the medium-shadow term (e.g. the light's shadow
/// map sampled at the froxel). The result is `sigma_s * phase * L * vis`,
/// per channel.
#[must_use]
pub fn in_scatter(
    medium: &MediumSample,
    cos_theta: f32,
    g: f32,
    light_radiance: [f32; 3],
    visibility: f32,
) -> [f32; 3] {
    let phase = henyey_greenstein(cos_theta, g);
    let vis = visibility.clamp(0.0, 1.0);
    [
        medium.scattering[0] * phase * light_radiance[0] * vis,
        medium.scattering[1] * phase * light_radiance[1] * vis,
        medium.scattering[2] * phase * light_radiance[2] * vis,
    ]
}

/// The constant per-slice source radiance: in-scattered light plus the medium's
/// own emissive density.
#[must_use]
pub fn froxel_source(froxel: &Froxel) -> [f32; 3] {
    [
        froxel.in_scattering[0] + froxel.medium.emissive[0],
        froxel.in_scattering[1] + froxel.medium.emissive[1],
        froxel.in_scattering[2] + froxel.medium.emissive[2],
    ]
}

/// Energy-conserving analytic integration of a constant source `S` across a
/// slice of thickness `d` under its own extinction `sigma_t`.
///
/// Returns `(S - S * exp(-sigma_t * d)) / sigma_t` per channel, with the
/// `sigma_t -> 0` limit `S * d`. This is the closed form of
/// `integral_0^d S * exp(-sigma_t * t) dt`, so a thick, dense slice cannot
/// out-radiate the light actually scattered within it.
#[must_use]
pub fn integrate_slice(source: [f32; 3], extinction: [f32; 3], thickness: f32) -> [f32; 3] {
    let d = thickness.max(0.0);
    let t = transmittance(extinction, d);
    let mut out = [0.0f32; 3];
    let mut c = 0;
    while c < 3 {
        let sigma = extinction[c].max(0.0);
        out[c] = if sigma > EXTINCTION_EPSILON {
            (source[c] - source[c] * t[c]) / sigma
        } else {
            source[c] * d
        };
        c += 1;
    }
    out
}

/// Integrates one froxel column front-to-back.
///
/// Walks the froxels from the camera outward, accumulating each slice's
/// energy-conserving in-scattering under the transmittance built up by the
/// nearer slices, then multiplying the running transmittance by the slice's
/// own. The returned [`VolumetricIntegration`] is applied as
/// `final = background * transmittance + in_scattering`.
#[must_use]
pub fn integrate_froxel_column(froxels: &[Froxel]) -> VolumetricIntegration {
    let mut accum = [0.0f32; 3];
    let mut trans = [1.0f32; 3];
    for froxel in froxels {
        let ext = froxel.medium.extinction();
        let source = froxel_source(froxel);
        let slice = integrate_slice(source, ext, froxel.thickness);
        let mut c = 0;
        while c < 3 {
            accum[c] += trans[c] * slice[c];
            c += 1;
        }
        let slice_t = transmittance(ext, froxel.thickness);
        trans[0] *= slice_t[0];
        trans[1] *= slice_t[1];
        trans[2] *= slice_t[2];
    }
    VolumetricIntegration {
        in_scattering: accum,
        transmittance: trans,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INV_4PI: f32 = 1.0 / (4.0 * core::f32::consts::PI);

    #[test]
    fn isotropic_phase_is_uniform() {
        for &cos in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            assert!((henyey_greenstein(cos, 0.0) - INV_4PI).abs() < 1.0e-6);
        }
    }

    #[test]
    fn forward_scattering_peaks_toward_the_light() {
        // g > 0 must be brightest looking into the light (cos = 1) and dimmest
        // looking away (cos = -1).
        let g = 0.6;
        let fwd = henyey_greenstein(1.0, g);
        let side = henyey_greenstein(0.0, g);
        let back = henyey_greenstein(-1.0, g);
        assert!(fwd > side && side > back, "{fwd} {side} {back}");
    }

    #[test]
    fn backward_scattering_mirrors_forward() {
        let g = 0.5;
        let fwd = henyey_greenstein(1.0, g);
        let back_of_neg = henyey_greenstein(-1.0, -g);
        assert!((fwd - back_of_neg).abs() < 1.0e-6);
    }

    #[test]
    fn phase_integrates_to_one_over_the_sphere() {
        // Numerically integrate p(cos) over the sphere: 2*pi * integral_-1^1.
        for &g in &[0.0f32, 0.3, -0.4, 0.7] {
            let n = 200_000;
            let mut acc = 0.0f64;
            for i in 0..n {
                let cos = -1.0 + 2.0 * (i as f32 + 0.5) / n as f32;
                acc += henyey_greenstein(cos, g) as f64;
            }
            let dcos = 2.0 / n as f64;
            let integral = 2.0 * core::f64::consts::PI * acc * dcos;
            assert!((integral - 1.0).abs() < 2.0e-3, "g={g} integral={integral}");
        }
    }

    #[test]
    fn transmittance_is_one_at_zero_distance_and_decays() {
        let ext = [0.5, 1.0, 2.0];
        assert_eq!(transmittance(ext, 0.0), [1.0, 1.0, 1.0]);
        let t1 = transmittance(ext, 1.0);
        let t2 = transmittance(ext, 2.0);
        for c in 0..3 {
            assert!(t2[c] < t1[c] && t1[c] < 1.0);
            // Blue (highest extinction) attenuates fastest.
        }
        assert!(t1[2] < t1[0], "coloured extinction: blue faster than red");
    }

    #[test]
    fn slice_integral_reduces_to_source_times_depth_in_vacuum() {
        let s = [2.0, 3.0, 4.0];
        let out = integrate_slice(s, [0.0; 3], 1.5);
        for c in 0..3 {
            assert!((out[c] - s[c] * 1.5).abs() < 1.0e-6);
        }
    }

    #[test]
    fn slice_integral_is_bounded_by_unattenuated_energy() {
        // With real extinction the integrated slice must be < S * d.
        let s = [5.0, 5.0, 5.0];
        let d = 2.0;
        let out = integrate_slice(s, [1.0, 1.0, 1.0], d);
        for c in 0..3 {
            assert!(out[c] > 0.0 && out[c] < s[c] * d, "{}", out[c]);
        }
    }

    #[test]
    fn slice_integral_matches_closed_form() {
        let s = [1.0, 1.0, 1.0];
        let sigma = 0.75f32;
        let d = 1.3f32;
        let expected = (1.0 - ops::exp(-sigma * d)) / sigma;
        let out = integrate_slice(s, [sigma; 3], d);
        assert!((out[0] - expected).abs() < 1.0e-6);
    }

    fn clear_froxel(source: [f32; 3], d: f32) -> Froxel {
        Froxel {
            medium: MediumSample::default(),
            in_scattering: source,
            thickness: d,
        }
    }

    #[test]
    fn empty_column_is_fully_transmissive_with_no_fog() {
        let out = integrate_froxel_column(&[]);
        assert_eq!(out.in_scattering, [0.0; 3]);
        assert_eq!(out.transmittance, [1.0; 3]);
    }

    #[test]
    fn clear_slices_pass_light_through_and_sum_sources() {
        // Vacuum froxels (no extinction) => transmittance stays 1, in-scatter
        // is just the sum of source * thickness.
        let col = [
            clear_froxel([1.0, 0.0, 0.0], 1.0),
            clear_froxel([0.0, 2.0, 0.0], 0.5),
        ];
        let out = integrate_froxel_column(&col);
        assert_eq!(out.transmittance, [1.0; 3]);
        assert!((out.in_scattering[0] - 1.0).abs() < 1.0e-6);
        assert!((out.in_scattering[1] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn column_transmittance_is_the_product_of_slices() {
        let dense = Froxel {
            medium: MediumSample {
                scattering: [0.0; 3],
                absorption: [1.0; 3],
                emissive: [0.0; 3],
            },
            in_scattering: [0.0; 3],
            thickness: 1.0,
        };
        let out = integrate_froxel_column(&[dense, dense]);
        let single = transmittance([1.0; 3], 1.0)[0];
        assert!((out.transmittance[0] - single * single).abs() < 1.0e-6);
    }

    #[test]
    fn near_opaque_slice_occludes_farther_in_scattering() {
        // A near, fully opaque scattering slice must dominate; a bright far
        // slice behind it barely contributes because transmittance collapses.
        let opaque = Froxel {
            medium: MediumSample {
                scattering: [0.0; 3],
                absorption: [50.0; 3],
                emissive: [0.0; 3],
            },
            in_scattering: [0.1, 0.1, 0.1],
            thickness: 1.0,
        };
        let far_bright = clear_froxel([10.0, 10.0, 10.0], 1.0);
        let out = integrate_froxel_column(&[opaque, far_bright]);
        // Transmittance after the opaque slice is ~0, so the far slice's 10.0
        // is almost entirely blocked.
        assert!(out.transmittance[0] < 1.0e-10);
        assert!(
            out.in_scattering[0] < 0.2,
            "far slice leaked: {}",
            out.in_scattering[0]
        );
    }

    #[test]
    fn front_to_back_ordering_matters() {
        // Same two slices in opposite order give different eye radiance because
        // the near slice attenuates the far one.
        let a = Froxel {
            medium: MediumSample {
                scattering: [1.0; 3],
                absorption: [1.0; 3],
                emissive: [0.0; 3],
            },
            in_scattering: [1.0, 0.0, 0.0],
            thickness: 1.0,
        };
        let b = Froxel {
            medium: MediumSample {
                scattering: [1.0; 3],
                absorption: [1.0; 3],
                emissive: [0.0; 3],
            },
            in_scattering: [0.0, 0.0, 1.0],
            thickness: 1.0,
        };
        let ab = integrate_froxel_column(&[a, b]);
        let ba = integrate_froxel_column(&[b, a]);
        assert!((ab.in_scattering[0] - ba.in_scattering[0]).abs() > 1.0e-3);
    }

    #[test]
    fn in_scatter_scales_with_scattering_phase_and_visibility() {
        let medium = MediumSample {
            scattering: [0.5, 0.5, 0.5],
            absorption: [0.1; 3],
            emissive: [0.0; 3],
        };
        let lit = in_scatter(&medium, 1.0, 0.0, [4.0, 4.0, 4.0], 1.0);
        let shadowed = in_scatter(&medium, 1.0, 0.0, [4.0, 4.0, 4.0], 0.0);
        assert_eq!(shadowed, [0.0; 3]);
        // sigma_s * (1/4pi) * L * 1
        let expect = 0.5 * INV_4PI * 4.0;
        assert!((lit[0] - expect).abs() < 1.0e-6);
    }

    #[test]
    fn emissive_medium_glows_without_a_light() {
        let froxel = Froxel {
            medium: MediumSample {
                scattering: [0.0; 3],
                absorption: [0.0; 3],
                emissive: [1.0, 0.5, 0.25],
            },
            in_scattering: [0.0; 3],
            thickness: 2.0,
        };
        let out = integrate_froxel_column(&[froxel]);
        assert!((out.in_scattering[0] - 2.0).abs() < 1.0e-6);
        assert!((out.in_scattering[1] - 1.0).abs() < 1.0e-6);
    }
}
