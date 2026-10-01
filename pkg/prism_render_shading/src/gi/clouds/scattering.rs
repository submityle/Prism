//! In-cloud lighting: Beer-Lambert transmittance, the Beer-powder two-term
//! model, Henyey-Greenstein and dual-lobe phase functions, and an
//! energy-conserving multiple-scattering octave approximation — CPU golden.
//!
//! Light transport inside a cloud is governed by extinction (how fast radiance
//! is attenuated) and the angular phase function (how a scattering event
//! redistributes radiance). This module provides the backend-neutral reference
//! for those terms, independent of the volumetric-GI kernels so clouds can own
//! their characteristic *powder* darkening and dual-lobe anisotropy:
//!
//! * [`optical_depth`] / [`transmittance`] / [`beer_lambert`] — the extinction
//!   term `T = exp(-sigma_t * d)`.
//! * [`powder`] / [`beer_powder`] — the "powder sugar" two-term model that
//!   darkens cloud edges facing the light (approximating the in-scatter
//!   deficit near low-density boundaries).
//! * [`henyey_greenstein`] — the single-lobe HG phase.
//! * [`dual_lobe_hg`] — a forward/backward blend of two HG lobes, the standard
//!   cloud phase with a strong forward peak and a weaker backward glow.
//! * [`multiple_scattering_transmittance`] / [`multiple_scattering_phase`] — an
//!   octave-attenuated multiple-scattering approximation (Wrenninge-style):
//!   successive octaves scatter with reduced extinction and eccentricity,
//!   brightening deep cloud interiors while conserving energy.
//!
//! # Conventions
//! * `g` is the HG anisotropy in `(-1, 1)` (`> 0` forward, `< 0` backward,
//!   `0` isotropic); it is clamped just inside the open interval so the
//!   denominator stays positive. `cos_theta` is clamped to `[-1, 1]`.
//! * Optical depths, extinction, and distances are clamped non-negative;
//!   transmittances lie in `[0, 1]` with `T(0) = 1` and decrease monotonically
//!   with optical depth.
//! * Every phase function integrates to `1` over the sphere (energy conserving)
//!   and every result is finite and non-negative (never `NaN`).
//! * Transcendental maths goes through [`bevy_math::ops`]. Pure deterministic
//!   functions: no RNG, I/O, GPU, or `unsafe`.

use bevy_math::ops;
use core::f32::consts::PI;

/// Reciprocal of `4π`, the isotropic phase-function value.
const INV_4PI: f32 = 1.0 / (4.0 * PI);
/// Largest magnitude the anisotropy `g` may take; keeps `1 + g^2 - 2gc > 0`.
const MAX_G: f32 = 1.0 - 1.0e-4;
/// Exponent clamp guarding against denormal underflow on the GPU twin.
const MAX_EXPONENT: f32 = 80.0;
/// Maximum octave count honoured by the multiple-scattering sums.
const MAX_OCTAVES: u32 = 8;

/// Clamps `value` into `[lo, hi]`, mapping non-finite inputs to `lo`.
#[inline]
fn clamp_finite(value: f32, lo: f32, hi: f32) -> f32 {
    if value.is_finite() { value.clamp(lo, hi) } else { lo }
}

/// Clamps a value non-negative, mapping non-finite inputs to `0`.
#[inline]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() { value.max(0.0) } else { 0.0 }
}

/// Optical depth `tau = sigma_t * d` (dimensionless), clamped non-negative.
#[inline]
pub fn optical_depth(sigma_t: f32, distance: f32) -> f32 {
    let tau = clamp_non_negative(sigma_t) * clamp_non_negative(distance);
    clamp_non_negative(tau)
}

/// Transmittance `T = exp(-tau)` for a given optical depth, in `[0, 1]`.
///
/// `tau` is clamped non-negative and the exponent saturated so `T(0) = 1`,
/// `T` decreases monotonically with `tau`, and the result never underflows to a
/// non-finite value.
#[inline]
pub fn transmittance(tau: f32) -> f32 {
    let tau = clamp_non_negative(tau).min(MAX_EXPONENT);
    ops::exp(-tau).clamp(0.0, 1.0)
}

/// Beer-Lambert transmittance `T = exp(-sigma_t * d)` for scalar extinction.
///
/// Convenience wrapper over [`optical_depth`] + [`transmittance`]; `sigma_t`
/// and `d` are clamped non-negative so `T` lies in `[0, 1]`.
#[inline]
pub fn beer_lambert(sigma_t: f32, distance: f32) -> f32 {
    transmittance(optical_depth(sigma_t, distance))
}

/// Powder-sugar term `1 - exp(-2 tau)`, in `[0, 1]`.
///
/// This factor rises from `0` at `tau = 0` to `1` as the optical depth grows,
/// modelling the suppressed in-scattering near a low-density boundary that
/// darkens cloud edges facing the light. It is monotonically increasing in
/// `tau`.
#[inline]
pub fn powder(tau: f32) -> f32 {
    let tau = clamp_non_negative(tau).min(MAX_EXPONENT);
    (1.0 - ops::exp(-2.0 * tau)).clamp(0.0, 1.0)
}

/// Beer-powder two-term light energy, in `[0, 1]`.
///
/// Blends plain Beer-Lambert transmittance with the powder-darkened variant
/// `2 * beer * powder` by `strength in [0, 1]`:
/// `beer * ((1 - strength) + strength * 2 * powder)`. At `strength = 0` it is
/// exactly [`transmittance`] (monotonically decreasing in `tau`); larger
/// `strength` adds the characteristic edge darkening. The result stays in
/// `[0, 1]`.
#[inline]
pub fn beer_powder(tau: f32, strength: f32) -> f32 {
    let tau = clamp_non_negative(tau);
    let strength = clamp_finite(strength, 0.0, 1.0);
    let beer = transmittance(tau);
    let pw = powder(tau);
    let energy = beer * ((1.0 - strength) + strength * 2.0 * pw);
    energy.clamp(0.0, 1.0)
}

/// Henyey-Greenstein phase function `p(cos_theta, g)`.
///
/// Returns `(1 - g^2) / (4π (1 + g^2 - 2 g cos_theta)^{3/2})`, the normalised
/// angular distribution (sphere integral `1`). `g` is clamped to `(-1, 1)` and
/// `cos_theta` to `[-1, 1]`; as `g -> 0` it collapses to the isotropic
/// `1 / 4π`.
#[inline]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let cos_theta = clamp_finite(cos_theta, -1.0, 1.0);
    let g = clamp_finite(g, -MAX_G, MAX_G);
    let denom = (1.0 + g * g - 2.0 * g * cos_theta).max(1.0e-12);
    let value = (1.0 - g * g) * INV_4PI / ops::powf(denom, 1.5);
    if value.is_finite() { value.max(0.0) } else { INV_4PI }
}

/// Dual-lobe Henyey-Greenstein phase: a forward/backward blend.
///
/// Returns `(1 - blend) * HG(cos, g_forward) + blend * HG(cos, g_backward)`,
/// with `blend in [0, 1]`. Because each lobe integrates to `1` and the weights
/// sum to `1`, the blend is itself energy conserving. The canonical cloud phase
/// uses a strong forward `g_forward` and a weaker backward `g_backward` to
/// capture both the sun-side silver lining and the ambient backward glow.
#[inline]
pub fn dual_lobe_hg(cos_theta: f32, g_forward: f32, g_backward: f32, blend: f32) -> f32 {
    let blend = clamp_finite(blend, 0.0, 1.0);
    let fwd = henyey_greenstein(cos_theta, g_forward);
    let bwd = henyey_greenstein(cos_theta, g_backward);
    let value = (1.0 - blend) * fwd + blend * bwd;
    if value.is_finite() { value.max(0.0) } else { INV_4PI }
}

/// Clamps an octave count into `[1, MAX_OCTAVES]`.
#[inline]
fn clamp_octaves(octaves: u32) -> u32 {
    octaves.clamp(1, MAX_OCTAVES)
}

/// Energy-conserving multiple-scattering transmittance, in `[0, 1]`.
///
/// Approximates deep multiple scattering as a normalised weighted average of
/// Beer-Lambert transmittances evaluated at geometrically reduced optical
/// depths: octave `n` contributes weight `attenuation^n` at optical depth
/// `tau * attenuation^n`. Because `attenuation in [0, 1)` reduces the effective
/// optical depth of higher octaves, the result brightens cloud interiors
/// relative to single scattering while remaining in `[0, 1]`, monotonically
/// decreasing in `tau`, and reducing to plain [`transmittance`] at
/// `octaves = 1`.
#[inline]
pub fn multiple_scattering_transmittance(tau: f32, octaves: u32, attenuation: f32) -> f32 {
    let tau = clamp_non_negative(tau);
    let octaves = clamp_octaves(octaves);
    let attenuation = clamp_finite(attenuation, 0.0, 1.0);
    let mut weight = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves {
        sum += weight * transmittance(tau * weight);
        norm += weight;
        weight *= attenuation;
    }
    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        transmittance(tau)
    }
}

/// Energy-conserving multiple-scattering phase, integrating to `1`.
///
/// Mirrors [`multiple_scattering_transmittance`] in the angular domain: octave
/// `n` is a Henyey-Greenstein lobe with eccentricity `g * eccentricity^n` and
/// weight `attenuation^n`, the lot normalised by the summed weights. Each lobe
/// integrates to `1`, so the normalised blend also integrates to `1` — energy
/// conserving — while higher octaves flatten toward isotropic to model the
/// diffusion of repeatedly scattered light.
#[inline]
pub fn multiple_scattering_phase(
    cos_theta: f32,
    g: f32,
    octaves: u32,
    attenuation: f32,
    eccentricity: f32,
) -> f32 {
    let octaves = clamp_octaves(octaves);
    let attenuation = clamp_finite(attenuation, 0.0, 1.0);
    let eccentricity = clamp_finite(eccentricity, 0.0, 1.0);
    let mut weight = 1.0f32;
    let mut g_octave = clamp_finite(g, -MAX_G, MAX_G);
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves {
        sum += weight * henyey_greenstein(cos_theta, g_octave);
        norm += weight;
        weight *= attenuation;
        g_octave *= eccentricity;
    }
    if norm > 0.0 {
        let value = sum / norm;
        if value.is_finite() { value.max(0.0) } else { INV_4PI }
    } else {
        henyey_greenstein(cos_theta, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Midpoint quadrature of a `cos_theta` function over the unit sphere:
    /// `∫ f dω = 2π ∫_{-1}^{1} f dμ`.
    fn sphere_integral(f: impl Fn(f32) -> f32) -> f32 {
        let steps = 40_000;
        let mut acc = 0.0f64;
        for i in 0..steps {
            let mu = -1.0 + 2.0 * (i as f32 + 0.5) / steps as f32;
            acc += f(mu) as f64;
        }
        let dmu = 2.0 / steps as f32;
        (2.0 * PI as f64 * acc * dmu as f64) as f32
    }

    #[test]
    fn transmittance_unit_at_zero_and_bounded() {
        assert!((transmittance(0.0) - 1.0).abs() < 1e-6);
        assert!((beer_lambert(1.0, 0.0) - 1.0).abs() < 1e-6);
        for tau in [0.0f32, 0.5, 1.0, 5.0, 100.0, f32::INFINITY] {
            let t = transmittance(tau);
            assert!((0.0..=1.0).contains(&t), "transmittance out of range: {t}");
        }
    }

    #[test]
    fn transmittance_monotonically_decreases() {
        let mut prev = transmittance(0.0);
        for i in 0..=200 {
            let tau = i as f32 * 0.1;
            let t = transmittance(tau);
            assert!(t <= prev + 1e-6, "not monotone at tau={tau}: {t} > {prev}");
            prev = t;
        }
    }

    #[test]
    fn powder_rises_from_zero_to_one() {
        assert!(powder(0.0) < 1e-6);
        assert!(powder(50.0) > 1.0 - 1e-6);
        let mut prev = powder(0.0);
        for i in 0..=100 {
            let tau = i as f32 * 0.1;
            let p = powder(tau);
            assert!(p + 1e-6 >= prev, "powder not monotone at tau={tau}");
            assert!((0.0..=1.0).contains(&p));
            prev = p;
        }
    }

    #[test]
    fn beer_powder_reduces_to_beer_at_zero_strength() {
        for i in 0..=100 {
            let tau = i as f32 * 0.1;
            assert!((beer_powder(tau, 0.0) - transmittance(tau)).abs() < 1e-6);
        }
    }

    #[test]
    fn beer_powder_in_unit_range() {
        for &strength in &[0.0f32, 0.3, 0.7, 1.0] {
            for i in 0..=200 {
                let tau = i as f32 * 0.1;
                let v = beer_powder(tau, strength);
                assert!((0.0..=1.0).contains(&v), "beer-powder out of range: {v}");
            }
        }
    }

    #[test]
    fn hg_integrates_to_unity() {
        for g in [-0.8f32, -0.3, 0.0, 0.3, 0.8] {
            let integral = sphere_integral(|mu| henyey_greenstein(mu, g));
            assert!((integral - 1.0).abs() < 3e-3, "g={g} integral={integral}");
        }
    }

    #[test]
    fn hg_isotropic_at_zero_g() {
        for mu in [-1.0f32, -0.4, 0.0, 0.4, 1.0] {
            assert!((henyey_greenstein(mu, 0.0) - INV_4PI).abs() < 1e-6);
        }
    }

    #[test]
    fn dual_lobe_integrates_to_unity() {
        for blend in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let integral = sphere_integral(|mu| dual_lobe_hg(mu, 0.8, -0.3, blend));
            assert!((integral - 1.0).abs() < 3e-3, "blend={blend} integral={integral}");
        }
    }

    #[test]
    fn dual_lobe_has_forward_and_backward_lobes() {
        // Strong forward lobe dominates when the backward weight is small.
        let fwd = dual_lobe_hg(1.0, 0.8, -0.3, 0.2);
        let side = dual_lobe_hg(0.0, 0.8, -0.3, 0.2);
        assert!(fwd > side, "forward lobe not peaked: fwd={fwd} side={side}");
    }

    #[test]
    fn ms_transmittance_reduces_to_beer_at_one_octave() {
        for i in 0..=50 {
            let tau = i as f32 * 0.2;
            assert!(
                (multiple_scattering_transmittance(tau, 1, 0.5) - transmittance(tau)).abs() < 1e-6
            );
        }
    }

    #[test]
    fn ms_transmittance_bounded_and_monotone() {
        let mut prev = multiple_scattering_transmittance(0.0, 5, 0.5);
        for i in 0..=200 {
            let tau = i as f32 * 0.1;
            let t = multiple_scattering_transmittance(tau, 5, 0.5);
            assert!((0.0..=1.0).contains(&t), "MS transmittance out of range: {t}");
            assert!(t <= prev + 1e-6, "MS transmittance not monotone at tau={tau}");
            prev = t;
        }
    }

    #[test]
    fn ms_transmittance_brightens_interior() {
        // Multiple scattering lets more light through than single scattering.
        let tau = 4.0;
        let single = transmittance(tau);
        let multi = multiple_scattering_transmittance(tau, 5, 0.5);
        assert!(multi >= single, "MS darker than single: {multi} < {single}");
    }

    #[test]
    fn ms_phase_integrates_to_unity() {
        for octaves in [1u32, 3, 5] {
            let integral =
                sphere_integral(|mu| multiple_scattering_phase(mu, 0.8, octaves, 0.5, 0.5));
            assert!((integral - 1.0).abs() < 3e-3, "octaves={octaves} integral={integral}");
        }
    }

    #[test]
    fn everything_finite_on_extreme_inputs() {
        for g in [-5.0f32, -1.0, 1.0, 5.0, f32::NAN, f32::INFINITY] {
            for mu in [-2.0f32, 0.0, 2.0, f32::NAN] {
                assert!(henyey_greenstein(mu, g).is_finite());
                assert!(dual_lobe_hg(mu, g, -g, 0.5).is_finite());
                assert!(multiple_scattering_phase(mu, g, 4, 0.5, 0.5).is_finite());
            }
        }
        assert!(beer_powder(f32::NAN, f32::NAN).is_finite());
        assert!(multiple_scattering_transmittance(f32::NAN, 4, f32::NAN).is_finite());
    }
}
