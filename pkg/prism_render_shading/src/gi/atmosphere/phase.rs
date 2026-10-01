//! Atmospheric angular phase functions: Rayleigh and Cornette-Shanks (Mie).
//!
//! The phase function `p(cosθ)` describes how a single scattering event
//! redistributes incident radiance over outgoing directions, where `θ` is the
//! angle between the incoming light direction and the outgoing view direction.
//! Both functions here are *normalised*: their integral over the unit sphere is
//! exactly `1`, so they conserve energy per scattering event.
//!
//! * [`rayleigh_phase`] — the molecular Rayleigh phase
//!   `3 / (16π) · (1 + cos²θ)`, symmetric about `θ = 90°`.
//! * [`cornette_shanks_phase`] — the Cornette-Shanks approximation to the Mie
//!   phase, parameterised by the asymmetry `g`. For `g = 0` it degenerates
//!   exactly to the Rayleigh phase above.
//!
//! # Conventions
//! * `cos_theta` is clamped to `[-1, 1]`.
//! * `g` is the Mie asymmetry in `(-1, 1)`: `g > 0` is forward scattering,
//!   `g < 0` backward, `g = 0` the symmetric Rayleigh-shaped degeneracy. It is
//!   clamped just inside the open interval so the denominator stays positive.
//! * Every result is finite and non-negative (never `NaN`); non-finite inputs
//!   fall back to the isotropic-safe branch.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   value method. Every function is a deterministic pure function with no
//!   RNG, I/O, GPU, or `unsafe`.

use bevy_math::ops;
use core::f32::consts::PI;

/// `3 / (16π)`, the Rayleigh phase normalisation constant.
const THREE_OVER_16PI: f32 = 3.0 / (16.0 * PI);
/// `3 / (8π)`, the Cornette-Shanks leading constant.
const THREE_OVER_8PI: f32 = 3.0 / (8.0 * PI);
/// Largest magnitude the asymmetry `g` may take; keeps `1 + g² - 2gμ > 0`.
const MAX_G: f32 = 1.0 - 1.0e-4;

/// Rayleigh phase function `p(cosθ) = 3 / (16π) · (1 + cos²θ)`.
///
/// The classic molecular-scattering distribution: symmetric, with equal
/// forward and backward lobes and a minimum at `θ = 90°`. `cos_theta` is
/// clamped to `[-1, 1]`; the result integrates to `1` over the sphere.
#[inline]
pub fn rayleigh_phase(cos_theta: f32) -> f32 {
    let mu = clamp_finite(cos_theta, -1.0, 1.0);
    let value = THREE_OVER_16PI * (1.0 + mu * mu);
    if value.is_finite() {
        value.max(0.0)
    } else {
        THREE_OVER_16PI
    }
}

/// Cornette-Shanks Mie phase function with asymmetry `g`.
///
/// Returns
/// `3 / (8π) · (1 - g²) / (2 + g²) · (1 + cos²θ) / (1 + g² - 2g·cosθ)^{3/2}`.
/// This is the energy-conserving Mie approximation (its sphere integral is
/// `1`). As `g → 0` it collapses *exactly* to [`rayleigh_phase`]; for `g > 0`
/// it concentrates radiance into the forward lobe. `g` is clamped to
/// `(-1, 1)` and `cos_theta` to `[-1, 1]`.
#[inline]
pub fn cornette_shanks_phase(cos_theta: f32, g: f32) -> f32 {
    let mu = clamp_finite(cos_theta, -1.0, 1.0);
    let g = clamp_finite(g, -MAX_G, MAX_G);
    let g2 = g * g;
    let numerator = (1.0 - g2) * (1.0 + mu * mu);
    // `denom` is strictly positive for |g| < 1; guard round-off at the poles.
    let denom = (1.0 + g2 - 2.0 * g * mu).max(1.0e-12);
    let value = THREE_OVER_8PI * numerator / ((2.0 + g2) * ops::powf(denom, 1.5));
    if value.is_finite() {
        value.max(0.0)
    } else {
        rayleigh_phase(mu)
    }
}

/// Clamps `value` into `[lo, hi]`, mapping non-finite inputs to `lo`.
#[inline]
fn clamp_finite(value: f32, lo: f32, hi: f32) -> f32 {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Midpoint quadrature of a `cos_theta` function over the sphere:
    /// `∫_sphere f dω = 2π ∫_{-1}^{1} f dμ`.
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
    fn rayleigh_integrates_to_unity() {
        let integral = sphere_integral(rayleigh_phase);
        assert!((integral - 1.0).abs() < 2e-3, "integral={integral}");
    }

    #[test]
    fn rayleigh_is_symmetric_with_min_at_right_angle() {
        for mu in [0.1f32, 0.4, 0.7, 1.0] {
            let a = rayleigh_phase(mu);
            let b = rayleigh_phase(-mu);
            assert!((a - b).abs() < 1e-6, "mu={mu} a={a} b={b}");
        }
        // Minimum at mu = 0 (theta = 90 deg), maxima at the poles.
        assert!(rayleigh_phase(0.0) < rayleigh_phase(1.0));
        assert!(rayleigh_phase(0.0) < rayleigh_phase(-1.0));
    }

    #[test]
    fn cornette_shanks_integrates_to_unity_for_several_g() {
        for g in [-0.8f32, -0.3, 0.0, 0.3, 0.76, 0.9] {
            let integral = sphere_integral(|mu| cornette_shanks_phase(mu, g));
            assert!((integral - 1.0).abs() < 3e-3, "g={g} integral={integral}");
        }
    }

    #[test]
    fn cornette_shanks_degenerates_to_rayleigh_at_zero_g() {
        for mu in [-1.0f32, -0.5, 0.0, 0.33, 0.5, 1.0] {
            let cs = cornette_shanks_phase(mu, 0.0);
            let r = rayleigh_phase(mu);
            assert!((cs - r).abs() < 1e-6, "mu={mu} cs={cs} r={r}");
        }
    }

    #[test]
    fn cornette_shanks_forward_peaks_for_positive_g() {
        let g = 0.76;
        let fwd = cornette_shanks_phase(1.0, g);
        let bwd = cornette_shanks_phase(-1.0, g);
        assert!(fwd > bwd, "fwd={fwd} bwd={bwd}");
        // Mirror symmetry p(mu, g) == p(-mu, -g).
        for mu in [-0.7f32, -0.2, 0.4, 0.9] {
            let a = cornette_shanks_phase(mu, g);
            let b = cornette_shanks_phase(-mu, -g);
            assert!((a - b).abs() < 1e-5, "mu={mu} a={a} b={b}");
        }
    }

    #[test]
    fn phases_stay_finite_and_nonnegative_on_extreme_inputs() {
        for g in [-5.0f32, -1.0, 1.0, 5.0, f32::NAN, f32::INFINITY] {
            for mu in [-2.0f32, -1.0, 0.0, 1.0, 2.0, f32::NAN] {
                let cs = cornette_shanks_phase(mu, g);
                let r = rayleigh_phase(mu);
                assert!(cs.is_finite() && cs >= 0.0, "g={g} mu={mu} cs={cs}");
                assert!(r.is_finite() && r >= 0.0, "mu={mu} r={r}");
            }
        }
    }
}
