//! Poisson-Gaussian sensor-noise model — CPU golden reference.
//!
//! A digital camera sensor corrupts an ideal signal with two physically
//! distinct noise sources:
//!
//! * **Photon shot noise** — photon arrivals are Poisson-distributed, so the
//!   variance of the collected charge equals its mean. Referred to the output
//!   through the analog gain this gives a variance term that is **linear in the
//!   signal** and scales with gain (ISO).
//! * **Read noise** — a signal-independent, roughly Gaussian floor from the
//!   readout electronics / ADC. It sets the noise in the deep shadows where
//!   almost no photons were collected.
//!
//! The classic Foi et al. "Poissonian-Gaussian" model collapses both into an
//! affine noise-variance function `σ²(s) = a·s + b`, with `a` the shot slope
//! (gain-dependent) and `b` the read-noise floor. This module is the backend-
//! neutral reference for that model plus a *deterministic* Gaussian sampler so
//! the WESL / GPU twin reproduces the exact noise field.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, IO, GPU, global state, or `unsafe`.
//! * `signal` is a normalised sensor response in `[0, 1]`; it is clamped there.
//! * `iso` is clamped to be `>= `[`ISO_BASE`] and finite; `read_noise` to be
//!   finite and `>= 0`.
//! * Variance / std are always finite and non-negative.
//! * Transcendentals go through [`bevy_math::ops`]; `sqrt` uses the inherent
//!   `f32` method. The Gaussian uses Box-Muller with a CLT cross-check.
//! * Integer hashing is shared with [`super::grain`] for a single reproducible
//!   PRNG across the whole film-grain subsystem.

use bevy_math::{ops, UVec2};

use super::grain::{hash_pixel, uniform01};

/// Base ISO at which the analog gain is unity.
pub const ISO_BASE: f32 = 100.0;

/// Shot-noise slope coefficient (variance per unit signal at unit gain).
///
/// Chosen small so that at base ISO the shot term is a few percent of the
/// signal — a reasonable normalised stand-in for a full-well-scaled Poisson
/// variance without needing an absolute electron count.
const SHOT_COEFF: f32 = 0.02;

/// Smallest Gaussian probability mass we allow into `ln`, to avoid `ln(0)`.
const MIN_U: f32 = 1.0e-7;

/// Replaces a non-finite scalar with `fallback`.
#[inline]
#[must_use]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Relative analog gain for an ISO value: `iso / ISO_BASE`, clamped `>= 1`.
///
/// ISO below base is treated as base (unity gain); the gain is what amplifies
/// both the shot slope and the read-noise floor.
#[inline]
#[must_use]
pub fn iso_gain(iso: f32) -> f32 {
    let i = finite_or(iso, ISO_BASE).max(ISO_BASE);
    i / ISO_BASE
}

/// Affine Poisson-Gaussian noise **variance** `σ²(signal)` in output units².
///
/// `σ² = SHOT_COEFF · gain · signal + read_noise² · gain`, where
/// `gain = iso / ISO_BASE`. The first term is the gain-amplified photon shot
/// noise (linear in signal); the second is the gain-amplified read-noise floor
/// (independent of signal). Both terms, and hence the whole variance, grow with
/// ISO. The result is finite and non-negative.
#[must_use]
pub fn noise_variance(signal: f32, iso: f32, read_noise: f32) -> f32 {
    let s = finite_or(signal, 0.0).clamp(0.0, 1.0);
    let gain = iso_gain(iso);
    let read = finite_or(read_noise, 0.0).max(0.0);

    let shot = SHOT_COEFF * gain * s;
    let read_var = read * read * gain;
    (shot + read_var).max(0.0)
}

/// Noise **standard deviation** `σ(signal)` — the square root of
/// [`noise_variance`]. Finite and non-negative.
#[must_use]
pub fn noise_std(signal: f32, iso: f32, read_noise: f32) -> f32 {
    noise_variance(signal, iso, read_noise).max(0.0).sqrt()
}

/// A standard-normal sample `N(0, 1)` via the Box-Muller transform.
///
/// Draws two decorrelated uniforms from the pixel hash and maps them to one
/// Gaussian deviate `z = sqrt(-2 ln u1) · cos(2π u2)`. `u1` is floored to
/// [`MIN_U`] so `ln` never sees zero. Deterministic in `(p, seed)` and finite.
#[must_use]
pub fn gaussian_boxmuller(p: UVec2, seed: u32) -> f32 {
    let u1 = uniform01(hash_pixel(p, seed)).max(MIN_U);
    let u2 = uniform01(hash_pixel(p, seed ^ 0x27D4_EB2F));
    let radius = (-2.0 * ops::ln(u1)).max(0.0).sqrt();
    let z = radius * ops::cos(core::f32::consts::TAU * u2);
    finite_or(z, 0.0)
}

/// A standard-normal sample `N(0, 1)` via the central-limit sum of 12 uniforms.
///
/// `z = (Σ_{i=0}^{11} u_i) - 6`, which has mean `0` and variance `1` exactly for
/// independent uniforms. Cheaper and branch-free; used as a cross-check of the
/// Box-Muller path and available as an alternative sampler. Deterministic.
#[must_use]
pub fn gaussian_clt(p: UVec2, seed: u32) -> f32 {
    let mut acc = 0.0_f32;
    let mut i = 0u32;
    while i < 12 {
        acc += uniform01(hash_pixel(p, seed ^ i.wrapping_mul(0x9E37_79B9)));
        i += 1;
    }
    finite_or(acc - 6.0, 0.0)
}

/// The signed sensor-noise delta for a pixel: `σ(signal) · N(0, 1)`.
///
/// Uses [`noise_std`] for the amplitude and [`gaussian_boxmuller`] for the unit
/// deviate, so the delta's variance equals [`noise_variance`] in expectation.
/// Deterministic in `(signal, iso, read_noise, p, seed)` and finite.
#[must_use]
pub fn sensor_noise(signal: f32, iso: f32, read_noise: f32, p: UVec2, seed: u32) -> f32 {
    let std = noise_std(signal, iso, read_noise);
    std * gaussian_boxmuller(p, seed)
}

/// Applies sensor noise to a signal and clamps the result back to `[0, 1]`.
///
/// Returns `clamp(signal + sensor_noise(...), 0, 1)`. The input `signal` is
/// sanitized to `[0, 1]` first; the output is always finite and in range.
#[must_use]
pub fn apply_sensor_noise(signal: f32, iso: f32, read_noise: f32, p: UVec2, seed: u32) -> f32 {
    let s = finite_or(signal, 0.0).clamp(0.0, 1.0);
    let noisy = s + sensor_noise(s, iso, read_noise, p, seed);
    finite_or(noisy, s).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `iso_gain` is unity at base ISO, grows linearly above it, and never
    /// drops below one.
    #[test]
    fn iso_gain_behaviour() {
        assert_eq!(iso_gain(ISO_BASE), 1.0);
        assert_eq!(iso_gain(50.0), 1.0, "sub-base ISO clamps to unity gain");
        assert!((iso_gain(400.0) - 4.0).abs() < 1.0e-6);
        assert_eq!(iso_gain(f32::NAN), 1.0);
    }

    /// Variance is affine in the signal: the slope is constant across the
    /// signal range (the defining property of the shot-noise term).
    #[test]
    fn variance_is_linear_in_signal() {
        let iso = 200.0;
        let read = 0.01;
        let slope = |a: f32, b: f32| {
            (noise_variance(b, iso, read) - noise_variance(a, iso, read)) / (b - a)
        };
        let s1 = slope(0.1, 0.2);
        let s2 = slope(0.5, 0.9);
        assert!((s1 - s2).abs() < 1.0e-6, "slope not constant: {s1} vs {s2}");
        // Intercept at signal 0 is the read-noise floor only, and positive.
        let intercept = noise_variance(0.0, iso, read);
        assert!(intercept > 0.0, "read floor should be positive: {intercept}");
    }

    /// The shot slope and read floor both scale with ISO, so variance grows
    /// monotonically with ISO at any fixed signal.
    #[test]
    fn variance_scales_with_iso() {
        let read = 0.02;
        let v100 = noise_variance(0.5, 100.0, read);
        let v400 = noise_variance(0.5, 400.0, read);
        let v1600 = noise_variance(0.5, 1600.0, read);
        assert!(v100 < v400, "v100={v100} v400={v400}");
        assert!(v400 < v1600, "v400={v400} v1600={v1600}");
        // Shot slope scales exactly with gain.
        let slope_at = |iso: f32| noise_variance(1.0, iso, 0.0) - noise_variance(0.0, iso, 0.0);
        assert!((slope_at(400.0) - 4.0 * slope_at(100.0)).abs() < 1.0e-6);
    }

    /// `noise_std` is the square root of `noise_variance`.
    #[test]
    fn std_matches_sqrt_variance() {
        for &s in &[0.0_f32, 0.25, 0.5, 1.0] {
            let var = noise_variance(s, 800.0, 0.03);
            let std = noise_std(s, 800.0, 0.03);
            assert!((std - var.sqrt()).abs() < 1.0e-6, "s={s} std={std} var={var}");
        }
    }

    /// Variance / std stay finite and non-negative under degenerate inputs.
    #[test]
    fn variance_is_sanitized() {
        let v = noise_variance(f32::NAN, f32::INFINITY, -5.0);
        assert!(v.is_finite() && v >= 0.0, "v={v}");
        let s = noise_std(f32::INFINITY, f32::NAN, f32::NAN);
        assert!(s.is_finite() && s >= 0.0, "s={s}");
    }

    /// Box-Muller samples are deterministic, finite, and approximately
    /// zero-mean with unit variance over a large grid.
    #[test]
    fn boxmuller_is_standard_normal() {
        assert_eq!(gaussian_boxmuller(UVec2::new(3, 4), 1), gaussian_boxmuller(UVec2::new(3, 4), 1));
        let mut sum = 0.0_f64;
        let mut sq = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..128u32 {
            for x in 0..128u32 {
                let z = gaussian_boxmuller(UVec2::new(x, y), 7);
                assert!(z.is_finite(), "z={z}");
                sum += z as f64;
                sq += (z as f64) * (z as f64);
                n += 1.0;
            }
        }
        let mean = sum / n;
        let var = sq / n - mean * mean;
        assert!(mean.abs() < 0.03, "mean={mean}");
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }

    /// The CLT sampler is also approximately standard-normal and zero-mean.
    #[test]
    fn clt_is_standard_normal() {
        let mut sum = 0.0_f64;
        let mut sq = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..128u32 {
            for x in 0..128u32 {
                let z = gaussian_clt(UVec2::new(x, y), 11);
                sum += z as f64;
                sq += (z as f64) * (z as f64);
                n += 1.0;
            }
        }
        let mean = sum / n;
        let var = sq / n - mean * mean;
        assert!(mean.abs() < 0.03, "mean={mean}");
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }

    /// The empirical variance of the sampled noise matches `noise_variance`.
    #[test]
    fn sampled_noise_variance_matches_model() {
        let signal = 0.6;
        let iso = 800.0;
        let read = 0.02;
        let expected = noise_variance(signal, iso, read) as f64;
        let mut sum = 0.0_f64;
        let mut sq = 0.0_f64;
        let mut n = 0.0_f64;
        for y in 0..160u32 {
            for x in 0..160u32 {
                let d = sensor_noise(signal, iso, read, UVec2::new(x, y), 5) as f64;
                sum += d;
                sq += d * d;
                n += 1.0;
            }
        }
        let mean = sum / n;
        let var = sq / n - mean * mean;
        // Within 15% of the analytic variance.
        assert!((var - expected).abs() <= 0.15 * expected + 1.0e-6, "var={var} expected={expected}");
    }

    /// `apply_sensor_noise` is deterministic and stays within `[0, 1]`.
    #[test]
    fn apply_is_deterministic_and_clamped() {
        let a = apply_sensor_noise(0.5, 1600.0, 0.05, UVec2::new(9, 2), 3);
        assert_eq!(a, apply_sensor_noise(0.5, 1600.0, 0.05, UVec2::new(9, 2), 3));
        for y in 0..32u32 {
            for x in 0..32u32 {
                let v = apply_sensor_noise(1.0, 3200.0, 0.1, UVec2::new(x, y), 1);
                assert!((0.0..=1.0).contains(&v), "v={v}");
                let w = apply_sensor_noise(0.0, 3200.0, 0.1, UVec2::new(x, y), 1);
                assert!((0.0..=1.0).contains(&w), "w={w}");
            }
        }
    }
}
