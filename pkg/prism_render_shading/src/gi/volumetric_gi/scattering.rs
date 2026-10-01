//! Participating-media scattering: Henyey-Greenstein phase, Beer-Lambert
//! transmittance, and analytic single / multiple scattering — CPU golden.
//!
//! For a homogeneous participating medium the radiative transfer along a ray is
//! governed by two coefficients: the *scattering* coefficient `sigma_s` (how
//! much light is redirected into the ray) and the *extinction* coefficient
//! `sigma_t = sigma_s + sigma_a` (how fast radiance is attenuated, where
//! `sigma_a` is absorption).  Their ratio, the *single-scattering albedo*
//! `albedo = sigma_s / sigma_t`, bounds how much energy survives each bounce.
//!
//! This module is the backend-neutral numerical reference for the volumetric
//! lighting kernels.  It provides:
//!
//! * [`henyey_greenstein`] — the HG phase function `p(cos_theta, g)`, the
//!   workhorse angular scattering distribution, degenerating to the isotropic
//!   `1 / 4π` as the anisotropy `g → 0`.
//! * [`beer_lambert`] / [`beer_lambert_rgb`] — the transmittance
//!   `T = exp(-sigma_t * d)` for scalar and spectral extinction.
//! * [`optical_depth`] — the dimensionless `sigma_t * d`.
//! * [`in_scattering`] — the differential in-scattered radiance
//!   `sigma_s * p * L` at a point.
//! * [`single_scattering_homogeneous`] — the closed-form single-scattering
//!   integral along a ray through a homogeneous slab.
//! * [`multiple_scattering_octaves`] — a geometric-series multiple-scattering
//!   gain layered on top of the single-scattering estimate.
//!
//! # Conventions
//! * `g` is the HG anisotropy in `(-1, 1)`: `g > 0` is forward scattering,
//!   `g < 0` backward, `g = 0` isotropic.  Inputs are clamped just inside the
//!   open interval so the phase denominator stays strictly positive.
//! * `cos_theta` is the cosine of the angle between the incoming and outgoing
//!   scattering directions; it is clamped to `[-1, 1]`.
//! * All coefficients and distances are clamped non-negative; transmittance is
//!   always in `[0, 1]` and every result is finite (never `NaN`).
//! * Spectral quantities are linear-RGB [`Vec3`]s matching the GPU twin.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Reciprocal of `4π`, the isotropic phase-function value.
const INV_4PI: f32 = 1.0 / (4.0 * PI);

/// Largest magnitude the anisotropy `g` may take; keeps `1 + g^2 - 2 g c > 0`.
const MAX_G: f32 = 1.0 - 1.0e-4;

/// Henyey-Greenstein phase function `p(cos_theta, g)`.
///
/// Returns
/// `(1 - g^2) / (4π * (1 + g^2 - 2 g * cos_theta)^{3/2})`, the normalised
/// angular distribution of scattered radiance (its integral over the sphere is
/// `1`).  `g` is clamped to `(-1, 1)` and `cos_theta` to `[-1, 1]`; as
/// `g → 0` the result collapses to the isotropic `1 / 4π`.
#[inline]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let cos_theta = clamp_finite(cos_theta, -1.0, 1.0);
    let g = clamp_finite(g, -MAX_G, MAX_G);
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    // `denom` is strictly positive for |g| < 1, but guard against round-off.
    let denom = denom.max(1.0e-12);
    let value = (1.0 - g * g) * INV_4PI / ops::powf(denom, 1.5);
    if value.is_finite() {
        value.max(0.0)
    } else {
        INV_4PI
    }
}

/// Beer-Lambert transmittance `T = exp(-sigma_t * d)` for scalar extinction.
///
/// `sigma_t` and `d` are clamped non-negative, so the optical depth is
/// non-negative and `T` lies in `[0, 1]` with `T(0) = 1` and `T` decreasing
/// monotonically as either factor grows.
#[inline]
pub fn beer_lambert(sigma_t: f32, distance: f32) -> f32 {
    let tau = optical_depth(sigma_t, distance);
    // Clamp the exponent to avoid denormal underflow surprises on the twin.
    let t = ops::exp(-tau.min(80.0));
    t.clamp(0.0, 1.0)
}

/// Spectral Beer-Lambert transmittance: per-channel `exp(-sigma_t * d)`.
///
/// Each RGB channel of `sigma_t` is treated independently; every component of
/// the result is in `[0, 1]`.
#[inline]
pub fn beer_lambert_rgb(sigma_t: Vec3, distance: f32) -> Vec3 {
    Vec3::new(
        beer_lambert(sigma_t.x, distance),
        beer_lambert(sigma_t.y, distance),
        beer_lambert(sigma_t.z, distance),
    )
}

/// Optical depth `tau = sigma_t * d` (dimensionless), clamped non-negative.
#[inline]
pub fn optical_depth(sigma_t: f32, distance: f32) -> f32 {
    let sigma_t = clamp_non_negative(sigma_t);
    let distance = clamp_non_negative(distance);
    let tau = sigma_t * distance;
    if tau.is_finite() {
        tau.max(0.0)
    } else {
        0.0
    }
}

/// Differential in-scattered radiance at a point: `sigma_s * p(cos_theta, g) *
/// radiance`.
///
/// This is the integrand of the volume rendering equation's in-scattering term
/// (before transmittance weighting): the fraction of `radiance` arriving from
/// the sampled direction that is redirected into the view ray.  `sigma_s` is
/// clamped non-negative; the result is non-negative per channel.
#[inline]
pub fn in_scattering(sigma_s: f32, cos_theta: f32, g: f32, radiance: Vec3) -> Vec3 {
    let sigma_s = clamp_non_negative(sigma_s);
    let phase = henyey_greenstein(cos_theta, g);
    let scale = sigma_s * phase;
    sanitize_rgb(radiance * scale)
}

/// Closed-form single-scattering radiance along a homogeneous slab of length
/// `distance`.
///
/// Integrating the in-scattered, transmittance-weighted radiance of a constant
/// source `radiance` over the ray yields
/// `sigma_s * p * L * (1 - exp(-sigma_t * d)) / sigma_t`.  As `sigma_t → 0` the
/// factor tends to `d` (the optically-thin limit), which this function returns
/// directly to stay finite.  `sigma_s`, `sigma_t`, and `distance` are clamped
/// non-negative.
#[inline]
pub fn single_scattering_homogeneous(
    sigma_s: f32,
    sigma_t: f32,
    cos_theta: f32,
    g: f32,
    radiance: Vec3,
    distance: f32,
) -> Vec3 {
    let sigma_s = clamp_non_negative(sigma_s);
    let sigma_t = clamp_non_negative(sigma_t);
    let distance = clamp_non_negative(distance);
    let phase = henyey_greenstein(cos_theta, g);
    // ∫_0^d exp(-sigma_t t) dt = (1 - exp(-sigma_t d)) / sigma_t, with the
    // optically-thin limit d as sigma_t -> 0.
    let path = if sigma_t > 1.0e-6 {
        (1.0 - beer_lambert(sigma_t, distance)) / sigma_t
    } else {
        distance
    };
    let scale = sigma_s * phase * path;
    sanitize_rgb(radiance * scale)
}

/// Geometric-series multiple-scattering gain applied to a single-scattering
/// estimate.
///
/// Approximates higher-order scattering as a geometric series of `octaves`
/// bounces, each attenuated by the single-scattering `albedo`
/// (`sigma_s / sigma_t`): the total is
/// `single * (1 - albedo^octaves) / (1 - albedo)`.  `albedo` is clamped to
/// `[0, 1)` so the series converges; `octaves = 1` returns `single` unchanged
/// and the limit as `octaves → ∞` is `single / (1 - albedo)`.
#[inline]
pub fn multiple_scattering_octaves(single: Vec3, albedo: f32, octaves: u32) -> Vec3 {
    if octaves <= 1 {
        return sanitize_rgb(single);
    }
    let albedo = clamp_finite(albedo, 0.0, 1.0 - 1.0e-4);
    let n = octaves as f32;
    // sum_{k=0}^{octaves-1} albedo^k = (1 - albedo^octaves) / (1 - albedo).
    let gain = (1.0 - ops::powf(albedo, n)) / (1.0 - albedo);
    let gain = if gain.is_finite() { gain.max(1.0) } else { 1.0 };
    sanitize_rgb(single * gain)
}

/// Single-scattering albedo `sigma_s / sigma_t`, clamped to `[0, 1]`.
///
/// Returns `0` for a non-positive or non-finite extinction, matching an opaque
/// / absorption-only medium.
#[inline]
pub fn albedo(sigma_s: f32, sigma_t: f32) -> f32 {
    let sigma_s = clamp_non_negative(sigma_s);
    let sigma_t = clamp_non_negative(sigma_t);
    if sigma_t > 1.0e-12 {
        (sigma_s / sigma_t).clamp(0.0, 1.0)
    } else {
        0.0
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

/// Clamps `value` to be non-negative and finite.
#[inline]
fn clamp_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Replaces any non-finite channel with `0` and clamps every channel
/// non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed Gauss-Legendre-free midpoint quadrature of a function of
    /// `cos_theta` over the sphere: `∫_sphere f dω = 2π ∫_{-1}^{1} f dμ`.
    fn sphere_integral(f: impl Fn(f32) -> f32) -> f32 {
        let steps = 20_000;
        let mut acc = 0.0f64;
        for i in 0..steps {
            let mu = -1.0 + 2.0 * (i as f32 + 0.5) / steps as f32;
            acc += f(mu) as f64;
        }
        let dmu = 2.0 / steps as f32;
        (2.0 * PI as f64 * acc * dmu as f64) as f32
    }

    #[test]
    fn hg_integrates_to_unity_for_several_g() {
        for g in [-0.7f32, -0.3, 0.0, 0.25, 0.6, 0.9] {
            let integral = sphere_integral(|mu| henyey_greenstein(mu, g));
            assert!((integral - 1.0).abs() < 2e-3, "g={g} integral={integral}");
        }
    }

    #[test]
    fn hg_degenerates_to_isotropic_at_zero_g() {
        for mu in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let p = henyey_greenstein(mu, 0.0);
            assert!((p - INV_4PI).abs() < 1e-6, "mu={mu} p={p}");
        }
    }

    #[test]
    fn hg_forward_vs_backward_peak() {
        // Forward g > 0 peaks towards cos_theta = 1; backward g < 0 towards -1.
        let g = 0.6;
        let fwd = henyey_greenstein(1.0, g);
        let bwd = henyey_greenstein(-1.0, g);
        assert!(fwd > bwd);
        // Mirror symmetry: p(c, g) == p(-c, -g).
        for c in [-0.8f32, -0.2, 0.4, 0.9] {
            let a = henyey_greenstein(c, g);
            let b = henyey_greenstein(-c, -g);
            assert!((a - b).abs() < 1e-5, "c={c} a={a} b={b}");
        }
    }

    #[test]
    fn hg_clamps_extreme_anisotropy() {
        // |g| >= 1 must not blow up to NaN/inf.
        for g in [-5.0f32, -1.0, 1.0, 5.0, f32::NAN] {
            for c in [-1.0f32, 0.0, 1.0] {
                let p = henyey_greenstein(c, g);
                assert!(p.is_finite() && p >= 0.0, "g={g} c={c} p={p}");
            }
        }
    }

    #[test]
    fn beer_lambert_monotonic_and_bounded() {
        assert!((beer_lambert(1.0, 0.0) - 1.0).abs() < 1e-6);
        let mut prev = 1.0;
        for d in 1..=20 {
            let t = beer_lambert(0.5, d as f32);
            assert!(t <= prev + 1e-6, "not monotonic at d={d}");
            assert!((0.0..=1.0).contains(&t));
            prev = t;
        }
        // Analytic value check.
        let t = beer_lambert(2.0, 3.0);
        assert!((t - ops::exp(-6.0)).abs() < 1e-6, "t={t}");
    }

    #[test]
    fn beer_lambert_rgb_is_per_channel() {
        let t = beer_lambert_rgb(Vec3::new(0.0, 1.0, 2.0), 1.0);
        assert!((t.x - 1.0).abs() < 1e-6);
        assert!((t.y - ops::exp(-1.0)).abs() < 1e-6);
        assert!((t.z - ops::exp(-2.0)).abs() < 1e-6);
    }

    #[test]
    fn single_scattering_matches_numeric_integral() {
        let sigma_s = 0.4;
        let sigma_t = 0.9;
        let g = 0.3;
        let cos_theta = 0.5;
        let radiance = Vec3::new(2.0, 1.0, 0.5);
        let distance = 4.0;
        let analytic =
            single_scattering_homogeneous(sigma_s, sigma_t, cos_theta, g, radiance, distance);

        // Numerically integrate sigma_s * p * L * exp(-sigma_t * t) dt.
        let phase = henyey_greenstein(cos_theta, g);
        let steps = 20_000;
        let mut acc = 0.0f64;
        for i in 0..steps {
            let t = (i as f32 + 0.5) / steps as f32 * distance;
            acc += (sigma_s * phase * ops::exp(-sigma_t * t)) as f64;
        }
        let dt = distance / steps as f32;
        let factor = (acc * dt as f64) as f32;
        let numeric = radiance * factor;
        assert!((analytic - numeric).abs().max_element() < 2e-3, "a={analytic} n={numeric}");
    }

    #[test]
    fn single_scattering_thin_limit_is_linear_in_distance() {
        // sigma_t -> 0: single scattering -> sigma_s * p * L * d.
        let out = single_scattering_homogeneous(0.5, 0.0, 0.0, 0.0, Vec3::ONE, 3.0);
        let expected = 0.5 * INV_4PI * 3.0;
        assert!((out.x - expected).abs() < 1e-5, "out={out} exp={expected}");
    }

    #[test]
    fn in_scattering_is_phase_weighted() {
        let out = in_scattering(2.0, 0.0, 0.0, Vec3::new(1.0, 2.0, 4.0));
        let expected = Vec3::new(1.0, 2.0, 4.0) * (2.0 * INV_4PI);
        assert!((out - expected).abs().max_element() < 1e-6, "out={out}");
    }

    #[test]
    fn multiple_scattering_grows_with_octaves_towards_limit() {
        let single = Vec3::new(1.0, 1.0, 1.0);
        let albedo = 0.5;
        let mut prev = multiple_scattering_octaves(single, albedo, 1);
        assert!((prev.x - 1.0).abs() < 1e-6);
        for n in 2..=12 {
            let cur = multiple_scattering_octaves(single, albedo, n);
            assert!(cur.x >= prev.x - 1e-6, "not increasing at n={n}");
            prev = cur;
        }
        // Converges to single / (1 - albedo) = 2.
        let limit = multiple_scattering_octaves(single, albedo, 64);
        assert!((limit.x - 2.0).abs() < 1e-3, "limit={limit}");
    }

    #[test]
    fn albedo_is_bounded_and_guarded() {
        assert!((albedo(0.5, 1.0) - 0.5).abs() < 1e-6);
        assert_eq!(albedo(1.0, 0.0), 0.0);
        assert!((albedo(5.0, 1.0) - 1.0).abs() < 1e-6);
        assert_eq!(albedo(f32::NAN, 1.0), 0.0);
    }

    #[test]
    fn no_nan_on_degenerate_scattering_inputs() {
        let out = single_scattering_homogeneous(
            f32::NAN,
            f32::INFINITY,
            f32::NAN,
            f32::NAN,
            Vec3::splat(f32::NAN),
            f32::NAN,
        );
        assert!(out.is_finite());
        assert!(beer_lambert(f32::NAN, f32::NAN).is_finite());
    }
}
