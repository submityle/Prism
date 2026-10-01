//! Christensen-Burley normalized subsurface-scattering diffusion profile — the
//! CPU golden reference for the engine's screen-space / world-space diffusion
//! blur passes.
//!
//! Subsurface light transport is modelled here with the *normalized diffusion*
//! approximation of Christensen & Burley (2015), "Approximate Reflectance
//! Profiles for Efficient Subsurface Scattering".  The radial reflectance
//! profile of a flat, semi-infinite scattering slab is approximated by a sum of
//! two exponentials,
//!
//! ```text
//!   R(r) = (e^{-r/d} + e^{-r/(3d)}) / (8 π d r),
//! ```
//!
//! whose radial integral `∫_0^∞ 2π r R(r) dr = 1` is energy-conserving.  The
//! scaling length `d` is derived from a user-facing *diffuse mean free path*
//! (dmfp) and the surface albedo through Burley's shaping fit `s(A)`, so that
//! `d = dmfp / s(A)`.  Scaling the profile by the per-channel albedo recovers
//! the fraction of incident light that re-emerges at radial distance `r`.
//!
//! The profile doubles as a sampling distribution: its normalised radial pdf
//! `p(r) = 2π r R(r)` is a two-component exponential mixture, so radii can be
//! drawn with an *exact analytic inverse CDF* (component pick + exponential
//! inversion) rather than a numeric root find.  This is what the separable /
//! importance-sampled diffusion kernels on the GPU twin reproduce.
//!
//! # Conventions
//! * `albedo` is the per-channel diffuse surface albedo in `[0, 1]`; it is
//!   clamped to that range.  `dmfp` is the diffuse mean free path in world
//!   units and is clamped non-negative.
//! * `d` is the normalized-diffusion scaling length (`d = dmfp / s`), clamped
//!   strictly positive so the profile stays finite away from the origin.
//! * The profile `R(r)` is energy-normalized (unit radial integral); scaling by
//!   `albedo` yields the physically meaningful reflectance that integrates to
//!   `albedo`.  `R(r)` is singular as `r → 0` (integrably, since `2π r R(r)`
//!   stays finite); the radius is clamped to a tiny epsilon so the evaluated
//!   value is always finite.
//! * Spectral quantities are linear-RGB [`Vec3`]s matching the GPU twin.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// `1 / (8π)`, the leading constant of the normalized diffusion profile.
const INV_8PI: f32 = 1.0 / (8.0 * PI);

/// Smallest radius used when evaluating the profile, guarding the `1/r` term.
const MIN_RADIUS: f32 = 1.0e-6;

/// Smallest scaling length `d`, keeping the profile finite and invertible.
const MIN_SCALING: f32 = 1.0e-6;

/// Burley's shaping parameter `s(A)` for the normalized diffusion profile.
///
/// Uses the "searchlight" fit `s = 1.9 - A + 3.5 (A - 0.8)^2` from
/// Christensen-Burley (2015), which maps a surface albedo `A ∈ [0, 1]` to the
/// dimensionless shape factor relating the diffuse mean free path to the
/// profile's scaling length (`d = dmfp / s`).  `albedo` is clamped to `[0, 1]`
/// and the returned `s` is strictly positive.
#[inline]
pub fn burley_shaping(albedo: f32) -> f32 {
    let a = clamp_finite(albedo, 0.0, 1.0);
    let s = 1.9 - a + 3.5 * (a - 0.8) * (a - 0.8);
    if s.is_finite() {
        s.max(MIN_SCALING)
    } else {
        1.0
    }
}

/// Per-channel [`burley_shaping`] over a linear-RGB albedo.
#[inline]
pub fn burley_shaping_rgb(albedo: Vec3) -> Vec3 {
    Vec3::new(
        burley_shaping(albedo.x),
        burley_shaping(albedo.y),
        burley_shaping(albedo.z),
    )
}

/// Normalized-diffusion scaling length `d = dmfp / s(albedo)`.
///
/// Converts a user-facing diffuse mean free path `dmfp` (world units) and a
/// surface `albedo` into the exponential scaling length that parameterises
/// [`profile`].  `dmfp` is clamped non-negative; the result is clamped strictly
/// positive.
#[inline]
pub fn burley_scaling(dmfp: f32, albedo: f32) -> f32 {
    let dmfp = clamp_non_negative(dmfp);
    let s = burley_shaping(albedo);
    let d = dmfp / s;
    if d.is_finite() {
        d.max(MIN_SCALING)
    } else {
        MIN_SCALING
    }
}

/// Per-channel [`burley_scaling`] over linear-RGB `dmfp` and `albedo`.
#[inline]
pub fn burley_scaling_rgb(dmfp: Vec3, albedo: Vec3) -> Vec3 {
    Vec3::new(
        burley_scaling(dmfp.x, albedo.x),
        burley_scaling(dmfp.y, albedo.y),
        burley_scaling(dmfp.z, albedo.z),
    )
}

/// Energy-normalized Burley diffusion profile `R(r)` at radius `r`.
///
/// Returns `(e^{-r/d} + e^{-r/(3d)}) / (8π d r)`, whose radial integral
/// `∫_0^∞ 2π r R(r) dr = 1`.  The radius is clamped to a tiny positive epsilon
/// so the `1/r` singularity evaluates to a large but finite value; `d` is
/// clamped strictly positive.  The result is non-negative and finite.
#[inline]
pub fn profile(r: f32, d: f32) -> f32 {
    let d = clamp_scaling(d);
    let r = clamp_non_negative(r).max(MIN_RADIUS);
    let inv_d = 1.0 / d;
    let value = (ops::exp(-r * inv_d) + ops::exp(-r * inv_d / 3.0)) * INV_8PI / (d * r);
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Albedo-weighted profile `A · R(r)`.
///
/// This is the physically meaningful reflectance whose radial integral equals
/// the diffuse `albedo`.  `albedo` is clamped to `[0, 1]`.
#[inline]
pub fn profile_albedo(r: f32, d: f32, albedo: f32) -> f32 {
    let a = clamp_finite(albedo, 0.0, 1.0);
    (a * profile(r, d)).max(0.0)
}

/// Per-channel albedo-weighted profile over linear-RGB `d` and `albedo`.
///
/// Each channel evaluates [`profile_albedo`] with its own scaling length and
/// albedo, so coloured subsurface materials (e.g. the reddish bleed of skin)
/// are reproduced exactly.
#[inline]
pub fn profile_rgb(r: f32, d: Vec3, albedo: Vec3) -> Vec3 {
    Vec3::new(
        profile_albedo(r, d.x, albedo.x),
        profile_albedo(r, d.y, albedo.y),
        profile_albedo(r, d.z, albedo.z),
    )
}

/// Normalized radial pdf `p(r) = 2π r R(r)` of the diffusion profile.
///
/// Equal to `(e^{-r/d} + e^{-r/(3d)}) / (4 d)`, a unit-integral density over
/// `r ∈ [0, ∞)`.  Unlike [`profile`] this is finite at `r = 0`
/// (`p(0) = 1 / (2d)`), making it the quantity the importance sampler targets.
/// `r` is clamped non-negative and `d` strictly positive.
#[inline]
pub fn radius_pdf(r: f32, d: f32) -> f32 {
    let d = clamp_scaling(d);
    let r = clamp_non_negative(r);
    let inv_d = 1.0 / d;
    let value = (ops::exp(-r * inv_d) + ops::exp(-r * inv_d / 3.0)) / (4.0 * d);
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Radial CDF `F(r) = 1 - ¼ e^{-r/d} - ¾ e^{-r/(3d)}` of the diffusion profile.
///
/// This is the exact integral of [`radius_pdf`] from `0` to `r`: `F(0) = 0` and
/// `F(r) → 1` as `r → ∞`, monotonically increasing in between.  `r` is clamped
/// non-negative and `d` strictly positive; the result lies in `[0, 1]`.
#[inline]
pub fn radius_cdf(r: f32, d: f32) -> f32 {
    let d = clamp_scaling(d);
    let r = clamp_non_negative(r);
    let inv_d = 1.0 / d;
    let value = 1.0 - 0.25 * ops::exp(-r * inv_d) - 0.75 * ops::exp(-r * inv_d / 3.0);
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Importance-samples a radius from the diffusion profile via its exact analytic
/// inverse CDF.
///
/// The radial pdf is the two-component exponential mixture
/// `p(r) = ¼ · (1/d) e^{-r/d} + ¾ · (1/(3d)) e^{-r/(3d)}`, so a radius is drawn
/// by first picking a component with `u_select` (weight `¼` for scale `d`,
/// weight `¾` for scale `3d`) and then inverting that exponential with
/// `r = -scale · ln(1 - u_radius)`.  The returned radius is distributed exactly
/// according to [`radius_pdf`] — no numerical root finding is required.
///
/// Both `u_select` and `u_radius` are expected in `[0, 1)` (as produced by
/// [`crate::gi::sample::low_discrepancy_sample_2d`]) and are clamped into
/// `[0, 1)` defensively; `d` is clamped strictly positive.  The result is
/// non-negative and finite.
#[inline]
pub fn sample_radius(d: f32, u_select: f32, u_radius: f32) -> f32 {
    let d = clamp_scaling(d);
    let u_select = clamp_finite(u_select, 0.0, 1.0);
    // Keep `1 - u` strictly positive so `ln` never sees zero.
    let u = clamp_finite(u_radius, 0.0, 1.0 - 1.0e-7);
    let scale = if u_select < 0.25 { d } else { 3.0 * d };
    let r = -scale * ops::ln(1.0 - u);
    if r.is_finite() {
        r.max(0.0)
    } else {
        0.0
    }
}

/// Convenience wrapper feeding a 2-D low-discrepancy sample into [`sample_radius`].
///
/// `uv.0` selects the mixture component and `uv.1` inverts the exponential,
/// matching the `(u, v)` layout returned by the shared GI samplers.
#[inline]
pub fn sample_radius_2d(d: f32, uv: (f32, f32)) -> f32 {
    sample_radius(d, uv.0, uv.1)
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

/// Clamps a scaling length to be finite and strictly positive.
#[inline]
fn clamp_scaling(value: f32) -> f32 {
    if value.is_finite() {
        value.max(MIN_SCALING)
    } else {
        MIN_SCALING
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::sample::low_discrepancy_sample_2d;

    /// Midpoint quadrature of `2π r R(r)` over `[0, r_max]`.
    fn radial_integral(f: impl Fn(f32) -> f32, r_max: f32) -> f32 {
        let steps = 200_000;
        let dr = r_max / steps as f32;
        let mut acc = 0.0f64;
        for i in 0..steps {
            let r = (i as f32 + 0.5) * dr;
            acc += (2.0 * PI * r * f(r)) as f64;
        }
        (acc * dr as f64) as f32
    }

    #[test]
    fn profile_integrates_to_unity() {
        for d in [0.1f32, 0.5, 1.0, 2.5] {
            let integral = radial_integral(|r| profile(r, d), 40.0 * d);
            assert!((integral - 1.0).abs() < 5e-3, "d={d} integral={integral}");
        }
    }

    #[test]
    fn albedo_weighted_profile_integrates_to_albedo() {
        for a in [0.2f32, 0.5, 0.9] {
            let d = 1.0;
            let integral = radial_integral(|r| profile_albedo(r, d, a), 40.0 * d);
            assert!((integral - a).abs() < 5e-3, "a={a} integral={integral}");
        }
    }

    #[test]
    fn pdf_integrates_to_unity() {
        for d in [0.25f32, 1.0, 3.0] {
            let steps = 200_000;
            let r_max = 60.0 * d;
            let dr = r_max / steps as f32;
            let mut acc = 0.0f64;
            for i in 0..steps {
                let r = (i as f32 + 0.5) * dr;
                acc += radius_pdf(r, d) as f64;
            }
            let integral = (acc * dr as f64) as f32;
            assert!((integral - 1.0).abs() < 5e-3, "d={d} integral={integral}");
        }
    }

    #[test]
    fn cdf_derivative_matches_pdf() {
        let d = 1.3;
        let h = 1.0e-3;
        for r in [0.2f32, 0.8, 1.5, 3.0, 6.0] {
            let num = (radius_cdf(r + h, d) - radius_cdf(r - h, d)) / (2.0 * h);
            let analytic = radius_pdf(r, d);
            assert!((num - analytic).abs() < 2e-3, "r={r} num={num} pdf={analytic}");
        }
    }

    #[test]
    fn cdf_is_monotonic_and_bounded() {
        let d = 0.7;
        assert!(radius_cdf(0.0, d).abs() < 1e-6);
        let mut prev = 0.0;
        for i in 1..=60 {
            let r = i as f32 * 0.5;
            let c = radius_cdf(r, d);
            assert!(c >= prev - 1e-6, "not monotonic at r={r}");
            assert!((0.0..=1.0).contains(&c));
            prev = c;
        }
        assert!(radius_cdf(10_000.0, d) > 0.999);
    }

    #[test]
    fn sampled_radius_mean_matches_analytic() {
        // E[r] for the mixture = 0.25 d + 0.75 * 3d = 2.5 d.
        let d = 1.5;
        let n = 200_000u32;
        let mut acc = 0.0f64;
        for i in 0..n {
            let (u, v) = low_discrepancy_sample_2d((7, 3), 0, i, 11);
            acc += sample_radius(d, u, v) as f64;
        }
        let mean = (acc / n as f64) as f32;
        let expected = 2.5 * d;
        assert!((mean - expected).abs() < 0.05, "mean={mean} expected={expected}");
    }

    #[test]
    fn sampled_radius_histogram_matches_cdf() {
        // Fraction of samples below a radius must track the analytic CDF.
        let d = 1.0;
        let n = 200_000u32;
        for probe in [0.5f32, 1.0, 2.0, 4.0] {
            let mut below = 0u32;
            for i in 0..n {
                let (u, v) = low_discrepancy_sample_2d((21, 9), 1, i, 5);
                if sample_radius(d, u, v) <= probe {
                    below += 1;
                }
            }
            let empirical = below as f32 / n as f32;
            let analytic = radius_cdf(probe, d);
            assert!(
                (empirical - analytic).abs() < 1e-2,
                "probe={probe} empirical={empirical} cdf={analytic}"
            );
        }
    }

    #[test]
    fn shaping_is_positive_and_matches_fit() {
        assert!((burley_shaping(0.8) - 1.1).abs() < 1e-5);
        for a in [-1.0f32, 0.0, 0.5, 1.0, 2.0, f32::NAN] {
            assert!(burley_shaping(a) > 0.0, "a={a}");
        }
    }

    #[test]
    fn scaling_is_dmfp_over_shaping() {
        let a = 0.5;
        let dmfp = 2.0;
        let d = burley_scaling(dmfp, a);
        assert!((d - dmfp / burley_shaping(a)).abs() < 1e-5, "d={d}");
    }

    #[test]
    fn rgb_helpers_are_per_channel() {
        let d = burley_scaling_rgb(Vec3::new(1.0, 2.0, 3.0), Vec3::splat(0.5));
        assert!((d.x - burley_scaling(1.0, 0.5)).abs() < 1e-6);
        assert!((d.z - burley_scaling(3.0, 0.5)).abs() < 1e-6);
        let p = profile_rgb(0.5, d, Vec3::new(0.2, 0.5, 0.9));
        assert!((p.y - profile_albedo(0.5, d.y, 0.5)).abs() < 1e-6);
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(profile(0.4, 1.2), profile(0.4, 1.2));
        assert_eq!(sample_radius(1.0, 0.3, 0.6), sample_radius(1.0, 0.3, 0.6));
        assert_eq!(sample_radius_2d(1.0, (0.3, 0.6)), sample_radius(1.0, 0.3, 0.6));
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        assert!(profile(f32::NAN, f32::NAN).is_finite());
        assert!(profile(0.0, 0.0).is_finite());
        assert!(radius_pdf(f32::INFINITY, 0.0).is_finite());
        assert!(radius_cdf(f32::NAN, f32::NAN).is_finite());
        assert!(sample_radius(0.0, f32::NAN, f32::NAN).is_finite());
        assert!(profile_rgb(f32::NAN, Vec3::ZERO, Vec3::splat(f32::NAN)).is_finite());
        assert!(burley_scaling(f32::NAN, f32::NAN).is_finite());
    }
}
