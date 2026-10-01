//! Single scattering, isotropic multiple-scattering coupling, and aerial
//! perspective — the sky-radiance integrators built on the medium, phase, and
//! transmittance primitives.
//!
//! Along a view ray the in-scattered radiance is
//! `∫ T_view(t) · T_sun(t) · (σ_s^R·p_R + σ_s^M·p_M) · E_sun dt`, where
//! `T_view` attenuates from the camera to the sample, `T_sun` attenuates sunlight
//! from the sample to the top of the atmosphere, and the bracket is the
//! altitude-dependent scattering weighted by the Rayleigh and Mie phases. This
//! is the single-scattering term of Hillaire 2020.
//!
//! Higher orders are folded in with Hillaire's isotropic multiple-scattering
//! approximation: assuming each additional bounce rescatters a fixed spectral
//! fraction `f` of the previous order, the infinite series sums geometrically to
//! `L_2 · (1 + f + f² + …) = L_2 / (1 - f)`, i.e. the extra energy beyond the
//! second order is `L_2 · f / (1 - f)` ([`multiple_scattering_contribution`]).
//!
//! * [`single_scattering`] — view-ray single-scattered sky radiance to the
//!   first boundary.
//! * [`multiple_scattering_contribution`] — the geometric higher-order gain
//!   `L_2 · f / (1 - f)`.
//! * [`multiscatter_estimate`] — Hillaire's `(L_2, f)` evaluation at a point and
//!   its coupled total `L_2 / (1 - f)`.
//! * [`aerial_perspective`] / [`AerialPerspective`] — segment-bounded
//!   in-scattering plus transmittance for compositing distant geometry.
//!
//! # Conventions
//! * Positions are planet-centred in kilometres; `view_dir`/`sun_dir` are
//!   normalised internally. `cos θ = view_dir · sun_dir`.
//! * Sunlight below the local horizon (the sun ray hits the ground) is fully
//!   shadowed (`T_sun = 0`).
//! * Radiance is spectral linear-RGB, non-negative and finite (never `NaN`);
//!   transmittance channels stay in `(0, 1]`.
//! * Transcendental maths goes through [`bevy_math::ops`]. Every function is a
//!   deterministic pure function with no RNG, I/O, GPU, or `unsafe`.

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

use super::medium::Atmosphere;
use super::phase::{cornette_shanks_phase, rayleigh_phase};
use super::transmittance::{
    distance_to_boundary, nearest_positive_intersection, transmittance_to_boundary,
};

/// Reciprocal of `4π`, the isotropic phase value used for multiple scattering.
const INV_4PI: f32 = 1.0 / (4.0 * PI);
/// Golden-angle increment for the Fibonacci-lattice sphere sampling.
const GOLDEN_ANGLE: f32 = PI * (3.0 - 2.2360679_f32); // π·(3 - √5)
/// Upper bound on the per-order rescattering fraction, keeping `1 - f > 0`.
const MAX_FRACTION: f32 = 1.0 - 1.0e-4;

/// Result of an aerial-perspective evaluation along a bounded view segment.
///
/// Composite distant radiance `L_bg` as `in_scatter + transmittance * L_bg`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AerialPerspective {
    /// Spectral in-scattered radiance accumulated over the segment.
    pub in_scatter: Vec3,
    /// Spectral transmittance across the full segment, in `(0, 1]`.
    pub transmittance: Vec3,
}

/// Single-scattered sky radiance along a view ray to the first atmosphere
/// boundary.
///
/// Marches `samples` midpoint steps from `origin` along `view_dir`, weighting
/// each sample by view transmittance, sun transmittance (`sun_samples` sub-steps
/// each, zero when the sun is below the local horizon), the altitude-dependent
/// Rayleigh/Mie scattering, and their phases toward `sun_dir`. `sun_irradiance`
/// is the top-of-atmosphere spectral solar irradiance.
#[inline]
pub fn single_scattering(
    atmosphere: &Atmosphere,
    origin: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    sun_irradiance: Vec3,
    samples: u32,
    sun_samples: u32,
) -> Vec3 {
    let dir = view_dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return Vec3::ZERO;
    }
    let distance = distance_to_boundary(atmosphere, origin, dir);
    integrate_inscatter(
        atmosphere,
        origin,
        dir,
        distance,
        sun_dir,
        sun_irradiance,
        samples,
        sun_samples,
    )
    .in_scatter
}

/// Aerial-perspective in-scattering and transmittance over a bounded view
/// segment.
///
/// Identical to [`single_scattering`] but the march stops at
/// `min(max_distance, boundary)`, and the accumulated segment transmittance is
/// returned alongside the in-scattered radiance so distant geometry can be
/// composited as `in_scatter + transmittance · L_bg`.
#[inline]
pub fn aerial_perspective(
    atmosphere: &Atmosphere,
    origin: Vec3,
    view_dir: Vec3,
    max_distance: f32,
    sun_dir: Vec3,
    sun_irradiance: Vec3,
    samples: u32,
    sun_samples: u32,
) -> AerialPerspective {
    let dir = view_dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return AerialPerspective {
            in_scatter: Vec3::ZERO,
            transmittance: Vec3::ONE,
        };
    }
    let boundary = distance_to_boundary(atmosphere, origin, dir);
    let max_distance = if max_distance.is_finite() {
        max_distance.max(0.0)
    } else {
        boundary
    };
    let distance = max_distance.min(boundary);
    integrate_inscatter(
        atmosphere,
        origin,
        dir,
        distance,
        sun_dir,
        sun_irradiance,
        samples,
        sun_samples,
    )
}

/// Higher-order multiple-scattering gain `L_2 · f / (1 - f)` (per channel).
///
/// Models Hillaire's isotropic infinite series: given the second-order
/// in-scattered radiance `l_second` and the per-order rescattering fraction
/// `f` (clamped to `[0, 1)` so the series converges), returns the summed
/// contribution of *all orders beyond the second*. The full multiple-scattering
/// radiance is `l_second + multiple_scattering_contribution(l_second, f)`.
#[inline]
pub fn multiple_scattering_contribution(l_second: Vec3, f: Vec3) -> Vec3 {
    let gain = Vec3::new(
        geometric_tail(f.x),
        geometric_tail(f.y),
        geometric_tail(f.z),
    );
    sanitize_rgb(l_second * gain)
}

/// Hillaire's point-wise multiple-scattering evaluation.
///
/// Samples `dir_samples` directions on a Fibonacci sphere about a point at
/// `altitude` (on the `+Y` axis). Along each direction it marches
/// `march_samples` steps, accumulating two spectral integrals with the
/// isotropic phase `1/4π`:
///
/// * `l_second` — second-order in-scattered radiance (sunlight scattered once
///   at the sample and re-scattered toward the point),
/// * `f` — the per-order rescattering fraction `∫ σ_s · T · (1/4π) dω ds`.
///
/// The returned [`Vec3`] is the coupled total `l_second / (1 - f)` (second order
/// plus every higher order), always finite and `≥ l_second`. `sun_cos_zenith`
/// is `cos` of the sun's zenith angle.
#[inline]
pub fn multiscatter_estimate(
    atmosphere: &Atmosphere,
    altitude: f32,
    sun_cos_zenith: f32,
    sun_irradiance: Vec3,
    dir_samples: u32,
    march_samples: u32,
) -> Vec3 {
    let (l_second, f) = multiscatter_terms(
        atmosphere,
        altitude,
        sun_cos_zenith,
        sun_irradiance,
        dir_samples,
        march_samples,
    );
    let coupling = Vec3::new(
        geometric_total(f.x),
        geometric_total(f.y),
        geometric_total(f.z),
    );
    sanitize_rgb(l_second * coupling)
}

/// Evaluates Hillaire's `(l_second, f)` pair at a point; see
/// [`multiscatter_estimate`]. Exposed for callers that need the raw terms (for
/// example to pair with [`multiple_scattering_contribution`]).
#[inline]
pub fn multiscatter_terms(
    atmosphere: &Atmosphere,
    altitude: f32,
    sun_cos_zenith: f32,
    sun_irradiance: Vec3,
    dir_samples: u32,
    march_samples: u32,
) -> (Vec3, Vec3) {
    let altitude = if altitude.is_finite() {
        altitude.clamp(0.0, atmosphere.thickness())
    } else {
        0.0
    };
    let point = Vec3::new(0.0, atmosphere.bottom_radius + altitude, 0.0);
    let c = clamp_finite(sun_cos_zenith, -1.0, 1.0);
    let s = (1.0 - c * c).max(0.0).sqrt();
    let sun_dir = Vec3::new(s, c, 0.0).normalize_or_zero();

    let n = dir_samples.max(1);
    let mut l_second = Vec3::ZERO;
    let mut f = Vec3::ZERO;
    for i in 0..n {
        let dir = fibonacci_direction(i, n);
        let (l_dir, f_dir) = march_second_order(
            atmosphere,
            point,
            dir,
            sun_dir,
            sun_irradiance,
            march_samples,
        );
        l_second += l_dir;
        f += f_dir;
    }
    // Sphere-integral weight 4π/N combined with the isotropic phase 1/4π gives
    // 1/N; the phase is already folded into the per-direction march.
    let inv_n = 1.0 / n as f32;
    (sanitize_rgb(l_second * inv_n), sanitize_rgb(f * inv_n))
}

/// Marches one direction from a point for the multiple-scattering integrals,
/// returning its `(l_second, f)` contribution (phase `1/4π` already applied).
#[inline]
fn march_second_order(
    atmosphere: &Atmosphere,
    point: Vec3,
    dir: Vec3,
    sun_dir: Vec3,
    sun_irradiance: Vec3,
    march_samples: u32,
) -> (Vec3, Vec3) {
    let distance = distance_to_boundary(atmosphere, point, dir);
    if !(distance > 0.0) {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let steps = march_samples.max(1);
    let ds = distance / steps as f32;
    let mut optical_depth = Vec3::ZERO;
    let mut l_second = Vec3::ZERO;
    let mut f = Vec3::ZERO;
    for i in 0..steps {
        let t_mid = (i as f32 + 0.5) * ds;
        let pos = point + dir * t_mid;
        let altitude = atmosphere.altitude_at(pos);
        let ext = atmosphere.extinction(altitude);
        let t_view = exp_neg(optical_depth + ext * (0.5 * ds));
        let scatter = atmosphere.scattering(altitude);
        let t_sun = sun_transmittance(atmosphere, pos, sun_dir, 8);
        // Isotropic phase 1/4π for both the gathered sunlight and the rescatter
        // probability.
        f += t_view * scatter * (INV_4PI * ds);
        l_second += t_view * scatter * t_sun * sun_irradiance * (INV_4PI * ds);
        optical_depth += ext * ds;
    }
    (sanitize_rgb(l_second), sanitize_rgb(f))
}

/// Shared view-ray in-scattering + transmittance march (`dir` normalised).
#[inline]
fn integrate_inscatter(
    atmosphere: &Atmosphere,
    origin: Vec3,
    dir: Vec3,
    distance: f32,
    sun_dir: Vec3,
    sun_irradiance: Vec3,
    samples: u32,
    sun_samples: u32,
) -> AerialPerspective {
    if !(distance > 0.0) || !distance.is_finite() {
        return AerialPerspective {
            in_scatter: Vec3::ZERO,
            transmittance: Vec3::ONE,
        };
    }
    let sun = sun_dir.normalize_or_zero();
    let cos_theta = clamp_finite(dir.dot(sun), -1.0, 1.0);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = cornette_shanks_phase(cos_theta, atmosphere.mie_g);

    let steps = samples.max(1);
    let ds = distance / steps as f32;
    let mut optical_depth = Vec3::ZERO;
    let mut in_scatter = Vec3::ZERO;
    for i in 0..steps {
        let t_mid = (i as f32 + 0.5) * ds;
        let pos = origin + dir * t_mid;
        let altitude = atmosphere.altitude_at(pos);
        let ext = atmosphere.extinction(altitude);
        let t_view = exp_neg(optical_depth + ext * (0.5 * ds));
        let rayleigh_s = atmosphere.rayleigh_scattering_at(altitude);
        let mie_s = atmosphere.mie_scattering_at(altitude);
        let scatter = rayleigh_s * phase_r + Vec3::splat(mie_s * phase_m);
        let t_sun = sun_transmittance(atmosphere, pos, sun, sun_samples);
        in_scatter += t_view * t_sun * scatter * sun_irradiance * ds;
        optical_depth += ext * ds;
    }
    AerialPerspective {
        in_scatter: sanitize_rgb(in_scatter),
        transmittance: exp_neg(optical_depth),
    }
}

/// Spectral sunlight transmittance from `pos` toward `sun_dir`, or `0` when the
/// sun ray is blocked by the planet (below the local horizon).
#[inline]
fn sun_transmittance(atmosphere: &Atmosphere, pos: Vec3, sun_dir: Vec3, samples: u32) -> Vec3 {
    let sun = sun_dir.normalize_or_zero();
    if sun == Vec3::ZERO {
        return Vec3::ZERO;
    }
    if nearest_positive_intersection(pos, sun, atmosphere.bottom_radius).is_some() {
        return Vec3::ZERO;
    }
    transmittance_to_boundary(atmosphere, pos, sun, samples)
}

/// Geometric tail `f / (1 - f)` for `f ∈ [0, 1)`, clamped and finite.
#[inline]
fn geometric_tail(f: f32) -> f32 {
    let f = clamp_finite(f, 0.0, MAX_FRACTION);
    let value = f / (1.0 - f);
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

/// Geometric total `1 / (1 - f)` for `f ∈ [0, 1)`, clamped to `≥ 1`.
#[inline]
fn geometric_total(f: f32) -> f32 {
    let f = clamp_finite(f, 0.0, MAX_FRACTION);
    let value = 1.0 / (1.0 - f);
    if value.is_finite() {
        value.max(1.0)
    } else {
        1.0
    }
}

/// Deterministic Fibonacci-lattice direction `i` of `n` on the unit sphere.
#[inline]
fn fibonacci_direction(i: u32, n: u32) -> Vec3 {
    let n = n.max(1) as f32;
    let z = 1.0 - 2.0 * (i as f32 + 0.5) / n;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = GOLDEN_ANGLE * i as f32;
    let (sin_phi, cos_phi) = ops::sin_cos(phi);
    Vec3::new(r * cos_phi, z, r * sin_phi)
}

/// Per-channel `exp(-tau)` clamped to `[0, 1]`.
#[inline]
fn exp_neg(tau: Vec3) -> Vec3 {
    Vec3::new(
        channel_exp_neg(tau.x),
        channel_exp_neg(tau.y),
        channel_exp_neg(tau.z),
    )
}

/// Scalar `exp(-tau)` clamped to `[0, 1]`, saturating large exponents.
#[inline]
fn channel_exp_neg(tau: f32) -> f32 {
    let tau = if tau.is_finite() { tau.max(0.0) } else { 0.0 };
    ops::exp(-tau.min(80.0)).clamp(0.0, 1.0)
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

    fn ground_origin(a: &Atmosphere) -> Vec3 {
        Vec3::new(0.0, a.bottom_radius + 0.5, 0.0)
    }

    #[test]
    fn single_scattering_is_finite_and_nonnegative() {
        let a = Atmosphere::earth();
        let origin = ground_origin(&a);
        let sun = Vec3::new(0.3, 0.95, 0.0);
        for view in [Vec3::Y, Vec3::new(0.5, 0.5, 0.0), Vec3::new(1.0, 0.1, 0.0)] {
            let l = single_scattering(&a, origin, view, sun, Vec3::splat(20.0), 48, 16);
            assert!(l.is_finite() && l.min_element() >= 0.0, "view={view:?} l={l:?}");
            assert!(l.max_element() > 0.0, "expected some sky radiance: {l:?}");
        }
    }

    #[test]
    fn single_scattering_scales_linearly_with_sun_irradiance() {
        let a = Atmosphere::earth();
        let origin = ground_origin(&a);
        let view = Vec3::new(0.4, 0.7, 0.0);
        let sun = Vec3::new(0.3, 0.95, 0.0);
        let base = single_scattering(&a, origin, view, sun, Vec3::splat(1.0), 48, 16);
        let scaled = single_scattering(&a, origin, view, sun, Vec3::splat(5.0), 48, 16);
        assert!((scaled - base * 5.0).length() < 1e-4, "base={base:?} scaled={scaled:?}");
    }

    #[test]
    fn sun_below_horizon_gives_no_direct_single_scatter() {
        let a = Atmosphere::earth();
        let origin = ground_origin(&a);
        // Sun pointing straight down is occluded by the planet everywhere.
        let sun = -Vec3::Y;
        let l = single_scattering(&a, origin, Vec3::Y, sun, Vec3::splat(20.0), 48, 16);
        assert!(l.max_element() < 1e-6, "expected shadowed sky: {l:?}");
    }

    #[test]
    fn aerial_perspective_transmittance_bounded_and_inscatter_grows() {
        let a = Atmosphere::earth();
        let origin = ground_origin(&a);
        let view = Vec3::new(1.0, 0.2, 0.0);
        let sun = Vec3::new(0.3, 0.95, 0.0);
        let near = aerial_perspective(&a, origin, view, 5.0, sun, Vec3::splat(20.0), 48, 12);
        let far = aerial_perspective(&a, origin, view, 40.0, sun, Vec3::splat(20.0), 48, 12);
        assert!(near.transmittance.min_element() > 0.0 && near.transmittance.max_element() <= 1.0);
        // Farther segment attenuates more and in-scatters more.
        assert!(far.transmittance.x <= near.transmittance.x + 1e-6, "{far:?} {near:?}");
        assert!(far.in_scatter.max_element() >= near.in_scatter.max_element() - 1e-6);
    }

    #[test]
    fn multiple_scattering_contribution_matches_geometric_series() {
        let l2 = Vec3::new(1.0, 2.0, 0.5);
        for f_val in [0.0f32, 0.1, 0.35, 0.6, 0.85] {
            let f = Vec3::splat(f_val);
            let got = multiple_scattering_contribution(l2, f);
            // Partial sum of l2 * (f + f^2 + ... + f^N) converges to l2*f/(1-f).
            let mut partial = 0.0f64;
            let mut term = f_val as f64;
            for _ in 0..4096 {
                partial += term;
                term *= f_val as f64;
            }
            let expected = l2 * partial as f32;
            assert!((got - expected).length() < 1e-3, "f={f_val} got={got:?} exp={expected:?}");
        }
    }

    #[test]
    fn multiple_scattering_contribution_is_zero_without_rescatter() {
        let got = multiple_scattering_contribution(Vec3::splat(3.0), Vec3::ZERO);
        assert_eq!(got, Vec3::ZERO);
    }

    #[test]
    fn multiscatter_estimate_is_at_least_second_order_and_finite() {
        let a = Atmosphere::earth();
        let (l2, f) =
            multiscatter_terms(&a, 1.0, 0.9, Vec3::splat(20.0), 64, 16);
        let total = multiscatter_estimate(&a, 1.0, 0.9, Vec3::splat(20.0), 64, 16);
        assert!(l2.is_finite() && f.is_finite() && total.is_finite());
        assert!(f.min_element() >= 0.0 && f.max_element() < 1.0, "f={f:?}");
        // Total (geometric sum) must dominate the raw second order.
        assert!(total.x >= l2.x - 1e-6, "total={total:?} l2={l2:?}");
        assert!(l2.max_element() > 0.0, "expected non-trivial second order: {l2:?}");
    }

    #[test]
    fn fibonacci_directions_are_unit_length() {
        for n in [16u32, 32, 64] {
            for i in 0..n {
                let d = fibonacci_direction(i, n);
                assert!((d.length() - 1.0).abs() < 1e-5, "n={n} i={i} d={d:?}");
            }
        }
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let a = Atmosphere::earth();
        let origin = ground_origin(&a);
        // Zero view direction -> zero radiance, no NaN.
        assert_eq!(
            single_scattering(&a, origin, Vec3::ZERO, Vec3::Y, Vec3::splat(20.0), 16, 8),
            Vec3::ZERO
        );
        let ap = aerial_perspective(
            &a,
            origin,
            Vec3::ZERO,
            f32::NAN,
            Vec3::Y,
            Vec3::splat(20.0),
            16,
            8,
        );
        assert!(ap.in_scatter.is_finite() && ap.transmittance.is_finite());
        let c = multiple_scattering_contribution(Vec3::splat(f32::NAN), Vec3::splat(2.0));
        assert!(c.is_finite());
        let m = multiscatter_estimate(&a, f32::NAN, f32::NAN, Vec3::splat(f32::NAN), 8, 4);
        assert!(m.is_finite());
    }
}
