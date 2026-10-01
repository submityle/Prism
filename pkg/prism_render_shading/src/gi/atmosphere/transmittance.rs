//! Ray/shell intersection and spectral transmittance integration.
//!
//! Transmittance `T = exp(-∫ σ_t ds)` is the fraction of radiance that survives
//! travel through the atmosphere between two points. Because the extinction
//! `σ_t` varies with altitude, the optical depth `∫ σ_t ds` is evaluated by
//! deterministic midpoint ray-march quadrature along the segment.
//!
//! Rays are intersected against the planet-centred spherical shells (ground and
//! atmosphere top) by solving the quadratic `|O + tD|² = r²`, with careful
//! fallbacks for the no-intersection and negative-discriminant cases so the
//! marcher never walks past a boundary or returns `NaN`.
//!
//! * [`ray_sphere_intersections`] — both signed roots of a ray/sphere hit.
//! * [`nearest_positive_intersection`] — the closest hit strictly ahead of the
//!   origin.
//! * [`distance_to_boundary`] — distance to the first shell the ray leaves the
//!   atmosphere through (top exit or ground).
//! * [`transmittance_to_boundary`] — spectral `T` from a point to that boundary
//!   along a direction.
//! * [`transmittance_between`] — spectral `T` between two explicit points.
//!
//! # Conventions
//! * Positions are planet-centred in kilometres; directions are normalised
//!   internally (a zero direction yields an empty / zero-length march).
//! * Transmittance is spectral linear-RGB in `(0, 1]` with `T(0) = 1`, and
//!   decreases monotonically with optical depth.
//! * Every result is finite (never `NaN`); negative discriminants and missing
//!   intersections fall back to "no hit".
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   value method. Every function is a deterministic pure function with no RNG,
//!   I/O, GPU, or `unsafe`.

use bevy_math::{ops, Vec3};

use super::medium::Atmosphere;

/// Minimum positive distance treated as "strictly ahead" of the origin (km).
const T_EPSILON: f32 = 1.0e-4;

/// Signed roots `(t0, t1)` with `t0 ≤ t1` of the ray/sphere intersection
/// `|origin + t·dir|² = radius²`.
///
/// `dir` is normalised internally. Returns [`None`] when the ray misses the
/// sphere (negative discriminant) or when `dir`/`radius` are degenerate. Roots
/// may be negative (intersection behind the origin).
#[inline]
pub fn ray_sphere_intersections(origin: Vec3, dir: Vec3, radius: f32) -> Option<(f32, f32)> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO || !(radius > 0.0) || !origin.is_finite() {
        return None;
    }
    // t² + 2b t + c = 0, with b = O·D and c = |O|² - r².
    let b = origin.dot(dir);
    let c = origin.length_squared() - radius * radius;
    let disc = b * b - c;
    if !disc.is_finite() || disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t0 = -b - sq;
    let t1 = -b + sq;
    if t0.is_finite() && t1.is_finite() {
        Some((t0, t1))
    } else {
        None
    }
}

/// Smallest intersection distance strictly ahead of the origin, if any.
///
/// Returns the nearest root `> T_EPSILON`, i.e. the first time the forward ray
/// crosses the sphere. [`None`] when the ray misses or only hits behind the
/// origin.
#[inline]
pub fn nearest_positive_intersection(origin: Vec3, dir: Vec3, radius: f32) -> Option<f32> {
    let (t0, t1) = ray_sphere_intersections(origin, dir, radius)?;
    if t0 > T_EPSILON {
        Some(t0)
    } else if t1 > T_EPSILON {
        Some(t1)
    } else {
        None
    }
}

/// Distance from `origin` along `dir` to the first atmosphere boundary.
///
/// The ray leaves the atmosphere either by hitting the ground
/// (`bottom_radius`) or by exiting the top shell (`top_radius`); this returns
/// whichever comes first ahead of the origin. Returns `0` for a degenerate ray
/// that intersects nothing ahead.
#[inline]
pub fn distance_to_boundary(atmosphere: &Atmosphere, origin: Vec3, dir: Vec3) -> f32 {
    let top = top_exit_distance(atmosphere, origin, dir);
    match nearest_positive_intersection(origin, dir, atmosphere.bottom_radius) {
        Some(ground) if ground < top => ground,
        _ => top,
    }
}

/// Distance to the far side of the atmosphere-top shell ahead of the origin.
///
/// For a point inside the shell the near root is behind the origin, so the far
/// (positive) root is the exit. Returns `0` when the ray never meets the top
/// shell ahead (degenerate).
#[inline]
fn top_exit_distance(atmosphere: &Atmosphere, origin: Vec3, dir: Vec3) -> f32 {
    match ray_sphere_intersections(origin, dir, atmosphere.top_radius) {
        Some((t0, t1)) => {
            if t1 > T_EPSILON {
                t1
            } else if t0 > T_EPSILON {
                t0
            } else {
                0.0
            }
        }
        None => 0.0,
    }
}

/// Spectral transmittance from `origin` along `dir` to the first atmosphere
/// boundary, via `samples` midpoint quadrature steps.
///
/// Each channel is `exp(-∫ σ_t ds)` and lies in `(0, 1]`.
#[inline]
pub fn transmittance_to_boundary(
    atmosphere: &Atmosphere,
    origin: Vec3,
    dir: Vec3,
    samples: u32,
) -> Vec3 {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return Vec3::ONE;
    }
    let distance = distance_to_boundary(atmosphere, origin, dir);
    let tau = optical_depth(atmosphere, origin, dir, distance, samples);
    exp_neg(tau)
}

/// Spectral transmittance between two explicit points `a` and `b`, via
/// `samples` midpoint quadrature steps.
///
/// Each channel is `exp(-∫_a^b σ_t ds)` in `(0, 1]`; coincident points give
/// `T = 1`.
#[inline]
pub fn transmittance_between(atmosphere: &Atmosphere, a: Vec3, b: Vec3, samples: u32) -> Vec3 {
    let delta = b - a;
    let distance = delta.length();
    if !(distance > 0.0) || !distance.is_finite() {
        return Vec3::ONE;
    }
    let dir = delta / distance;
    let tau = optical_depth(atmosphere, a, dir, distance, samples);
    exp_neg(tau)
}

/// Spectral optical depth `∫_0^distance σ_t(origin + t·dir) dt` via midpoint
/// quadrature. `dir` is assumed normalised.
#[inline]
fn optical_depth(
    atmosphere: &Atmosphere,
    origin: Vec3,
    dir: Vec3,
    distance: f32,
    samples: u32,
) -> Vec3 {
    if !(distance > 0.0) || !distance.is_finite() {
        return Vec3::ZERO;
    }
    let steps = samples.max(1);
    let ds = distance / steps as f32;
    let mut tau = Vec3::ZERO;
    for i in 0..steps {
        let t = (i as f32 + 0.5) * ds;
        let pos = origin + dir * t;
        let altitude = atmosphere.altitude_at(pos);
        tau += atmosphere.extinction(altitude) * ds;
    }
    sanitize_rgb(tau)
}

/// Per-channel `exp(-tau)` clamped to `[0, 1]`, with large exponents saturated
/// to avoid denormal underflow surprises on the GPU twin.
#[inline]
fn exp_neg(tau: Vec3) -> Vec3 {
    Vec3::new(
        channel_exp_neg(tau.x),
        channel_exp_neg(tau.y),
        channel_exp_neg(tau.z),
    )
}

/// Scalar `exp(-tau)` clamped to `[0, 1]`.
#[inline]
fn channel_exp_neg(tau: f32) -> f32 {
    let tau = if tau.is_finite() { tau.max(0.0) } else { 0.0 };
    let value = ops::exp(-tau.min(80.0));
    value.clamp(0.0, 1.0)
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

    #[test]
    fn ray_sphere_known_roots() {
        // Origin 5 units out on -Z, aimed at +Z through a unit sphere: hits
        // at t = 4 and t = 6.
        let hit = ray_sphere_intersections(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, 1.0).unwrap();
        assert!((hit.0 - 4.0).abs() < 1e-5, "t0={}", hit.0);
        assert!((hit.1 - 6.0).abs() < 1e-5, "t1={}", hit.1);
        // Tangent / miss returns None.
        assert!(ray_sphere_intersections(Vec3::new(0.0, 2.0, -5.0), Vec3::Z, 1.0).is_none());
    }

    #[test]
    fn nearest_positive_skips_hits_behind_origin() {
        // Inside the sphere: one root behind, one ahead.
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let t = nearest_positive_intersection(origin, Vec3::Z, 2.0).unwrap();
        assert!((t - 2.0).abs() < 1e-5, "t={t}");
        // Pointing away from a sphere entirely behind: no positive hit.
        let behind = nearest_positive_intersection(Vec3::new(0.0, 0.0, -5.0), -Vec3::Z, 1.0);
        assert!(behind.is_none());
    }

    #[test]
    fn sun_above_horizon_is_unoccluded_but_below_hits_ground() {
        let a = Atmosphere::earth();
        let p = Vec3::new(0.0, a.bottom_radius + 1.0, 0.0);
        // Straight up: no ground intersection ahead.
        assert!(nearest_positive_intersection(p, Vec3::Y, a.bottom_radius).is_none());
        // Straight down: hits the ground ahead.
        let down = nearest_positive_intersection(p, -Vec3::Y, a.bottom_radius).unwrap();
        assert!((down - 1.0).abs() < 1e-2, "down={down}");
    }

    #[test]
    fn transmittance_is_unit_at_zero_distance_and_bounded() {
        let a = Atmosphere::earth();
        let p = Vec3::new(0.0, a.bottom_radius, 0.0);
        let t0 = transmittance_between(&a, p, p, 16);
        assert!((t0 - Vec3::ONE).length() < 1e-6, "t0={t0:?}");
        let t = transmittance_to_boundary(&a, p, Vec3::Y, 64);
        assert!(t.min_element() > 0.0 && t.max_element() <= 1.0, "t={t:?}");
    }

    #[test]
    fn transmittance_decreases_monotonically_with_distance() {
        let a = Atmosphere::earth();
        let origin = Vec3::new(0.0, a.bottom_radius, 0.0);
        let dir = Vec3::Y;
        let mut prev = Vec3::ONE;
        for d in 1..=40 {
            let b = origin + dir * (d as f32);
            let t = transmittance_between(&a, origin, b, 128);
            assert!(t.x <= prev.x + 1e-6, "not monotonic at d={d}: {t:?}");
            assert!(t.min_element() > 0.0 && t.max_element() <= 1.0 + 1e-6);
            prev = t;
        }
    }

    #[test]
    fn zenith_transmittance_matches_analytic_rayleigh() {
        // Mie- and ozone-free atmosphere: a zenith ray from the ground has the
        // closed-form optical depth tau = sigma_s * H * (1 - exp(-thickness/H)).
        let mut a = Atmosphere::earth();
        a.mie_scattering = 0.0;
        a.mie_extinction = 0.0;
        a.ozone_absorption = Vec3::ZERO;
        let origin = Vec3::new(0.0, a.bottom_radius, 0.0);
        let numeric = transmittance_to_boundary(&a, origin, Vec3::Y, 4096);

        let h = a.rayleigh_scale_height;
        let column = h * (1.0 - ops::exp(-a.thickness() / h));
        let tau = a.rayleigh_scattering * column;
        let analytic = Vec3::new(ops::exp(-tau.x), ops::exp(-tau.y), ops::exp(-tau.z));
        assert!((numeric - analytic).length() < 2e-3, "num={numeric:?} ana={analytic:?}");
    }

    #[test]
    fn degenerate_rays_never_produce_nan() {
        let a = Atmosphere::earth();
        let p = Vec3::new(0.0, a.bottom_radius + 5.0, 0.0);
        // Zero direction -> full transmittance, no NaN.
        assert_eq!(transmittance_to_boundary(&a, p, Vec3::ZERO, 16), Vec3::ONE);
        // NaN origin stays finite.
        let t = transmittance_to_boundary(&a, Vec3::splat(f32::NAN), Vec3::Y, 16);
        assert!(t.is_finite());
        assert!(ray_sphere_intersections(p, Vec3::ZERO, 1.0).is_none());
        assert!(distance_to_boundary(&a, p, Vec3::Y).is_finite());
    }
}
