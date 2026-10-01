//! Cloud-layer integration: shell intersection, view-ray scattering/
//! transmittance accumulation, and the secondary light march — CPU golden.
//!
//! A cloud layer occupies the spherical shell between an inner and outer radius
//! around the planet centre (reusing [`crate::gi::atmosphere`]'s ray/sphere
//! intersection). Rendering it means marching the view ray through the shell,
//! and at each step marching a short ray toward the sun to estimate how much
//! light reaches that point. Radiance and transmittance are accumulated
//! *front-to-back, premultiplied* so the integration can terminate early once
//! the ray is opaque.
//!
//! * [`shell_march_interval`] — the nearest forward `[t_near, t_far]` span of a
//!   ray inside the cloud shell, accounting for the inner "hole".
//! * [`CloudMarchParams`] — extinction, albedo, dual-lobe phase, powder, and
//!   march-resolution controls.
//! * [`light_energy`] — the secondary sun-ward march returning in-cloud light
//!   energy in `[0, 1]`.
//! * [`march_clouds`] — the primary view-ray integration returning
//!   `(radiance, transmittance)`.
//!
//! # Conventions
//! * Positions are planet-centred in kilometres; directions are normalised
//!   internally (a zero direction yields an empty march).
//! * The density callback must return values in `[0, 1]`; its output is clamped
//!   defensively regardless.
//! * Transmittance is scalar (grey) in `[0, 1]`, starts at `1`, and decreases
//!   monotonically along the march. Radiance is linear-RGB and non-negative.
//! * An empty or degenerate interval is the identity: `(0, 1)` (no light added,
//!   full transmittance). Every result is finite (never `NaN`).
//! * Pure deterministic functions: no RNG, I/O, GPU, or `unsafe`.

use bevy_math::Vec3;

use super::scattering::{
    beer_lambert, beer_powder, dual_lobe_hg, multiple_scattering_transmittance,
};
use crate::gi::atmosphere::transmittance::ray_sphere_intersections;

/// Smallest span treated as a non-empty march interval (km).
const INTERVAL_EPS: f32 = 1.0e-5;
/// Transmittance below which the front-to-back march terminates early.
const TRANSMITTANCE_CUTOFF: f32 = 1.0e-4;
/// Upper bound on honoured view-ray steps.
const MAX_VIEW_STEPS: u32 = 1024;
/// Upper bound on honoured light-ray steps.
const MAX_LIGHT_STEPS: u32 = 256;

/// Nearest forward march interval `[t_near, t_far]` of a ray inside the cloud
/// shell bounded by `inner_radius` and `outer_radius`.
///
/// The shell is the region `inner <= |origin + t·dir| <= outer`. This returns
/// the nearest sub-interval ahead of the origin (`t >= 0`), correctly skipping
/// the inner "hole" when the ray dips below `inner_radius`. Returns [`None`]
/// when the ray misses the outer sphere ahead, when the direction is
/// degenerate, or when the radii are non-positive. Radii are reordered if
/// supplied the wrong way round.
#[inline]
pub fn shell_march_interval(
    origin: Vec3,
    dir: Vec3,
    inner_radius: f32,
    outer_radius: f32,
) -> Option<(f32, f32)> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO || !origin.is_finite() {
        return None;
    }
    let inner = inner_radius.min(outer_radius);
    let outer = inner_radius.max(outer_radius);
    if !(outer > 0.0) {
        return None;
    }

    // Outer shell span, clipped to the forward ray.
    let (a0, a1) = ray_sphere_intersections(origin, dir, outer)?;
    let t_lo = a0.max(0.0);
    let t_hi = a1;
    if !(t_hi - t_lo > INTERVAL_EPS) {
        return None;
    }

    // Subtract the inner "hole" (b0, b1) where |p| < inner, if the ray hits it.
    match ray_sphere_intersections(origin, dir, inner) {
        Some((b0, b1)) if inner > 0.0 => {
            let seg_near = (t_lo, t_hi.min(b0));
            let seg_far = (t_lo.max(b1), t_hi);
            nearest_segment(seg_near, seg_far)
        }
        _ => Some((t_lo, t_hi)),
    }
}

/// Picks the nearest non-empty of two candidate `[lo, hi]` segments.
#[inline]
fn nearest_segment(seg_a: (f32, f32), seg_b: (f32, f32)) -> Option<(f32, f32)> {
    let valid_a = seg_a.1 - seg_a.0 > INTERVAL_EPS;
    let valid_b = seg_b.1 - seg_b.0 > INTERVAL_EPS;
    match (valid_a, valid_b) {
        (true, true) => {
            if seg_a.0 <= seg_b.0 {
                Some(seg_a)
            } else {
                Some(seg_b)
            }
        }
        (true, false) => Some(seg_a),
        (false, true) => Some(seg_b),
        (false, false) => None,
    }
}

/// Controls for the cloud march: medium coefficients, phase, and resolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudMarchParams {
    /// Extinction coefficient per unit density (per km) at density `1`.
    pub extinction: f32,
    /// Single-scatter albedo in `[0, 1]`.
    pub albedo: f32,
    /// Forward Henyey-Greenstein lobe asymmetry.
    pub forward_g: f32,
    /// Backward Henyey-Greenstein lobe asymmetry.
    pub backward_g: f32,
    /// Dual-lobe blend weight toward the backward lobe, in `[0, 1]`.
    pub lobe_blend: f32,
    /// Beer-powder darkening strength in `[0, 1]`.
    pub powder_strength: f32,
    /// Number of view-ray integration steps.
    pub view_steps: u32,
    /// Number of secondary light-ray steps.
    pub light_steps: u32,
    /// Maximum secondary light-march distance (km).
    pub light_span: f32,
    /// Multiple-scattering octave count for the light energy.
    pub ms_octaves: u32,
    /// Multiple-scattering per-octave attenuation in `[0, 1)`.
    pub ms_attenuation: f32,
}

impl CloudMarchParams {
    /// A reasonable Earth-like preset: moderate extinction, bright albedo, a
    /// strong forward lobe with a weak backward glow, and modest powder.
    #[inline]
    pub fn earthlike() -> Self {
        Self {
            extinction: 0.1,
            albedo: 0.9,
            forward_g: 0.8,
            backward_g: -0.2,
            lobe_blend: 0.3,
            powder_strength: 0.5,
            view_steps: 64,
            light_steps: 6,
            light_span: 6.0,
            ms_octaves: 3,
            ms_attenuation: 0.5,
        }
    }
}

/// Secondary sun-ward march returning in-cloud light energy in `[0, 1]`.
///
/// Marches from `pos` toward `light_dir` through the cloud shell (capped at
/// `params.light_span` km), accumulating optical depth, then combines the
/// Beer-powder two-term model with the multiple-scattering transmittance
/// approximation. Returns `1.0` (full light) when the ray immediately escapes
/// the shell. The result is always finite and in `[0, 1]`.
#[inline]
pub fn light_energy<F>(
    pos: Vec3,
    light_dir: Vec3,
    inner_radius: f32,
    outer_radius: f32,
    params: &CloudMarchParams,
    density: &F,
) -> f32
where
    F: Fn(Vec3) -> f32,
{
    let dir = light_dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return 1.0;
    }
    let (l0, l1) = match shell_march_interval(pos, dir, inner_radius, outer_radius) {
        Some(seg) => seg,
        None => return 1.0,
    };
    let span_cap = if params.light_span.is_finite() {
        params.light_span.max(0.0)
    } else {
        0.0
    };
    let span = (l1 - l0).min(span_cap);
    if !(span > INTERVAL_EPS) {
        return 1.0;
    }
    let steps = params.light_steps.clamp(1, MAX_LIGHT_STEPS);
    let ls = span / steps as f32;
    let extinction = if params.extinction.is_finite() {
        params.extinction.max(0.0)
    } else {
        0.0
    };

    let mut tau = 0.0f32;
    for j in 0..steps {
        let t = l0 + (j as f32 + 0.5) * ls;
        let sample = pos + dir * t;
        let d = sample_density(density, sample);
        tau += extinction * d * ls;
    }

    let edge = beer_powder(tau, params.powder_strength);
    let multi = multiple_scattering_transmittance(tau, params.ms_octaves, params.ms_attenuation);
    (0.5 * edge + 0.5 * multi).clamp(0.0, 1.0)
}

/// Primary view-ray integration through the cloud shell.
///
/// Computes the forward march interval via [`shell_march_interval`], then
/// accumulates in-scattered radiance and transmittance *front-to-back,
/// premultiplied*: each step adds `transmittance · (1 - step_T) · albedo ·
/// phase · light · sun_radiance` and multiplies the running transmittance by
/// the step's Beer-Lambert factor. The dual-lobe phase is evaluated once for
/// the fixed view/light geometry. Returns `(radiance, transmittance)`; an empty
/// interval or degenerate ray yields the identity `(0, 1)`.
#[inline]
pub fn march_clouds<F>(
    origin: Vec3,
    view_dir: Vec3,
    light_dir: Vec3,
    sun_radiance: Vec3,
    inner_radius: f32,
    outer_radius: f32,
    params: &CloudMarchParams,
    density: &F,
) -> (Vec3, f32)
where
    F: Fn(Vec3) -> f32,
{
    let view = view_dir.normalize_or_zero();
    let light = light_dir.normalize_or_zero();
    if view == Vec3::ZERO {
        return (Vec3::ZERO, 1.0);
    }
    let (t0, t1) = match shell_march_interval(origin, view, inner_radius, outer_radius) {
        Some(seg) => seg,
        None => return (Vec3::ZERO, 1.0),
    };
    if !(t1 - t0 > INTERVAL_EPS) {
        return (Vec3::ZERO, 1.0);
    }

    let steps = params.view_steps.clamp(1, MAX_VIEW_STEPS);
    let ds = (t1 - t0) / steps as f32;
    let extinction = if params.extinction.is_finite() {
        params.extinction.max(0.0)
    } else {
        0.0
    };
    let albedo = if params.albedo.is_finite() {
        params.albedo.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cos_theta = view.dot(light);
    let phase = dual_lobe_hg(cos_theta, params.forward_g, params.backward_g, params.lobe_blend);
    let sun = sanitize_rgb(sun_radiance);

    let mut transmittance = 1.0f32;
    let mut radiance = Vec3::ZERO;
    for i in 0..steps {
        let t = t0 + (i as f32 + 0.5) * ds;
        let sample = origin + view * t;
        let d = sample_density(density, sample);
        if d > 0.0 {
            let sigma = extinction * d;
            let step_transmittance = beer_lambert(sigma, ds);
            let absorbed = (1.0 - step_transmittance).clamp(0.0, 1.0);
            let energy = light_energy(sample, light, inner_radius, outer_radius, params, density);
            let in_scatter = sun * (albedo * phase * energy * absorbed);
            radiance += in_scatter * transmittance;
            transmittance *= step_transmittance;
            if transmittance < TRANSMITTANCE_CUTOFF {
                break;
            }
        }
    }

    (sanitize_rgb(radiance), transmittance.clamp(0.0, 1.0))
}

/// Clamps a density sample to `[0, 1]`, mapping non-finite values to `0`.
#[inline]
fn sample_density<F>(density: &F, p: Vec3) -> f32
where
    F: Fn(Vec3) -> f32,
{
    let d = density(p);
    if d.is_finite() { d.clamp(0.0, 1.0) } else { 0.0 }
}

/// Replaces non-finite channels with `0` and clamps every channel non-negative.
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

    /// Shell radii used across the tests (km).
    const INNER: f32 = 6362.0;
    const OUTER: f32 = 6365.0;

    #[test]
    fn centred_ray_skips_the_inner_hole() {
        // From the planet centre along +Z the shell span is [inner, outer].
        let seg = shell_march_interval(Vec3::ZERO, Vec3::Z, 1.0, 2.0).unwrap();
        assert!((seg.0 - 1.0).abs() < 1e-4, "near={}", seg.0);
        assert!((seg.1 - 2.0).abs() < 1e-4, "far={}", seg.1);
    }

    #[test]
    fn ray_without_inner_hit_spans_full_outer_chord() {
        // Tangent to the inner sphere but through the outer: no hole removed.
        let origin = Vec3::new(0.0, 1.5, 0.0);
        let seg = shell_march_interval(origin, Vec3::Z, 1.0, 2.0).unwrap();
        // |origin + t Z| = 2 -> t = sqrt(4 - 2.25) ~ 1.3229 either side.
        let expected = (4.0f32 - 2.25).sqrt();
        assert!((seg.0 - 0.0).abs() < 1e-4, "near={}", seg.0);
        assert!((seg.1 - expected).abs() < 1e-3, "far={} exp={expected}", seg.1);
    }

    #[test]
    fn degenerate_and_missing_rays_return_none() {
        assert!(shell_march_interval(Vec3::ZERO, Vec3::ZERO, 1.0, 2.0).is_none());
        // Ray entirely outside pointing away from the shell.
        assert!(shell_march_interval(Vec3::new(0.0, 0.0, 10.0), Vec3::Z, 1.0, 2.0).is_none());
        // Non-positive radii.
        assert!(shell_march_interval(Vec3::ZERO, Vec3::Z, 0.0, 0.0).is_none());
    }

    #[test]
    fn empty_interval_is_identity() {
        let params = CloudMarchParams::earthlike();
        let density = |_p: Vec3| 0.5;
        // Degenerate view direction -> identity.
        let (rad, tr) = march_clouds(
            Vec3::new(0.0, INNER - 10.0, 0.0),
            Vec3::ZERO,
            Vec3::Y,
            Vec3::splat(10.0),
            INNER,
            OUTER,
            &params,
            &density,
        );
        assert_eq!(rad, Vec3::ZERO);
        assert_eq!(tr, 1.0);
    }

    #[test]
    fn empty_cloud_is_fully_transmissive() {
        let params = CloudMarchParams::earthlike();
        let density = |_p: Vec3| 0.0;
        let origin = Vec3::new(0.0, INNER - 5.0, 0.0);
        let (rad, tr) = march_clouds(
            origin,
            Vec3::Y,
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::splat(10.0),
            INNER,
            OUTER,
            &params,
            &density,
        );
        assert_eq!(rad, Vec3::ZERO, "empty cloud scattered light");
        assert!((tr - 1.0).abs() < 1e-6, "empty cloud attenuated: {tr}");
    }

    #[test]
    fn transmittance_in_range_and_decreases_with_density() {
        let params = CloudMarchParams::earthlike();
        let origin = Vec3::new(0.0, INNER - 5.0, 0.0);
        let sun = Vec3::splat(10.0);
        let light = Vec3::new(0.2, 1.0, 0.0);
        let mut prev = 1.0f32;
        for i in 0..=10 {
            let d = i as f32 / 10.0;
            let density = move |_p: Vec3| d;
            let (rad, tr) = march_clouds(
                origin, Vec3::Y, light, sun, INNER, OUTER, &params, &density,
            );
            assert!((0.0..=1.0).contains(&tr), "transmittance out of range: {tr}");
            assert!(rad.is_finite() && rad.min_element() >= 0.0, "bad radiance: {rad:?}");
            assert!(tr <= prev + 1e-6, "transmittance not decreasing at d={d}: {tr} > {prev}");
            prev = tr;
        }
        // A dense cloud must attenuate noticeably.
        assert!(prev < 0.9, "dense cloud barely attenuated: {prev}");
    }

    #[test]
    fn light_energy_bounded_and_monotone_without_powder() {
        let mut params = CloudMarchParams::earthlike();
        params.powder_strength = 0.0; // disable the non-monotone powder term
        let pos = Vec3::new(0.0, INNER + 1.0, 0.0);
        let light = Vec3::Y;
        let mut prev = 2.0f32;
        for i in 0..=10 {
            let d = i as f32 / 10.0;
            let density = move |_p: Vec3| d;
            let e = light_energy(pos, light, INNER, OUTER, &params, &density);
            assert!((0.0..=1.0).contains(&e), "light energy out of range: {e}");
            assert!(e <= prev + 1e-6, "light energy not decreasing at d={d}: {e} > {prev}");
            prev = e;
        }
    }

    #[test]
    fn light_energy_full_when_ray_escapes() {
        let params = CloudMarchParams::earthlike();
        let density = |_p: Vec3| 1.0;
        // A point above the shell marching further up never re-enters it.
        let pos = Vec3::new(0.0, OUTER + 10.0, 0.0);
        let e = light_energy(pos, Vec3::Y, INNER, OUTER, &params, &density);
        assert!((e - 1.0).abs() < 1e-6, "expected full light, got {e}");
    }

    #[test]
    fn march_never_produces_nan_on_bad_inputs() {
        let params = CloudMarchParams::earthlike();
        let density = |_p: Vec3| f32::NAN;
        let (rad, tr) = march_clouds(
            Vec3::new(0.0, INNER - 5.0, 0.0),
            Vec3::Y,
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
            INNER,
            OUTER,
            &params,
            &density,
        );
        assert!(rad.is_finite(), "radiance not finite: {rad:?}");
        assert!(tr.is_finite() && (0.0..=1.0).contains(&tr), "transmittance bad: {tr}");
    }

    #[test]
    fn radiance_scales_with_sun_intensity() {
        let params = CloudMarchParams::earthlike();
        let origin = Vec3::new(0.0, INNER - 5.0, 0.0);
        let light = Vec3::new(0.2, 1.0, 0.0);
        let density = |_p: Vec3| 0.3;
        let (dim, _) = march_clouds(
            origin, Vec3::Y, light, Vec3::splat(1.0), INNER, OUTER, &params, &density,
        );
        let (bright, _) = march_clouds(
            origin, Vec3::Y, light, Vec3::splat(10.0), INNER, OUTER, &params, &density,
        );
        assert!(bright.length() > dim.length(), "radiance did not scale with sun");
    }
}
