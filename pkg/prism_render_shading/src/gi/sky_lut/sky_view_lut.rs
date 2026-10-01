//! Sky-View LUT: bake and sample the full-sky radiance seen from a fixed
//! altitude, in Hillaire's horizon-centred latitude/longitude parameterization.
//!
//! The sky-view LUT caches the in-scattered radiance along every view direction
//! from a single camera height `r`, given a fixed sun direction. At runtime the
//! sky is then a cheap bilinear lookup rather than a full ray-march per pixel.
//!
//! Two angles parameterize the hemisphere-plus-ground dome:
//!
//! * **latitude** (`v`) — the view's zenith angle, warped so the horizon sits at
//!   `v = 0.5`. Hillaire squares the normalized angle on each side of the
//!   horizon, concentrating texels where the sky gradient is steepest and
//!   avoiding the banding artifacts a linear map produces near the horizon.
//! * **longitude** (`u`) — the azimuth of the view relative to the sun,
//!   encoded through `u = sqrt(0.5 - 0.5 · cosγ)` where `γ` is the view→sun
//!   azimuthal angle. This clusters resolution toward the solar meridian.
//!
//! Each texel's radiance couples **single scattering** (sunlight scattered once
//! toward the eye) with **multiple scattering** sampled from a
//! [`MultiscatterLut`]: along the view ray the isotropic multiple-scattering
//! factor `Ψ_ms` is weighted by the local scattering coefficient and view
//! transmittance, exactly as in Hillaire 2020.
//!
//! See [`uv_to_sky_params`] / [`sky_params_to_uv`] for the maps,
//! [`bake_sky_view_lut`] for the bake, and [`sample_sky_view`] for runtime
//! lookup from world-space view/sun directions.
//!
//! # Conventions
//! * The local zenith is `+Y`; the camera sits at `(0, r, 0)` with
//!   `r ∈ [bottom_radius, top_radius]`.
//! * `u`/`v` are clamped to `[0, 1]` (clamp-to-edge); the horizon maps to
//!   `v = 0.5`.
//! * Radiance is spectral linear-RGB, non-negative and finite (never `NaN`).
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the value
//!   method. Every function is a deterministic pure function with no RNG, I/O,
//!   GPU, or `unsafe`.

use alloc::vec::Vec;
use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

use super::multiscatter_lut::{sample_multiscatter_lut, MultiscatterLut};
use crate::gi::atmosphere::medium::Atmosphere;
use crate::gi::atmosphere::phase::{cornette_shanks_phase, rayleigh_phase};
use crate::gi::atmosphere::transmittance::{
    distance_to_boundary, nearest_positive_intersection, transmittance_to_boundary,
};

/// Smallest angle / denominator treated as non-degenerate.
const EPSILON: f32 = 1.0e-6;

/// A baked 2D sky radiance table in `(u = longitude, v = latitude)` space.
///
/// Stored row-major (`v` outer, `u` inner) as linear-RGB [`Vec3`] radiance.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyViewLut {
    width: usize,
    height: usize,
    /// Camera radius the table was baked at (km).
    view_radius: f32,
    texels: Vec<Vec3>,
}

impl SkyViewLut {
    /// Texel-grid width (`u` / longitude axis).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Texel-grid height (`v` / latitude axis).
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Camera radius (km) the table was baked at.
    #[inline]
    pub fn view_radius(&self) -> f32 {
        self.view_radius
    }

    /// Immutable view of the row-major texel storage.
    #[inline]
    pub fn texels(&self) -> &[Vec3] {
        &self.texels
    }

    /// Row-major index for integer texel `(x, y)`, clamped to the grid.
    #[inline]
    pub fn texel_index(&self, x: usize, y: usize) -> usize {
        if self.width == 0 || self.height == 0 {
            return 0;
        }
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        y * self.width + x
    }

    /// Clamp-to-edge nearest fetch of integer texel `(x, y)`.
    #[inline]
    fn fetch(&self, x: usize, y: usize) -> Vec3 {
        if self.texels.is_empty() {
            return Vec3::ZERO;
        }
        self.texels[self.texel_index(x, y)]
    }

    /// Clamp-to-edge bilinear fetch at continuous UV in `[0, 1]²`.
    ///
    /// An empty table yields `Vec3::ZERO`.
    #[inline]
    pub fn sample_uv(&self, u: f32, v: f32) -> Vec3 {
        if self.width == 0 || self.height == 0 || self.texels.is_empty() {
            return Vec3::ZERO;
        }
        let u = clamp01(u);
        let v = clamp01(v);
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0 = floor_usize(fx);
        let y0 = floor_usize(fy);
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let tx = (fx - fx.floor()).clamp(0.0, 1.0);
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);
        let c00 = self.fetch(x0, y0);
        let c10 = self.fetch(x1, y0);
        let c01 = self.fetch(x0, y1);
        let c11 = self.fetch(x1, y1);
        let top = c00.lerp(c10, tx);
        let bottom = c01.lerp(c11, tx);
        sanitize_rgb(top.lerp(bottom, ty))
    }
}

/// Floor of `value` as a non-negative `usize` (negatives clamp to `0`).
#[inline]
fn floor_usize(value: f32) -> usize {
    if value <= 0.0 || !value.is_finite() {
        0
    } else {
        value.floor() as usize
    }
}

/// Clamps `value` to `[0, 1]`, mapping non-finite inputs to `0`.
#[inline]
fn clamp01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
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

/// Replaces non-finite channels with `0` and clamps each non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

/// Per-channel `exp(-tau)` clamped to `[0, 1]`, saturating large exponents.
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
    ops::exp(-tau.min(80.0)).clamp(0.0, 1.0)
}

/// The ground-to-horizon angle `β` (rad) and its cosine at radius `r`.
///
/// `cos β = sqrt(r² - bottom²) / r` is the cosine of the angle between the
/// local zenith and the horizon tangent; `β` is clamped to `[0, π]`.
#[inline]
fn horizon_angle(atmosphere: &Atmosphere, r: f32) -> (f32, f32) {
    let bottom = atmosphere.bottom_radius.max(0.0);
    let r = if r.is_finite() { r.max(bottom.max(EPSILON)) } else { bottom.max(EPSILON) };
    let v_horizon = (r * r - bottom * bottom).max(0.0).sqrt();
    let cos_beta = clamp_finite(v_horizon / r, -1.0, 1.0);
    let beta = ops::acos(cos_beta);
    (beta, cos_beta)
}

/// Converts a UV coordinate in `[0, 1]²` to `(view_zenith_cos, light_view_cos)`
/// at camera radius `r`.
///
/// `v < 0.5` decodes directions above the horizon, `v ≥ 0.5` below it, using the
/// squared horizon-centred warp. `u` decodes the view→sun azimuthal cosine.
#[inline]
pub fn uv_to_sky_params(atmosphere: &Atmosphere, r: f32, u: f32, v: f32) -> (f32, f32) {
    let u = clamp01(u);
    let v = clamp01(v);
    let (beta, _) = horizon_angle(atmosphere, r);
    let zenith_horizon_angle = (PI - beta).max(EPSILON);

    let view_zenith_cos = if v < 0.5 {
        let mut coord = 2.0 * v;
        coord = 1.0 - coord;
        coord *= coord;
        coord = 1.0 - coord;
        ops::cos(zenith_horizon_angle * coord).clamp(-1.0, 1.0)
    } else {
        let mut coord = v * 2.0 - 1.0;
        coord *= coord;
        ops::cos(zenith_horizon_angle + beta * coord).clamp(-1.0, 1.0)
    };

    // Azimuthal cosine: inverse of u = sqrt(0.5 - 0.5·cosγ).
    let coord = u * u;
    let light_view_cos = (1.0 - 2.0 * coord).clamp(-1.0, 1.0);
    (view_zenith_cos, light_view_cos)
}

/// Converts `(view_zenith_cos, light_view_cos)` to a UV coordinate in `[0, 1]²`.
///
/// Inverse of [`uv_to_sky_params`]. `intersect_ground` selects the below-horizon
/// branch so grazing rays land on the correct side of `v = 0.5`.
#[inline]
pub fn sky_params_to_uv(
    atmosphere: &Atmosphere,
    r: f32,
    view_zenith_cos: f32,
    light_view_cos: f32,
    intersect_ground: bool,
) -> (f32, f32) {
    let (beta, _) = horizon_angle(atmosphere, r);
    let zenith_horizon_angle = (PI - beta).max(EPSILON);
    let view_zenith_angle = ops::acos(clamp_finite(view_zenith_cos, -1.0, 1.0));

    let v = if !intersect_ground {
        let mut coord = view_zenith_angle / zenith_horizon_angle;
        coord = (1.0 - coord).max(0.0);
        coord = coord.sqrt();
        coord = 1.0 - coord;
        clamp01(coord * 0.5)
    } else {
        let denom = beta.max(EPSILON);
        let mut coord = (view_zenith_angle - zenith_horizon_angle) / denom;
        coord = coord.max(0.0).sqrt();
        clamp01(coord * 0.5 + 0.5)
    };

    let lvc = clamp_finite(light_view_cos, -1.0, 1.0);
    let u = clamp01((0.5 - 0.5 * lvc).max(0.0).sqrt());
    (u, v)
}

/// Builds the world-space view direction for `(view_zenith_cos, light_view_cos)`
/// in the bake frame (zenith `+Y`, sun azimuth along `+X`).
#[inline]
fn view_direction(view_zenith_cos: f32, light_view_cos: f32) -> Vec3 {
    let vzc = view_zenith_cos.clamp(-1.0, 1.0);
    let lvc = light_view_cos.clamp(-1.0, 1.0);
    let sin_vz = (1.0 - vzc * vzc).max(0.0).sqrt();
    let sin_az = (1.0 - lvc * lvc).max(0.0).sqrt();
    // Horizontal azimuth measured from the sun's horizontal direction (+X).
    Vec3::new(lvc * sin_vz, vzc, sin_az * sin_vz)
}

/// Sun direction with zenith cosine `sun_cos_zenith` in the bake frame.
#[inline]
fn sun_direction(sun_cos_zenith: f32) -> Vec3 {
    let c = clamp_finite(sun_cos_zenith, -1.0, 1.0);
    let s = (1.0 - c * c).max(0.0).sqrt();
    Vec3::new(s, c, 0.0)
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

/// Marches one view ray, coupling single scattering with the multiple-scattering
/// LUT, returning the total in-scattered radiance.
///
/// The single-scattering term mirrors
/// [`crate::gi::atmosphere::scattering::single_scattering`] exactly; the
/// multiple-scattering term adds `T_view · σ_s · Ψ_ms` sampled from `ms_lut` at
/// each step, using the local sun-zenith cosine along the ray.
#[inline]
fn march_sky_radiance(
    atmosphere: &Atmosphere,
    origin: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    sun_irradiance: Vec3,
    ms_lut: &MultiscatterLut,
    view_samples: u32,
    sun_samples: u32,
) -> Vec3 {
    let dir = view_dir.normalize_or_zero();
    let sun = sun_dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return Vec3::ZERO;
    }
    let distance = distance_to_boundary(atmosphere, origin, dir);
    if !(distance > 0.0) || !distance.is_finite() {
        return Vec3::ZERO;
    }
    let cos_theta = clamp_finite(dir.dot(sun), -1.0, 1.0);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = cornette_shanks_phase(cos_theta, atmosphere.mie_g);

    let steps = view_samples.max(1);
    let ds = distance / steps as f32;
    let mut optical_depth = Vec3::ZERO;
    let mut radiance = Vec3::ZERO;
    for i in 0..steps {
        let t_mid = (i as f32 + 0.5) * ds;
        let pos = origin + dir * t_mid;
        let altitude = atmosphere.altitude_at(pos);
        let ext = atmosphere.extinction(altitude);
        let t_view = exp_neg(optical_depth + ext * (0.5 * ds));

        // Single scattering (directional phase-weighted).
        let rayleigh_s = atmosphere.rayleigh_scattering_at(altitude);
        let mie_s = atmosphere.mie_scattering_at(altitude);
        let scatter_phase = rayleigh_s * phase_r + Vec3::splat(mie_s * phase_m);
        let t_sun = sun_transmittance(atmosphere, pos, sun, sun_samples);
        radiance += t_view * t_sun * scatter_phase * sun_irradiance * ds;

        // Multiple scattering (isotropic, from the LUT).
        let total_scatter = atmosphere.scattering(altitude);
        let r_local = pos.length();
        let up = if r_local > EPSILON { pos / r_local } else { Vec3::Y };
        let mu_sun_local = clamp_finite(up.dot(sun), -1.0, 1.0);
        let psi_ms = sample_multiscatter_lut(ms_lut, atmosphere, r_local, mu_sun_local);
        radiance += t_view * total_scatter * psi_ms * ds;

        optical_depth += ext * ds;
    }
    sanitize_rgb(radiance)
}

/// Bakes a [`SkyViewLut`] at camera radius `r` for a fixed sun.
///
/// Each texel decodes to `(view_zenith_cos, light_view_cos)`, reconstructs the
/// world-space view direction, and marches [`march_sky_radiance`] with
/// `view_samples` steps (and `sun_samples` sunlight sub-steps), coupling the
/// `ms_lut` multiple-scattering factor. Zero dimensions yield an empty table.
#[inline]
pub fn bake_sky_view_lut(
    atmosphere: &Atmosphere,
    r: f32,
    sun_cos_zenith: f32,
    sun_irradiance: Vec3,
    ms_lut: &MultiscatterLut,
    width: usize,
    height: usize,
    view_samples: u32,
    sun_samples: u32,
) -> SkyViewLut {
    let bottom = atmosphere.bottom_radius.max(0.0);
    let top = atmosphere.top_radius.max(bottom);
    let view_radius = if r.is_finite() { r.clamp(bottom, top) } else { bottom };
    if width == 0 || height == 0 {
        return SkyViewLut {
            width: 0,
            height: 0,
            view_radius,
            texels: Vec::new(),
        };
    }
    let origin = Vec3::new(0.0, view_radius, 0.0);
    let sun = sun_direction(sun_cos_zenith);
    let view_samples = view_samples.max(1);
    let sun_samples = sun_samples.max(1);
    let mut texels = Vec::with_capacity(width * height);
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let (vzc, lvc) = uv_to_sky_params(atmosphere, view_radius, u, v);
            let view_dir = view_direction(vzc, lvc);
            let l = march_sky_radiance(
                atmosphere,
                origin,
                view_dir,
                sun,
                sun_irradiance,
                ms_lut,
                view_samples,
                sun_samples,
            );
            texels.push(l);
        }
    }
    SkyViewLut {
        width,
        height,
        view_radius,
        texels,
    }
}

/// Samples the baked sky radiance for world-space `view_dir`/`sun_dir`.
///
/// Both directions are interpreted in the bake frame (local zenith `+Y`). The
/// view zenith cosine and the view→sun azimuthal cosine are recovered, the
/// below-horizon branch is detected by ray/ground intersection, and the LUT is
/// sampled with clamp-to-edge filtering.
#[inline]
pub fn sample_sky_view(
    lut: &SkyViewLut,
    atmosphere: &Atmosphere,
    view_dir: Vec3,
    sun_dir: Vec3,
) -> Vec3 {
    let view = view_dir.normalize_or_zero();
    let sun = sun_dir.normalize_or_zero();
    if view == Vec3::ZERO {
        return Vec3::ZERO;
    }
    let up = Vec3::Y;
    let vzc = clamp_finite(view.dot(up), -1.0, 1.0);
    // Horizontal (azimuthal) components relative to the zenith.
    let view_h = (view - up * vzc).normalize_or_zero();
    let sun_h = (sun - up * sun.dot(up)).normalize_or_zero();
    let lvc = if view_h == Vec3::ZERO || sun_h == Vec3::ZERO {
        1.0
    } else {
        clamp_finite(view_h.dot(sun_h), -1.0, 1.0)
    };
    let origin = Vec3::new(0.0, lut.view_radius(), 0.0);
    let intersect_ground =
        nearest_positive_intersection(origin, view, atmosphere.bottom_radius).is_some();
    let (u, v) = sky_params_to_uv(atmosphere, lut.view_radius(), vzc, lvc, intersect_ground);
    lut.sample_uv(u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::atmosphere::scattering::single_scattering;

    fn earth() -> Atmosphere {
        Atmosphere::earth()
    }

    /// A zero-filled multiple-scattering LUT (contributes nothing).
    fn zero_ms() -> MultiscatterLut {
        super::super::multiscatter_lut::bake_multiscatter_lut(&earth(), Vec3::ZERO, 2, 2, 2, 2)
    }

    #[test]
    fn uv_param_roundtrip_is_stable() {
        let a = earth();
        let r = a.bottom_radius + 1.0;
        for &v in &[0.05f32, 0.2, 0.45, 0.55, 0.8, 0.95] {
            for &u in &[0.02f32, 0.3, 0.6, 0.9, 0.99] {
                let (vzc, lvc) = uv_to_sky_params(&a, r, u, v);
                let intersect = v >= 0.5;
                let (u2, v2) = sky_params_to_uv(&a, r, vzc, lvc, intersect);
                assert!((u - u2).abs() < 3e-3, "u={u} u2={u2}");
                assert!((v - v2).abs() < 3e-3, "v={v} v2={v2} vzc={vzc}");
            }
        }
    }

    #[test]
    fn horizon_maps_to_half() {
        let a = earth();
        let r = a.bottom_radius + 5.0;
        let (beta, _) = horizon_angle(&a, r);
        let zenith_horizon_angle = PI - beta;
        // A ray exactly at the horizon zenith angle maps to v = 0.5.
        let vzc = ops::cos(zenith_horizon_angle);
        let (_, v_above) = sky_params_to_uv(&a, r, vzc, 0.0, false);
        assert!((v_above - 0.5).abs() < 1e-3, "v_above={v_above}");
    }

    #[test]
    fn baked_single_term_matches_single_scattering() {
        // With a zero multiple-scattering LUT, each texel must reproduce the
        // analytic single-scattering integral for its decoded direction.
        let a = earth();
        let ms = zero_ms();
        let r = a.bottom_radius + 2.0;
        let sun_cos = 0.6;
        let sun_irr = Vec3::splat(20.0);
        let (w, h) = (32usize, 32usize);
        let lut = bake_sky_view_lut(&a, r, sun_cos, sun_irr, &ms, w, h, 48, 16);
        let origin = Vec3::new(0.0, r, 0.0);
        let sun = sun_direction(sun_cos);
        for &(x, y) in &[(8usize, 8usize), (16, 20), (24, 12)] {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            let (vzc, lvc) = uv_to_sky_params(&a, r, u, v);
            let view_dir = view_direction(vzc, lvc);
            let got = lut.texels()[lut.texel_index(x, y)];
            let want = single_scattering(&a, origin, view_dir, sun, sun_irr, 48, 16);
            assert!((got - want).length() < 1e-4, "got={got:?} want={want:?}");
        }
    }

    #[test]
    fn multiscatter_adds_energy() {
        let a = earth();
        let r = a.bottom_radius + 2.0;
        let sun_cos = 0.5;
        let sun_irr = Vec3::splat(20.0);
        let zero = zero_ms();
        let full =
            super::super::multiscatter_lut::bake_multiscatter_lut(&a, sun_irr, 16, 16, 32, 12);
        let lut_single = bake_sky_view_lut(&a, r, sun_cos, sun_irr, &zero, 24, 24, 32, 12);
        let lut_multi = bake_sky_view_lut(&a, r, sun_cos, sun_irr, &full, 24, 24, 32, 12);
        let view = Vec3::new(0.3, 0.7, 0.1);
        let sun = sun_direction(sun_cos);
        let s = sample_sky_view(&lut_single, &a, view, sun);
        let m = sample_sky_view(&lut_multi, &a, view, sun);
        assert!(m.is_finite() && s.is_finite());
        assert!(m.max_element() >= s.max_element() - 1e-5, "m={m:?} s={s:?}");
    }

    #[test]
    fn radiance_is_non_negative_and_finite() {
        let a = earth();
        let ms = zero_ms();
        let lut = bake_sky_view_lut(&a, a.bottom_radius + 1.0, 0.3, Vec3::splat(15.0), &ms, 16, 16, 24, 8);
        for texel in lut.texels() {
            assert!(texel.is_finite() && texel.min_element() >= 0.0, "texel={texel:?}");
        }
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let a = earth();
        let ms = zero_ms();
        let empty = bake_sky_view_lut(&a, a.bottom_radius + 1.0, 0.5, Vec3::splat(1.0), &ms, 0, 0, 8, 4);
        assert_eq!(sample_sky_view(&empty, &a, Vec3::Y, Vec3::Y), Vec3::ZERO);
        let lut = bake_sky_view_lut(&a, a.bottom_radius + 1.0, 0.5, Vec3::splat(1.0), &ms, 8, 8, 8, 4);
        // Zero view direction -> zero radiance, no NaN.
        assert_eq!(sample_sky_view(&lut, &a, Vec3::ZERO, Vec3::Y), Vec3::ZERO);
        // NaN directions stay finite.
        let s = sample_sky_view(&lut, &a, Vec3::splat(f32::NAN), Vec3::splat(f32::NAN));
        assert!(s.is_finite());
        let (vzc, lvc) = uv_to_sky_params(&a, f32::NAN, f32::NAN, f32::INFINITY);
        assert!(vzc.is_finite() && lvc.is_finite());
        let (u, v) = sky_params_to_uv(&a, f32::NAN, f32::NAN, f32::NAN, true);
        assert!(u.is_finite() && v.is_finite());
    }
}
