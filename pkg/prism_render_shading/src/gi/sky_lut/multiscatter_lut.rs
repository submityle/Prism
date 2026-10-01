//! Multiple-scattering LUT: bake and sample Hillaire's isotropic
//! multiple-scattering factor `Ψ_ms = L₂ / (1 - f)` over `(r, μ_sun)`.
//!
//! Hillaire approximates all scattering orders beyond the first as an isotropic
//! field: a second-order estimate `L₂` is amplified by the geometric series of
//! a per-order rescattering fraction `f`, giving `L₂ / (1 - f)`. Because this
//! factor depends only on altitude and the sun's zenith angle (the medium is
//! spherically symmetric and the second order is treated isotropically), it is
//! cheaply tabulated in a small 2D LUT and reused by the sky-view and
//! aerial-perspective passes.
//!
//! The parameterization is deliberately simple and matches Hillaire's reference:
//!
//! * `u = 0.5 + 0.5 · μ_sun` — linear in the sun-zenith cosine `μ_sun ∈ [-1, 1]`,
//! * `v = (r - bottom) / thickness` — linear in altitude.
//!
//! See [`uv_to_r_mu_sun`] / [`r_mu_sun_to_uv`] for the maps, and
//! [`bake_multiscatter_lut`] for the bake that calls
//! [`crate::gi::atmosphere::scattering::multiscatter_estimate`].
//!
//! # Conventions
//! * `r` is clamped to `[bottom_radius, top_radius]`; `μ_sun = cos θ_sun ∈ [-1, 1]`.
//! * `u`/`v` are clamped to `[0, 1]` (clamp-to-edge filtering).
//! * The stored factor is spectral linear-RGB, non-negative and finite (never
//!   `NaN`); it is dimensionless (per unit sun irradiance when baked with unit
//!   irradiance).
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the value
//!   method. Every function is a deterministic pure function with no RNG, I/O,
//!   GPU, or `unsafe`.

use alloc::vec::Vec;
use bevy_math::Vec3;

use crate::gi::atmosphere::medium::Atmosphere;
use crate::gi::atmosphere::scattering::multiscatter_estimate;

/// Smallest shell thickness treated as non-degenerate (km-scale).
const EPSILON: f32 = 1.0e-6;

/// A baked 2D multiple-scattering factor table in `(u = μ_sun, v = altitude)`.
///
/// Stored row-major (`v` outer, `u` inner) as linear-RGB [`Vec3`] factors.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiscatterLut {
    width: usize,
    height: usize,
    texels: Vec<Vec3>,
}

impl MultiscatterLut {
    /// Texel-grid width (`u` / `μ_sun` axis).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Texel-grid height (`v` / altitude axis).
    #[inline]
    pub fn height(&self) -> usize {
        self.height
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
    /// An empty table yields `Vec3::ZERO` (no multiple scattering).
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

/// Replaces non-finite channels with `0` and clamps each non-negative.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.max(0.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.max(0.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.max(0.0) } else { 0.0 },
    )
}

/// Converts a UV coordinate in `[0, 1]²` to physical `(r, μ_sun)`.
///
/// `u` is linear in `μ_sun ∈ [-1, 1]`; `v` is linear in altitude so
/// `r = bottom + v · thickness ∈ [bottom, top]`. A degenerate shell returns
/// `(bottom, 2u - 1)`.
#[inline]
pub fn uv_to_r_mu_sun(atmosphere: &Atmosphere, u: f32, v: f32) -> (f32, f32) {
    let bottom = atmosphere.bottom_radius.max(0.0);
    let thickness = atmosphere.thickness();
    let u = clamp01(u);
    let v = clamp01(v);
    let mu_sun = (2.0 * u - 1.0).clamp(-1.0, 1.0);
    if !(thickness > EPSILON) {
        return (bottom, mu_sun);
    }
    let r = bottom + v * thickness;
    (r.clamp(bottom, bottom + thickness), mu_sun)
}

/// Converts physical `(r, μ_sun)` to a UV coordinate in `[0, 1]²`.
///
/// Inverse of [`uv_to_r_mu_sun`]. Inputs are clamped; a degenerate shell returns
/// `v = 0`.
#[inline]
pub fn r_mu_sun_to_uv(atmosphere: &Atmosphere, r: f32, mu_sun: f32) -> (f32, f32) {
    let bottom = atmosphere.bottom_radius.max(0.0);
    let thickness = atmosphere.thickness();
    let mu_sun = if mu_sun.is_finite() {
        mu_sun.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let u = clamp01(0.5 + 0.5 * mu_sun);
    let v = if thickness > EPSILON {
        let r = if r.is_finite() { r } else { bottom };
        clamp01((r - bottom) / thickness)
    } else {
        0.0
    };
    (u, v)
}

/// Bakes a [`MultiscatterLut`] of the given dimensions for `atmosphere`.
///
/// Each texel decodes to `(r, μ_sun)` at its centre and evaluates
/// [`multiscatter_estimate`] with `dir_samples` Fibonacci directions and
/// `march_samples` march steps, baking against `sun_irradiance`. Zero
/// dimensions yield an empty table.
#[inline]
pub fn bake_multiscatter_lut(
    atmosphere: &Atmosphere,
    sun_irradiance: Vec3,
    width: usize,
    height: usize,
    dir_samples: u32,
    march_samples: u32,
) -> MultiscatterLut {
    if width == 0 || height == 0 {
        return MultiscatterLut {
            width: 0,
            height: 0,
            texels: Vec::new(),
        };
    }
    let mut texels = Vec::with_capacity(width * height);
    let dir_samples = dir_samples.max(1);
    let march_samples = march_samples.max(1);
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let (r, mu_sun) = uv_to_r_mu_sun(atmosphere, u, v);
            let altitude = (r - atmosphere.bottom_radius).clamp(0.0, atmosphere.thickness());
            let factor = multiscatter_estimate(
                atmosphere,
                altitude,
                mu_sun,
                sun_irradiance,
                dir_samples,
                march_samples,
            );
            texels.push(sanitize_rgb(factor));
        }
    }
    MultiscatterLut {
        width,
        height,
        texels,
    }
}

/// Samples the baked multiple-scattering factor for physical `(r, μ_sun)`.
///
/// Encodes through [`r_mu_sun_to_uv`] and samples `lut` with clamp-to-edge
/// filtering. Spectral linear-RGB, non-negative.
#[inline]
pub fn sample_multiscatter_lut(
    lut: &MultiscatterLut,
    atmosphere: &Atmosphere,
    r: f32,
    mu_sun: f32,
) -> Vec3 {
    let (u, v) = r_mu_sun_to_uv(atmosphere, r, mu_sun);
    lut.sample_uv(u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth() -> Atmosphere {
        Atmosphere::earth()
    }

    #[test]
    fn uv_param_roundtrip_is_exact() {
        let a = earth();
        for &v in &[0.0f32, 0.2, 0.5, 0.8, 1.0] {
            for &u in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let (r, mu) = uv_to_r_mu_sun(&a, u, v);
                let (u2, v2) = r_mu_sun_to_uv(&a, r, mu);
                assert!((u - u2).abs() < 1e-6, "u={u} u2={u2}");
                assert!((v - v2).abs() < 1e-6, "v={v} v2={v2}");
            }
        }
    }

    #[test]
    fn decoded_params_in_range() {
        let a = earth();
        for y in 0..8 {
            for x in 0..8 {
                let u = (x as f32 + 0.5) / 8.0;
                let v = (y as f32 + 0.5) / 8.0;
                let (r, mu) = uv_to_r_mu_sun(&a, u, v);
                assert!(r >= a.bottom_radius - 1e-3 && r <= a.top_radius + 1e-3, "r={r}");
                assert!((-1.0..=1.0).contains(&mu), "mu={mu}");
            }
        }
    }

    #[test]
    fn baked_sample_matches_direct_estimate() {
        let a = earth();
        let sun = Vec3::splat(1.0);
        let lut = bake_multiscatter_lut(&a, sun, 32, 32, 32, 16);
        for &(x, y) in &[(8usize, 6usize), (16, 16), (28, 24)] {
            let u = (x as f32 + 0.5) / 32.0;
            let v = (y as f32 + 0.5) / 32.0;
            let (r, mu) = uv_to_r_mu_sun(&a, u, v);
            let got = sample_multiscatter_lut(&lut, &a, r, mu);
            let altitude = (r - a.bottom_radius).clamp(0.0, a.thickness());
            let want = multiscatter_estimate(&a, altitude, mu, sun, 32, 16);
            assert!((got - want).length() < 1e-4, "got={got:?} want={want:?}");
        }
    }

    #[test]
    fn higher_sun_scatters_at_least_as_much() {
        let a = earth();
        let sun = Vec3::splat(1.0);
        let lut = bake_multiscatter_lut(&a, sun, 48, 32, 48, 16);
        let r = a.bottom_radius + 2.0;
        let overhead = sample_multiscatter_lut(&lut, &a, r, 1.0);
        let below = sample_multiscatter_lut(&lut, &a, r, -1.0);
        assert!(overhead.is_finite() && below.is_finite());
        assert!(
            overhead.max_element() >= below.max_element() - 1e-4,
            "overhead={overhead:?} below={below:?}"
        );
        assert!(below.min_element() >= 0.0);
    }

    #[test]
    fn factor_is_non_negative_everywhere() {
        let a = earth();
        let lut = bake_multiscatter_lut(&a, Vec3::splat(10.0), 24, 24, 24, 12);
        for texel in lut.texels() {
            assert!(texel.is_finite() && texel.min_element() >= 0.0, "texel={texel:?}");
        }
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let a = earth();
        let empty = bake_multiscatter_lut(&a, Vec3::splat(1.0), 0, 0, 8, 4);
        assert_eq!(sample_multiscatter_lut(&empty, &a, a.bottom_radius, 0.0), Vec3::ZERO);
        let mut flat = a;
        flat.top_radius = flat.bottom_radius;
        let (r, mu) = uv_to_r_mu_sun(&flat, 0.5, 0.5);
        assert!(r.is_finite() && mu.is_finite());
        let (u, v) = r_mu_sun_to_uv(&flat, r, mu);
        assert!(u.is_finite() && v.is_finite());
        let lut = bake_multiscatter_lut(&a, Vec3::splat(1.0), 8, 8, 8, 4);
        let s = sample_multiscatter_lut(&lut, &a, f32::NAN, f32::NAN);
        assert!(s.is_finite());
        let (ru, rv) = uv_to_r_mu_sun(&a, f32::INFINITY, f32::NAN);
        assert!(ru.is_finite() && rv.is_finite());
    }
}
