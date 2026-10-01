//! Transmittance LUT: bake and sample the atmosphere's spectral transmittance
//! to the top boundary, parameterized in the Bruneton/Hillaire `(r, mu)` layout.
//!
//! The transmittance LUT caches `T(r, mu) = exp(-∫ σ_t ds)` from a point at
//! planet-centred radius `r` along a direction whose cosine with the local
//! zenith is `mu`, integrated to the first atmosphere boundary. Because the sky
//! and multiple-scattering passes evaluate transmittance millions of times, it
//! is tabulated once and sampled bilinearly at runtime.
//!
//! The texel layout follows Bruneton 2017 / Hillaire 2020: rather than a naive
//! `(altitude, mu)` grid — which wastes resolution near the horizon — the
//! parameterization warps `mu` by the *distance to the atmosphere boundary*, so
//! grazing rays (where transmittance changes fastest) receive proportionally
//! more texels. The warp uses the auxiliary lengths
//!
//! * `H  = sqrt(top² - bottom²)` — the ground-tangent chord to the top shell,
//! * `ρ  = sqrt(r² - bottom²)`   — the horizon distance from radius `r`,
//! * `d`                          — the distance from `(r, mu)` to the top shell,
//!
//! with `u = (d - d_min) / (d_max - d_min)` and `v = ρ / H`, where
//! `d_min = top - r` (straight up) and `d_max = ρ + H` (the horizon-grazing
//! ray). Inverting this gives [`uv_to_r_mu`]; the forward map is
//! [`r_mu_to_uv`].
//!
//! # Conventions
//! * `r` is clamped to `[bottom_radius, top_radius]`; `mu = cos θ_view ∈ [-1, 1]`.
//! * `u` warps `mu` by boundary distance, `v` warps `r` by horizon distance;
//!   both are clamped to `[0, 1]` (clamp-to-edge).
//! * Transmittance is spectral linear-RGB in `(0, 1]`, backed by
//!   [`crate::gi::atmosphere::transmittance::transmittance_to_boundary`].
//! * Every result is finite (never `NaN`); degenerate atmospheres and texels
//!   fall back to full transmittance / zero-size tables return `Vec3::ONE`.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the value
//!   method. Every function is a deterministic pure function with no RNG, I/O,
//!   GPU, or `unsafe`.

use alloc::vec::Vec;
use bevy_math::Vec3;

use crate::gi::atmosphere::medium::Atmosphere;
use crate::gi::atmosphere::transmittance::transmittance_to_boundary;

/// Smallest denominator / distance treated as non-degenerate (km-scale).
const EPSILON: f32 = 1.0e-6;

/// A baked 2D spectral transmittance table in `(u = mu-warp, v = r-warp)` space.
///
/// Stored row-major (`v` outer, `u` inner) as linear-RGB [`Vec3`] texels.
#[derive(Clone, Debug, PartialEq)]
pub struct TransmittanceLut {
    width: usize,
    height: usize,
    texels: Vec<Vec3>,
}

impl TransmittanceLut {
    /// Texel-grid width (`u` / `mu`-warp axis).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Texel-grid height (`v` / `r`-warp axis).
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
    ///
    /// Returns `0` for an empty table so callers never index out of bounds.
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
            return Vec3::ONE;
        }
        self.texels[self.texel_index(x, y)]
    }

    /// Clamp-to-edge bilinear fetch at continuous UV in `[0, 1]²`.
    ///
    /// `u`/`v` are clamped to the valid range; an empty table yields
    /// `Vec3::ONE` (full transmittance).
    #[inline]
    pub fn sample_uv(&self, u: f32, v: f32) -> Vec3 {
        if self.width == 0 || self.height == 0 || self.texels.is_empty() {
            return Vec3::ONE;
        }
        let u = clamp01(u);
        let v = clamp01(v);
        // Map UV to texel-centre space: coordinate c ∈ [-0.5, n-0.5].
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

/// Replaces non-finite channels with `0` and clamps each to `[0, 1]`.
#[inline]
fn sanitize_rgb(rgb: Vec3) -> Vec3 {
    Vec3::new(
        if rgb.x.is_finite() { rgb.x.clamp(0.0, 1.0) } else { 0.0 },
        if rgb.y.is_finite() { rgb.y.clamp(0.0, 1.0) } else { 0.0 },
        if rgb.z.is_finite() { rgb.z.clamp(0.0, 1.0) } else { 0.0 },
    )
}

/// Ground-tangent chord `H = sqrt(top² - bottom²)` for `atmosphere`.
///
/// Clamped non-negative; a degenerate shell returns `0`.
#[inline]
fn tangent_chord(atmosphere: &Atmosphere) -> f32 {
    let top = atmosphere.top_radius;
    let bottom = atmosphere.bottom_radius;
    (top * top - bottom * bottom).max(0.0).sqrt()
}

/// Converts a UV coordinate in `[0, 1]²` to physical `(r, mu)`.
///
/// `u` is the boundary-distance warp of `mu`; `v` is the horizon-distance warp
/// of `r`. The result is clamped to the physical shell: `r ∈ [bottom, top]` and
/// `mu ∈ [-1, 1]`. A degenerate atmosphere falls back to `(bottom, 1)`.
#[inline]
pub fn uv_to_r_mu(atmosphere: &Atmosphere, u: f32, v: f32) -> (f32, f32) {
    let bottom = atmosphere.bottom_radius.max(0.0);
    let top = atmosphere.top_radius.max(bottom);
    let h = tangent_chord(atmosphere);
    if !(h > EPSILON) {
        return (bottom, 1.0);
    }
    let u = clamp01(u);
    let v = clamp01(v);
    let rho = h * v;
    let r = (rho * rho + bottom * bottom).sqrt().clamp(bottom, top);
    let d_min = (top - r).max(0.0);
    let d_max = rho + h;
    let d = d_min + u * (d_max - d_min);
    let mu = if d <= EPSILON {
        1.0
    } else {
        let value = (h * h - rho * rho - d * d) / (2.0 * r * d);
        if value.is_finite() {
            value.clamp(-1.0, 1.0)
        } else {
            1.0
        }
    };
    (r, mu)
}

/// Converts physical `(r, mu)` to a UV coordinate in `[0, 1]²`.
///
/// Inverse of [`uv_to_r_mu`]. Inputs are clamped (`r ∈ [bottom, top]`,
/// `mu ∈ [-1, 1]`); a degenerate atmosphere returns `(0, 0)`.
#[inline]
pub fn r_mu_to_uv(atmosphere: &Atmosphere, r: f32, mu: f32) -> (f32, f32) {
    let bottom = atmosphere.bottom_radius.max(0.0);
    let top = atmosphere.top_radius.max(bottom);
    let h = tangent_chord(atmosphere);
    if !(h > EPSILON) {
        return (0.0, 0.0);
    }
    let r = if r.is_finite() { r.clamp(bottom, top) } else { bottom };
    let mu = if mu.is_finite() { mu.clamp(-1.0, 1.0) } else { 1.0 };
    let rho = (r * r - bottom * bottom).max(0.0).sqrt();
    // Distance to the top shell: positive root of d² + 2 r mu d + (r² - top²) = 0.
    let disc = (r * r * (mu * mu - 1.0) + top * top).max(0.0);
    let d = (-r * mu + disc.sqrt()).max(0.0);
    let d_min = (top - r).max(0.0);
    let d_max = rho + h;
    let denom = d_max - d_min;
    let u = if denom > EPSILON {
        clamp01((d - d_min) / denom)
    } else {
        0.0
    };
    let v = clamp01(rho / h);
    (u, v)
}

/// Builds the view direction at radius `r` whose local-zenith cosine is `mu`.
///
/// The point sits on the `+Y` axis, so the zenith is `+Y`; the direction lies
/// in the `XY` plane, `(sin θ, mu, 0)`.
#[inline]
fn direction_for_mu(mu: f32) -> Vec3 {
    let mu = mu.clamp(-1.0, 1.0);
    let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
    Vec3::new(sin_theta, mu, 0.0)
}

/// Bakes a [`TransmittanceLut`] of the given dimensions for `atmosphere`.
///
/// Each texel decodes to `(r, mu)` at its centre and integrates
/// [`transmittance_to_boundary`] with `samples` midpoint steps. Zero dimensions
/// yield an empty table.
#[inline]
pub fn bake_transmittance_lut(
    atmosphere: &Atmosphere,
    width: usize,
    height: usize,
    samples: u32,
) -> TransmittanceLut {
    if width == 0 || height == 0 {
        return TransmittanceLut {
            width: 0,
            height: 0,
            texels: Vec::new(),
        };
    }
    let mut texels = Vec::with_capacity(width * height);
    let samples = samples.max(1);
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let (r, mu) = uv_to_r_mu(atmosphere, u, v);
            let origin = Vec3::new(0.0, r, 0.0);
            let dir = direction_for_mu(mu);
            let t = transmittance_to_boundary(atmosphere, origin, dir, samples);
            texels.push(sanitize_rgb(t));
        }
    }
    TransmittanceLut {
        width,
        height,
        texels,
    }
}

/// Samples the baked transmittance for physical `(r, mu)` via bilinear fetch.
///
/// Encodes `(r, mu)` through [`r_mu_to_uv`] and samples `lut` with clamp-to-edge
/// filtering. Spectral linear-RGB in `(0, 1]`.
#[inline]
pub fn sample_transmittance_lut(
    lut: &TransmittanceLut,
    atmosphere: &Atmosphere,
    r: f32,
    mu: f32,
) -> Vec3 {
    let (u, v) = r_mu_to_uv(atmosphere, r, mu);
    lut.sample_uv(u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn earth() -> Atmosphere {
        Atmosphere::earth()
    }

    #[test]
    fn uv_param_roundtrip_is_stable() {
        let a = earth();
        for &v in &[0.05f32, 0.25, 0.5, 0.75, 0.95] {
            for &u in &[0.02f32, 0.3, 0.5, 0.8, 0.98] {
                let (r, mu) = uv_to_r_mu(&a, u, v);
                let (u2, v2) = r_mu_to_uv(&a, r, mu);
                assert!((u - u2).abs() < 2e-3, "u={u} u2={u2} (r={r} mu={mu})");
                assert!((v - v2).abs() < 2e-3, "v={v} v2={v2}");
            }
        }
    }

    #[test]
    fn decoded_radius_and_mu_in_range() {
        let a = earth();
        for y in 0..8 {
            for x in 0..8 {
                let u = (x as f32 + 0.5) / 8.0;
                let v = (y as f32 + 0.5) / 8.0;
                let (r, mu) = uv_to_r_mu(&a, u, v);
                assert!(r >= a.bottom_radius - 1e-3 && r <= a.top_radius + 1e-3, "r={r}");
                assert!((-1.0..=1.0).contains(&mu), "mu={mu}");
            }
        }
    }

    #[test]
    fn baked_sample_matches_direct_integration() {
        let a = earth();
        let lut = bake_transmittance_lut(&a, 128, 64, 64);
        // Sample at a handful of texel centres and compare to the analytic march.
        for &(x, y) in &[(10usize, 8usize), (64, 32), (100, 50)] {
            let u = (x as f32 + 0.5) / 128.0;
            let v = (y as f32 + 0.5) / 64.0;
            let (r, mu) = uv_to_r_mu(&a, u, v);
            let got = sample_transmittance_lut(&lut, &a, r, mu);
            let origin = Vec3::new(0.0, r, 0.0);
            let dir = direction_for_mu(mu);
            let want = transmittance_to_boundary(&a, origin, dir, 64);
            assert!((got - want).length() < 2e-3, "got={got:?} want={want:?}");
        }
    }

    #[test]
    fn transmittance_decreases_toward_horizon() {
        let a = earth();
        let lut = bake_transmittance_lut(&a, 256, 64, 96);
        let r = a.bottom_radius + 1.0;
        let mut prev = sample_transmittance_lut(&lut, &a, r, 1.0);
        // Lower mu -> longer slant path -> smaller transmittance.
        for i in 1..=20 {
            let mu = 1.0 - i as f32 * 0.045;
            let t = sample_transmittance_lut(&lut, &a, r, mu);
            assert!(t.x <= prev.x + 2e-3, "not monotonic at mu={mu}: {t:?} prev {prev:?}");
            assert!(t.min_element() >= 0.0 && t.max_element() <= 1.0 + 1e-4);
            prev = t;
        }
    }

    #[test]
    fn straight_up_at_top_is_near_unit() {
        let a = earth();
        let lut = bake_transmittance_lut(&a, 64, 32, 32);
        let t = sample_transmittance_lut(&lut, &a, a.top_radius, 1.0);
        assert!(t.min_element() > 0.99, "expected ~unit transmittance: {t:?}");
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let a = earth();
        // Empty table returns full transmittance.
        let empty = bake_transmittance_lut(&a, 0, 0, 16);
        assert_eq!(sample_transmittance_lut(&empty, &a, a.bottom_radius, 0.5), Vec3::ONE);
        // Degenerate shell.
        let mut flat = a;
        flat.top_radius = flat.bottom_radius;
        let (r, mu) = uv_to_r_mu(&flat, 0.5, 0.5);
        assert!(r.is_finite() && mu.is_finite());
        let (u, v) = r_mu_to_uv(&flat, r, mu);
        assert!(u.is_finite() && v.is_finite());
        // NaN inputs stay finite.
        let lut = bake_transmittance_lut(&a, 16, 16, 16);
        let t = sample_transmittance_lut(&lut, &a, f32::NAN, f32::NAN);
        assert!(t.is_finite());
        let (ru, rv) = uv_to_r_mu(&a, f32::NAN, f32::INFINITY);
        assert!(ru.is_finite() && rv.is_finite());
    }
}
