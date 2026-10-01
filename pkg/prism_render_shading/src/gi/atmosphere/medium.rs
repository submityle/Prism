//! Atmosphere medium model: planetary geometry, exponential density profiles,
//! and the spectral extinction / scattering coefficients they induce.
//!
//! The medium follows the layered model of Hillaire 2020 ("A Scalable and
//! Production Ready Sky and Atmosphere"). Two scattering species share a common
//! planet-centred radial frame, each with its own exponential density
//! `ρ(h) = exp(-h / H)` for scale height `H`, plus an optional ozone absorption
//! layer with a triangular ("tent") altitude profile:
//!
//! * **Rayleigh** — spectral molecular scattering, no absorption, scale height
//!   `~8 km`.
//! * **Mie** — grey aerosol scattering with additional absorption
//!   (`extinction ≥ scattering`), scale height `~1.2 km`.
//! * **Ozone** — pure absorption, peaking around `25 km`, contributing nothing
//!   to scattering (shapes the sky's twilight hue).
//!
//! All lengths are in kilometres and all coefficients are per-kilometre, so an
//! optical depth is `coefficient · distance` with both in consistent units.
//!
//! # Conventions
//! * Altitude is measured from the planet surface (`bottom_radius`) and clamped
//!   to `[0, thickness]`; densities use the clamped altitude.
//! * [`Atmosphere::extinction`] (`σ_t`) is always `≥` [`Atmosphere::scattering`]
//!   (`σ_s`) per channel, because extinction folds in absorption.
//! * Every coefficient is finite and non-negative (never `NaN`); non-finite or
//!   non-positive scale heights fall back to a surface density of `0`.
//! * Transcendental maths goes through [`bevy_math::ops`]. Every method is a
//!   deterministic pure function with no RNG, I/O, GPU, or `unsafe`.

use bevy_math::{ops, Vec3};

/// Physically based layered atmosphere parameters (kilometre / per-kilometre).
///
/// Construct the Earth-like preset with [`Atmosphere::earth`], or set fields
/// directly for other planets or ablated media (e.g. Mie-free test cases).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atmosphere {
    /// Planet (ground) radius in kilometres.
    pub bottom_radius: f32,
    /// Atmosphere top radius in kilometres (`> bottom_radius`).
    pub top_radius: f32,
    /// Spectral Rayleigh scattering coefficient at the surface (per km).
    pub rayleigh_scattering: Vec3,
    /// Rayleigh density scale height in kilometres.
    pub rayleigh_scale_height: f32,
    /// Grey Mie scattering coefficient at the surface (per km).
    pub mie_scattering: f32,
    /// Grey Mie extinction coefficient at the surface (per km, `≥` scattering).
    pub mie_extinction: f32,
    /// Mie density scale height in kilometres.
    pub mie_scale_height: f32,
    /// Mie phase asymmetry `g` in `(-1, 1)`.
    pub mie_g: f32,
    /// Spectral ozone absorption coefficient at the tent peak (per km).
    pub ozone_absorption: Vec3,
    /// Ozone tent centre altitude in kilometres.
    pub ozone_center: f32,
    /// Ozone tent half-width in kilometres (density is `0` beyond this).
    pub ozone_width: f32,
}

impl Atmosphere {
    /// Earth-like preset mirroring Hillaire 2020's reference parameters.
    ///
    /// Radii `6360 / 6460 km`; Rayleigh `σ_s = (5.802, 13.558, 33.1)·10⁻³`
    /// with `H = 8 km`; Mie `σ_s = 3.996·10⁻³`, `σ_t = 4.44·10⁻³`,
    /// `H = 1.2 km`, `g = 0.8`; ozone `σ_a = (0.650, 1.881, 0.085)·10⁻³`
    /// centred at `25 km` with a `15 km` half-width.
    #[inline]
    pub fn earth() -> Self {
        Self {
            bottom_radius: 6360.0,
            top_radius: 6460.0,
            rayleigh_scattering: Vec3::new(5.802e-3, 13.558e-3, 33.1e-3),
            rayleigh_scale_height: 8.0,
            mie_scattering: 3.996e-3,
            mie_extinction: 4.44e-3,
            mie_scale_height: 1.2,
            mie_g: 0.8,
            ozone_absorption: Vec3::new(0.650e-3, 1.881e-3, 0.085e-3),
            ozone_center: 25.0,
            ozone_width: 15.0,
        }
    }

    /// Atmosphere shell thickness `top_radius - bottom_radius` (km), clamped
    /// non-negative.
    #[inline]
    pub fn thickness(&self) -> f32 {
        (self.top_radius - self.bottom_radius).max(0.0)
    }

    /// Altitude of a planet-centred position in kilometres, clamped to
    /// `[0, thickness]`.
    #[inline]
    pub fn altitude_at(&self, position: Vec3) -> f32 {
        let r = position.length();
        if r.is_finite() {
            (r - self.bottom_radius).clamp(0.0, self.thickness())
        } else {
            0.0
        }
    }

    /// Normalised Rayleigh density `exp(-h / H_R)` in `[0, 1]`.
    #[inline]
    pub fn rayleigh_density(&self, altitude: f32) -> f32 {
        exp_density(altitude, self.rayleigh_scale_height)
    }

    /// Normalised Mie density `exp(-h / H_M)` in `[0, 1]`.
    #[inline]
    pub fn mie_density(&self, altitude: f32) -> f32 {
        exp_density(altitude, self.mie_scale_height)
    }

    /// Normalised ozone density: a triangular tent peaking at
    /// `ozone_center`, linearly falling to `0` at `± ozone_width`.
    #[inline]
    pub fn ozone_density(&self, altitude: f32) -> f32 {
        let h = altitude.max(0.0);
        let width = self.ozone_width;
        if !(width > 0.0) || !h.is_finite() {
            return 0.0;
        }
        let t = 1.0 - (h - self.ozone_center).abs() / width;
        t.clamp(0.0, 1.0)
    }

    /// Spectral Rayleigh scattering coefficient `σ_s^R(h)` (per km).
    #[inline]
    pub fn rayleigh_scattering_at(&self, altitude: f32) -> Vec3 {
        sanitize_rgb(self.rayleigh_scattering * self.rayleigh_density(altitude))
    }

    /// Grey Mie scattering coefficient `σ_s^M(h)` (per km).
    #[inline]
    pub fn mie_scattering_at(&self, altitude: f32) -> f32 {
        clamp_non_negative(self.mie_scattering) * self.mie_density(altitude)
    }

    /// Grey Mie extinction coefficient `σ_t^M(h)` (per km).
    #[inline]
    pub fn mie_extinction_at(&self, altitude: f32) -> f32 {
        clamp_non_negative(self.mie_extinction) * self.mie_density(altitude)
    }

    /// Spectral ozone absorption coefficient `σ_a^O(h)` (per km).
    #[inline]
    pub fn ozone_absorption_at(&self, altitude: f32) -> Vec3 {
        sanitize_rgb(self.ozone_absorption * self.ozone_density(altitude))
    }

    /// Total scattering coefficient `σ_s(h)` = Rayleigh + Mie (per km).
    ///
    /// Ozone is pure absorption and contributes nothing here.
    #[inline]
    pub fn scattering(&self, altitude: f32) -> Vec3 {
        let rayleigh = self.rayleigh_scattering_at(altitude);
        let mie = self.mie_scattering_at(altitude);
        sanitize_rgb(rayleigh + Vec3::splat(mie))
    }

    /// Total extinction coefficient `σ_t(h)` = Rayleigh scattering + Mie
    /// extinction + ozone absorption (per km).
    ///
    /// Always `≥` [`Atmosphere::scattering`] per channel.
    #[inline]
    pub fn extinction(&self, altitude: f32) -> Vec3 {
        let rayleigh = self.rayleigh_scattering_at(altitude);
        let mie = self.mie_extinction_at(altitude);
        let ozone = self.ozone_absorption_at(altitude);
        sanitize_rgb(rayleigh + Vec3::splat(mie) + ozone)
    }
}

/// Exponential density `exp(-h / H)`, clamped to `[0, 1]`.
///
/// Returns `0` for a non-positive or non-finite scale height.
#[inline]
fn exp_density(altitude: f32, scale_height: f32) -> f32 {
    let h = altitude.max(0.0);
    if !(scale_height > 0.0) || !h.is_finite() {
        return 0.0;
    }
    let value = ops::exp(-h / scale_height);
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
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
        clamp_non_negative(rgb.x),
        clamp_non_negative(rgb.y),
        clamp_non_negative(rgb.z),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_is_unit_at_surface_and_decreasing() {
        let a = Atmosphere::earth();
        assert!((a.rayleigh_density(0.0) - 1.0).abs() < 1e-6);
        assert!((a.mie_density(0.0) - 1.0).abs() < 1e-6);
        let mut prev_r = a.rayleigh_density(0.0);
        let mut prev_m = a.mie_density(0.0);
        for h in 1..=50 {
            let r = a.rayleigh_density(h as f32);
            let m = a.mie_density(h as f32);
            assert!(r <= prev_r + 1e-7 && (0.0..=1.0).contains(&r), "r={r}");
            assert!(m <= prev_m + 1e-7 && (0.0..=1.0).contains(&m), "m={m}");
            prev_r = r;
            prev_m = m;
        }
    }

    #[test]
    fn rayleigh_density_matches_analytic_exp() {
        let a = Atmosphere::earth();
        for h in [0.0f32, 2.0, 8.0, 16.0, 32.0] {
            let got = a.rayleigh_density(h);
            let want = ops::exp(-h / a.rayleigh_scale_height);
            assert!((got - want).abs() < 1e-6, "h={h} got={got} want={want}");
        }
    }

    #[test]
    fn ozone_tent_peaks_at_center_and_vanishes_outside() {
        let a = Atmosphere::earth();
        assert!((a.ozone_density(a.ozone_center) - 1.0).abs() < 1e-6);
        // Edges of the tent are zero.
        assert!(a.ozone_density(a.ozone_center + a.ozone_width) < 1e-6);
        assert!(a.ozone_density(a.ozone_center - a.ozone_width) < 1e-6);
        // Far outside stays clamped to zero.
        assert_eq!(a.ozone_density(a.ozone_center + 2.0 * a.ozone_width), 0.0);
        // Half way up a limb is ~0.5.
        let mid = a.ozone_density(a.ozone_center + 0.5 * a.ozone_width);
        assert!((mid - 0.5).abs() < 1e-5, "mid={mid}");
    }

    #[test]
    fn extinction_dominates_scattering_per_channel() {
        let a = Atmosphere::earth();
        for h in [0.0f32, 1.0, 10.0, 25.0, 60.0, 100.0] {
            let s = a.scattering(h);
            let t = a.extinction(h);
            assert!(t.x >= s.x - 1e-9, "h={h} tx={} sx={}", t.x, s.x);
            assert!(t.y >= s.y - 1e-9, "h={h}");
            assert!(t.z >= s.z - 1e-9, "h={h}");
            assert!(s.min_element() >= 0.0 && t.min_element() >= 0.0);
        }
    }

    #[test]
    fn altitude_is_clamped_to_shell() {
        let a = Atmosphere::earth();
        let below = a.altitude_at(Vec3::new(0.0, a.bottom_radius - 10.0, 0.0));
        assert_eq!(below, 0.0);
        let above = a.altitude_at(Vec3::new(0.0, a.top_radius + 10.0, 0.0));
        assert!((above - a.thickness()).abs() < 1e-3, "above={above}");
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let mut a = Atmosphere::earth();
        a.rayleigh_scale_height = 0.0;
        a.mie_scale_height = -1.0;
        a.ozone_width = 0.0;
        assert_eq!(a.rayleigh_density(5.0), 0.0);
        assert_eq!(a.mie_density(5.0), 0.0);
        assert_eq!(a.ozone_density(25.0), 0.0);
        let t = a.extinction(f32::NAN);
        assert!(t.is_finite());
        let s = a.scattering(f32::INFINITY);
        assert!(s.is_finite());
        assert!(a.altitude_at(Vec3::splat(f32::NAN)).is_finite());
    }
}
