//! Angle-of-repose measurement for a settled granular pile.
//!
//! After pouring grains onto a floor and relaxing them to rest (see
//! [`GravitySettler`](super::gravity_settle::GravitySettler)), the heap holds a
//! characteristic free-surface slope: the **angle of repose**. It is the single
//! most-reported bulk property of a granular material — dry sand sits near
//! `34°`, gravel near `45°`, and a nearly frictionless powder collapses toward
//! `0°` — so being able to read it back off a settled packing closes the loop
//! between the DEM drivers and a measurable material parameter.
//!
//! This module measures the angle from the settled grain centres alone. It:
//!
//! 1. picks the pile axis as the horizontal centroid of all grain centres and
//!    the floor as the lowest centre;
//! 2. bins the grains by horizontal distance `ρ` from that axis and records the
//!    *surface* grain (the tallest centre) in each radial bin, giving a
//!    descending height profile `h(ρ)` from apex to base;
//! 3. fits a straight line to that profile by least squares and reports the
//!    repose angle as `atan(|slope|)`.
//!
//! A simpler conical estimate `atan(apex_height / base_radius)` is also exposed
//! for cross-checking against the regression. Everything here is a pure,
//! deterministic geometric analysis of the supplied arrays; nothing is derived
//! from Unreal Engine source.

use glam::{Vec2, Vec3};

/// Measured repose geometry of a settled pile.
///
/// Build one with [`AngleOfRepose::measure`]. Heights are measured relative to
/// the floor (the lowest grain centre), and radii relative to the pile axis
/// (the horizontal centroid of the grain centres).
#[derive(Clone, Debug, PartialEq)]
pub struct AngleOfRepose {
    grain_count: usize,
    axis_xy: Vec2,
    floor_height: f32,
    apex_height: f32,
    base_radius: f32,
    repose_angle: f32,
    cone_angle: f32,
    surface_profile: Vec<(f32, f32)>,
}

impl AngleOfRepose {
    /// Measures the repose geometry of a pile given parallel `positions` and
    /// `radii`, binning the free surface into `radial_bins` radial shells.
    ///
    /// Returns `None` unless the two arrays share the same non-zero length,
    /// every value is finite, every radius is strictly positive, `radial_bins`
    /// is non-zero, and the pile has a strictly positive horizontal extent
    /// (`base_radius > 0`). At least two non-empty radial bins are required so
    /// the surface slope can be fitted; a single tight column therefore yields
    /// `None`.
    #[must_use]
    pub fn measure(positions: &[Vec3], radii: &[f32], radial_bins: usize) -> Option<Self> {
        if positions.len() != radii.len() || positions.is_empty() {
            return None;
        }
        if radial_bins == 0 {
            return None;
        }
        if positions.iter().any(|p| !p.is_finite()) {
            return None;
        }
        if radii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return None;
        }

        let grain_count = positions.len();

        // Pile axis = horizontal centroid; floor = lowest grain centre.
        let mut sum_xy = Vec2::ZERO;
        let mut floor_height = f32::INFINITY;
        for p in positions {
            sum_xy += Vec2::new(p.x, p.y);
            floor_height = floor_height.min(p.z);
        }
        let axis_xy = sum_xy / grain_count as f32;

        // Per-grain radius from the axis and height above the floor.
        let mut radius_of = Vec::with_capacity(grain_count);
        let mut height_of = Vec::with_capacity(grain_count);
        let mut base_radius = 0.0_f32;
        let mut apex_height = 0.0_f32;
        for p in positions {
            let rho = (Vec2::new(p.x, p.y) - axis_xy).length();
            let h = p.z - floor_height;
            base_radius = base_radius.max(rho);
            apex_height = apex_height.max(h);
            radius_of.push(rho);
            height_of.push(h);
        }

        if base_radius <= 0.0 {
            return None;
        }

        // Surface profile: tallest grain in each radial bin.
        let bin_width = base_radius / radial_bins as f32;
        let mut surface: Vec<Option<(f32, f32)>> = vec![None; radial_bins];
        for (&rho, &h) in radius_of.iter().zip(height_of.iter()) {
            let mut bin = (rho / bin_width).floor() as usize;
            if bin >= radial_bins {
                bin = radial_bins - 1;
            }
            match surface[bin] {
                Some((_, best_h)) if best_h >= h => {}
                _ => surface[bin] = Some((rho, h)),
            }
        }
        let surface_profile: Vec<(f32, f32)> = surface.into_iter().flatten().collect();

        if surface_profile.len() < 2 {
            return None;
        }

        // Least-squares slope of height against radius, accumulated in f64.
        let n = surface_profile.len() as f64;
        let mut sum_r = 0.0_f64;
        let mut sum_h = 0.0_f64;
        for &(r, h) in &surface_profile {
            sum_r += r as f64;
            sum_h += h as f64;
        }
        let mean_r = sum_r / n;
        let mean_h = sum_h / n;
        let mut sxx = 0.0_f64;
        let mut sxy = 0.0_f64;
        for &(r, h) in &surface_profile {
            let dr = r as f64 - mean_r;
            sxx += dr * dr;
            sxy += dr * (h as f64 - mean_h);
        }
        if sxx <= 0.0 {
            return None;
        }
        let slope = sxy / sxx;
        let repose_angle = slope.abs().atan() as f32;

        let cone_angle = (apex_height as f64 / base_radius as f64).atan() as f32;

        Some(Self {
            grain_count,
            axis_xy,
            floor_height,
            apex_height,
            base_radius,
            repose_angle,
            cone_angle,
            surface_profile,
        })
    }

    /// Number of grains analysed.
    #[must_use]
    pub fn grain_count(&self) -> usize {
        self.grain_count
    }

    /// Horizontal pile axis (centroid of the grain centres in XY).
    #[must_use]
    pub fn axis_xy(&self) -> Vec2 {
        self.axis_xy
    }

    /// Floor level: the lowest grain-centre height.
    #[must_use]
    pub fn floor_height(&self) -> f32 {
        self.floor_height
    }

    /// Apex height above the floor.
    #[must_use]
    pub fn apex_height(&self) -> f32 {
        self.apex_height
    }

    /// Base radius: the largest horizontal distance of any grain from the axis.
    #[must_use]
    pub fn base_radius(&self) -> f32 {
        self.base_radius
    }

    /// Repose angle in radians, from the least-squares fit of the free surface.
    #[must_use]
    pub fn repose_angle(&self) -> f32 {
        self.repose_angle
    }

    /// Repose angle in degrees.
    #[must_use]
    pub fn repose_angle_degrees(&self) -> f32 {
        (self.repose_angle as f64 * 180.0 / std::f64::consts::PI) as f32
    }

    /// Conical estimate `atan(apex_height / base_radius)` in radians, a quick
    /// cross-check against the regression-based [`repose_angle`](Self::repose_angle).
    #[must_use]
    pub fn cone_angle(&self) -> f32 {
        self.cone_angle
    }

    /// Conical estimate in degrees.
    #[must_use]
    pub fn cone_angle_degrees(&self) -> f32 {
        (self.cone_angle as f64 * 180.0 / std::f64::consts::PI) as f32
    }

    /// Free-surface profile as `(radius, height)` samples, one per non-empty
    /// radial bin, ordered from the axis outward.
    #[must_use]
    pub fn surface_profile(&self) -> &[(f32, f32)] {
        &self.surface_profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a conical pile: apex at `(0, 0, height)`, base radius `base` at
    /// `z = 0`, with `rings` stacked rings of `spokes` grains each lying on the
    /// cone surface `h = height · (1 − ρ / base)`.
    fn cone(height: f32, base: f32, rings: usize, spokes: usize) -> (Vec<Vec3>, Vec<f32>) {
        let mut positions = Vec::new();
        for ri in 0..=rings {
            let z = height * ri as f32 / rings as f32;
            let rho = base * (1.0 - z / height);
            if rho <= 1e-4 {
                positions.push(Vec3::new(0.0, 0.0, z));
                continue;
            }
            for k in 0..spokes {
                let theta = 2.0 * std::f64::consts::PI * k as f64 / spokes as f64;
                let (s, c) = theta.sin_cos();
                positions.push(Vec3::new(rho * c as f32, rho * s as f32, z));
            }
        }
        let radii = vec![0.05_f32; positions.len()];
        (positions, radii)
    }

    #[test]
    fn rejects_invalid_inputs() {
        let (p, r) = cone(2.0, 2.0, 6, 8);
        // Length mismatch.
        assert!(AngleOfRepose::measure(&p, &r[..r.len() - 1], 8).is_none());
        // Empty.
        assert!(AngleOfRepose::measure(&[], &[], 8).is_none());
        // Zero bins.
        assert!(AngleOfRepose::measure(&p, &r, 0).is_none());
        // Non-finite position.
        let mut bad = p.clone();
        bad[0] = Vec3::splat(f32::NAN);
        assert!(AngleOfRepose::measure(&bad, &r, 8).is_none());
        // Non-positive radius.
        let mut badr = r.clone();
        badr[0] = 0.0;
        assert!(AngleOfRepose::measure(&p, &badr, 8).is_none());
    }

    #[test]
    fn vertical_column_has_no_repose_surface() {
        // All grains on the axis → base_radius 0 → None.
        let p: Vec<Vec3> = (0..5).map(|i| Vec3::new(0.0, 0.0, i as f32)).collect();
        let r = vec![0.1_f32; p.len()];
        assert!(AngleOfRepose::measure(&p, &r, 8).is_none());
    }

    #[test]
    fn forty_five_degree_cone() {
        let (p, r) = cone(4.0, 4.0, 8, 12);
        let m = AngleOfRepose::measure(&p, &r, 12).unwrap();
        assert!(
            (m.repose_angle_degrees() - 45.0).abs() < 1.0,
            "repose {}",
            m.repose_angle_degrees()
        );
        assert!(
            (m.cone_angle_degrees() - 45.0).abs() < 1.0,
            "cone {}",
            m.cone_angle_degrees()
        );
        assert!((m.base_radius() - 4.0).abs() < 1e-3);
        assert!((m.apex_height() - 4.0).abs() < 1e-3);
    }

    #[test]
    fn thirty_degree_cone() {
        // tan(30°) ≈ 0.5774 → height = base · tan(30°).
        let base = 4.0_f32;
        let height = base * (30.0_f64 * std::f64::consts::PI / 180.0).tan() as f32;
        let (p, r) = cone(height, base, 10, 16);
        let m = AngleOfRepose::measure(&p, &r, 14).unwrap();
        assert!(
            (m.repose_angle_degrees() - 30.0).abs() < 1.5,
            "repose {}",
            m.repose_angle_degrees()
        );
    }

    #[test]
    fn flat_disk_has_near_zero_repose() {
        // A single flat layer of grains: surface height constant → ~0°.
        let mut p = Vec::new();
        for gx in -4..=4 {
            for gy in -4..=4 {
                p.push(Vec3::new(gx as f32, gy as f32, 0.0));
            }
        }
        let r = vec![0.4_f32; p.len()];
        let m = AngleOfRepose::measure(&p, &r, 8).unwrap();
        assert!(
            m.repose_angle_degrees() < 1.0,
            "repose {}",
            m.repose_angle_degrees()
        );
    }

    #[test]
    fn surface_profile_descends_from_apex() {
        let (p, r) = cone(4.0, 4.0, 8, 12);
        let m = AngleOfRepose::measure(&p, &r, 12).unwrap();
        let prof = m.surface_profile();
        assert!(prof.len() >= 2);
        // Heights fall monotonically as radius grows.
        for w in prof.windows(2) {
            assert!(w[0].0 < w[1].0, "radius should increase");
            assert!(w[0].1 >= w[1].1 - 1e-4, "height should not increase");
        }
    }
}
