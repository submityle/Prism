//! Deterministic path-guided importance sampling over a hemisphere — CPU golden.
//!
//! Practical path guiding (Müller et al. 2017, *Practical Path Guiding for
//! Efficient Light-Transport Simulation*) learns *where* radiance comes from
//! and biases future samples towards those directions, dramatically cutting
//! variance in hard-to-sample scenes.  The original work stores that learned
//! distribution in an adaptively subdivided directional `SD-tree`.  This module
//! is the backend-neutral reference for the *sampling* half of that idea using
//! a fully deterministic, closed-form structure instead of any learned,
//! neural, or online-adapted model:
//!
//! * [`GuidingDistribution`] is a piecewise-constant directional density over
//!   the upper hemisphere about `+Z`, discretised on an *equal-solid-angle*
//!   grid (uniform in `cos(theta)` by `phi`).  Equal-area cells make the
//!   normalisation and the per-cell solid angle a single constant, so the
//!   piecewise-constant density integrates to exactly one by construction.
//! * [`GuidingDistribution::accumulate`] folds directional radiance statistics
//!   into the grid; [`pdf`](GuidingDistribution::pdf) evaluates the normalised
//!   solid-angle density; [`sample`](GuidingDistribution::sample) draws a
//!   direction whose returned pdf is *identical* to `pdf(dir)` (the inverse-CDF
//!   draw and the density evaluation address the same cell).
//! * [`GuidingDistribution::mis_weight`] returns the balance-heuristic weight
//!   that blends the guided density with a cosine-weighted hemisphere lobe, the
//!   standard defensive mixture that keeps guiding robust when the learned
//!   statistics are poor or empty.
//!
//! # Conventions
//! * Directions live in a local tangent frame with the surface normal along
//!   `+Z`; the lower hemisphere (`z <= 0`) has zero density.  The cosine lobe
//!   used for mixing is `cos(theta) / pi`, matching
//!   [`super::super::sample::mapping::cosine_hemisphere`].
//! * The grid is `nz` cosine bands by `nphi` azimuth sectors in row-major
//!   order (`index = band * nphi + sector`).  Each cell subtends the same solid
//!   angle `2*pi / (nz * nphi)`, stored implicitly; only per-cell accumulated
//!   weights and their total live in the `f32` buffer twin.
//! * Densities are solid-angle pdfs: non-negative and integrating to one over
//!   the hemisphere.  Accumulated weights must be non-negative; non-finite or
//!   non-positive contributions are ignored.
//! * Degenerate fallback: an *empty* distribution (no accumulated weight)
//!   reports and samples the cosine-weighted hemisphere exactly, so `sample`
//!   and `pdf` stay mutually consistent and never divide by zero or return
//!   `NaN`.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no global state, and no `unsafe`.  The only allocation is the weight grid
//!   owned by [`GuidingDistribution`].

use alloc::vec::Vec;
use bevy_math::{ops, Vec3};
use core::f32::consts::TAU;

use super::super::sample::mapping::{cosine_hemisphere, cosine_hemisphere_pdf};
use super::restir::luminance;

/// Normalises `dir`, returning `+Z` for a degenerate (near-zero) input so the
/// cell lookup and trigonometry never produce `NaN`s.
#[inline]
fn normalize_or_z(dir: Vec3) -> Vec3 {
    let len_sq = dir.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        dir * len_sq.sqrt().recip()
    } else {
        Vec3::Z
    }
}

/// A piecewise-constant guided directional density over the `+Z` hemisphere.
///
/// Weights are accumulated per equal-solid-angle cell and normalised on demand,
/// so the structure can keep absorbing statistics without an explicit
/// normalisation pass.  See the module docs for the grid layout and
/// conventions.
#[derive(Clone, Debug, PartialEq)]
pub struct GuidingDistribution {
    /// Number of cosine (`cos(theta)`) bands; always `>= 1`.
    nz: usize,
    /// Number of azimuth (`phi`) sectors; always `>= 1`.
    nphi: usize,
    /// Row-major `nz * nphi` non-negative accumulated weights.
    weights: Vec<f32>,
    /// Sum of all cell weights (`sum_i weights[i]`).
    total: f32,
}

impl GuidingDistribution {
    /// Creates a zero (empty) distribution with `nz` cosine bands by `nphi`
    /// azimuth sectors.  Both counts are clamped to at least `1` so the grid is
    /// never empty.  An empty distribution behaves as a cosine-weighted
    /// hemisphere (the degenerate fallback).
    #[inline]
    pub fn new(nz: usize, nphi: usize) -> Self {
        let nz = nz.max(1);
        let nphi = nphi.max(1);
        Self {
            nz,
            nphi,
            weights: alloc::vec::from_elem(0.0, nz * nphi),
            total: 0.0,
        }
    }

    /// Number of cosine bands.
    #[inline]
    pub fn band_count(&self) -> usize {
        self.nz
    }

    /// Number of azimuth sectors.
    #[inline]
    pub fn sector_count(&self) -> usize {
        self.nphi
    }

    /// Total number of cells (`band_count * sector_count`).
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.weights.len()
    }

    /// Sum of all accumulated cell weights.
    #[inline]
    pub fn total_weight(&self) -> f32 {
        self.total
    }

    /// Whether the distribution has no usable statistics (cosine fallback).
    #[inline]
    pub fn is_empty(&self) -> bool {
        // `total` only ever accumulates non-negative, finite contributions.
        self.total <= 0.0
    }

    /// The constant solid angle subtended by one cell: `2*pi / (nz * nphi)`.
    #[inline]
    fn cell_solid_angle(&self) -> f32 {
        TAU / (self.nz as f32 * self.nphi as f32)
    }

    /// Returns the cell index for a direction, or `None` for the lower
    /// hemisphere / a degenerate direction.
    #[inline]
    fn cell_index(&self, dir: Vec3) -> Option<usize> {
        let d = normalize_or_z(dir);
        let cos_theta = d.z;
        if cos_theta <= 0.0 {
            return None;
        }
        // Cosine band: cos_theta in [band / nz, (band + 1) / nz].
        let band = ((cos_theta * self.nz as f32).floor() as usize).min(self.nz - 1);
        // Azimuth sector: phi in [0, 2*pi).
        let mut phi = ops::atan2(d.y, d.x);
        if phi < 0.0 {
            phi += TAU;
        }
        let frac = (phi / TAU).clamp(0.0, 1.0);
        let sector = ((frac * self.nphi as f32).floor() as usize).min(self.nphi - 1);
        Some(band * self.nphi + sector)
    }

    /// Folds a non-negative scalar statistic `value` into the cell containing
    /// `dir`.  Non-finite / non-positive values and lower-hemisphere directions
    /// are ignored.
    #[inline]
    pub fn accumulate(&mut self, dir: Vec3, value: f32) {
        if !value.is_finite() || value <= 0.0 {
            return;
        }
        if let Some(idx) = self.cell_index(dir) {
            self.weights[idx] += value;
            self.total += value;
        }
    }

    /// Convenience: accumulate the luminance of an RGB radiance sample arriving
    /// from direction `dir`, the usual path-guiding statistic.
    #[inline]
    pub fn accumulate_radiance(&mut self, dir: Vec3, radiance: Vec3) {
        self.accumulate(dir, luminance(radiance));
    }

    /// Evaluates the normalised solid-angle density for `dir`.
    ///
    /// Returns `0` for the lower hemisphere.  When the distribution is empty it
    /// falls back to the cosine-weighted density `cos(theta) / pi` so that it
    /// stays consistent with [`sample`](Self::sample)'s fallback.  Otherwise it
    /// returns `(weight_cell / total) / cell_solid_angle`.
    #[inline]
    pub fn pdf(&self, dir: Vec3) -> f32 {
        let d = normalize_or_z(dir);
        if d.z <= 0.0 {
            return 0.0;
        }
        if self.is_empty() {
            return cosine_hemisphere_pdf(d.z);
        }
        match self.cell_index(d) {
            Some(idx) => {
                let prob = self.weights[idx] / self.total;
                let pdf = prob / self.cell_solid_angle();
                if pdf.is_finite() {
                    pdf.max(0.0)
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }

    /// Draws a direction from the distribution given two uniforms in `[0, 1]`.
    ///
    /// Returns the unit direction (upper hemisphere) and its solid-angle pdf,
    /// which equals [`pdf`](Self::pdf) evaluated at that direction.  An empty
    /// distribution draws from the cosine-weighted hemisphere instead.
    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> (Vec3, f32) {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        if self.is_empty() {
            let local = cosine_hemisphere(u, v);
            let dir = Vec3::new(local[0], local[1], local[2]);
            return (dir, cosine_hemisphere_pdf(dir.z));
        }

        // Inverse-CDF cell selection. Keep the target strictly below `total` so
        // a cell is always found and the in-cell residual stays in [0, 1).
        let target = (u * self.total).min(self.total * (1.0 - 1e-6));
        let mut acc = 0.0f32;
        let mut chosen = self.weights.len() - 1;
        let mut residual = 0.5f32;
        for (k, &w) in self.weights.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            let next = acc + w;
            if target < next {
                chosen = k;
                residual = ((target - acc) / w).clamp(0.0, 1.0);
                break;
            }
            acc = next;
        }

        let band = chosen / self.nphi;
        let sector = chosen % self.nphi;

        // Uniform placement inside the chosen equal-area cell.
        let cos_theta = ((band as f32 + residual) / self.nz as f32).clamp(0.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = (sector as f32 + v) / self.nphi as f32 * TAU;
        let dir = Vec3::new(sin_theta * ops::cos(phi), sin_theta * ops::sin(phi), cos_theta);

        let prob = self.weights[chosen] / self.total;
        let pdf = (prob / self.cell_solid_angle()).max(0.0);
        (dir, pdf)
    }

    /// Balance-heuristic MIS weight for the *guided* strategy against a
    /// cosine-weighted hemisphere lobe: `p_guide / (p_guide + p_cosine)`.
    ///
    /// Returns a value in `[0, 1]`.  For an empty distribution the guided
    /// density equals the cosine density, so the weight is `0.5` — an even
    /// split that gracefully defers to the cosine lobe until statistics exist.
    #[inline]
    pub fn mis_weight(&self, dir: Vec3) -> f32 {
        let d = normalize_or_z(dir);
        if d.z <= 0.0 {
            return 0.0;
        }
        let p_guide = self.pdf(d);
        let p_cosine = cosine_hemisphere_pdf(d.z);
        let denom = p_guide + p_cosine;
        if denom <= 0.0 || !denom.is_finite() {
            return 0.0;
        }
        (p_guide / denom).clamp(0.0, 1.0)
    }

    /// Defensive mixture density: `fraction * p_guide + (1 - fraction) *
    /// p_cosine`, with `fraction` clamped to `[0, 1]`.
    ///
    /// This is the one-sample-MIS density of a guided/cosine mixture sampler,
    /// provided for callers that estimate with the combined density.
    #[inline]
    pub fn mixture_pdf(&self, dir: Vec3, guide_fraction: f32) -> f32 {
        let f = guide_fraction.clamp(0.0, 1.0);
        let d = normalize_or_z(dir);
        if d.z <= 0.0 {
            return 0.0;
        }
        let p_guide = self.pdf(d);
        let p_cosine = cosine_hemisphere_pdf(d.z);
        (f * p_guide + (1.0 - f) * p_cosine).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerically integrates `pdf` over the hemisphere with a deterministic
    /// uniform-solid-angle quadrature, returning the estimated integral.
    fn integrate_pdf(dist: &GuidingDistribution, n: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..n {
            // Uniform over solid angle: cos(theta) uniform in [0, 1], phi in
            // [0, 2*pi). Golden-ratio azimuth decorrelates the strata.
            let cos_theta = (i as f32 + 0.5) / n as f32;
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let phi = TAU * ((i as f32 * 0.618_034).fract());
            let dir = Vec3::new(sin_theta * ops::cos(phi), sin_theta * ops::sin(phi), cos_theta);
            sum += dist.pdf(dir);
        }
        // Each sample carries solid angle 2*pi / n.
        sum * TAU / n as f32
    }

    #[test]
    fn new_clamps_dimensions_and_starts_empty() {
        let d = GuidingDistribution::new(0, 0);
        assert_eq!(d.band_count(), 1);
        assert_eq!(d.sector_count(), 1);
        assert_eq!(d.cell_count(), 1);
        assert!(d.is_empty());
        assert_eq!(d.total_weight(), 0.0);
    }

    #[test]
    fn accumulate_ignores_degenerate_inputs() {
        let mut d = GuidingDistribution::new(4, 8);
        d.accumulate(Vec3::Z, -1.0);
        d.accumulate(Vec3::Z, 0.0);
        d.accumulate(Vec3::Z, f32::NAN);
        d.accumulate(Vec3::Z, f32::INFINITY);
        // Lower hemisphere is ignored.
        d.accumulate(Vec3::NEG_Z, 5.0);
        assert!(d.is_empty());
        assert_eq!(d.total_weight(), 0.0);
    }

    #[test]
    fn empty_distribution_matches_cosine_fallback() {
        let d = GuidingDistribution::new(8, 16);
        for (u, v) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.0, 0.0)] {
            let (dir, pdf) = d.sample(u, v);
            // Direction matches the cosine-hemisphere mapping exactly.
            let expected = cosine_hemisphere(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
            assert!((dir.x - expected[0]).abs() < 1e-6);
            assert!((dir.y - expected[1]).abs() < 1e-6);
            assert!((dir.z - expected[2]).abs() < 1e-6);
            // pdf matches the cosine density and sample/pdf stay consistent.
            assert!((pdf - cosine_hemisphere_pdf(dir.z)).abs() < 1e-6);
            assert!((pdf - d.pdf(dir)).abs() < 1e-6);
        }
    }

    #[test]
    fn lower_hemisphere_has_zero_density() {
        let mut d = GuidingDistribution::new(4, 8);
        d.accumulate(Vec3::Z, 1.0);
        assert_eq!(d.pdf(Vec3::NEG_Z), 0.0);
        assert_eq!(d.pdf(Vec3::new(0.0, 0.0, -0.5)), 0.0);
        assert_eq!(d.mis_weight(Vec3::NEG_Z), 0.0);
        assert_eq!(d.mixture_pdf(Vec3::NEG_Z, 0.5), 0.0);
    }

    #[test]
    fn empty_pdf_integrates_to_one() {
        let d = GuidingDistribution::new(8, 16);
        let integral = integrate_pdf(&d, 40_000);
        assert!((integral - 1.0).abs() < 0.02, "integral={integral}");
    }

    #[test]
    fn populated_pdf_integrates_to_one() {
        let mut d = GuidingDistribution::new(8, 16);
        // Accumulate an uneven set of directional statistics.
        d.accumulate(normalize(0.1, 0.1, 1.0), 5.0);
        d.accumulate(normalize(0.8, 0.0, 0.6), 2.0);
        d.accumulate(normalize(-0.3, 0.5, 0.8), 1.0);
        d.accumulate(normalize(0.0, -0.9, 0.4), 3.0);
        let integral = integrate_pdf(&d, 60_000);
        assert!((integral - 1.0).abs() < 0.03, "integral={integral}");
    }

    #[test]
    fn sample_pdf_is_consistent_with_pdf() {
        let mut d = GuidingDistribution::new(6, 12);
        d.accumulate(normalize(0.2, 0.1, 1.0), 4.0);
        d.accumulate(normalize(-0.6, 0.2, 0.7), 2.0);
        d.accumulate(normalize(0.1, -0.8, 0.5), 1.0);

        let n = 64;
        for i in 0..n {
            for j in 0..n {
                let u = (i as f32 + 0.5) / n as f32;
                let v = (j as f32 + 0.5) / n as f32;
                let (dir, pdf) = d.sample(u, v);
                // Upper hemisphere, unit length.
                assert!(dir.z >= 0.0, "dir.z={}", dir.z);
                assert!((dir.length() - 1.0).abs() < 1e-4, "len={}", dir.length());
                // The drawn pdf must equal the density at the drawn direction.
                let p = d.pdf(dir);
                assert!(pdf.is_finite() && p.is_finite());
                assert!((pdf - p).abs() <= 1e-4 * p.max(1.0), "pdf={pdf} p={p}");
            }
        }
    }

    #[test]
    fn sampling_concentrates_toward_accumulated_direction() {
        // Pour all statistics into one direction; most samples should land in
        // the same cosine band, and the density there must dominate.
        let mut d = GuidingDistribution::new(8, 16);
        let hot = normalize(0.0, 0.0, 1.0); // straight up: top cosine band
        d.accumulate(hot, 100.0);

        let hot_pdf = d.pdf(hot);
        let side = normalize(1.0, 0.0, 0.05); // grazing: near-zero band
        assert!(hot_pdf > d.pdf(side), "hot={hot_pdf} side={}", d.pdf(side));

        let n = 4000;
        let mut near = 0usize;
        for i in 0..n {
            let u = (i as f32 + 0.5) / n as f32;
            let v = ((i as f32 * 0.618_034).fract()).clamp(0.0, 1.0);
            let (dir, _) = d.sample(u, v);
            if dir.z > 0.875 {
                // Top band of 8: cos(theta) in [7/8, 1].
                near += 1;
            }
        }
        // All weight is in the top band, so (nearly) every sample lands there.
        assert!(near as f32 / n as f32 > 0.95, "near frac={}", near as f32 / n as f32);
    }

    #[test]
    fn mis_weight_is_half_when_empty_and_in_range_when_populated() {
        let empty = GuidingDistribution::new(4, 8);
        assert!((empty.mis_weight(Vec3::Z) - 0.5).abs() < 1e-5);

        let mut d = GuidingDistribution::new(8, 16);
        d.accumulate(Vec3::Z, 10.0);
        for dir in [
            normalize(0.0, 0.0, 1.0),
            normalize(0.5, 0.5, 0.7),
            normalize(-0.3, 0.1, 0.9),
            normalize(0.9, 0.0, 0.1),
        ] {
            let w = d.mis_weight(dir);
            assert!((0.0..=1.0).contains(&w), "w={w}");
        }
        // Where guided density dominates, the guided weight exceeds 0.5.
        assert!(d.mis_weight(Vec3::Z) > 0.5);
    }

    #[test]
    fn mixture_pdf_interpolates_endpoints() {
        let mut d = GuidingDistribution::new(8, 16);
        d.accumulate(Vec3::Z, 10.0);
        let dir = normalize(0.1, 0.0, 1.0);
        let pg = d.pdf(dir);
        let pc = cosine_hemisphere_pdf(normalize_or_z(dir).z);
        assert!((d.mixture_pdf(dir, 1.0) - pg).abs() < 1e-5);
        assert!((d.mixture_pdf(dir, 0.0) - pc).abs() < 1e-5);
        let mid = d.mixture_pdf(dir, 0.5);
        assert!((mid - 0.5 * (pg + pc)).abs() < 1e-5);
    }

    #[test]
    fn results_are_deterministic() {
        let build = || {
            let mut d = GuidingDistribution::new(6, 12);
            d.accumulate(normalize(0.2, 0.3, 0.9), 2.0);
            d.accumulate(normalize(-0.5, 0.1, 0.8), 1.5);
            d
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
        assert_eq!(a.sample(0.3, 0.7), b.sample(0.3, 0.7));
        let dir = normalize(0.2, 0.3, 0.9);
        assert_eq!(a.pdf(dir), b.pdf(dir));
    }

    /// Test helper: build a normalised [`Vec3`] from raw components.
    fn normalize(x: f32, y: f32, z: f32) -> Vec3 {
        normalize_or_z(Vec3::new(x, y, z))
    }
}
