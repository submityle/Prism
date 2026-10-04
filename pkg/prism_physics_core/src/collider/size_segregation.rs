//! Size-segregation / vertical stratification diagnostics.
//!
//! Polydisperse granular media rarely stay mixed: shaking, pouring, and
//! shearing drive size segregation, most famously the "Brazil-nut" rise of
//! large grains. This module quantifies vertical stratification by binning
//! grains into horizontal (constant-`z`) layers and comparing the mean grain
//! size of each layer.
//!
//! Two complementary scalars are reported:
//!
//! * **Segregation intensity** `I = σ_b / d̄`, the count-weighted coefficient
//!   of variation of the per-layer mean diameters, where
//!   `σ_b² = (1/N) Σ_k n_k (d_k − d̄)²` is the between-layer variance. `I = 0`
//!   means every layer shares the global mean size (well mixed).
//! * **Vertical size gradient**, the count-weighted least-squares slope of the
//!   per-layer mean diameter against layer height, plus a dimensionless form
//!   normalised by the global mean size and the domain height. A positive
//!   gradient means coarse grains concentrate near the top.
//!
//! The module composes depth-binning (as in the porosity profile) with pure
//! size statistics and does not couple to the simulation step.

use glam::Vec3;

/// Vertical size-segregation diagnostics for a sphere packing.
#[derive(Debug, Clone, PartialEq)]
pub struct SizeSegregation {
    layer_bounds: Vec<(f32, f32)>,
    layer_centers: Vec<f32>,
    layer_counts: Vec<usize>,
    layer_mean_diameters: Vec<f32>,
    grain_count: usize,
    overall_mean_diameter: f32,
    segregation_intensity: f32,
    size_gradient: f32,
    normalized_gradient: f32,
}

impl SizeSegregation {
    /// Bins grains by centre height into `layers` equal `z`-slabs between
    /// `box_min.z` and `box_max.z` and computes stratification statistics.
    ///
    /// Returns `None` when lengths disagree, inputs are empty/non-finite, any
    /// radius is non-positive, `layers == 0`, the box is degenerate in `z`, or
    /// fewer than two layers end up populated (a gradient needs two points).
    pub fn analyze(
        positions: &[Vec3],
        radii: &[f32],
        box_min: Vec3,
        box_max: Vec3,
        layers: usize,
    ) -> Option<Self> {
        if positions.is_empty() || positions.len() != radii.len() || layers == 0 {
            return None;
        }
        if !box_min.is_finite() || !box_max.is_finite() {
            return None;
        }
        let z0 = box_min.z;
        let z1 = box_max.z;
        let height = z1 - z0;
        if !height.is_finite() || height <= 0.0 {
            return None;
        }
        for (p, &r) in positions.iter().zip(radii.iter()) {
            if !p.is_finite() || !r.is_finite() || r <= 0.0 {
                return None;
            }
        }

        let slab = height / layers as f32;
        let inv_slab = 1.0 / slab;
        let mut layer_counts = vec![0usize; layers];
        // Accumulate diameters in f64 per layer.
        let mut layer_diam_sum = vec![0.0_f64; layers];

        for (p, &r) in positions.iter().zip(radii.iter()) {
            // Grains whose centre falls outside the box are ignored.
            if p.z < z0 || p.z > z1 {
                continue;
            }
            let mut idx = ((p.z - z0) * inv_slab).floor() as isize;
            if idx < 0 {
                idx = 0;
            }
            if idx as usize >= layers {
                idx = layers as isize - 1;
            }
            let k = idx as usize;
            layer_counts[k] += 1;
            layer_diam_sum[k] += (2.0 * r) as f64;
        }

        let mut layer_bounds = Vec::with_capacity(layers);
        let mut layer_centers = Vec::with_capacity(layers);
        let mut layer_mean_diameters = Vec::with_capacity(layers);
        for (k, (&count, &sum)) in layer_counts.iter().zip(layer_diam_sum.iter()).enumerate() {
            let lo = z0 + slab * k as f32;
            let hi = z0 + slab * (k + 1) as f32;
            layer_bounds.push((lo, hi));
            layer_centers.push(0.5 * (lo + hi));
            let mean = if count > 0 {
                (sum / count as f64) as f32
            } else {
                0.0
            };
            layer_mean_diameters.push(mean);
        }

        let grain_count: usize = layer_counts.iter().sum();
        if grain_count == 0 {
            return None;
        }
        let populated = layer_counts.iter().filter(|&&c| c > 0).count();
        if populated < 2 {
            return None;
        }

        // Global count-weighted mean diameter.
        let total_diam: f64 = layer_diam_sum.iter().sum();
        let overall_mean_diameter = (total_diam / grain_count as f64) as f32;

        // Between-layer variance σ_b² = (1/N) Σ n_k (d_k − d̄)².
        let dbar = overall_mean_diameter as f64;
        let mut var_b = 0.0_f64;
        for (&count, mean) in layer_counts.iter().zip(layer_mean_diameters.iter()) {
            if count == 0 {
                continue;
            }
            let diff = *mean as f64 - dbar;
            var_b += count as f64 * diff * diff;
        }
        var_b /= grain_count as f64;
        let sigma_b = var_b.max(0.0).sqrt();
        let segregation_intensity = if dbar > 0.0 {
            (sigma_b / dbar) as f32
        } else {
            0.0
        };

        // Count-weighted least-squares slope of d_k vs z_center.
        let mut sw = 0.0_f64;
        let mut swz = 0.0_f64;
        let mut swd = 0.0_f64;
        let mut swzz = 0.0_f64;
        let mut swzd = 0.0_f64;
        for ((&count, &center), mean) in layer_counts
            .iter()
            .zip(layer_centers.iter())
            .zip(layer_mean_diameters.iter())
        {
            if count == 0 {
                continue;
            }
            let w = count as f64;
            let z = center as f64;
            let d = *mean as f64;
            sw += w;
            swz += w * z;
            swd += w * d;
            swzz += w * z * z;
            swzd += w * z * d;
        }
        let denom = sw * swzz - swz * swz;
        let size_gradient = if denom.abs() > 1e-12 {
            ((sw * swzd - swz * swd) / denom) as f32
        } else {
            0.0
        };
        // Dimensionless: slope · H / d̄.
        let normalized_gradient = if overall_mean_diameter > 0.0 {
            size_gradient * height / overall_mean_diameter
        } else {
            0.0
        };

        Some(Self {
            layer_bounds,
            layer_centers,
            layer_counts,
            layer_mean_diameters,
            grain_count,
            overall_mean_diameter,
            segregation_intensity,
            size_gradient,
            normalized_gradient,
        })
    }

    /// Number of layers.
    pub fn layer_count(&self) -> usize {
        self.layer_bounds.len()
    }

    /// Total grains binned (centres inside the box).
    pub fn grain_count(&self) -> usize {
        self.grain_count
    }

    /// `(z_lo, z_hi)` bounds of each layer, bottom to top.
    pub fn layer_bounds(&self) -> &[(f32, f32)] {
        &self.layer_bounds
    }

    /// Mid-height of each layer.
    pub fn layer_centers(&self) -> &[f32] {
        &self.layer_centers
    }

    /// Grain count in each layer.
    pub fn layer_counts(&self) -> &[usize] {
        &self.layer_counts
    }

    /// Mean grain diameter in each layer (`0` for empty layers).
    pub fn layer_mean_diameters(&self) -> &[f32] {
        &self.layer_mean_diameters
    }

    /// Global count-weighted mean diameter `d̄`.
    pub fn overall_mean_diameter(&self) -> f32 {
        self.overall_mean_diameter
    }

    /// Segregation intensity `I = σ_b / d̄` (`0` = well mixed).
    pub fn segregation_intensity(&self) -> f32 {
        self.segregation_intensity
    }

    /// Count-weighted least-squares slope of mean diameter vs height
    /// (diameter units per length). Positive = coarse grains on top.
    pub fn size_gradient(&self) -> f32 {
        self.size_gradient
    }

    /// Dimensionless size gradient `slope · H / d̄`.
    pub fn normalized_gradient(&self) -> f32 {
        self.normalized_gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_layer(z: f32, diameter: f32, n: usize) -> (Vec<Vec3>, Vec<f32>) {
        let r = 0.5 * diameter;
        let mut p = Vec::with_capacity(n);
        let mut rad = Vec::with_capacity(n);
        for i in 0..n {
            p.push(Vec3::new(i as f32, 0.0, z));
            rad.push(r);
        }
        (p, rad)
    }

    #[test]
    fn rejects_bad_input() {
        let p = [Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)];
        let r = [0.5, 0.5];
        assert!(SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::ZERO, 2).is_none());
        assert!(
            SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::new(0.0, 0.0, 2.0), 0).is_none()
        );
        assert!(SizeSegregation::analyze(&p, &[0.5], Vec3::ZERO, Vec3::Z, 2).is_none());
        assert!(SizeSegregation::analyze(&[], &[], Vec3::ZERO, Vec3::Z, 2).is_none());
        // Degenerate (zero-height) box.
        assert!(
            SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::new(1.0, 1.0, 0.0), 2).is_none()
        );
    }

    #[test]
    fn needs_two_populated_layers() {
        // All grains land in one layer → no gradient defined.
        let (p, r) = grid_layer(0.25, 0.2, 5);
        let res = SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::new(10.0, 10.0, 1.0), 4);
        assert!(res.is_none());
    }

    #[test]
    fn well_mixed_has_zero_intensity() {
        // Same diameter in both layers → I = 0, gradient = 0.
        let (mut p, mut r) = grid_layer(0.25, 0.4, 6);
        let (p2, r2) = grid_layer(0.75, 0.4, 6);
        p.extend(p2);
        r.extend(r2);
        let s =
            SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::new(10.0, 10.0, 1.0), 2).unwrap();
        assert_eq!(s.layer_count(), 2);
        assert_eq!(s.grain_count(), 12);
        assert!((s.overall_mean_diameter() - 0.4).abs() < 1e-5);
        assert!(s.segregation_intensity() < 1e-5);
        assert!(s.size_gradient().abs() < 1e-5);
        assert!(s.normalized_gradient().abs() < 1e-5);
    }

    #[test]
    fn coarse_on_top_gives_positive_gradient() {
        // Small grains low, large grains high → positive gradient.
        let (mut p, mut r) = grid_layer(0.25, 0.2, 8); // bottom, d=0.2
        let (p2, r2) = grid_layer(0.75, 0.6, 8); // top, d=0.6
        p.extend(p2);
        r.extend(r2);
        let s =
            SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::new(100.0, 100.0, 1.0), 2).unwrap();
        // Layer means: bottom 0.2, top 0.6.
        let d = s.layer_mean_diameters();
        assert!((d[0] - 0.2).abs() < 1e-5);
        assert!((d[1] - 0.6).abs() < 1e-5);
        // Overall mean = 0.4; σ_b = 0.2 → I = 0.5.
        assert!((s.overall_mean_diameter() - 0.4).abs() < 1e-5);
        assert!((s.segregation_intensity() - 0.5).abs() < 1e-4);
        // Centers 0.25 & 0.75; slope = (0.6-0.2)/0.5 = 0.8.
        assert!((s.size_gradient() - 0.8).abs() < 1e-3);
        // Normalized = 0.8 · 1.0 / 0.4 = 2.0.
        assert!((s.normalized_gradient() - 2.0).abs() < 1e-3);
    }

    #[test]
    fn coarse_on_bottom_gives_negative_gradient() {
        let (mut p, mut r) = grid_layer(0.25, 0.6, 8); // bottom, d=0.6
        let (p2, r2) = grid_layer(0.75, 0.2, 8); // top, d=0.2
        p.extend(p2);
        r.extend(r2);
        let s =
            SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::new(100.0, 100.0, 1.0), 2).unwrap();
        assert!(s.size_gradient() < 0.0);
        assert!(s.normalized_gradient() < 0.0);
        assert!((s.segregation_intensity() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn counts_and_bounds_are_consistent() {
        let (mut p, mut r) = grid_layer(0.25, 0.2, 3);
        let (p2, r2) = grid_layer(0.75, 0.4, 5);
        p.extend(p2);
        r.extend(r2);
        let s =
            SizeSegregation::analyze(&p, &r, Vec3::ZERO, Vec3::new(10.0, 10.0, 1.0), 2).unwrap();
        assert_eq!(s.layer_counts(), &[3, 5]);
        assert_eq!(s.grain_count(), 8);
        let b = s.layer_bounds();
        assert!((b[0].0 - 0.0).abs() < 1e-6 && (b[0].1 - 0.5).abs() < 1e-6);
        assert!((b[1].0 - 0.5).abs() < 1e-6 && (b[1].1 - 1.0).abs() < 1e-6);
        // Count-weighted overall mean = (3·0.2 + 5·0.4)/8 = 0.325.
        assert!((s.overall_mean_diameter() - 0.325).abs() < 1e-5);
    }
}
