//! Radial distribution function `g(r)` for a sphere-centre point cloud.
//!
//! Once a granular scene has been authored and settled, a single scalar like
//! the solid volume fraction (see
//! [`PackingDiagnostics`](super::packing_diagnostics::PackingDiagnostics)) tells
//! you *how* dense the pack is, but not *how* the grains are arranged. The
//! radial distribution function answers the arrangement question: given a grain
//! picked at random, `g(r)` is the density of other grain centres a distance
//! `r` away, normalised by the density a structureless (ideal-gas) cloud of the
//! same average density would show.
//!
//! * `g(r) → 0` for `r` smaller than a grain diameter — hard spheres cannot
//!   overlap, so no centres sit that close.
//! * `g(r)` has a sharp first peak near the mean nearest-neighbour spacing (one
//!   grain diameter for a monodisperse touching pack), then softer peaks at the
//!   second and third shells.
//! * `g(r) → 1` at large `r` — beyond a few diameters the arrangement looks
//!   uniform and the local density matches the global average.
//!
//! This module histograms every centre-to-centre distance below the analysis
//! cutoff `r_max = bin_width · bin_count` and normalises each shell by the
//! ideal-gas expectation
//! `N · ρ · V_shell` with number density `ρ = N / V_box` and shell volume
//! `V_shell = (4/3)·π·(r_{k+1}³ − r_k³)`. The pair search reuses the same
//! uniform-spatial-hash neighbour walk as the rest of the granular stack, so
//! the histogram is an expected `O(n)` rather than the naive `O(n²)`.
//! Everything here is a pure, deterministic analysis of the supplied centres;
//! nothing is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::HashMap;

/// Integer cell coordinate in the uniform spatial hash.
type Cell = (i32, i32, i32);

fn cell_of(point: Vec3, origin: Vec3, inv_cell: f32) -> Cell {
    let local = (point - origin) * inv_cell;
    (
        local.x.floor() as i32,
        local.y.floor() as i32,
        local.z.floor() as i32,
    )
}

/// A binned radial distribution function `g(r)` computed from a point cloud.
///
/// Build one with [`RadialDistribution::compute`]. The histogram covers the
/// half-open range `[0, r_max)` split into `bin_count` equal-width bins; bin `k`
/// spans `[k·bin_width, (k+1)·bin_width)`.
#[derive(Clone, Debug, PartialEq)]
pub struct RadialDistribution {
    grain_count: usize,
    box_volume: f64,
    bin_width: f32,
    /// Unordered pair counts per bin: `pair_counts[k]` is the number of
    /// `{i, j}` pairs whose separation falls in bin `k`.
    pair_counts: Vec<u64>,
}

impl RadialDistribution {
    /// Computes `g(r)` from grain centres `positions` inside the axis-aligned
    /// reference box `[box_min, box_max]`.
    ///
    /// The analysis uses `bin_count` shells of width `bin_width`, so the cutoff
    /// radius is `r_max = bin_width · bin_count`. Radii are intentionally not
    /// consumed: `g(r)` is a function of grain *centres*, and the reference box
    /// supplies the number density `ρ = N / V_box`.
    ///
    /// Returns `None` unless every position is finite, the box is finite with a
    /// strictly positive extent on every axis, `bin_width` is finite and
    /// strictly positive, and `bin_count` is non-zero. A cloud with fewer than
    /// two grains is accepted and yields all-zero pair counts (and hence a
    /// zero `g(r)`).
    #[must_use]
    pub fn compute(
        positions: &[Vec3],
        box_min: Vec3,
        box_max: Vec3,
        bin_width: f32,
        bin_count: usize,
    ) -> Option<Self> {
        if bin_count == 0 {
            return None;
        }
        if !bin_width.is_finite() || bin_width <= 0.0 {
            return None;
        }
        if !(box_min.is_finite() && box_max.is_finite()) {
            return None;
        }
        let extent = box_max - box_min;
        if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
            return None;
        }
        if positions.iter().any(|p| !p.is_finite()) {
            return None;
        }

        let grain_count = positions.len();
        let box_volume = (extent.x as f64) * (extent.y as f64) * (extent.z as f64);
        let r_max = bin_width * bin_count as f32;

        let mut pair_counts = vec![0_u64; bin_count];

        if grain_count >= 2 {
            // Cell size equal to the cutoff means every pair closer than
            // `r_max` lies within the 3×3×3 neighbourhood of a grain's cell.
            let inv_cell = 1.0 / r_max;
            let origin = positions
                .iter()
                .copied()
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);

            let mut grid: HashMap<Cell, Vec<usize>> = HashMap::new();
            for (i, &p) in positions.iter().enumerate() {
                grid.entry(cell_of(p, origin, inv_cell))
                    .or_default()
                    .push(i);
            }

            for (i, &ci) in positions.iter().enumerate() {
                let (cx, cy, cz) = cell_of(ci, origin, inv_cell);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let key = (cx + dx, cy + dy, cz + dz);
                            let Some(indices) = grid.get(&key) else {
                                continue;
                            };
                            for &j in indices {
                                if j <= i {
                                    continue;
                                }
                                let dist = ci.distance(positions[j]);
                                if dist >= r_max {
                                    continue;
                                }
                                let bin = (dist / bin_width).floor() as usize;
                                // Guard against the `dist == r_max` edge slipping
                                // past the `>=` check through float rounding.
                                if bin < bin_count {
                                    pair_counts[bin] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(Self {
            grain_count,
            box_volume,
            bin_width,
            pair_counts,
        })
    }

    /// Number of grains analysed.
    #[must_use]
    pub fn grain_count(&self) -> usize {
        self.grain_count
    }

    /// Number of histogram shells.
    #[must_use]
    pub fn bin_count(&self) -> usize {
        self.pair_counts.len()
    }

    /// Width of each histogram shell.
    #[must_use]
    pub fn bin_width(&self) -> f32 {
        self.bin_width
    }

    /// Analysis cutoff radius `r_max = bin_width · bin_count`.
    #[must_use]
    pub fn cutoff(&self) -> f32 {
        self.bin_width * self.pair_counts.len() as f32
    }

    /// Average number density `ρ = N / V_box`.
    #[must_use]
    pub fn number_density(&self) -> f32 {
        (self.grain_count as f64 / self.box_volume) as f32
    }

    /// Unordered pair counts per bin: entry `k` is the number of `{i, j}`
    /// pairs whose centre separation falls in bin `k`.
    #[must_use]
    pub fn pair_counts(&self) -> &[u64] {
        &self.pair_counts
    }

    /// Inclusive-lower, exclusive-upper radial bounds `[r_k, r_{k+1})` of bin
    /// `k`. Returns `None` when `k` is out of range.
    #[must_use]
    pub fn shell_bounds(&self, k: usize) -> Option<(f32, f32)> {
        if k >= self.pair_counts.len() {
            return None;
        }
        let lo = k as f32 * self.bin_width;
        let hi = (k + 1) as f32 * self.bin_width;
        Some((lo, hi))
    }

    /// Radius at the centre of bin `k`, `(k + 0.5)·bin_width`. Returns `None`
    /// when `k` is out of range.
    #[must_use]
    pub fn bin_center(&self, k: usize) -> Option<f32> {
        if k >= self.pair_counts.len() {
            return None;
        }
        Some((k as f32 + 0.5) * self.bin_width)
    }

    /// Normalised radial distribution function value for bin `k`.
    ///
    /// `g(r_k) = 2·pair_counts[k] / (N · ρ · V_shell)` where the factor of two
    /// turns the unordered pair count into the ordered count the ideal-gas
    /// expectation `N · ρ · V_shell` is written against. Returns `None` when
    /// `k` is out of range, and `Some(0.0)` whenever fewer than two grains were
    /// analysed (no pairs exist to normalise).
    #[must_use]
    pub fn g(&self, k: usize) -> Option<f32> {
        if k >= self.pair_counts.len() {
            return None;
        }
        if self.grain_count < 2 {
            return Some(0.0);
        }
        let r_lo = k as f64 * self.bin_width as f64;
        let r_hi = (k + 1) as f64 * self.bin_width as f64;
        let shell_volume =
            4.0 / 3.0 * std::f64::consts::PI * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
        if shell_volume <= 0.0 {
            return Some(0.0);
        }
        let n = self.grain_count as f64;
        let density = n / self.box_volume;
        let ideal_ordered = n * density * shell_volume;
        let ordered = 2.0 * self.pair_counts[k] as f64;
        Some((ordered / ideal_ordered) as f32)
    }

    /// The full `g(r)` curve, one value per bin, in bin order.
    #[must_use]
    pub fn g_curve(&self) -> Vec<f32> {
        (0..self.pair_counts.len())
            .map(|k| self.g(k).unwrap_or(0.0))
            .collect()
    }

    /// Total number of unordered pairs counted across every bin (i.e. all pairs
    /// closer than the cutoff radius).
    #[must_use]
    pub fn total_pairs(&self) -> u64 {
        self.pair_counts.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> (Vec3, Vec3) {
        (Vec3::ZERO, Vec3::splat(1.0))
    }

    #[test]
    fn rejects_invalid_inputs() {
        let (lo, hi) = unit_box();
        let p = vec![Vec3::ZERO, Vec3::splat(0.5)];
        // Zero bins.
        assert!(RadialDistribution::compute(&p, lo, hi, 0.1, 0).is_none());
        // Non-positive / non-finite bin width.
        assert!(RadialDistribution::compute(&p, lo, hi, 0.0, 4).is_none());
        assert!(RadialDistribution::compute(&p, lo, hi, f32::NAN, 4).is_none());
        // Degenerate box.
        assert!(RadialDistribution::compute(&p, lo, lo, 0.1, 4).is_none());
        // Non-finite box.
        assert!(RadialDistribution::compute(&p, Vec3::splat(f32::INFINITY), hi, 0.1, 4).is_none());
        // Non-finite position.
        let bad = vec![Vec3::ZERO, Vec3::splat(f32::NAN)];
        assert!(RadialDistribution::compute(&bad, lo, hi, 0.1, 4).is_none());
    }

    #[test]
    fn empty_and_single_cloud_have_zero_pairs() {
        let (lo, hi) = unit_box();
        for cloud in [vec![], vec![Vec3::splat(0.3)]] {
            let rdf = RadialDistribution::compute(&cloud, lo, hi, 0.1, 5).unwrap();
            assert_eq!(rdf.total_pairs(), 0);
            assert!(rdf.g_curve().iter().all(|&v| v == 0.0));
            assert_eq!(rdf.bin_count(), 5);
        }
    }

    #[test]
    fn two_grains_land_in_the_expected_bin() {
        let (lo, hi) = unit_box();
        // Separation 0.35 → bin index floor(0.35 / 0.1) = 3.
        let p = vec![Vec3::new(0.1, 0.5, 0.5), Vec3::new(0.45, 0.5, 0.5)];
        let rdf = RadialDistribution::compute(&p, lo, hi, 0.1, 8).unwrap();
        assert_eq!(rdf.total_pairs(), 1);
        assert_eq!(rdf.pair_counts()[3], 1);
        for (k, &c) in rdf.pair_counts().iter().enumerate() {
            if k != 3 {
                assert_eq!(c, 0);
            }
        }
    }

    #[test]
    fn pair_beyond_cutoff_is_dropped() {
        let (lo, hi) = (Vec3::ZERO, Vec3::splat(10.0));
        // Separation 5.0, cutoff = 0.5 * 4 = 2.0 → no pair recorded.
        let p = vec![Vec3::new(1.0, 1.0, 1.0), Vec3::new(6.0, 1.0, 1.0)];
        let rdf = RadialDistribution::compute(&p, lo, hi, 0.5, 4).unwrap();
        assert_eq!(rdf.total_pairs(), 0);
    }

    #[test]
    fn shell_and_center_geometry() {
        let (lo, hi) = unit_box();
        let p = vec![Vec3::ZERO, Vec3::splat(0.2)];
        let rdf = RadialDistribution::compute(&p, lo, hi, 0.25, 4).unwrap();
        assert_eq!(rdf.cutoff(), 1.0);
        let (slo, shi) = rdf.shell_bounds(2).unwrap();
        assert!((slo - 0.5).abs() < 1e-6);
        assert!((shi - 0.75).abs() < 1e-6);
        assert!((rdf.bin_center(2).unwrap() - 0.625).abs() < 1e-6);
        assert!(rdf.shell_bounds(4).is_none());
        assert!(rdf.bin_center(4).is_none());
    }

    #[test]
    fn number_density_matches_definition() {
        let (lo, hi) = (Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        let p = vec![
            Vec3::splat(0.1),
            Vec3::splat(0.2),
            Vec3::splat(0.3),
            Vec3::splat(0.4),
        ];
        let rdf = RadialDistribution::compute(&p, lo, hi, 0.1, 4).unwrap();
        // 4 grains / 8 unit box = 0.5.
        assert!((rdf.number_density() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn histogram_matches_brute_force() {
        let (lo, hi) = (Vec3::ZERO, Vec3::splat(4.0));
        // Deterministic pseudo-cloud via a cheap LCG so the test is stable.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) * 4.0
        };
        let p: Vec<Vec3> = (0..200)
            .map(|_| Vec3::new(next(), next(), next()))
            .collect();

        let bin_width = 0.2_f32;
        let bin_count = 6;
        let r_max = bin_width * bin_count as f32;
        let rdf = RadialDistribution::compute(&p, lo, hi, bin_width, bin_count).unwrap();

        let mut brute = vec![0_u64; bin_count];
        for i in 0..p.len() {
            for j in (i + 1)..p.len() {
                let d = p[i].distance(p[j]);
                if d < r_max {
                    let b = (d / bin_width).floor() as usize;
                    if b < bin_count {
                        brute[b] += 1;
                    }
                }
            }
        }
        assert_eq!(rdf.pair_counts(), brute.as_slice());
    }

    #[test]
    fn simple_cubic_lattice_peaks_at_the_spacing() {
        // 5×5×5 simple-cubic lattice, spacing 1.0.
        let n = 5;
        let spacing = 1.0_f32;
        let mut p = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    p.push(Vec3::new(i as f32, j as f32, k as f32) * spacing);
                }
            }
        }
        // The reference box pads the lattice by half a cell on every side so
        // each grain owns exactly one unit cell: density = 125 / 5³ = 1.0,
        // the true simple-cubic density. A tight bounding box would overstate
        // the density and depress the whole g(r) curve.
        let span = (n - 1) as f32 * spacing;
        let (lo, hi) = (
            Vec3::splat(-0.5 * spacing),
            Vec3::splat(span + 0.5 * spacing),
        );
        // Bin width 0.3 keeps the nearest-neighbour shell (r = 1.0 → bin 3)
        // comfortably inside a bin rather than on a bin boundary.
        let rdf = RadialDistribution::compute(&p, lo, hi, 0.3, 10).unwrap();
        assert!((rdf.number_density() - 1.0).abs() < 1e-5);

        // No pairs closer than the lattice spacing (bins below r = 1.0).
        for k in 0..3 {
            assert_eq!(rdf.pair_counts()[k], 0, "bin {k} should be empty");
        }
        // First shell (r = 1.0) falls in bin 3 = [0.9, 1.2).
        assert!(rdf.pair_counts()[3] > 0);
        // g(r) peaks there well above the uniform baseline of 1.
        assert!(rdf.g(3).unwrap() > 1.0);
    }
}
