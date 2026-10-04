//! Quality diagnostics for a generated sphere packing.
//!
//! The packers in this crate
//! ([`pack_spheres`](super::sphere_packing::pack_spheres),
//! [`pack_spheres_grid`](super::grid_sphere_packing::pack_spheres_grid),
//! [`pack_spheres_from_distribution`](super::distribution_packing::pack_spheres_from_distribution))
//! produce an initial grain cloud, but authoring a *good* granular scene also
//! means checking that cloud: is it dense enough, and are the grains actually
//! touching their neighbours or floating in a loose gas? Two numbers answer
//! that for a sphere assembly:
//!
//! * **Solid volume fraction** `φ = Σ Vᵢ / V_box` — the share of the containing
//!   box occupied by grain material. A random loose pack sits near `φ ≈ 0.55`
//!   and a dense random pack near `φ ≈ 0.64`; a value far below that means the
//!   dart-throwing budget was too small or the box too large.
//! * **Coordination number** `Z` — how many neighbours each grain touches. A
//!   mechanically stable 3-D pack needs a mean `Z` at or above the isostatic
//!   value (`≈ 6` for frictionless spheres, lower with friction); a cloud with
//!   `Z ≈ 0` has not been compacted and will collapse on the first DEM step.
//!
//! This module computes both from the parallel `positions`/`radii` arrays the
//! packers emit, using the same uniform-spatial-hash neighbour search as the
//! grid packer so the contact census is an expected `O(n)` rather than the
//! naive `O(n²)`. Everything here is a pure, deterministic analysis of the
//! supplied arrays; nothing is derived from Unreal Engine source.

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

/// Computed quality metrics for a sphere packing.
///
/// Build one with [`PackingDiagnostics::analyze`]. The coordination census
/// counts a pair as *touching* when the surface gap
/// `‖cᵢ − cⱼ‖ − rᵢ − rⱼ` is at most the `contact_tolerance` passed to
/// `analyze`, which absorbs the small separation a packer leaves between
/// grains.
#[derive(Clone, Debug, PartialEq)]
pub struct PackingDiagnostics {
    grain_count: usize,
    solid_volume: f64,
    coordination: Vec<u32>,
    contacting_pairs: usize,
}

impl PackingDiagnostics {
    /// Analyses a packing given parallel `positions` and `radii` and a
    /// non-negative `contact_tolerance` for the touching test.
    ///
    /// Returns `None` unless the two arrays share the same length, every value
    /// is finite, every radius is strictly positive, and `contact_tolerance`
    /// is finite and non-negative. An empty packing is accepted and yields
    /// zeroed metrics.
    #[must_use]
    pub fn analyze(positions: &[Vec3], radii: &[f32], contact_tolerance: f32) -> Option<Self> {
        if positions.len() != radii.len() {
            return None;
        }
        if !contact_tolerance.is_finite() || contact_tolerance < 0.0 {
            return None;
        }
        if positions.iter().any(|p| !p.is_finite()) {
            return None;
        }
        if radii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
            return None;
        }

        let grain_count = positions.len();
        // Solid volume Σ (4/3)·π·rᵢ³, accumulated in f64 for stability.
        let four_thirds_pi = 4.0 / 3.0 * std::f64::consts::PI;
        let solid_volume: f64 = radii
            .iter()
            .map(|&r| {
                let rd = r as f64;
                four_thirds_pi * rd * rd * rd
            })
            .sum();

        let mut coordination = vec![0_u32; grain_count];
        let mut contacting_pairs = 0_usize;

        if grain_count >= 2 {
            let radius_max = radii.iter().copied().fold(0.0_f32, f32::max);
            // Cell size = maximum touching distance, so any touching pair lies
            // within the 3×3×3 neighbourhood of a grain's cell.
            let cell = 2.0 * radius_max + contact_tolerance;
            let inv_cell = 1.0 / cell;
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

            for (i, (&ci, &ri)) in positions.iter().zip(radii.iter()).enumerate() {
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
                                let gap = ci.distance(positions[j]) - ri - radii[j];
                                if gap <= contact_tolerance {
                                    coordination[i] += 1;
                                    coordination[j] += 1;
                                    contacting_pairs += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Some(Self {
            grain_count,
            solid_volume,
            coordination,
            contacting_pairs,
        })
    }

    /// Number of grains analysed.
    #[must_use]
    pub fn grain_count(&self) -> usize {
        self.grain_count
    }

    /// Total solid volume `Σ (4/3)·π·rᵢ³` of all grains.
    #[must_use]
    pub fn solid_volume(&self) -> f32 {
        self.solid_volume as f32
    }

    /// Solid volume fraction `φ = solid_volume / V_box` for the axis-aligned
    /// box `[box_min, box_max]`.
    ///
    /// Returns `None` when the box is non-finite or has a non-positive extent
    /// on any axis.
    #[must_use]
    pub fn packing_fraction(&self, box_min: Vec3, box_max: Vec3) -> Option<f32> {
        if !(box_min.is_finite() && box_max.is_finite()) {
            return None;
        }
        let extent = box_max - box_min;
        if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
            return None;
        }
        let box_volume = (extent.x as f64) * (extent.y as f64) * (extent.z as f64);
        Some((self.solid_volume / box_volume) as f32)
    }

    /// Per-grain coordination numbers, indexed in lockstep with the analysed
    /// `positions`.
    #[must_use]
    pub fn coordination_numbers(&self) -> &[u32] {
        &self.coordination
    }

    /// Number of distinct touching grain pairs.
    #[must_use]
    pub fn contacting_pairs(&self) -> usize {
        self.contacting_pairs
    }

    /// Mean coordination number `Z = 2·contacting_pairs / grain_count`.
    ///
    /// Returns `0` for an empty packing.
    #[must_use]
    pub fn mean_coordination(&self) -> f32 {
        if self.grain_count == 0 {
            0.0
        } else {
            2.0 * self.contacting_pairs as f32 / self.grain_count as f32
        }
    }

    /// Smallest per-grain coordination number, or `0` for an empty packing.
    #[must_use]
    pub fn min_coordination(&self) -> u32 {
        self.coordination.iter().copied().min().unwrap_or(0)
    }

    /// Largest per-grain coordination number, or `0` for an empty packing.
    #[must_use]
    pub fn max_coordination(&self) -> u32 {
        self.coordination.iter().copied().max().unwrap_or(0)
    }

    /// Number of *rattlers*: grains with fewer than `threshold` contacts, which
    /// carry no load and are free to rattle in their cages. For frictionless
    /// spheres a common choice is `threshold = 4`.
    #[must_use]
    pub fn rattler_count(&self, threshold: u32) -> usize {
        self.coordination.iter().filter(|&&z| z < threshold).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_or_invalid_input() {
        assert!(PackingDiagnostics::analyze(&[Vec3::ZERO], &[], 0.0).is_none());
        assert!(PackingDiagnostics::analyze(&[Vec3::ZERO], &[0.0], 0.0).is_none());
        assert!(PackingDiagnostics::analyze(&[Vec3::ZERO], &[1.0], -0.1).is_none());
        assert!(
            PackingDiagnostics::analyze(&[Vec3::new(f32::NAN, 0.0, 0.0)], &[1.0], 0.0).is_none()
        );
        assert!(PackingDiagnostics::analyze(&[Vec3::ZERO], &[1.0], 0.0).is_some());
    }

    #[test]
    fn empty_packing_yields_zeroed_metrics() {
        let d = PackingDiagnostics::analyze(&[], &[], 0.0).unwrap();
        assert_eq!(d.grain_count(), 0);
        assert_eq!(d.contacting_pairs(), 0);
        assert_eq!(d.mean_coordination(), 0.0);
        assert_eq!(d.min_coordination(), 0);
        assert_eq!(d.max_coordination(), 0);
        assert!(d.solid_volume().abs() <= 1.0e-12);
    }

    #[test]
    fn solid_volume_matches_closed_form() {
        let r = 0.5_f32;
        let d = PackingDiagnostics::analyze(&[Vec3::ZERO], &[r], 0.0).unwrap();
        let expected = 4.0 / 3.0 * std::f32::consts::PI * r * r * r;
        assert!((d.solid_volume() - expected).abs() <= 1.0e-6);
    }

    #[test]
    fn packing_fraction_rejects_degenerate_box() {
        let d = PackingDiagnostics::analyze(&[Vec3::ZERO], &[0.5], 0.0).unwrap();
        assert!(d
            .packing_fraction(Vec3::ZERO, Vec3::new(0.0, 1.0, 1.0))
            .is_none());
        let phi = d
            .packing_fraction(Vec3::splat(-1.0), Vec3::splat(1.0))
            .unwrap();
        // One r=0.5 sphere in a 2×2×2 box: φ = (π/6)/8 ≈ 0.0654.
        assert!((phi - 0.06545).abs() <= 1.0e-3, "phi {phi}");
    }

    #[test]
    fn two_touching_spheres_are_mutually_coordinated() {
        // Centres 1.0 apart, radii 0.5 each: gap exactly 0 ⇒ touching.
        let positions = [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let radii = [0.5_f32, 0.5];
        let d = PackingDiagnostics::analyze(&positions, &radii, 1.0e-4).unwrap();
        assert_eq!(d.contacting_pairs(), 1);
        assert_eq!(d.coordination_numbers(), &[1, 1]);
        assert_eq!(d.mean_coordination(), 1.0);
    }

    #[test]
    fn separated_spheres_are_not_coordinated() {
        // Centres 2.0 apart, radii 0.5: gap 1.0 ≫ tolerance ⇒ no contact.
        let positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let radii = [0.5_f32, 0.5];
        let d = PackingDiagnostics::analyze(&positions, &radii, 0.01).unwrap();
        assert_eq!(d.contacting_pairs(), 0);
        assert_eq!(d.coordination_numbers(), &[0, 0]);
        assert_eq!(d.rattler_count(1), 2);
    }

    #[test]
    fn chain_of_three_counts_interior_grain_twice() {
        // A line of three unit-gap spheres: ends touch one neighbour, the
        // middle touches two.
        let positions = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let radii = [0.5_f32, 0.5, 0.5];
        let d = PackingDiagnostics::analyze(&positions, &radii, 1.0e-4).unwrap();
        assert_eq!(d.contacting_pairs(), 2);
        assert_eq!(d.coordination_numbers(), &[1, 2, 1]);
        assert_eq!(d.min_coordination(), 1);
        assert_eq!(d.max_coordination(), 2);
        assert_eq!(d.rattler_count(2), 2); // the two ends have Z < 2
    }

    #[test]
    fn matches_brute_force_census_on_a_grid_packing() {
        use crate::collider::grid_sphere_packing::pack_spheres_grid;
        use crate::collider::sphere_packing::SpherePackingParams;

        let params = SpherePackingParams {
            min: Vec3::ZERO,
            max: Vec3::splat(8.0),
            radius_min: 0.4,
            radius_max: 0.6,
            target_count: 120,
            max_attempts: 100_000,
            separation: 0.0,
            seed: 7,
        };
        let packing = pack_spheres_grid(&params).unwrap();
        let positions = packing.positions();
        let radii = packing.radii();
        let tol = 0.02_f32;
        let d = PackingDiagnostics::analyze(positions, radii, tol).unwrap();

        // Independent O(n²) census must agree with the grid-accelerated one.
        let mut pairs = 0usize;
        for (i, (&ci, &ri)) in positions.iter().zip(radii.iter()).enumerate() {
            for (&cj, &rj) in positions.iter().zip(radii.iter()).skip(i + 1) {
                if ci.distance(cj) - ri - rj <= tol {
                    pairs += 1;
                }
            }
        }
        assert_eq!(d.contacting_pairs(), pairs);
    }
}
