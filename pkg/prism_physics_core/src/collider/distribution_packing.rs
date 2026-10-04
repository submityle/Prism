//! Grid-accelerated sphere packing whose grain sizes follow a prescribed
//! [`GrainSizeDistribution`].
//!
//! [`pack_spheres_grid`](super::grid_sphere_packing::pack_spheres_grid) draws
//! each grain's radius *uniformly* from `[radius_min, radius_max]`. Authoring a
//! realistic granular bed usually means matching a measured grading curve —
//! a log-normal spread or an explicit sieve analysis — rather than a flat band.
//! This module packs exactly such a bed: it reuses the uniform-spatial-hash
//! acceptance test of the grid packer but takes each candidate radius from a
//! caller-supplied [`GrainSizeDistribution`], tying together the size-sampling
//! and packing stages of scene setup.
//!
//! The output is a [`SpherePacking`] ready for the DEM integrators and the
//! boundary driver, with no two grains overlapping and every grain fully inside
//! the authored box. Randomness comes from the crate's
//! [`DeterministicRng`], so a given [`DistributionPackingParams`] and
//! distribution always yield the identical packing. Nothing here is derived
//! from Unreal Engine source.

use super::grain_size_distribution::GrainSizeDistribution;
use super::sphere_packing::SpherePacking;
use crate::fracture::rng::DeterministicRng;
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

/// Tuning for [`pack_spheres_from_distribution`], excluding the radius spread,
/// which is supplied separately as a [`GrainSizeDistribution`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistributionPackingParams {
    /// Lower corner of the axis-aligned packing box.
    pub min: Vec3,
    /// Upper corner of the axis-aligned packing box; strictly greater than
    /// `min` on every axis.
    pub max: Vec3,
    /// Desired number of grains. The result may contain fewer if the dart
    /// budget is exhausted first.
    pub target_count: usize,
    /// Total dart-throw budget. Must be at least `target_count`.
    pub max_attempts: usize,
    /// Extra clearance required between grain surfaces; non-negative.
    pub separation: f32,
    /// Seed for the deterministic generator.
    pub seed: u64,
}

/// Packs a non-overlapping sphere bed whose radii follow `distribution`.
///
/// Candidate radii are drawn from `distribution` and candidate centres are
/// drawn uniformly from the box shrunk by the candidate radius, so every grain
/// is guaranteed to fit; a candidate is accepted only when its surface clears
/// every already-placed grain by at least `params.separation`, tested against
/// the 3×3×3 spatial-hash neighbourhood for an expected `O(n)` cost.
///
/// Returns `None` when `params` is invalid: a non-finite or non-strictly-
/// ordered box, a negative or non-finite `separation`, a zero `target_count`,
/// a `max_attempts` smaller than `target_count`, or a box too small to contain
/// a grain of the distribution's maximum radius on every axis.
#[must_use]
pub fn pack_spheres_from_distribution(
    params: &DistributionPackingParams,
    distribution: &GrainSizeDistribution,
) -> Option<SpherePacking> {
    let &DistributionPackingParams {
        min,
        max,
        target_count,
        max_attempts,
        separation,
        seed,
    } = params;

    if !(min.is_finite() && max.is_finite()) {
        return None;
    }
    if !separation.is_finite() || separation < 0.0 {
        return None;
    }
    if target_count == 0 || max_attempts < target_count {
        return None;
    }
    let extent = max - min;
    if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
        return None;
    }

    let radius_max = distribution.max_radius();
    if !(radius_max.is_finite() && radius_max > 0.0) {
        return None;
    }
    let diameter = 2.0 * radius_max;
    if extent.x < diameter || extent.y < diameter || extent.z < diameter {
        return None;
    }

    // Cell size = maximum centre-to-centre conflict distance, so any conflict
    // lies within the 3×3×3 neighbourhood of the candidate's cell.
    let cell = diameter + separation;
    let inv_cell = 1.0 / cell;

    let mut rng = DeterministicRng::new(seed);
    let mut positions: Vec<Vec3> = Vec::with_capacity(target_count);
    let mut radii: Vec<f32> = Vec::with_capacity(target_count);
    let mut grid: HashMap<Cell, Vec<usize>> = HashMap::new();
    let mut attempts = 0_usize;

    while positions.len() < target_count && attempts < max_attempts {
        attempts += 1;
        let radius = distribution.sample(&mut rng);
        let lo = min + Vec3::splat(radius);
        let hi = max - Vec3::splat(radius);
        // The distribution guarantees radius ≤ radius_max, so the shrunk box is
        // non-empty; guard defensively regardless.
        if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
            continue;
        }
        let center = rng.next_in_box(lo, hi);
        let (cx, cy, cz) = cell_of(center, min, inv_cell);

        let mut accepted = true;
        'scan: for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    let Some(indices) = grid.get(&key) else {
                        continue;
                    };
                    for &idx in indices {
                        let min_dist = radius + radii[idx] + separation;
                        if center.distance_squared(positions[idx]) < min_dist * min_dist {
                            accepted = false;
                            break 'scan;
                        }
                    }
                }
            }
        }

        if accepted {
            let index = positions.len();
            positions.push(center);
            radii.push(radius);
            grid.entry((cx, cy, cz)).or_default().push(index);
        }
    }

    SpherePacking::from_parts(positions, radii)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::grain_size_distribution::SieveBin;

    fn base_params() -> DistributionPackingParams {
        DistributionPackingParams {
            min: Vec3::ZERO,
            max: Vec3::splat(10.0),
            target_count: 150,
            max_attempts: 200_000,
            separation: 0.05,
            seed: 42,
        }
    }

    fn log_normal() -> GrainSizeDistribution {
        GrainSizeDistribution::log_normal(0.5, 1.3, 0.35, 0.7).unwrap()
    }

    #[test]
    fn rejects_invalid_params() {
        let dist = log_normal();

        let mut p = base_params();
        p.max = Vec3::new(-1.0, 10.0, 10.0);
        assert!(pack_spheres_from_distribution(&p, &dist).is_none());

        let mut p = base_params();
        p.separation = -1.0;
        assert!(pack_spheres_from_distribution(&p, &dist).is_none());

        let mut p = base_params();
        p.target_count = 0;
        assert!(pack_spheres_from_distribution(&p, &dist).is_none());

        let mut p = base_params();
        p.max_attempts = p.target_count - 1;
        assert!(pack_spheres_from_distribution(&p, &dist).is_none());

        // Box narrower than one maximum diameter (2 * 0.7 = 1.4).
        let mut p = base_params();
        p.max = Vec3::new(1.0, 10.0, 10.0);
        assert!(pack_spheres_from_distribution(&p, &dist).is_none());
    }

    #[test]
    fn packing_is_non_overlapping_inside_box_and_within_distribution() {
        let p = base_params();
        let dist = log_normal();
        let packing = pack_spheres_from_distribution(&p, &dist).expect("valid params pack");
        assert!(!packing.is_empty());
        assert!(packing.min_separation() >= p.separation - 1.0e-5);
        for (&c, &r) in packing.positions().iter().zip(packing.radii().iter()) {
            assert!(
                (dist.min_radius()..=dist.max_radius()).contains(&r),
                "radius {r} outside the distribution envelope"
            );
            assert!(c.x - r >= p.min.x - 1.0e-5 && c.x + r <= p.max.x + 1.0e-5);
            assert!(c.y - r >= p.min.y - 1.0e-5 && c.y + r <= p.max.y + 1.0e-5);
            assert!(c.z - r >= p.min.z - 1.0e-5 && c.z + r <= p.max.z + 1.0e-5);
        }
    }

    #[test]
    fn packing_is_deterministic() {
        let p = base_params();
        let dist = log_normal();
        let a = pack_spheres_from_distribution(&p, &dist).expect("packs");
        let b = pack_spheres_from_distribution(&p, &dist).expect("packs");
        assert_eq!(a.positions(), b.positions());
        assert_eq!(a.radii(), b.radii());
    }

    #[test]
    fn graded_distribution_packs_within_its_bins() {
        let dist = GrainSizeDistribution::graded(vec![
            SieveBin::new(0.3, 0.4, 1.0).unwrap(),
            SieveBin::new(0.6, 0.8, 2.0).unwrap(),
        ])
        .unwrap();
        let p = base_params();
        let packing = pack_spheres_from_distribution(&p, &dist).expect("packs");
        assert!(!packing.is_empty());
        for &r in packing.radii() {
            let in_fine = (0.3..0.4).contains(&r);
            let in_coarse = (0.6..0.8).contains(&r);
            assert!(in_fine || in_coarse, "radius {r} fell outside every bin");
        }
    }
}
