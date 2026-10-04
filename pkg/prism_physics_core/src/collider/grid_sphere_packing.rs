//! Grid-accelerated deterministic sphere packing for large granular scenes.
//!
//! [`pack_spheres`](super::sphere_packing::pack_spheres) places grains by dart
//! throwing and tests each candidate against *every* already-placed grain — an
//! `O(n²)` acceptance loop that is fine for the hundreds-to-low-thousands grain
//! counts of a small scene but becomes the bottleneck for the tens of thousands
//! of grains a production hopper, soil column or ballast bed needs.
//!
//! This module packs the identical polydisperse, non-overlapping initial
//! condition using a **uniform spatial hash** so each candidate is tested only
//! against the grains in its own cell and the 26 neighbours — expected `O(1)`
//! per placement and `O(n)` overall.
//!
//! # Why the result is identical to the brute-force packer
//!
//! Two grains conflict only when their centres are closer than
//! `rᵢ + rⱼ + separation`, which is at most `2·radius_max + separation`. Choosing
//! the cell size to be exactly that maximum interaction distance guarantees any
//! conflicting grain shares the candidate's cell or an immediate neighbour: if
//! two centres are strictly closer than one cell on an axis their cell indices
//! differ by at most one on that axis, so scanning the 3×3×3 neighbourhood can
//! never miss a conflict. Because the candidate radius and centre are drawn in
//! the same order from the same [`DeterministicRng`] as the brute-force packer,
//! and the accept/reject decision is identical, `pack_spheres_grid` reproduces
//! the *exact* packing of [`pack_spheres`] for the same
//! [`SpherePackingParams`] — only faster. That equivalence is asserted directly
//! in the tests. Nothing here is derived from Unreal Engine source.

use super::sphere_packing::{SpherePacking, SpherePackingParams};
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

/// Generates a non-overlapping polydisperse sphere packing using a uniform
/// spatial hash to accelerate the overlap test.
///
/// Accepts the same [`SpherePackingParams`] as
/// [`pack_spheres`](super::sphere_packing::pack_spheres) and returns the
/// identical packing — the grid only changes the cost of the acceptance test
/// from `O(n²)` to an expected `O(n)`. Returns `None` on exactly the same
/// invalid-parameter conditions: a non-finite or non-strictly-ordered box, a
/// non-positive `radius_min`, `radius_max < radius_min`, a negative
/// `separation`, a zero `target_count`, a `max_attempts` smaller than
/// `target_count`, or a box too small to hold a grain of radius `radius_max`.
#[must_use]
pub fn pack_spheres_grid(params: &SpherePackingParams) -> Option<SpherePacking> {
    let &SpherePackingParams {
        min,
        max,
        radius_min,
        radius_max,
        target_count,
        max_attempts,
        separation,
        seed,
    } = params;

    if !(min.is_finite() && max.is_finite()) {
        return None;
    }
    if !(radius_min.is_finite() && radius_max.is_finite() && separation.is_finite()) {
        return None;
    }
    if radius_min <= 0.0 || radius_max < radius_min || separation < 0.0 {
        return None;
    }
    if target_count == 0 || max_attempts < target_count {
        return None;
    }
    let extent = max - min;
    if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
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
        let radius = rng.next_range(radius_min, radius_max);
        let lo = min + Vec3::splat(radius);
        let hi = max - Vec3::splat(radius);
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
    use crate::collider::sphere_packing::pack_spheres;

    fn base_params() -> SpherePackingParams {
        SpherePackingParams {
            min: Vec3::ZERO,
            max: Vec3::splat(10.0),
            radius_min: 0.4,
            radius_max: 0.6,
            target_count: 200,
            max_attempts: 200_000,
            separation: 0.05,
            seed: 42,
        }
    }

    #[test]
    fn rejects_invalid_params() {
        let mut p = base_params();
        p.max = Vec3::new(-1.0, 10.0, 10.0);
        assert!(pack_spheres_grid(&p).is_none());

        let mut p = base_params();
        p.radius_min = 0.0;
        assert!(pack_spheres_grid(&p).is_none());

        let mut p = base_params();
        p.radius_max = 0.1;
        assert!(pack_spheres_grid(&p).is_none());

        let mut p = base_params();
        p.separation = -1.0;
        assert!(pack_spheres_grid(&p).is_none());

        let mut p = base_params();
        p.target_count = 0;
        assert!(pack_spheres_grid(&p).is_none());

        let mut p = base_params();
        p.max_attempts = p.target_count - 1;
        assert!(pack_spheres_grid(&p).is_none());

        let mut p = base_params();
        p.max = Vec3::new(1.0, 10.0, 10.0); // narrower than one diameter
        assert!(pack_spheres_grid(&p).is_none());
    }

    #[test]
    fn grid_packing_is_non_overlapping_and_inside_the_box() {
        let p = base_params();
        let packing = pack_spheres_grid(&p).expect("valid params pack");
        assert!(!packing.is_empty());
        assert!(packing.min_separation() >= p.separation - 1.0e-5);
        for (&c, &r) in packing.positions().iter().zip(packing.radii().iter()) {
            assert!(c.x - r >= p.min.x - 1.0e-5 && c.x + r <= p.max.x + 1.0e-5);
            assert!(c.y - r >= p.min.y - 1.0e-5 && c.y + r <= p.max.y + 1.0e-5);
            assert!(c.z - r >= p.min.z - 1.0e-5 && c.z + r <= p.max.z + 1.0e-5);
        }
    }

    #[test]
    fn grid_packing_matches_brute_force_exactly() {
        // The grid acceleration must not change the outcome: for identical
        // params and seed it must reproduce the brute-force packing bit for bit.
        let p = base_params();
        let brute = pack_spheres(&p).expect("brute force packs");
        let grid = pack_spheres_grid(&p).expect("grid packs");
        assert_eq!(grid.len(), brute.len(), "grain count diverged");
        assert_eq!(grid.positions(), brute.positions(), "positions diverged");
        assert_eq!(grid.radii(), brute.radii(), "radii diverged");
    }

    #[test]
    fn grid_packing_is_deterministic() {
        let p = base_params();
        let a = pack_spheres_grid(&p).expect("packs");
        let b = pack_spheres_grid(&p).expect("packs");
        assert_eq!(a.positions(), b.positions());
        assert_eq!(a.radii(), b.radii());
    }

    #[test]
    fn negative_origin_box_matches_brute_force() {
        // Negative cell coordinates exercise the signed floor in cell_of.
        let mut p = base_params();
        p.min = Vec3::new(-8.0, -5.0, -3.0);
        p.max = Vec3::new(2.0, 5.0, 7.0);
        let brute = pack_spheres(&p).expect("brute force packs");
        let grid = pack_spheres_grid(&p).expect("grid packs");
        assert_eq!(grid.positions(), brute.positions());
        assert_eq!(grid.radii(), brute.radii());
    }
}
