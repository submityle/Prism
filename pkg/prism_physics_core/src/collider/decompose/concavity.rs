//! Concavity measurement for a voxel part.
//!
//! A part's *concavity* is the extra volume its convex hull encloses beyond the
//! part's own solid volume -- the gap the hull cannot represent. The driver
//! splits the part with the largest concavity until every part is "convex
//! enough" (its concavity falls below a fraction of the whole mesh's volume).
//! This volumetric concavity is the same acceptance signal the V-HACD family
//! uses to decide when a part no longer needs subdividing.
//!
//! Volumes are measured in voxel units: the part volume is simply the occupied
//! cell count times the cell volume, and the hull volume is the signed volume
//! of the convex hull of the part's boundary-cell corners.
//!
//! # Provenance
//!
//! Volumetric concavity and signed-tetrahedron volume integration are standard
//! geometry. This module contains **no Unreal Engine source or derived code**.

use std::collections::HashSet;

use glam::Vec3;

use super::voxel::VoxelGrid;
use crate::collider::convex_hull;

/// Collects the deduplicated world-space corners of the part's boundary cells.
///
/// Only cells with a missing face-neighbour (outside the part or outside the
/// grid) contribute corners; a cell buried inside the part can only yield
/// corners strictly interior to the hull, so dropping it changes neither the
/// hull nor its volume while cutting the input point count dramatically.
///
/// Corners are emitted in ascending cell order and deduplicated by integer
/// lattice coordinate, so the point list is deterministic for a given part.
#[must_use]
pub fn part_corner_points(part: &[usize], grid: &VoxelGrid) -> Vec<Vec3> {
    let members: HashSet<usize> = part.iter().copied().collect();
    let [nx, ny, nz] = grid.dims();
    let mut seen: HashSet<[usize; 3]> = HashSet::new();
    let mut points: Vec<Vec3> = Vec::new();

    for &cell in part {
        let [i, j, k] = grid.cell_coord(cell);
        if !is_boundary_cell(i, j, k, [nx, ny, nz], grid, &members) {
            continue;
        }
        for (di, dj, dk) in corner_offsets() {
            let corner = [i + di, j + dj, k + dk];
            if seen.insert(corner) {
                // Reconstruct the corner position without a dedicated accessor:
                // the cell's base corner plus the unit offset scaled by the cell.
                let base = grid.cell_center(i, j, k) - Vec3::splat(0.5 * grid.cell_size());
                points.push(base + Vec3::new(di as f32, dj as f32, dk as f32) * grid.cell_size());
            }
        }
    }
    points
}

/// The eight unit corner offsets of a cell, in a fixed order.
fn corner_offsets() -> [(usize, usize, usize); 8] {
    [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
}

/// Whether cell `(i, j, k)` has at least one face-neighbour missing from the
/// part (either outside the grid or not an occupied part member).
fn is_boundary_cell(
    i: usize,
    j: usize,
    k: usize,
    dims: [usize; 3],
    grid: &VoxelGrid,
    members: &HashSet<usize>,
) -> bool {
    let [nx, ny, nz] = dims;
    let neighbors = [
        (i > 0, i.wrapping_sub(1), j, k),
        (i + 1 < nx, i + 1, j, k),
        (j > 0, i, j.wrapping_sub(1), k),
        (j + 1 < ny, i, j + 1, k),
        (k > 0, i, j, k.wrapping_sub(1)),
        (k + 1 < nz, i, j, k + 1),
    ];
    for (in_bounds, ni, nj, nk) in neighbors {
        if !in_bounds {
            return true;
        }
        if !members.contains(&grid.linear_index(ni, nj, nk)) {
            return true;
        }
    }
    false
}

/// The solid volume of the part: occupied cell count times the cell volume.
#[must_use]
pub fn part_volume(part: &[usize], grid: &VoxelGrid) -> f32 {
    part.len() as f32 * grid.cell_volume()
}

/// The volume enclosed by the convex hull of the part's boundary corners.
///
/// Returns `0.0` for a part too small or degenerate to form a convex solid.
#[must_use]
pub fn hull_volume(part: &[usize], grid: &VoxelGrid) -> f32 {
    let points = part_corner_points(part, grid);
    let Some((vertices, triangles)) = convex_hull(&points) else {
        return 0.0;
    };
    signed_mesh_volume(&vertices, &triangles)
}

/// The concavity of the part: how much volume its convex hull adds beyond the
/// part's own solid volume. Clamped to be non-negative (voxel discretization
/// can make the hull marginally smaller than the stair-stepped part).
#[must_use]
pub fn concavity(part: &[usize], grid: &VoxelGrid) -> f32 {
    (hull_volume(part, grid) - part_volume(part, grid)).max(0.0)
}

/// Signed volume of a closed, outward-wound triangle mesh (the sum of signed
/// tetrahedron volumes from the origin). Returns the magnitude.
fn signed_mesh_volume(vertices: &[Vec3], triangles: &[[u32; 3]]) -> f32 {
    let mut v6 = 0.0f32;
    for tri in triangles {
        let a = vertices[tri[0] as usize];
        let b = vertices[tri[1] as usize];
        let c = vertices[tri[2] as usize];
        v6 += a.dot(b.cross(c));
    }
    (v6 / 6.0).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_mesh(min: Vec3, max: Vec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        let t = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        (v, t)
    }

    #[test]
    fn convex_box_has_near_zero_concavity() {
        let (v, t) = box_mesh(Vec3::splat(-1.0), Vec3::splat(1.0));
        let grid = VoxelGrid::voxelize(&v, &t, 24).unwrap();
        let part = grid.occupied_indices();
        let total = part_volume(&part, &grid);
        // A convex solid's hull coincides with the part up to voxel aliasing.
        assert!(concavity(&part, &grid) < 0.05 * total);
    }

    #[test]
    fn hull_volume_matches_box() {
        let (v, t) = box_mesh(Vec3::splat(-1.0), Vec3::splat(1.0));
        let grid = VoxelGrid::voxelize(&v, &t, 32).unwrap();
        let part = grid.occupied_indices();
        let hv = hull_volume(&part, &grid);
        assert!((hv - 8.0).abs() < 0.1 * 8.0, "hull volume {hv}");
    }

    #[test]
    fn signed_volume_of_unit_box() {
        let (v, t) = box_mesh(Vec3::ZERO, Vec3::ONE);
        assert!((signed_mesh_volume(&v, &t) - 1.0).abs() < 1e-5);
    }
}
