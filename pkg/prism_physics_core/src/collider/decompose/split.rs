//! Axis-aligned split-plane search for one voxel part.
//!
//! When a part is too concave the driver cuts it in two along the plane that
//! best reduces concavity. Candidate planes lie on cell boundaries along each
//! of the three axes; each candidate is scored by the summed concavity of the
//! two halves plus a small balance penalty that discourages shaving off a
//! single sliver. The lowest-scoring plane wins. This mirrors the V-HACD
//! family's "best clipping plane" step, restricted to axis-aligned planes for
//! determinism and speed.
//!
//! The number of planes probed per axis is capped so the search cost stays
//! bounded even at high voxel resolutions; the probed coordinates are spread
//! evenly across the part's extent.
//!
//! # Provenance
//!
//! Greedy concavity-minimizing plane selection over a voxel grid is a standard
//! decomposition heuristic. This module contains **no Unreal Engine source or
//! derived code**.

use super::concavity::{concavity, part_volume};
use super::voxel::VoxelGrid;

/// The outcome of a successful binary split.
#[derive(Clone, Debug)]
pub struct SplitResult {
    /// Cells on the low side of the plane (coordinate `< coord` on `axis`).
    pub left: Vec<usize>,
    /// Cells on the high side of the plane (coordinate `>= coord`).
    pub right: Vec<usize>,
    /// The split axis (`0 = x`, `1 = y`, `2 = z`).
    pub axis: usize,
    /// The cell coordinate the plane sits in front of.
    pub coord: usize,
}

/// Relative weight of the balance penalty against raw concavity. Small enough
/// that concavity dominates, large enough to break ties toward central cuts.
const BALANCE_WEIGHT: f32 = 0.05;

/// Searches every axis for the split plane that minimizes
/// `concavity(left) + concavity(right) + balance_penalty`.
///
/// `max_planes_per_axis` caps how many candidate planes are probed on each axis
/// (evenly spaced across the part's extent); it is treated as at least 1.
/// Returns [`None`] when the part cannot be split (it spans a single cell on
/// every axis, or no candidate leaves both halves non-empty).
#[must_use]
pub fn best_split(
    part: &[usize],
    grid: &VoxelGrid,
    max_planes_per_axis: usize,
) -> Option<SplitResult> {
    if part.len() < 2 {
        return None;
    }
    let max_planes_per_axis = max_planes_per_axis.max(1);
    let part_vol = part_volume(part, grid);

    let mut best: Option<(f32, SplitResult)> = None;
    for axis in 0..3usize {
        let (lo, hi) = axis_range(part, grid, axis);
        if hi <= lo {
            continue;
        }
        for coord in plane_coords(lo, hi, max_planes_per_axis) {
            let (left, right) = partition(part, grid, axis, coord);
            if left.is_empty() || right.is_empty() {
                continue;
            }
            let imbalance = (left.len() as f32 - right.len() as f32).abs() / part.len() as f32;
            let score = concavity(&left, grid)
                + concavity(&right, grid)
                + BALANCE_WEIGHT * imbalance * part_vol;
            let better = match &best {
                None => true,
                Some((best_score, _)) => score < *best_score,
            };
            if better {
                best = Some((
                    score,
                    SplitResult {
                        left,
                        right,
                        axis,
                        coord,
                    },
                ));
            }
        }
    }
    best.map(|(_, result)| result)
}

/// The inclusive `[min, max]` cell-coordinate range the part occupies on `axis`.
fn axis_range(part: &[usize], grid: &VoxelGrid, axis: usize) -> (usize, usize) {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for &cell in part {
        let c = grid.cell_coord(cell)[axis];
        lo = lo.min(c);
        hi = hi.max(c);
    }
    (lo, hi)
}

/// Evenly spaced split coordinates in the open interval `(lo, hi]`, i.e. the
/// planes that separate cell `coord - 1` from cell `coord`, deduplicated and
/// sorted ascending.
fn plane_coords(lo: usize, hi: usize, max_planes: usize) -> Vec<usize> {
    // Valid plane coordinates are lo+1 ..= hi (each leaves cells on both sides).
    let span = hi - lo; // number of candidate planes available
    let count = span.min(max_planes);
    if count == 0 {
        return Vec::new();
    }
    let mut coords = Vec::with_capacity(count);
    let mut prev = usize::MAX;
    for s in 0..count {
        // Map step s in [0, count) to a plane coordinate in [lo+1, hi].
        let num = (s * 2 + 1) * span;
        let coord = lo + 1 + num / (count * 2);
        let coord = coord.min(hi);
        if coord != prev {
            coords.push(coord);
            prev = coord;
        }
    }
    coords
}

/// Partitions `part` into cells with `coord_on_axis < coord` (left) and the
/// rest (right).
fn partition(
    part: &[usize],
    grid: &VoxelGrid,
    axis: usize,
    coord: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &cell in part {
        if grid.cell_coord(cell)[axis] < coord {
            left.push(cell);
        } else {
            right.push(cell);
        }
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::super::concavity::concavity;
    use super::*;
    use glam::Vec3;

    /// Builds a closed box mesh (outward CCW) spanning `[min, max]`.
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

    fn box_grid() -> VoxelGrid {
        let (v, t) = box_mesh(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        VoxelGrid::voxelize(&v, &t, 16).unwrap()
    }

    #[test]
    fn plane_coords_are_in_range_and_sorted() {
        let coords = plane_coords(2, 10, 4);
        assert!(!coords.is_empty());
        for w in coords.windows(2) {
            assert!(w[0] < w[1]);
        }
        for &c in &coords {
            assert!(c > 2 && c <= 10);
        }
    }

    #[test]
    fn plane_coords_capped() {
        let coords = plane_coords(0, 100, 8);
        assert!(coords.len() <= 8);
    }

    #[test]
    fn split_halves_cover_the_part() {
        let grid = box_grid();
        let part = grid.occupied_indices();
        let split = best_split(&part, &grid, 8).expect("box is splittable");
        assert_eq!(split.left.len() + split.right.len(), part.len());
        assert!(!split.left.is_empty() && !split.right.is_empty());
    }

    #[test]
    fn splitting_convex_box_does_not_raise_per_part_concavity() {
        let grid = box_grid();
        let part = grid.occupied_indices();
        let whole = concavity(&part, &grid);
        let split = best_split(&part, &grid, 8).unwrap();
        let child_max = concavity(&split.left, &grid).max(concavity(&split.right, &grid));
        assert!(child_max <= whole + 1e-4);
    }

    #[test]
    fn deterministic_split() {
        let grid = box_grid();
        let part = grid.occupied_indices();
        let a = best_split(&part, &grid, 8).unwrap();
        let b = best_split(&part, &grid, 8).unwrap();
        assert_eq!(a.axis, b.axis);
        assert_eq!(a.coord, b.coord);
        assert_eq!(a.left, b.left);
    }
}
