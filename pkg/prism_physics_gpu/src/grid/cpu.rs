//! `CPU` golden twin for the bounded uniform-grid build.
//!
//! [`cpu_cell_index`] maps a point to its dense linear cell index with the same
//! `floor`/`clamp` arithmetic the `WGSL` hash kernel uses, and [`cpu_grid_sort`]
//! performs the whole build: assign every particle its cell, stably sort the
//! particle indices by cell, then scan the sorted stream for each occupied
//! cell's `[start, end)` half-open range into the sorted array.
//!
//! Because the cell index is an integer and the sort is stable, the build is a
//! pure permutation plus a range scan, so a real-device parity test can compare
//! the outputs of this twin and the `GPU` path bit-for-bit.
//!
//! # Provenance
//!
//! The sort-by-cell then find-cell-start construction is the classical uniform
//! grid of Green, "Particle Simulation using CUDA" (NVIDIA 2008). No Unreal
//! Engine source or derived code.

use glam::Vec3;

use super::config::GridConfig;

/// Sentinel stored in [`GridBuild::cell_start`] and [`GridBuild::cell_end`] for
/// a cell that holds no particles.
pub const EMPTY: u32 = u32::MAX;

/// The result of a uniform-grid build.
///
/// `sorted_indices` lists the input particle indices reordered so that all
/// particles sharing a cell are contiguous and in ascending cell order; ties
/// keep their input order. For an occupied cell `c`, `cell_start[c]..cell_end[c]`
/// is the half-open range of positions into `sorted_indices` that belong to `c`;
/// both are [`EMPTY`] for a cell with no particles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridBuild {
    /// Particle indices ordered by cell (stable within a cell).
    pub sorted_indices: Vec<u32>,
    /// Per-cell first position into `sorted_indices`, or [`EMPTY`].
    pub cell_start: Vec<u32>,
    /// Per-cell one-past-last position into `sorted_indices`, or [`EMPTY`].
    pub cell_end: Vec<u32>,
}

/// Maps `position` to its dense linear cell index in `config`.
///
/// Each axis coordinate is `floor((p - origin) / cell_size)` clamped to
/// `[0, dim - 1]`, so a point outside the lattice lands in the nearest boundary
/// cell. The clamp is performed in floating point before the cast so the value
/// converted to an integer is always in range, exactly mirroring the kernel.
#[must_use]
pub fn cpu_cell_index(position: Vec3, config: &GridConfig) -> u32 {
    let [nx, ny, nz] = config.dims;
    let cx = axis_coord(position.x, config.origin.x, config.cell_size, nx);
    let cy = axis_coord(position.y, config.origin.y, config.cell_size, ny);
    let cz = axis_coord(position.z, config.origin.z, config.cell_size, nz);
    cx + cy * nx + cz * nx * ny
}

/// Clamped, floored coordinate of `p` along one axis with `dim` cells.
fn axis_coord(p: f32, origin: f32, cell_size: f32, dim: u32) -> u32 {
    let hi = dim.saturating_sub(1) as f32;
    let coord = ((p - origin) / cell_size).floor().clamp(0.0, hi);
    coord as u32
}

/// Builds the uniform grid for `positions` under `config`.
///
/// Returns the stably sorted particle indices and the per-cell ranges into
/// them. The `cell_start`/`cell_end` vectors always have `config.num_cells()`
/// entries, with [`EMPTY`] for cells that hold no particles.
#[must_use]
pub fn cpu_grid_sort(positions: &[Vec3], config: &GridConfig) -> GridBuild {
    let n = positions.len();
    let num_cells = config.num_cells() as usize;

    let cells: Vec<u32> = positions
        .iter()
        .map(|p| cpu_cell_index(*p, config))
        .collect();

    let mut sorted_indices: Vec<u32> = (0..u32::try_from(n).unwrap_or(u32::MAX)).collect();
    sorted_indices.sort_by_key(|&i| cells[i as usize]);

    let mut cell_start = vec![EMPTY; num_cells];
    let mut cell_end = vec![EMPTY; num_cells];
    for pos in 0..n {
        let cell = cells[sorted_indices[pos] as usize];
        let first = pos == 0 || cells[sorted_indices[pos - 1] as usize] != cell;
        let last = pos == n - 1 || cells[sorted_indices[pos + 1] as usize] != cell;
        if first {
            cell_start[cell as usize] = pos as u32;
        }
        if last {
            cell_end[cell as usize] = (pos + 1) as u32;
        }
    }

    GridBuild {
        sorted_indices,
        cell_start,
        cell_end,
    }
}

#[cfg(test)]
mod tests {
    use super::{cpu_cell_index, cpu_grid_sort, EMPTY};
    use crate::grid::config::GridConfig;
    use glam::Vec3;

    #[test]
    fn cell_index_is_dense_and_linear() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [4, 4, 4]);
        assert_eq!(cpu_cell_index(Vec3::new(0.5, 0.5, 0.5), &config), 0);
        assert_eq!(cpu_cell_index(Vec3::new(1.5, 0.5, 0.5), &config), 1);
        assert_eq!(cpu_cell_index(Vec3::new(0.5, 1.5, 0.5), &config), 4);
        assert_eq!(cpu_cell_index(Vec3::new(0.5, 0.5, 1.5), &config), 16);
    }

    #[test]
    fn out_of_bounds_points_clamp_to_boundary_cells() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [4, 4, 4]);
        // Far negative clamps to cell 0 on every axis.
        assert_eq!(
            cpu_cell_index(Vec3::new(-100.0, -100.0, -100.0), &config),
            0
        );
        // Far positive clamps to the last cell (3, 3, 3) -> 3 + 12 + 48 = 63.
        assert_eq!(cpu_cell_index(Vec3::new(100.0, 100.0, 100.0), &config), 63);
    }

    #[test]
    fn build_orders_by_cell_and_records_ranges() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [2, 1, 1]);
        // Cells: p0 -> 1, p1 -> 0, p2 -> 1, p3 -> 0.
        let positions = [
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(1.2, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
        ];
        let build = cpu_grid_sort(&positions, &config);
        // Cell 0 keeps input order (p1, p3), then cell 1 (p0, p2).
        assert_eq!(build.sorted_indices, vec![1, 3, 0, 2]);
        assert_eq!(build.cell_start, vec![0, 2]);
        assert_eq!(build.cell_end, vec![2, 4]);
    }

    #[test]
    fn empty_cells_stay_sentinel() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [3, 1, 1]);
        // Only cell 2 is occupied.
        let positions = [Vec3::new(2.5, 0.0, 0.0), Vec3::new(2.1, 0.0, 0.0)];
        let build = cpu_grid_sort(&positions, &config);
        assert_eq!(build.sorted_indices, vec![0, 1]);
        assert_eq!(build.cell_start, vec![EMPTY, EMPTY, 0]);
        assert_eq!(build.cell_end, vec![EMPTY, EMPTY, 2]);
    }

    #[test]
    fn empty_input_yields_all_sentinel_ranges() {
        let config = GridConfig::new(Vec3::ZERO, 1.0, [2, 2, 1]);
        let build = cpu_grid_sort(&[], &config);
        assert!(build.sorted_indices.is_empty());
        assert_eq!(build.cell_start, vec![EMPTY; 4]);
        assert_eq!(build.cell_end, vec![EMPTY; 4]);
    }
}
