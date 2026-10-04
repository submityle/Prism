//! Deterministic cell-sort schedule for `FLIP`/`APIC` particle transfer.
//!
//! The measured `MAC` transfer bottleneck is the particle-to-grid scatter
//! (`P2G`) and grid-to-particle gather (`G2P`): on an Apple `M2` a `24^3`
//! domain of ~10^5 particles spends ~0.46 ms in `P2G` and ~0.50 ms in `G2P`
//! (see the `water::gpu_bench` real-device profile). The cost is dominated by
//! incoherent memory traffic and atomic contention when neighbouring threads
//! touch unrelated faces: particles arrive in spawn order, so two particles
//! adjacent in the pool may splat to faces far apart in memory.
//!
//! Production `FLIP` solvers (and particle fluids generally) fix this with a
//! counting sort that reorders particles into cell-major order before the
//! transfer, so threads in a workgroup touch a compact face neighbourhood and
//! the scatter/gather become cache-friendly. This module owns the *schedule*
//! of that sort as pure, deterministic functions: the per-cell histogram, the
//! exclusive prefix sum that turns counts into `CSR` offsets, and the stable
//! permutation that lists particle indices cell-major. It is the golden `CPU`
//! reference and the `ABI` contract for the companion `GPU` counting-sort
//! kernels (histogram -> scan -> scatter), exactly as the other `water::gpu`
//! kernels carry a `CPU` twin.
//!
//! The sort is a permutation only: it never moves or mutates particle data, so
//! it is lossless and reversible, and the transfer kernels read particles
//! through [`FlipParticleOrder::sorted_indices`] instead of being rewritten.
//! The result is fully determined by the input order — within one cell the
//! particles keep ascending original index — so the solve stays reproducible
//! regardless of `GPU` thread scheduling. Out-of-grid particles are dropped
//! from the order (never panicked on), matching the transfer kernels, which
//! already skip particles outside the active grid. Only `+ - * /` and
//! comparisons appear; no `f32` equality and no AI/ML.

use alloc::vec::Vec;

use super::{Vec3, EPS};

/// Uniform pressure-cell grid of a `FLIP`/`APIC` `MAC` domain.
///
/// Cells are the pressure cells of edge `dx` (the same cells the divergence
/// and projection stages solve on), indexed row-major with `x` fastest, then
/// `y`, then `z`. This is deliberately a distinct type from the `PBF`
/// smoothing-radius grid: the `FLIP` cell size is the grid spacing, not a
/// kernel radius, and the sort produces a flat `CSR` layout for the device
/// rather than per-cell neighbour buckets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacCellGrid {
    /// Minimum corner of the domain in world space.
    pub origin: Vec3,
    /// Cell edge length `dx` (`> 0`).
    pub dx: f32,
    /// Cell count along x (`>= 1`).
    pub nx: u32,
    /// Cell count along y (`>= 1`).
    pub ny: u32,
    /// Cell count along z (`>= 1`).
    pub nz: u32,
}

impl MacCellGrid {
    /// Total number of pressure cells.
    #[must_use]
    pub fn cell_count(self) -> usize {
        (self.nx as usize)
            .saturating_mul(self.ny as usize)
            .saturating_mul(self.nz as usize)
    }

    /// Integer cell coordinate of a world position, or `None` when the position
    /// falls outside the grid (including a degenerate `dx <= EPS`).
    #[must_use]
    pub fn cell_coord(self, p: Vec3) -> Option<(u32, u32, u32)> {
        if self.dx <= EPS {
            return None;
        }
        let local = p.sub(self.origin);
        if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
            return None;
        }
        let cx = (local.x / self.dx) as u32;
        let cy = (local.y / self.dx) as u32;
        let cz = (local.z / self.dx) as u32;
        if cx >= self.nx || cy >= self.ny || cz >= self.nz {
            return None;
        }
        Some((cx, cy, cz))
    }

    /// Row-major flat index for an in-range cell coordinate.
    #[must_use]
    pub fn flat_index(self, cx: u32, cy: u32, cz: u32) -> Option<usize> {
        if cx >= self.nx || cy >= self.ny || cz >= self.nz {
            return None;
        }
        let nx = self.nx as usize;
        let ny = self.ny as usize;
        Some((cz as usize) * nx * ny + (cy as usize) * nx + (cx as usize))
    }

    /// Flat cell index of a world position, or `None` when out of the grid.
    #[must_use]
    pub fn cell_of(self, p: Vec3) -> Option<usize> {
        let (cx, cy, cz) = self.cell_coord(p)?;
        self.flat_index(cx, cy, cz)
    }
}

/// Per-cell particle histogram: `counts[flat]` is the number of in-grid
/// particles whose centre falls in cell `flat`, row-major.
///
/// Walks particles in index order; out-of-grid particles are skipped. This is
/// the `CPU` reference for the `GPU` histogram pass (one `atomicAdd` per
/// particle into its cell bin).
#[must_use]
pub fn cell_histogram(grid: MacCellGrid, positions: &[Vec3]) -> Vec<u32> {
    let mut counts = vec![0u32; grid.cell_count()];
    for &p in positions {
        if let Some(flat) = grid.cell_of(p) {
            counts[flat] = counts[flat].saturating_add(1);
        }
    }
    counts
}

/// Exclusive prefix sum (scan) of `counts` into `CSR` cell offsets.
///
/// Returns a vector of length `counts.len() + 1`: `offsets[c]` is the start of
/// cell `c` in the sorted order and `offsets[c + 1]` its end, so
/// `offsets[last]` is the total in-grid particle count. This is the `CPU`
/// reference for the `GPU` scan pass that converts the histogram into write
/// cursors. Addition saturates so a pathological overflow clamps rather than
/// wrapping into a bogus cursor.
#[must_use]
pub fn exclusive_prefix_sum(counts: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut running = 0u32;
    offsets.push(0);
    for &c in counts {
        running = running.saturating_add(c);
        offsets.push(running);
    }
    offsets
}

/// Particles reordered cell-major for coherent `FLIP`/`APIC` transfer.
///
/// A `CSR`-style layout: [`cell_offsets`](Self::cell_offsets) has one entry per
/// cell plus a trailing total, and [`sorted_indices`](Self::sorted_indices)
/// lists the original particle indices grouped by cell. The transfer kernels
/// read particles through `sorted_indices`, so adjacent threads touch a compact
/// face neighbourhood. Out-of-grid particles are absent from `sorted_indices`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlipParticleOrder {
    /// `CSR` cell offsets, length `cell_count + 1`; `cell_offsets[last]` equals
    /// the number of in-grid particles.
    pub cell_offsets: Vec<u32>,
    /// Original particle indices in cell-major, ascending-within-cell order.
    pub sorted_indices: Vec<u32>,
}

impl FlipParticleOrder {
    /// Number of in-grid particles placed in the order.
    #[must_use]
    pub fn total(&self) -> usize {
        self.sorted_indices.len()
    }

    /// `true` when no particle was placed (empty input or all out of grid).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sorted_indices.is_empty()
    }

    /// Half-open `[start, end)` slice of [`sorted_indices`](Self::sorted_indices)
    /// holding cell `flat`'s particles, or `None` when `flat` is out of range.
    #[must_use]
    pub fn cell_range(&self, flat: usize) -> Option<(usize, usize)> {
        if flat + 1 >= self.cell_offsets.len() {
            return None;
        }
        Some((
            self.cell_offsets[flat] as usize,
            self.cell_offsets[flat + 1] as usize,
        ))
    }
}

/// Counting-sorts particle indices into cell-major order.
///
/// Deterministic and stable: particles are scanned in index order, so within a
/// cell the original indices stay ascending. Out-of-grid particles are dropped.
/// This is the full `CPU` golden for the `GPU` counting-sort chain (histogram
/// -> exclusive scan -> scatter): the returned [`FlipParticleOrder::cell_offsets`]
/// matches the scan output and [`FlipParticleOrder::sorted_indices`] matches
/// the scatter output when the device walks a per-cell write cursor seeded from
/// those offsets.
#[must_use]
pub fn counting_sort_particles(grid: MacCellGrid, positions: &[Vec3]) -> FlipParticleOrder {
    let counts = cell_histogram(grid, positions);
    let cell_offsets = exclusive_prefix_sum(&counts);
    let total = *cell_offsets.last().unwrap_or(&0) as usize;

    // Per-cell write cursor seeded from the exclusive-scan start of each cell.
    // Scattering particles in index order with an advancing cursor yields the
    // stable ascending-within-cell permutation, identical to what the device
    // scatter pass produces from the same seed.
    let mut cursor: Vec<u32> = cell_offsets[..counts.len()].to_vec();
    let mut sorted_indices = vec![0u32; total];
    for (i, &p) in positions.iter().enumerate() {
        let Some(flat) = grid.cell_of(p) else {
            continue;
        };
        let slot = cursor[flat] as usize;
        if slot < sorted_indices.len() {
            sorted_indices[slot] = i as u32;
            cursor[flat] = cursor[flat].saturating_add(1);
        }
    }

    FlipParticleOrder {
        cell_offsets,
        sorted_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> MacCellGrid {
        MacCellGrid {
            origin: Vec3::new(0.0, 0.0, 0.0),
            dx: 1.0,
            nx: 2,
            ny: 2,
            nz: 2,
        }
    }

    fn at(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn cell_coord_and_flat_index_are_row_major() {
        let g = grid();
        assert_eq!(g.cell_count(), 8);
        assert_eq!(g.cell_of(at(0.5, 0.5, 0.5)), Some(0));
        assert_eq!(g.cell_of(at(1.5, 0.5, 0.5)), Some(1));
        assert_eq!(g.cell_of(at(0.5, 1.5, 0.5)), Some(2));
        assert_eq!(g.cell_of(at(0.5, 0.5, 1.5)), Some(4));
        assert_eq!(g.cell_of(at(1.5, 1.5, 1.5)), Some(7));
    }

    #[test]
    fn out_of_grid_particles_are_dropped_not_panicked() {
        let g = grid();
        // Negative, beyond extent, and degenerate-grid cases.
        assert_eq!(g.cell_of(at(-0.1, 0.5, 0.5)), None);
        assert_eq!(g.cell_of(at(2.5, 0.5, 0.5)), None);
        let degenerate = MacCellGrid { dx: 0.0, ..g };
        assert_eq!(degenerate.cell_of(at(0.5, 0.5, 0.5)), None);
        let order = counting_sort_particles(g, &[at(-5.0, 0.0, 0.0), at(9.0, 9.0, 9.0)]);
        assert!(order.is_empty());
        assert_eq!(order.total(), 0);
        // Offsets still well-formed (all zero, length cells + 1).
        assert_eq!(order.cell_offsets, vec![0; g.cell_count() + 1]);
    }

    #[test]
    fn histogram_counts_only_in_grid_particles() {
        let g = grid();
        let pos = [
            at(0.5, 0.5, 0.5),  // cell 0
            at(0.6, 0.6, 0.6),  // cell 0
            at(1.5, 0.5, 0.5),  // cell 1
            at(-1.0, 0.0, 0.0), // dropped
        ];
        let counts = cell_histogram(g, &pos);
        assert_eq!(counts.len(), 8);
        assert_eq!(counts[0], 2);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[2..].iter().sum::<u32>(), 0);
    }

    #[test]
    fn exclusive_prefix_sum_is_csr_offsets() {
        let counts = [2u32, 1, 0, 3];
        let offsets = exclusive_prefix_sum(&counts);
        assert_eq!(offsets, vec![0, 2, 3, 3, 6]);
        assert_eq!(*offsets.last().unwrap(), counts.iter().sum::<u32>());
    }

    #[test]
    fn counting_sort_is_cell_major_and_stable_within_cell() {
        let g = grid();
        // Interleave two cells so a naive stable requirement is non-trivial:
        // indices 0,2,4 -> cell 0; 1,3 -> cell 1; 5 -> cell 7.
        let pos = [
            at(0.1, 0.1, 0.1), // 0 -> cell 0
            at(1.1, 0.1, 0.1), // 1 -> cell 1
            at(0.2, 0.2, 0.2), // 2 -> cell 0
            at(1.2, 0.2, 0.2), // 3 -> cell 1
            at(0.3, 0.3, 0.3), // 4 -> cell 0
            at(1.9, 1.9, 1.9), // 5 -> cell 7
        ];
        let order = counting_sort_particles(g, &pos);
        assert_eq!(order.total(), 6);
        // CSR: cell 0 has 3, cell 1 has 2, cell 7 has 1.
        assert_eq!(order.cell_range(0), Some((0, 3)));
        assert_eq!(order.cell_range(1), Some((3, 5)));
        assert_eq!(order.cell_range(7), Some((5, 6)));
        // Within each cell, original indices ascend (stable).
        assert_eq!(&order.sorted_indices[0..3], &[0, 2, 4]);
        assert_eq!(&order.sorted_indices[3..5], &[1, 3]);
        assert_eq!(&order.sorted_indices[5..6], &[5]);
    }

    #[test]
    fn sorted_order_is_a_permutation_of_in_grid_particles() {
        let g = grid();
        let pos = [
            at(1.9, 1.9, 1.9),
            at(0.1, 0.1, 0.1),
            at(1.1, 0.1, 1.1),
            at(0.1, 1.1, 0.1),
            at(1.1, 1.1, 1.1),
        ];
        let order = counting_sort_particles(g, &pos);
        let mut seen = order.sorted_indices.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        // cell_range over the whole grid tiles sorted_indices exactly once.
        let mut covered = 0usize;
        for flat in 0..g.cell_count() {
            let (s, e) = order.cell_range(flat).unwrap();
            assert!(s <= e);
            covered += e - s;
        }
        assert_eq!(covered, order.total());
    }

    #[test]
    fn cell_range_out_of_bounds_is_none() {
        let g = grid();
        let order = counting_sort_particles(g, &[at(0.5, 0.5, 0.5)]);
        assert_eq!(order.cell_range(g.cell_count()), None);
        assert!(order.cell_range(g.cell_count() - 1).is_some());
    }

    #[test]
    fn empty_input_yields_empty_well_formed_order() {
        let g = grid();
        let order = counting_sort_particles(g, &[]);
        assert!(order.is_empty());
        assert_eq!(order.cell_offsets, vec![0; g.cell_count() + 1]);
        assert_eq!(cell_histogram(g, &[]), vec![0; g.cell_count()]);
    }

    #[test]
    fn result_is_deterministic() {
        let g = grid();
        let pos = [
            at(1.1, 0.1, 0.1),
            at(0.1, 0.1, 0.1),
            at(0.9, 0.9, 1.9),
            at(1.9, 1.1, 0.1),
        ];
        let a = counting_sort_particles(g, &pos);
        let b = counting_sort_particles(g, &pos);
        assert_eq!(a, b);
    }
}
