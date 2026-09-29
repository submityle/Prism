//! Uniform-grid spatial hashing: the `CPU`-verifiable neighbor-acceleration
//! contract shared by particle collision, boids/flocking and grid-fluid
//! neighborhood queries (design §7 `PerNeighborCell`, §10).
//!
//! Production `GPU` VFX stacks (Unreal `Niagara`'s neighbor grid, `Frostbite`'s
//! FX neighborhood pass, and generic broad-phase grids) accelerate "who is near
//! me" queries by binning particles into a uniform grid and building a
//! counting-sort bucket layout: count per cell, exclusive prefix-sum the counts
//! into per-cell offsets, then stably scatter particle indices into the bucket
//! array. A query then walks the `3x3x3` (or larger) cell neighborhood and only
//! visits the handful of particles that share those buckets, turning an
//! `O(n^2)` all-pairs test into a near-linear sweep.
//!
//! This module owns only the deterministic `CPU` reference of that layout so
//! the `GPU` build can be validated bit for bit. It provides both a *bounded*
//! [`UniformGrid`] (a fixed `AABB` of cells, indices clamped inside) and an
//! *unbounded* spatial hash ([`hash_cell`]) that folds arbitrary integer cell
//! coordinates into a fixed hash-table size for open worlds. Everything is
//! integer/`floor`/`sqrt`-only arithmetic with saturating products and
//! wrapping hash mixes, so nothing panics, wraps unexpectedly, or produces
//! `NaN`. The bucket byte-size helper reuses the `SoA` `u32` stride from
//! [`crate::particle::gpu_layout`] rather than hand-coding it.

use alloc::vec;
use alloc::vec::Vec;

use crate::particle::gpu_layout::U32_STRIDE;

/// Epsilon for float comparisons; direct `==`/`!=` on `f32` is forbidden in
/// this crate, so magnitudes are compared against this threshold instead.
const CMP_EPS: f32 = 1e-6;

/// Large primes used to mix integer cell coordinates in the unbounded hash.
/// These are the widely used spatial-hash constants (Teschner et al.).
const HASH_P1: i32 = 73_856_093;
/// Second mixing prime for the `y` coordinate.
const HASH_P2: i32 = 19_349_663;
/// Third mixing prime for the `z` coordinate.
const HASH_P3: i32 = 83_492_791;

/// A hand-rolled three-component vector, local to this contract layer so the
/// module stays self-contained (it does not depend on the sibling particle
/// vector math).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The `x` component.
    pub x: f32,
    /// The `y` component.
    pub y: f32,
    /// The `z` component.
    pub z: f32,
}

impl Vec3 {
    /// Builds a vector from its three components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// Clamps an already-`floor`ed `f32` into the `i32` domain without tripping a
/// truncating cast: `NaN` maps to `0`, and out-of-range magnitudes saturate at
/// the `i32` bounds (Rust's float-to-int `as` cast saturates by definition).
#[must_use]
fn floor_to_i32(v: f32) -> i32 {
    if v.is_nan() {
        return 0;
    }
    // `i32::MIN` is exactly representable in `f32`; `i32::MAX` rounds up to
    // 2^31, but the saturating float cast maps it back down to `i32::MAX`.
    let lo = -2_147_483_648.0_f32;
    let hi = 2_147_483_647.0_f32;
    let clamped = v.clamp(lo, hi);
    clamped as i32
}

/// A bounded uniform grid: a regular lattice of cubic cells anchored at an
/// origin, used to bin particles for neighbor queries (design §10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformGrid {
    /// Minimum corner (the `AABB` min) of cell `[0, 0, 0]`.
    origin: Vec3,
    /// Edge length of every (cubic) cell; always `>= CMP_EPS`.
    cell_size: f32,
    /// Cell counts along each axis; every component is `>= 1`.
    dims: [u32; 3],
}

impl UniformGrid {
    /// Builds a grid, clamping `cell_size` up to at least `CMP_EPS` (so cell
    /// lookups never divide by zero and `NaN` collapses to the epsilon) and
    /// clamping every `dims` component up to at least `1` (so the grid always
    /// owns at least one cell).
    #[must_use]
    pub fn new(origin: Vec3, cell_size: f32, dims: [u32; 3]) -> Self {
        // `f32::max` returns the non-`NaN` argument, so a `NaN` cell size also
        // collapses to `CMP_EPS`.
        let cell_size = cell_size.max(CMP_EPS);
        let dims = [dims[0].max(1), dims[1].max(1), dims[2].max(1)];
        Self {
            origin,
            cell_size,
            dims,
        }
    }

    /// Returns the grid origin (the `AABB` min corner).
    #[must_use]
    pub const fn origin(&self) -> Vec3 {
        self.origin
    }

    /// Returns the (clamped) cell edge length.
    #[must_use]
    pub const fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Returns the (clamped) per-axis cell counts.
    #[must_use]
    pub const fn dims(&self) -> [u32; 3] {
        self.dims
    }

    /// Total number of cells (`dimX * dimY * dimZ`), computed in `u64` with
    /// saturating multiplies so an enormous grid reports `u64::MAX` instead of
    /// overflowing.
    #[must_use]
    pub fn cell_count(&self) -> u64 {
        u64::from(self.dims[0])
            .saturating_mul(u64::from(self.dims[1]))
            .saturating_mul(u64::from(self.dims[2]))
    }

    /// Integer cell coordinate of a point: `floor((p - origin) / cell_size)`
    /// per axis. Coordinates may be negative or beyond the grid; use
    /// [`UniformGrid::clamp_coord`] to fold them inside.
    #[must_use]
    pub fn cell_coord(&self, p: Vec3) -> [i32; 3] {
        let inv = self.cell_size;
        [
            floor_to_i32(((p.x - self.origin.x) / inv).floor()),
            floor_to_i32(((p.y - self.origin.y) / inv).floor()),
            floor_to_i32(((p.z - self.origin.z) / inv).floor()),
        ]
    }

    /// Clamps a (possibly out-of-range) integer cell coordinate into the valid
    /// `[0, dims - 1]` range on every axis.
    #[must_use]
    pub fn clamp_coord(&self, c: [i32; 3]) -> [u32; 3] {
        [
            clamp_axis(c[0], self.dims[0]),
            clamp_axis(c[1], self.dims[1]),
            clamp_axis(c[2], self.dims[2]),
        ]
    }

    /// Flattens an in-range cell coordinate to a linear index using
    /// `x + y * dimX + z * dimX * dimY`, with saturating arithmetic so a
    /// degenerate coordinate cannot overflow.
    #[must_use]
    pub fn flatten(&self, c: [u32; 3]) -> u32 {
        let dx = self.dims[0];
        let dy = self.dims[1];
        let row = c[1].saturating_mul(dx);
        let slab = c[2].saturating_mul(dx.saturating_mul(dy));
        c[0].saturating_add(row).saturating_add(slab)
    }

    /// Convenience: the linear cell index of a point
    /// (`cell_coord` → `clamp_coord` → `flatten`).
    #[must_use]
    pub fn cell_index(&self, p: Vec3) -> u32 {
        let coord = self.cell_coord(p);
        let clamped = self.clamp_coord(coord);
        self.flatten(clamped)
    }

    /// Whether a point falls inside the grid's `AABB`
    /// (`origin ..= origin + dims * cell_size`), inclusive of the far face.
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        let ext_x = self.cell_size * (self.dims[0] as f32);
        let ext_y = self.cell_size * (self.dims[1] as f32);
        let ext_z = self.cell_size * (self.dims[2] as f32);
        p.x >= self.origin.x
            && p.y >= self.origin.y
            && p.z >= self.origin.z
            && p.x <= self.origin.x + ext_x
            && p.y <= self.origin.y + ext_y
            && p.z <= self.origin.z + ext_z
    }
}

/// Clamps a signed axis coordinate into `[0, dim - 1]` without a lossy cast.
#[must_use]
fn clamp_axis(v: i32, dim: u32) -> u32 {
    if v <= 0 {
        return 0;
    }
    let max_idx = dim.saturating_sub(1);
    let vu = u32::try_from(v).unwrap_or(u32::MAX);
    vu.min(max_idx)
}

/// Unbounded spatial hash: folds an arbitrary integer cell coordinate into a
/// slot in `[0, table_size)` by mixing the axes with large primes.
///
/// The mix uses wrapping multiplies (so extreme coordinates never panic) and a
/// bitwise reinterpret to `u32` before the modulus, matching the standard
/// open-world spatial-hash construction. A `table_size` of `0` yields `0`.
#[must_use]
pub fn hash_cell(c: [i32; 3], table_size: u32) -> u32 {
    if table_size == 0 {
        return 0;
    }
    let mixed =
        c[0].wrapping_mul(HASH_P1) ^ c[1].wrapping_mul(HASH_P2) ^ c[2].wrapping_mul(HASH_P3);
    // Reinterpret the sign bits into an unsigned key without a lossy cast.
    let key = u32::from_ne_bytes(mixed.to_ne_bytes());
    key % table_size
}

/// Counts how many entries land in each cell, given each entry's flattened cell
/// index. The result has length `cell_count`; indices at or beyond
/// `cell_count` are skipped, and per-cell counts saturate.
#[must_use]
pub fn count_cells(indices: &[u32], cell_count: u32) -> Vec<u32> {
    let n = usize::try_from(cell_count).unwrap_or(usize::MAX);
    let mut counts = vec![0u32; n];
    for &idx in indices {
        let slot = usize::try_from(idx).unwrap_or(usize::MAX);
        if slot < n {
            counts[slot] = counts[slot].saturating_add(1);
        }
    }
    counts
}

/// Exclusive prefix sum of per-cell counts, i.e. the starting bucket offset of
/// each cell. The result has length `counts.len() + 1`; element `i` is the sum
/// of the first `i` counts, so the final element is the saturating grand total.
#[must_use]
pub fn prefix_sum(counts: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut acc: u32 = 0;
    offsets.push(acc);
    for &c in counts {
        acc = acc.saturating_add(c);
        offsets.push(acc);
    }
    offsets
}

/// Stably scatters entry positions into their cell buckets (the counting-sort
/// build step). `indices[i]` is entry `i`'s flattened cell; `offsets` is the
/// [`prefix_sum`] of the matching [`count_cells`].
///
/// The returned array holds the reordered entry indices grouped by cell; within
/// a cell the original input order is preserved (stability). Entries whose cell
/// is at or beyond the cell count are skipped.
#[must_use]
pub fn scatter_stable(indices: &[u32], offsets: &[u32]) -> Vec<u32> {
    let total = usize::try_from(offsets.last().copied().unwrap_or(0)).unwrap_or(usize::MAX);
    let cell_count = offsets.len().saturating_sub(1);
    let mut out = vec![0u32; total];
    // A mutable copy of the offsets acts as the per-cell write cursor.
    let mut cursors: Vec<u32> = offsets.to_vec();
    for (i, &cell) in indices.iter().enumerate() {
        let ci = usize::try_from(cell).unwrap_or(usize::MAX);
        if ci >= cell_count {
            continue;
        }
        let slot = usize::try_from(cursors[ci]).unwrap_or(usize::MAX);
        if slot < out.len() {
            out[slot] = u32::try_from(i).unwrap_or(u32::MAX);
            cursors[ci] = cursors[ci].saturating_add(1);
        }
    }
    out
}

/// The 27 cell offsets of the `3x3x3` neighborhood centered on a cell,
/// including the center `[0, 0, 0]`. Ordered by increasing `z`, then `y`, then
/// `x`, so the center is the 14th entry (index `13`).
#[must_use]
pub const fn neighbor_cell_offsets_3x3x3() -> [[i32; 3]; 27] {
    let mut out = [[0i32; 3]; 27];
    let mut n = 0usize;
    let mut dz = -1i32;
    while dz <= 1 {
        let mut dy = -1i32;
        while dy <= 1 {
            let mut dx = -1i32;
            while dx <= 1 {
                out[n] = [dx, dy, dz];
                n += 1;
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    out
}

/// Byte size of the `GPU` per-cell bucket-offset buffer for a grid of
/// `cell_count` cells: one `u32` per cell plus a trailing total, using the
/// shared `SoA` `u32` stride. Saturates rather than overflowing.
#[must_use]
pub fn bucket_offset_buffer_bytes(cell_count: u32) -> usize {
    let entries = usize::try_from(cell_count)
        .unwrap_or(usize::MAX)
        .saturating_add(1);
    entries.saturating_mul(U32_STRIDE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_size_is_clamped_to_epsilon() {
        let o = Vec3::new(0.0, 0.0, 0.0);
        assert!(UniformGrid::new(o, -5.0, [1, 1, 1]).cell_size() >= CMP_EPS);
        assert!(UniformGrid::new(o, 0.0, [1, 1, 1]).cell_size() >= CMP_EPS);
        let nan_grid = UniformGrid::new(o, f32::NAN, [1, 1, 1]);
        assert!(!nan_grid.cell_size().is_nan());
        assert!(nan_grid.cell_size() >= CMP_EPS);
    }

    #[test]
    fn dims_are_clamped_to_one() {
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [0, 0, 0]);
        assert_eq!(g.dims(), [1, 1, 1]);
        assert_eq!(g.cell_count(), 1);
    }

    #[test]
    fn cell_coord_floors_negative_coordinates() {
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [8, 8, 8]);
        assert_eq!(g.cell_coord(Vec3::new(-0.5, -1.5, 2.5)), [-1, -2, 2]);
    }

    #[test]
    fn cell_coord_handles_offset_origin() {
        let g = UniformGrid::new(Vec3::new(10.0, -10.0, 0.0), 2.0, [8, 8, 8]);
        assert_eq!(g.cell_coord(Vec3::new(13.0, -5.0, 5.0)), [1, 2, 2]);
    }

    #[test]
    fn clamp_coord_folds_out_of_range() {
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [4, 5, 6]);
        assert_eq!(g.clamp_coord([-3, 10, 2]), [0, 4, 2]);
        assert_eq!(g.clamp_coord([3, 4, 5]), [3, 4, 5]);
    }

    #[test]
    fn flatten_matches_manual_layout() {
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [5, 6, 7]);
        let flat = g.flatten([2, 3, 1]);
        // 2 + 3*dimX + 1*dimX*dimY = 2 + 15 + 30 = 47.
        assert_eq!(flat, 47);
    }

    #[test]
    fn flatten_round_trips_through_manual_unflatten() {
        let dims = [5u32, 6u32, 7u32];
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, dims);
        let coord = [2u32, 3u32, 1u32];
        let flat = g.flatten(coord);
        let x = flat % dims[0];
        let y = (flat / dims[0]) % dims[1];
        let z = flat / (dims[0] * dims[1]);
        assert_eq!([x, y, z], coord);
    }

    #[test]
    fn cell_index_composes_the_pipeline() {
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [4, 4, 4]);
        // (1.5, 2.5, 0.5) -> coord [1, 2, 0] -> flatten 1 + 2*4 + 0 = 9.
        assert_eq!(g.cell_index(Vec3::new(1.5, 2.5, 0.5)), 9);
    }

    #[test]
    fn cell_index_clamps_points_outside_the_grid() {
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [4, 4, 4]);
        // Far outside -> clamps to the max corner cell [3, 3, 3].
        assert_eq!(
            g.cell_index(Vec3::new(100.0, 100.0, 100.0)),
            g.flatten([3, 3, 3])
        );
        // Far below -> clamps to [0, 0, 0].
        assert_eq!(g.cell_index(Vec3::new(-100.0, -100.0, -100.0)), 0);
    }

    #[test]
    fn contains_distinguishes_inside_and_outside() {
        let g = UniformGrid::new(Vec3::new(0.0, 0.0, 0.0), 1.0, [2, 2, 2]);
        assert!(g.contains(Vec3::new(1.0, 1.0, 1.0)));
        assert!(g.contains(Vec3::new(2.0, 0.0, 0.0)));
        assert!(!g.contains(Vec3::new(2.5, 0.0, 0.0)));
        assert!(!g.contains(Vec3::new(-0.1, 0.0, 0.0)));
    }

    #[test]
    fn hash_cell_is_deterministic_and_in_range() {
        let c = [3, -7, 12];
        let a = hash_cell(c, 1024);
        let b = hash_cell(c, 1024);
        assert_eq!(a, b);
        assert!(a < 1024);
    }

    #[test]
    fn hash_cell_zero_table_is_zero() {
        assert_eq!(hash_cell([9, 9, 9], 0), 0);
    }

    #[test]
    fn hash_cell_survives_extreme_coordinates() {
        // Wrapping multiplies must not panic on `i32::MIN`/`i32::MAX`.
        let h = hash_cell([i32::MIN, i32::MAX, i32::MIN], 97);
        assert!(h < 97);
    }

    #[test]
    fn count_cells_distributes_hits() {
        let counts = count_cells(&[0, 2, 2, 5], 6);
        assert_eq!(counts, vec![1, 0, 2, 0, 0, 1]);
    }

    #[test]
    fn count_cells_skips_out_of_range_indices() {
        let counts = count_cells(&[0, 7, 2], 3);
        assert_eq!(counts, vec![1, 0, 1]);
    }

    #[test]
    fn prefix_sum_is_exclusive_with_total_last() {
        let counts = vec![1u32, 0, 2, 0, 0, 1];
        let offsets = prefix_sum(&counts);
        assert_eq!(offsets, vec![0, 1, 1, 3, 3, 3, 4]);
        assert_eq!(offsets.len(), counts.len() + 1);
        assert_eq!(*offsets.last().unwrap(), 4);
    }

    #[test]
    fn prefix_sum_of_empty_is_single_zero() {
        assert_eq!(prefix_sum(&[]), vec![0]);
    }

    #[test]
    fn scatter_stable_preserves_same_cell_order() {
        // Particle -> cell: [2, 0, 2, 1, 2], 3 cells.
        let indices = [2u32, 0, 2, 1, 2];
        let counts = count_cells(&indices, 3);
        let offsets = prefix_sum(&counts);
        let bucketed = scatter_stable(&indices, &offsets);
        // Cell 2 owns particles 0, 2, 4 in that original order.
        assert_eq!(bucketed, vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn scatter_stable_skips_out_of_bounds_cells() {
        let indices = [0u32, 5, 1];
        let offsets = prefix_sum(&count_cells(&indices, 2));
        let bucketed = scatter_stable(&indices, &offsets);
        assert_eq!(bucketed, vec![0, 2]);
    }

    #[test]
    fn neighbor_offsets_cover_the_full_cube() {
        let offsets = neighbor_cell_offsets_3x3x3();
        assert_eq!(offsets.len(), 27);
        assert_eq!(offsets[13], [0, 0, 0]);
        assert!(offsets.contains(&[0, 0, 0]));
        assert!(offsets.contains(&[-1, -1, -1]));
        assert!(offsets.contains(&[1, 1, 1]));
        // All 27 offsets are distinct.
        let mut seen = 0usize;
        for (i, a) in offsets.iter().enumerate() {
            for b in offsets.iter().skip(i + 1) {
                if a == b {
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, 0);
    }

    #[test]
    fn cell_count_saturates_on_huge_grids() {
        let g = UniformGrid::new(
            Vec3::new(0.0, 0.0, 0.0),
            1.0,
            [u32::MAX, u32::MAX, u32::MAX],
        );
        assert_eq!(g.cell_count(), u64::MAX);
    }

    #[test]
    fn bucket_offset_buffer_bytes_uses_the_shared_stride() {
        assert_eq!(bucket_offset_buffer_bytes(3), 4 * U32_STRIDE);
        assert_eq!(bucket_offset_buffer_bytes(0), U32_STRIDE);
    }
}
