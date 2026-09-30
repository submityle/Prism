//! Deterministic uniform-grid acceleration structure for strand self-collision.
//!
//! [`super::self_collision`] owns the CPU golden for approximate strand
//! self-collision: it buckets particles into a uniform spatial hash and pushes
//! each particle apart from the neighbors in its own and adjacent cells
//! (design §6.2 / §8 "自碰撞近似", `TressFX` 4 style). That golden builds its
//! grid on the fly with a throwaway [`BTreeMap`] of dynamic buckets and never
//! exposes it, so there is nothing a `GPU` self-collision pass could upload.
//!
//! This module fills that gap purely and deterministically (design §9). It
//! reifies the same bucketing as a reusable [`UniformGrid`] and then flattens
//! it into a `GPU`-uploadable compressed-sparse-row (`CSR`) form
//! ([`GridCsr`]): a `cell_keys` list, a `cell_starts` prefix-sum offset table,
//! and a flat `indices` array. Because the backing map is an ordered
//! [`BTreeMap`] and every bucket is filled in ascending particle index, the
//! `CSR` layout is a pure function of the input array — the same particles
//! always produce byte-identical offsets and indices, which is exactly the
//! determinism a future `hair_self_collision.wesl` twin needs.
//!
//! The cell hashing here is value-for-value identical to
//! [`super::self_collision`] so both sides agree on which particles share a
//! cell. Nothing samples a real random source and no input panics: non-finite
//! particles are skipped, a non-positive or non-finite `cell_size` yields an
//! empty grid, and out-of-range cell lookups return an empty bucket.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::dynamics::{StrandParticle, Vec3};

/// Vectors and coordinates use the same finite-only discipline as the golden.
///
/// Integer grid cell coordinate: `(x, y, z)` cell indices.
pub type GridCell = (i32, i32, i32);

/// Maps a world position to its grid cell for the given `cell_size`.
///
/// This mirrors the private hashing in [`super::self_collision`] exactly:
/// `inv = 1 / cell_size` and each axis is `floor(coord * inv)` truncated to
/// `i32`. The caller is responsible for a positive, finite `cell_size`
/// (see [`UniformGrid::build`]); with a degenerate `cell_size` the result is
/// unspecified but still total (never panics).
#[must_use]
pub fn grid_cell_of(position: Vec3, cell_size: f32) -> GridCell {
    let inv = 1.0 / cell_size;
    (
        (position.x * inv).floor() as i32,
        (position.y * inv).floor() as i32,
        (position.z * inv).floor() as i32,
    )
}

/// A uniform spatial-hash grid over strand particles.
///
/// Buckets are keyed by integer [`GridCell`] and hold ascending particle
/// indices. The backing store is an ordered [`BTreeMap`] so iteration and the
/// derived [`GridCsr`] are deterministic. Build it once per frame from the
/// current particle positions and query neighbors per particle.
#[derive(Clone, Debug, Default)]
pub struct UniformGrid {
    cell_size: f32,
    cells: BTreeMap<GridCell, Vec<u32>>,
    total: usize,
}

impl UniformGrid {
    /// Builds a grid from the particle array using `cell_size`.
    ///
    /// Every particle with a fully finite position is bucketed in ascending
    /// index order, so each bucket stays sorted. A non-positive or non-finite
    /// `cell_size` produces an empty grid rather than a panic, matching the
    /// no-op guard in [`super::self_collision::resolve_self_collision`].
    #[must_use]
    pub fn build(particles: &[StrandParticle], cell_size: f32) -> Self {
        let mut cells: BTreeMap<GridCell, Vec<u32>> = BTreeMap::new();
        let mut total = 0usize;
        if cell_size > 0.0 && cell_size.is_finite() {
            for (index, particle) in particles.iter().enumerate() {
                let p = particle.position;
                if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
                    cells
                        .entry(grid_cell_of(p, cell_size))
                        .or_default()
                        .push(index as u32);
                    total += 1;
                }
            }
        }
        Self {
            cell_size,
            cells,
            total,
        }
    }

    /// The cell edge length this grid was built with.
    #[must_use]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Ascending particle indices stored in `cell`, or an empty slice.
    #[must_use]
    pub fn bucket(&self, cell: GridCell) -> &[u32] {
        self.cells.get(&cell).map_or(&[], Vec::as_slice)
    }

    /// Number of particles bucketed (non-finite ones are excluded).
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// Whether the grid holds no bucketed particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Number of occupied cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Gathers every particle index in the 27 cells around `cell` into `out`.
    ///
    /// `out` is cleared first, then filled from the `-1..=1` neighborhood on
    /// each axis and sorted ascending. Filtering the result by `index > i`
    /// reproduces the candidate set that
    /// [`super::self_collision::resolve_self_collision`] gathers inline for
    /// particle `i`, so the two stay value-for-value in parity. The per-axis
    /// offset uses [`i32::saturating_add`] so a boundary cell (a finite but
    /// extreme position that saturates to `i32::MAX`/`MIN` in
    /// [`grid_cell_of`]) can never overflow — the query stays total and
    /// panic-free, matching the golden.
    pub fn neighbors(&self, cell: GridCell, out: &mut Vec<u32>) {
        out.clear();
        let (cx, cy, cz) = cell;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = self.cells.get(&(
                        cx.saturating_add(dx),
                        cy.saturating_add(dy),
                        cz.saturating_add(dz),
                    )) {
                        out.extend_from_slice(bucket);
                    }
                }
            }
        }
        out.sort_unstable();
    }
}

/// `GPU`-uploadable compressed-sparse-row view of a [`UniformGrid`].
///
/// The three arrays together describe the buckets without any pointers, ready
/// for a `GPU` buffer upload:
/// - `cell_keys[k]` is the `(x, y, z)` coordinate of the `k`-th occupied cell,
///   in ascending [`BTreeMap`] order.
/// - `cell_starts` has `cell_count + 1` entries; bucket `k` occupies
///   `indices[cell_starts[k] .. cell_starts[k + 1]]`, and the final entry
///   equals the total index count.
/// - `indices` is the concatenation of every bucket, each ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GridCsr {
    /// Coordinates of each occupied cell, in ascending order.
    pub cell_keys: Vec<[i32; 3]>,
    /// Prefix-sum offsets; length is `cell_keys.len() + 1`.
    pub cell_starts: Vec<u32>,
    /// Flat concatenation of every bucket's ascending particle indices.
    pub indices: Vec<u32>,
}

impl GridCsr {
    /// Number of occupied cells described by this `CSR`.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cell_keys.len()
    }

    /// Total number of particle indices across all buckets.
    #[must_use]
    pub fn total(&self) -> usize {
        self.indices.len()
    }

    /// Whether this `CSR` describes no buckets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cell_keys.is_empty()
    }

    /// Ascending particle indices of the `k`-th bucket, or an empty slice.
    #[must_use]
    pub fn bucket(&self, k: usize) -> &[u32] {
        if k + 1 >= self.cell_starts.len() {
            return &[];
        }
        let start = self.cell_starts[k] as usize;
        let end = self.cell_starts[k + 1] as usize;
        &self.indices[start..end]
    }
}

/// Flattens a [`UniformGrid`] into its deterministic [`GridCsr`] form.
///
/// Occupied cells are emitted in ascending [`BTreeMap`] order, so the offsets
/// and indices are a pure function of the source grid. An empty grid yields a
/// `CSR` with empty `cell_keys`/`indices` and a single-element `cell_starts`
/// of `[0]`.
#[must_use]
pub fn build_csr(grid: &UniformGrid) -> GridCsr {
    let mut cell_keys: Vec<[i32; 3]> = Vec::with_capacity(grid.cells.len());
    let mut cell_starts: Vec<u32> = Vec::with_capacity(grid.cells.len() + 1);
    let mut indices: Vec<u32> = Vec::with_capacity(grid.total);
    let mut offset: u32 = 0;
    cell_starts.push(offset);
    for (&(x, y, z), bucket) in &grid.cells {
        cell_keys.push([x, y, z]);
        indices.extend_from_slice(bucket);
        offset += bucket.len() as u32;
        cell_starts.push(offset);
    }
    GridCsr {
        cell_keys,
        cell_starts,
        indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn particle(x: f32, y: f32, z: f32) -> StrandParticle {
        StrandParticle::free(Vec3::new(x, y, z))
    }

    /// The private cell hashing in `self_collision` and the public one here
    /// must agree value-for-value, including negative-coordinate `floor`.
    fn reference_cell_of(position: Vec3, cell_size: f32) -> GridCell {
        let inv = 1.0 / cell_size;
        (
            (position.x * inv).floor() as i32,
            (position.y * inv).floor() as i32,
            (position.z * inv).floor() as i32,
        )
    }

    #[test]
    fn cell_hash_matches_reference_including_negatives() {
        let cell_size = 0.5;
        let samples = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.49, 0.51, 0.99),
            Vec3::new(-0.01, -0.5, -0.51),
            Vec3::new(-1.25, 2.75, -3.1),
        ];
        for p in samples {
            assert_eq!(grid_cell_of(p, cell_size), reference_cell_of(p, cell_size));
        }
    }

    #[test]
    fn build_buckets_finite_particles_in_ascending_index_order() {
        let cell_size = 1.0;
        // indices 0 and 2 share a cell; 1 is elsewhere; 3 is non-finite.
        let particles = [
            particle(0.1, 0.1, 0.1),
            particle(5.0, 5.0, 5.0),
            particle(0.2, 0.3, 0.4),
            particle(f32::NAN, 0.0, 0.0),
        ];
        let grid = UniformGrid::build(&particles, cell_size);
        assert_eq!(grid.total(), 3);
        assert_eq!(grid.cell_count(), 2);
        assert!(!grid.is_empty());
        assert_eq!(
            grid.bucket(grid_cell_of(particles[0].position, cell_size)),
            &[0, 2]
        );
        assert_eq!(
            grid.bucket(grid_cell_of(particles[1].position, cell_size)),
            &[1]
        );
        // An unoccupied cell returns an empty slice, not a panic.
        assert!(grid.bucket((100, 100, 100)).is_empty());
    }

    #[test]
    fn invalid_cell_size_yields_empty_grid() {
        let particles = [particle(0.0, 0.0, 0.0), particle(1.0, 1.0, 1.0)];
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let grid = UniformGrid::build(&particles, bad);
            assert!(grid.is_empty());
            assert_eq!(grid.total(), 0);
            assert_eq!(grid.cell_count(), 0);
            let csr = build_csr(&grid);
            assert!(csr.is_empty());
            assert_eq!(csr.total(), 0);
            assert_eq!(csr.cell_starts, alloc::vec![0]);
        }
    }

    #[test]
    fn empty_input_yields_empty_grid_and_csr() {
        let grid = UniformGrid::build(&[], 1.0);
        assert!(grid.is_empty());
        let csr = build_csr(&grid);
        assert!(csr.is_empty());
        assert_eq!(csr.cell_starts, alloc::vec![0]);
        assert!(csr.bucket(0).is_empty());
    }

    #[test]
    fn neighbors_parity_with_inline_gather() {
        let cell_size = 0.5;
        let particles = [
            particle(0.0, 0.0, 0.0),
            particle(0.1, 0.0, 0.0),
            particle(0.4, 0.2, -0.1),
            particle(-0.3, 0.1, 0.2),
            particle(0.6, -0.4, 0.3),
            particle(1.9, 1.9, 1.9),
        ];
        let grid = UniformGrid::build(&particles, cell_size);
        let mut scratch: Vec<u32> = Vec::new();
        for (i, particle) in particles.iter().enumerate() {
            let cell = grid_cell_of(particle.position, cell_size);
            grid.neighbors(cell, &mut scratch);
            // Public neighbors filtered by j > i.
            let from_grid: Vec<u32> = scratch.iter().copied().filter(|&j| j > i as u32).collect();

            // Reference: inline 27-cell gather exactly like `self_collision`.
            let mut reference: Vec<u32> = Vec::new();
            let (cx, cy, cz) = cell;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cx + dx, cy + dy, cz + dz);
                        for &j in grid.bucket(neighbor) {
                            if j > i as u32 {
                                reference.push(j);
                            }
                        }
                    }
                }
            }
            reference.sort_unstable();

            assert_eq!(from_grid, reference, "candidate mismatch at particle {i}");
        }
    }

    #[test]
    fn csr_prefix_sum_and_grouping_are_correct() {
        let cell_size = 1.0;
        let particles = [
            particle(0.1, 0.1, 0.1), // cell A, index 0
            particle(5.0, 5.0, 5.0), // cell B, index 1
            particle(0.3, 0.2, 0.4), // cell A, index 2
            particle(5.1, 5.2, 5.3), // cell B, index 3
            particle(9.0, 0.0, 0.0), // cell C, index 4
        ];
        let grid = UniformGrid::build(&particles, cell_size);
        let csr = build_csr(&grid);

        assert_eq!(csr.cell_count(), grid.cell_count());
        assert_eq!(csr.total(), grid.total());
        // cell_starts length is cell_count + 1, last equals total.
        assert_eq!(csr.cell_starts.len(), csr.cell_count() + 1);
        assert_eq!(*csr.cell_starts.last().unwrap() as usize, csr.total());

        // Each CSR bucket matches the grid bucket for the same cell key, and
        // per-cell length equals the prefix-sum delta.
        for (k, key) in csr.cell_keys.iter().enumerate() {
            let cell = (key[0], key[1], key[2]);
            let grid_bucket = grid.bucket(cell);
            let csr_bucket = csr.bucket(k);
            assert_eq!(csr_bucket, grid_bucket);
            let delta = csr.cell_starts[k + 1] - csr.cell_starts[k];
            assert_eq!(delta as usize, grid_bucket.len());
            // Bucket stays ascending.
            for w in csr_bucket.windows(2) {
                assert!(w[0] < w[1]);
            }
        }

        // cell_keys are strictly ascending in BTreeMap order.
        for w in csr.cell_keys.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn neighbors_at_i32_boundary_never_overflows() {
        // A finite but extreme position saturates to the i32 cell boundary in
        // `grid_cell_of`; gathering its 27-cell neighborhood must not overflow
        // `cx + 1` / `cx - 1`. This asserts the saturating-add contract.
        let grid = UniformGrid::default();
        let mut out = Vec::new();
        grid.neighbors((i32::MAX, i32::MAX, i32::MAX), &mut out);
        assert!(out.is_empty());
        grid.neighbors((i32::MIN, i32::MIN, i32::MIN), &mut out);
        assert!(out.is_empty());
        // Same via a real saturating position fed through grid_cell_of.
        let cell = grid_cell_of(Vec3::new(1.0e30, -1.0e30, 1.0e30), 1.0);
        assert_eq!(cell, (i32::MAX, i32::MIN, i32::MAX));
        grid.neighbors(cell, &mut out);
        assert!(out.is_empty());
    }
}
