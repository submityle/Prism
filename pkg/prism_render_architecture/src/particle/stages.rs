//! Simulation-stage scheduling: iteration domains, spatial-hash neighborhoods,
//! graph-colored constraint batches, and the grid-fluid pass order.
//!
//! An emitter runs an ordered list of *stages* (design §7). Each stage declares
//! the [`IterationDomain`] it iterates so the scheduler can size the compute
//! dispatch and place barriers, and some stages need auxiliary acceleration
//! structures — a spatial hash for neighbor queries (flocking, `SPH`), a
//! graph-colored constraint set for `XPBD`/`VBD` position solves, or the fixed
//! advect/project pass sequence of a voxel fluid (design §10).
//!
//! Everything here is a pure, deterministic `CPU` reference: the spatial hash
//! is a stable counting-sort into a fixed bucket table, the constraint coloring
//! is the same input-order greedy coloring the cloth kernel uses (so the two
//! subsystems batch identically), and the fluid schedule is a fixed ordering.
//! Only `floor` and ordinary arithmetic are used, so the output is bit-
//! reproducible against a future `GPU` kernel.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{IterationDomain, Vec3};

/// Live element counts an emitter exposes for domain-sized dispatch.
///
/// A stage's dispatch size is read from the field matching its
/// [`IterationDomain`]; [`IterationDomain::Custom`] carries its own count and
/// ignores this table.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct StageCounts {
    /// Live particles this frame (drives [`IterationDomain::PerParticle`]).
    pub live_particles: u32,
    /// Occupied spatial-hash cells (drives [`IterationDomain::PerNeighborCell`]).
    pub neighbor_cells: u32,
    /// Voxels in the fluid grid (drives [`IterationDomain::PerGridVoxel`]).
    pub grid_voxels: u32,
    /// Constraints in the batch being solved (drives
    /// [`IterationDomain::PerConstraint`]).
    pub constraints: u32,
    /// Buffered events (drives [`IterationDomain::PerEvent`]).
    pub events: u32,
}

/// Returns the number of invocations a stage over `domain` dispatches.
///
/// Total and deterministic: every domain maps to exactly one field of `counts`
/// (or the domain's own `Custom` count), so a scheduler never has to special-
/// case a missing size.
#[must_use]
pub fn dispatch_domain_size(domain: IterationDomain, counts: StageCounts) -> u32 {
    match domain {
        IterationDomain::PerParticle => counts.live_particles,
        IterationDomain::PerNeighborCell => counts.neighbor_cells,
        IterationDomain::PerGridVoxel => counts.grid_voxels,
        IterationDomain::PerConstraint => counts.constraints,
        IterationDomain::PerEvent => counts.events,
        IterationDomain::Custom(n) => n,
    }
}

/// Number of workgroups needed to cover `invocations` at `workgroup_size`.
///
/// A saturating ceiling division: a zero workgroup size yields zero groups
/// (nothing to dispatch) rather than dividing by zero, and the count never
/// overflows.
#[must_use]
pub fn workgroup_count(invocations: u32, workgroup_size: u32) -> u32 {
    if workgroup_size == 0 || invocations == 0 {
        return 0;
    }
    let last = invocations - 1;
    (last / workgroup_size) + 1
}

/// Integer lattice coordinate of a spatial-hash cell.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CellCoord {
    /// Cell index along X.
    pub x: i32,
    /// Cell index along Y.
    pub y: i32,
    /// Cell index along Z.
    pub z: i32,
}

impl CellCoord {
    /// Builds a cell coordinate.
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// A uniform spatial hash over a fixed bucket table, the neighbor-query
/// acceleration structure for `PerNeighborCell` stages (design §10).
///
/// Positions are bucketed into cubic cells of side `cell_size`; each cell maps
/// to one of `table_size` buckets by an integer hash. The table is filled with
/// a stable counting sort, so entries within a bucket keep insertion (particle-
/// index) order and every query is deterministic. Choosing `cell_size` equal to
/// the interaction radius makes the 27-cell block around a point a superset of
/// every particle within that radius, so [`NeighborGrid::query_ball`] is exact
/// after the per-candidate distance test.
#[derive(Clone, Debug)]
pub struct NeighborGrid {
    cell_size: f32,
    table_size: u32,
    /// CSR bucket offsets, length `table_size + 1`.
    bucket_start: Vec<u32>,
    /// Particle indices grouped by bucket (stable within a bucket).
    entries: Vec<u32>,
    /// Cell coordinate of each entry, parallel to `entries`, so a bucket-hash
    /// collision between two different cells is rejected by an exact compare.
    entry_cells: Vec<CellCoord>,
}

const HASH_PX: u32 = 0x9E37_79B1;
const HASH_PY: u32 = 0x85EB_CA77;
const HASH_PZ: u32 = 0xC2B2_AE3D;

impl NeighborGrid {
    /// Cell coordinate containing `p` for a given cell size.
    ///
    /// Uses `floor` so negative coordinates map to the cell below zero, keeping
    /// the lattice uniform across the origin. A non-positive cell size collapses
    /// every point to the origin cell (a total fallback, not a panic).
    #[must_use]
    pub fn cell_of(p: Vec3, cell_size: f32) -> CellCoord {
        if cell_size <= 0.0 {
            return CellCoord::new(0, 0, 0);
        }
        let inv = 1.0 / cell_size;
        CellCoord::new(
            (p.x * inv).floor() as i32,
            (p.y * inv).floor() as i32,
            (p.z * inv).floor() as i32,
        )
    }

    /// Hashes a cell coordinate into `0..table_size`.
    #[must_use]
    pub fn bucket_of(cell: CellCoord, table_size: u32) -> u32 {
        if table_size == 0 {
            return 0;
        }
        let hx = (cell.x as u32).wrapping_mul(HASH_PX);
        let hy = (cell.y as u32).wrapping_mul(HASH_PY);
        let hz = (cell.z as u32).wrapping_mul(HASH_PZ);
        (hx ^ hy ^ hz) % table_size
    }

    /// Builds the grid from particle positions with a stable counting sort.
    ///
    /// A `table_size` of zero is promoted to one so the structure is always
    /// well-formed. Cost is `O(n + table_size)`.
    #[must_use]
    pub fn build(positions: &[Vec3], cell_size: f32, table_size: u32) -> Self {
        let table_size = table_size.max(1);
        let n = positions.len();
        let mut bucket_start = vec![0u32; table_size as usize + 1];
        let mut buckets = Vec::with_capacity(n);
        for &p in positions {
            let cell = Self::cell_of(p, cell_size);
            let bucket = Self::bucket_of(cell, table_size);
            buckets.push((bucket, cell));
            bucket_start[bucket as usize + 1] += 1;
        }
        for i in 0..table_size as usize {
            bucket_start[i + 1] += bucket_start[i];
        }
        let mut entries = vec![0u32; n];
        let mut entry_cells = vec![CellCoord::new(0, 0, 0); n];
        let mut cursor: Vec<u32> = bucket_start[..table_size as usize].to_vec();
        for (index, &(bucket, cell)) in buckets.iter().enumerate() {
            let slot = cursor[bucket as usize];
            entries[slot as usize] = index as u32;
            entry_cells[slot as usize] = cell;
            cursor[bucket as usize] = slot + 1;
        }
        Self {
            cell_size,
            table_size,
            bucket_start,
            entries,
            entry_cells,
        }
    }

    /// Cell side length.
    #[must_use]
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Bucket table size.
    #[must_use]
    pub fn table_size(&self) -> u32 {
        self.table_size
    }

    /// Total indexed particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no particle is indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Appends the particle indices stored in exactly `cell` to `out`.
    ///
    /// Filters out bucket-hash collisions by comparing the stored cell, so only
    /// particles truly in `cell` are returned, in stable index order.
    fn collect_cell(&self, cell: CellCoord, out: &mut Vec<u32>) {
        let bucket = Self::bucket_of(cell, self.table_size) as usize;
        let start = self.bucket_start[bucket] as usize;
        let end = self.bucket_start[bucket + 1] as usize;
        for slot in start..end {
            if self.entry_cells[slot] == cell {
                out.push(self.entries[slot]);
            }
        }
    }

    /// Returns every particle in the 27-cell block centered on `cell`.
    ///
    /// Cells are visited in a fixed `z`,`y`,`x` order and entries within a cell
    /// keep index order, so the result is deterministic.
    #[must_use]
    pub fn query_cell_block(&self, cell: CellCoord) -> Vec<u32> {
        let mut out = Vec::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    self.collect_cell(
                        CellCoord::new(cell.x + dx, cell.y + dy, cell.z + dz),
                        &mut out,
                    );
                }
            }
        }
        out
    }

    /// Returns the particles within `radius` of `center`.
    ///
    /// Scans the 27-cell block around `center` and keeps candidates inside the
    /// radius. When `cell_size >= radius` the block is a superset of the true
    /// neighborhood, so the result is exact; the query point itself is included
    /// only if a particle sits on it. `positions` must be the slice the grid was
    /// built from.
    #[must_use]
    pub fn query_ball(&self, positions: &[Vec3], center: Vec3, radius: f32) -> Vec<u32> {
        let candidates = self.query_cell_block(Self::cell_of(center, self.cell_size));
        let r2 = radius * radius;
        let mut out = Vec::new();
        for index in candidates {
            if let Some(&p) = positions.get(index as usize)
                && center.distance_squared(p) <= r2
            {
                out.push(index);
            }
        }
        out
    }
}

/// A positional distance constraint between two particles (`XPBD`).
///
/// `rest_length` is the target separation and `compliance` the inverse
/// stiffness (`0` is rigid). This is the particle analogue of the cloth
/// constraint the shared physics kernel projects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleConstraint {
    /// First particle index.
    pub a: u32,
    /// Second particle index.
    pub b: u32,
    /// Target separation.
    pub rest_length: f32,
    /// Inverse stiffness (`0` is rigid).
    pub compliance: f32,
}

impl ParticleConstraint {
    /// Builds a distance constraint.
    #[must_use]
    pub fn new(a: u32, b: u32, rest_length: f32, compliance: f32) -> Self {
        Self {
            a,
            b,
            rest_length,
            compliance,
        }
    }
}

/// A contiguous run of same-color constraints inside a [`ConstraintColoring`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ColorBatch {
    /// Offset of the batch's first constraint in the reordered list.
    pub start: u32,
    /// Number of constraints in the batch.
    pub len: u32,
}

/// A constraint list reordered into disjoint color batches.
///
/// `constraints` is the input reordered so each color is a contiguous run
/// described by the matching entry of `batches`; within a color the projections
/// touch no shared particle and can run in parallel, while colors are applied
/// in order.
#[derive(Clone, Debug, Default)]
pub struct ConstraintColoring {
    /// Constraints reordered so each color is contiguous.
    pub constraints: Vec<ParticleConstraint>,
    /// One batch per color, in application order.
    pub batches: Vec<ColorBatch>,
}

impl ConstraintColoring {
    /// Number of colors (batches).
    #[must_use]
    pub fn color_count(&self) -> usize {
        self.batches.len()
    }

    /// Returns `true` when no constraint was colored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

/// Colors particle constraints into batches with no shared particle per batch.
///
/// The same deterministic, input-order greedy coloring the cloth kernel uses
/// (see [`crate::cloth::constraints`]): each constraint takes the lowest color
/// whose particle set is disjoint from its two endpoints, opening a new color
/// only when none fits. The output reorders the constraints so every color is a
/// contiguous run, preserving input order within a color, so replay and
/// networked simulation see identical batching. Cost is
/// `O(constraints * colors)` with the color count bounded by the maximum
/// particle degree.
#[must_use]
pub fn color_particle_constraints(constraints: &[ParticleConstraint]) -> ConstraintColoring {
    let count = constraints.len();
    let mut color_of = vec![0usize; count];
    let mut used_particles: Vec<BTreeSet<u32>> = Vec::new();

    for (i, constraint) in constraints.iter().enumerate() {
        let a = constraint.a;
        let b = constraint.b;
        let mut chosen: Option<usize> = None;
        for (color, particles) in used_particles.iter().enumerate() {
            if !particles.contains(&a) && !particles.contains(&b) {
                chosen = Some(color);
                break;
            }
        }
        let color = match chosen {
            Some(color) => color,
            None => {
                used_particles.push(BTreeSet::new());
                used_particles.len() - 1
            }
        };
        used_particles[color].insert(a);
        used_particles[color].insert(b);
        color_of[i] = color;
    }

    let mut coloring = ConstraintColoring::default();
    for color in 0..used_particles.len() {
        let start = coloring.constraints.len() as u32;
        for (i, constraint) in constraints.iter().enumerate() {
            if color_of[i] == color {
                coloring.constraints.push(*constraint);
            }
        }
        let len = coloring.constraints.len() as u32 - start;
        coloring.batches.push(ColorBatch { start, len });
    }
    coloring
}

/// One pass of the grid-fluid solve (design §10).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FluidStage {
    /// Inject body forces (gravity, buoyancy, emitter impulses) into velocity.
    AddForces,
    /// Semi-Lagrangian advection of velocity (and scalar fields) along itself.
    Advect,
    /// Viscous diffusion of the velocity field.
    Diffuse,
    /// Compute the divergence of the intermediate velocity field.
    ComputeDivergence,
    /// One Jacobi relaxation of the pressure Poisson solve.
    PressureJacobi,
    /// Subtract the pressure gradient to project velocity divergence-free.
    SubtractGradient,
}

/// Builds the ordered voxel-fluid pass list for a projection iteration count.
///
/// The classic stable-fluids order: add forces, advect, diffuse, take the
/// divergence, relax pressure `pressure_iterations` times, then subtract the
/// gradient to project the field divergence-free. A zero iteration count still
/// emits the surrounding passes so the field advects even without a pressure
/// solve; the returned order is fixed and deterministic.
#[must_use]
pub fn grid_fluid_schedule(pressure_iterations: u32) -> Vec<FluidStage> {
    let mut stages = Vec::with_capacity(4 + pressure_iterations as usize + 1);
    stages.push(FluidStage::AddForces);
    stages.push(FluidStage::Advect);
    stages.push(FluidStage::Diffuse);
    stages.push(FluidStage::ComputeDivergence);
    for _ in 0..pressure_iterations {
        stages.push(FluidStage::PressureJacobi);
    }
    stages.push(FluidStage::SubtractGradient);
    stages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_size_reads_the_matching_domain_field() {
        let counts = StageCounts {
            live_particles: 100,
            neighbor_cells: 12,
            grid_voxels: 4096,
            constraints: 33,
            events: 7,
        };
        assert_eq!(
            dispatch_domain_size(IterationDomain::PerParticle, counts),
            100
        );
        assert_eq!(
            dispatch_domain_size(IterationDomain::PerNeighborCell, counts),
            12
        );
        assert_eq!(
            dispatch_domain_size(IterationDomain::PerGridVoxel, counts),
            4096
        );
        assert_eq!(
            dispatch_domain_size(IterationDomain::PerConstraint, counts),
            33
        );
        assert_eq!(dispatch_domain_size(IterationDomain::PerEvent, counts), 7);
        assert_eq!(dispatch_domain_size(IterationDomain::Custom(9), counts), 9);
    }

    #[test]
    fn workgroup_count_is_saturating_ceiling() {
        assert_eq!(workgroup_count(0, 64), 0);
        assert_eq!(workgroup_count(1, 64), 1);
        assert_eq!(workgroup_count(64, 64), 1);
        assert_eq!(workgroup_count(65, 64), 2);
        assert_eq!(workgroup_count(128, 64), 2);
        // Zero workgroup size never divides by zero.
        assert_eq!(workgroup_count(100, 0), 0);
    }

    #[test]
    fn cell_of_uses_floor_across_the_origin() {
        assert_eq!(
            NeighborGrid::cell_of(Vec3::new(0.5, 1.5, 2.5), 1.0),
            CellCoord::new(0, 1, 2)
        );
        assert_eq!(
            NeighborGrid::cell_of(Vec3::new(-0.5, -1.5, -0.01), 1.0),
            CellCoord::new(-1, -2, -1)
        );
        // Non-positive cell size collapses to the origin cell.
        assert_eq!(
            NeighborGrid::cell_of(Vec3::new(9.0, 9.0, 9.0), 0.0),
            CellCoord::new(0, 0, 0)
        );
    }

    #[test]
    fn empty_grid_is_empty() {
        let grid = NeighborGrid::build(&[], 1.0, 16);
        assert!(grid.is_empty());
        assert_eq!(grid.len(), 0);
        assert!(grid.query_cell_block(CellCoord::new(0, 0, 0)).is_empty());
        assert!(grid.query_ball(&[], Vec3::ZERO, 1.0).is_empty());
    }

    #[test]
    fn query_ball_finds_neighbors_within_radius() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.9, 0.0, 0.0),
            Vec3::new(5.0, 5.0, 5.0),
        ];
        let grid = NeighborGrid::build(&positions, 1.0, 64);
        let mut near = grid.query_ball(&positions, Vec3::new(0.0, 0.0, 0.0), 1.0);
        near.sort_unstable();
        assert_eq!(near, vec![0, 1, 2]);
        // The far particle is excluded.
        assert!(!near.contains(&3));
    }

    #[test]
    fn query_ball_excludes_out_of_radius_within_the_block() {
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.9, 0.9, 0.0)];
        let grid = NeighborGrid::build(&positions, 1.0, 64);
        // Both fall in the 27-cell block, but the second is > 1.0 away.
        let near = grid.query_ball(&positions, Vec3::ZERO, 1.0);
        assert_eq!(near, vec![0]);
    }

    #[test]
    fn cell_block_gathers_only_the_true_cell_members() {
        // Two particles in distinct, non-adjacent cells must not leak into each
        // other's block even if their buckets collide.
        let positions = [Vec3::new(0.5, 0.5, 0.5), Vec3::new(50.5, 0.5, 0.5)];
        let grid = NeighborGrid::build(&positions, 1.0, 8);
        let block = grid.query_cell_block(CellCoord::new(0, 0, 0));
        assert_eq!(block, vec![0]);
    }

    #[test]
    fn query_is_deterministic_and_index_ordered() {
        let positions: Vec<Vec3> = (0..8)
            .map(|i| Vec3::new(0.1 * i as f32, 0.0, 0.0))
            .collect();
        let grid = NeighborGrid::build(&positions, 1.0, 32);
        let a = grid.query_ball(&positions, Vec3::ZERO, 1.0);
        let b = grid.query_ball(&positions, Vec3::ZERO, 1.0);
        assert_eq!(a, b);
        // All eight are within the same cell and returned in index order.
        assert_eq!(a, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    fn constraint(a: u32, b: u32) -> ParticleConstraint {
        ParticleConstraint::new(a, b, 1.0, 0.0)
    }

    #[test]
    fn coloring_empty_is_empty() {
        let coloring = color_particle_constraints(&[]);
        assert!(coloring.is_empty());
        assert_eq!(coloring.color_count(), 0);
    }

    #[test]
    fn coloring_partitions_a_shared_fan() {
        // A fan through particle 0: every constraint shares 0, so each needs
        // its own color.
        let constraints = [constraint(0, 1), constraint(0, 2), constraint(0, 3)];
        let coloring = color_particle_constraints(&constraints);
        assert_eq!(coloring.color_count(), 3);
        for batch in &coloring.batches {
            assert_eq!(batch.len, 1);
        }
    }

    #[test]
    fn coloring_packs_disjoint_constraints_into_one_color() {
        // Three disjoint edges share no particle, so one color holds all three.
        let constraints = [constraint(0, 1), constraint(2, 3), constraint(4, 5)];
        let coloring = color_particle_constraints(&constraints);
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.batches[0].len, 3);
        assert_eq!(coloring.constraints.len(), 3);
    }

    #[test]
    fn coloring_has_no_shared_particle_within_a_batch() {
        let constraints = [
            constraint(0, 1),
            constraint(1, 2),
            constraint(2, 3),
            constraint(3, 4),
        ];
        let coloring = color_particle_constraints(&constraints);
        for batch in &coloring.batches {
            let start = batch.start as usize;
            let end = start + batch.len as usize;
            let mut seen = BTreeSet::new();
            for c in &coloring.constraints[start..end] {
                assert!(seen.insert(c.a), "duplicate particle in batch");
                assert!(seen.insert(c.b), "duplicate particle in batch");
            }
        }
    }

    #[test]
    fn coloring_is_deterministic() {
        let constraints = [
            constraint(0, 1),
            constraint(0, 2),
            constraint(1, 2),
            constraint(3, 4),
        ];
        let a = color_particle_constraints(&constraints);
        let b = color_particle_constraints(&constraints);
        assert_eq!(a.batches, b.batches);
        assert_eq!(a.constraints, b.constraints);
    }

    #[test]
    fn fluid_schedule_orders_passes_around_the_pressure_solve() {
        let schedule = grid_fluid_schedule(3);
        assert_eq!(
            schedule,
            vec![
                FluidStage::AddForces,
                FluidStage::Advect,
                FluidStage::Diffuse,
                FluidStage::ComputeDivergence,
                FluidStage::PressureJacobi,
                FluidStage::PressureJacobi,
                FluidStage::PressureJacobi,
                FluidStage::SubtractGradient,
            ]
        );
    }

    #[test]
    fn fluid_schedule_with_no_pressure_iterations_still_advects() {
        let schedule = grid_fluid_schedule(0);
        assert_eq!(
            schedule,
            vec![
                FluidStage::AddForces,
                FluidStage::Advect,
                FluidStage::Diffuse,
                FluidStage::ComputeDivergence,
                FluidStage::SubtractGradient,
            ]
        );
        assert!(!schedule.contains(&FluidStage::PressureJacobi));
    }
}
