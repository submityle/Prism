//! Persistent-thread constraint-graph coloring contract (design doc §8.5
//! item15).
//!
//! `XPBD`/`PBD` cloth and hair solvers are Gauss-Seidel: a constraint reads and
//! writes the positions of the few particles it couples, so two constraints
//! that touch a shared particle must not run at the same instant or they race.
//! The classic parallelisation is *graph coloring*: treat each constraint as a
//! node, connect constraints that share a particle, and color the graph so that
//! every color class is an independent set. All constraints of one color then
//! run fully in parallel, and the solver sweeps colors in sequence — a parallel
//! Gauss-Seidel. On the `GPU` the §8 persistence story keeps a resident
//! `compute` kernel alive across color sweeps (persistent threads that loop over
//! `tile`s of work) so the driver pays per-dispatch setup once instead of once
//! per color batch.
//!
//! This module is the deterministic, panic-free *contract* layer for that
//! scheme. The real resident `GPU` kernel and its dispatch belong to the render
//! graph; this layer only produces the `CPU` contract the dispatch is validated
//! against, value for value against a `golden`. It does several pure jobs
//! (array in, array out, no device state):
//!
//! * **Greedy coloring** — [`greedy_color_constraints`] walks the constraints in
//!   input order and assigns each the smallest color not already used by any
//!   constraint sharing one of its particles. Colors are scanned from `0`
//!   upward, so the same input always yields the same per-constraint color.
//! * **Batch grouping** — [`ColorBatches`] regroups the per-constraint colors
//!   into one list of constraint indices per color, preserving input order
//!   within each batch.
//! * **Persistent-thread planning** — [`plan_persistent_batches`] turns each
//!   color batch into a [`BatchDispatch`], the integer coverage a resident
//!   kernel needs (`tile` count, resident workgroups, work per workgroup). It is
//!   pure integer ceiling arithmetic.
//! * **Bucketing** — [`bin_dispatches`] fans a dispatch list into empty /
//!   partial-`tile` / full-`tile` buckets in one deterministic order-preserving
//!   pass, reusing the `push`/`total`/`is_empty` bucket shape of
//!   [`crate::virtual_geometry::bins`].
//! * **Self-check** — [`verify_coloring`] confirms a coloring is a valid
//!   independent-set partition (no two same-color constraints share a particle),
//!   so a `golden` can be cross-checked. It returns a `bool` and never panics.
//!
//! Everything here is integer / set work — no floating point, no `bitset`
//! hashing with a nondeterministic iteration order — so it needs no `libm`
//! determinism shim and is byte-for-byte reproducible.

use alloc::vec::Vec;

/// Maximum particles a single [`Constraint`] can couple.
///
/// An edge/stretch constraint couples `2` particles and a bending constraint
/// `3`; `4` leaves room for a tetrahedral volume constraint while keeping the
/// descriptor a fixed inline array (no per-constraint heap allocation).
pub const MAX_PARTICLES_PER_CONSTRAINT: usize = 4;

/// Hard upper bound on resident persistent threads, so a bogus parameter cannot
/// request an absurd plan.
pub const MAX_RESIDENT_THREADS: usize = 1 << 20;

/// Hard upper bound on the per-`tile` constraint count.
pub const MAX_TILE_SIZE: usize = 1 << 16;

/// Hard upper bound on the resident workgroups budgeted per color batch.
pub const MAX_WORKGROUP_BUDGET: usize = 1 << 16;

// ---------------------------------------------------------------------------
// Constraint descriptor.
// ---------------------------------------------------------------------------

/// A single solver constraint, described only by the particle indices it
/// couples.
///
/// Particle indices are stored inline in a fixed array capped at
/// [`MAX_PARTICLES_PER_CONSTRAINT`]; constructors truncate anything longer. The
/// descriptor carries no stiffness or rest value because coloring depends only
/// on *which* particles are shared, not on the constraint's physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constraint {
    particles: [usize; MAX_PARTICLES_PER_CONSTRAINT],
    len: usize,
}

impl Constraint {
    /// A two-particle edge / stretch constraint.
    #[must_use]
    pub const fn edge(a: usize, b: usize) -> Self {
        Self {
            particles: [a, b, 0, 0],
            len: 2,
        }
    }

    /// A three-particle bending constraint.
    #[must_use]
    pub const fn bend(a: usize, b: usize, c: usize) -> Self {
        Self {
            particles: [a, b, c, 0],
            len: 3,
        }
    }

    /// A constraint over an arbitrary particle set, truncated to
    /// [`MAX_PARTICLES_PER_CONSTRAINT`]. An empty slice yields a constraint that
    /// couples no particles (it never conflicts with anything).
    #[must_use]
    pub fn from_slice(particles: &[usize]) -> Self {
        let mut data = [0usize; MAX_PARTICLES_PER_CONSTRAINT];
        let len = if particles.len() > MAX_PARTICLES_PER_CONSTRAINT {
            MAX_PARTICLES_PER_CONSTRAINT
        } else {
            particles.len()
        };
        let mut i = 0;
        while i < len {
            data[i] = particles[i];
            i += 1;
        }
        Self {
            particles: data,
            len,
        }
    }

    /// The particle indices this constraint couples.
    #[must_use]
    pub fn particles(&self) -> &[usize] {
        &self.particles[..self.len]
    }

    /// Number of particles coupled.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when this constraint couples no particles.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Greedy graph coloring.
// ---------------------------------------------------------------------------

/// Greedily colors a constraint graph so each color class is independent.
///
/// Returns one color per input constraint, in input order. Constraints are
/// processed in order; each is assigned the smallest color (scanning from `0`
/// upward) not already taken by a previously colored constraint that shares one
/// of its particles. The per-particle "colors already used" state is kept as a
/// growable `Vec<bool>` per particle (a deterministic `bitset`), so the result
/// is a pure, reproducible function of the input.
///
/// A particle index `>= particle_count` references no real particle, so it is
/// ignored for conflict purposes rather than indexed (never a panic). An empty
/// constraint list returns an empty vector; a constraint that couples no valid
/// particles always receives color `0`.
#[must_use]
pub fn greedy_color_constraints(constraints: &[Constraint], particle_count: usize) -> Vec<usize> {
    let mut colors = Vec::with_capacity(constraints.len());
    // `used[p][c] == true` once a colored constraint touching particle `p` has
    // claimed color `c`.
    let mut used: Vec<Vec<bool>> = Vec::new();
    used.resize_with(particle_count, Vec::new);

    for constraint in constraints {
        // Smallest color free for every valid particle of this constraint.
        let mut color = 0usize;
        loop {
            let mut free = true;
            for &p in constraint.particles() {
                if p >= particle_count {
                    continue;
                }
                let per_particle = &used[p];
                if color < per_particle.len() && per_particle[color] {
                    free = false;
                    break;
                }
            }
            if free {
                break;
            }
            color += 1;
        }

        // Record the claim so later constraints see it.
        for &p in constraint.particles() {
            if p >= particle_count {
                continue;
            }
            let per_particle = &mut used[p];
            if per_particle.len() <= color {
                per_particle.resize(color + 1, false);
            }
            per_particle[color] = true;
        }

        colors.push(color);
    }

    colors
}

/// Checks that `colors` is a valid independent-set coloring of `constraints`.
///
/// Returns `true` when no two constraints sharing a (valid) particle were given
/// the same color — i.e. every color batch can run in parallel without a
/// Gauss-Seidel race. A particle index `>= particle_count` is ignored, matching
/// [`greedy_color_constraints`]. Returns `false` (never panics) when the two
/// slices differ in length so a mismatched `golden` is reported rather than
/// crashing.
#[must_use]
pub fn verify_coloring(
    constraints: &[Constraint],
    colors: &[usize],
    particle_count: usize,
) -> bool {
    if constraints.len() != colors.len() {
        return false;
    }
    // `seen[c]` lists the valid particles already claimed by color `c`.
    let mut seen: Vec<Vec<usize>> = Vec::new();
    for (constraint, &color) in constraints.iter().zip(colors.iter()) {
        if seen.len() <= color {
            seen.resize_with(color + 1, Vec::new);
        }
        for &p in constraint.particles() {
            if p >= particle_count {
                continue;
            }
            if seen[color].contains(&p) {
                return false;
            }
        }
        for &p in constraint.particles() {
            if p >= particle_count {
                continue;
            }
            seen[color].push(p);
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Color batches.
// ---------------------------------------------------------------------------

/// Constraints regrouped by color: one list of constraint indices per color.
///
/// Batch `k` holds the indices of every constraint assigned color `k`, in input
/// order. Produced from a [`greedy_color_constraints`] result; empty batches are
/// preserved so batch index equals color index even if a caller supplies a
/// coloring with color gaps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColorBatches {
    batches: Vec<Vec<usize>>,
}

impl ColorBatches {
    /// Groups a per-constraint color list into per-color constraint-index lists.
    #[must_use]
    pub fn from_colors(colors: &[usize]) -> Self {
        let mut batches: Vec<Vec<usize>> = Vec::new();
        for (index, &color) in colors.iter().enumerate() {
            if batches.len() <= color {
                batches.resize_with(color + 1, Vec::new);
            }
            batches[color].push(index);
        }
        Self { batches }
    }

    /// Number of color batches (the highest color used, plus one).
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Total number of constraints across every batch.
    #[must_use]
    pub fn total(&self) -> usize {
        let mut sum = 0;
        for batch in &self.batches {
            sum += batch.len();
        }
        sum
    }

    /// Returns `true` when no constraint landed in any batch.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Constraint indices of the `k`-th color batch, or `None` when `k` is out
    /// of range.
    #[must_use]
    pub fn batch(&self, k: usize) -> Option<&[usize]> {
        self.batches.get(k).map(Vec::as_slice)
    }
}

// ---------------------------------------------------------------------------
// Persistent-thread parameters.
// ---------------------------------------------------------------------------

/// Tuning for the resident persistent-thread `compute` kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistentThreadParams {
    /// Total persistent threads kept resident across color sweeps.
    pub resident_threads: usize,
    /// Constraints packed into one `tile` of work.
    pub tile_size: usize,
    /// Maximum resident workgroups budgeted for one color batch.
    pub workgroup_budget: usize,
}

impl Default for PersistentThreadParams {
    fn default() -> Self {
        Self {
            resident_threads: 256,
            tile_size: 64,
            workgroup_budget: 64,
        }
    }
}

impl PersistentThreadParams {
    /// Explicit parameters (not clamped; call [`sanitized`](Self::sanitized)
    /// before planning).
    #[must_use]
    pub const fn new(resident_threads: usize, tile_size: usize, workgroup_budget: usize) -> Self {
        Self {
            resident_threads,
            tile_size,
            workgroup_budget,
        }
    }

    /// Clamps every field into `[1, max]` so planning never divides by zero or
    /// emits an absurd plan.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            resident_threads: clamp_pos(self.resident_threads, MAX_RESIDENT_THREADS),
            tile_size: clamp_pos(self.tile_size, MAX_TILE_SIZE),
            workgroup_budget: clamp_pos(self.workgroup_budget, MAX_WORKGROUP_BUDGET),
        }
    }
}

/// Clamps `v` into `[1, max]`.
#[must_use]
fn clamp_pos(v: usize, max: usize) -> usize {
    if v == 0 {
        1
    } else if v > max {
        max
    } else {
        v
    }
}

// ---------------------------------------------------------------------------
// Persistent-thread batch planning.
// ---------------------------------------------------------------------------

/// Integer coverage a resident kernel needs for one color batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchDispatch {
    /// Color index this dispatch covers.
    pub color: usize,
    /// Constraints in the batch.
    pub constraint_count: usize,
    /// `tile` size this plan was built against.
    pub tile_size: usize,
    /// `tile`s needed: `ceil(constraint_count / tile_size)`.
    pub tiles: usize,
    /// Resident workgroups launched: `min(tiles, workgroup_budget)`.
    pub resident_workgroups: usize,
    /// `tile`s each resident workgroup loops over:
    /// `ceil(tiles / resident_workgroups)`.
    pub tiles_per_workgroup: usize,
    /// Threads actually doing work: `min(resident_threads, constraint_count)`.
    pub active_threads: usize,
}

impl BatchDispatch {
    /// Returns `true` when the batch is empty (no constraints, no work).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.constraint_count == 0
    }

    /// Returns `true` when the batch exactly fills its last `tile` (no ragged
    /// remainder).
    #[must_use]
    pub const fn is_full_tile(&self) -> bool {
        self.constraint_count != 0 && self.constraint_count == self.tiles * self.tile_size
    }
}

/// Plans a persistent-thread dispatch for every color batch.
///
/// Produces one [`BatchDispatch`] per batch index `0..batch_count`, so an empty
/// batch yields an all-zero-work dispatch rather than being dropped. The
/// parameters are sanitised first, so all divisions are by a positive `tile`
/// size / workgroup count. Pure integer ceiling arithmetic, fully
/// deterministic.
#[must_use]
pub fn plan_persistent_batches(
    batches: &ColorBatches,
    params: PersistentThreadParams,
) -> Vec<BatchDispatch> {
    let params = params.sanitized();
    let mut out = Vec::with_capacity(batches.batch_count());
    for color in 0..batches.batch_count() {
        let constraint_count = batches.batch(color).map_or(0, <[usize]>::len);
        let tiles = constraint_count.div_ceil(params.tile_size);
        let resident_workgroups = if tiles == 0 {
            0
        } else {
            tiles.min(params.workgroup_budget)
        };
        let tiles_per_workgroup = if resident_workgroups == 0 {
            0
        } else {
            tiles.div_ceil(resident_workgroups)
        };
        let active_threads = if constraint_count == 0 {
            0
        } else {
            params.resident_threads.min(constraint_count)
        };
        out.push(BatchDispatch {
            color,
            constraint_count,
            tile_size: params.tile_size,
            tiles,
            resident_workgroups,
            tiles_per_workgroup,
            active_threads,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Dispatch bucketing (mirrors crate::virtual_geometry::bins).
// ---------------------------------------------------------------------------

/// The bucket a [`BatchDispatch`] falls into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchClass {
    /// No constraints, no work.
    Empty,
    /// Has work but a ragged final `tile`.
    PartialTile,
    /// Has work and exactly fills every `tile`.
    FullTile,
}

/// Classifies one dispatch.
#[must_use]
pub fn classify_dispatch(dispatch: &BatchDispatch) -> DispatchClass {
    if dispatch.is_empty() {
        DispatchClass::Empty
    } else if dispatch.is_full_tile() {
        DispatchClass::FullTile
    } else {
        DispatchClass::PartialTile
    }
}

/// Dispatch indices fanned out by [`DispatchClass`].
///
/// Each bucket holds dispatch indices (into the planned slice) in input order,
/// so submission stays deterministic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DispatchBins {
    /// Indices of empty (no-work) dispatches.
    pub empty: Vec<usize>,
    /// Indices of dispatches with a ragged final `tile`.
    pub partial_tile: Vec<usize>,
    /// Indices of dispatches that fully fill every `tile`.
    pub full_tile: Vec<usize>,
}

impl DispatchBins {
    /// Pushes a dispatch index into the bucket named by `class`.
    pub fn push(&mut self, index: usize, class: DispatchClass) {
        match class {
            DispatchClass::Empty => self.empty.push(index),
            DispatchClass::PartialTile => self.partial_tile.push(index),
            DispatchClass::FullTile => self.full_tile.push(index),
        }
    }

    /// Total dispatch indices across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.empty.len() + self.partial_tile.len() + self.full_tile.len()
    }

    /// Returns `true` when no dispatch landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.empty.is_empty() && self.partial_tile.is_empty() && self.full_tile.is_empty()
    }

    /// Read-only view of the bucket named by `class`.
    #[must_use]
    pub fn bucket(&self, class: DispatchClass) -> &[usize] {
        match class {
            DispatchClass::Empty => &self.empty,
            DispatchClass::PartialTile => &self.partial_tile,
            DispatchClass::FullTile => &self.full_tile,
        }
    }
}

/// Fans a planned dispatch slice into per-class buckets in one order-preserving
/// pass.
#[must_use]
pub fn bin_dispatches(dispatches: &[BatchDispatch]) -> DispatchBins {
    let mut bins = DispatchBins::default();
    for (index, dispatch) in dispatches.iter().enumerate() {
        bins.push(index, classify_dispatch(dispatch));
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn empty_input_colors_empty() {
        let colors = greedy_color_constraints(&[], 0);
        assert!(colors.is_empty());
        assert!(verify_coloring(&[], &colors, 0));
    }

    #[test]
    fn single_edge_is_color_zero() {
        let constraints = [Constraint::edge(0, 1)];
        let colors = greedy_color_constraints(&constraints, 2);
        assert_eq!(colors, vec![0]);
        assert!(verify_coloring(&constraints, &colors, 2));
    }

    #[test]
    fn edge_chain_alternates_two_colors() {
        // Path 0-1-2-3: adjacent edges share a particle, so colors alternate.
        let constraints = [
            Constraint::edge(0, 1),
            Constraint::edge(1, 2),
            Constraint::edge(2, 3),
        ];
        let colors = greedy_color_constraints(&constraints, 4);
        assert_eq!(colors, vec![0, 1, 0]);
        assert!(verify_coloring(&constraints, &colors, 4));
    }

    #[test]
    fn edge_ring_needs_three_colors() {
        // Triangle 0-1-2-0: three mutually adjacent edges.
        let constraints = [
            Constraint::edge(0, 1),
            Constraint::edge(1, 2),
            Constraint::edge(2, 0),
        ];
        let colors = greedy_color_constraints(&constraints, 3);
        assert_eq!(colors, vec![0, 1, 2]);
        assert!(verify_coloring(&constraints, &colors, 3));
    }

    #[test]
    fn star_fan_needs_distinct_colors() {
        // All edges share the hub particle 0, so none may share a color.
        let constraints = [
            Constraint::edge(0, 1),
            Constraint::edge(0, 2),
            Constraint::edge(0, 3),
        ];
        let colors = greedy_color_constraints(&constraints, 4);
        assert_eq!(colors, vec![0, 1, 2]);
        assert!(verify_coloring(&constraints, &colors, 4));
    }

    #[test]
    fn disjoint_edges_share_a_color() {
        // Two edges with no shared particle both take color 0.
        let constraints = [Constraint::edge(0, 1), Constraint::edge(2, 3)];
        let colors = greedy_color_constraints(&constraints, 4);
        assert_eq!(colors, vec![0, 0]);
        assert!(verify_coloring(&constraints, &colors, 4));
    }

    #[test]
    fn mixed_edge_and_bend_coloring_is_valid() {
        let constraints = [
            Constraint::edge(0, 1),
            Constraint::bend(0, 1, 2),
            Constraint::edge(2, 3),
            Constraint::from_slice(&[3, 4, 5, 6]),
        ];
        let colors = greedy_color_constraints(&constraints, 7);
        // edge{0,1}=0; bend{0,1,2} conflicts on 0,1 ->1; edge{2,3} conflicts on
        // 2 (color1) ->0; quad{3,4,5,6} conflicts on 3 (color0) ->1.
        assert_eq!(colors, vec![0, 1, 0, 1]);
        assert!(verify_coloring(&constraints, &colors, 7));
    }

    #[test]
    fn out_of_range_particles_are_ignored() {
        // Both constraints reference only invalid particles -> no conflict.
        let constraints = [
            Constraint::edge(10, 11),
            Constraint::from_slice(&[12, 13, 14]),
        ];
        let colors = greedy_color_constraints(&constraints, 2);
        assert_eq!(colors, vec![0, 0]);
        assert!(verify_coloring(&constraints, &colors, 2));
    }

    #[test]
    fn isolated_and_empty_constraints_do_not_panic() {
        let constraints = [
            Constraint::from_slice(&[]),
            Constraint::edge(0, 1),
            Constraint::from_slice(&[]),
        ];
        let colors = greedy_color_constraints(&constraints, 2);
        assert_eq!(colors, vec![0, 0, 0]);
        assert!(verify_coloring(&constraints, &colors, 2));
    }

    #[test]
    fn from_slice_truncates_to_max() {
        let c = Constraint::from_slice(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(c.len(), MAX_PARTICLES_PER_CONSTRAINT);
        assert_eq!(c.particles(), &[1, 2, 3, 4]);
        assert!(!c.is_empty());
    }

    #[test]
    fn color_batches_group_and_report() {
        let colors = vec![0, 1, 0, 2, 1];
        let batches = ColorBatches::from_colors(&colors);
        assert_eq!(batches.batch_count(), 3);
        assert_eq!(batches.total(), 5);
        assert!(!batches.is_empty());
        assert_eq!(batches.batch(0), Some(&[0usize, 2][..]));
        assert_eq!(batches.batch(1), Some(&[1usize, 4][..]));
        assert_eq!(batches.batch(2), Some(&[3usize][..]));
        assert_eq!(batches.batch(3), None);
    }

    #[test]
    fn empty_color_batches_are_empty() {
        let batches = ColorBatches::from_colors(&[]);
        assert_eq!(batches.batch_count(), 0);
        assert_eq!(batches.total(), 0);
        assert!(batches.is_empty());
        assert_eq!(batches.batch(0), None);
    }

    #[test]
    fn params_default_and_sanitize_bounds() {
        let d = PersistentThreadParams::default();
        assert_eq!(d.resident_threads, 256);
        assert_eq!(d.tile_size, 64);
        assert_eq!(d.workgroup_budget, 64);

        // Zeros clamp up to 1; oversized clamps down to the hard caps.
        let s = PersistentThreadParams::new(0, 0, 0).sanitized();
        assert_eq!(s.resident_threads, 1);
        assert_eq!(s.tile_size, 1);
        assert_eq!(s.workgroup_budget, 1);

        let big = PersistentThreadParams::new(usize::MAX, usize::MAX, usize::MAX).sanitized();
        assert_eq!(big.resident_threads, MAX_RESIDENT_THREADS);
        assert_eq!(big.tile_size, MAX_TILE_SIZE);
        assert_eq!(big.workgroup_budget, MAX_WORKGROUP_BUDGET);
    }

    #[test]
    fn plan_rounds_tiles_and_workgroups_up() {
        // One batch of 5 constraints, tile_size 2, budget 2.
        let batches = ColorBatches::from_colors(&[0, 0, 0, 0, 0]);
        let params = PersistentThreadParams::new(8, 2, 2);
        let plan = plan_persistent_batches(&batches, params);
        assert_eq!(plan.len(), 1);
        let d = plan[0];
        assert_eq!(d.color, 0);
        assert_eq!(d.constraint_count, 5);
        assert_eq!(d.tile_size, 2);
        assert_eq!(d.tiles, 3); // ceil(5/2)
        assert_eq!(d.resident_workgroups, 2); // min(3, budget 2)
        assert_eq!(d.tiles_per_workgroup, 2); // ceil(3/2)
        assert_eq!(d.active_threads, 5); // min(8, 5)
        assert!(!d.is_empty());
        assert!(!d.is_full_tile()); // 5 != 3*2
    }

    #[test]
    fn plan_marks_full_tile_batches() {
        let batches = ColorBatches::from_colors(&[0, 0, 0, 0]);
        let params = PersistentThreadParams::new(100, 2, 16);
        let plan = plan_persistent_batches(&batches, params);
        let d = plan[0];
        assert_eq!(d.tiles, 2); // ceil(4/2)
        assert!(d.is_full_tile()); // 4 == 2*2
        assert_eq!(d.active_threads, 4); // min(100, 4)
    }

    #[test]
    fn plan_emits_zero_work_for_empty_batches() {
        // Color 1 is skipped in the input, so batch 1 is empty.
        let batches = ColorBatches::from_colors(&[0, 2]);
        let plan = plan_persistent_batches(&batches, PersistentThreadParams::default());
        assert_eq!(plan.len(), 3);
        let empty = plan[1];
        assert_eq!(empty.constraint_count, 0);
        assert_eq!(empty.tiles, 0);
        assert_eq!(empty.resident_workgroups, 0);
        assert_eq!(empty.tiles_per_workgroup, 0);
        assert_eq!(empty.active_threads, 0);
        assert!(empty.is_empty());
        assert!(!empty.is_full_tile());
    }

    #[test]
    fn bin_dispatches_routes_and_preserves_order() {
        // batch0: 4 constraints (full tile), batch1: empty, batch2: 3 (partial).
        let batches = ColorBatches::from_colors(&[0, 0, 0, 0, 2, 2, 2]);
        let params = PersistentThreadParams::new(64, 2, 16);
        let plan = plan_persistent_batches(&batches, params);
        let bins = bin_dispatches(&plan);
        assert_eq!(bins.total(), 3);
        assert_eq!(bins.full_tile, vec![0]);
        assert_eq!(bins.empty, vec![1]);
        assert_eq!(bins.partial_tile, vec![2]);
        assert_eq!(bins.bucket(DispatchClass::FullTile), &[0usize][..]);
        assert!(!bins.is_empty());
    }

    #[test]
    fn bin_dispatches_empty_input_is_empty() {
        let bins = bin_dispatches(&[]);
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }

    #[test]
    fn verify_rejects_a_conflicting_coloring() {
        // Two edges share particle 1 but are given the same color -> invalid.
        let constraints = [Constraint::edge(0, 1), Constraint::edge(1, 2)];
        assert!(!verify_coloring(&constraints, &[0, 0], 3));
    }

    #[test]
    fn verify_rejects_length_mismatch() {
        let constraints = [Constraint::edge(0, 1)];
        assert!(!verify_coloring(&constraints, &[0, 0], 2));
        assert!(!verify_coloring(&constraints, &[], 2));
    }

    #[test]
    fn coloring_is_bit_exact_across_runs() {
        let constraints = [
            Constraint::edge(0, 1),
            Constraint::bend(1, 2, 3),
            Constraint::edge(3, 4),
            Constraint::edge(0, 4),
            Constraint::from_slice(&[2, 5, 6]),
        ];
        let a = greedy_color_constraints(&constraints, 7);
        let b = greedy_color_constraints(&constraints, 7);
        assert_eq!(a, b);
        assert!(verify_coloring(&constraints, &a, 7));

        let ba = ColorBatches::from_colors(&a);
        let bb = ColorBatches::from_colors(&b);
        assert_eq!(ba, bb);

        let pa = plan_persistent_batches(&ba, PersistentThreadParams::default());
        let pb = plan_persistent_batches(&bb, PersistentThreadParams::default());
        assert_eq!(pa, pb);
        assert_eq!(bin_dispatches(&pa), bin_dispatches(&pb));
    }
}
