//! Constraint-graph construction and deterministic graph coloring.
//!
//! The cloth solver drives a garment by projecting positional distance
//! constraints between sim-mesh particles (see [`super::Constraint`]). This
//! module turns a woven sim mesh into that constraint set and colors the set
//! into batches that the XPBD solver can process in parallel within a color and
//! serially across colors (design §3, aligning with the physics kernel §3.5).
//!
//! Two responsibilities live here, both pure array-in / array-out so they can
//! be golden-tested on the CPU:
//!
//! 1. **Construction** — a rectangular warp/weft sim grid is expanded into
//!    stretch (structural), shear (quad-diagonal) and bend (skip-a-vertex)
//!    constraints with per-kind compliance, plus long-range attachment (LRA)
//!    and tether leashes to pinned anchors. This mirrors how `Chaos` Cloth and
//!    `NvCloth` derive their constraint sets from a woven panel.
//! 2. **Graph coloring** — a deterministic greedy coloring partitions any
//!    constraint list into color batches where no two constraints in a batch
//!    touch a shared particle. Within a color the projections are independent
//!    (Jacobi / GPU dispatch); colors are applied in order (Gauss-Seidel). The
//!    coloring depends only on the input order, so it is bit-reproducible for
//!    networked or replay determinism.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::{ColorBatch, Compliance, Constraint, ConstraintGraph, ConstraintKind, Vec3};

/// Per-kind compliance for expanding a woven sim grid into constraints.
///
/// Woven anisotropy is expressed as different warp and weft compliance: a
/// fabric that resists stretch along the warp threads but gives along the weft
/// sets a smaller `warp` than `weft`. Bending is usually far softer than
/// stretch, so `bend` is typically the largest compliance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridConstraintParams {
    /// Compliance of structural edges along the warp (row) direction.
    pub warp: Compliance,
    /// Compliance of structural edges along the weft (column) direction.
    pub weft: Compliance,
    /// Compliance of the quad-diagonal shear constraints.
    pub shear: Compliance,
    /// Compliance of the skip-a-vertex bending constraints.
    pub bend: Compliance,
}

/// A rectangular warp/weft sim grid: `rows * cols` particles in row-major order.
///
/// Particle `(r, c)` lives at linear index `r * cols + c`. This is the woven
/// panel the constraint builder walks; a fully stitched 3D garment is a union
/// of such grids plus seam constraints, but the grid is the deterministic unit
/// the golden tests exercise.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClothGrid {
    /// Number of particle rows (warp lines).
    pub rows: u32,
    /// Number of particle columns (weft lines).
    pub cols: u32,
}

impl ClothGrid {
    /// Builds a grid description.
    #[must_use]
    pub fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }

    /// Total particle count `rows * cols`.
    #[must_use]
    pub fn particle_count(self) -> u32 {
        self.rows.saturating_mul(self.cols)
    }

    /// Row-major linear index of particle `(r, c)`.
    #[must_use]
    pub fn index(self, r: u32, c: u32) -> u32 {
        r.saturating_mul(self.cols).saturating_add(c)
    }
}

/// Rest distance between two particle indices, or `None` when either index is
/// out of bounds (so a malformed grid skips the edge instead of panicking).
fn rest_length(positions: &[Vec3], a: u32, b: u32) -> Option<f32> {
    let pa = positions.get(a as usize)?;
    let pb = positions.get(b as usize)?;
    Some(pa.distance(*pb))
}

/// Expands a woven sim grid into stretch, shear and bend constraints.
///
/// * **Stretch** — structural edges between 4-neighbors: warp compliance along
///   rows, weft compliance along columns.
/// * **Shear** — both quad diagonals of every cell, resisting in-plane shear.
/// * **Bend** — skip-a-vertex distance constraints (`(r,c)`–`(r,c+2)` and
///   `(r,c)`–`(r+2,c)`), a distance-based bending model that keeps a single
///   solver kernel; the full dihedral formulation is a future slot.
///
/// Rest lengths are read from `positions`, so the authored (draped) pose is the
/// rest state. Edges whose endpoints fall outside `positions` are skipped. The
/// emitted order is deterministic: all warp edges, then weft, then shear, then
/// bend, each in row-major order.
#[must_use]
pub fn build_grid_constraints(
    grid: ClothGrid,
    positions: &[Vec3],
    params: GridConstraintParams,
) -> Vec<Constraint> {
    let mut out: Vec<Constraint> = Vec::new();
    let rows = grid.rows;
    let cols = grid.cols;

    // Warp structural edges: (r, c) - (r + 1, c).
    for c in 0..cols {
        for r in 0..rows.saturating_sub(1) {
            let a = grid.index(r, c);
            let b = grid.index(r + 1, c);
            if let Some(len) = rest_length(positions, a, b) {
                out.push(Constraint::new(
                    a,
                    b,
                    len,
                    params.warp,
                    ConstraintKind::Stretch,
                ));
            }
        }
    }

    // Weft structural edges: (r, c) - (r, c + 1).
    for r in 0..rows {
        for c in 0..cols.saturating_sub(1) {
            let a = grid.index(r, c);
            let b = grid.index(r, c + 1);
            if let Some(len) = rest_length(positions, a, b) {
                out.push(Constraint::new(
                    a,
                    b,
                    len,
                    params.weft,
                    ConstraintKind::Stretch,
                ));
            }
        }
    }

    // Shear diagonals: both diagonals of each quad cell.
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols.saturating_sub(1) {
            let tl = grid.index(r, c);
            let tr = grid.index(r, c + 1);
            let bl = grid.index(r + 1, c);
            let br = grid.index(r + 1, c + 1);
            if let Some(len) = rest_length(positions, tl, br) {
                out.push(Constraint::new(
                    tl,
                    br,
                    len,
                    params.shear,
                    ConstraintKind::Shear,
                ));
            }
            if let Some(len) = rest_length(positions, tr, bl) {
                out.push(Constraint::new(
                    tr,
                    bl,
                    len,
                    params.shear,
                    ConstraintKind::Shear,
                ));
            }
        }
    }

    // Bend: skip-a-vertex along both axes.
    for c in 0..cols {
        for r in 0..rows.saturating_sub(2) {
            let a = grid.index(r, c);
            let b = grid.index(r + 2, c);
            if let Some(len) = rest_length(positions, a, b) {
                out.push(Constraint::new(
                    a,
                    b,
                    len,
                    params.bend,
                    ConstraintKind::Bend,
                ));
            }
        }
    }
    for r in 0..rows {
        for c in 0..cols.saturating_sub(2) {
            let a = grid.index(r, c);
            let b = grid.index(r, c + 2);
            if let Some(len) = rest_length(positions, a, b) {
                out.push(Constraint::new(
                    a,
                    b,
                    len,
                    params.bend,
                    ConstraintKind::Bend,
                ));
            }
        }
    }

    out
}

/// One long-range or tether leash from a particle to a (usually pinned) anchor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchorLeash {
    /// The moving particle being leashed.
    pub particle: u32,
    /// The anchor particle (typically pinned) the leash is measured against.
    pub anchor: u32,
    /// Maximum allowed distance from the anchor.
    pub max_distance: f32,
}

/// Builds long-range attachment (LRA) constraints from leashes.
///
/// LRA caps a particle's distance to an anchor so fast body motion cannot
/// stretch the cloth into a spike; it is rigid (zero compliance) and one-sided
/// (the solver only projects when the particle is farther than `max_distance`).
/// Leashes referencing `particle == anchor` are dropped as degenerate.
#[must_use]
pub fn build_lra_constraints(leashes: &[AnchorLeash]) -> Vec<Constraint> {
    build_leashes(leashes, ConstraintKind::Lra)
}

/// Builds tether constraints from leashes.
///
/// A tether is the same one-sided leash as [`build_lra_constraints`] but tagged
/// [`ConstraintKind::Tether`], for authored fixed points that should only pull
/// the cloth back when it drifts too far.
#[must_use]
pub fn build_tether_constraints(leashes: &[AnchorLeash]) -> Vec<Constraint> {
    build_leashes(leashes, ConstraintKind::Tether)
}

fn build_leashes(leashes: &[AnchorLeash], kind: ConstraintKind) -> Vec<Constraint> {
    let mut out: Vec<Constraint> = Vec::new();
    for leash in leashes {
        if leash.particle == leash.anchor {
            continue;
        }
        out.push(Constraint::new(
            leash.particle,
            leash.anchor,
            leash.max_distance.max(0.0),
            Compliance::RIGID,
            kind,
        ));
    }
    out
}

/// Colors a constraint list into batches with no shared particle per batch.
///
/// A deterministic greedy coloring: constraints are processed in input order
/// and each is placed in the lowest-indexed color whose particle set is
/// disjoint from the constraint's two endpoints. The output reorders the
/// constraints so every color is a contiguous run described by a
/// [`ColorBatch`], preserving input order within each color.
///
/// The result is fully determined by the input order, so the same constraint
/// list always yields the same coloring — the property networked simulation and
/// replay rely on. Cost is `O(constraints * colors)`; the color count stays
/// near the maximum particle degree (a small constant for a woven grid), so
/// there is no hidden quadratic in the particle count.
#[must_use]
pub fn color_constraints(constraints: &[Constraint]) -> ConstraintGraph {
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

    let mut graph = ConstraintGraph::default();
    for color in 0..used_particles.len() {
        let start = graph.constraints.len() as u32;
        for (i, constraint) in constraints.iter().enumerate() {
            if color_of[i] == color {
                graph.constraints.push(*constraint);
            }
        }
        let len = graph.constraints.len() as u32 - start;
        graph.batches.push(ColorBatch { start, len });
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_grid_positions(grid: ClothGrid, spacing: f32) -> Vec<Vec3> {
        let mut positions = Vec::new();
        for r in 0..grid.rows {
            for c in 0..grid.cols {
                positions.push(Vec3::new(c as f32 * spacing, 0.0, r as f32 * spacing));
            }
        }
        positions
    }

    fn params() -> GridConstraintParams {
        GridConstraintParams {
            warp: Compliance(0.0),
            weft: Compliance(0.001),
            shear: Compliance(0.01),
            bend: Compliance(0.1),
        }
    }

    #[test]
    fn grid_index_is_row_major() {
        let grid = ClothGrid::new(3, 4);
        assert_eq!(grid.index(0, 0), 0);
        assert_eq!(grid.index(1, 0), 4);
        assert_eq!(grid.index(2, 3), 11);
        assert_eq!(grid.particle_count(), 12);
    }

    #[test]
    fn two_by_two_grid_constraint_counts() {
        // 2x2: 2 warp + 2 weft structural, 2 shear diagonals, 0 bend.
        let grid = ClothGrid::new(2, 2);
        let positions = flat_grid_positions(grid, 1.0);
        let constraints = build_grid_constraints(grid, &positions, params());
        let stretch = constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Stretch)
            .count();
        let shear = constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Shear)
            .count();
        let bend = constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Bend)
            .count();
        assert_eq!(stretch, 4);
        assert_eq!(shear, 2);
        assert_eq!(bend, 0);
    }

    #[test]
    fn three_by_three_grid_has_bend_constraints() {
        // 3x3: warp 3*2=6, weft 3*2=6 stretch; shear 2*2*2=8; bend axis
        // warp 3*1=3 + weft 3*1=3 = 6.
        let grid = ClothGrid::new(3, 3);
        let positions = flat_grid_positions(grid, 1.0);
        let constraints = build_grid_constraints(grid, &positions, params());
        let stretch = constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Stretch)
            .count();
        let shear = constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Shear)
            .count();
        let bend = constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Bend)
            .count();
        assert_eq!(stretch, 12);
        assert_eq!(shear, 8);
        assert_eq!(bend, 6);
    }

    #[test]
    fn rest_lengths_come_from_positions() {
        let grid = ClothGrid::new(2, 2);
        let positions = flat_grid_positions(grid, 2.0);
        let constraints = build_grid_constraints(grid, &positions, params());
        // Structural edges span one 2.0 spacing; shear diagonals span sqrt(8).
        let structural = constraints
            .iter()
            .find(|c| c.kind == ConstraintKind::Stretch)
            .expect("has stretch");
        assert!((structural.rest_length - 2.0).abs() < 1.0e-5);
        let diagonal = constraints
            .iter()
            .find(|c| c.kind == ConstraintKind::Shear)
            .expect("has shear");
        assert!((diagonal.rest_length - 8.0_f32.sqrt()).abs() < 1.0e-5);
    }

    #[test]
    fn warp_and_weft_compliance_are_anisotropic() {
        let grid = ClothGrid::new(2, 2);
        let positions = flat_grid_positions(grid, 1.0);
        let constraints = build_grid_constraints(grid, &positions, params());
        // Warp edges are emitted first (column-major over rows), weft after.
        let warp = constraints[0];
        assert_eq!(warp.compliance, Compliance(0.0));
        let weft = constraints
            .iter()
            .filter(|c| c.kind == ConstraintKind::Stretch)
            .nth(2)
            .copied()
            .expect("weft edge exists");
        assert_eq!(weft.compliance, Compliance(0.001));
    }

    #[test]
    fn out_of_bounds_positions_skip_edges() {
        let grid = ClothGrid::new(2, 2);
        // Only one position: every edge references a missing particle.
        let positions = [Vec3::ZERO];
        let constraints = build_grid_constraints(grid, &positions, params());
        assert!(constraints.is_empty());
    }

    #[test]
    fn lra_and_tether_tag_and_drop_degenerate() {
        let leashes = [
            AnchorLeash {
                particle: 5,
                anchor: 0,
                max_distance: 3.0,
            },
            // Degenerate self-leash is dropped.
            AnchorLeash {
                particle: 2,
                anchor: 2,
                max_distance: 1.0,
            },
        ];
        let lra = build_lra_constraints(&leashes);
        assert_eq!(lra.len(), 1);
        assert_eq!(lra[0].kind, ConstraintKind::Lra);
        assert!(lra[0].kind.is_one_sided());
        assert_eq!(lra[0].a, 5);
        assert_eq!(lra[0].b, 0);
        assert!((lra[0].rest_length - 3.0).abs() < 1.0e-6);

        let tether = build_tether_constraints(&leashes);
        assert_eq!(tether.len(), 1);
        assert_eq!(tether[0].kind, ConstraintKind::Tether);
    }

    #[test]
    fn negative_leash_distance_clamps_to_zero() {
        let leashes = [AnchorLeash {
            particle: 1,
            anchor: 0,
            max_distance: -4.0,
        }];
        let lra = build_lra_constraints(&leashes);
        assert!((lra[0].rest_length - 0.0).abs() < 1.0e-6);
    }

    #[test]
    fn coloring_partitions_all_constraints() {
        let grid = ClothGrid::new(4, 4);
        let positions = flat_grid_positions(grid, 1.0);
        let constraints = build_grid_constraints(grid, &positions, params());
        let graph = color_constraints(&constraints);
        assert_eq!(graph.constraint_count(), constraints.len());
        let batched: usize = graph.batches.iter().map(|b| b.len as usize).sum();
        assert_eq!(batched, constraints.len());
    }

    #[test]
    fn no_batch_shares_a_particle() {
        let grid = ClothGrid::new(4, 4);
        let positions = flat_grid_positions(grid, 1.0);
        let constraints = build_grid_constraints(grid, &positions, params());
        let graph = color_constraints(&constraints);
        for batch in &graph.batches {
            let mut seen: BTreeSet<u32> = BTreeSet::new();
            for constraint in graph.batch(*batch) {
                assert!(seen.insert(constraint.a), "particle a repeated in batch");
                assert!(seen.insert(constraint.b), "particle b repeated in batch");
            }
        }
    }

    #[test]
    fn coloring_is_deterministic() {
        let grid = ClothGrid::new(5, 3);
        let positions = flat_grid_positions(grid, 1.0);
        let constraints = build_grid_constraints(grid, &positions, params());
        let a = color_constraints(&constraints);
        let b = color_constraints(&constraints);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_constraints_color_to_empty_graph() {
        let graph = color_constraints(&[]);
        assert!(graph.is_empty());
        assert_eq!(graph.color_count(), 0);
    }

    #[test]
    fn single_particle_chain_needs_multiple_colors() {
        // A fan of edges all sharing particle 0 must each get its own color.
        let constraints = [
            Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
            Constraint::new(0, 2, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
            Constraint::new(0, 3, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
        ];
        let graph = color_constraints(&constraints);
        assert_eq!(graph.color_count(), 3);
    }
}
