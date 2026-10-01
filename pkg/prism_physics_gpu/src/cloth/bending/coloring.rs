//! Three-endpoint greedy graph colouring for the cloth bending kernel.
//!
//! A single bending constraint touches three particles — the two neighbours
//! `a`/`b` and the hinge `center` — so two constraints may run in the same
//! parallel batch only when they share *no* particle at all. This is the
//! classical graph-colouring reformulation of a Gauss-Seidel sweep: colour the
//! constraint conflict graph (an edge for every shared particle), then every
//! colour class is an independent set whose constraints write disjoint position
//! slots and can be dispatched as one race-free compute pass.
//!
//! The engine's own distance-constraint colourer in `crate::xpbd::coloring`
//! handles only the two-endpoint case, so this is the three-endpoint sibling
//! kept local to the cloth tree rather than widening that module's contract.
//!
//! # Provenance
//!
//! Greedy first-fit graph colouring is a textbook algorithm; the application to
//! batched position-based-dynamics constraints is standard GPU PBD practice. No
//! Unreal Engine source or derived code.

use alloc::vec::Vec;

use super::ClothBendingConstraint;

/// A deterministic colouring of a bending-constraint set.
///
/// `order` is a permutation of the constraint indices grouped by colour, and
/// `ranges` gives the `(start, count)` span into `order` for each colour. A
/// colour class is an independent set: no two of its constraints share any
/// particle, so a GPU pass may process the whole class concurrently.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BendingColoring {
    /// Constraint indices grouped by colour; the reorder permutation uploaded
    /// to the device.
    pub order: Vec<u32>,
    /// `(start, count)` span into [`BendingColoring::order`] per colour.
    pub ranges: Vec<(u32, u32)>,
}

impl BendingColoring {
    /// Number of colour classes (independent compute passes per iteration).
    #[must_use]
    pub fn colour_count(&self) -> usize {
        self.ranges.len()
    }
}

/// Marks `colour` used on a growable per-particle bitset.
fn mark(used: &mut Vec<u64>, colour: usize) {
    let word = colour / 64;
    while used.len() <= word {
        used.push(0);
    }
    used[word] |= 1_u64 << (colour % 64);
}

/// Returns whether `colour` is already used on `used`.
fn is_used(used: &[u64], colour: usize) -> bool {
    let word = colour / 64;
    word < used.len() && (used[word] >> (colour % 64)) & 1 == 1
}

/// Greedily colours `constraints` so each colour class is an independent set.
///
/// Each constraint is assigned the lowest colour not yet used by any of its
/// three particles; a per-particle growable bitset tracks the used colours, so
/// the number of colours is unbounded (it stays small — bounded by the maximum
/// particle degree — for realistic cloth meshes). The result is deterministic:
/// constraints are visited in input order and colours are grouped ascending.
///
/// Returns an empty colouring when `constraints` is empty.
#[must_use]
pub fn colour_bending(constraints: &[ClothBendingConstraint]) -> BendingColoring {
    if constraints.is_empty() {
        return BendingColoring::default();
    }

    let particle_count = constraints
        .iter()
        .map(|c| c.a.max(c.center).max(c.b) as usize + 1)
        .max()
        .unwrap_or(0);

    // Per-particle growable bitset of the colours already incident on it.
    let mut particle_colours: Vec<Vec<u64>> = Vec::new();
    particle_colours.resize(particle_count, Vec::new());

    // Colour assigned to each constraint, and the running colour-class sizes.
    let mut constraint_colour = Vec::with_capacity(constraints.len());
    let mut class_sizes: Vec<u32> = Vec::new();

    for c in constraints {
        let (ia, ic, ib) = (c.a as usize, c.center as usize, c.b as usize);
        // Lowest colour free on all three particles.
        let mut colour = 0_usize;
        loop {
            let free = !is_used(&particle_colours[ia], colour)
                && !is_used(&particle_colours[ic], colour)
                && !is_used(&particle_colours[ib], colour);
            if free {
                break;
            }
            colour += 1;
        }
        mark(&mut particle_colours[ia], colour);
        mark(&mut particle_colours[ic], colour);
        mark(&mut particle_colours[ib], colour);
        constraint_colour.push(colour);
        if colour >= class_sizes.len() {
            class_sizes.resize(colour + 1, 0);
        }
        class_sizes[colour] += 1;
    }

    // Prefix-sum the class sizes into contiguous `(start, count)` spans.
    let colour_count = class_sizes.len();
    let mut ranges = Vec::with_capacity(colour_count);
    let mut cursor = 0_u32;
    for &size in &class_sizes {
        ranges.push((cursor, size));
        cursor += size;
    }

    // Scatter each constraint index into its colour's span. `fill` tracks the
    // next free offset within each colour class.
    let mut order = alloc::vec![0_u32; constraints.len()];
    let mut fill: Vec<u32> = ranges.iter().map(|&(start, _)| start).collect();
    for (idx, &colour) in constraint_colour.iter().enumerate() {
        let slot = fill[colour] as usize;
        order[slot] = u32::try_from(idx).unwrap_or(u32::MAX);
        fill[colour] += 1;
    }

    BendingColoring { order, ranges }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cst(a: u32, center: u32, b: u32) -> ClothBendingConstraint {
        ClothBendingConstraint {
            a,
            center,
            b,
            rest_offset: 0.0,
            compliance: 0.0,
        }
    }

    /// Asserts that no colour class contains two constraints sharing a particle,
    /// which is the whole correctness contract of the colourer.
    fn assert_independent(constraints: &[ClothBendingConstraint], c: &BendingColoring) {
        for &(start, count) in &c.ranges {
            for i in start..start + count {
                for j in (i + 1)..start + count {
                    let ci = &constraints[c.order[i as usize] as usize];
                    let cj = &constraints[c.order[j as usize] as usize];
                    let si = [ci.a, ci.center, ci.b];
                    let sj = [cj.a, cj.center, cj.b];
                    for p in si {
                        assert!(
                            !sj.contains(&p),
                            "colour class shares particle {p} between constraints"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn empty_is_empty() {
        let c = colour_bending(&[]);
        assert!(c.order.is_empty());
        assert!(c.ranges.is_empty());
        assert_eq!(c.colour_count(), 0);
    }

    #[test]
    fn disjoint_constraints_share_one_colour() {
        let constraints = [cst(0, 1, 2), cst(3, 4, 5), cst(6, 7, 8)];
        let c = colour_bending(&constraints);
        assert_eq!(c.colour_count(), 1);
        assert_eq!(c.ranges[0], (0, 3));
        assert_independent(&constraints, &c);
    }

    #[test]
    fn fully_overlapping_center_needs_distinct_colours() {
        // All three hinge on particle 1, so none can share a colour.
        let constraints = [cst(0, 1, 2), cst(3, 1, 4), cst(5, 1, 6)];
        let c = colour_bending(&constraints);
        assert_eq!(c.colour_count(), 3);
        for &(_, count) in &c.ranges {
            assert_eq!(count, 1);
        }
        assert_independent(&constraints, &c);
    }

    #[test]
    fn order_is_a_permutation() {
        let constraints = [
            cst(0, 1, 2),
            cst(2, 3, 4),
            cst(1, 5, 6),
            cst(7, 8, 9),
            cst(4, 10, 11),
        ];
        let c = colour_bending(&constraints);
        let mut seen = c.order.clone();
        seen.sort_unstable();
        let expected: Vec<u32> = (0..constraints.len() as u32).collect();
        assert_eq!(seen, expected);
        // Spans must tile `order` contiguously with no gaps or overlaps.
        let mut cursor = 0_u32;
        for &(start, count) in &c.ranges {
            assert_eq!(start, cursor);
            cursor += count;
        }
        assert_eq!(cursor as usize, constraints.len());
        assert_independent(&constraints, &c);
    }

    #[test]
    fn grid_strip_colouring_stays_independent() {
        // A row of horizontal bending triples (i, i+1, i+2) overlaps its
        // neighbour, so adjacent triples must land in different colours.
        let constraints: Vec<ClothBendingConstraint> =
            (0..10).map(|i| cst(i, i + 1, i + 2)).collect();
        let c = colour_bending(&constraints);
        assert!(c.colour_count() >= 2);
        assert_independent(&constraints, &c);
    }
}
