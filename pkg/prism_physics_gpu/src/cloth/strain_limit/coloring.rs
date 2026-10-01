//! Two-endpoint greedy graph colouring for the cloth strain-limit kernel.
//!
//! A strain-limit edge touches two particles `a`/`b`, so two edges may run in
//! the same parallel batch only when they share *no* particle. This is the
//! classical graph-colouring reformulation of a Gauss-Seidel sweep: colour the
//! edge conflict graph (an edge for every shared particle), then every colour
//! class is an independent set whose edges write disjoint position slots and
//! can be dispatched as one race-free compute pass.
//!
//! The engine's generic distance-constraint colourer in `crate::xpbd::coloring`
//! also covers the two-endpoint case, but it is fallible (it errors on an
//! out-of-range index and caps at 64 colours). The cloth kernels deliberately
//! use the infallible "skip out-of-range edges individually, grow colours as
//! needed" model (see the bending and long-range siblings), so this keeps a
//! local colourer with that same contract rather than widening the engine
//! colourer's.
//!
//! # Provenance
//!
//! Greedy first-fit graph colouring is a textbook algorithm; the application to
//! batched position-based-dynamics constraints is standard GPU PBD practice. No
//! Unreal Engine source or derived code.

use alloc::vec::Vec;

use super::ClothStrainLimitConstraint;

/// A deterministic colouring of a strain-limit edge set.
///
/// `order` is a permutation of the edge indices grouped by colour, and `ranges`
/// gives the `(start, count)` span into `order` for each colour. A colour class
/// is an independent set: no two of its edges share any particle, so a GPU pass
/// may process the whole class concurrently.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StrainLimitColoring {
    /// Edge indices grouped by colour; the reorder permutation uploaded to the
    /// device.
    pub order: Vec<u32>,
    /// `(start, count)` span into [`StrainLimitColoring::order`] per colour.
    pub ranges: Vec<(u32, u32)>,
}

impl StrainLimitColoring {
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
/// Each edge is assigned the lowest colour not yet used by either of its two
/// particles; a per-particle growable bitset tracks the used colours, so the
/// number of colours is unbounded (it stays small — bounded by the maximum
/// particle degree — for realistic cloth meshes). The result is deterministic:
/// edges are visited in input order and colours are grouped ascending.
///
/// Returns an empty colouring when `constraints` is empty.
#[must_use]
pub fn colour_strain_limit(constraints: &[ClothStrainLimitConstraint]) -> StrainLimitColoring {
    if constraints.is_empty() {
        return StrainLimitColoring::default();
    }

    let particle_count = constraints
        .iter()
        .map(|c| c.a.max(c.b) as usize + 1)
        .max()
        .unwrap_or(0);

    // Per-particle growable bitset of the colours already incident on it.
    let mut particle_colours: Vec<Vec<u64>> = Vec::new();
    particle_colours.resize(particle_count, Vec::new());

    // Colour assigned to each edge, and the running colour-class sizes.
    let mut constraint_colour = Vec::with_capacity(constraints.len());
    let mut class_sizes: Vec<u32> = Vec::new();

    for c in constraints {
        let (ia, ib) = (c.a as usize, c.b as usize);
        // Lowest colour free on both particles.
        let mut colour = 0_usize;
        loop {
            let free =
                !is_used(&particle_colours[ia], colour) && !is_used(&particle_colours[ib], colour);
            if free {
                break;
            }
            colour += 1;
        }
        mark(&mut particle_colours[ia], colour);
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

    // Scatter each edge index into its colour's span. `fill` tracks the next
    // free offset within each colour class.
    let mut order = alloc::vec![0_u32; constraints.len()];
    let mut fill: Vec<u32> = ranges.iter().map(|&(start, _)| start).collect();
    for (idx, &colour) in constraint_colour.iter().enumerate() {
        let slot = fill[colour] as usize;
        order[slot] = u32::try_from(idx).unwrap_or(u32::MAX);
        fill[colour] += 1;
    }

    StrainLimitColoring { order, ranges }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(a: u32, b: u32) -> ClothStrainLimitConstraint {
        ClothStrainLimitConstraint::new(a, b, 1.0, 1.1, 0.0)
    }

    /// Asserts that no colour class contains two edges sharing a particle,
    /// which is the whole correctness contract of the colourer.
    fn assert_independent(constraints: &[ClothStrainLimitConstraint], c: &StrainLimitColoring) {
        for &(start, count) in &c.ranges {
            for i in start..start + count {
                for j in (i + 1)..start + count {
                    let ci = &constraints[c.order[i as usize] as usize];
                    let cj = &constraints[c.order[j as usize] as usize];
                    let si = [ci.a, ci.b];
                    let sj = [cj.a, cj.b];
                    for p in si {
                        assert!(
                            !sj.contains(&p),
                            "colour class shares particle {p} between edges"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn empty_is_empty() {
        let c = colour_strain_limit(&[]);
        assert!(c.order.is_empty());
        assert!(c.ranges.is_empty());
        assert_eq!(c.colour_count(), 0);
    }

    #[test]
    fn disjoint_edges_share_one_colour() {
        let constraints = [edge(0, 1), edge(2, 3), edge(4, 5)];
        let c = colour_strain_limit(&constraints);
        assert_eq!(c.colour_count(), 1);
        assert_eq!(c.ranges[0], (0, 3));
        assert_independent(&constraints, &c);
    }

    #[test]
    fn edges_sharing_a_particle_need_distinct_colours() {
        // All three incident on particle 0, so none can share a colour.
        let constraints = [edge(0, 1), edge(0, 2), edge(0, 3)];
        let c = colour_strain_limit(&constraints);
        assert_eq!(c.colour_count(), 3);
        for &(_, count) in &c.ranges {
            assert_eq!(count, 1);
        }
        assert_independent(&constraints, &c);
    }

    #[test]
    fn order_is_a_permutation() {
        let constraints = [edge(0, 1), edge(1, 2), edge(2, 3), edge(4, 5), edge(3, 6)];
        let c = colour_strain_limit(&constraints);
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
    fn grid_row_colouring_stays_independent() {
        // A chain of edges (i, i+1) overlaps its neighbour, so adjacent edges
        // must land in different colours.
        let constraints: Vec<ClothStrainLimitConstraint> =
            (0..10).map(|i| edge(i, i + 1)).collect();
        let c = colour_strain_limit(&constraints);
        assert!(c.colour_count() >= 2);
        assert_independent(&constraints, &c);
    }
}
