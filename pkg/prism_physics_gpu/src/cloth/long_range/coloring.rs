//! Single-endpoint greedy graph colouring for the cloth long-range kernel.
//!
//! A long-range-attachment leash touches exactly one particle, so two leashes
//! conflict only when they share that particle. In the common case each
//! particle carries at most one leash and the whole set is a single colour
//! class; when two leashes do land on the same particle they must run in
//! different passes so their sequential Gauss-Seidel order is preserved. This
//! is the single-endpoint sibling of the three-endpoint bending colourer kept
//! local to the cloth tree.
//!
//! # Provenance
//!
//! Greedy first-fit graph colouring is a textbook algorithm; the application to
//! batched position-based-dynamics constraints is standard GPU PBD practice. No
//! Unreal Engine source or derived code.

use alloc::vec::Vec;

use super::ClothLongRangeConstraint;

/// A deterministic colouring of a long-range-leash set.
///
/// `order` is a permutation of the constraint indices grouped by colour, and
/// `ranges` gives the `(start, count)` span into `order` for each colour. A
/// colour class is an independent set: no two of its leashes share a particle,
/// so a `GPU` pass may process the whole class concurrently.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LongRangeColoring {
    /// Constraint indices grouped by colour; the reorder permutation uploaded
    /// to the device.
    pub order: Vec<u32>,
    /// `(start, count)` span into [`LongRangeColoring::order`] per colour.
    pub ranges: Vec<(u32, u32)>,
}

impl LongRangeColoring {
    /// Number of colour classes (independent compute passes per iteration).
    #[must_use]
    pub fn colour_count(&self) -> usize {
        self.ranges.len()
    }
}

/// Greedily colours `constraints` so no colour class contains two leashes on
/// the same particle, preserving the sequential visit order within each
/// particle.
///
/// The colour of a leash is the number of earlier leashes already assigned to
/// the same particle (first-fit on a single endpoint), which is the minimum
/// number of classes needed. Returns an empty colouring for an empty input.
#[must_use]
pub fn colour_long_range(constraints: &[ClothLongRangeConstraint]) -> LongRangeColoring {
    if constraints.is_empty() {
        return LongRangeColoring::default();
    }

    // How many leashes have already been assigned to each particle; the next
    // one on that particle takes the next colour up.
    let mut per_particle_count: Vec<u32> = Vec::new();
    let mut constraint_colour: Vec<usize> = Vec::with_capacity(constraints.len());
    let mut class_sizes: Vec<u32> = Vec::new();

    for c in constraints {
        let p = c.particle as usize;
        if per_particle_count.len() <= p {
            per_particle_count.resize(p + 1, 0);
        }
        let colour = per_particle_count[p] as usize;
        per_particle_count[p] += 1;
        constraint_colour.push(colour);
        if colour >= class_sizes.len() {
            class_sizes.resize(colour + 1, 0);
        }
        class_sizes[colour] += 1;
    }

    // Prefix-sum the class sizes into contiguous `(start, count)` spans.
    let mut ranges = Vec::with_capacity(class_sizes.len());
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

    LongRangeColoring { order, ranges }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn leash(particle: u32) -> ClothLongRangeConstraint {
        ClothLongRangeConstraint::new(particle, Vec3::ZERO, 1.0, 0.0)
    }

    /// Asserts no colour class contains two leashes on the same particle, the
    /// whole correctness contract of the colourer.
    fn assert_independent(constraints: &[ClothLongRangeConstraint], c: &LongRangeColoring) {
        for &(start, count) in &c.ranges {
            for i in start..start + count {
                for j in (i + 1)..start + count {
                    let pi = constraints[c.order[i as usize] as usize].particle;
                    let pj = constraints[c.order[j as usize] as usize].particle;
                    assert_ne!(pi, pj, "colour class shares particle {pi}");
                }
            }
        }
    }

    #[test]
    fn empty_is_empty() {
        let c = colour_long_range(&[]);
        assert!(c.order.is_empty());
        assert!(c.ranges.is_empty());
        assert_eq!(c.colour_count(), 0);
    }

    #[test]
    fn distinct_particles_share_one_colour() {
        let constraints = [leash(0), leash(1), leash(2), leash(3)];
        let c = colour_long_range(&constraints);
        assert_eq!(c.colour_count(), 1);
        assert_eq!(c.ranges[0], (0, 4));
        assert_independent(&constraints, &c);
    }

    #[test]
    fn repeated_particle_needs_distinct_colours() {
        // Three leashes all on particle 1, so none can share a colour.
        let constraints = [leash(1), leash(1), leash(1)];
        let c = colour_long_range(&constraints);
        assert_eq!(c.colour_count(), 3);
        for &(_, count) in &c.ranges {
            assert_eq!(count, 1);
        }
        assert_independent(&constraints, &c);
    }

    #[test]
    fn order_is_a_permutation() {
        let constraints = [leash(0), leash(1), leash(0), leash(2), leash(1)];
        let c = colour_long_range(&constraints);
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
}
