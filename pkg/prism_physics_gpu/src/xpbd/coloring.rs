//! Greedy graph colouring of the distance-constraint graph.
//!
//! Parallel `XPBD` projection is only race-free when the constraints running
//! concurrently touch disjoint particles. [`Colouring`] partitions the
//! constraints into *colours* such that no two constraints of the same colour
//! share a particle, so every colour can be projected as one parallel dispatch
//! with each particle written by at most one thread. Projecting the colours in
//! sequence (with a barrier between them) then reproduces a Gauss-Seidel sweep
//! exactly, which is why the `CPU` golden and the `GPU` kernel — both walking
//! the identical colour order over the identical reordered constraint list —
//! agree.
//!
//! The colouring is a deterministic first-fit (greedy) assignment: constraints
//! are visited in input order and each takes the lowest colour not already used
//! by a constraint sharing either of its particles. First-fit is not guaranteed
//! optimal, but for the near-regular graphs of cloth, rope, and soft-body
//! lattices it uses close to the minimum (`max degree + 1`) colours, which is
//! all the solver needs.
//!
//! Provenance: textbook greedy (first-fit) graph colouring. No Unreal Engine
//! source or derived code.

use super::config::XpbdError;
use super::constraint::DistanceConstraint;

/// An edge in the constraint graph the colouring partitions.
///
/// The greedy first-fit only ever needs a constraint's two particle indices, so
/// any constraint type that couples exactly two particles (a distance
/// constraint, a sphere contact) implements this and shares the identical
/// colouring machinery. Keeping the colouring generic over the edge — rather
/// than copying it per constraint type — is what lets the contact solver reuse
/// the proven partitioner unchanged.
pub trait ColouredEdge {
    /// Returns the two particle indices this constraint couples.
    fn endpoints(&self) -> (u32, u32);
}

impl ColouredEdge for DistanceConstraint {
    fn endpoints(&self) -> (u32, u32) {
        (self.a, self.b)
    }
}

/// The maximum number of colours the first-fit packing supports.
///
/// Each particle's used-colour set is a single `u64` bitmask, so the ceiling is
/// `64`. Physical meshes never approach this (vertex degree is a handful), so
/// the limit only rejects pathological fully-connected inputs.
pub const MAX_COLOURS: u32 = 64;

/// A colour-partitioned view of a constraint set.
///
/// [`order`](Colouring::order) lists the original constraint indices grouped by
/// colour; [`ranges`](Colouring::ranges) gives the `[start, end)` half-open
/// slice of `order` occupied by each colour. Projecting colour `c` means
/// visiting `order[ranges[c].0 .. ranges[c].1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Colouring {
    /// Original constraint indices, grouped contiguously by colour.
    order: Vec<u32>,
    /// Half-open `[start, end)` ranges into `order`, one per colour.
    ranges: Vec<(u32, u32)>,
}

impl Colouring {
    /// Colours `constraints` so that same-colour constraints share no particle.
    ///
    /// `particle_count` bounds the valid particle indices; every constraint is
    /// range-checked so an out-of-bounds index is reported rather than silently
    /// corrupting the colouring.
    ///
    /// # Errors
    ///
    /// Returns [`XpbdError::ConstraintOutOfRange`] if a constraint references a
    /// particle `>= particle_count`, or [`XpbdError::TooManyColours`] if the
    /// graph needs more than [`MAX_COLOURS`] colours.
    pub fn build<E: ColouredEdge>(
        constraints: &[E],
        particle_count: u32,
    ) -> Result<Colouring, XpbdError> {
        // Per-particle bitmask of colours already taken by an incident
        // constraint. Bit `c` set means "this particle already has a colour-`c`
        // constraint", so a new incident constraint must avoid colour `c`.
        let mut particle_mask = vec![0u64; particle_count as usize];
        let mut colour_of = vec![0u32; constraints.len()];
        let mut colour_count = 0u32;

        for (ci, con) in constraints.iter().enumerate() {
            let (a, b) = con.endpoints();
            check_index(ci as u32, a, particle_count)?;
            check_index(ci as u32, b, particle_count)?;
            let taken = particle_mask[a as usize] | particle_mask[b as usize];
            let colour = lowest_free_bit(taken);
            if colour >= MAX_COLOURS {
                return Err(XpbdError::TooManyColours {
                    colour,
                    maximum: MAX_COLOURS,
                });
            }
            let bit = 1u64 << colour;
            particle_mask[a as usize] |= bit;
            particle_mask[b as usize] |= bit;
            colour_of[ci] = colour;
            colour_count = colour_count.max(colour + 1);
        }

        Ok(Self::group_by_colour(&colour_of, colour_count))
    }

    /// Groups constraint indices contiguously by their assigned colour.
    fn group_by_colour(colour_of: &[u32], colour_count: u32) -> Colouring {
        // Counting sort by colour: stable within a colour, preserving input
        // order so the reordered list is deterministic.
        let mut counts = vec![0u32; colour_count as usize];
        for &c in colour_of {
            counts[c as usize] += 1;
        }
        let mut ranges = Vec::with_capacity(colour_count as usize);
        let mut cursor = 0u32;
        for &count in &counts {
            ranges.push((cursor, cursor + count));
            cursor += count;
        }
        let mut fill = ranges.iter().map(|&(start, _)| start).collect::<Vec<_>>();
        let mut order = vec![0u32; colour_of.len()];
        for (ci, &c) in colour_of.iter().enumerate() {
            let slot = &mut fill[c as usize];
            order[*slot as usize] = ci as u32;
            *slot += 1;
        }
        Colouring { order, ranges }
    }

    /// Number of colours used.
    #[must_use]
    pub fn colour_count(&self) -> u32 {
        self.ranges.len() as u32
    }

    /// The reordered constraint indices, grouped by colour.
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The half-open `[start, end)` ranges into [`order`](Self::order).
    #[must_use]
    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    /// Reorders `items` into colour-grouped order, ready for upload.
    ///
    /// Generic over the element so the same permutation reorders a distance
    /// constraint list or a contact constraint list with one implementation.
    #[must_use]
    pub fn reorder<T: Copy>(&self, items: &[T]) -> Vec<T> {
        self.order.iter().map(|&i| items[i as usize]).collect()
    }
}

/// Range-checks one endpoint of a constraint.
fn check_index(constraint: u32, particle: u32, particle_count: u32) -> Result<(), XpbdError> {
    if particle >= particle_count {
        return Err(XpbdError::ConstraintOutOfRange {
            constraint,
            particle,
            particle_count,
        });
    }
    Ok(())
}

/// Returns the index of the lowest zero bit in `mask`.
///
/// Returns `64` when every bit is set, which the caller treats as
/// [`XpbdError::TooManyColours`].
fn lowest_free_bit(mask: u64) -> u32 {
    (!mask).trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(a: u32, b: u32) -> DistanceConstraint {
        DistanceConstraint::new(a, b, 1.0, 0.0)
    }

    #[test]
    fn lowest_free_bit_finds_first_gap() {
        assert_eq!(lowest_free_bit(0b0), 0);
        assert_eq!(lowest_free_bit(0b1), 1);
        assert_eq!(lowest_free_bit(0b1011), 2);
        assert_eq!(lowest_free_bit(u64::MAX), 64);
    }

    #[test]
    fn disjoint_constraints_share_one_colour() {
        // Two constraints on four distinct particles never conflict.
        let cons = [c(0, 1), c(2, 3)];
        let colouring = Colouring::build(&cons, 4).unwrap();
        assert_eq!(colouring.colour_count(), 1);
        assert_eq!(colouring.ranges(), &[(0, 2)]);
    }

    #[test]
    fn chain_sharing_particles_needs_two_colours() {
        // A path 0-1-2: edges (0,1) and (1,2) share particle 1.
        let cons = [c(0, 1), c(1, 2)];
        let colouring = Colouring::build(&cons, 3).unwrap();
        assert_eq!(colouring.colour_count(), 2);
    }

    #[test]
    fn same_colour_constraints_never_share_a_particle() {
        // A denser fan graph; verify the invariant holds for every colour.
        let cons = [c(0, 1), c(0, 2), c(0, 3), c(1, 2), c(2, 3), c(1, 3)];
        let colouring = Colouring::build(&cons, 4).unwrap();
        for &(start, end) in colouring.ranges() {
            let mut seen = std::collections::HashSet::new();
            for &oi in &colouring.order()[start as usize..end as usize] {
                let con = cons[oi as usize];
                assert!(seen.insert(con.a), "particle {} repeated in colour", con.a);
                assert!(seen.insert(con.b), "particle {} repeated in colour", con.b);
            }
        }
    }

    #[test]
    fn order_is_a_permutation_of_all_constraints() {
        let cons = [c(0, 1), c(1, 2), c(2, 3), c(0, 3)];
        let colouring = Colouring::build(&cons, 4).unwrap();
        let mut sorted = colouring.order().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn out_of_range_particle_is_reported() {
        let cons = [c(0, 5)];
        let err = Colouring::build(&cons, 2).unwrap_err();
        assert_eq!(
            err,
            XpbdError::ConstraintOutOfRange {
                constraint: 0,
                particle: 5,
                particle_count: 2,
            }
        );
    }

    #[test]
    fn reorder_follows_colour_order() {
        let cons = [c(0, 1), c(1, 2)];
        let colouring = Colouring::build(&cons, 3).unwrap();
        let reordered = colouring.reorder(&cons);
        assert_eq!(reordered.len(), 2);
        // Colour 0 holds the first edge, colour 1 the second.
        assert_eq!(reordered[0], cons[colouring.order()[0] as usize]);
    }
}
