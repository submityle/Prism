//! The result of partitioning a constraint graph into independent islands.
//!
//! An *island* is a connected component of the constraint graph: a maximal set
//! of dynamic particles that are transitively coupled by constraints, together
//! with the constraints that couple them. Two particles in different islands
//! share no constraint, so their sub-solves are completely independent — they
//! can be solved in parallel, and, crucially for this module's purpose, an
//! entire island can be put to *sleep* as a unit once all of its particles come
//! to rest (see [`crate::island::sleep`]).
//!
//! Pinned particles (inverse mass `0`) are deliberately *not* island members and
//! never bridge two islands: an immovable anchor touched by two otherwise
//! separate stacks must leave them independent, exactly as `PhysX`, `Box2D`, and
//! Chaos treat static bodies. A constraint that touches a pinned particle still
//! belongs to the island of its dynamic endpoint.
//!
//! Provenance: standard rigid/particle-solver island partitioning. No Unreal
//! Engine source or derived code.

/// One connected component of the constraint graph.
///
/// Holds the dynamic particles the component couples and the indices (into the
/// caller's constraint list) of the constraints that couple them. Both lists are
/// sorted ascending and deduplicated, so an island has one canonical form
/// regardless of the order the union-find happened to visit edges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Island {
    /// The dynamic particle indices in this island, sorted ascending.
    particles: Vec<u32>,
    /// The constraint indices (into the input list) in this island, sorted
    /// ascending.
    constraints: Vec<u32>,
}

impl Island {
    /// Builds an island from already-sorted, deduplicated member lists.
    ///
    /// This is an internal constructor used by the partitioner; callers obtain
    /// islands through [`build_islands`](crate::island::build_islands).
    #[must_use]
    pub(crate) fn from_sorted(particles: Vec<u32>, constraints: Vec<u32>) -> Island {
        Island {
            particles,
            constraints,
        }
    }

    /// The dynamic particle indices in this island, sorted ascending.
    #[must_use]
    pub fn particles(&self) -> &[u32] {
        &self.particles
    }

    /// The constraint indices (into the input list) in this island, sorted
    /// ascending.
    #[must_use]
    pub fn constraints(&self) -> &[u32] {
        &self.constraints
    }

    /// The number of dynamic particles in this island.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// The number of constraints in this island.
    #[must_use]
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }
}

/// A complete partition of a constraint graph into [`Island`]s.
///
/// Alongside the island list this keeps a reverse index — for each particle, the
/// id of the island it belongs to, or [`None`] for a pinned or unconstrained
/// particle — so a solver or the sleep tracker can look up a particle's island
/// in `O(1)` without scanning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IslandSet {
    /// The islands, indexed by island id.
    islands: Vec<Island>,
    /// For each particle, the id of its island, or [`None`] when the particle is
    /// pinned or touched by no constraint.
    particle_island: Vec<Option<u32>>,
}

impl IslandSet {
    /// Builds an island set from its parts.
    ///
    /// Internal constructor used by the partitioner.
    #[must_use]
    pub(crate) fn new(islands: Vec<Island>, particle_island: Vec<Option<u32>>) -> IslandSet {
        IslandSet {
            islands,
            particle_island,
        }
    }

    /// The islands, indexed by island id.
    #[must_use]
    pub fn islands(&self) -> &[Island] {
        &self.islands
    }

    /// The number of islands in the partition.
    #[must_use]
    pub fn len(&self) -> usize {
        self.islands.len()
    }

    /// Whether the partition contains no islands (no dynamic particle is
    /// coupled by any constraint).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.islands.is_empty()
    }

    /// The id of the island that `particle` belongs to, or [`None`] when it is
    /// pinned or unconstrained.
    ///
    /// Returns [`None`] for an out-of-range particle as well, so a stale index
    /// degrades to "no island" rather than panicking.
    #[must_use]
    pub fn island_of(&self, particle: u32) -> Option<u32> {
        self.particle_island
            .get(particle as usize)
            .copied()
            .flatten()
    }

    /// Borrows the island with the given id, or [`None`] when the id is out of
    /// range.
    #[must_use]
    pub fn island(&self, id: u32) -> Option<&Island> {
        self.islands.get(id as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A freshly built island exposes its sorted members and their counts.
    #[test]
    fn island_reports_members_and_counts() {
        let island = Island::from_sorted(vec![1, 4, 7], vec![0, 2]);
        assert_eq!(island.particles(), &[1, 4, 7]);
        assert_eq!(island.constraints(), &[0, 2]);
        assert_eq!(island.particle_count(), 3);
        assert_eq!(island.constraint_count(), 2);
    }

    /// The reverse index resolves a particle to its island and rejects
    /// out-of-range and unmapped particles with [`None`].
    #[test]
    fn island_of_maps_particles_and_guards_range() {
        let islands = vec![
            Island::from_sorted(vec![0, 2], vec![0]),
            Island::from_sorted(vec![3], vec![]),
        ];
        let mapping = vec![Some(0), None, Some(0), Some(1)];
        let set = IslandSet::new(islands, mapping);

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert_eq!(set.island_of(0), Some(0));
        assert_eq!(set.island_of(2), Some(0));
        assert_eq!(set.island_of(3), Some(1));
        // Particle 1 is unmapped (pinned or free).
        assert_eq!(set.island_of(1), None);
        // Out-of-range degrades to None rather than panicking.
        assert_eq!(set.island_of(99), None);
    }

    /// `island(id)` borrows a valid island and returns [`None`] out of range.
    #[test]
    fn island_lookup_guards_range() {
        let set = IslandSet::new(vec![Island::from_sorted(vec![0], vec![])], vec![Some(0)]);
        assert_eq!(set.island(0).map(Island::particle_count), Some(1));
        assert!(set.island(1).is_none());
    }

    /// An empty partition reports itself empty.
    #[test]
    fn empty_partition_is_empty() {
        let set = IslandSet::new(Vec::new(), vec![None, None]);
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
        assert_eq!(set.island_of(0), None);
    }
}
