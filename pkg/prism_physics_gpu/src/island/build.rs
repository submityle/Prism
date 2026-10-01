//! Partitioning a constraint graph into independent [`Island`]s.
//!
//! [`build_islands`] runs a disjoint-set (union-find) pass over the constraint
//! list, treating each constraint between two *dynamic* particles as an edge
//! that merges their components. Constraints that touch a pinned particle
//! (inverse mass `0`) do not merge through it — a static anchor is shared, never
//! a bridge — so two stacks resting on the same ground floor stay independent
//! islands and can sleep separately. The result is a deterministic, canonical
//! [`IslandSet`]: islands are ordered by their smallest particle index and every
//! member list is sorted ascending, so the same graph always yields byte-equal
//! output regardless of union order.
//!
//! Provenance: textbook union-find (path compression + union by size) applied to
//! solver island partitioning. No Unreal Engine source or derived code.

use crate::xpbd::{ColouredEdge, XpbdError};

use super::set::{Island, IslandSet};

/// A disjoint-set forest with path compression and union by size.
///
/// Only used inside [`build_islands`]; kept private so the island partition is
/// the module's sole public surface.
struct DisjointSet {
    /// Parent link of each element; a root points at itself.
    parent: Vec<u32>,
    /// Subtree size of each root, used to keep unions shallow.
    size: Vec<u32>,
}

impl DisjointSet {
    /// Creates `count` singleton sets.
    fn new(count: usize) -> DisjointSet {
        DisjointSet {
            parent: (0..count as u32).collect(),
            size: vec![1; count],
        }
    }

    /// Returns the representative of `x`'s set, compressing the path to the root.
    fn find(&mut self, x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Second pass: point every node on the path directly at the root.
        let mut node = x;
        while self.parent[node as usize] != root {
            let next = self.parent[node as usize];
            self.parent[node as usize] = root;
            node = next;
        }
        root
    }

    /// Merges the sets containing `a` and `b` (union by size).
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (small, large) = if self.size[ra as usize] < self.size[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = large;
        self.size[large as usize] += self.size[small as usize];
    }
}

/// Partitions `constraints` over `particle_count` particles into islands.
///
/// `inverse_masses` classifies each particle: an inverse mass `> 0` is dynamic,
/// `0` (or less) is a pinned anchor that never bridges islands. A constraint is
/// an island edge only when *both* of its endpoints are dynamic; a constraint
/// with one dynamic endpoint still joins that endpoint's island, and a
/// constraint between two anchors belongs to no island (it can move nothing).
///
/// The returned [`IslandSet`] is canonical: islands sorted by least particle,
/// members sorted ascending.
///
/// # Errors
///
/// Returns [`XpbdError::ConstraintOutOfRange`] when a constraint references a
/// particle index `>= particle_count`, and
/// [`XpbdError::InvalidConfig`] when `inverse_masses` is shorter than
/// `particle_count` (the classification would be undefined).
pub fn build_islands<E: ColouredEdge>(
    constraints: &[E],
    inverse_masses: &[f32],
    particle_count: u32,
) -> Result<IslandSet, XpbdError> {
    if (inverse_masses.len() as u32) < particle_count {
        return Err(XpbdError::InvalidConfig(
            "inverse mass array shorter than particle count",
        ));
    }

    let is_dynamic = |p: u32| inverse_masses[p as usize] > 0.0;

    let mut dsu = DisjointSet::new(particle_count as usize);
    // A particle is an island member only if it is a dynamic endpoint of at
    // least one constraint; free dynamic particles form no island.
    let mut touched = vec![false; particle_count as usize];

    for (ci, con) in constraints.iter().enumerate() {
        let (a, b) = con.endpoints();
        check_index(ci as u32, a, particle_count)?;
        check_index(ci as u32, b, particle_count)?;
        let da = is_dynamic(a);
        let db = is_dynamic(b);
        if da {
            touched[a as usize] = true;
        }
        if db {
            touched[b as usize] = true;
        }
        if da && db {
            dsu.union(a, b);
        }
    }

    // Group dynamic, touched particles by their set root; assign each a slot in
    // a temporary map from root -> growing island.
    let mut root_to_slot: Vec<Option<usize>> = vec![None; particle_count as usize];
    let mut particles_by_slot: Vec<Vec<u32>> = Vec::new();
    let mut constraints_by_slot: Vec<Vec<u32>> = Vec::new();

    for p in 0..particle_count {
        if !is_dynamic(p) || !touched[p as usize] {
            continue;
        }
        let root = dsu.find(p);
        let slot = ensure_slot(
            &mut root_to_slot,
            &mut particles_by_slot,
            &mut constraints_by_slot,
            root,
        );
        particles_by_slot[slot].push(p);
    }

    // Attribute each constraint to the island of its dynamic endpoint. A
    // both-static constraint has no dynamic endpoint and is dropped.
    for (ci, con) in constraints.iter().enumerate() {
        let (a, b) = con.endpoints();
        let anchor = if is_dynamic(a) {
            Some(a)
        } else if is_dynamic(b) {
            Some(b)
        } else {
            None
        };
        let Some(anchor) = anchor else {
            continue;
        };
        let root = dsu.find(anchor);
        let slot = ensure_slot(
            &mut root_to_slot,
            &mut particles_by_slot,
            &mut constraints_by_slot,
            root,
        );
        constraints_by_slot[slot].push(ci as u32);
    }

    // Canonicalise: order islands by their least particle index. Every
    // per-slot particle list is already ascending (built by scanning particles
    // in order) and every constraint list is ascending (built by scanning
    // constraints in order), so only the island order itself needs sorting.
    let mut order: Vec<usize> = (0..particles_by_slot.len()).collect();
    order.sort_by_key(|&slot| particles_by_slot[slot].first().copied().unwrap_or(u32::MAX));

    let mut islands = Vec::with_capacity(order.len());
    let mut particle_island = vec![None; particle_count as usize];
    for (island_id, &slot) in order.iter().enumerate() {
        let particles = std::mem::take(&mut particles_by_slot[slot]);
        let constraints = std::mem::take(&mut constraints_by_slot[slot]);
        for &p in &particles {
            particle_island[p as usize] = Some(island_id as u32);
        }
        islands.push(Island::from_sorted(particles, constraints));
    }

    Ok(IslandSet::new(islands, particle_island))
}

/// Returns the island slot for `root`, allocating a fresh one on first use.
fn ensure_slot(
    root_to_slot: &mut [Option<usize>],
    particles_by_slot: &mut Vec<Vec<u32>>,
    constraints_by_slot: &mut Vec<Vec<u32>>,
    root: u32,
) -> usize {
    match root_to_slot[root as usize] {
        Some(slot) => slot,
        None => {
            let slot = particles_by_slot.len();
            root_to_slot[root as usize] = Some(slot);
            particles_by_slot.push(Vec::new());
            constraints_by_slot.push(Vec::new());
            slot
        }
    }
}

/// Validates that `particle` is a legal index, mirroring the colouring guard so
/// the two stages reject the same malformed graphs identically.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal constraint edge used to drive the partitioner in isolation.
    struct Edge(u32, u32);

    impl ColouredEdge for Edge {
        fn endpoints(&self) -> (u32, u32) {
            (self.0, self.1)
        }
    }

    /// All-dynamic inverse masses of length `n`.
    fn dynamic(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    /// A single edge between two dynamic particles forms one island holding both
    /// endpoints and the constraint.
    #[test]
    fn single_edge_forms_one_island() {
        let edges = [Edge(0, 1)];
        let set = build_islands(&edges, &dynamic(2), 2).expect("valid graph");
        assert_eq!(set.len(), 1);
        assert_eq!(set.island(0).unwrap().particles(), &[0, 1]);
        assert_eq!(set.island(0).unwrap().constraints(), &[0]);
        assert_eq!(set.island_of(0), Some(0));
        assert_eq!(set.island_of(1), Some(0));
    }

    /// A chain 0-1-2-3 collapses transitively into a single island regardless of
    /// the order edges are supplied.
    #[test]
    fn chain_merges_transitively() {
        let edges = [Edge(2, 3), Edge(0, 1), Edge(1, 2)];
        let set = build_islands(&edges, &dynamic(4), 4).expect("valid graph");
        assert_eq!(set.len(), 1);
        assert_eq!(set.island(0).unwrap().particles(), &[0, 1, 2, 3]);
        // Constraint indices are attributed in ascending order.
        assert_eq!(set.island(0).unwrap().constraints(), &[0, 1, 2]);
    }

    /// Two stacks resting on a shared pinned floor stay independent islands: a
    /// static anchor is shared, never a bridge.
    #[test]
    fn shared_pinned_anchor_does_not_bridge() {
        // Particle 0 pinned (floor). Stack A = {1, 2}, stack B = {3, 4}, each
        // resting on the floor and internally linked.
        let mut inv = dynamic(5);
        inv[0] = 0.0;
        let edges = [Edge(0, 1), Edge(1, 2), Edge(0, 3), Edge(3, 4)];
        let set = build_islands(&edges, &inv, 5).expect("valid graph");

        assert_eq!(set.len(), 2);
        // Canonical ordering: island of least particle index first.
        assert_eq!(set.island(0).unwrap().particles(), &[1, 2]);
        assert_eq!(set.island(1).unwrap().particles(), &[3, 4]);
        // The pinned floor belongs to no island.
        assert_eq!(set.island_of(0), None);
        // Constraints touching the floor attach to their dynamic endpoint's
        // island (edge 0 -> island of particle 1, edge 2 -> island of 3).
        assert_eq!(set.island(0).unwrap().constraints(), &[0, 1]);
        assert_eq!(set.island(1).unwrap().constraints(), &[2, 3]);
    }

    /// Islands are ordered by their least particle index, and members within an
    /// island are sorted ascending, independent of edge order.
    #[test]
    fn output_is_canonical() {
        // Build the high-index island first to prove ordering is by content,
        // not by discovery order.
        let edges = [Edge(5, 4), Edge(1, 0)];
        let set = build_islands(&edges, &dynamic(6), 6).expect("valid graph");
        assert_eq!(set.len(), 2);
        assert_eq!(set.island(0).unwrap().particles(), &[0, 1]);
        assert_eq!(set.island(1).unwrap().particles(), &[4, 5]);
    }

    /// A constraint between two static particles moves nothing and is dropped.
    #[test]
    fn both_static_constraint_is_dropped() {
        let inv = vec![0.0, 0.0];
        let edges = [Edge(0, 1)];
        let set = build_islands(&edges, &inv, 2).expect("valid graph");
        assert!(set.is_empty());
        assert_eq!(set.island_of(0), None);
        assert_eq!(set.island_of(1), None);
    }

    /// A free dynamic particle that no constraint touches forms no island.
    #[test]
    fn untouched_dynamic_particle_forms_no_island() {
        // Particle 2 is dynamic but referenced by no edge.
        let edges = [Edge(0, 1)];
        let set = build_islands(&edges, &dynamic(3), 3).expect("valid graph");
        assert_eq!(set.len(), 1);
        assert_eq!(set.island_of(2), None);
    }

    /// An out-of-range endpoint is rejected with a precise error.
    #[test]
    fn out_of_range_endpoint_is_rejected() {
        let edges = [Edge(0, 5)];
        let err = build_islands(&edges, &dynamic(3), 3).expect_err("index 5 is out of range");
        match err {
            XpbdError::ConstraintOutOfRange {
                constraint,
                particle,
                particle_count,
            } => {
                assert_eq!(constraint, 0);
                assert_eq!(particle, 5);
                assert_eq!(particle_count, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// An inverse-mass array shorter than the particle count is rejected before
    /// any indexing can occur.
    #[test]
    fn short_inverse_mass_array_is_rejected() {
        let edges = [Edge(0, 1)];
        let err = build_islands(&edges, &dynamic(1), 2).expect_err("array too short");
        assert!(matches!(err, XpbdError::InvalidConfig(_)));
    }
}
