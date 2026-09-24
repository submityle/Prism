//! Simulation islands via union-find.
//!
//! An *island* is a maximal set of bodies connected (directly or transitively)
//! by constraints or contacts; islands can be solved independently and in
//! parallel. This module provides a real weighted-union / path-compression
//! union-find ([`IslandBuilder`]) and the resulting partition ([`IslandSet`]).
//!
//! This is a standard disjoint-set data structure and is not derived from
//! Unreal Engine source.

/// Identifies a single island within an [`IslandSet`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IslandId(pub u32);

/// Incremental union-find over a fixed number of elements.
///
/// Elements are `0..count`. Call [`IslandBuilder::union`] to connect pairs,
/// then [`IslandBuilder::build`] to produce an [`IslandSet`]. Uses union by
/// rank and path compression for near-constant amortized operations.
#[derive(Clone, Debug)]
pub struct IslandBuilder {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl IslandBuilder {
    /// Creates a builder over `count` initially disconnected elements.
    #[must_use]
    pub fn new(count: usize) -> IslandBuilder {
        IslandBuilder {
            parent: (0..count).collect(),
            rank: vec![0; count],
        }
    }

    /// Returns the number of elements this builder tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Returns `true` if the builder tracks no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Finds the representative (root) of element `a`, applying path
    /// compression.
    ///
    /// # Panics
    ///
    /// Panics if `a` is out of range (`a >= self.len()`).
    pub fn find(&mut self, a: usize) -> usize {
        let mut root = a;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression: point every node on the path directly at the root.
        let mut node = a;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    /// Unions the sets containing `a` and `b`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of range.
    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Union by rank: attach the shallower tree under the deeper one.
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }

    /// Freezes the current partition into an [`IslandSet`].
    ///
    /// Island ids are assigned in order of the first element that belongs to
    /// each island, so results are deterministic.
    pub fn build(&mut self) -> IslandSet {
        let count = self.parent.len();
        let mut island_of: Vec<IslandId> = Vec::with_capacity(count);
        let mut members: Vec<Vec<usize>> = Vec::new();
        let mut root_to_island: Vec<Option<u32>> = vec![None; count];

        for i in 0..count {
            let root = self.find(i);
            let island = match root_to_island[root] {
                Some(id) => id,
                None => {
                    let id = members.len() as u32;
                    root_to_island[root] = Some(id);
                    members.push(Vec::new());
                    id
                }
            };
            island_of.push(IslandId(island));
            members[island as usize].push(i);
        }

        IslandSet { island_of, members }
    }
}

/// A frozen partition of elements into islands.
#[derive(Clone, Debug, Default)]
pub struct IslandSet {
    island_of: Vec<IslandId>,
    members: Vec<Vec<usize>>,
}

impl IslandSet {
    /// Returns the island element `i` belongs to.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range.
    #[must_use]
    pub fn island_of(&self, i: usize) -> IslandId {
        self.island_of[i]
    }

    /// Returns the number of islands.
    #[must_use]
    pub fn island_count(&self) -> usize {
        self.members.len()
    }

    /// Returns the members of `island`, sorted ascending by element index.
    ///
    /// Returns an empty slice if the island id is out of range.
    #[must_use]
    pub fn members(&self, island: IslandId) -> &[usize] {
        self.members
            .get(island.0 as usize)
            .map_or(&[], Vec::as_slice)
    }

    /// Returns the total number of elements across all islands.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.island_of.len()
    }

    /// Returns `true` if there are no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.island_of.is_empty()
    }
}

/// Builds islands directly from a set of connectivity `pairs` over `count`
/// elements.
///
/// This is a convenience wrapper around [`IslandBuilder`]: it unions every pair
/// and returns the resulting [`IslandSet`]. Elements not mentioned in any pair
/// form singleton islands.
#[must_use]
pub fn islands_from_pairs(count: usize, pairs: &[(usize, usize)]) -> IslandSet {
    let mut builder = IslandBuilder::new(count);
    for &(a, b) in pairs {
        builder.union(a, b);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_example_partition() {
        // 6 elements, pairs {(0,1),(2,3),(3,4)} -> {0,1}, {2,3,4}, {5}.
        let set = islands_from_pairs(6, &[(0, 1), (2, 3), (3, 4)]);
        assert_eq!(set.island_count(), 3);
        assert_eq!(set.element_count(), 6);

        // 0 and 1 share an island.
        assert_eq!(set.island_of(0), set.island_of(1));
        // 2, 3, 4 share an island.
        assert_eq!(set.island_of(2), set.island_of(3));
        assert_eq!(set.island_of(3), set.island_of(4));
        // 5 is alone.
        assert_ne!(set.island_of(5), set.island_of(0));
        assert_ne!(set.island_of(5), set.island_of(2));

        // Deterministic ids by first appearance.
        assert_eq!(set.island_of(0), IslandId(0));
        assert_eq!(set.island_of(2), IslandId(1));
        assert_eq!(set.island_of(5), IslandId(2));

        assert_eq!(set.members(IslandId(0)), &[0, 1]);
        assert_eq!(set.members(IslandId(1)), &[2, 3, 4]);
        assert_eq!(set.members(IslandId(2)), &[5]);
        assert_eq!(set.members(IslandId(99)), &[] as &[usize]);
    }

    #[test]
    fn find_and_union_are_transitive() {
        let mut b = IslandBuilder::new(5);
        assert_eq!(b.len(), 5);
        b.union(0, 1);
        b.union(1, 2);
        assert_eq!(b.find(0), b.find(2));
        assert_ne!(b.find(0), b.find(3));
    }

    #[test]
    fn no_pairs_yields_all_singletons() {
        let set = islands_from_pairs(4, &[]);
        assert_eq!(set.island_count(), 4);
        for i in 0..4 {
            assert_eq!(set.members(set.island_of(i)), &[i]);
        }
    }

    #[test]
    fn empty_builder_is_empty() {
        let set = islands_from_pairs(0, &[]);
        assert!(set.is_empty());
        assert_eq!(set.island_count(), 0);
    }
}
