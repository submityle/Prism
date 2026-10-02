//! Weakly-connected component ("island") partitioning via union-find.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. A disjoint-set
//! forest with path compression and union by rank groups nodes that are
//! reachable from one another when edges are treated as undirected.
//!
//! # Relationship
//!
//! Independent islands never share a buffer and can be rendered on separate
//! worker threads with no synchronization between them; the parallel scheduler
//! (`schedule`) and the CPU budget governor use this partition to distribute and
//! bound work. Island membership is also a fast rejection test for the liveness
//! allocator, which only ever reuses a slot within the island that produced it.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::graph_desc::{GraphDesc, NodeId};

/// Partition of a graph into weakly-connected components.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IslandPartition {
    island_of: Vec<usize>,
    members: Vec<Vec<NodeId>>,
}

impl IslandPartition {
    /// Returns the island index that `node` belongs to.
    #[must_use]
    pub fn island_of(&self, node: NodeId) -> usize {
        self.island_of[node.0]
    }

    /// Returns the number of islands.
    #[must_use]
    pub fn island_count(&self) -> usize {
        self.members.len()
    }

    /// Returns the member nodes of each island; members within an island are in
    /// ascending index order and islands are ordered by their lowest member.
    #[must_use]
    pub fn islands(&self) -> &[Vec<NodeId>] {
        &self.members
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u32>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Partitions `desc` into weakly-connected islands.
///
/// Isolated nodes (no edges) each form their own island. The result is
/// deterministic: islands are keyed by their lowest-index member and emitted in
/// ascending order of that key.
#[must_use]
pub fn partition_islands(desc: &GraphDesc) -> IslandPartition {
    let n = desc.node_count();
    let mut ds = DisjointSet::new(n);
    for edge in desc.edges() {
        ds.union(edge.from_node.0, edge.to_node.0);
    }

    // Canonical root per node, then compact roots into dense island indices in
    // ascending order of the smallest member they contain.
    let mut root = vec![0_usize; n];
    for (i, slot) in root.iter_mut().enumerate() {
        *slot = ds.find(i);
    }

    let mut island_index = vec![usize::MAX; n];
    let mut members: Vec<Vec<NodeId>> = Vec::new();
    for (i, &r) in root.iter().enumerate() {
        if island_index[r] == usize::MAX {
            island_index[r] = members.len();
            members.push(Vec::new());
        }
        members[island_index[r]].push(NodeId(i));
    }

    let island_of = root.iter().map(|&r| island_index[r]).collect();
    IslandPartition {
        island_of,
        members,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_desc::{EdgeDesc, NodeDesc};

    #[test]
    fn single_chain_is_one_island() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.set_master(b);
        let part = partition_islands(&g);
        assert_eq!(part.island_count(), 1);
        assert_eq!(part.island_of(a), part.island_of(b));
    }

    #[test]
    fn disjoint_subgraphs_are_separate_islands() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(0, 1));
        let d = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(c, 0, d, 0));
        g.set_master(b);
        let part = partition_islands(&g);
        assert_eq!(part.island_count(), 2);
        assert_ne!(part.island_of(a), part.island_of(c));
        assert_eq!(part.island_of(a), part.island_of(b));
        assert_eq!(part.island_of(c), part.island_of(d));
        assert_eq!(part.islands()[0], vec![NodeId(0), NodeId(1)]);
        assert_eq!(part.islands()[1], vec![NodeId(2), NodeId(3)]);
    }

    #[test]
    fn isolated_node_is_its_own_island() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(0, 1));
        g.set_master(a);
        let part = partition_islands(&g);
        assert_eq!(part.island_count(), 2);
        assert_ne!(part.island_of(a), part.island_of(b));
    }
}
