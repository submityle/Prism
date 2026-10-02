//! Longest-path depth layering (wavefront levels).
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The level of a node
//! is the length of the longest directed path reaching it, computed in a single
//! pass over the topological order.
//!
//! # Relationship
//!
//! Nodes sharing a level have no path between them and are therefore safe to run
//! concurrently; `schedule` turns these levels into parallel wavefronts, and the
//! level count is the critical-path length that bounds achievable parallel
//! speed-up.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::graph_desc::GraphDesc;
use crate::graph_desc::NodeId;

/// Per-node longest-path depth plus derived parallelism metrics.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LevelAssignment {
    level: Vec<u32>,
    depth: u32,
}

impl LevelAssignment {
    /// Returns the level (longest-path depth) of every node, indexed by
    /// [`NodeId`] value.
    #[must_use]
    pub fn levels(&self) -> &[u32] {
        &self.level
    }

    /// Returns the level of a single node.
    #[must_use]
    pub fn level_of(&self, node: NodeId) -> u32 {
        self.level[node.0]
    }

    /// Returns the number of distinct levels, i.e. the critical-path length in
    /// nodes. A single wavefront graph has depth `1`; an empty graph `0`.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.depth
    }
}

/// Assigns each node its longest-path depth from any source node.
///
/// `order` must be a valid topological order of `desc` (see
/// [`crate::topo::topological_order`]); the computation relies on every
/// predecessor being visited before its successors.
#[must_use]
pub fn assign_levels(desc: &GraphDesc, order: &[NodeId]) -> LevelAssignment {
    let n = desc.node_count();
    let mut level = vec![0_u32; n];
    for &node in order {
        for edge in desc.edges() {
            if edge.from_node == node {
                let candidate = level[node.0] + 1;
                if candidate > level[edge.to_node.0] {
                    level[edge.to_node.0] = candidate;
                }
            }
        }
    }
    let depth = if n == 0 {
        0
    } else {
        level.iter().copied().max().unwrap_or(0) + 1
    };
    LevelAssignment { level, depth }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_desc::{EdgeDesc, NodeDesc};
    use crate::topo::topological_order;

    #[test]
    fn chain_levels_increment() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, c, 0));
        g.set_master(c);
        let order = topological_order(&g).unwrap();
        let levels = assign_levels(&g, &order);
        assert_eq!(levels.levels(), &[0, 1, 2]);
        assert_eq!(levels.depth(), 3);
    }

    #[test]
    fn diamond_uses_longest_path() {
        // a -> b -> d, a -> c -> d, plus a -> d directly.
        // d must sit at level 2 (longest path a-b-d), not level 1.
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(1, 1));
        let d = g.add_node(NodeDesc::new(3, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(a, 0, c, 0));
        g.connect(EdgeDesc::new(b, 0, d, 0));
        g.connect(EdgeDesc::new(c, 0, d, 1));
        g.connect(EdgeDesc::new(a, 0, d, 2));
        g.set_master(d);
        let order = topological_order(&g).unwrap();
        let levels = assign_levels(&g, &order);
        assert_eq!(levels.level_of(a), 0);
        assert_eq!(levels.level_of(b), 1);
        assert_eq!(levels.level_of(c), 1);
        assert_eq!(levels.level_of(d), 2);
        assert_eq!(levels.depth(), 3);
    }

    #[test]
    fn independent_nodes_share_level_zero() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(0, 1));
        let sink = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(a, 0, sink, 0));
        g.connect(EdgeDesc::new(b, 0, sink, 1));
        g.set_master(sink);
        let order = topological_order(&g).unwrap();
        let levels = assign_levels(&g, &order);
        assert_eq!(levels.level_of(a), 0);
        assert_eq!(levels.level_of(b), 0);
        assert_eq!(levels.level_of(sink), 1);
    }
}
