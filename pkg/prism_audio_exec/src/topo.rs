//! Deterministic topological ordering with cycle detection.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. This is a textbook
//! Kahn in-degree algorithm with a strict deterministic tie-break (lowest node
//! index first), implemented over the integer [`GraphDesc`].
//!
//! # Relationship
//!
//! The order produced here is the serial execution order and the position space
//! that `levels`, `liveness`, and `pdc` build on. Determinism matters because the
//! whole engine promises bit-identical renders across runs and across the serial
//! and parallel schedulers (`schedule`).

use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::cmp::Reverse;

use crate::graph_desc::{ExecError, GraphDesc, NodeId};

/// Computes a deterministic topological order of the graph's nodes.
///
/// Nodes with no remaining incoming edges are emitted in ascending index order,
/// which makes the result a pure function of the graph structure regardless of
/// edge insertion order.
///
/// # Errors
///
/// Returns [`ExecError::Cycle`] if the graph is not a DAG, and the validation
/// errors from [`GraphDesc::validate`] if node or port references are invalid.
pub fn topological_order(desc: &GraphDesc) -> Result<Vec<NodeId>, ExecError> {
    desc.validate()?;

    let n = desc.node_count();
    let mut indegree = vec![0_usize; n];
    for edge in desc.edges() {
        indegree[edge.to_node.0] += 1;
    }

    // Min-heap over node index for deterministic emission.
    let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    for (i, &deg) in indegree.iter().enumerate() {
        if deg == 0 {
            ready.push(Reverse(i));
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(Reverse(node)) = ready.pop() {
        order.push(NodeId(node));
        for edge in desc.edges() {
            if edge.from_node.0 == node {
                let d = &mut indegree[edge.to_node.0];
                *d -= 1;
                if *d == 0 {
                    ready.push(Reverse(edge.to_node.0));
                }
            }
        }
    }

    if order.len() == n {
        Ok(order)
    } else {
        Err(ExecError::Cycle)
    }
}

/// Returns, for each node index, its position in the topological order.
///
/// This is the inverse permutation of [`topological_order`] and is the integer
/// "time axis" used by buffer liveness and delay compensation.
#[must_use]
pub fn position_map(order: &[NodeId], node_count: usize) -> Vec<usize> {
    let mut pos = vec![0_usize; node_count];
    for (i, &nid) in order.iter().enumerate() {
        pos[nid.0] = i;
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_desc::{EdgeDesc, NodeDesc};

    fn chain() -> GraphDesc {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, c, 0));
        g.set_master(c);
        g
    }

    #[test]
    fn orders_a_chain() {
        let g = chain();
        let order = topological_order(&g).unwrap();
        assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn tie_break_is_ascending_index() {
        // Two independent sources feed one sink; sources must come out in index
        // order regardless of edge order.
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(0, 1));
        let sink = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(b, 0, sink, 1));
        g.connect(EdgeDesc::new(a, 0, sink, 0));
        g.set_master(sink);
        let order = topological_order(&g).unwrap();
        assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn detects_cycle() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(1, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, a, 0));
        g.set_master(b);
        assert_eq!(topological_order(&g), Err(ExecError::Cycle));
    }

    #[test]
    fn position_map_is_inverse() {
        let g = chain();
        let order = topological_order(&g).unwrap();
        let pos = position_map(&order, g.node_count());
        assert_eq!(pos, vec![0, 1, 2]);
    }
}
