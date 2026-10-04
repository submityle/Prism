//! Distance-aware node selection (design §24.6).
//!
//! When work produced on one NUMA node needs to spill elsewhere — because the
//! home node is saturated — it should land on the *nearest* node by the
//! [`NumaDistanceMatrix`], not an arbitrary one, to keep remote-memory traffic
//! cheap. These pure functions implement that selection with a fully
//! deterministic tie-break so the choice never depends on thread timing.
//!
//! - [`rank_nodes_by_distance`] totally orders every node by distance from a
//!   home node (ties broken by ascending index).
//! - [`nearest_node`] picks the closest node from an explicit candidate set.
//! - [`choose_balanced_node`] adds load balancing: among nodes it minimizes
//!   `(distance, load, index)`, so the nearest node wins, ties go to the least
//!   loaded node, and remaining ties to the lowest index.

use alloc::vec::Vec;

use crate::numa::NumaNodeId;

use super::distance::NumaDistanceMatrix;

/// Every node ordered by distance from `from` (ascending), ties broken by
/// ascending node index. `from` itself ranks first (its self-distance is the
/// row minimum). Returns an empty vector if `from` is out of range.
pub fn rank_nodes_by_distance(matrix: &NumaDistanceMatrix, from: NumaNodeId) -> Vec<NumaNodeId> {
    let n = matrix.node_count();
    if (from.index() as usize) >= n {
        return Vec::new();
    }
    let mut nodes: Vec<NumaNodeId> = (0..n as u16).map(NumaNodeId::new).collect();
    nodes.sort_by_key(|&node| (matrix.distance(from, node), node.index()));
    nodes
}

/// The nearest node to `from` among `candidates`, by `(distance, index)`.
///
/// Candidates that are out of range for `matrix` are skipped. Returns `None`
/// if `from` is out of range or no in-range candidate remains.
pub fn nearest_node(
    matrix: &NumaDistanceMatrix,
    from: NumaNodeId,
    candidates: &[NumaNodeId],
) -> Option<NumaNodeId> {
    candidates
        .iter()
        .copied()
        .filter_map(|c| matrix.get(from, c).map(|d| (d, c.index(), c)))
        .min_by_key(|&(d, idx, _)| (d, idx))
        .map(|(_, _, c)| c)
}

/// Choose a target node for work homed on `home`, balancing distance against
/// current load.
///
/// `loads[i]` is the current load of node `i`; only the first
/// `min(loads.len(), node_count)` nodes are considered. The winner minimizes
/// `(distance(home, node), load, index)`: nearest first, then least loaded,
/// then lowest index — a total, deterministic order. Returns `None` if `home`
/// is out of range or there are no nodes to consider.
pub fn choose_balanced_node(
    matrix: &NumaDistanceMatrix,
    home: NumaNodeId,
    loads: &[u32],
) -> Option<NumaNodeId> {
    let n = matrix.node_count().min(loads.len());
    if n == 0 || (home.index() as usize) >= matrix.node_count() {
        return None;
    }
    (0..n as u16)
        .map(NumaNodeId::new)
        .map(|node| {
            let distance = matrix.distance(home, node);
            let load = loads[node.index() as usize];
            (distance, load, node.index(), node)
        })
        .min_by_key(|&(distance, load, idx, _)| (distance, load, idx))
        .map(|(_, _, _, node)| node)
}
