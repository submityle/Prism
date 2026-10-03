//! Read-only inspection of a compiled audio processing graph.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the graph-inspection part of design section 26. The exec graph
//! compiler (design section 20) exports its node list and edges into a
//! [`GraphInspection`] so tooling can draw the signal flow, read per-node
//! reported latency, and compute the critical-path latency without touching
//! the RT structures. The compiler guarantees an acyclic graph; the queries
//! here nonetheless detect cycles defensively and report them rather than
//! looping.

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Metadata for one node in the compiled graph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GraphNodeInfo {
    /// Stable identifier of the node within the compiled graph.
    pub id: u64,
    /// Human-readable name for display.
    pub name: String,
    /// Node category label (for example `"reverb"` or `"mixer"`).
    pub kind: String,
    /// Channel count the node processes.
    pub channels: usize,
    /// Processing latency the node introduces, in frames.
    pub latency_frames: u32,
    /// `true` when the node is currently bypassed.
    pub bypassed: bool,
}

impl GraphNodeInfo {
    /// Create node metadata with no reported latency and not bypassed.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>, kind: impl Into<String>, channels: usize) -> Self {
        Self {
            id,
            name: name.into(),
            kind: kind.into(),
            channels,
            latency_frames: 0,
            bypassed: false,
        }
    }

    /// Builder-style setter for the reported latency, in frames.
    #[must_use]
    pub fn with_latency(mut self, latency_frames: u32) -> Self {
        self.latency_frames = latency_frames;
        self
    }
}

/// A directed edge from one node's output port to another's input port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GraphConnection {
    /// Source node identifier.
    pub from_node: u64,
    /// Source output port index.
    pub from_port: u32,
    /// Destination node identifier.
    pub to_node: u64,
    /// Destination input port index.
    pub to_port: u32,
}

impl GraphConnection {
    /// Create a connection between two node ports.
    #[must_use]
    #[inline]
    pub const fn new(from_node: u64, from_port: u32, to_node: u64, to_port: u32) -> Self {
        Self {
            from_node,
            from_port,
            to_node,
            to_port,
        }
    }
}

/// A read-only view of a compiled graph: its nodes and their connections.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GraphInspection {
    nodes: Vec<GraphNodeInfo>,
    connections: Vec<GraphConnection>,
}

impl GraphInspection {
    /// Create an empty inspection.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
        }
    }

    /// Add a node to the inspection.
    pub fn add_node(&mut self, node: GraphNodeInfo) {
        self.nodes.push(node);
    }

    /// Add a connection to the inspection.
    pub fn connect(&mut self, connection: GraphConnection) {
        self.connections.push(connection);
    }

    /// Number of nodes.
    #[must_use]
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of connections.
    #[must_use]
    #[inline]
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Iterate the nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &GraphNodeInfo> {
        self.nodes.iter()
    }

    /// Iterate the connections.
    pub fn connections(&self) -> impl Iterator<Item = &GraphConnection> {
        self.connections.iter()
    }

    /// Find a node by identifier.
    #[must_use]
    pub fn node(&self, id: u64) -> Option<&GraphNodeInfo> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// Compute a topological ordering of node identifiers via Kahn's algorithm,
    /// or `None` when the graph contains a cycle or references a node that was
    /// not added.
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<u64>> {
        let mut indegree: BTreeMap<u64, usize> = BTreeMap::new();
        for node in &self.nodes {
            indegree.insert(node.id, 0);
        }
        for connection in &self.connections {
            if !indegree.contains_key(&connection.from_node)
                || !indegree.contains_key(&connection.to_node)
            {
                return None;
            }
            if let Some(count) = indegree.get_mut(&connection.to_node) {
                *count += 1;
            }
        }

        let mut ready: VecDeque<u64> = indegree
            .iter()
            .filter_map(|(id, degree)| if *degree == 0 { Some(*id) } else { None })
            .collect();
        let mut order: Vec<u64> = Vec::with_capacity(self.nodes.len());

        while let Some(id) = ready.pop_front() {
            order.push(id);
            for connection in &self.connections {
                if connection.from_node != id {
                    continue;
                }
                if let Some(count) = indegree.get_mut(&connection.to_node) {
                    *count -= 1;
                    if *count == 0 {
                        ready.push_back(connection.to_node);
                    }
                }
            }
        }

        if order.len() == self.nodes.len() {
            Some(order)
        } else {
            None
        }
    }

    /// `true` when the graph has no cycles and every edge references a known
    /// node.
    #[must_use]
    pub fn is_acyclic(&self) -> bool {
        self.topological_order().is_some()
    }

    /// Longest cumulative latency, in frames, along any path through the graph.
    ///
    /// Returns `None` when the graph is not acyclic. An empty graph has a
    /// critical-path latency of `0`.
    #[must_use]
    pub fn critical_path_latency(&self) -> Option<u32> {
        let order = self.topological_order()?;
        let mut best: BTreeMap<u64, u32> = BTreeMap::new();
        for node in &self.nodes {
            best.insert(node.id, 0);
        }

        let mut peak: u32 = 0;
        for id in order {
            let incoming = best.get(&id).copied().unwrap_or(0);
            let own = self.node(id).map(|node| node.latency_frames).unwrap_or(0);
            let through = incoming.saturating_add(own);
            if through > peak {
                peak = through;
            }
            for connection in &self.connections {
                if connection.from_node != id {
                    continue;
                }
                if let Some(downstream) = best.get_mut(&connection.to_node)
                    && through > *downstream
                {
                    *downstream = through;
                }
            }
        }
        Some(peak)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> GraphInspection {
        let mut graph = GraphInspection::new();
        graph.add_node(GraphNodeInfo::new(1, "src", "source", 2).with_latency(0));
        graph.add_node(GraphNodeInfo::new(2, "verb", "reverb", 2).with_latency(64));
        graph.add_node(GraphNodeInfo::new(3, "out", "mixer", 2).with_latency(16));
        graph.connect(GraphConnection::new(1, 0, 2, 0));
        graph.connect(GraphConnection::new(2, 0, 3, 0));
        graph
    }

    #[test]
    fn empty_graph_is_acyclic_with_zero_latency() {
        let graph = GraphInspection::new();
        assert!(graph.is_acyclic());
        assert_eq!(graph.critical_path_latency(), Some(0));
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn chain_topological_order_is_source_first() {
        let graph = chain();
        let order = graph.topological_order().expect("acyclic");
        assert_eq!(order, alloc::vec![1, 2, 3]);
        assert_eq!(graph.connection_count(), 2);
    }

    #[test]
    fn critical_path_sums_latency_along_chain() {
        let graph = chain();
        assert_eq!(graph.critical_path_latency(), Some(80));
    }

    #[test]
    fn diamond_takes_longest_branch() {
        let mut graph = GraphInspection::new();
        graph.add_node(GraphNodeInfo::new(1, "src", "source", 2));
        graph.add_node(GraphNodeInfo::new(2, "short", "gain", 2).with_latency(8));
        graph.add_node(GraphNodeInfo::new(3, "long", "reverb", 2).with_latency(128));
        graph.add_node(GraphNodeInfo::new(4, "out", "mixer", 2).with_latency(4));
        graph.connect(GraphConnection::new(1, 0, 2, 0));
        graph.connect(GraphConnection::new(1, 0, 3, 0));
        graph.connect(GraphConnection::new(2, 0, 4, 0));
        graph.connect(GraphConnection::new(3, 0, 4, 1));
        assert_eq!(graph.critical_path_latency(), Some(132));
    }

    #[test]
    fn cycle_detected() {
        let mut graph = GraphInspection::new();
        graph.add_node(GraphNodeInfo::new(1, "a", "gain", 2));
        graph.add_node(GraphNodeInfo::new(2, "b", "gain", 2));
        graph.connect(GraphConnection::new(1, 0, 2, 0));
        graph.connect(GraphConnection::new(2, 0, 1, 0));
        assert!(!graph.is_acyclic());
        assert_eq!(graph.topological_order(), None);
        assert_eq!(graph.critical_path_latency(), None);
    }

    #[test]
    fn edge_to_unknown_node_is_rejected() {
        let mut graph = GraphInspection::new();
        graph.add_node(GraphNodeInfo::new(1, "a", "gain", 2));
        graph.connect(GraphConnection::new(1, 0, 99, 0));
        assert_eq!(graph.topological_order(), None);
    }

    #[test]
    fn node_lookup_works() {
        let graph = chain();
        assert_eq!(graph.node(2).map(|n| n.kind.as_str()), Some("reverb"));
        assert_eq!(graph.node(42), None);
    }
}
