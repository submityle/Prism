//! [`GraphModel`] — the node editor's pure data model.
//!
//! This module is deliberately render-free: it describes *what* a node graph
//! contains (nodes, ports and the edges between them) without producing any
//! [`Element`](prism_ui::Element). It is `no_std` (`alloc` only) so the model
//! can live in headless tooling, serialization and tests without a runtime.
//!
//! The visual controls ([`super::node`], [`super::port`], [`super::edge`],
//! [`super::node_canvas`]) consume data shaped like this model but never depend
//! on it directly; keeping the model separate lets layout and painting evolve
//! independently of the data representation.

use alloc::string::String;
use alloc::vec::Vec;

/// A stable identifier for a node within a [`GraphModel`].
///
/// Ids are opaque: the model never interprets the wrapped integer, it only
/// compares ids for equality, so callers may allocate them however they like.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct NodeId(pub u64);

/// Identifies a single port on a node.
///
/// A port is addressed by its owning [`NodeId`] plus its `index` within that
/// node's input *or* output list. Whether the index refers to an input or an
/// output is implied by context (the `from` side of an [`Edge`] is an output,
/// the `to` side is an input).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortId {
    /// The node this port belongs to.
    pub node: NodeId,
    /// The port's position within its node's input or output list.
    pub index: u32,
}

impl PortId {
    /// Creates a port reference for `node` at `index`.
    #[must_use]
    pub const fn new(node: NodeId, index: u32) -> Self {
        Self { node, index }
    }
}

/// A directed connection from an output port (`from`) to an input port (`to`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Edge {
    /// The source output port.
    pub from: PortId,
    /// The destination input port.
    pub to: PortId,
}

impl Edge {
    /// Creates an edge from an output port to an input port.
    #[must_use]
    pub const fn new(from: PortId, to: PortId) -> Self {
        Self { from, to }
    }

    /// Returns `true` if either endpoint of this edge is on `node`.
    #[must_use]
    pub fn touches(&self, node: NodeId) -> bool {
        self.from.node == node || self.to.node == node
    }
}

/// A single node: identity, a display title, a canvas position and the labelled
/// input/output ports it exposes.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NodeData {
    /// The node's stable id.
    pub id: NodeId,
    /// Human-readable title shown in the node header.
    pub title: String,
    /// Canvas position as `(x, y)` in logical pixels.
    pub pos: (f32, f32),
    /// Labels for the node's input ports, in order.
    pub inputs: Vec<String>,
    /// Labels for the node's output ports, in order.
    pub outputs: Vec<String>,
}

impl NodeData {
    /// Creates a node with the given id and title at the origin, with no ports.
    #[must_use]
    pub fn new(id: NodeId, title: impl Into<String>) -> Self {
        Self {
            id,
            title: title.into(),
            pos: (0.0, 0.0),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Sets the canvas position.
    #[must_use]
    pub fn pos(mut self, x: f32, y: f32) -> Self {
        self.pos = (x, y);
        self
    }

    /// Appends an input port label.
    #[must_use]
    pub fn input(mut self, label: impl Into<String>) -> Self {
        self.inputs.push(label.into());
        self
    }

    /// Appends an output port label.
    #[must_use]
    pub fn output(mut self, label: impl Into<String>) -> Self {
        self.outputs.push(label.into());
        self
    }

    /// Replaces the input port labels.
    #[must_use]
    pub fn inputs<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inputs = labels.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces the output port labels.
    #[must_use]
    pub fn outputs<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.outputs = labels.into_iter().map(Into::into).collect();
        self
    }
}

/// The whole node graph: a flat list of nodes plus the edges between their
/// ports.
///
/// Mutating methods keep the two lists consistent — removing a node drops every
/// edge that touched it — but the model stays intentionally small; richer
/// invariants (acyclicity, type checking) belong to higher layers.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct GraphModel {
    /// All nodes in the graph.
    pub nodes: Vec<NodeData>,
    /// All edges in the graph.
    pub edges: Vec<Edge>,
}

impl GraphModel {
    /// Creates an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a node. Does not deduplicate ids; callers own id allocation.
    pub fn add_node(&mut self, node: NodeData) {
        self.nodes.push(node);
    }

    /// Removes the node with `id` (if present) along with every edge that
    /// touched it. Returns `true` if a node was removed.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != id);
        let removed = self.nodes.len() != before;
        if removed {
            self.edges.retain(|e| !e.touches(id));
        }
        removed
    }

    /// Looks up a node by id.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&NodeData> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Returns `true` if a port reference is in range for its node and side.
    ///
    /// `output` selects which list the index is validated against: the output
    /// list when `true`, the input list otherwise.
    #[must_use]
    fn port_valid(&self, port: PortId, output: bool) -> bool {
        match self.node(port.node) {
            Some(node) => {
                let len = if output {
                    node.outputs.len()
                } else {
                    node.inputs.len()
                };
                (port.index as usize) < len
            }
            None => false,
        }
    }

    /// Connects an output port to an input port.
    ///
    /// Returns `false` (adding nothing) when either endpoint is out of range or
    /// an identical edge already exists; otherwise appends the edge and returns
    /// `true`.
    pub fn connect(&mut self, from: PortId, to: PortId) -> bool {
        if !self.port_valid(from, true) || !self.port_valid(to, false) {
            return false;
        }
        let edge = Edge::new(from, to);
        if self.edges.contains(&edge) {
            return false;
        }
        self.edges.push(edge);
        true
    }

    /// Removes an edge equal to `edge`. Returns `true` if one was removed.
    pub fn disconnect(&mut self, edge: Edge) -> bool {
        let before = self.edges.len();
        self.edges.retain(|e| *e != edge);
        self.edges.len() != before
    }

    /// Collects every edge with an endpoint on `node`, preserving order.
    #[must_use]
    pub fn edges_of(&self, node: NodeId) -> Vec<Edge> {
        self.edges.iter().copied().filter(|e| e.touches(node)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> GraphModel {
        let mut g = GraphModel::new();
        g.add_node(
            NodeData::new(NodeId(1), "Source")
                .pos(10.0, 20.0)
                .output("out"),
        );
        g.add_node(
            NodeData::new(NodeId(2), "Sink")
                .input("a")
                .input("b"),
        );
        g
    }

    #[test]
    fn add_and_lookup_nodes() {
        let g = graph();
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.node(NodeId(1)).map(|n| n.title.as_str()), Some("Source"));
        assert!(g.node(NodeId(99)).is_none());
    }

    #[test]
    fn connect_validates_ports_and_dedups() {
        let mut g = graph();
        let from = PortId::new(NodeId(1), 0);
        let to = PortId::new(NodeId(2), 1);
        assert!(g.connect(from, to));
        assert_eq!(g.edges.len(), 1);
        // Duplicate is rejected.
        assert!(!g.connect(from, to));
        // Out-of-range input index is rejected.
        assert!(!g.connect(from, PortId::new(NodeId(2), 5)));
        // Unknown node is rejected.
        assert!(!g.connect(from, PortId::new(NodeId(42), 0)));
        assert_eq!(g.edges.len(), 1);
    }

    #[test]
    fn edges_of_lists_incident_edges() {
        let mut g = graph();
        g.connect(PortId::new(NodeId(1), 0), PortId::new(NodeId(2), 0));
        let incident = g.edges_of(NodeId(2));
        assert_eq!(incident.len(), 1);
        assert_eq!(incident[0].to, PortId::new(NodeId(2), 0));
        assert!(g.edges_of(NodeId(99)).is_empty());
    }

    #[test]
    fn remove_node_cascades_edges() {
        let mut g = graph();
        g.connect(PortId::new(NodeId(1), 0), PortId::new(NodeId(2), 0));
        assert!(g.remove_node(NodeId(1)));
        assert_eq!(g.nodes.len(), 1);
        assert!(g.edges.is_empty(), "edges touching a removed node are dropped");
        assert!(!g.remove_node(NodeId(1)), "second removal is a no-op");
    }

    #[test]
    fn disconnect_removes_matching_edge() {
        let mut g = graph();
        let edge = Edge::new(PortId::new(NodeId(1), 0), PortId::new(NodeId(2), 0));
        g.connect(edge.from, edge.to);
        assert!(g.disconnect(edge));
        assert!(g.edges.is_empty());
        assert!(!g.disconnect(edge));
    }
}
