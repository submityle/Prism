//! Lightweight structural description of an audio processing graph.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The structures here
//! are a deliberately minimal, float-free mirror of the runtime graph defined in
//! `prism_audio_core::graph`, carrying only the information the execution planner
//! needs: per-node port counts and processing latency, directed port-to-port
//! edges, and a designated master output node.
//!
//! # Relationship
//!
//! [`GraphDesc`] is the single input consumed by every planning stage in this
//! crate (`topo`, `levels`, `islands`, `liveness`, `pdc`, `schedule`) and by the
//! assembled [`crate::exec_plan::ExecPlan`]. Keeping it dependency-free (integer
//! identifiers and counts only) makes the planner deterministic, trivially
//! testable, and reusable by a host that converts its own graph representation at
//! the boundary.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Stable identifier of a node inside a [`GraphDesc`].
///
/// The wrapped value is the node's insertion index, which doubles as its
/// position in the `nodes` vector. Deterministic tie-breaking throughout the
/// planner uses this ordering.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct NodeId(pub usize);

/// Structural description of a single processing node.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct NodeDesc {
    /// Number of input ports the node consumes.
    pub input_ports: u32,
    /// Number of output ports the node produces.
    pub output_ports: u32,
    /// Intrinsic processing latency of the node in frames (PDC input).
    pub latency_frames: u32,
    /// Whether the node may write an output into one of its input buffers,
    /// enabling slot aliasing when that input's last consumer is this node.
    pub can_process_in_place: bool,
}

impl NodeDesc {
    /// Creates a node description with the given port counts and zero latency,
    /// not in-place capable.
    #[must_use]
    pub const fn new(input_ports: u32, output_ports: u32) -> Self {
        Self {
            input_ports,
            output_ports,
            latency_frames: 0,
            can_process_in_place: false,
        }
    }

    /// Returns a copy with `latency_frames` set.
    #[must_use]
    pub const fn with_latency(mut self, latency_frames: u32) -> Self {
        self.latency_frames = latency_frames;
        self
    }

    /// Returns a copy with `can_process_in_place` set.
    #[must_use]
    pub const fn with_in_place(mut self, can_process_in_place: bool) -> Self {
        self.can_process_in_place = can_process_in_place;
        self
    }
}

/// Directed connection from one node's output port to another node's input port.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct EdgeDesc {
    /// Producing node.
    pub from_node: NodeId,
    /// Producing node's output port index.
    pub from_port: u32,
    /// Consuming node.
    pub to_node: NodeId,
    /// Consuming node's input port index.
    pub to_port: u32,
}

impl EdgeDesc {
    /// Creates an edge between two ports.
    #[must_use]
    pub const fn new(from_node: NodeId, from_port: u32, to_node: NodeId, to_port: u32) -> Self {
        Self {
            from_node,
            from_port,
            to_node,
            to_port,
        }
    }
}

/// Error raised while describing or planning a graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum ExecError {
    /// The graph contains a directed cycle and cannot be scheduled.
    Cycle,
    /// An edge or master reference names a node index that does not exist.
    UnknownNode(NodeId),
    /// An edge references a port index outside a node's declared port count.
    PortOutOfRange {
        /// Node whose port range was violated.
        node: NodeId,
        /// Offending port index.
        port: u32,
        /// Declared number of ports on that side of the node.
        port_count: u32,
    },
    /// No master output node was designated before planning.
    NoMaster,
}

/// Structural description of a processing graph consumed by the planner.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct GraphDesc {
    nodes: Vec<NodeDesc>,
    edges: Vec<EdgeDesc>,
    master: Option<NodeId>,
}

impl GraphDesc {
    /// Creates an empty graph description.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            master: None,
        }
    }

    /// Appends a node and returns its assigned [`NodeId`].
    pub fn add_node(&mut self, desc: NodeDesc) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(desc);
        id
    }

    /// Records a directed port-to-port connection.
    ///
    /// The edge is validated lazily by [`GraphDesc::validate`]; this method only
    /// stores it so that batch construction stays allocation-cheap.
    pub fn connect(&mut self, edge: EdgeDesc) {
        self.edges.push(edge);
    }

    /// Designates the node whose output is the final rendered mix.
    pub fn set_master(&mut self, node: NodeId) {
        self.master = Some(node);
    }

    /// Returns the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns the slice of node descriptions in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeDesc] {
        &self.nodes
    }

    /// Returns the slice of edges in insertion order.
    #[must_use]
    pub fn edges(&self) -> &[EdgeDesc] {
        &self.edges
    }

    /// Returns the designated master node, if any.
    #[must_use]
    pub fn master(&self) -> Option<NodeId> {
        self.master
    }

    /// Returns the description for `node`, or `None` if the id is out of range.
    #[must_use]
    pub fn node(&self, node: NodeId) -> Option<&NodeDesc> {
        self.nodes.get(node.0)
    }

    /// Validates node references, port ranges, and master designation.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::UnknownNode`] for any edge or master referencing a
    /// missing node, [`ExecError::PortOutOfRange`] for a port index beyond a
    /// node's declared count, and [`ExecError::NoMaster`] when no master is set.
    /// Cycle detection is performed separately by [`crate::topo::topological_order`].
    pub fn validate(&self) -> Result<(), ExecError> {
        let n = self.nodes.len();
        for edge in &self.edges {
            if edge.from_node.0 >= n {
                return Err(ExecError::UnknownNode(edge.from_node));
            }
            if edge.to_node.0 >= n {
                return Err(ExecError::UnknownNode(edge.to_node));
            }
            let from = &self.nodes[edge.from_node.0];
            if edge.from_port >= from.output_ports {
                return Err(ExecError::PortOutOfRange {
                    node: edge.from_node,
                    port: edge.from_port,
                    port_count: from.output_ports,
                });
            }
            let to = &self.nodes[edge.to_node.0];
            if edge.to_port >= to.input_ports {
                return Err(ExecError::PortOutOfRange {
                    node: edge.to_node,
                    port: edge.to_port,
                    port_count: to.input_ports,
                });
            }
        }
        match self.master {
            None => Err(ExecError::NoMaster),
            Some(m) if m.0 >= n => Err(ExecError::UnknownNode(m)),
            Some(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_node_assigns_sequential_ids() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        assert_eq!(a, NodeId(0));
        assert_eq!(b, NodeId(1));
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn validate_rejects_unknown_node() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        g.connect(EdgeDesc::new(a, 0, NodeId(9), 0));
        g.set_master(a);
        assert_eq!(g.validate(), Err(ExecError::UnknownNode(NodeId(9))));
    }

    #[test]
    fn validate_rejects_port_out_of_range() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 3, b, 0));
        g.set_master(b);
        assert_eq!(
            g.validate(),
            Err(ExecError::PortOutOfRange {
                node: a,
                port: 3,
                port_count: 1,
            })
        );
    }

    #[test]
    fn validate_requires_master() {
        let mut g = GraphDesc::new();
        g.add_node(NodeDesc::new(0, 1));
        assert_eq!(g.validate(), Err(ExecError::NoMaster));
    }

    #[test]
    fn validate_accepts_well_formed_graph() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.set_master(b);
        assert_eq!(g.validate(), Ok(()));
    }

    #[test]
    fn builder_helpers_set_fields() {
        let d = NodeDesc::new(1, 1).with_latency(64).with_in_place(true);
        assert_eq!(d.latency_frames, 64);
        assert!(d.can_process_in_place);
    }
}
