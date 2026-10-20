//! Deterministic, type-aware node graph kernel for editor and authoring tools.
//!
//! The kernel owns graph structure and execution order only. Values, widgets,
//! serialization formats, and project-specific node semantics remain outside
//! this crate. Nodes, ports, edges, and groups use monotonic opaque IDs: an ID
//! is never reused after removal, so references remain unambiguous for the
//! lifetime of the graph.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

mod graph;

pub use graph::{
    ConnectionPolicy, CycleError, Edge, EvaluationError, EvaluationNode, Graph, GraphError, Group,
    GroupId, GroupSpec, NameKind, Node, NodeId, NodeSpec, Port, PortDirection, PortId, PortSpec,
    PortType,
};
