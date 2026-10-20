use std::any::type_name;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Monotonic identity for a graph node.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Constructs an ID from its serialized numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the serialized numeric representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "node-{}", self.0)
    }
}

/// Monotonic identity for a node port.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PortId(u64);

impl PortId {
    /// Constructs an ID from its serialized numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the serialized numeric representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for PortId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "port-{}", self.0)
    }
}

/// Monotonic identity for a connection edge.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct EdgeId(u64);

impl EdgeId {
    /// Constructs an ID from its serialized numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the serialized numeric representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "edge-{}", self.0)
    }
}

/// Monotonic identity for a hierarchical group.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct GroupId(u64);

impl GroupId {
    /// Constructs an ID from its serialized numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the serialized numeric representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for GroupId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "group-{}", self.0)
    }
}

/// A stable, caller-defined type key used to validate port compatibility.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PortType(String);

impl PortType {
    /// Creates a type key from a stable name such as `"prism.color.linear"`.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Creates a type key from a Rust type name.
    ///
    /// Persisted graphs should prefer [`Self::named`] so a Rust module rename
    /// does not change the graph's type key.
    #[must_use]
    pub fn of<T: 'static>() -> Self {
        Self(type_name::<T>().to_owned())
    }

    /// Returns the stable type key.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.0
    }
}

/// Direction of a node port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortDirection {
    /// A value consumed by a node.
    Input,
    /// A value produced by a node.
    Output,
}

/// Connection cardinality for one port.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConnectionPolicy {
    /// At most one edge may target the port.
    Single,
    /// Any number of edges may target the port.
    Multiple,
}

/// A declarative port description used while constructing a node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortSpec {
    name: String,
    direction: PortDirection,
    port_type: PortType,
    connection_policy: ConnectionPolicy,
}

impl PortSpec {
    /// Creates a single-connection input port.
    #[must_use]
    pub fn input(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            direction: PortDirection::Input,
            port_type,
            connection_policy: ConnectionPolicy::Single,
        }
    }

    /// Creates a multiple-connection output port.
    #[must_use]
    pub fn output(name: impl Into<String>, port_type: PortType) -> Self {
        Self {
            name: name.into(),
            direction: PortDirection::Output,
            port_type,
            connection_policy: ConnectionPolicy::Multiple,
        }
    }

    /// Changes the connection cardinality of this port.
    #[must_use]
    pub const fn with_connection_policy(mut self, policy: ConnectionPolicy) -> Self {
        self.connection_policy = policy;
        self
    }

    /// Returns the port name local to its node.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the port direction.
    #[must_use]
    pub const fn direction(&self) -> PortDirection {
        self.direction
    }

    /// Returns the port type key.
    #[must_use]
    pub const fn port_type(&self) -> &PortType {
        &self.port_type
    }

    /// Returns the connection cardinality.
    #[must_use]
    pub const fn connection_policy(&self) -> ConnectionPolicy {
        self.connection_policy
    }
}

/// A node description with caller-owned payload.
#[derive(Debug)]
pub struct NodeSpec<N = ()> {
    name: String,
    payload: N,
    ports: Vec<PortSpec>,
}

impl<N> NodeSpec<N> {
    /// Creates a node description with no ports.
    #[must_use]
    pub fn new(name: impl Into<String>, payload: N) -> Self {
        Self {
            name: name.into(),
            payload,
            ports: Vec::new(),
        }
    }

    /// Adds a port to the node description.
    #[must_use]
    pub fn with_port(mut self, port: PortSpec) -> Self {
        self.ports.push(port);
        self
    }

    /// Adds a port to the node description in place.
    pub fn add_port(&mut self, port: PortSpec) {
        self.ports.push(port);
    }

    /// Returns the declared node name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the declared ports in insertion order.
    #[must_use]
    pub fn ports(&self) -> &[PortSpec] {
        &self.ports
    }
}

/// A graph node stored after a [`NodeSpec`] is committed.
#[derive(Debug)]
pub struct Node<N = ()> {
    id: NodeId,
    name: String,
    payload: N,
    group: Option<GroupId>,
    ports: Vec<PortId>,
}

impl<N> Node<N> {
    /// Returns this node's stable ID.
    #[must_use]
    pub const fn id(&self) -> NodeId {
        self.id
    }

    /// Returns this node's display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the caller-owned payload.
    #[must_use]
    pub const fn payload(&self) -> &N {
        &self.payload
    }

    /// Returns a mutable reference to the caller-owned payload.
    pub fn payload_mut(&mut self) -> &mut N {
        &mut self.payload
    }

    /// Returns the containing group, if any.
    #[must_use]
    pub const fn group(&self) -> Option<GroupId> {
        self.group
    }

    /// Returns this node's ports in declaration order.
    #[must_use]
    pub fn ports(&self) -> &[PortId] {
        &self.ports
    }
}

/// A committed node port.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Port {
    id: PortId,
    node: NodeId,
    name: String,
    direction: PortDirection,
    port_type: PortType,
    connection_policy: ConnectionPolicy,
}

impl Port {
    /// Returns this port's stable ID.
    #[must_use]
    pub const fn id(&self) -> PortId {
        self.id
    }

    /// Returns the owning node.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// Returns the local port name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the port direction.
    #[must_use]
    pub const fn direction(&self) -> PortDirection {
        self.direction
    }

    /// Returns the port type key.
    #[must_use]
    pub const fn port_type(&self) -> &PortType {
        &self.port_type
    }

    /// Returns the connection cardinality.
    #[must_use]
    pub const fn connection_policy(&self) -> ConnectionPolicy {
        self.connection_policy
    }
}

/// A committed directed edge from an output port to an input port.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Edge {
    id: EdgeId,
    output: PortId,
    input: PortId,
}

impl Edge {
    /// Returns this edge's stable ID.
    #[must_use]
    pub const fn id(&self) -> EdgeId {
        self.id
    }

    /// Returns the source output port.
    #[must_use]
    pub const fn output(&self) -> PortId {
        self.output
    }

    /// Returns the destination input port.
    #[must_use]
    pub const fn input(&self) -> PortId {
        self.input
    }
}

/// A declarative hierarchical group description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupSpec {
    name: String,
    parent: Option<GroupId>,
}

impl GroupSpec {
    /// Creates a root group description.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
        }
    }

    /// Assigns the group to an existing parent group when committed.
    #[must_use]
    pub const fn with_parent(mut self, parent: GroupId) -> Self {
        self.parent = Some(parent);
        self
    }
}

/// A committed hierarchical group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Group {
    id: GroupId,
    name: String,
    parent: Option<GroupId>,
    nodes: BTreeSet<NodeId>,
    children: BTreeSet<GroupId>,
}

impl Group {
    /// Returns this group's stable ID.
    #[must_use]
    pub const fn id(&self) -> GroupId {
        self.id
    }

    /// Returns this group's display name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns this group's parent, if any.
    #[must_use]
    pub const fn parent(&self) -> Option<GroupId> {
        self.parent
    }

    /// Returns directly assigned nodes in stable ID order.
    #[must_use]
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().copied()
    }

    /// Returns direct child groups in stable ID order.
    #[must_use]
    pub fn children(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.children.iter().copied()
    }
}

/// The portion of a graph passed to one evaluation callback.
pub struct EvaluationNode<'a, N> {
    graph: &'a Graph<N>,
    id: NodeId,
}

impl<N> EvaluationNode<'_, N> {
    /// Returns the node being evaluated.
    #[must_use]
    pub const fn id(&self) -> NodeId {
        self.id
    }

    /// Returns the node metadata and payload.
    #[must_use]
    pub fn node(&self) -> &Node<N> {
        self.graph
            .node(self.id)
            .expect("evaluation order contains only live nodes")
    }

    /// Returns incoming edges in stable edge-ID order.
    #[must_use]
    pub fn input_edges(&self) -> Vec<&Edge> {
        self.graph.incoming_edges(self.id)
    }

    /// Returns outgoing edges in stable edge-ID order.
    #[must_use]
    pub fn output_edges(&self) -> Vec<&Edge> {
        self.graph.outgoing_edges(self.id)
    }
}

/// A cycle reported by topological validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CycleError {
    nodes: Vec<NodeId>,
}

impl CycleError {
    /// Creates a cycle from a non-empty deterministic node sequence.
    fn new(nodes: Vec<NodeId>) -> Self {
        Self { nodes }
    }

    /// Returns the nodes in one detected cycle, in deterministic traversal order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }
}

impl fmt::Display for CycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("graph contains a dependency cycle through ")?;
        for (index, node) in self.nodes.iter().enumerate() {
            if index != 0 {
                formatter.write_str(" -> ")?;
            }
            write!(formatter, "{node}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CycleError {}

/// Structural or validation failure from graph operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphError {
    /// A required graph name was empty.
    InvalidName { kind: NameKind, name: String },
    /// A port type key was empty.
    InvalidPortType { port: String },
    /// A node declared two ports with the same local name.
    DuplicatePortName { node: NodeId, name: String },
    /// A node ID was not found.
    MissingNode(NodeId),
    /// A port ID was not found.
    MissingPort(PortId),
    /// An edge ID was not found.
    MissingEdge(EdgeId),
    /// A group ID was not found.
    MissingGroup(GroupId),
    /// A connection used the wrong endpoint direction.
    WrongDirection {
        port: PortId,
        expected: PortDirection,
        actual: PortDirection,
    },
    /// The output and input port type keys differ.
    TypeMismatch {
        output: PortId,
        input: PortId,
        output_type: PortType,
        input_type: PortType,
    },
    /// A single-connection input already has an edge.
    InputAlreadyConnected(PortId),
    /// A proposed group parent would create a group cycle.
    GroupCycle { group: GroupId, parent: GroupId },
    /// A group cannot be removed while it owns nodes or child groups.
    GroupNotEmpty(GroupId),
    /// The graph's node dependency edges contain a cycle.
    Cycle(CycleError),
    /// A monotonic ID counter cannot allocate another ID.
    IdExhausted,
}

/// Name categories used in [`GraphError::InvalidName`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NameKind {
    /// A node name.
    Node,
    /// A port name.
    Port,
    /// A group name.
    Group,
}

impl fmt::Display for NameKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Node => "node",
            Self::Port => "port",
            Self::Group => "group",
        })
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName { kind, name } => {
                write!(formatter, "{kind} name cannot be empty: {name:?}")
            }
            Self::InvalidPortType { port } => {
                write!(formatter, "port {port:?} has an empty type key")
            }
            Self::DuplicatePortName { node, name } => {
                write!(formatter, "node {node} has duplicate port {name:?}")
            }
            Self::MissingNode(id) => write!(formatter, "{id} does not exist"),
            Self::MissingPort(id) => write!(formatter, "{id} does not exist"),
            Self::MissingEdge(id) => write!(formatter, "{id} does not exist"),
            Self::MissingGroup(id) => write!(formatter, "{id} does not exist"),
            Self::WrongDirection {
                port,
                expected,
                actual,
            } => {
                write!(formatter, "{port} is {actual:?}, expected {expected:?}")
            }
            Self::TypeMismatch {
                output,
                input,
                output_type,
                input_type,
            } => write!(
                formatter,
                "cannot connect {output} ({}) to {input} ({})",
                output_type.key(),
                input_type.key()
            ),
            Self::InputAlreadyConnected(port) => {
                write!(formatter, "input {port} already has a connection")
            }
            Self::GroupCycle { group, parent } => write!(
                formatter,
                "assigning {parent} below {group} would create a group cycle"
            ),
            Self::GroupNotEmpty(id) => write!(formatter, "{id} still owns nodes or child groups"),
            Self::Cycle(error) => error.fmt(formatter),
            Self::IdExhausted => formatter.write_str("graph ID space is exhausted"),
        }
    }
}

impl std::error::Error for GraphError {}

/// Failure while invoking a node evaluator.
#[derive(Debug)]
pub enum EvaluationError<E> {
    /// The graph could not produce a valid deterministic order.
    Graph(GraphError),
    /// The callback rejected one node.
    Node { node: NodeId, error: E },
}

impl<E: fmt::Display> fmt::Display for EvaluationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => error.fmt(formatter),
            Self::Node { node, error } => write!(formatter, "evaluation of {node} failed: {error}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for EvaluationError<E> {}

/// A deterministic node graph with typed ports, hierarchical groups, and
/// callback-driven topological evaluation.
#[derive(Debug)]
pub struct Graph<N = ()> {
    nodes: BTreeMap<NodeId, Node<N>>,
    ports: BTreeMap<PortId, Port>,
    port_names: BTreeMap<(NodeId, String), PortId>,
    edges: BTreeMap<EdgeId, Edge>,
    incoming_by_port: BTreeMap<PortId, BTreeSet<EdgeId>>,
    outgoing_by_port: BTreeMap<PortId, BTreeSet<EdgeId>>,
    outgoing_nodes: BTreeMap<NodeId, BTreeSet<NodeId>>,
    node_pair_counts: BTreeMap<(NodeId, NodeId), usize>,
    groups: BTreeMap<GroupId, Group>,
    next_node: u64,
    next_port: u64,
    next_edge: u64,
    next_group: u64,
}

impl<N> Default for Graph<N> {
    fn default() -> Self {
        Self {
            nodes: BTreeMap::new(),
            ports: BTreeMap::new(),
            port_names: BTreeMap::new(),
            edges: BTreeMap::new(),
            incoming_by_port: BTreeMap::new(),
            outgoing_by_port: BTreeMap::new(),
            outgoing_nodes: BTreeMap::new(),
            node_pair_counts: BTreeMap::new(),
            groups: BTreeMap::new(),
            next_node: 1,
            next_port: 1,
            next_edge: 1,
            next_group: 1,
        }
    }
}

impl<N> Graph<N> {
    /// Creates an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of live nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of live ports.
    #[must_use]
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Returns the number of live edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns the number of live groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns node IDs in ascending stable-ID order.
    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Returns a node by stable ID.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&Node<N>> {
        self.nodes.get(&id)
    }

    /// Returns a mutable node by stable ID.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node<N>> {
        self.nodes.get_mut(&id)
    }

    /// Returns a port by stable ID.
    #[must_use]
    pub fn port(&self, id: PortId) -> Option<&Port> {
        self.ports.get(&id)
    }

    /// Finds a port by its owning node and local name.
    #[must_use]
    pub fn port_named(&self, node: NodeId, name: &str) -> Option<PortId> {
        self.port_names.get(&(node, name.to_owned())).copied()
    }

    /// Returns an edge by stable ID.
    #[must_use]
    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(&id)
    }

    /// Returns edges in ascending stable-ID order.
    #[must_use]
    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.edges.values()
    }

    /// Returns a group by stable ID.
    #[must_use]
    pub fn group(&self, id: GroupId) -> Option<&Group> {
        self.groups.get(&id)
    }

    /// Adds a validated node and allocates all of its stable port IDs.
    pub fn add_node(&mut self, spec: NodeSpec<N>) -> Result<NodeId, GraphError> {
        self.validate_node_spec(&spec)?;
        self.ensure_capacity(self.next_node, 1)?;
        self.ensure_capacity(self.next_port, spec.ports.len() as u64)?;

        let node_id = NodeId(Self::allocate(&mut self.next_node)?);
        let mut port_ids = Vec::with_capacity(spec.ports.len());
        for port_spec in spec.ports {
            let port_id = PortId(Self::allocate(&mut self.next_port)?);
            self.port_names
                .insert((node_id, port_spec.name.clone()), port_id);
            self.ports.insert(
                port_id,
                Port {
                    id: port_id,
                    node: node_id,
                    name: port_spec.name,
                    direction: port_spec.direction,
                    port_type: port_spec.port_type,
                    connection_policy: port_spec.connection_policy,
                },
            );
            port_ids.push(port_id);
        }
        self.nodes.insert(
            node_id,
            Node {
                id: node_id,
                name: spec.name,
                payload: spec.payload,
                group: None,
                ports: port_ids,
            },
        );
        Ok(node_id)
    }

    /// Removes a node and all incident edges, returning its payload-bearing node.
    pub fn remove_node(&mut self, id: NodeId) -> Result<Node<N>, GraphError> {
        let port_ids = self
            .nodes
            .get(&id)
            .ok_or(GraphError::MissingNode(id))?
            .ports
            .clone();
        let group = self.nodes.get(&id).and_then(|node| node.group);
        let mut edge_ids = BTreeSet::new();
        for port_id in &port_ids {
            if let Some(edges) = self.incoming_by_port.get(port_id) {
                edge_ids.extend(edges.iter().copied());
            }
            if let Some(edges) = self.outgoing_by_port.get(port_id) {
                edge_ids.extend(edges.iter().copied());
            }
        }
        for edge_id in edge_ids {
            self.disconnect(edge_id)?;
        }
        for port_id in port_ids {
            let port = self
                .ports
                .remove(&port_id)
                .expect("node port index is consistent");
            self.port_names.remove(&(id, port.name));
            self.incoming_by_port.remove(&port_id);
            self.outgoing_by_port.remove(&port_id);
        }
        if let Some(group_id) = group {
            self.groups
                .get_mut(&group_id)
                .expect("node group index is consistent")
                .nodes
                .remove(&id);
        }
        Ok(self.nodes.remove(&id).expect("node existence was checked"))
    }

    /// Adds a root or nested group.
    pub fn add_group(&mut self, spec: GroupSpec) -> Result<GroupId, GraphError> {
        if spec.name.is_empty() {
            return Err(GraphError::InvalidName {
                kind: NameKind::Group,
                name: spec.name,
            });
        }
        if let Some(parent) = spec.parent {
            if !self.groups.contains_key(&parent) {
                return Err(GraphError::MissingGroup(parent));
            }
        }
        self.ensure_capacity(self.next_group, 1)?;
        let id = GroupId(Self::allocate(&mut self.next_group)?);
        if let Some(parent) = spec.parent {
            self.groups
                .get_mut(&parent)
                .expect("parent existence was checked")
                .children
                .insert(id);
        }
        self.groups.insert(
            id,
            Group {
                id,
                name: spec.name,
                parent: spec.parent,
                nodes: BTreeSet::new(),
                children: BTreeSet::new(),
            },
        );
        Ok(id)
    }

    /// Removes an empty group without reusing its ID.
    pub fn remove_group(&mut self, id: GroupId) -> Result<(), GraphError> {
        let group = self.groups.get(&id).ok_or(GraphError::MissingGroup(id))?;
        if !group.nodes.is_empty() || !group.children.is_empty() {
            return Err(GraphError::GroupNotEmpty(id));
        }
        let parent = group.parent;
        self.groups.remove(&id);
        if let Some(parent) = parent {
            self.groups
                .get_mut(&parent)
                .expect("parent existence is maintained")
                .children
                .remove(&id);
        }
        Ok(())
    }

    /// Changes a group's parent while rejecting hierarchy cycles.
    pub fn set_group_parent(
        &mut self,
        group: GroupId,
        parent: Option<GroupId>,
    ) -> Result<(), GraphError> {
        if !self.groups.contains_key(&group) {
            return Err(GraphError::MissingGroup(group));
        }
        if let Some(parent_id) = parent {
            if !self.groups.contains_key(&parent_id) {
                return Err(GraphError::MissingGroup(parent_id));
            }
            if parent_id == group || self.is_group_descendant(parent_id, group) {
                return Err(GraphError::GroupCycle {
                    group,
                    parent: parent_id,
                });
            }
        }
        let old_parent = self.groups.get(&group).and_then(|entry| entry.parent);
        if old_parent == parent {
            return Ok(());
        }
        if let Some(old_parent) = old_parent {
            self.groups
                .get_mut(&old_parent)
                .expect("old parent existence is maintained")
                .children
                .remove(&group);
        }
        if let Some(parent_id) = parent {
            self.groups
                .get_mut(&parent_id)
                .expect("new parent existence was checked")
                .children
                .insert(group);
        }
        self.groups
            .get_mut(&group)
            .expect("group existence was checked")
            .parent = parent;
        Ok(())
    }

    /// Assigns a node to a group or removes it from its current group.
    pub fn set_node_group(
        &mut self,
        node: NodeId,
        group: Option<GroupId>,
    ) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&node) {
            return Err(GraphError::MissingNode(node));
        }
        if let Some(group_id) = group {
            if !self.groups.contains_key(&group_id) {
                return Err(GraphError::MissingGroup(group_id));
            }
        }
        let old_group = self.nodes.get(&node).and_then(|entry| entry.group);
        if old_group == group {
            return Ok(());
        }
        if let Some(old_group) = old_group {
            self.groups
                .get_mut(&old_group)
                .expect("old group existence is maintained")
                .nodes
                .remove(&node);
        }
        if let Some(group_id) = group {
            self.groups
                .get_mut(&group_id)
                .expect("new group existence was checked")
                .nodes
                .insert(node);
        }
        self.nodes
            .get_mut(&node)
            .expect("node existence was checked")
            .group = group;
        Ok(())
    }

    /// Returns all nodes directly in a group, or recursively in its hierarchy.
    pub fn group_nodes(&self, group: GroupId, recursive: bool) -> Result<Vec<NodeId>, GraphError> {
        let entry = self
            .groups
            .get(&group)
            .ok_or(GraphError::MissingGroup(group))?;
        let mut result = BTreeSet::new();
        result.extend(entry.nodes.iter().copied());
        if recursive {
            let mut pending: Vec<GroupId> = entry.children.iter().copied().collect();
            while let Some(child) = pending.pop() {
                let child_group = self
                    .groups
                    .get(&child)
                    .expect("group child index is consistent");
                result.extend(child_group.nodes.iter().copied());
                pending.extend(child_group.children.iter().copied());
            }
        }
        Ok(result.into_iter().collect())
    }

    /// Connects an output port to an input port after validating direction,
    /// type, ownership, and input cardinality.
    pub fn connect(&mut self, output: PortId, input: PortId) -> Result<EdgeId, GraphError> {
        let output_port = self
            .ports
            .get(&output)
            .ok_or(GraphError::MissingPort(output))?;
        let input_port = self
            .ports
            .get(&input)
            .ok_or(GraphError::MissingPort(input))?;
        if output_port.direction != PortDirection::Output {
            return Err(GraphError::WrongDirection {
                port: output,
                expected: PortDirection::Output,
                actual: output_port.direction,
            });
        }
        if input_port.direction != PortDirection::Input {
            return Err(GraphError::WrongDirection {
                port: input,
                expected: PortDirection::Input,
                actual: input_port.direction,
            });
        }
        if output_port.port_type != input_port.port_type {
            return Err(GraphError::TypeMismatch {
                output,
                input,
                output_type: output_port.port_type.clone(),
                input_type: input_port.port_type.clone(),
            });
        }
        if input_port.connection_policy == ConnectionPolicy::Single
            && self
                .incoming_by_port
                .get(&input)
                .is_some_and(|edges| !edges.is_empty())
        {
            return Err(GraphError::InputAlreadyConnected(input));
        }
        self.ensure_capacity(self.next_edge, 1)?;
        let id = EdgeId(Self::allocate(&mut self.next_edge)?);
        let source = output_port.node;
        let target = input_port.node;
        self.edges.insert(id, Edge { id, output, input });
        self.outgoing_by_port.entry(output).or_default().insert(id);
        self.incoming_by_port.entry(input).or_default().insert(id);
        self.outgoing_nodes
            .entry(source)
            .or_default()
            .insert(target);
        *self.node_pair_counts.entry((source, target)).or_default() += 1;
        Ok(id)
    }

    /// Disconnects an edge and returns its stable edge record.
    pub fn disconnect(&mut self, id: EdgeId) -> Result<Edge, GraphError> {
        let edge = self.edges.remove(&id).ok_or(GraphError::MissingEdge(id))?;
        if let Some(edges) = self.outgoing_by_port.get_mut(&edge.output) {
            edges.remove(&id);
            if edges.is_empty() {
                self.outgoing_by_port.remove(&edge.output);
            }
        }
        if let Some(edges) = self.incoming_by_port.get_mut(&edge.input) {
            edges.remove(&id);
            if edges.is_empty() {
                self.incoming_by_port.remove(&edge.input);
            }
        }
        let source = self
            .ports
            .get(&edge.output)
            .expect("edge output is live")
            .node;
        let target = self
            .ports
            .get(&edge.input)
            .expect("edge input is live")
            .node;
        let pair = (source, target);
        let count = self
            .node_pair_counts
            .get_mut(&pair)
            .expect("edge pair index is consistent");
        *count -= 1;
        if *count == 0 {
            self.node_pair_counts.remove(&pair);
            if let Some(targets) = self.outgoing_nodes.get_mut(&source) {
                targets.remove(&target);
                if targets.is_empty() {
                    self.outgoing_nodes.remove(&source);
                }
            }
        }
        Ok(edge)
    }

    /// Returns incoming edges for a node in stable edge-ID order.
    #[must_use]
    pub fn incoming_edges(&self, node: NodeId) -> Vec<&Edge> {
        self.edges_for_node(node, true)
    }

    /// Returns outgoing edges for a node in stable edge-ID order.
    #[must_use]
    pub fn outgoing_edges(&self, node: NodeId) -> Vec<&Edge> {
        self.edges_for_node(node, false)
    }

    /// Computes a deterministic dependency-first order for every live node.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, GraphError> {
        let mut indegree: BTreeMap<NodeId, usize> =
            self.nodes.keys().copied().map(|id| (id, 0)).collect();
        for ((source, target), count) in &self.node_pair_counts {
            if source != target {
                *indegree.get_mut(target).expect("edge target is live") += count;
            } else {
                *indegree.get_mut(target).expect("edge target is live") += count;
            }
        }
        let mut ready: BTreeSet<NodeId> = indegree
            .iter()
            .filter_map(|(&node, &degree)| (degree == 0).then_some(node))
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(node) = ready.pop_first() {
            order.push(node);
            if let Some(targets) = self.outgoing_nodes.get(&node) {
                for &target in targets {
                    let degree = indegree.get_mut(&target).expect("edge target is live");
                    let count = self
                        .node_pair_counts
                        .get(&(node, target))
                        .copied()
                        .expect("node pair index is consistent");
                    *degree -= count;
                    if *degree == 0 {
                        ready.insert(target);
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            let residual: BTreeSet<NodeId> = indegree
                .into_iter()
                .filter_map(|(node, degree)| (degree != 0).then_some(node))
                .collect();
            return Err(GraphError::Cycle(CycleError::new(
                self.find_cycle(&residual),
            )));
        }
        Ok(order)
    }

    /// Validates graph structure and requires an acyclic node dependency graph.
    pub fn validate(&self) -> Result<(), GraphError> {
        self.validate_indexes()?;
        self.topological_order().map(|_| ())
    }

    /// Evaluates every node in deterministic topological order.
    pub fn evaluate<F, E>(&self, mut callback: F) -> Result<Vec<NodeId>, EvaluationError<E>>
    where
        F: FnMut(EvaluationNode<'_, N>) -> Result<(), E>,
    {
        let order = self.topological_order().map_err(EvaluationError::Graph)?;
        for &node in &order {
            callback(EvaluationNode {
                graph: self,
                id: node,
            })
            .map_err(|error| EvaluationError::Node { node, error })?;
        }
        Ok(order)
    }

    fn validate_node_spec(&self, spec: &NodeSpec<N>) -> Result<(), GraphError> {
        if spec.name.is_empty() {
            return Err(GraphError::InvalidName {
                kind: NameKind::Node,
                name: spec.name.clone(),
            });
        }
        let mut names = BTreeSet::new();
        for port in &spec.ports {
            if port.name.is_empty() {
                return Err(GraphError::InvalidName {
                    kind: NameKind::Port,
                    name: port.name.clone(),
                });
            }
            if !names.insert(port.name.clone()) {
                return Err(GraphError::DuplicatePortName {
                    node: NodeId(0),
                    name: port.name.clone(),
                });
            }
            if port.port_type.key().is_empty() {
                return Err(GraphError::InvalidPortType {
                    port: port.name.clone(),
                });
            }
        }
        Ok(())
    }

    fn validate_indexes(&self) -> Result<(), GraphError> {
        for (id, node) in &self.nodes {
            if node.id != *id {
                return Err(GraphError::MissingNode(*id));
            }
            for port_id in &node.ports {
                let port = self
                    .ports
                    .get(port_id)
                    .ok_or(GraphError::MissingPort(*port_id))?;
                if port.node != *id {
                    return Err(GraphError::MissingPort(*port_id));
                }
            }
            if let Some(group) = node.group {
                if !self
                    .groups
                    .get(&group)
                    .is_some_and(|entry| entry.nodes.contains(id))
                {
                    return Err(GraphError::MissingGroup(group));
                }
            }
        }
        for edge in self.edges.values() {
            let output = self
                .ports
                .get(&edge.output)
                .ok_or(GraphError::MissingPort(edge.output))?;
            let input = self
                .ports
                .get(&edge.input)
                .ok_or(GraphError::MissingPort(edge.input))?;
            if output.direction != PortDirection::Output || input.direction != PortDirection::Input
            {
                return Err(GraphError::WrongDirection {
                    port: edge.output,
                    expected: PortDirection::Output,
                    actual: output.direction,
                });
            }
            if output.port_type != input.port_type {
                return Err(GraphError::TypeMismatch {
                    output: edge.output,
                    input: edge.input,
                    output_type: output.port_type.clone(),
                    input_type: input.port_type.clone(),
                });
            }
        }
        Ok(())
    }

    fn edges_for_node(&self, node: NodeId, incoming: bool) -> Vec<&Edge> {
        let Some(entry) = self.nodes.get(&node) else {
            return Vec::new();
        };
        let mut ids = BTreeSet::new();
        for port in &entry.ports {
            let edges = if incoming {
                self.incoming_by_port.get(port)
            } else {
                self.outgoing_by_port.get(port)
            };
            if let Some(edges) = edges {
                ids.extend(edges.iter().copied());
            }
        }
        ids.into_iter()
            .filter_map(|id| self.edges.get(&id))
            .collect()
    }

    fn is_group_descendant(&self, candidate: GroupId, ancestor: GroupId) -> bool {
        let mut current = Some(candidate);
        while let Some(group) = current {
            if group == ancestor {
                return true;
            }
            current = self.groups.get(&group).and_then(|entry| entry.parent);
        }
        false
    }

    fn find_cycle(&self, residual: &BTreeSet<NodeId>) -> Vec<NodeId> {
        let mut color = BTreeMap::<NodeId, u8>::new();
        let mut parent = BTreeMap::<NodeId, NodeId>::new();
        for &start in residual {
            if color.get(&start).copied().unwrap_or(0) != 0 {
                continue;
            }
            color.insert(start, 1);
            let mut stack = vec![(start, self.residual_targets(start, residual), 0_usize)];
            while let Some((node, targets, next)) = stack.last_mut() {
                if *next == targets.len() {
                    color.insert(*node, 2);
                    stack.pop();
                    continue;
                }
                let target = targets[*next];
                *next += 1;
                match color.get(&target).copied().unwrap_or(0) {
                    0 => {
                        parent.insert(target, *node);
                        color.insert(target, 1);
                        stack.push((target, self.residual_targets(target, residual), 0));
                    }
                    1 => {
                        let mut cycle = vec![target];
                        let mut cursor = *node;
                        while cursor != target {
                            cycle.push(cursor);
                            cursor = parent[&cursor];
                        }
                        cycle.reverse();
                        return cycle;
                    }
                    _ => {}
                }
            }
        }
        residual.iter().copied().collect()
    }

    fn residual_targets(&self, node: NodeId, residual: &BTreeSet<NodeId>) -> Vec<NodeId> {
        self.outgoing_nodes
            .get(&node)
            .into_iter()
            .flat_map(|targets| targets.iter().copied())
            .filter(|target| residual.contains(target))
            .collect()
    }

    fn ensure_capacity(&self, next: u64, amount: u64) -> Result<(), GraphError> {
        next.checked_add(amount)
            .ok_or(GraphError::IdExhausted)
            .map(|_| ())
    }

    fn allocate(counter: &mut u64) -> Result<u64, GraphError> {
        let id = *counter;
        *counter = counter.checked_add(1).ok_or(GraphError::IdExhausted)?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number_type() -> PortType {
        PortType::named("test.number")
    }

    fn node(name: &str, input: bool, output: bool) -> NodeSpec<&'static str> {
        let mut spec = NodeSpec::new(name, "payload");
        if input {
            spec.add_port(PortSpec::input("in", number_type()));
        }
        if output {
            spec.add_port(PortSpec::output("out", number_type()));
        }
        spec
    }

    #[test]
    fn ids_are_monotonic_and_are_not_reused() {
        let mut graph = Graph::new();
        let first = graph.add_node(node("first", false, true)).expect("node");
        let second = graph.add_node(node("second", true, true)).expect("node");
        let third = graph.add_node(node("third", true, false)).expect("node");
        let first_out = graph.port_named(first, "out").expect("port");
        let second_in = graph.port_named(second, "in").expect("port");
        let second_out = graph.port_named(second, "out").expect("port");
        let third_in = graph.port_named(third, "in").expect("port");
        let edge_a = graph.connect(first_out, second_in).expect("edge");
        let edge_b = graph.connect(second_out, third_in).expect("edge");
        assert_eq!(first.raw(), 1);
        assert_eq!(edge_a.raw(), 1);
        graph.disconnect(edge_a).expect("disconnect");
        graph.remove_node(second).expect("remove");
        let replacement = graph
            .add_node(node("replacement", false, false))
            .expect("node");
        let edge_c = graph.connect(first_out, third_in);
        assert_eq!(replacement.raw(), 4);
        assert!(edge_c.is_ok());
        assert!(edge_b.raw() < edge_c.expect("edge").raw());
    }

    #[test]
    fn typed_connections_validate_direction_type_and_cardinality() {
        let mut graph = Graph::new();
        let source = graph.add_node(node("source", false, true)).expect("node");
        let target = graph.add_node(node("target", true, false)).expect("node");
        let output = graph.port_named(source, "out").expect("port");
        let input = graph.port_named(target, "in").expect("port");
        graph.connect(output, input).expect("connect");
        assert_eq!(
            graph.connect(output, input),
            Err(GraphError::InputAlreadyConnected(input))
        );
        assert_eq!(
            graph.connect(input, output),
            Err(GraphError::WrongDirection {
                port: input,
                expected: PortDirection::Output,
                actual: PortDirection::Input,
            })
        );
        let other = graph
            .add_node(NodeSpec::new("other", "payload"))
            .expect("node");
        let other_input = graph
            .add_node(NodeSpec::new("other-input", "payload"))
            .expect("node");
        let other_output = graph.node(other).expect("node").id();
        assert_eq!(other_output, other);
        let invalid = graph.port_named(other_input, "missing");
        assert!(invalid.is_none());
    }

    #[test]
    fn topological_order_is_stable_and_cycles_are_explicit() {
        let mut graph = Graph::new();
        let a = graph.add_node(node("a", true, true)).expect("node");
        let b = graph.add_node(node("b", true, true)).expect("node");
        let c = graph.add_node(node("c", true, true)).expect("node");
        let a_out = graph.port_named(a, "out").expect("port");
        let b_in = graph.port_named(b, "in").expect("port");
        let b_out = graph.port_named(b, "out").expect("port");
        let c_in = graph.port_named(c, "in").expect("port");
        graph.connect(a_out, b_in).expect("edge");
        graph.connect(b_out, c_in).expect("edge");
        assert_eq!(graph.topological_order().expect("acyclic"), vec![a, b, c]);
        let c_out_port = graph.port_named(c, "out").expect("port");
        let a_input_port = graph.port_named(a, "in").expect("port");
        graph.connect(c_out_port, a_input_port).expect("edge");
        let error = graph.topological_order().expect_err("cycle");
        match error {
            GraphError::Cycle(cycle) => assert!(!cycle.nodes().is_empty()),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn evaluation_uses_topological_order_and_exposes_edges() {
        let mut graph = Graph::new();
        let source = graph.add_node(node("source", false, true)).expect("node");
        let middle = graph.add_node(node("middle", true, true)).expect("node");
        let sink = graph.add_node(node("sink", true, false)).expect("node");
        let source_out = graph.port_named(source, "out").expect("port");
        let middle_in = graph.port_named(middle, "in").expect("port");
        let middle_out = graph.port_named(middle, "out").expect("port");
        let sink_in = graph.port_named(sink, "in").expect("port");
        graph.connect(source_out, middle_in).expect("edge");
        graph.connect(middle_out, sink_in).expect("edge");
        let mut visited = Vec::new();
        let order = graph
            .evaluate(|entry| {
                if entry.id() == middle {
                    assert_eq!(entry.input_edges().len(), 1);
                    assert_eq!(entry.output_edges().len(), 1);
                }
                visited.push(entry.id());
                Ok::<(), &'static str>(())
            })
            .expect("evaluation");
        assert_eq!(visited, vec![source, middle, sink]);
        assert_eq!(order, visited);
    }

    #[test]
    fn groups_are_hierarchical_and_reject_cycles() {
        let mut graph = Graph::new();
        let root = graph.add_group(GroupSpec::new("root")).expect("group");
        let child = graph
            .add_group(GroupSpec::new("child").with_parent(root))
            .expect("group");
        let node_id = graph.add_node(node("member", false, false)).expect("node");
        graph
            .set_node_group(node_id, Some(child))
            .expect("membership");
        assert_eq!(
            graph.group_nodes(root, true).expect("members"),
            vec![node_id]
        );
        assert_eq!(
            graph.set_group_parent(root, Some(child)),
            Err(GraphError::GroupCycle {
                group: root,
                parent: child
            })
        );
        assert_eq!(
            graph.remove_group(root),
            Err(GraphError::GroupNotEmpty(root))
        );
        graph.set_node_group(node_id, None).expect("unassign");
        graph.remove_group(child).expect("remove");
        graph.remove_group(root).expect("remove");
    }

    #[test]
    fn invalid_specs_and_type_mismatch_are_rejected() {
        let mut graph = Graph::new();
        assert!(matches!(
            graph.add_node(NodeSpec::new("", ())),
            Err(GraphError::InvalidName {
                kind: NameKind::Node,
                ..
            })
        ));
        assert!(matches!(
            graph.add_node(
                NodeSpec::new("duplicate", ())
                    .with_port(PortSpec::input("same", number_type()))
                    .with_port(PortSpec::output("same", number_type()))
            ),
            Err(GraphError::DuplicatePortName { .. })
        ));
        let source = graph
            .add_node(
                NodeSpec::new("source", ())
                    .with_port(PortSpec::output("out", PortType::named("test.number"))),
            )
            .expect("node");
        let target = graph
            .add_node(
                NodeSpec::new("target", ())
                    .with_port(PortSpec::input("in", PortType::named("test.text"))),
            )
            .expect("node");
        let error = graph
            .connect(
                graph.port_named(source, "out").expect("port"),
                graph.port_named(target, "in").expect("port"),
            )
            .expect_err("type mismatch");
        assert!(matches!(error, GraphError::TypeMismatch { .. }));
    }
}
