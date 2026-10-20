//! Atomic authoring over the generic kernel. Inverses own only touched records.
use super::*;
use alloc::collections::VecDeque;

mod codec;

/// Integer document coordinates, independent of UI scale or floating point.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GraphPoint {
    /// Horizontal document coordinate; absolute value at most one million.
    pub x: i32,
    /// Vertical document coordinate; absolute value at most one million.
    pub y: i32,
}
/// Explicit identity and description for a newly created port.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphPortDraft {
    /// Nonzero, monotonically increasing identity.
    pub id: PortId,
    /// Host-contributed type, direction, cardinality and local name.
    pub spec: PortSpec,
}
/// One host-contributed node. Payload interpretation belongs to its host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphNodeDraft {
    /// Nonzero identity, never reused after a successful edit.
    pub id: NodeId,
    /// Display name.
    pub name: String,
    /// Engine-neutral opaque data.
    pub payload: Vec<u8>,
    /// Ports in stable presentation order, with increasing explicit IDs.
    pub ports: Vec<GraphPortDraft>,
    /// Document position.
    pub position: GraphPoint,
}
/// One mutation inside an atomic command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphEdit {
    /// Insert a node and its ports.
    CreateNode(GraphNodeDraft),
    /// Remove a node and all incident connections.
    DeleteNode(NodeId),
    /// Replace the document position.
    MoveNode {
        /// Existing node identity.
        node: NodeId,
        /// New document position.
        position: GraphPoint,
    },
    /// Replace opaque host data.
    SetPayload {
        /// Existing node identity.
        node: NodeId,
        /// Replacement opaque bytes.
        payload: Vec<u8>,
    },
    /// Insert a typed directed edge.
    Connect {
        /// Fresh edge identity.
        edge: EdgeId,
        /// Typed source port.
        output: PortId,
        /// Typed destination port.
        input: PortId,
    },
    /// Remove an edge.
    Disconnect(EdgeId),
}
/// Deterministically serializable edit/history operation, replayable from empty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphCommand {
    /// All edits succeed together; final topology must be acyclic.
    Edit(Vec<GraphEdit>),
    /// Restore the most recent retained edit.
    Undo,
    /// Reapply the most recently undone edit.
    Redo,
}
/// Configurable bounds, each also restricted by a hard implementation ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphAuthoringLimits {
    /// Maximum live nodes, at most 16384.
    pub nodes: usize,
    /// Maximum live edges, at most 65536.
    pub edges: usize,
    /// Maximum ports on one node, at most 64.
    pub ports_per_node: usize,
    /// Maximum UTF-8 bytes per name/type key, at most 4096.
    pub text_bytes: usize,
    /// Maximum opaque bytes per node, at most 1 MiB.
    pub payload_bytes: usize,
    /// Maximum encoded command bytes, at most 16 MiB.
    pub command_bytes: usize,
    /// Maximum edits per batch, at most 1024.
    pub edits: usize,
    /// Maximum retained entries across undo and redo, at most 1024.
    pub history_entries: usize,
    /// Conservative retained-history charge, at most 64 MiB.
    pub history_bytes: usize,
}
impl Default for GraphAuthoringLimits {
    fn default() -> Self {
        Self {
            nodes: 4096,
            edges: 8192,
            ports_per_node: 32,
            text_bytes: 256,
            payload_bytes: 65536,
            command_bytes: 1048576,
            edits: 128,
            history_entries: 128,
            history_bytes: 8388608,
        }
    }
}
impl GraphAuthoringLimits {
    /// Validate nonzero limits and hard ceilings before accepting untrusted data.
    pub fn validate(self) -> Result<(), GraphAuthoringError> {
        for (n, max) in [
            (self.nodes, 16384),
            (self.edges, 65536),
            (self.ports_per_node, 64),
            (self.text_bytes, 4096),
            (self.payload_bytes, 1048576),
            (self.command_bytes, 16777216),
            (self.edits, 1024),
            (self.history_entries, 1024),
            (self.history_bytes, 67108864),
        ] {
            if n == 0 || n > max {
                return Err(GraphAuthoringError::Limit);
            }
        }
        Ok(())
    }
}
/// Rejected input leaves graph, IDs, revision and history unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphAuthoringError {
    /// Invalid configured bounds, coordinates, counts or byte budget.
    Limit,
    /// Zero, exhausted, reused or unordered explicit identity.
    Identity,
    /// No retained history in the requested direction.
    EmptyHistory,
    /// Invalid version, tag, UTF-8, truncation or trailing bytes.
    Codec,
    /// Kernel validation failure, including type/cardinality/topology.
    Graph(GraphError),
}
impl fmt::Display for GraphAuthoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GraphAuthoringError {}
impl From<GraphError> for GraphAuthoringError {
    fn from(e: GraphError) -> Self {
        Self::Graph(e)
    }
}

#[derive(Debug)]
struct Removed {
    node: Node<Vec<u8>>,
    ports: Vec<Port>,
    edges: Vec<Edge>,
    position: GraphPoint,
}
#[derive(Debug)]
enum Inverse {
    Delete(NodeId),
    Restore(Removed),
    Position(NodeId, GraphPoint),
    Payload(NodeId, Vec<u8>),
    Disconnect(EdgeId),
    Connect(Edge),
}
impl Inverse {
    fn charge(&self) -> usize {
        // Slack covers enum/vector/map entry overhead in either direction.
        512 + match self {
            Self::Restore(r) => {
                r.node.name.len()
                    + r.node.payload.len()
                    + r.ports
                        .iter()
                        .map(|p| 512 + p.name.len() + p.port_type.key().len())
                        .sum::<usize>()
                    + r.edges.len() * 512
            }
            Self::Payload(_, p) => p.len(),
            _ => 0,
        }
    }
}
#[derive(Debug)]
struct Entry {
    inverse: Vec<Inverse>,
    charge: usize,
}
/// Bounded authoring state. Mutable kernel access is deliberately not exposed.
///
/// Each batch journals only touched records. History reserves both forward and
/// reverse data, so undo cannot grow beyond its admission budget. Eviction is
/// oldest-first. Allocation failure/panics are outside the atomic error contract.
#[derive(Debug)]
pub struct GraphEditor {
    graph: Graph<Vec<u8>>,
    positions: BTreeMap<NodeId, GraphPoint>,
    limits: GraphAuthoringLimits,
    undo: VecDeque<Entry>,
    redo: Vec<Entry>,
    history_bytes: usize,
    revision: u64,
}
impl GraphEditor {
    /// Construct an empty editor with validated resource bounds.
    pub fn new(limits: GraphAuthoringLimits) -> Result<Self, GraphAuthoringError> {
        limits.validate()?;
        Ok(Self {
            graph: Graph::new(),
            positions: BTreeMap::new(),
            limits,
            undo: VecDeque::new(),
            redo: Vec::new(),
            history_bytes: 0,
            revision: 0,
        })
    }
    /// Read typed structure or evaluate using host semantics.
    pub fn graph(&self) -> &Graph<Vec<u8>> {
        &self.graph
    }
    /// Configured bounds.
    pub const fn limits(&self) -> GraphAuthoringLimits {
        self.limits
    }
    /// Monotonic successful command revision, including undo and redo.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Position of an existing node.
    pub fn position(&self, id: NodeId) -> Option<GraphPoint> {
        self.positions.get(&id).copied()
    }
    /// Ordered positions without allocating or copying graph records.
    pub fn positions(&self) -> impl Iterator<Item = (NodeId, GraphPoint)> + '_ {
        self.positions.iter().map(|(id, p)| (*id, *p))
    }
    /// Allocation high-water marks; undo never rewinds them.
    pub fn next_ids(&self) -> (NodeId, PortId, EdgeId) {
        (
            NodeId(self.graph.next_node),
            PortId(self.graph.next_port),
            EdgeId(self.graph.next_edge),
        )
    }
    /// Number of retained undo commands.
    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }
    /// Number of retained redo commands.
    pub fn redo_depth(&self) -> usize {
        self.redo.len()
    }
    /// Conservative admitted bytes for both stacks together.
    pub const fn history_bytes(&self) -> usize {
        self.history_bytes
    }
    /// Execute one atomic command. Failed commands do not truncate redo history.
    pub fn execute(&mut self, cmd: &GraphCommand) -> Result<(), GraphAuthoringError> {
        let encoded = cmd.encode(self.limits)?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(GraphAuthoringError::Limit)?;
        match cmd {
            GraphCommand::Undo => {
                let e = self
                    .undo
                    .pop_back()
                    .ok_or(GraphAuthoringError::EmptyHistory)?;
                let inverse = self.reverse(e.inverse);
                self.redo.push(Entry {
                    inverse,
                    charge: e.charge,
                });
            }
            GraphCommand::Redo => {
                let e = self.redo.pop().ok_or(GraphAuthoringError::EmptyHistory)?;
                let inverse = self.reverse(e.inverse);
                self.undo.push_back(Entry {
                    inverse,
                    charge: e.charge,
                });
            }
            GraphCommand::Edit(edits) => {
                let counters = (
                    self.graph.next_node,
                    self.graph.next_port,
                    self.graph.next_edge,
                );
                let mut inverse = Vec::new();
                let result = (|| {
                    let mut charge = encoded.len() * 4;
                    for edit in edits {
                        inverse.push(self.edit(edit)?);
                        charge += inverse.last().expect("just pushed").charge() * 2;
                        if let GraphEdit::CreateNode(n) = edit {
                            charge += 1024 + n.ports.len() * 1024;
                        }
                        if charge > self.limits.history_bytes {
                            return Err(GraphAuthoringError::Limit);
                        }
                        if self.graph.node_count() > self.limits.nodes
                            || self.graph.edge_count() > self.limits.edges
                        {
                            return Err(GraphAuthoringError::Limit);
                        }
                    }
                    self.graph.validate()?;
                    Ok(charge)
                })();
                let charge = match result {
                    Ok(c) => c,
                    Err(e) => {
                        self.reverse(inverse);
                        (
                            self.graph.next_node,
                            self.graph.next_port,
                            self.graph.next_edge,
                        ) = counters;
                        return Err(e);
                    }
                };
                for entry in self.redo.drain(..) {
                    self.history_bytes -= entry.charge;
                }
                while self.undo.len() >= self.limits.history_entries
                    || self.history_bytes + charge > self.limits.history_bytes
                {
                    self.history_bytes -= self
                        .undo
                        .pop_front()
                        .expect("admitted entry fits empty history")
                        .charge;
                }
                self.history_bytes += charge;
                self.undo.push_back(Entry { inverse, charge });
            }
        }
        self.revision = revision;
        Ok(())
    }
    fn edit(&mut self, edit: &GraphEdit) -> Result<Inverse, GraphAuthoringError> {
        match edit {
            GraphEdit::CreateNode(d) => {
                id(d.id.0, self.graph.next_node)?;
                let mut next = self.graph.next_port;
                for p in &d.ports {
                    id(p.id.0, next)?;
                    next = p.id.0 + 1;
                }
                let spec = NodeSpec {
                    name: d.name.clone(),
                    payload: Vec::new(),
                    ports: d.ports.iter().map(|p| p.spec.clone()).collect(),
                };
                self.graph.validate_node_spec(&spec)?;
                let mut ports = Vec::new();
                for p in &d.ports {
                    let s = &p.spec;
                    self.graph.port_names.insert((d.id, s.name.clone()), p.id);
                    self.graph.ports.insert(
                        p.id,
                        Port {
                            id: p.id,
                            node: d.id,
                            name: s.name.clone(),
                            direction: s.direction,
                            port_type: s.port_type.clone(),
                            connection_policy: s.connection_policy,
                        },
                    );
                    ports.push(p.id);
                }
                self.graph.nodes.insert(
                    d.id,
                    Node {
                        id: d.id,
                        name: d.name.clone(),
                        payload: d.payload.clone(),
                        group: None,
                        ports,
                    },
                );
                self.graph.next_node = d.id.0 + 1;
                self.graph.next_port = next;
                self.positions.insert(d.id, d.position);
                Ok(Inverse::Delete(d.id))
            }
            GraphEdit::DeleteNode(n) => Ok(Inverse::Restore(self.remove(*n)?)),
            GraphEdit::MoveNode { node, position } => {
                let p = self
                    .positions
                    .get_mut(node)
                    .ok_or(GraphError::MissingNode(*node))?;
                Ok(Inverse::Position(*node, core::mem::replace(p, *position)))
            }
            GraphEdit::SetPayload { node, payload } => {
                let n = self
                    .graph
                    .nodes
                    .get_mut(node)
                    .ok_or(GraphError::MissingNode(*node))?;
                Ok(Inverse::Payload(
                    *node,
                    core::mem::replace(&mut n.payload, payload.clone()),
                ))
            }
            GraphEdit::Connect {
                edge,
                output,
                input,
            } => {
                id(edge.0, self.graph.next_edge)?;
                let counter = self.graph.next_edge;
                self.graph.next_edge = edge.0;
                if let Err(e) = self.graph.connect(*output, *input) {
                    self.graph.next_edge = counter;
                    return Err(e.into());
                }
                Ok(Inverse::Disconnect(*edge))
            }
            GraphEdit::Disconnect(e) => Ok(Inverse::Connect(self.graph.disconnect(*e)?)),
        }
    }
    fn remove(&mut self, n: NodeId) -> Result<Removed, GraphError> {
        let node = self.graph.nodes.get(&n).ok_or(GraphError::MissingNode(n))?;
        let ports = node
            .ports
            .iter()
            .map(|p| self.graph.ports[p].clone())
            .collect();
        let ids: BTreeSet<_> = self
            .graph
            .incoming_edges(n)
            .into_iter()
            .chain(self.graph.outgoing_edges(n))
            .map(Edge::id)
            .collect();
        let edges = ids
            .into_iter()
            .map(|e| self.graph.disconnect(e).expect("incident edge exists"))
            .collect();
        let node = self.graph.remove_node(n)?;
        Ok(Removed {
            node,
            ports,
            edges,
            position: self.positions.remove(&n).expect("editor position index"),
        })
    }
    fn reverse(&mut self, inverse: Vec<Inverse>) -> Vec<Inverse> {
        inverse
            .into_iter()
            .rev()
            .map(|i| match i {
                Inverse::Delete(n) => Inverse::Restore(self.remove(n).expect("private inverse")),
                Inverse::Restore(r) => {
                    let n = r.node.id;
                    for p in r.ports {
                        self.graph.port_names.insert((n, p.name.clone()), p.id);
                        self.graph.ports.insert(p.id, p);
                    }
                    self.graph.nodes.insert(n, r.node);
                    self.positions.insert(n, r.position);
                    for e in r.edges {
                        self.graph.insert_edge(e);
                    }
                    Inverse::Delete(n)
                }
                Inverse::Position(n, p) => {
                    Inverse::Position(n, self.positions.insert(n, p).expect("private inverse"))
                }
                Inverse::Payload(n, p) => Inverse::Payload(
                    n,
                    core::mem::replace(
                        &mut self
                            .graph
                            .nodes
                            .get_mut(&n)
                            .expect("private inverse")
                            .payload,
                        p,
                    ),
                ),
                Inverse::Disconnect(e) => {
                    Inverse::Connect(self.graph.disconnect(e).expect("private inverse"))
                }
                Inverse::Connect(e) => {
                    let id = e.id;
                    self.graph.insert_edge(e);
                    Inverse::Disconnect(id)
                }
            })
            .collect()
    }
}
fn id(raw: u64, next: u64) -> Result<(), GraphAuthoringError> {
    if raw == 0 || raw < next || raw == u64::MAX {
        Err(GraphAuthoringError::Identity)
    } else {
        Ok(())
    }
}
