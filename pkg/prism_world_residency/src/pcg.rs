use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt;

/// PCG-style avalanche hash for stable ids, seeds, and dither inputs.
pub const fn pcg_hash_u32(input: u64) -> u32 {
    let mut state = input.wrapping_add(0x853c_49e6_748f_ea9b);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let word = ((state >> 18) ^ state) >> 27;
    let rotation = (state >> 59) as u32;
    (word as u32).rotate_right(rotation)
}

fn hash64(input: u64) -> u64 {
    (u64::from(pcg_hash_u32(input)) << 32) | u64::from(pcg_hash_u32(input ^ 0x9e37_79b9_7f4a_7c15))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_i64(bytes: &mut Vec<u8>, value: i64) {
    put_u64(bytes, value as u64);
}

/// Signed sector coordinate used as a deterministic PCG input.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SectorCoord {
    /// X coordinate.
    pub x: i64,
    /// Y coordinate.
    pub y: i64,
    /// Z coordinate.
    pub z: i64,
}

impl SectorCoord {
    /// Creates a sector coordinate.
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    fn append(self, bytes: &mut Vec<u8>) {
        put_i64(bytes, self.x);
        put_i64(bytes, self.y);
        put_i64(bytes, self.z);
    }
}

/// Stable numeric identifier for a graph node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(u32);

impl NodeId {
    /// Creates a node identifier.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the numeric identifier.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for NodeId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

/// A literal or a reference to another node output.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NodeInput {
    /// Signed integer literal.
    Literal(i64),
    /// Reference to a node output.
    Node(NodeId),
}

impl NodeInput {
    /// Creates a node reference.
    pub const fn node(id: NodeId) -> Self {
        Self::Node(id)
    }

    /// Creates a literal.
    pub const fn literal(value: i64) -> Self {
        Self::Literal(value)
    }
}

/// Integer-only PCG operation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NodeKind {
    /// Produces a constant.
    Constant {
        /// Constant signed integer value.
        value: i64,
    },
    /// Wrapping integer addition.
    Add {
        /// Left operand.
        lhs: NodeInput,
        /// Right operand.
        rhs: NodeInput,
    },
    /// Wrapping integer multiplication.
    Multiply {
        /// Left operand.
        lhs: NodeInput,
        /// Right operand.
        rhs: NodeInput,
    },
    /// Produces sector/seed/node/salt-derived hash noise.
    HashNoise {
        /// Stable salt mixed into the noise input.
        salt: u64,
    },
    /// Selects true when condition is non-zero.
    Select {
        /// Branch condition.
        condition: NodeInput,
        /// Value selected for a non-zero condition.
        when_true: NodeInput,
        /// Value selected for a zero condition.
        when_false: NodeInput,
    },
}

impl NodeKind {
    fn inputs(self) -> [Option<NodeInput>; 3] {
        match self {
            Self::Constant { .. } | Self::HashNoise { .. } => [None, None, None],
            Self::Add { lhs, rhs } | Self::Multiply { lhs, rhs } => [Some(lhs), Some(rhs), None],
            Self::Select {
                condition,
                when_true,
                when_false,
            } => [Some(condition), Some(when_true), Some(when_false)],
        }
    }

    fn append(&self, bytes: &mut Vec<u8>) {
        match *self {
            Self::Constant { value } => {
                bytes.push(0);
                put_i64(bytes, value);
            }
            Self::Add { lhs, rhs } => {
                bytes.push(1);
                append_input(bytes, lhs);
                append_input(bytes, rhs);
            }
            Self::Multiply { lhs, rhs } => {
                bytes.push(2);
                append_input(bytes, lhs);
                append_input(bytes, rhs);
            }
            Self::HashNoise { salt } => {
                bytes.push(3);
                put_u64(bytes, salt);
            }
            Self::Select {
                condition,
                when_true,
                when_false,
            } => {
                bytes.push(4);
                append_input(bytes, condition);
                append_input(bytes, when_true);
                append_input(bytes, when_false);
            }
        }
    }
}

fn append_input(bytes: &mut Vec<u8>, input: NodeInput) {
    match input {
        NodeInput::Literal(value) => {
            bytes.push(0);
            put_i64(bytes, value);
        }
        NodeInput::Node(id) => {
            bytes.push(1);
            put_u32(bytes, id.get());
        }
    }
}

/// A graph node declaration.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PcgNode {
    /// Stable node identifier.
    pub id: NodeId,
    /// Node operation.
    pub kind: NodeKind,
}

impl PcgNode {
    /// Creates a node declaration.
    pub const fn new(id: NodeId, kind: NodeKind) -> Self {
        Self { id, kind }
    }
}

/// Graph validation failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcgGraphError {
    /// No nodes were supplied.
    EmptyGraph,
    /// A node id occurs more than once.
    DuplicateNode(NodeId),
    /// The selected output is absent.
    MissingOutput(NodeId),
    /// An input references an absent node.
    MissingInput {
        /// Node containing the invalid reference.
        node: NodeId,
        /// Referenced node that is absent.
        input: NodeId,
    },
    /// The graph has a cycle.
    Cycle,
    /// The supplied order is not canonical topological order.
    NonCanonicalGraph,
    /// An invalidation request names an absent node.
    UnknownNode(NodeId),
}

impl fmt::Display for PcgGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyGraph => f.write_str("PCG graph is empty"),
            Self::DuplicateNode(id) => write!(f, "PCG graph contains duplicate node {}", id.get()),
            Self::MissingOutput(id) => write!(f, "PCG graph output node {} is missing", id.get()),
            Self::MissingInput { node, input } => write!(
                f,
                "PCG node {} references missing input node {}",
                node.get(),
                input.get()
            ),
            Self::Cycle => f.write_str("PCG graph contains a dependency cycle"),
            Self::NonCanonicalGraph => f.write_str("PCG graph is not canonical"),
            Self::UnknownNode(id) => write!(f, "PCG invalidation names unknown node {}", id.get()),
        }
    }
}

/// A node invalidation scoped to one sector.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DirtySector {
    /// Sector to regenerate.
    pub sector: SectorCoord,
    /// Changed node in that sector.
    pub node: NodeId,
}

impl DirtySector {
    /// Creates a dirty sector entry.
    pub const fn new(sector: SectorCoord, node: NodeId) -> Self {
        Self { sector, node }
    }
}

/// Direct and reverse dependency relations for incremental evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencySet {
    direct: BTreeMap<NodeId, Vec<NodeId>>,
    downstream: BTreeMap<NodeId, Vec<NodeId>>,
}

impl DependencySet {
    fn build(nodes: &[PcgNode]) -> Result<Self, PcgGraphError> {
        let declared: BTreeSet<NodeId> = nodes.iter().map(|node| node.id).collect();
        let mut direct = BTreeMap::new();
        let mut reverse: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for node in nodes {
            let mut dependencies = Vec::new();
            for input in node.kind.inputs().into_iter().flatten() {
                if let NodeInput::Node(input_id) = input {
                    if !declared.contains(&input_id) {
                        return Err(PcgGraphError::MissingInput {
                            node: node.id,
                            input: input_id,
                        });
                    }
                    if !dependencies.contains(&input_id) {
                        dependencies.push(input_id);
                    }
                }
            }
            dependencies.sort_unstable();
            for dependency in &dependencies {
                reverse.entry(*dependency).or_default().push(node.id);
            }
            direct.insert(node.id, dependencies);
        }
        let mut downstream = BTreeMap::new();
        for node in nodes {
            let mut found = BTreeSet::new();
            let mut pending = reverse.get(&node.id).cloned().unwrap_or_default();
            while let Some(next) = pending.pop() {
                if found.insert(next)
                    && let Some(children) = reverse.get(&next)
                {
                    pending.extend(children.iter().copied());
                }
            }
            downstream.insert(node.id, found.into_iter().collect());
        }
        Ok(Self { direct, downstream })
    }

    /// Returns direct input node ids.
    pub fn direct_inputs(&self, node: NodeId) -> Option<&[NodeId]> {
        self.direct.get(&node).map(Vec::as_slice)
    }

    /// Returns all transitive downstream node ids.
    pub fn downstream(&self, node: NodeId) -> Option<&[NodeId]> {
        self.downstream.get(&node).map(Vec::as_slice)
    }

    /// Returns changed nodes and all of their downstream consumers.
    pub fn affected_nodes(&self, changed: &[NodeId]) -> Result<Vec<NodeId>, PcgGraphError> {
        let mut result = BTreeSet::new();
        for node in changed {
            if !self.direct.contains_key(node) {
                return Err(PcgGraphError::UnknownNode(*node));
            }
            result.insert(*node);
            if let Some(children) = self.downstream.get(node) {
                result.extend(children.iter().copied());
            }
        }
        Ok(result.into_iter().collect())
    }

    /// Propagates invalidation without crossing sector boundaries.
    pub fn propagate_dirty_sectors(
        &self,
        dirty: &[DirtySector],
    ) -> Result<Vec<DirtySector>, PcgGraphError> {
        let mut result = BTreeSet::new();
        for entry in dirty {
            for node in self.affected_nodes(core::slice::from_ref(&entry.node))? {
                result.insert(DirtySector::new(entry.sector, node));
            }
        }
        Ok(result.into_iter().collect())
    }
}

/// A validated PCG graph in canonical topological order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PcgGraph {
    schema_version: u32,
    algorithm_version: u32,
    output: NodeId,
    nodes: Vec<PcgNode>,
    index: BTreeMap<NodeId, usize>,
    dependencies: DependencySet,
    graph_hash: u64,
}

impl PcgGraph {
    /// Canonicalizes insertion order by node id and validates the result.
    ///
    /// Use [`Self::new`] when callers must reject an already non-canonical
    /// serialized graph. This constructor is for authoring/import paths where
    /// insertion order is not part of the graph identity.
    pub fn from_nodes(
        schema_version: u32,
        algorithm_version: u32,
        mut nodes: Vec<PcgNode>,
        output: NodeId,
    ) -> Result<Self, PcgGraphError> {
        nodes.sort_unstable_by_key(|node| node.id);
        Self::new(schema_version, algorithm_version, nodes, output)
    }

    /// Validates a graph. Nodes must be supplied in the unique increasing-id
    /// topological order, making the serialized form canonical.
    pub fn new(
        schema_version: u32,
        algorithm_version: u32,
        nodes: Vec<PcgNode>,
        output: NodeId,
    ) -> Result<Self, PcgGraphError> {
        if nodes.is_empty() {
            return Err(PcgGraphError::EmptyGraph);
        }
        let mut declared = BTreeSet::new();
        for node in &nodes {
            if !declared.insert(node.id) {
                return Err(PcgGraphError::DuplicateNode(node.id));
            }
        }
        if !declared.contains(&output) {
            return Err(PcgGraphError::MissingOutput(output));
        }
        for pair in nodes.windows(2) {
            if pair[0].id >= pair[1].id {
                return Err(PcgGraphError::NonCanonicalGraph);
            }
        }

        let mut indegree = BTreeMap::new();
        let mut reverse: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for node in &nodes {
            indegree.insert(node.id, 0_usize);
        }
        for node in &nodes {
            let mut dependencies = BTreeSet::new();
            for input in node.kind.inputs().into_iter().flatten() {
                if let NodeInput::Node(input_id) = input {
                    if !declared.contains(&input_id) {
                        return Err(PcgGraphError::MissingInput {
                            node: node.id,
                            input: input_id,
                        });
                    }
                    dependencies.insert(input_id);
                }
            }
            for dependency in dependencies {
                if let Some(degree) = indegree.get_mut(&node.id) {
                    *degree += 1;
                }
                reverse.entry(dependency).or_default().push(node.id);
            }
        }

        let mut ready: BTreeSet<NodeId> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect();
        let mut topology = Vec::with_capacity(nodes.len());
        while let Some(id) = ready.pop_first() {
            topology.push(id);
            if let Some(children) = reverse.get(&id) {
                for child in children {
                    if let Some(degree) = indegree.get_mut(child) {
                        *degree -= 1;
                        if *degree == 0 {
                            ready.insert(*child);
                        }
                    }
                }
            }
        }
        let supplied: Vec<NodeId> = nodes.iter().map(|node| node.id).collect();
        if topology.len() != nodes.len() {
            return Err(PcgGraphError::Cycle);
        }
        if topology != supplied {
            return Err(PcgGraphError::NonCanonicalGraph);
        }

        let mut index = BTreeMap::new();
        for (position, node) in nodes.iter().enumerate() {
            index.insert(node.id, position);
        }
        let dependencies = DependencySet::build(&nodes)?;
        let graph_hash = graph_hash(schema_version, algorithm_version, &nodes, output);
        Ok(Self {
            schema_version,
            algorithm_version,
            output,
            nodes,
            index,
            dependencies,
            graph_hash,
        })
    }

    /// Returns the graph schema version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the evaluation algorithm version.
    pub const fn algorithm_version(&self) -> u32 {
        self.algorithm_version
    }

    /// Returns the output node.
    pub const fn output_node(&self) -> NodeId {
        self.output
    }

    /// Returns canonical graph nodes.
    pub fn nodes(&self) -> &[PcgNode] {
        &self.nodes
    }

    /// Returns canonical topological node ids.
    pub fn topology(&self) -> Vec<NodeId> {
        self.nodes.iter().map(|node| node.id).collect()
    }

    /// Returns the dependency index.
    pub const fn dependencies(&self) -> &DependencySet {
        &self.dependencies
    }

    /// Returns the canonical graph hash.
    pub const fn graph_hash(&self) -> u64 {
        self.graph_hash
    }

    /// Evaluates all nodes using only integer operations.
    pub fn evaluate(&self, sector: SectorCoord, seed: u64) -> PcgEvaluation {
        let mut values = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let value = match node.kind {
                NodeKind::Constant { value } => value,
                NodeKind::Add { lhs, rhs } => resolve(lhs, &self.index, &values)
                    .wrapping_add(resolve(rhs, &self.index, &values)),
                NodeKind::Multiply { lhs, rhs } => resolve(lhs, &self.index, &values)
                    .wrapping_mul(resolve(rhs, &self.index, &values)),
                NodeKind::HashNoise { salt } => hash_noise(sector, seed, node.id, salt) as i64,
                NodeKind::Select {
                    condition,
                    when_true,
                    when_false,
                } => {
                    if resolve(condition, &self.index, &values) != 0 {
                        resolve(when_true, &self.index, &values)
                    } else {
                        resolve(when_false, &self.index, &values)
                    }
                }
            };
            values.push(value);
        }
        let output = values[self.index[&self.output]];
        let bytes = self.result_bytes(sector, seed, &values, output);
        let hash = fnv1a(&bytes);
        PcgEvaluation {
            output,
            bytes,
            hash,
        }
    }

    fn result_bytes(&self, sector: SectorCoord, seed: u64, values: &[i64], output: i64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PRISM-PCG-RESULT-1");
        put_u32(&mut bytes, self.schema_version);
        put_u32(&mut bytes, self.algorithm_version);
        put_u64(&mut bytes, self.graph_hash);
        sector.append(&mut bytes);
        put_u64(&mut bytes, seed);
        put_u32(&mut bytes, self.nodes.len() as u32);
        for (node, value) in self.nodes.iter().zip(values) {
            put_u32(&mut bytes, node.id.get());
            put_i64(&mut bytes, *value);
        }
        put_i64(&mut bytes, output);
        bytes
    }
}

fn resolve(input: NodeInput, index: &BTreeMap<NodeId, usize>, values: &[i64]) -> i64 {
    match input {
        NodeInput::Literal(value) => value,
        NodeInput::Node(id) => values[index[&id]],
    }
}

fn graph_hash(schema: u32, algorithm: u32, nodes: &[PcgNode], output: NodeId) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"PRISM-PCG-GRAPH-1");
    put_u32(&mut bytes, schema);
    put_u32(&mut bytes, algorithm);
    put_u32(&mut bytes, output.get());
    for node in nodes {
        put_u32(&mut bytes, node.id.get());
        node.kind.append(&mut bytes);
    }
    fnv1a(&bytes)
}

fn hash_noise(sector: SectorCoord, seed: u64, node: NodeId, salt: u64) -> u64 {
    let state = hash64(seed ^ salt ^ u64::from(node.get()).rotate_left(23) ^ sector.x as u64);
    let state = hash64(state ^ (sector.y as u64).rotate_left(17));
    hash64(state ^ (sector.z as u64).rotate_left(37))
}

#[cfg(test)]
mod tests {
    use super::{
        DerivedCacheKey, DirtySector, NodeId, NodeInput, NodeKind, PcgGraph, PcgGraphError,
        PcgNode, SectorCoord,
    };
    use alloc::vec;

    fn graph() -> PcgGraph {
        PcgGraph::new(
            7,
            11,
            vec![
                PcgNode::new(NodeId::new(1), NodeKind::Constant { value: 3 }),
                PcgNode::new(NodeId::new(2), NodeKind::HashNoise { salt: 0x55aa }),
                PcgNode::new(
                    NodeId::new(3),
                    NodeKind::Add {
                        lhs: NodeInput::node(NodeId::new(1)),
                        rhs: NodeInput::node(NodeId::new(2)),
                    },
                ),
                PcgNode::new(
                    NodeId::new(4),
                    NodeKind::Select {
                        condition: NodeInput::node(NodeId::new(1)),
                        when_true: NodeInput::node(NodeId::new(3)),
                        when_false: NodeInput::literal(-1),
                    },
                ),
            ],
            NodeId::new(4),
        )
        .expect("test graph is canonical")
    }

    #[test]
    fn topology_and_all_integer_operations_are_deterministic() {
        let graph = graph();
        assert_eq!(
            graph.topology(),
            vec![
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(3),
                NodeId::new(4)
            ]
        );
        let sector = SectorCoord::new(-4, 2, 9);
        let first = graph.evaluate(sector, 123);
        let second = graph.evaluate(sector, 123);
        assert_eq!(first, second);
        assert_eq!(first.stable_bytes(), second.stable_bytes());
        assert_eq!(first.stable_hash(), second.stable_hash());
        assert_ne!(
            first.stable_hash(),
            graph.evaluate(sector, 124).stable_hash()
        );

        let arithmetic = PcgGraph::new(
            1,
            1,
            vec![
                PcgNode::new(NodeId::new(1), NodeKind::Constant { value: 2 }),
                PcgNode::new(
                    NodeId::new(2),
                    NodeKind::Multiply {
                        lhs: NodeInput::node(NodeId::new(1)),
                        rhs: NodeInput::literal(5),
                    },
                ),
                PcgNode::new(
                    NodeId::new(3),
                    NodeKind::Select {
                        condition: NodeInput::node(NodeId::new(1)),
                        when_true: NodeInput::node(NodeId::new(2)),
                        when_false: NodeInput::literal(0),
                    },
                ),
            ],
            NodeId::new(3),
        )
        .expect("arithmetic graph is canonical");
        assert_eq!(
            arithmetic.evaluate(SectorCoord::new(0, 0, 0), 0).output(),
            10
        );
    }

    #[test]
    fn canonicalization_makes_authoring_order_irrelevant() {
        let canonical = graph();
        let imported = PcgGraph::from_nodes(
            7,
            11,
            vec![
                canonical.nodes()[3],
                canonical.nodes()[1],
                canonical.nodes()[0],
                canonical.nodes()[2],
            ],
            NodeId::new(4),
        )
        .expect("import path canonicalizes order");
        assert_eq!(canonical, imported);
        assert_eq!(
            canonical
                .evaluate(SectorCoord::new(1, 2, 3), 99)
                .stable_hash(),
            imported
                .evaluate(SectorCoord::new(1, 2, 3), 99)
                .stable_hash()
        );

        let rejected = PcgGraph::new(
            7,
            11,
            vec![
                canonical.nodes()[1],
                canonical.nodes()[0],
                canonical.nodes()[2],
                canonical.nodes()[3],
            ],
            NodeId::new(4),
        );
        assert_eq!(rejected, Err(PcgGraphError::NonCanonicalGraph));
    }

    #[test]
    fn invalid_graphs_are_rejected() {
        let duplicate = PcgGraph::new(
            1,
            1,
            vec![
                PcgNode::new(NodeId::new(1), NodeKind::Constant { value: 0 }),
                PcgNode::new(NodeId::new(1), NodeKind::Constant { value: 1 }),
            ],
            NodeId::new(1),
        );
        assert_eq!(duplicate, Err(PcgGraphError::DuplicateNode(NodeId::new(1))));

        let missing = PcgGraph::new(
            1,
            1,
            vec![PcgNode::new(
                NodeId::new(1),
                NodeKind::Add {
                    lhs: NodeInput::node(NodeId::new(99)),
                    rhs: NodeInput::literal(1),
                },
            )],
            NodeId::new(1),
        );
        assert_eq!(
            missing,
            Err(PcgGraphError::MissingInput {
                node: NodeId::new(1),
                input: NodeId::new(99)
            })
        );

        let cycle = PcgGraph::new(
            1,
            1,
            vec![
                PcgNode::new(
                    NodeId::new(1),
                    NodeKind::Add {
                        lhs: NodeInput::node(NodeId::new(2)),
                        rhs: NodeInput::literal(1),
                    },
                ),
                PcgNode::new(
                    NodeId::new(2),
                    NodeKind::Add {
                        lhs: NodeInput::node(NodeId::new(1)),
                        rhs: NodeInput::literal(1),
                    },
                ),
            ],
            NodeId::new(1),
        );
        assert_eq!(cycle, Err(PcgGraphError::Cycle));
    }

    #[test]
    fn dirty_propagation_is_transitive_but_sector_local() {
        let dirty = graph()
            .dependencies()
            .propagate_dirty_sectors(&[
                DirtySector::new(SectorCoord::new(1, 0, 0), NodeId::new(1)),
                DirtySector::new(SectorCoord::new(2, 0, 0), NodeId::new(2)),
            ])
            .expect("all dirty nodes are declared");
        assert_eq!(
            dirty,
            vec![
                DirtySector::new(SectorCoord::new(1, 0, 0), NodeId::new(1)),
                DirtySector::new(SectorCoord::new(1, 0, 0), NodeId::new(3)),
                DirtySector::new(SectorCoord::new(1, 0, 0), NodeId::new(4)),
                DirtySector::new(SectorCoord::new(2, 0, 0), NodeId::new(2)),
                DirtySector::new(SectorCoord::new(2, 0, 0), NodeId::new(3)),
                DirtySector::new(SectorCoord::new(2, 0, 0), NodeId::new(4)),
            ]
        );
    }

    #[test]
    fn cache_key_contains_graph_schema_algorithm_sector_and_seed() {
        let graph = graph();
        let sector = SectorCoord::new(8, -2, 4);
        let base = DerivedCacheKey::for_graph(&graph, sector, 77);
        assert_eq!(base.graph_hash, graph.graph_hash());
        assert_eq!(base.schema_version, 7);
        assert_eq!(base.algorithm_version, 11);
        assert_eq!(base.sector, sector);
        assert_eq!(base.seed, 77);
        assert_ne!(base, DerivedCacheKey::for_graph(&graph, sector, 78));
        assert_ne!(
            base,
            DerivedCacheKey::for_graph(&graph, SectorCoord::new(9, -2, 4), 77)
        );
        assert_ne!(
            base,
            DerivedCacheKey::new(base.graph_hash ^ 1, 7, 11, sector, 77)
        );
    }
}

/// Stable result of one graph evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PcgEvaluation {
    output: i64,
    bytes: Vec<u8>,
    hash: u64,
}

impl PcgEvaluation {
    /// Returns the output node value.
    pub const fn output(&self) -> i64 {
        self.output
    }

    /// Returns canonical result bytes.
    pub fn stable_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the stable result hash.
    pub const fn stable_hash(&self) -> u64 {
        self.hash
    }
}

/// Complete identity of a derived sector result.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DerivedCacheKey {
    /// Canonical graph hash.
    pub graph_hash: u64,
    /// Graph schema version.
    pub schema_version: u32,
    /// Evaluation algorithm version.
    pub algorithm_version: u32,
    /// Sector coordinate.
    pub sector: SectorCoord,
    /// World/content seed.
    pub seed: u64,
    /// Compact upper digest half.
    pub hi: u64,
    /// Compact lower digest half.
    pub lo: u64,
}

impl DerivedCacheKey {
    /// Creates a complete derived cache key.
    pub const fn new(
        graph_hash: u64,
        schema_version: u32,
        algorithm_version: u32,
        sector: SectorCoord,
        seed: u64,
    ) -> Self {
        let hi = graph_hash
            ^ seed.rotate_left(17)
            ^ (schema_version as u64).rotate_left(31)
            ^ (algorithm_version as u64).rotate_left(47)
            ^ (sector.x as u64).rotate_left(7)
            ^ (sector.y as u64).rotate_left(19)
            ^ (sector.z as u64).rotate_left(37);
        let lo = graph_hash.rotate_left(29)
            ^ seed.rotate_right(11)
            ^ (schema_version as u64).wrapping_mul(0x9e37_79b9)
            ^ (algorithm_version as u64).wrapping_mul(0x85eb_ca6b)
            ^ (sector.x as u64).rotate_right(13)
            ^ (sector.y as u64).rotate_right(23)
            ^ (sector.z as u64).rotate_right(41);
        Self {
            graph_hash,
            schema_version,
            algorithm_version,
            sector,
            seed,
            hi,
            lo,
        }
    }

    /// Creates a key for one graph and sector.
    pub const fn for_graph(graph: &PcgGraph, sector: SectorCoord, seed: u64) -> Self {
        Self::new(
            graph.graph_hash,
            graph.schema_version,
            graph.algorithm_version,
            sector,
            seed,
        )
    }
}

impl fmt::Display for DerivedCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}{:016x}", self.hi, self.lo)
    }
}

/// Creates a complete cache key without constructing a graph.
pub const fn derive_cache_key(
    graph_hash: u64,
    schema_version: u32,
    algorithm_version: u32,
    sector: SectorCoord,
    seed: u64,
) -> DerivedCacheKey {
    DerivedCacheKey::new(graph_hash, schema_version, algorithm_version, sector, seed)
}

/// Deterministic PCG32 generator with an explicit stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Pcg32 {
    state: u64,
    increment: u64,
}

impl Pcg32 {
    /// Creates a generator from a seed and stream selector.
    pub const fn new(seed: u64, stream: u64) -> Self {
        let increment = (stream << 1) | 1;
        let mut generator = Self {
            state: 0,
            increment,
        };
        let _ = generator.next_u32();
        generator.state = generator.state.wrapping_add(seed);
        let _ = generator.next_u32();
        generator
    }

    /// Returns the next deterministic 32-bit value.
    pub const fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.increment);
        let word = ((old_state >> 18) ^ old_state) >> 27;
        let rotation = (old_state >> 59) as u32;
        (word as u32).rotate_right(rotation)
    }

    /// Returns two consecutive values as one 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }

    /// Returns a value in `[0, upper)` using rejection sampling.
    pub fn next_bounded(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            return 0;
        }
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u32();
            if value >= threshold {
                return value % upper;
            }
        }
    }

    /// Returns a Q16 unit value.
    pub fn next_unit_q16(&mut self) -> u16 {
        (self.next_u32() >> 16) as u16
    }
}
