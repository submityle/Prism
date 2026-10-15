use std::collections::BTreeMap;

/// A parsed literal value and its MaterialX type spelling.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedValue {
    /// MaterialX type (for example `float`, `color3`, `vector3`, or `boolean`).
    pub type_name: String,
    /// Canonical lexical form used by deterministic export.
    pub value: String,
}

/// An input is either a literal or a connection; these forms are mutually exclusive.
#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    /// Input name.
    pub name: String,
    /// Declared type when present.
    pub type_name: Option<String>,
    /// Literal value.
    pub value: Option<TypedValue>,
    /// Connected source, represented by MaterialX node/graph/output identifiers.
    pub connection: Option<Connection>,
    /// Other supported standard attributes (for example `enum`), kept losslessly.
    pub attributes: BTreeMap<String, String>,
}

/// A MaterialX connection from a nodegraph output or node output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Connection {
    /// Referenced node identifier; absent when the connection targets a graph output.
    pub node: Option<String>,
    /// Optional output name.
    pub output: Option<String>,
    /// Optional nodegraph identifier. When present, `node` names the source node within that graph.
    pub nodegraph: Option<String>,
}

/// A node instance inside a nodegraph.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// Stable document identifier.
    pub name: String,
    /// Node category, such as `constant`, `multiply`, or `open_pbr_surface`.
    pub category: String,
    /// Literal value carried by value-producing nodes such as `constant`.
    pub value: Option<TypedValue>,
    /// Ordered inputs (source order is preserved).
    pub inputs: Vec<Input>,
    /// Output types declared on this node.
    pub outputs: Vec<Output>,
    /// Supported non-structural attributes.
    pub attributes: BTreeMap<String, String>,
}

/// A graph output.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    /// Output name.
    pub name: String,
    /// Output type.
    pub type_name: Option<String>,
    /// Connection source, absent only for a typed declaration in a nodedef.
    pub connection: Option<Connection>,
    /// Optional literal output value (supported in definitions).
    pub value: Option<TypedValue>,
}

/// Node definition signature.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeDef {
    /// MaterialX nodedef identifier.
    pub name: String,
    /// Category used by node instances.
    pub category: String,
    /// Input signature.
    pub inputs: Vec<Input>,
    /// Output signature.
    pub outputs: Vec<Output>,
    /// Supported non-structural attributes.
    pub attributes: BTreeMap<String, String>,
}

/// Node graph with deterministic child storage.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeGraph {
    /// Stable graph identifier.
    pub name: String,
    /// Compound graph interface inputs in source order.
    pub inputs: Vec<Input>,
    /// Node instances in source order.
    pub nodes: Vec<Node>,
    /// Graph outputs in source order.
    pub outputs: Vec<Output>,
    /// Supported non-structural attributes.
    pub attributes: BTreeMap<String, String>,
}

/// Loss-aware MaterialX authoring document for the supported structural subset.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Document {
    /// Root attributes supported by this frontend.
    pub attributes: BTreeMap<String, String>,
    /// Node definitions sorted by identifier.
    pub nodedefs: Vec<NodeDef>,
    /// Node graphs sorted by identifier.
    pub nodegraphs: Vec<NodeGraph>,
}

impl Document {
    /// Finds a node by identifier across nodegraphs.
    pub fn node(&self, name: &str) -> Option<&Node> {
        self.nodegraphs
            .iter()
            .flat_map(|graph| &graph.nodes)
            .find(|node| node.name == name)
    }
}
