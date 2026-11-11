use std::collections::BTreeMap;

/// The semantic family of a MaterialX value type.
///
/// MaterialX carries the concrete spelling on every port, so the authoring IR
/// keeps that spelling in [`TypedValue::type_name`]. This enum supplies the
/// checked interpretation needed by import/evaluation without collapsing
/// custom shader types into strings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueTypeKind {
    /// A finite floating-point scalar or vector/matrix.
    Float {
        /// Number of components, or zero for a variable-length array.
        components: usize,
    },
    /// An integer scalar or array.
    Integer {
        /// Number of components, or zero for a variable-length array.
        components: usize,
    },
    /// A boolean scalar.
    Boolean,
    /// A textual value whose whitespace is significant.
    String,
    /// A MaterialX token/identifier.
    Token,
    /// A shader, closure, or other opaque value that requires a runtime
    /// implementation rather than scalar evaluation.
    Opaque,
}

/// A parsed literal value and its MaterialX type spelling.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedValue {
    /// MaterialX type (for example `float`, `color3`, `vector3`, or `boolean`).
    pub type_name: String,
    /// Canonical lexical form used by deterministic export.
    pub value: String,
}

impl TypedValue {
    /// Parse and canonicalize one MaterialX lexical value.
    ///
    /// Numeric values are validated at the authoring boundary: wrong arity,
    /// malformed numbers, non-finite values, and invalid booleans are errors.
    /// Unknown-but-well-formed custom types remain lossless opaque values and
    /// are rejected later if a runtime consumer needs scalar semantics.
    pub fn from_lexical(type_name: impl Into<String>, raw: &str) -> Result<Self, String> {
        let type_name = type_name.into();
        let kind = value_type_kind(&type_name);
        let value = match kind {
            ValueTypeKind::Float { components } => {
                let numbers = numeric_tokens(raw)?;
                if components != 0 && numbers.len() != components {
                    return Err(format!(
                        "type '{type_name}' requires {components} numeric components, found {}",
                        numbers.len()
                    ));
                }
                numbers
                    .into_iter()
                    .map(canonical_number)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
            ValueTypeKind::Integer { components } => {
                let values = raw
                    .split(|character: char| character == ',' || character.is_ascii_whitespace())
                    .filter(|part| !part.is_empty())
                    .map(|part| {
                        part.parse::<i64>()
                            .map_err(|_| format!("invalid integer literal '{raw}'"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if components != 0 && values.len() != components {
                    return Err(format!(
                        "type '{type_name}' requires {components} integer components, found {}",
                        values.len()
                    ));
                }
                values
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
            ValueTypeKind::Boolean => match raw.trim() {
                "true" | "1" => "true".to_owned(),
                "false" | "0" => "false".to_owned(),
                _ => return Err(format!("invalid boolean literal '{raw}'")),
            },
            ValueTypeKind::String | ValueTypeKind::Token | ValueTypeKind::Opaque => raw.to_owned(),
        };
        Ok(Self { type_name, value })
    }

    /// Returns the checked type family and component count.
    #[must_use]
    pub fn kind(&self) -> ValueTypeKind {
        value_type_kind(&self.type_name)
    }

    /// Returns finite numeric components for scalar/vector/matrix values.
    pub fn numeric_components(&self) -> Result<Vec<f32>, String> {
        match self.kind() {
            ValueTypeKind::Float { .. } => numeric_tokens(&self.value),
            ValueTypeKind::Integer { .. } => self
                .value
                .split(|character: char| character == ',' || character.is_ascii_whitespace())
                .filter(|part| !part.is_empty())
                .map(|part| {
                    let value = part
                        .parse::<f32>()
                        .map_err(|_| format!("invalid integer literal '{}'", self.value))?;
                    value.is_finite().then_some(value).ok_or_else(|| {
                        format!("integer literal '{}' exceeds f32 range", self.value)
                    })
                })
                .collect(),
            _ => Err(format!("type '{}' is not numeric", self.type_name)),
        }
    }
}

/// Return the structural meaning of a MaterialX type spelling.
#[must_use]
pub fn value_type_kind(type_name: &str) -> ValueTypeKind {
    let scalar = match type_name {
        "float" | "half" | "double" => Some(1),
        "vector2" | "color2" => Some(2),
        "vector3" | "color3" | "point" | "normal" => Some(3),
        "vector4" | "color4" => Some(4),
        "matrix33" => Some(9),
        "matrix44" => Some(16),
        _ => None,
    };
    if let Some(components) = scalar {
        return ValueTypeKind::Float { components };
    }
    if matches!(type_name, "integer" | "int") {
        return ValueTypeKind::Integer { components: 1 };
    }
    if matches!(type_name, "boolean" | "bool") {
        return ValueTypeKind::Boolean;
    }
    if matches!(type_name, "string" | "filename") {
        return ValueTypeKind::String;
    }
    if type_name == "token" {
        return ValueTypeKind::Token;
    }
    if let Some(element) = type_name.strip_suffix("array") {
        return match value_type_kind(element) {
            ValueTypeKind::Float { .. } => ValueTypeKind::Float { components: 0 },
            ValueTypeKind::Integer { .. } => ValueTypeKind::Integer { components: 0 },
            ValueTypeKind::String | ValueTypeKind::Token => ValueTypeKind::String,
            _ => ValueTypeKind::Opaque,
        };
    }
    ValueTypeKind::Opaque
}

fn numeric_tokens(raw: &str) -> Result<Vec<f32>, String> {
    raw.split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let value = part
                .parse::<f32>()
                .map_err(|_| format!("invalid numeric literal '{raw}'"))?;
            if !value.is_finite() {
                return Err(format!("non-finite numeric literal '{raw}'"));
            }
            Ok(value)
        })
        .collect()
}

fn canonical_number(value: f32) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    value.to_string()
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

impl Input {
    /// Returns an explicit MaterialX default, preserving the distinction from
    /// an omitted input. `value` is the canonical spelling used by Prism's
    /// current subset; `default` is accepted for authoring sources that use
    /// the equivalent MaterialX declaration attribute.
    #[must_use]
    pub fn explicit_default(&self) -> Option<TypedValue> {
        self.value.clone().or_else(|| {
            self.attributes.get("default").map(|value| TypedValue {
                type_name: self
                    .type_name
                    .clone()
                    .unwrap_or_else(|| "string".to_owned()),
                value: value.clone(),
            })
        })
    }
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
    /// Whether the source used the generic `<node type="...">` spelling.
    ///
    /// MaterialX permits both a category element (`<add>`) and a generic node
    /// element (`<node type="add">`). Keeping this bit in the IR is required
    /// for a lossless semantic round trip; the two spellings have identical
    /// runtime meaning.
    pub generic_element: bool,
    /// Literal value carried by value-producing nodes such as `constant`.
    pub value: Option<TypedValue>,
    /// Type of a node-level value when the source uses generic `<node>` syntax.
    /// Category and value type are separate MaterialX concepts.
    pub value_type: Option<String>,
    /// Ordered inputs (source order is preserved).
    pub inputs: Vec<Input>,
    /// Output types declared on this node.
    pub outputs: Vec<Output>,
    /// Supported non-structural attributes.
    pub attributes: BTreeMap<String, String>,
}

impl Node {
    /// Returns the source `nodedef` binding, when this node is an instance of
    /// a declared definition.
    #[must_use]
    pub fn nodedef(&self) -> Option<&str> {
        self.attributes.get("nodedef").map(String::as_str)
    }
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
    /// Optional nodedef implemented by this reusable graph.
    pub nodedef: Option<String>,
    /// Compound graph interface inputs in source order.
    pub inputs: Vec<Input>,
    /// Node instances in source order.
    pub nodes: Vec<Node>,
    /// Graph outputs in source order.
    pub outputs: Vec<Output>,
    /// Supported non-structural attributes.
    pub attributes: BTreeMap<String, String>,
}

/// A non-executable MaterialX document element retained for lossless graph
/// package round trips (for example `surfacematerial`, `implementation`, or
/// `propertyset`). Its inputs and nested children remain typed/structured;
/// runtime lowering decides whether a particular element has a Prism mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct DocumentElement {
    /// MaterialX element category.
    pub category: String,
    /// Optional element name.
    pub name: Option<String>,
    /// Optional typed element value (for example a MaterialX `property`).
    pub value: Option<TypedValue>,
    /// Non-structural attributes in deterministic order.
    pub attributes: BTreeMap<String, String>,
    /// Typed input declarations/connections.
    pub inputs: Vec<Input>,
    /// Typed output declarations/connections.
    pub outputs: Vec<Output>,
    /// Nested structural elements.
    pub children: Vec<DocumentElement>,
    /// Character data for source-bearing elements such as `sourcecode`.
    pub text: Option<String>,
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
    /// Supported document-level MaterialX elements in source order.
    pub elements: Vec<DocumentElement>,
}

impl Document {
    /// Finds a node by identifier across nodegraphs.
    pub fn node(&self, name: &str) -> Option<&Node> {
        self.nodegraphs
            .iter()
            .flat_map(|graph| &graph.nodes)
            .find(|node| node.name == name)
    }

    /// Iterates authored image nodes with their owning graph in deterministic
    /// graph/node order. Node names are graph-scoped in MaterialX.
    pub(crate) fn nodegraph_images(&self) -> impl Iterator<Item = (&NodeGraph, &Node)> {
        self.nodegraphs
            .iter()
            .flat_map(|graph| graph.nodes.iter().map(move |node| (graph, node)))
            .filter(|(_, node)| node.category == "image")
    }
}
