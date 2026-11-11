//! Lower the supported MaterialX procedural subset into the runtime material IR.
//!
//! The lowering is deliberately explicit: unsupported nodes, external graph
//! references, dynamic noise parameters, and malformed curve data are errors.
//! This keeps authoring and runtime semantics aligned without pretending that a
//! general MaterialX evaluator exists.

use std::collections::{BTreeMap, BTreeSet};

use prism_render_material::{
    bake_procedural_vt, MaterialGraph, MaterialNode, MaterialNodeId, MaterialValue,
    MaterialVtSourceSnapshot, ProceduralVtBakeConfig, ProceduralVtBakeError,
};

use crate::{Connection, Document, Input, Node, NodeGraph, TypedValue};

/// Error while lowering a MaterialX procedural graph into the runtime IR.
#[derive(Clone, Debug, PartialEq)]
pub enum ProceduralGraphError {
    /// The graph must expose exactly one connected output for baking.
    OutputShape,
    /// A referenced node or graph output does not exist.
    InvalidReference(String),
    /// A connection points into a different graph, which this bounded ABI does not embed.
    ExternalGraph(String),
    /// The node category is outside the procedural bake ABI.
    UnsupportedCategory {
        /// Source node name.
        node: String,
        /// Source node category.
        category: String,
    },
    /// A required input is absent or has the wrong form.
    InvalidInput {
        /// Source node name.
        node: String,
        /// Input name or structural input description.
        input: String,
    },
    /// A literal could not be represented by the runtime material value ABI.
    InvalidValue {
        /// Source node name.
        node: String,
        /// Input carrying the invalid value.
        input: String,
    },
    /// A graph cycle was found while lowering.
    Cycle(String),
    /// The lowered graph would exceed the runtime node-id space.
    NodeIdExhausted,
    /// The runtime procedural baker rejected the lowered graph.
    Bake(ProceduralVtBakeError),
}

impl std::fmt::Display for ProceduralGraphError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutputShape => {
                formatter.write_str("procedural graph needs exactly one connected output")
            }
            Self::InvalidReference(reference) => {
                write!(formatter, "invalid procedural graph reference: {reference}")
            }
            Self::ExternalGraph(graph) => write!(
                formatter,
                "external graph '{graph}' is not supported by the procedural bake ABI"
            ),
            Self::UnsupportedCategory { node, category } => {
                write!(
                    formatter,
                    "node '{node}' uses unsupported procedural category '{category}'"
                )
            }
            Self::InvalidInput { node, input } => {
                write!(
                    formatter,
                    "node '{node}' has invalid procedural input '{input}'"
                )
            }
            Self::InvalidValue { node, input } => {
                write!(
                    formatter,
                    "node '{node}' has invalid literal input '{input}'"
                )
            }
            Self::Cycle(node) => write!(formatter, "procedural graph cycle at node '{node}'"),
            Self::NodeIdExhausted => {
                formatter.write_str("procedural graph node id space exhausted")
            }
            Self::Bake(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProceduralGraphError {}

/// Lowers one bounded MaterialX procedural graph into the runtime material IR.
///
/// Compound nodegraphs are expanded at this boundary. Interface values are
/// carried as graph-node ids, so an instance override remains dynamic in the
/// lowered graph instead of being replaced by a definition default.
pub fn lower_procedural_graph(
    document: &Document,
    graph: &NodeGraph,
) -> Result<MaterialGraph, ProceduralGraphError> {
    let outputs = graph
        .outputs
        .iter()
        .filter(|output| output.connection.is_some())
        .collect::<Vec<_>>();
    if outputs.len() != 1 {
        return Err(ProceduralGraphError::OutputShape);
    }

    let mut context = LowerContext {
        document,
        nodes: BTreeMap::new(),
        active: BTreeSet::new(),
        next_id: 1,
    };
    let output = outputs[0]
        .connection
        .as_ref()
        .ok_or(ProceduralGraphError::OutputShape)
        .and_then(|connection| context.lower_connection(graph, connection, &BTreeMap::new()))?;
    Ok(MaterialGraph {
        nodes: context.nodes,
        output: Some(output),
        plan: None,
    })
}

/// Lowers and bakes a supported MaterialX graph into the production VT source contract.
pub fn bake_procedural_graph(
    document: &Document,
    graph: &NodeGraph,
    config: ProceduralVtBakeConfig,
) -> Result<MaterialVtSourceSnapshot, ProceduralGraphError> {
    let lowered = lower_procedural_graph(document, graph)?;
    bake_procedural_vt(&lowered, config).map_err(ProceduralGraphError::Bake)
}

struct LowerContext<'a> {
    document: &'a Document,
    nodes: BTreeMap<MaterialNodeId, MaterialNode>,
    active: BTreeSet<String>,
    next_id: u32,
}

impl<'a> LowerContext<'a> {
    fn lower_connection(
        &mut self,
        graph: &'a NodeGraph,
        connection: &Connection,
        interfaces: &BTreeMap<String, MaterialNodeId>,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        let target = if let Some(name) = connection.nodegraph.as_deref() {
            self.document
                .nodegraphs
                .iter()
                .find(|candidate| candidate.name == name)
                .ok_or_else(|| {
                    ProceduralGraphError::InvalidReference(format!("nodegraph '{name}'"))
                })?
        } else {
            graph
        };
        let target_interfaces = if target.name == graph.name {
            interfaces
        } else {
            &BTreeMap::new()
        };
        if let Some(node) = connection.node.as_deref() {
            return self.lower_node_output(
                target,
                node,
                connection.output.as_deref(),
                target_interfaces,
            );
        }
        let output = connection
            .output
            .as_deref()
            .or_else(|| (target.outputs.len() == 1).then(|| target.outputs[0].name.as_str()))
            .ok_or_else(|| {
                ProceduralGraphError::InvalidReference("graph output is ambiguous".into())
            })?;
        self.lower_graph_output(target, output, target_interfaces)
    }

    fn lower_graph_output(
        &mut self,
        graph: &'a NodeGraph,
        output: &str,
        interfaces: &BTreeMap<String, MaterialNodeId>,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        let key = format!("{}::@output::{output}", graph.name);
        if !self.active.insert(key.clone()) {
            return Err(ProceduralGraphError::Cycle(key));
        }
        let result = graph
            .outputs
            .iter()
            .find(|candidate| candidate.name == output)
            .ok_or_else(|| {
                ProceduralGraphError::InvalidReference(format!("graph output '{output}'"))
            })
            .and_then(|graph_output| {
                graph_output.connection.as_ref().ok_or_else(|| {
                    ProceduralGraphError::InvalidReference(format!(
                        "unconnected graph output '{output}'"
                    ))
                })
            })
            .and_then(|source| self.lower_connection(graph, source, interfaces));
        self.active.remove(&key);
        result
    }

    fn lower_node_output(
        &mut self,
        graph: &'a NodeGraph,
        name: &str,
        output: Option<&str>,
        interfaces: &BTreeMap<String, MaterialNodeId>,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        let node = graph
            .nodes
            .iter()
            .find(|node| node.name == name)
            .ok_or_else(|| ProceduralGraphError::InvalidReference(format!("node '{name}'")))?
            .clone();
        let key = format!("{}::{name}", graph.name);
        if !self.active.insert(key.clone()) {
            return Err(ProceduralGraphError::Cycle(key));
        }
        let result = if let Some(definition_name) = node.nodedef() {
            if let Some(implementation) = self
                .document
                .nodegraphs
                .iter()
                .find(|candidate| candidate.nodedef.as_deref() == Some(definition_name))
            {
                self.lower_graph_instance(graph, &node, implementation, output, interfaces)
            } else {
                self.lower_node_value(graph, &node, interfaces)
            }
        } else {
            self.lower_node_value(graph, &node, interfaces)
        };
        self.active.remove(&key);
        result
    }

    fn lower_graph_instance(
        &mut self,
        caller: &'a NodeGraph,
        instance: &Node,
        implementation: &'a NodeGraph,
        selected_output: Option<&str>,
        caller_interfaces: &BTreeMap<String, MaterialNodeId>,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        let definition_name = implementation.nodedef.as_deref().ok_or_else(|| {
            ProceduralGraphError::InvalidReference(format!(
                "implementation graph '{}' has no nodedef",
                implementation.name
            ))
        })?;
        let definition = self
            .document
            .nodedefs
            .iter()
            .find(|candidate| candidate.name == definition_name)
            .ok_or_else(|| {
                ProceduralGraphError::InvalidReference(format!("nodedef '{definition_name}'"))
            })?;
        let mut bindings = BTreeMap::new();
        for interface in &implementation.inputs {
            let value = if let Some(override_input) = instance
                .inputs
                .iter()
                .find(|input| input.name == interface.name)
            {
                self.lower_input_node(caller, instance, override_input, caller_interfaces)?
            } else if interface.connection.is_some()
                || interface.explicit_default().is_some()
                || interface.attributes.contains_key("interfacename")
            {
                self.lower_interface(implementation, &interface.name, &bindings, definition)?
            } else {
                let declared = definition
                    .inputs
                    .iter()
                    .find(|input| input.name == interface.name)
                    .ok_or_else(|| {
                        ProceduralGraphError::InvalidReference(format!(
                            "interface '{}' is not declared by nodedef '{definition_name}'",
                            interface.name
                        ))
                    })?;
                let value = declared.explicit_default().ok_or_else(|| {
                    self.invalid_input(
                        instance,
                        &format!("external parameter '{}' is unbound", interface.name),
                    )
                })?;
                self.literal_node(instance, &interface.name, &value)?
            };
            bindings.insert(interface.name.clone(), value);
        }
        let output = selected_output
            .or_else(|| {
                (implementation.outputs.len() == 1).then(|| implementation.outputs[0].name.as_str())
            })
            .ok_or_else(|| {
                ProceduralGraphError::InvalidReference(format!(
                    "node '{}' has ambiguous graph output",
                    instance.name
                ))
            })?;
        self.lower_graph_output(implementation, output, &bindings)
    }

    fn lower_interface(
        &mut self,
        graph: &'a NodeGraph,
        name: &str,
        bindings: &BTreeMap<String, MaterialNodeId>,
        definition: &crate::NodeDef,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        if let Some(value) = bindings.get(name) {
            return Ok(*value);
        }
        let key = format!("{}::@interface::{name}", graph.name);
        if !self.active.insert(key.clone()) {
            return Err(ProceduralGraphError::Cycle(key));
        }
        let input = graph
            .inputs
            .iter()
            .find(|input| input.name == name)
            .ok_or_else(|| ProceduralGraphError::InvalidReference(format!("interface '{name}'")));
        let result = input.and_then(|input| {
            if let Some(alias) = input.attributes.get("interfacename") {
                return self.lower_interface(graph, alias, bindings, definition);
            }
            if let Some(connection) = input.connection.as_ref() {
                return self.lower_connection(graph, connection, bindings);
            }
            if let Some(value) = input.explicit_default() {
                return self.literal_node_from_name(name, &value);
            }
            let declared = definition
                .inputs
                .iter()
                .find(|candidate| candidate.name == name)
                .ok_or_else(|| {
                    ProceduralGraphError::InvalidReference(format!("interface '{name}'"))
                })?;
            let value = declared.explicit_default().ok_or_else(|| {
                ProceduralGraphError::InvalidReference(format!(
                    "unbound external parameter '{name}'"
                ))
            })?;
            self.literal_node_from_name(name, &value)
        });
        self.active.remove(&key);
        result
    }

    fn lower_node_value(
        &mut self,
        graph: &'a NodeGraph,
        node: &Node,
        interfaces: &BTreeMap<String, MaterialNodeId>,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        match node.category.as_str() {
            "constant" => node
                .value
                .as_ref()
                .ok_or_else(|| self.invalid_input(node, "value"))
                .and_then(|value| {
                    parse_material_value(value).map_err(|_| self.invalid_input(node, "value"))
                })
                .and_then(|value| self.insert(MaterialNode::Constant(value))),
            "texcoord" | "texcoord2" => self.insert(MaterialNode::Uv),
            "noise" => {
                let scale = self.literal_scalar(node, &["scale", "frequency"])?;
                let seed = self.literal_u32(node, "seed")?;
                self.insert(MaterialNode::Noise { scale, seed })
            }
            "curve" | "ramp" => {
                let input = self.required_input(node, &["in", "input", "value"])?;
                let points = node
                    .attributes
                    .get("points")
                    .cloned()
                    .or_else(|| {
                        self.input(node, &["points"])
                            .and_then(|input| input.explicit_default())
                            .map(|value| value.value)
                    })
                    .ok_or_else(|| self.invalid_input(node, "points"))?;
                let input_id = self.lower_input_node(graph, node, input, interfaces)?;
                self.insert(MaterialNode::Curve {
                    input: input_id,
                    points: parse_curve_points(&points)
                        .map_err(|_| self.invalid_input(node, "points"))?,
                })
            }
            "add" | "plus" => {
                let (a, b) = self.required_binary_inputs(node)?;
                let a = self.lower_input_node(graph, node, a, interfaces)?;
                let b = self.lower_input_node(graph, node, b, interfaces)?;
                self.insert(MaterialNode::Add(a, b))
            }
            "multiply" | "mult" => {
                let (a, b) = self.required_binary_inputs(node)?;
                let a = self.lower_input_node(graph, node, a, interfaces)?;
                let b = self.lower_input_node(graph, node, b, interfaces)?;
                self.insert(MaterialNode::Multiply(a, b))
            }
            "subtract" => {
                let (a, b) = self.required_binary_inputs(node)?;
                let a = self.lower_input_node(graph, node, a, interfaces)?;
                let b = self.lower_input_node(graph, node, b, interfaces)?;
                let negative = self.insert(MaterialNode::Constant(MaterialValue::Scalar(-1.0)))?;
                let negated = self.insert(MaterialNode::Multiply(b, negative))?;
                self.insert(MaterialNode::Add(a, negated))
            }
            "convert" => {
                let input = self.required_input(node, &["in", "input"])?;
                self.lower_input_node(graph, node, input, interfaces)
            }
            "mix" | "blend" => {
                let a = self.required_input(node, &["in1", "a", "left"])?;
                let b = self.required_input(node, &["in2", "b", "right"])?;
                let weight = self.required_input(node, &["mix", "weight", "factor"])?;
                let a = self.lower_input_node(graph, node, a, interfaces)?;
                let b = self.lower_input_node(graph, node, b, interfaces)?;
                let weight = self.lower_input_node(graph, node, weight, interfaces)?;
                self.insert(MaterialNode::Blend { a, b, weight })
            }
            category => Err(ProceduralGraphError::UnsupportedCategory {
                node: node.name.clone(),
                category: category.to_owned(),
            }),
        }
    }

    fn lower_input_node(
        &mut self,
        graph: &'a NodeGraph,
        node: &Node,
        input: &Input,
        interfaces: &BTreeMap<String, MaterialNodeId>,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        if let Some(connection) = &input.connection {
            return self.lower_connection(graph, connection, interfaces);
        }
        if let Some(interface) = input.attributes.get("interfacename") {
            return interfaces
                .get(interface)
                .copied()
                .ok_or_else(|| self.invalid_input(node, &input.name));
        }
        let value = input
            .explicit_default()
            .ok_or_else(|| self.invalid_input(node, &input.name))?;
        self.literal_node(node, &input.name, &value)
    }

    fn literal_node(
        &mut self,
        node: &Node,
        input: &str,
        value: &TypedValue,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        let material_value =
            parse_material_value(value).map_err(|_| self.invalid_input(node, input))?;
        self.insert(MaterialNode::Constant(material_value))
    }

    fn literal_node_from_name(
        &mut self,
        name: &str,
        value: &TypedValue,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        let material_value = parse_material_value(value).map_err(|_| {
            ProceduralGraphError::InvalidReference(format!("invalid interface '{name}'"))
        })?;
        self.insert(MaterialNode::Constant(material_value))
    }

    fn insert(&mut self, node: MaterialNode) -> Result<MaterialNodeId, ProceduralGraphError> {
        let next = self
            .next_id
            .checked_add(1)
            .ok_or(ProceduralGraphError::NodeIdExhausted)?;
        let id = MaterialNodeId(self.next_id);
        self.next_id = next;
        self.nodes.insert(id, node);
        Ok(id)
    }

    fn required_binary_inputs<'b>(
        &self,
        node: &'b Node,
    ) -> Result<(&'b Input, &'b Input), ProceduralGraphError> {
        let inputs = node
            .inputs
            .iter()
            .filter(|input| input.name != "points")
            .collect::<Vec<_>>();
        if inputs.len() != 2 {
            return Err(self.invalid_input(node, "binary inputs"));
        }
        Ok((inputs[0], inputs[1]))
    }

    fn required_input<'b>(
        &self,
        node: &'b Node,
        names: &[&str],
    ) -> Result<&'b Input, ProceduralGraphError> {
        self.input(node, names)
            .ok_or_else(|| self.invalid_input(node, names[0]))
    }

    fn input<'b>(&self, node: &'b Node, names: &[&str]) -> Option<&'b Input> {
        node.inputs
            .iter()
            .find(|input| names.iter().any(|name| *name == input.name))
    }

    fn literal_scalar(&self, node: &Node, names: &[&str]) -> Result<f32, ProceduralGraphError> {
        let input = self
            .input(node, names)
            .ok_or_else(|| self.invalid_input(node, names[0]))?;
        let value = input
            .explicit_default()
            .ok_or_else(|| self.invalid_input(node, &input.name))?;
        let parsed = parse_numbers(&value).map_err(|_| self.invalid_input(node, &input.name))?;
        if parsed.len() != 1 || !parsed[0].is_finite() {
            return Err(self.invalid_input(node, &input.name));
        }
        Ok(parsed[0])
    }

    fn literal_u32(&self, node: &Node, name: &str) -> Result<u32, ProceduralGraphError> {
        let input = self
            .input(node, &[name])
            .ok_or_else(|| self.invalid_input(node, name))?;
        let value = input
            .explicit_default()
            .ok_or_else(|| self.invalid_input(node, name))?;
        let parsed = parse_numbers(&value).map_err(|_| self.invalid_input(node, name))?;
        if parsed.len() != 1
            || !parsed[0].is_finite()
            || parsed[0] < 0.0
            || parsed[0].fract() != 0.0
        {
            return Err(self.invalid_input(node, name));
        }
        Ok(parsed[0] as u32)
    }

    fn invalid_input(&self, node: &Node, input: &str) -> ProceduralGraphError {
        ProceduralGraphError::InvalidInput {
            node: node.name.clone(),
            input: input.to_owned(),
        }
    }
}

fn parse_material_value(value: &TypedValue) -> Result<MaterialValue, ()> {
    if value.type_name == "boolean" || value.type_name == "bool" {
        return match value.value.trim() {
            "true" => Ok(MaterialValue::Boolean(true)),
            "false" => Ok(MaterialValue::Boolean(false)),
            _ => Err(()),
        };
    }
    let numbers = parse_numbers(value)?;
    match numbers.as_slice() {
        [value] => Ok(MaterialValue::Scalar(*value)),
        [a, b] => Ok(MaterialValue::Vector([*a, *b, 0.0, 1.0])),
        [a, b, c] => Ok(MaterialValue::Vector([*a, *b, *c, 1.0])),
        [a, b, c, d] => Ok(MaterialValue::Vector([*a, *b, *c, *d])),
        _ => Err(()),
    }
}

fn parse_numbers(value: &TypedValue) -> Result<Vec<f32>, ()> {
    let values = value
        .value
        .split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<f32>().map_err(|_| ()))
        .collect::<Result<Vec<_>, _>>()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(());
    }
    Ok(values)
}

fn parse_curve_points(source: &str) -> Result<Vec<[f32; 2]>, ()> {
    let mut points = Vec::new();
    for pair in source.split(';') {
        let values = pair
            .split(|character: char| {
                character == ',' || character == ':' || character.is_ascii_whitespace()
            })
            .filter(|part| !part.is_empty())
            .map(|part| part.parse::<f32>().map_err(|_| ()))
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != 2 || values.iter().any(|value| !value.is_finite()) {
            return Err(());
        }
        points.push([values[0], values[1]]);
    }
    if points.is_empty() {
        Err(())
    } else {
        Ok(points)
    }
}
