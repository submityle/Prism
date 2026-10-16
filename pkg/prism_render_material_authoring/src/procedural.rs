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
            Self::Bake(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProceduralGraphError {}

/// Lowers one bounded MaterialX procedural graph into the runtime material IR.
pub fn lower_procedural_graph(
    _document: &Document,
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

    let source_ids = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.name.clone(), MaterialNodeId(index as u32 + 1)))
        .collect::<BTreeMap<_, _>>();
    let next_synthetic = graph.nodes.len() as u32 + 1;
    let mut context = LowerContext {
        graph,
        source_ids,
        nodes: BTreeMap::new(),
        visiting: BTreeSet::new(),
        next_synthetic,
    };
    let output = context.lower_connection(
        outputs[0]
            .connection
            .as_ref()
            .expect("filtered connected output"),
    )?;
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
    graph: &'a NodeGraph,
    source_ids: BTreeMap<String, MaterialNodeId>,
    nodes: BTreeMap<MaterialNodeId, MaterialNode>,
    visiting: BTreeSet<String>,
    next_synthetic: u32,
}

impl<'a> LowerContext<'a> {
    fn lower_connection(
        &mut self,
        connection: &Connection,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        if let Some(graph) = connection.nodegraph.as_deref()
            && graph != self.graph.name
        {
            return Err(ProceduralGraphError::ExternalGraph(graph.to_owned()));
        }
        if let Some(node) = connection.node.as_deref() {
            return self.lower_node(node);
        }
        let output = connection
            .output
            .as_deref()
            .or_else(|| {
                (self.graph.outputs.len() == 1).then(|| self.graph.outputs[0].name.as_str())
            })
            .ok_or_else(|| {
                ProceduralGraphError::InvalidReference("graph output is ambiguous".into())
            })?;
        let graph_output = self
            .graph
            .outputs
            .iter()
            .find(|candidate| candidate.name == output)
            .ok_or_else(|| {
                ProceduralGraphError::InvalidReference(format!("graph output '{output}'"))
            })?;
        let source = graph_output.connection.as_ref().ok_or_else(|| {
            ProceduralGraphError::InvalidReference(format!("unconnected graph output '{output}'"))
        })?;
        self.lower_connection(source)
    }

    fn lower_node(&mut self, name: &str) -> Result<MaterialNodeId, ProceduralGraphError> {
        let id = *self
            .source_ids
            .get(name)
            .ok_or_else(|| ProceduralGraphError::InvalidReference(format!("node '{name}'")))?;
        if self.nodes.contains_key(&id) {
            return Ok(id);
        }
        if !self.visiting.insert(name.to_owned()) {
            return Err(ProceduralGraphError::Cycle(name.to_owned()));
        }
        let node = self
            .graph
            .nodes
            .iter()
            .find(|node| node.name == name)
            .expect("source_ids and graph nodes are built together")
            .clone();
        let lowered = self.lower_node_value(&node);
        self.visiting.remove(name);
        let lowered = lowered?;
        self.nodes.insert(id, lowered);
        Ok(id)
    }

    fn lower_node_value(&mut self, node: &Node) -> Result<MaterialNode, ProceduralGraphError> {
        match node.category.as_str() {
            "constant" => node
                .value
                .as_ref()
                .ok_or_else(|| self.invalid_input(node, "value"))
                .and_then(|value| {
                    parse_material_value(value).map_err(|_| self.invalid_input(node, "value"))
                })
                .map(MaterialNode::Constant),
            "texcoord" | "texcoord2" => Ok(MaterialNode::Uv),
            "noise" => {
                let scale = self.literal_scalar(node, &["scale", "frequency"])?;
                let seed = self.literal_u32(node, "seed")?;
                Ok(MaterialNode::Noise { scale, seed })
            }
            "curve" | "ramp" => {
                let input = self.required_input(node, &["in", "input", "value"])?;
                let points = node
                    .attributes
                    .get("points")
                    .or_else(|| {
                        self.input(node, &["points"])
                            .and_then(|input| input.value.as_ref())
                            .map(|value| &value.value)
                    })
                    .ok_or_else(|| self.invalid_input(node, "points"))?;
                Ok(MaterialNode::Curve {
                    input: self.lower_input_node(node, input)?,
                    points: parse_curve_points(points)
                        .map_err(|_| self.invalid_input(node, "points"))?,
                })
            }
            "add" | "plus" => {
                let (a, b) = self.required_binary_inputs(node)?;
                Ok(MaterialNode::Add(
                    self.lower_input_node(node, a)?,
                    self.lower_input_node(node, b)?,
                ))
            }
            "multiply" | "mult" => {
                let (a, b) = self.required_binary_inputs(node)?;
                Ok(MaterialNode::Multiply(
                    self.lower_input_node(node, a)?,
                    self.lower_input_node(node, b)?,
                ))
            }
            "mix" | "blend" => {
                let a = self.required_input(node, &["in1", "a", "left"])?;
                let b = self.required_input(node, &["in2", "b", "right"])?;
                let weight = self.required_input(node, &["mix", "weight", "factor"])?;
                Ok(MaterialNode::Blend {
                    a: self.lower_input_node(node, a)?,
                    b: self.lower_input_node(node, b)?,
                    weight: self.lower_input_node(node, weight)?,
                })
            }
            category => Err(ProceduralGraphError::UnsupportedCategory {
                node: node.name.clone(),
                category: category.to_owned(),
            }),
        }
    }

    fn lower_input_node(
        &mut self,
        node: &Node,
        input: &Input,
    ) -> Result<MaterialNodeId, ProceduralGraphError> {
        if let Some(connection) = &input.connection {
            return self.lower_connection(connection);
        }
        let value = input
            .value
            .as_ref()
            .ok_or_else(|| self.invalid_input(node, &input.name))?;
        let material_value =
            parse_material_value(value).map_err(|_| self.invalid_input(node, &input.name))?;
        let id = MaterialNodeId(self.next_synthetic);
        self.next_synthetic = self.next_synthetic.saturating_add(1);
        self.nodes
            .insert(id, MaterialNode::Constant(material_value));
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
        let Some(input) = self.input(node, names) else {
            return Ok(1.0);
        };
        let value = input
            .value
            .as_ref()
            .ok_or_else(|| self.invalid_input(node, &input.name))?;
        let parsed = parse_numbers(value).map_err(|_| self.invalid_input(node, &input.name))?;
        if parsed.len() != 1 || !parsed[0].is_finite() {
            return Err(self.invalid_input(node, &input.name));
        }
        Ok(parsed[0])
    }

    fn literal_u32(&self, node: &Node, name: &str) -> Result<u32, ProceduralGraphError> {
        let Some(input) = self.input(node, &[name]) else {
            return Ok(0);
        };
        let value = input
            .value
            .as_ref()
            .ok_or_else(|| self.invalid_input(node, name))?;
        let parsed = parse_numbers(value).map_err(|_| self.invalid_input(node, name))?;
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
