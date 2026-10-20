//! Deterministic evaluator for the small, pure MaterialX expression subset.
use std::collections::BTreeSet;

use crate::{AuthoringError, Connection, DiagnosticCode, Document, Node, NodeGraph, TypedValue};

pub(crate) struct EvaluationIndex<'a> {
    graphs: std::collections::BTreeMap<&'a str, &'a NodeGraph>,
    nodes: std::collections::BTreeMap<(&'a str, &'a str), &'a Node>,
    implementations: std::collections::BTreeMap<&'a str, &'a NodeGraph>,
}

impl<'a> EvaluationIndex<'a> {
    pub(crate) fn new(document: &'a Document) -> Self {
        let mut index = Self {
            graphs: std::collections::BTreeMap::new(),
            nodes: std::collections::BTreeMap::new(),
            implementations: std::collections::BTreeMap::new(),
        };
        for graph in &document.nodegraphs {
            index.graphs.insert(&graph.name, graph);
            for node in &graph.nodes {
                index.nodes.insert((&graph.name, &node.name), node);
            }
            if let Some(nodedef) = graph.nodedef.as_deref() {
                index.implementations.insert(nodedef, graph);
            }
        }
        index
    }
}

/// Evaluate a node connection using only pure numeric graph operators.
///
/// This is intentionally a bounded authoring evaluator, not a general MaterialX
/// runtime: texture and procedural nodes remain explicit `UnsafeConnection`
/// errors rather than silently becoming defaults.
pub fn evaluate_connection(
    document: &Document,
    graph: &NodeGraph,
    connection: &Connection,
) -> Result<TypedValue, AuthoringError> {
    let index = EvaluationIndex::new(document);
    let target_graph = connection.nodegraph.as_deref().map_or(Ok(graph), |name| {
        index.graphs.get(name).copied().ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("nodegraph '{name}' does not exist"),
            )
        })
    })?;
    let mut stack = BTreeSet::new();
    evaluate_connection_inner(&index, document, target_graph, connection, &mut stack)
}

/// Resolves one node input using the caller's per-document index. `None` means
/// that the input is omitted. Batch adapters build the index once and reuse it
/// across all inputs/nodes in the evaluation.
pub(crate) fn evaluate_input_indexed(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    input: &crate::Input,
) -> Result<Option<TypedValue>, AuthoringError> {
    let mut stack = BTreeSet::new();
    evaluate_input_inner(index, document, graph, input, &mut stack)
}

fn eval_node(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    name: &str,
    output: Option<&str>,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    let key = format!("{}::{name}", graph.name);
    if !stack.insert(key.clone()) {
        return Err(error(
            DiagnosticCode::UnsafeConnection,
            format!("cyclic graph expression at node '{name}'"),
        ));
    }
    let node = index
        .nodes
        .get(&(graph.name.as_str(), name))
        .copied()
        .or_else(|| graph.nodes.iter().find(|n| n.name == name))
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("connected node '{name}' does not exist"),
            )
        })?;
    let result = eval_node_inner(index, document, graph, node, output, stack);
    stack.remove(&key);
    result
}

fn eval_node_inner(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    node: &Node,
    output: Option<&str>,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    if let Some(definition_name) = node.attributes.get("nodedef")
        && let Some(implementation) = index.implementations.get(definition_name.as_str()).copied()
    {
        return eval_graph_instance(index, document, graph, node, implementation, output, stack);
    }
    if let Some((definition, _)) = crate::author_node::resolve(document, node)? {
        return eval_author_node(index, document, graph, node, definition, output, stack);
    }
    if let Some(value) = &node.value {
        if output.is_some_and(|name| name != "out") {
            return Err(error(
                DiagnosticCode::UnsafeConnection,
                format!("node '{}' has no output '{}'", node.name, output.unwrap()),
            ));
        }
        return Ok(value.clone());
    }
    if output.is_some_and(|name| name != "out") {
        if let Some(declared) = node
            .outputs
            .iter()
            .find(|o| Some(o.name.as_str()) == output)
        {
            if let Some(value) = &declared.value {
                return Ok(value.clone());
            }
            return Err(error(
                DiagnosticCode::UnsafeConnection,
                format!(
                    "node '{}' output '{}' has no evaluable literal",
                    node.name,
                    output.unwrap()
                ),
            ));
        }
        return Err(error(
            DiagnosticCode::InvalidReference,
            format!(
                "output '{}' does not exist on node '{}'",
                output.unwrap(),
                node.name
            ),
        ));
    }
    let mut values = Vec::with_capacity(node.inputs.len());
    for input in &node.inputs {
        let Some(value) = evaluate_input_inner(index, document, graph, input, stack)? else {
            return Err(error(
                DiagnosticCode::UnsafeConnection,
                format!("node '{}' input '{}' is unconnected", node.name, input.name),
            ));
        };
        values.push(value);
    }
    evaluate_operator(&node.category, &values)
}

fn eval_graph_instance(
    index: &EvaluationIndex<'_>,
    document: &Document,
    caller: &NodeGraph,
    instance: &Node,
    implementation: &NodeGraph,
    selected_output: Option<&str>,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    let definition_name = implementation
        .nodedef
        .as_deref()
        .expect("matched graph nodedef");
    let definition = document
        .nodedefs
        .binary_search_by(|definition| definition.name.as_str().cmp(definition_name))
        .ok()
        .map(|pos| &document.nodedefs[pos])
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!(
                    "nodegraph '{}' references unknown nodedef '{definition_name}'",
                    implementation.name
                ),
            )
        })?;

    let mut effective = implementation.clone();
    for interface in &mut effective.inputs {
        if interface.value.is_none()
            && interface.connection.is_none()
            && !interface.attributes.contains_key("interfacename")
        {
            interface.value = definition
                .inputs
                .iter()
                .find(|declared| declared.name == interface.name)
                .and_then(|declared| declared.value.clone());
        }
        if let Some(override_input) = instance
            .inputs
            .iter()
            .find(|input| input.name == interface.name)
        {
            let mut value = evaluate_input_inner(index, document, caller, override_input, stack)?
                .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidStructure,
                    format!(
                        "node '{}' override '{}' has no value",
                        instance.name, interface.name
                    ),
                )
            })?;
            if let Some(expected) = interface.type_name.as_deref() {
                value = coerce_value(
                    value,
                    expected,
                    &format!("node '{}' input '{}'", instance.name, interface.name),
                    override_input.type_name.is_none()
                        && override_input
                            .value
                            .as_ref()
                            .is_some_and(|value| value.type_name == "string"),
                )?;
            }
            interface.type_name = interface
                .type_name
                .clone()
                .or_else(|| override_input.type_name.clone());
            interface.value = Some(value);
            interface.connection = None;
        }
    }
    for input in &instance.inputs {
        if !effective
            .inputs
            .iter()
            .any(|interface| interface.name == input.name)
        {
            return Err(error(
                DiagnosticCode::InvalidStructure,
                format!(
                    "node '{}' input '{}' is not declared by nodegraph '{}'",
                    instance.name, input.name, implementation.name
                ),
            ));
        }
    }

    let output_name = match selected_output {
        Some(name) => name,
        None if effective.outputs.len() == 1 => effective.outputs[0].name.as_str(),
        None => {
            return Err(error(
                DiagnosticCode::InvalidReference,
                format!(
                    "node '{}' references nodegraph '{}' without selecting one of its {} outputs",
                    instance.name,
                    implementation.name,
                    effective.outputs.len()
                ),
            ))
        }
    };
    let declared_output = definition
        .outputs
        .iter()
        .find(|output| output.name == output_name)
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!(
                    "output '{output_name}' is not declared by nodedef '{}'",
                    definition.name
                ),
            )
        })?;
    let value = eval_graph_output(index, document, &effective, output_name, stack)?;
    declared_output
        .type_name
        .as_deref()
        .map_or(Ok(value.clone()), |expected| {
            coerce_value(
                value,
                expected,
                &format!("node '{}' output '{}'", instance.name, output_name),
                false,
            )
        })
}

fn eval_author_node(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    node: &Node,
    definition: &crate::NodeDef,
    output: Option<&str>,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    let declared_output = match output {
        Some(name) => definition
            .outputs
            .iter()
            .find(|candidate| candidate.name == name)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidReference,
                    format!(
                        "output '{name}' does not exist on nodedef '{}'",
                        definition.name
                    ),
                )
            })?,
        None => definition.outputs.first().ok_or_else(|| {
            error(
                DiagnosticCode::InvalidStructure,
                format!("nodedef '{}' has no output", definition.name),
            )
        })?,
    };

    let mut values = Vec::with_capacity(definition.inputs.len());
    for declared in &definition.inputs {
        let instance = node
            .inputs
            .iter()
            .find(|candidate| candidate.name == declared.name);
        let allow_untyped = instance.is_some_and(|input| {
            input.type_name.is_none()
                && input
                    .value
                    .as_ref()
                    .is_some_and(|value| value.type_name == "string")
        });
        let mut value = if let Some(instance) = instance {
            evaluate_input_inner(index, document, graph, instance, stack)?.ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidStructure,
                    format!(
                        "node '{}' input '{}' has no value or connection",
                        node.name, instance.name
                    ),
                )
            })?
        } else {
            declared.value.clone().ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidStructure,
                    format!(
                        "node '{}' omits required nodedef input '{}'",
                        node.name, declared.name
                    ),
                )
            })?
        };
        let declared_type = declared
            .type_name
            .as_deref()
            .expect("parsed nodedef inputs always carry a type");
        value = coerce_value(
            value,
            declared_type,
            &format!("node '{}' input '{}'", node.name, declared.name),
            allow_untyped,
        )?;
        values.push(value);
    }

    let mut result = evaluate_operator(&definition.category, &values)?;
    let output_type = declared_output
        .type_name
        .as_deref()
        .expect("parsed nodedef outputs always carry a type");
    if definition.category == "convert" {
        let input_width = values
            .first()
            .and_then(|value| crate::author_node::numeric_width(&value.type_name));
        let output_width = crate::author_node::numeric_width(output_type);
        if input_width.is_none() || output_width.is_none() || input_width != output_width {
            return Err(error(
                DiagnosticCode::InvalidValue,
                format!(
                    "nodedef '{}' convert requires supported numeric input/output types with the same dimension",
                    definition.name
                ),
            ));
        }
    } else if result.type_name != output_type {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!(
                "nodedef '{}' evaluated type '{}', expected output type '{}'",
                definition.name, result.type_name, output_type
            ),
        ));
    }
    result.type_name = output_type.to_owned();
    Ok(result)
}

fn evaluate_operator(category: &str, values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    match category {
        "convert" => values
            .into_iter()
            .next()
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidStructure,
                    "convert requires one input",
                )
            })
            .and_then(|value| typed(&value.type_name, numbers(value)?)),
        "add" | "subtract" | "multiply" | "divide" => binary(category, values),
        "mix" => {
            if values.len() != 3 {
                return Err(error(
                    DiagnosticCode::InvalidStructure,
                    "mix requires three inputs",
                ));
            }
            let t = numbers(&values[2])?;
            let a = numbers(&values[0])?;
            let b = numbers(&values[1])?;
            if t.len() != 1 || a.len() != b.len() {
                return Err(error(
                    DiagnosticCode::InvalidValue,
                    "mix inputs have incompatible dimensions",
                ));
            }
            typed(
                &values[0].type_name,
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| x + (y - x) * t[0])
                    .collect(),
            )
        }
        "clamp" => {
            if values.len() != 3 {
                return Err(error(
                    DiagnosticCode::InvalidStructure,
                    "clamp requires value, low, and high",
                ));
            }
            let x = numbers(&values[0])?;
            let lo = numbers(&values[1])?;
            let hi = numbers(&values[2])?;
            if lo.len() != 1 || hi.len() != 1 {
                return Err(error(
                    DiagnosticCode::InvalidValue,
                    "clamp bounds must be scalar",
                ));
            }
            if lo[0] > hi[0] {
                return Err(error(
                    DiagnosticCode::InvalidValue,
                    "clamp lower bound exceeds upper bound",
                ));
            }
            typed(
                &values[0].type_name,
                x.into_iter().map(|v| v.clamp(lo[0], hi[0])).collect(),
            )
        }
        _ => Err(error(
            DiagnosticCode::UnsafeConnection,
            format!("node category '{category}' requires a bake/runtime"),
        )),
    }
}

fn evaluate_connection_inner(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    c: &Connection,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    let target = c.nodegraph.as_deref().map_or(Ok(graph), |n| {
        index.graphs.get(n).copied().ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!("nodegraph '{n}' does not exist"),
            )
        })
    })?;
    if let Some(name) = c.node.as_deref() {
        return eval_node(index, document, target, name, c.output.as_deref(), stack);
    }
    let output = c
        .output
        .as_deref()
        .or_else(|| (target.outputs.len() == 1).then(|| target.outputs[0].name.as_str()));
    let output = output.ok_or_else(|| {
        error(
            DiagnosticCode::InvalidReference,
            format!(
                "connection to nodegraph '{}' does not select one of its graph outputs",
                target.name
            ),
        )
    })?;
    eval_graph_output(index, document, target, output, stack)
}

fn evaluate_input_inner(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    input: &crate::Input,
    stack: &mut BTreeSet<String>,
) -> Result<Option<TypedValue>, AuthoringError> {
    if let Some(value) = &input.value {
        return Ok(Some(value.clone()));
    }
    if let Some(connection) = &input.connection {
        return evaluate_connection_inner(index, document, graph, connection, stack).map(Some);
    }
    if let Some(interface) = input.attributes.get("interfacename") {
        return evaluate_interface(index, document, graph, interface, stack).map(Some);
    }
    Ok(None)
}

fn evaluate_interface(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    name: &str,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    let key = format!("{}::@interface::{name}", graph.name);
    if !stack.insert(key.clone()) {
        return Err(error(
            DiagnosticCode::UnsafeConnection,
            format!("cyclic graph interface resolution at '{name}'"),
        ));
    }
    let input = graph
        .inputs
        .iter()
        .find(|input| input.name == name)
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!(
                    "interface input '{name}' does not exist in nodegraph '{}'",
                    graph.name
                ),
            )
        })?;
    let result = if let Some(value) = &input.value {
        Ok(value.clone())
    } else if let Some(connection) = &input.connection {
        evaluate_connection_inner(index, document, graph, connection, stack)
    } else if let Some(alias) = input.attributes.get("interfacename") {
        evaluate_interface(index, document, graph, alias, stack)
    } else {
        Err(error(
            DiagnosticCode::UnsafeConnection,
            format!("graph interface input '{name}' has no value or connection"),
        ))
    };
    let result = result.and_then(|value| {
        input
            .type_name
            .as_deref()
            .map_or(Ok(value.clone()), |expected| {
                coerce_value(value, expected, &format!("interface input '{name}'"), false)
            })
    });
    stack.remove(&key);
    result
}

fn eval_graph_output(
    index: &EvaluationIndex<'_>,
    document: &Document,
    graph: &NodeGraph,
    name: &str,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    let key = format!("{}::@output::{name}", graph.name);
    if !stack.insert(key.clone()) {
        return Err(error(
            DiagnosticCode::UnsafeConnection,
            format!("cyclic graph output resolution at '{name}'"),
        ));
    }
    let output = graph
        .outputs
        .iter()
        .find(|output| output.name == name)
        .ok_or_else(|| {
            error(
                DiagnosticCode::InvalidReference,
                format!(
                    "output '{name}' does not exist on nodegraph '{}'",
                    graph.name
                ),
            )
        })?;
    let result = if let Some(value) = &output.value {
        Ok(value.clone())
    } else if let Some(connection) = &output.connection {
        evaluate_connection_inner(index, document, graph, connection, stack)
    } else {
        Err(error(
            DiagnosticCode::UnsafeConnection,
            format!(
                "graph output '{name}' in nodegraph '{}' is unconnected",
                graph.name
            ),
        ))
    };
    let result = result.and_then(|value| {
        output
            .type_name
            .as_deref()
            .map_or(Ok(value.clone()), |expected| {
                coerce_value(value, expected, &format!("graph output '{name}'"), false)
            })
    });
    stack.remove(&key);
    result
}
fn numbers(v: &TypedValue) -> Result<Vec<f32>, AuthoringError> {
    v.value
        .split(',')
        .map(|s| {
            s.trim().parse::<f32>().map_err(|_| {
                error(
                    DiagnosticCode::InvalidValue,
                    format!("invalid numeric literal '{}'", v.value),
                )
            })
        })
        .collect()
}

fn coerce_value(
    value: TypedValue,
    expected: &str,
    context: &str,
    allow_untyped: bool,
) -> Result<TypedValue, AuthoringError> {
    if value.type_name == expected && expected == "string" {
        return Ok(value);
    }
    let untyped = value.type_name == "string";
    if untyped && !allow_untyped {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!("{context} has an untyped string value; expected '{expected}'"),
        ));
    }
    if !untyped && value.type_name != expected {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!(
                "{context} evaluated to type '{}', expected '{expected}'",
                value.type_name
            ),
        ));
    }

    if let Some(width) = crate::author_node::numeric_width(expected) {
        let components = numbers(&value)?;
        if components.len() != width {
            return Err(error(
                DiagnosticCode::InvalidValue,
                format!(
                    "{context} has {} numeric components, expected {width} for '{expected}'",
                    components.len()
                ),
            ));
        }
        return typed(expected, components);
    }
    if expected == "boolean" {
        if matches!(value.value.as_str(), "true" | "false") {
            return Ok(TypedValue {
                type_name: expected.to_owned(),
                value: value.value,
            });
        }
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!("{context} has invalid boolean literal '{}'", value.value),
        ));
    }
    if !untyped {
        return Ok(value);
    }
    Err(error(
        DiagnosticCode::InvalidValue,
        format!("{context} is an untyped string and cannot be safely coerced to '{expected}'"),
    ))
}

fn typed(name: &str, values: Vec<f32>) -> Result<TypedValue, AuthoringError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "expression produced a non-finite value",
        ));
    }
    Ok(TypedValue {
        type_name: name.to_owned(),
        value: values
            .iter()
            .map(|value| canonical_number(*value))
            .collect::<Vec<_>>()
            .join(", "),
    })
}

fn canonical_number(value: f32) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }

    // Authoring literals carry the meaningful decimal precision of an f32,
    // not the incidental tail introduced by one arithmetic operation. Round
    // to seven significant decimal digits before using the shortest decimal
    // representation. This keeps small/large magnitudes while canonicalizing
    // results such as `0.3 + 0.4 == 0.70000005f32` back to `0.7`.
    let exponent = value.abs().log10().floor() as i32;
    let shift = 6 - exponent;
    let scale = 10_f64.powi(shift.unsigned_abs() as i32);
    let rounded = if shift >= 0 {
        ((value as f64) * scale).round() / scale
    } else {
        ((value as f64) / scale).round() * scale
    };
    if rounded == 0.0 {
        "0".to_owned()
    } else {
        rounded.to_string()
    }
}
fn binary(op: &str, v: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if v.len() != 2 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            format!("{op} requires two inputs"),
        ));
    }
    let a = numbers(&v[0])?;
    let b = numbers(&v[1])?;
    if a.len() != b.len() {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "arithmetic inputs have incompatible dimensions",
        ));
    }
    typed(
        &v[0].type_name,
        a.into_iter()
            .zip(b)
            .map(|(x, y)| match op {
                "add" => x + y,
                "subtract" => x - y,
                "multiply" => x * y,
                "divide" => {
                    if y == 0.0 {
                        f32::NAN
                    } else {
                        x / y
                    }
                }
                _ => unreachable!(),
            })
            .collect(),
    )
}
fn error(code: DiagnosticCode, message: impl Into<String>) -> AuthoringError {
    AuthoringError::new(code, message, None)
}
