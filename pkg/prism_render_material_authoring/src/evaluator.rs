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
    if let Some(selected) = output
        && selected != "out"
        && !matches!(
            node.category.as_str(),
            "separate2" | "separate3" | "separate4"
        )
    {
        return Err(error(
            DiagnosticCode::UnsafeConnection,
            format!(
                "node '{}' output '{}' has no executable output semantics",
                node.name, selected
            ),
        ));
    }
    if output.is_some_and(|name| name != "out") && node.inputs.is_empty() {
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
    evaluate_operator(&node.category, &values, output)
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
        if interface.explicit_default().is_none()
            && interface.connection.is_none()
            && !interface.attributes.contains_key("interfacename")
        {
            interface.value = definition
                .inputs
                .iter()
                .find(|declared| declared.name == interface.name)
                .and_then(crate::Input::explicit_default);
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
            declared.explicit_default().ok_or_else(|| {
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

    let mut result = evaluate_operator(&definition.category, &values, output)?;
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

fn evaluate_operator(
    category: &str,
    values: &[TypedValue],
    selected_output: Option<&str>,
) -> Result<TypedValue, AuthoringError> {
    let category = crate::author_node::canonical_category(category).unwrap_or(category);
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
        "add" | "subtract" | "multiply" | "divide" | "modulo" | "power" | "pow" | "max" | "min" => {
            binary(category, values)
        }
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
        "smoothstep" => smoothstep(values),
        "step" => step(values),
        "remap" => remap(values),
        "abs" => unary(values, f32::abs),
        "acos" => unary(values, f32::acos),
        "asin" => unary(values, f32::asin),
        "ceil" => unary(values, f32::ceil),
        "cos" => unary(values, f32::cos),
        "degrees" => unary(values, f32::to_degrees),
        "exp" => unary(values, f32::exp),
        "floor" => unary(values, f32::floor),
        "ln" => unary(values, f32::ln),
        "log" => unary(values, |value| value.log10()),
        "radians" => unary(values, f32::to_radians),
        "round" => unary(values, f32::round),
        "sign" => unary(values, f32::signum),
        "sin" => unary(values, f32::sin),
        "sqrt" => unary(values, f32::sqrt),
        "tan" => unary(values, f32::tan),
        "atan2" => binary(category, values),
        "dotproduct" => dotproduct(values),
        "crossproduct" => crossproduct(values),
        "normalize" => normalize(values),
        "magnitude" => magnitude(values),
        "distance" => distance(values),
        "combine2" | "combine3" | "combine4" => combine(values, category),
        "separate2" | "separate3" | "separate4" => separate(values, selected_output),
        "extract" => extract(values),
        "ifgreater" => conditional(values, |a, b| a > b),
        "ifequal" => conditional(values, |a, b| a == b),
        "switch" => switch_value(values),
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
    if let Some(value) = input.explicit_default() {
        return Ok(Some(value));
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
    let result = if let Some(value) = input.explicit_default() {
        Ok(value)
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
        .split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
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
    let width = a.len().max(b.len());
    if ![a.len(), b.len()]
        .iter()
        .all(|length| *length == 1 || *length == width)
    {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "arithmetic inputs have incompatible dimensions",
        ));
    }
    typed(
        &v[0].type_name,
        (0..width)
            .map(|index| {
                let x = a[if a.len() == 1 { 0 } else { index }];
                let y = b[if b.len() == 1 { 0 } else { index }];
                match op {
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
                    "modulo" => {
                        if y == 0.0 {
                            f32::NAN
                        } else {
                            x % y
                        }
                    }
                    "power" | "pow" => x.powf(y),
                    "max" => x.max(y),
                    "min" => x.min(y),
                    "atan2" => x.atan2(y),
                    _ => f32::NAN,
                }
            })
            .collect(),
    )
}

fn unary(
    values: &[TypedValue],
    operation: impl Fn(f32) -> f32,
) -> Result<TypedValue, AuthoringError> {
    if values.len() != 1 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "unary operator requires one input",
        ));
    }
    let input = numbers(&values[0])?;
    typed(
        &values[0].type_name,
        input.into_iter().map(operation).collect(),
    )
}

fn smoothstep(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 3 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "smoothstep requires low, high, and value",
        ));
    }
    let low = scalar_value(&values[0])?;
    let high = scalar_value(&values[1])?;
    if low >= high {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "smoothstep low bound must be lower than high bound",
        ));
    }
    let value = numbers(&values[2])?;
    typed(
        &values[2].type_name,
        value
            .into_iter()
            .map(|x| {
                let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            })
            .collect(),
    )
}

fn step(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 2 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "step requires edge and value",
        ));
    }
    let edge = scalar_value(&values[0])?;
    let input = numbers(&values[1])?;
    typed(
        &values[1].type_name,
        input
            .into_iter()
            .map(|value| if value < edge { 0.0 } else { 1.0 })
            .collect(),
    )
}

fn remap(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 5 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "remap requires input, in_low, in_high, out_low, and out_high",
        ));
    }
    let in_low = scalar_value(&values[1])?;
    let in_high = scalar_value(&values[2])?;
    let out_low = scalar_value(&values[3])?;
    let out_high = scalar_value(&values[4])?;
    if in_low == in_high {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "remap input range cannot have zero length",
        ));
    }
    typed(
        &values[0].type_name,
        numbers(&values[0])?
            .into_iter()
            .map(|value| out_low + (value - in_low) * (out_high - out_low) / (in_high - in_low))
            .collect(),
    )
}

fn scalar_value(value: &TypedValue) -> Result<f32, AuthoringError> {
    let values = numbers(value)?;
    if values.len() != 1 {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!("expected scalar, got type '{}'", value.type_name),
        ));
    }
    Ok(values[0])
}

fn dotproduct(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 2 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "dotproduct requires two inputs",
        ));
    }
    let a = numbers(&values[0])?;
    let b = numbers(&values[1])?;
    if a.len() != b.len() || a.is_empty() {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "dotproduct dimensions differ",
        ));
    }
    typed(
        "float",
        vec![a.into_iter().zip(b).map(|(x, y)| x * y).sum()],
    )
}

fn crossproduct(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 2 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "crossproduct requires two inputs",
        ));
    }
    let a = numbers(&values[0])?;
    let b = numbers(&values[1])?;
    if a.len() != 3 || b.len() != 3 {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "crossproduct requires vector3 inputs",
        ));
    }
    typed(
        "vector3",
        vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
    )
}

fn normalize(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 1 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "normalize requires one input",
        ));
    }
    let mut value = numbers(&values[0])?;
    let length = value.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length == 0.0 {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "cannot normalize a zero vector",
        ));
    }
    for component in &mut value {
        *component /= length;
    }
    typed(&values[0].type_name, value)
}

fn magnitude(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 1 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "magnitude requires one input",
        ));
    }
    let value = numbers(&values[0])?;
    typed(
        "float",
        vec![value.iter().map(|v| v * v).sum::<f32>().sqrt()],
    )
}

fn distance(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 2 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "distance requires two inputs",
        ));
    }
    let a = numbers(&values[0])?;
    let b = numbers(&values[1])?;
    if a.len() != b.len() {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "distance dimensions differ",
        ));
    }
    typed(
        "float",
        vec![a
            .into_iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()],
    )
}

fn combine(values: &[TypedValue], category: &str) -> Result<TypedValue, AuthoringError> {
    let expected = category
        .strip_prefix("combine")
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| error(DiagnosticCode::InvalidStructure, "invalid combine category"))?;
    if values.len() != expected
        || values
            .iter()
            .any(|value| numbers(value).map_or(true, |v| v.len() != 1))
    {
        return Err(error(
            DiagnosticCode::InvalidValue,
            format!("{category} requires {expected} scalar inputs"),
        ));
    }
    typed(
        &format!("vector{expected}"),
        values
            .iter()
            .map(scalar_value)
            .collect::<Result<Vec<_>, _>>()?,
    )
}

fn separate(
    values: &[TypedValue],
    selected_output: Option<&str>,
) -> Result<TypedValue, AuthoringError> {
    if values.len() != 1 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "separate requires one input",
        ));
    }
    let components = numbers(&values[0])?;
    let index = match selected_output {
        None => 0,
        Some("outx" | "x" | "out_x") => 0,
        Some("outy" | "y" | "out_y") => 1,
        Some("outz" | "z" | "out_z") => 2,
        Some("outw" | "w" | "out_w") => 3,
        Some(output) => {
            return Err(error(
                DiagnosticCode::InvalidReference,
                format!("separate output '{output}' is not a component output"),
            ));
        }
    };
    let value = components.get(index).copied().ok_or_else(|| {
        error(
            DiagnosticCode::InvalidReference,
            "separate output is outside the input dimension",
        )
    })?;
    typed("float", vec![value])
}

fn extract(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() != 2 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "extract requires value and index",
        ));
    }
    let components = numbers(&values[0])?;
    let index = scalar_value(&values[1])?;
    if index < 0.0 || index.fract() != 0.0 {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "extract index must be a non-negative integer",
        ));
    }
    typed(
        "float",
        vec![*components.get(index as usize).ok_or_else(|| {
            error(
                DiagnosticCode::InvalidValue,
                "extract index is outside the input dimension",
            )
        })?],
    )
}

fn conditional(
    values: &[TypedValue],
    predicate: impl Fn(f32, f32) -> bool,
) -> Result<TypedValue, AuthoringError> {
    if values.len() != 4 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "conditional requires comparison and two branches",
        ));
    }
    let a = scalar_value(&values[0])?;
    let b = scalar_value(&values[1])?;
    Ok(if predicate(a, b) {
        values[2].clone()
    } else {
        values[3].clone()
    })
}

fn switch_value(values: &[TypedValue]) -> Result<TypedValue, AuthoringError> {
    if values.len() < 2 {
        return Err(error(
            DiagnosticCode::InvalidStructure,
            "switch requires an index and at least one value",
        ));
    }
    let index = scalar_value(&values[0])?;
    if index < 0.0 || index.fract() != 0.0 {
        return Err(error(
            DiagnosticCode::InvalidValue,
            "switch index must be a non-negative integer",
        ));
    }
    values.get(index as usize + 1).cloned().ok_or_else(|| {
        error(
            DiagnosticCode::InvalidValue,
            "switch index is outside its inputs",
        )
    })
}
fn error(code: DiagnosticCode, message: impl Into<String>) -> AuthoringError {
    AuthoringError::new(code, message, None)
}
