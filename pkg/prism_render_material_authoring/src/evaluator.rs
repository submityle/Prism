//! Deterministic evaluator for the small, pure MaterialX expression subset.
use std::collections::BTreeSet;

use crate::{AuthoringError, Connection, DiagnosticCode, Document, Node, NodeGraph, TypedValue};

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
    let target_graph = connection.nodegraph.as_deref().map_or(Ok(graph), |name| {
        document
            .nodegraphs
            .iter()
            .find(|g| g.name == name)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidReference,
                    format!("nodegraph '{name}' does not exist"),
                )
            })
    })?;
    let node_name = connection.node.as_deref().ok_or_else(|| {
        error(
            DiagnosticCode::UnsafeConnection,
            "graph-output connections cannot be evaluated by the expression evaluator",
        )
    })?;
    let mut stack = BTreeSet::new();
    eval_node(
        document,
        target_graph,
        node_name,
        connection.output.as_deref(),
        &mut stack,
    )
}

fn eval_node(
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
    let node = graph.nodes.iter().find(|n| n.name == name).ok_or_else(|| {
        error(
            DiagnosticCode::InvalidReference,
            format!("connected node '{name}' does not exist"),
        )
    })?;
    let result = eval_node_inner(document, graph, node, output, stack);
    stack.remove(&key);
    result
}

fn eval_node_inner(
    document: &Document,
    graph: &NodeGraph,
    node: &Node,
    output: Option<&str>,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
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
        } else {
            return Err(error(
                DiagnosticCode::InvalidReference,
                format!(
                    "output '{}' does not exist on node '{}'",
                    output.unwrap(),
                    node.name
                ),
            ));
        }
    }
    let mut values = Vec::with_capacity(node.inputs.len());
    for input in &node.inputs {
        let value = if let Some(literal) = &input.value {
            literal.clone()
        } else if let Some(connection) = &input.connection {
            evaluate_connection_inner(document, graph, connection, stack)?
        } else {
            return Err(error(
                DiagnosticCode::UnsafeConnection,
                format!("node '{}' input '{}' is unconnected", node.name, input.name),
            ));
        };
        values.push(value);
    }
    match node.category.as_str() {
        "convert" => values.into_iter().next().ok_or_else(|| {
            error(
                DiagnosticCode::InvalidStructure,
                "convert requires one input",
            )
        }),
        "add" | "subtract" | "multiply" | "divide" => binary(&node.category, &values),
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
            typed(
                &values[0].type_name,
                x.into_iter().map(|v| v.clamp(lo[0], hi[0])).collect(),
            )
        }
        _ => Err(error(
            DiagnosticCode::UnsafeConnection,
            format!(
                "node '{}' uses '{}' which requires a bake/runtime",
                node.name, node.category
            ),
        )),
    }
}

fn evaluate_connection_inner(
    document: &Document,
    graph: &NodeGraph,
    c: &Connection,
    stack: &mut BTreeSet<String>,
) -> Result<TypedValue, AuthoringError> {
    let target = c.nodegraph.as_deref().map_or(Ok(graph), |n| {
        document
            .nodegraphs
            .iter()
            .find(|g| g.name == n)
            .ok_or_else(|| {
                error(
                    DiagnosticCode::InvalidReference,
                    format!("nodegraph '{n}' does not exist"),
                )
            })
    })?;
    let name = c.node.as_deref().ok_or_else(|| {
        error(
            DiagnosticCode::UnsafeConnection,
            "graph output cannot be evaluated",
        )
    })?;
    eval_node(document, target, name, c.output.as_deref(), stack)
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
            .map(|v| {
                let mut s = format!("{v:.9}");
                while s.contains('.') && s.ends_with('0') {
                    s.pop();
                }
                if s.ends_with('.') {
                    s.pop();
                }
                s
            })
            .collect::<Vec<_>>()
            .join(", "),
    })
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
