//! 静态校验：方向 / exec-data 一致 / 类型兼容 / Wildcard / Pure 子图无环。

use super::context::LowerCtx;
use crate::error::{VsError, VsResult};
use crate::model::{Edge, NodeId, Pin, PinDir, PinIndex};
use crate::registry::{Purity, RunnerKind};
use crate::value::ValueType;
use std::collections::HashMap;

#[derive(Copy, Clone, PartialEq)]
enum Mark {
    New,
    Visiting,
    Done,
}

pub(super) fn run(cx: &LowerCtx<'_>) -> VsResult<()> {
    for (&node_id, control) in &cx.graph.controls {
        let def = cx
            .defs_by_node
            .get(&node_id)
            .ok_or(VsError::UnknownNode(node_id))?;
        if def.purity != Purity::Impure {
            return Err(VsError::Validate(format!(
                "control node {node_id} must be impure"
            )));
        }
        if let crate::model::ControlNode::Fork {
            branches,
            max_concurrency,
            ..
        } = control
        {
            if branches.is_empty() {
                return Err(VsError::Validate(format!(
                    "fork node {node_id} has no branches"
                )));
            }
            if *max_concurrency == 0 {
                return Err(VsError::Validate(format!(
                    "fork node {node_id} max_concurrency must be greater than zero"
                )));
            }
        }
    }
    for e in &cx.graph.edges {
        validate_edge(cx, e)?;
    }
    for n in &cx.graph.nodes {
        let def = &cx.defs_by_node[&n.id];
        if let RunnerKind::Effect(spec) = &def.runner {
            if def.purity != Purity::Impure {
                return Err(VsError::Validate(format!(
                    "effect node {} ({}) must be impure",
                    n.id, def.id
                )));
            }
            spec.validate().map_err(|error| {
                VsError::Validate(format!("effect node {} ({}): {error}", n.id, def.id))
            })?;
        }
        for pin in def.inputs.iter().chain(def.outputs.iter()) {
            if matches!(pin.ty, ValueType::Wildcard) {
                return Err(VsError::Validate(format!(
                    "node {} ({}): pin '{}' has unresolved Wildcard type",
                    n.id, def.id, pin.name
                )));
            }
        }
    }
    check_pure_acyclic(cx)
}

fn validate_edge(cx: &LowerCtx<'_>, e: &Edge) -> VsResult<()> {
    let from_def = cx
        .defs_by_node
        .get(&e.from.node)
        .ok_or(VsError::UnknownNode(e.from.node))?;
    let to_def = cx
        .defs_by_node
        .get(&e.to.node)
        .ok_or(VsError::UnknownNode(e.to.node))?;
    let from_pin = pin_at(&from_def.outputs, e.from.pin, "output")?;
    let to_pin = pin_at(&to_def.inputs, e.to.pin, "input")?;

    if from_pin.dir != PinDir::Out {
        return Err(VsError::Validate(format!(
            "edge.from must be Out: node {}, pin '{}'",
            e.from.node, from_pin.name
        )));
    }
    if to_pin.dir != PinDir::In {
        return Err(VsError::Validate(format!(
            "edge.to must be In: node {}, pin '{}'",
            e.to.node, to_pin.name
        )));
    }
    if from_pin.is_exec() != to_pin.is_exec() {
        return Err(VsError::Validate(format!(
            "exec/data mismatch: from='{}' to='{}'",
            from_pin.name, to_pin.name
        )));
    }
    if !from_pin.is_exec() && !to_pin.ty.accepts(&from_pin.ty) {
        return Err(VsError::TypeMismatch {
            expected: to_pin.ty.clone(),
            got: from_pin.ty.clone(),
            location: format!(
                "edge ({},{}) -> ({},{})",
                e.from.node, from_pin.name, e.to.node, to_pin.name
            ),
        });
    }
    Ok(())
}

fn check_pure_acyclic(cx: &LowerCtx<'_>) -> VsResult<()> {
    let mut mark: HashMap<NodeId, Mark> = HashMap::new();
    for n in &cx.graph.nodes {
        mark.insert(n.id, Mark::New);
    }
    for n in &cx.graph.nodes {
        if cx.defs_by_node[&n.id].purity == Purity::Pure {
            dfs_pure(cx, n.id, &mut mark)?;
        }
    }
    Ok(())
}

fn dfs_pure(cx: &LowerCtx<'_>, node: NodeId, mark: &mut HashMap<NodeId, Mark>) -> VsResult<()> {
    match mark[&node] {
        Mark::Done => return Ok(()),
        Mark::Visiting => return Err(VsError::Cycle(node)),
        Mark::New => {}
    }
    mark.insert(node, Mark::Visiting);
    let def = &cx.defs_by_node[&node];
    for (i, p) in def.inputs.iter().enumerate() {
        if p.is_exec() {
            continue;
        }
        if let Some(src) = cx.incoming_by_endpoint.get(&(node, i as PinIndex))
            && cx.defs_by_node[&src.node].purity == Purity::Pure
        {
            dfs_pure(cx, src.node, mark)?;
        }
    }
    mark.insert(node, Mark::Done);
    Ok(())
}

fn pin_at<'p>(pins: &'p [Pin], idx: PinIndex, side: &str) -> VsResult<&'p Pin> {
    pins.get(idx as usize)
        .ok_or_else(|| VsError::Validate(format!("{side} pin index {} out of range", idx)))
}
