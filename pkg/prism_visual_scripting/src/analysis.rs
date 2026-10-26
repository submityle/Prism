//! Static safety analysis for workflow graphs.
//!
//! The compiler still accepts valid cyclic control flow (loops are a runtime
//! concern), but this pass makes the safety proof explicit and machine-readable.

use crate::effect::{EffectClass, IdempotencyPolicy};
use crate::model::{ControlNode, Graph, NodeId};
use crate::registry::{NodeRegistry, Purity, RunnerKind};
use crate::{Value, VsResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    Warning,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticFinding {
    pub severity: FindingSeverity,
    pub code: String,
    pub node_ids: Vec<NodeId>,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticBounds {
    pub node_count: usize,
    pub effect_count: usize,
    pub wait_count: usize,
    pub max_parallelism: usize,
    pub max_fork_branches: usize,
    pub max_call_depth: Option<usize>,
    pub cost_units: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticModelReport {
    pub graph: String,
    pub bounds: StaticBounds,
    pub findings: Vec<StaticFinding>,
}

impl StaticModelReport {
    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == FindingSeverity::Error)
    }
    pub fn errors(&self) -> impl Iterator<Item = &StaticFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Error)
    }
}

/// Analyze one graph against the same registry used by compilation.
pub fn analyze_graph(graph: &Graph, registry: &dyn NodeRegistry) -> VsResult<StaticModelReport> {
    let mut defs = HashMap::new();
    for node in &graph.nodes {
        defs.insert(node.id, registry.resolve(node)?);
    }
    let mut report = StaticModelReport {
        graph: graph.name.clone(),
        bounds: StaticBounds {
            node_count: graph.nodes.len(),
            ..Default::default()
        },
        findings: Vec::new(),
    };
    let outgoing = graph.outgoing();
    let mut exec_edges: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for edge in &graph.edges {
        let Some(from) = defs.get(&edge.from.node) else {
            continue;
        };
        let Some(to) = defs.get(&edge.to.node) else {
            continue;
        };
        if from
            .outputs
            .get(edge.from.pin as usize)
            .is_some_and(|p| p.is_exec())
            && to
                .inputs
                .get(edge.to.pin as usize)
                .is_some_and(|p| p.is_exec())
        {
            exec_edges
                .entry(edge.from.node)
                .or_default()
                .push(edge.to.node);
        }
    }

    for node in &graph.nodes {
        let def = &defs[&node.id];
        if let RunnerKind::Effect(spec) = &def.runner {
            report.bounds.effect_count += 1;
            report.bounds.cost_units = report
                .bounds
                .cost_units
                .saturating_add(1 + usize::from(spec.retry.max_attempts));
            if spec.class == EffectClass::ExternalWrite
                && spec.idempotency != IdempotencyPolicy::Required
            {
                report.findings.push(finding(
                    FindingSeverity::Error,
                    "NON_IDEMPOTENT_RETRY",
                    node.id,
                    "external write can be retried without a required idempotency key",
                ));
            }
            if spec.retry.max_attempts > 1 && spec.timeout_ms == 0 {
                report.findings.push(finding(
                    FindingSeverity::Error,
                    "RETRY_WITHOUT_TIMEOUT",
                    node.id,
                    "retry policy has no bounded timeout",
                ));
            }
        }
        if def.id.starts_with("await.") || def.id.starts_with("wait.") || def.id.contains("suspend")
        {
            report.bounds.wait_count += 1;
            let has_timeout = node
                .config
                .get("timeout_ms")
                .and_then(Value::as_i64)
                .is_some_and(|v| v > 0)
                || node
                    .config
                    .get("deadline_ms")
                    .and_then(Value::as_i64)
                    .is_some();
            if !has_timeout {
                report.findings.push(finding(FindingSeverity::Warning, "UNBOUNDED_WAIT", node.id, "wait/suspension has no statically visible timeout; liveness depends on an external actor"));
            }
        }
        if def.purity == Purity::Impure {
            report.bounds.cost_units = report.bounds.cost_units.saturating_add(1);
        }
    }

    for (&id, control) in &graph.controls {
        match control {
            ControlNode::Fork {
                branches,
                max_concurrency,
                ..
            } => {
                report.bounds.max_fork_branches =
                    report.bounds.max_fork_branches.max(branches.len());
                report.bounds.max_parallelism = report
                    .bounds
                    .max_parallelism
                    .saturating_add(usize::from(*max_concurrency).min(branches.len()));
                if branches.is_empty() || *max_concurrency == 0 {
                    report.findings.push(finding(
                        FindingSeverity::Error,
                        "UNBOUNDED_FORK",
                        id,
                        "fork has no executable branch or no positive concurrency bound",
                    ));
                }
                if branches.iter().any(|call| call.graph == graph.name) {
                    report.findings.push(finding(FindingSeverity::Error, "RECURSIVE_FORK", id, "fork recursively invokes the current graph; no finite task bound can be proven"));
                }
            }
            ControlNode::Call { target } if target.graph == graph.name => {
                report.bounds.max_call_depth = None;
                report.findings.push(finding(
                    FindingSeverity::Error,
                    "RECURSIVE_CALL",
                    id,
                    "graph call is recursive and has no declared depth bound",
                ));
            }
            _ => {}
        }
    }
    report.bounds.max_call_depth.get_or_insert(1);
    report.bounds.max_parallelism = report.bounds.max_parallelism.max(1);
    for component in strongly_connected_components(&exec_edges, graph.nodes.iter().map(|n| n.id)) {
        let cyclic = component.len() > 1
            || component
                .first()
                .is_some_and(|id| exec_edges.get(id).is_some_and(|xs| xs.contains(id)));
        if cyclic {
            let has_exit = component.iter().any(|id| {
                outgoing.get(id).is_some_and(|edges| {
                    edges
                        .iter()
                        .any(|e| !exec_edges.get(id).is_some_and(|xs| xs.contains(&e.to.node)))
                })
            });
            if !has_exit {
                report.findings.push(StaticFinding { severity: FindingSeverity::Error, code: "DEADLOCK_SCC".into(), node_ids: component.clone(), message: "control-flow strongly connected component has no terminal or escape edge; execution can wait forever".into() });
            } else {
                report.findings.push(StaticFinding {
                    severity: FindingSeverity::Warning,
                    code: "UNBOUNDED_LOOP".into(),
                    node_ids: component.clone(),
                    message: "cyclic control flow has no statically bounded iteration count".into(),
                });
            }
        }
    }
    for (&id, def) in &defs {
        if let RunnerKind::Effect(spec) = &def.runner
            && spec.class == EffectClass::ExternalWrite
            && !node_has_compensation(graph, id)
        {
            report.findings.push(finding(
                FindingSeverity::Warning,
                "MISSING_COMPENSATION",
                id,
                "external write is outside a declared saga compensation region",
            ));
        }
    }
    Ok(report)
}

fn node_has_compensation(graph: &Graph, id: NodeId) -> bool {
    graph
        .nodes
        .iter()
        .find(|n| n.id == id)
        .is_some_and(|n| n.config.contains_key("compensation") || n.config.contains_key("saga_id"))
}

fn finding(severity: FindingSeverity, code: &str, node_id: NodeId, message: &str) -> StaticFinding {
    StaticFinding {
        severity,
        code: code.into(),
        node_ids: vec![node_id],
        message: message.into(),
    }
}

fn strongly_connected_components<I: IntoIterator<Item = NodeId>>(
    edges: &HashMap<NodeId, Vec<NodeId>>,
    nodes: I,
) -> Vec<Vec<NodeId>> {
    struct Tarjan<'a> {
        edges: &'a HashMap<NodeId, Vec<NodeId>>,
        index: usize,
        stack: Vec<NodeId>,
        on_stack: HashSet<NodeId>,
        indices: HashMap<NodeId, usize>,
        low: HashMap<NodeId, usize>,
        out: Vec<Vec<NodeId>>,
    }
    fn visit(state: &mut Tarjan<'_>, v: NodeId) {
        state.indices.insert(v, state.index);
        state.low.insert(v, state.index);
        state.index += 1;
        state.stack.push(v);
        state.on_stack.insert(v);
        for &w in state.edges.get(&v).into_iter().flatten() {
            if !state.indices.contains_key(&w) {
                visit(state, w);
                let low = state.low[&v].min(state.low[&w]);
                state.low.insert(v, low);
            } else if state.on_stack.contains(&w) {
                let low = state.low[&v].min(state.indices[&w]);
                state.low.insert(v, low);
            }
        }
        if state.low[&v] == state.indices[&v] {
            let mut component = Vec::new();
            while let Some(w) = state.stack.pop() {
                state.on_stack.remove(&w);
                component.push(w);
                if w == v {
                    break;
                }
            }
            component.sort_unstable();
            state.out.push(component);
        }
    }
    let mut state = Tarjan {
        edges,
        index: 0,
        stack: Vec::new(),
        on_stack: HashSet::new(),
        indices: HashMap::new(),
        low: HashMap::new(),
        out: Vec::new(),
    };
    for node in nodes {
        if !state.indices.contains_key(&node) {
            visit(&mut state, node);
        }
    }
    state.out
}
