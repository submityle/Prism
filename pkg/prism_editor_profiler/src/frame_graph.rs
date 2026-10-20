//! Deterministic render-frame graph data and critical-path analysis.

use std::collections::{BTreeMap, BTreeSet};

use crate::{validate_limit, validate_text, ProfilerError, ProfilerLimits};

/// Stable identity for a frame-graph node.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct GraphNodeId(u64);

impl GraphNodeId {
    /// Creates an identifier from a producer-assigned value.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the producer-assigned value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// One frame-graph pass or scheduling node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphNode {
    id: GraphNodeId,
    label: String,
    duration_ns: u64,
}

impl GraphNode {
    /// Creates a graph node. Label validation occurs when inserted into a graph.
    #[must_use]
    pub fn new(id: GraphNodeId, label: impl Into<String>, duration_ns: u64) -> Self {
        Self {
            id,
            label: label.into(),
            duration_ns,
        }
    }

    /// Returns the node identity.
    #[must_use]
    pub const fn id(&self) -> GraphNodeId {
        self.id
    }

    /// Returns the display label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the measured node duration.
    #[must_use]
    pub const fn duration_ns(&self) -> u64 {
        self.duration_ns
    }
}

/// A deterministic weighted path through a frame graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CriticalPath {
    node_ids: Vec<GraphNodeId>,
    total_duration_ns: u64,
}

impl CriticalPath {
    /// Returns node IDs from source to sink.
    #[must_use]
    pub fn node_ids(&self) -> &[GraphNodeId] {
        &self.node_ids
    }

    /// Returns the sum of durations along the path.
    #[must_use]
    pub const fn total_duration_ns(&self) -> u64 {
        self.total_duration_ns
    }
}

/// A bounded directed acyclic frame graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameGraph {
    limits: ProfilerLimits,
    nodes: BTreeMap<GraphNodeId, GraphNode>,
    dependencies: BTreeSet<(GraphNodeId, GraphNodeId)>,
}

impl FrameGraph {
    /// Creates an empty graph.
    #[must_use]
    pub const fn new(limits: ProfilerLimits) -> Self {
        Self {
            limits,
            nodes: BTreeMap::new(),
            dependencies: BTreeSet::new(),
        }
    }

    /// Inserts a node in stable ID order.
    pub fn add_node(&mut self, node: GraphNode) -> Result<(), ProfilerError> {
        validate_limit(
            self.nodes.len() + 1,
            self.limits.max_graph_nodes,
            "graph nodes",
        )?;
        validate_text(&node.label, "graph node label", &self.limits)?;
        if self.nodes.contains_key(&node.id) {
            return Err(ProfilerError::DuplicateId {
                resource: "graph node",
                id: node.id.raw(),
            });
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// Adds a directed dependency where `before` must execute before `after`.
    pub fn add_dependency(
        &mut self,
        before: GraphNodeId,
        after: GraphNodeId,
    ) -> Result<(), ProfilerError> {
        if before == after || !self.nodes.contains_key(&before) || !self.nodes.contains_key(&after)
        {
            return Err(ProfilerError::InvalidDependency);
        }
        validate_limit(
            self.dependencies.len() + 1,
            self.limits.max_graph_dependencies,
            "graph dependencies",
        )?;
        if !self.dependencies.insert((before, after)) {
            return Err(ProfilerError::InvalidDependency);
        }
        Ok(())
    }

    /// Returns nodes in stable ID order.
    pub fn nodes(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.values()
    }

    /// Returns dependencies in `(before, after)` order.
    pub fn dependencies(&self) -> impl Iterator<Item = (GraphNodeId, GraphNodeId)> + '_ {
        self.dependencies.iter().copied()
    }

    /// Computes a deterministic topological order or reports a cycle.
    pub fn topological_order(&self) -> Result<Vec<GraphNodeId>, ProfilerError> {
        let mut indegree: BTreeMap<GraphNodeId, usize> =
            self.nodes.keys().copied().map(|id| (id, 0)).collect();
        let mut successors: BTreeMap<GraphNodeId, Vec<GraphNodeId>> = BTreeMap::new();
        for (before, after) in &self.dependencies {
            *indegree
                .get_mut(after)
                .expect("dependency endpoints are validated") += 1;
            successors.entry(*before).or_default().push(*after);
        }
        for values in successors.values_mut() {
            values.sort_unstable();
        }
        let mut ready: BTreeSet<GraphNodeId> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            for successor in successors.get(&id).into_iter().flatten() {
                let degree = indegree
                    .get_mut(successor)
                    .expect("dependency endpoints are validated");
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*successor);
                }
            }
        }
        if order.len() == self.nodes.len() {
            Ok(order)
        } else {
            Err(ProfilerError::Cycle)
        }
    }

    /// Computes the maximum-duration path with stable ID tie-breakers.
    pub fn critical_path(&self) -> Result<CriticalPath, ProfilerError> {
        let order = self.topological_order()?;
        let mut predecessors: BTreeMap<GraphNodeId, Vec<GraphNodeId>> = BTreeMap::new();
        for (before, after) in &self.dependencies {
            predecessors.entry(*after).or_default().push(*before);
        }
        for values in predecessors.values_mut() {
            values.sort_unstable();
        }
        let mut best: BTreeMap<GraphNodeId, u64> = BTreeMap::new();
        let mut previous: BTreeMap<GraphNodeId, Option<GraphNodeId>> = BTreeMap::new();
        for id in &order {
            let node = &self.nodes[id];
            let mut total = node.duration_ns;
            let mut prior = None;
            for predecessor in predecessors.get(id).into_iter().flatten() {
                let candidate = best[predecessor]
                    .checked_add(node.duration_ns)
                    .ok_or(ProfilerError::ArithmeticOverflow)?;
                if candidate > total
                    || (candidate == total && prior.is_none_or(|value| *predecessor < value))
                {
                    total = candidate;
                    prior = Some(*predecessor);
                }
            }
            best.insert(*id, total);
            previous.insert(*id, prior);
        }
        let end = order
            .iter()
            .copied()
            .max_by_key(|id| (best[id], std::cmp::Reverse(*id)))
            .ok_or(ProfilerError::InvalidRange {
                field: "empty graph",
            })?;
        let mut path = Vec::new();
        let mut cursor = Some(end);
        while let Some(id) = cursor {
            path.push(id);
            cursor = previous[&id];
        }
        path.reverse();
        Ok(CriticalPath {
            node_ids: path,
            total_duration_ns: best[&end],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> FrameGraph {
        FrameGraph::new(ProfilerLimits {
            max_graph_nodes: 4,
            max_graph_dependencies: 4,
            ..ProfilerLimits::default()
        })
    }

    #[test]
    fn order_and_critical_path_are_stable() {
        let mut graph = graph();
        for (id, duration) in [(3, 2), (1, 5), (2, 7), (4, 3)] {
            graph
                .add_node(GraphNode::new(
                    GraphNodeId::from_raw(id),
                    id.to_string(),
                    duration,
                ))
                .unwrap();
        }
        graph
            .add_dependency(GraphNodeId::from_raw(1), GraphNodeId::from_raw(2))
            .unwrap();
        graph
            .add_dependency(GraphNodeId::from_raw(1), GraphNodeId::from_raw(3))
            .unwrap();
        graph
            .add_dependency(GraphNodeId::from_raw(2), GraphNodeId::from_raw(4))
            .unwrap();
        graph
            .add_dependency(GraphNodeId::from_raw(3), GraphNodeId::from_raw(4))
            .unwrap();
        assert_eq!(
            graph.topological_order().unwrap(),
            vec![
                GraphNodeId::from_raw(1),
                GraphNodeId::from_raw(2),
                GraphNodeId::from_raw(3),
                GraphNodeId::from_raw(4)
            ]
        );
        let path = graph.critical_path().unwrap();
        assert_eq!(
            path.node_ids(),
            &[
                GraphNodeId::from_raw(1),
                GraphNodeId::from_raw(2),
                GraphNodeId::from_raw(4)
            ]
        );
        assert_eq!(path.total_duration_ns(), 15);
    }

    #[test]
    fn cycles_and_duplicate_edges_are_rejected() {
        let mut graph = graph();
        graph
            .add_node(GraphNode::new(GraphNodeId::from_raw(1), "a", 1))
            .unwrap();
        graph
            .add_node(GraphNode::new(GraphNodeId::from_raw(2), "b", 1))
            .unwrap();
        graph
            .add_dependency(GraphNodeId::from_raw(1), GraphNodeId::from_raw(2))
            .unwrap();
        assert!(matches!(
            graph.add_dependency(GraphNodeId::from_raw(1), GraphNodeId::from_raw(2)),
            Err(ProfilerError::InvalidDependency)
        ));
        graph
            .add_dependency(GraphNodeId::from_raw(2), GraphNodeId::from_raw(1))
            .unwrap();
        assert!(matches!(
            graph.topological_order(),
            Err(ProfilerError::Cycle)
        ));
    }
}
