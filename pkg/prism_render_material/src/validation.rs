use crate::{MaterialGraph, MaterialNode, MaterialNodeId};
use alloc::collections::BTreeSet;
use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialValidationError {
    MissingOutput,
    MissingNode(MaterialNodeId),
    Cycle(MaterialNodeId),
    EmptyClosure(MaterialNodeId),
    /// The `Layer`/`Mix` closure slab exceeded the bounded depth (design §3.2).
    ClosureSlabTooDeep {
        node: MaterialNodeId,
        depth: u32,
        max: u32,
    },
}

impl fmt::Display for MaterialValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "material graph validation failed: {self:?}")
    }
}
impl std::error::Error for MaterialValidationError {}

pub fn validate_graph(graph: &MaterialGraph) -> Result<(), MaterialValidationError> {
    let output = graph.output.ok_or(MaterialValidationError::MissingOutput)?;
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    visit(graph, output, &mut visiting, &mut visited)
}

fn visit(
    graph: &MaterialGraph,
    id: MaterialNodeId,
    visiting: &mut BTreeSet<MaterialNodeId>,
    visited: &mut BTreeSet<MaterialNodeId>,
) -> Result<(), MaterialValidationError> {
    if visited.contains(&id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return Err(MaterialValidationError::Cycle(id));
    }
    let node = graph
        .nodes
        .get(&id)
        .ok_or(MaterialValidationError::MissingNode(id))?;
    match node {
        MaterialNode::TextureSample { uv, .. } => visit(graph, *uv, visiting, visited)?,
        MaterialNode::Add(a, b) | MaterialNode::Multiply(a, b) => {
            visit(graph, *a, visiting, visited)?;
            visit(graph, *b, visiting, visited)?;
        }
        MaterialNode::Closure { inputs, .. } => {
            if inputs.is_empty() {
                return Err(MaterialValidationError::EmptyClosure(id));
            }
            for input in inputs {
                visit(graph, *input, visiting, visited)?;
            }
        }
        _ => {}
    }
    visiting.remove(&id);
    visited.insert(id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ClosureKind, MaterialValue};
    use alloc::collections::BTreeMap;

    fn node_id(raw: u32) -> MaterialNodeId {
        MaterialNodeId(raw)
    }

    #[test]
    fn missing_output_is_rejected() {
        let graph = MaterialGraph::default();
        assert_eq!(
            validate_graph(&graph),
            Err(MaterialValidationError::MissingOutput)
        );
    }

    #[test]
    fn dangling_output_reports_missing_node() {
        let output = node_id(9);
        let graph = MaterialGraph {
            nodes: BTreeMap::new(),
            output: Some(output),
        };
        assert_eq!(
            validate_graph(&graph),
            Err(MaterialValidationError::MissingNode(output))
        );
    }

    #[test]
    fn self_referential_node_is_a_cycle() {
        let output = node_id(0);
        let graph = MaterialGraph {
            nodes: BTreeMap::from([(output, MaterialNode::Multiply(output, output))]),
            output: Some(output),
        };
        assert_eq!(
            validate_graph(&graph),
            Err(MaterialValidationError::Cycle(output))
        );
    }

    #[test]
    fn mutually_recursive_nodes_are_a_cycle() {
        let a = node_id(0);
        let b = node_id(1);
        let graph = MaterialGraph {
            nodes: BTreeMap::from([(a, MaterialNode::Add(b, b)), (b, MaterialNode::Add(a, a))]),
            output: Some(a),
        };
        assert!(matches!(
            validate_graph(&graph),
            Err(MaterialValidationError::Cycle(_))
        ));
    }

    #[test]
    fn closure_without_inputs_is_rejected() {
        let output = node_id(0);
        let graph = MaterialGraph {
            nodes: BTreeMap::from([(
                output,
                MaterialNode::Closure {
                    kind: ClosureKind::Diffuse,
                    inputs: Vec::new(),
                },
            )]),
            output: Some(output),
        };
        assert_eq!(
            validate_graph(&graph),
            Err(MaterialValidationError::EmptyClosure(output))
        );
    }

    #[test]
    fn acyclic_graph_with_shared_subtree_validates_once() {
        let leaf = node_id(0);
        let output = node_id(1);
        // The leaf is referenced twice; the `visited` set must keep the walk
        // total instead of re-flagging the shared node as a cycle.
        let graph = MaterialGraph {
            nodes: BTreeMap::from([
                (leaf, MaterialNode::Constant(MaterialValue::Scalar(1.0))),
                (
                    output,
                    MaterialNode::Closure {
                        kind: ClosureKind::Diffuse,
                        inputs: vec![leaf, leaf],
                    },
                ),
            ]),
            output: Some(output),
        };
        assert_eq!(validate_graph(&graph), Ok(()));
    }
}
