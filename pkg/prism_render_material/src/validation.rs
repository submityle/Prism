use crate::{MaterialGraph, MaterialNode, MaterialNodeId};
use alloc::collections::BTreeSet;
use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialValidationError {
    MissingOutput,
    MissingNode(MaterialNodeId),
    Cycle(MaterialNodeId),
    EmptyClosure(MaterialNodeId),
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
