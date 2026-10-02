//! Error type for Patch validation and compilation.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Reports the failure modes of the design section 11 pipeline: structural
//! problems found by `super::validate` and graph-build problems surfaced by the
//! `prism_audio_core` compiler (such as cycles). Wraps
//! `prism_audio_core::graph::GraphError` so callers get one unified result.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use prism_audio_core::graph::GraphError;

/// A Patch validation or compilation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PatchError {
    /// A node index referenced a position outside `nodes`.
    UnknownNode(usize),
    /// A connection or output addressed a port the node does not have.
    PortOutOfRange {
        /// Offending node index.
        node: usize,
        /// Offending port index.
        port: usize,
    },
    /// An exposed parameter named a parameter the node does not have.
    UnknownParam {
        /// Owning node index.
        node: usize,
        /// Requested parameter name.
        param: String,
    },
    /// An exposed trigger targeted a node without a trigger input.
    UnknownTrigger {
        /// Owning node index.
        node: usize,
    },
    /// A connection targeted a node that has no audio inputs (a pure source).
    ConnectionIntoSource {
        /// Offending destination node index.
        node: usize,
    },
    /// Nested patches exceeded the maximum expansion depth.
    NestingTooDeep {
        /// The maximum supported nesting depth.
        max: usize,
    },
    /// The description declared no audio outputs.
    NoOutput,
    /// A `SubPatch` leaked past flattening (internal invariant violation).
    UnflattenedSubPatch,
    /// The underlying `prism_audio_core` graph failed to build.
    GraphBuild(GraphError),
}

impl core::fmt::Display for PatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PatchError::UnknownNode(node) => write!(f, "unknown node index {node}"),
            PatchError::PortOutOfRange { node, port } => {
                write!(f, "port {port} out of range for node {node}")
            }
            PatchError::UnknownParam { node, param } => {
                write!(f, "node {node} has no parameter named '{param}'")
            }
            PatchError::UnknownTrigger { node } => {
                write!(f, "node {node} has no trigger input")
            }
            PatchError::ConnectionIntoSource { node } => {
                write!(f, "node {node} has no audio inputs to connect into")
            }
            PatchError::NestingTooDeep { max } => {
                write!(f, "nested patches exceeded maximum depth {max}")
            }
            PatchError::NoOutput => write!(f, "patch declares no audio outputs"),
            PatchError::UnflattenedSubPatch => {
                write!(f, "internal error: sub-patch was not flattened")
            }
            PatchError::GraphBuild(err) => write!(f, "graph build failed: {err}"),
        }
    }
}

impl From<GraphError> for PatchError {
    fn from(err: GraphError) -> Self {
        PatchError::GraphBuild(err)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PatchError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn display_is_populated() {
        let err = PatchError::UnknownNode(3);
        assert!(!err.to_string().is_empty());
        let err = PatchError::GraphBuild(GraphError::Cycle);
        assert!(err.to_string().contains("graph build"));
    }

    #[test]
    fn graph_error_converts() {
        let err: PatchError = GraphError::NoMaster.into();
        assert_eq!(err, PatchError::GraphBuild(GraphError::NoMaster));
    }
}
