//! The serializable Patch graph description (the data asset).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Models the editable, hot-reloadable Patch data asset of design section 11: a
//! small DSP graph of nodes and connections, the named parameters and triggers
//! it exposes to the host, and which internal node ports form its audio
//! outputs. It is pure data; `super::compiler` turns it into a runnable graph
//! and `super::validate` checks it. Node indices are positions in `nodes`.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::Sample;

use super::node_kind::NodeKind;

/// A node instance within a [`PatchDescription`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PatchNode {
    /// The kind of primitive (or nested patch) this node realizes.
    pub kind: NodeKind,
    /// A human-readable name, used as the prefix when re-exposing nested
    /// parameters.
    pub name: String,
    /// Initial values for named parameters, overriding the kind defaults.
    pub initial_params: Vec<(String, Sample)>,
}

impl PatchNode {
    /// Builds a node of `kind` named `name` with no parameter overrides.
    #[must_use]
    pub fn new(kind: NodeKind, name: String) -> Self {
        Self {
            kind,
            name,
            initial_params: Vec::new(),
        }
    }

    /// Returns the initial value for `param`, if an override is present.
    #[must_use]
    pub fn initial_param(&self, param: &str) -> Option<Sample> {
        self.initial_params
            .iter()
            .find(|(name, _)| name == param)
            .map(|(_, value)| *value)
    }
}

/// A directed audio connection between two node ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PatchConnection {
    /// Source node index.
    pub from_node: usize,
    /// Source output port index.
    pub from_port: usize,
    /// Destination node index.
    pub to_node: usize,
    /// Destination input port index.
    pub to_port: usize,
}

/// A parameter of an internal node exposed to the host under a public name.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ExposedParam {
    /// Public name the host uses to address the parameter.
    pub public_name: String,
    /// Index of the owning node.
    pub node: usize,
    /// Parameter name on that node.
    pub param: String,
}

/// A trigger/gate input of an internal node exposed to the host.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ExposedTrigger {
    /// Public name the host uses to address the trigger.
    pub public_name: String,
    /// Index of the owning node.
    pub node: usize,
}

/// An internal node port designated as a Patch audio output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PatchOutput {
    /// Index of the source node.
    pub node: usize,
    /// Output port index on that node.
    pub port: usize,
}

/// A complete, editable Patch graph description.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PatchDescription {
    /// Node instances, addressed by position.
    pub nodes: Vec<PatchNode>,
    /// Directed audio connections between node ports.
    pub connections: Vec<PatchConnection>,
    /// Parameters exposed to the host.
    pub exposed_params: Vec<ExposedParam>,
    /// Triggers exposed to the host.
    pub exposed_triggers: Vec<ExposedTrigger>,
    /// Node ports forming the Patch audio outputs, in order.
    pub outputs: Vec<PatchOutput>,
}

impl PatchDescription {
    /// Builds an empty description.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of audio outputs the Patch produces.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn node_initial_param_lookup() {
        let mut node = PatchNode::new(NodeKind::Gain, "g".to_string());
        node.initial_params.push(("gain".to_string(), 0.5));
        assert!(node.initial_param("gain").is_some());
        assert!(node.initial_param("missing").is_none());
    }

    #[test]
    fn empty_description_has_no_outputs() {
        let desc = PatchDescription::new();
        assert_eq!(desc.output_count(), 0);
    }
}
