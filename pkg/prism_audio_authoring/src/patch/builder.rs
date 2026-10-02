//! Incremental builder API for assembling Patch descriptions.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the runtime incremental authoring API of design section 11 (the
//! `PatchBuilder`, aligned with a MetaSound-style builder). Code or data drives
//! it on a task thread to assemble or rewrite a Patch, which is then compiled
//! into a new node off the audio thread. It emits a plain
//! [`PatchDescription`]; it performs no validation itself (see
//! `super::validate`).

#[cfg(not(feature = "std"))]
use alloc::string::ToString;

use prism_audio_core::Sample;

use super::description::{
    ExposedParam, ExposedTrigger, PatchConnection, PatchDescription, PatchNode, PatchOutput,
};
use super::node_kind::NodeKind;

/// A mutable builder that accumulates nodes, wires, and exposed inputs.
#[derive(Debug, Default)]
pub struct PatchBuilder {
    desc: PatchDescription,
}

impl PatchBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node of `kind` named `name` and returns its index.
    pub fn add_node(&mut self, kind: NodeKind, name: &str) -> usize {
        let index = self.desc.nodes.len();
        self.desc.nodes.push(PatchNode::new(kind, name.to_string()));
        index
    }

    /// Sets an initial parameter override on a previously added node.
    ///
    /// A later call for the same parameter replaces the earlier value.
    pub fn set_param(&mut self, node: usize, name: &str, value: Sample) {
        if let Some(entry) = self.desc.nodes.get_mut(node) {
            if let Some(existing) = entry
                .initial_params
                .iter_mut()
                .find(|(param, _)| param == name)
            {
                existing.1 = value;
            } else {
                entry.initial_params.push((name.to_string(), value));
            }
        }
    }

    /// Connects an output port of one node to an input port of another.
    pub fn connect(&mut self, from_node: usize, from_port: usize, to_node: usize, to_port: usize) {
        self.desc.connections.push(PatchConnection {
            from_node,
            from_port,
            to_node,
            to_port,
        });
    }

    /// Exposes a node parameter to the host under `public_name`.
    pub fn expose_param(&mut self, public_name: &str, node: usize, param: &str) {
        self.desc.exposed_params.push(ExposedParam {
            public_name: public_name.to_string(),
            node,
            param: param.to_string(),
        });
    }

    /// Exposes a node trigger to the host under `public_name`.
    pub fn expose_trigger(&mut self, public_name: &str, node: usize) {
        self.desc.exposed_triggers.push(ExposedTrigger {
            public_name: public_name.to_string(),
            node,
        });
    }

    /// Designates a node output port as a Patch audio output.
    pub fn add_output(&mut self, node: usize, port: usize) {
        self.desc.outputs.push(PatchOutput { node, port });
    }

    /// Consumes the builder and returns the assembled description.
    #[must_use]
    pub fn build(self) -> PatchDescription {
        self.desc
    }

    /// Borrows the description under construction.
    #[must_use]
    pub fn description(&self) -> &PatchDescription {
        &self.desc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_expected_shape() {
        let mut b = PatchBuilder::new();
        let g = b.add_node(NodeKind::Gain, "g");
        b.set_param(g, "gain", 0.5);
        b.set_param(g, "gain", 0.25);
        b.expose_param("level", g, "gain");
        b.add_output(g, 0);
        let desc = b.build();
        assert_eq!(desc.nodes.len(), 1);
        assert_eq!(desc.nodes[0].initial_params.len(), 1);
        let value = desc.nodes[0].initial_param("gain").expect("override set");
        assert!((value - 0.25).abs() < 1.0e-6);
        assert_eq!(desc.exposed_params.len(), 1);
        assert_eq!(desc.outputs.len(), 1);
    }
}
