//! Structural validation of a Patch description.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Enforces the correctness preconditions of design section 11 before the
//! compiler runs: every connection and output addresses an existing node and
//! port, exposed parameters and triggers name real inputs, connections never
//! target a pure source, and nested patches stay within the bounded expansion
//! depth. Cycle detection is delegated to the `prism_audio_core` compiler,
//! which reports `prism_audio_core::graph::GraphError::Cycle`.

use super::description::PatchDescription;
use super::error::PatchError;
use super::node_kind::NodeKind;

/// Maximum nesting depth for `SubPatch` expansion.
pub const MAX_NESTING_DEPTH: usize = 4;

/// Validates `desc`, returning the first problem found.
///
/// # Errors
///
/// Returns a [`PatchError`] describing the first structural violation.
pub fn validate(desc: &PatchDescription) -> Result<(), PatchError> {
    validate_at_depth(desc, 0)
}

fn validate_at_depth(desc: &PatchDescription, depth: usize) -> Result<(), PatchError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(PatchError::NestingTooDeep {
            max: MAX_NESTING_DEPTH,
        });
    }

    let node_count = desc.nodes.len();

    // Connections: endpoints exist and target has an audio input.
    for conn in &desc.connections {
        if conn.from_node >= node_count {
            return Err(PatchError::UnknownNode(conn.from_node));
        }
        if conn.to_node >= node_count {
            return Err(PatchError::UnknownNode(conn.to_node));
        }
        let from_ports = desc.nodes[conn.from_node].kind.port_count();
        if conn.from_port >= from_ports.outputs {
            return Err(PatchError::PortOutOfRange {
                node: conn.from_node,
                port: conn.from_port,
            });
        }
        let to_ports = desc.nodes[conn.to_node].kind.port_count();
        if to_ports.inputs == 0 {
            return Err(PatchError::ConnectionIntoSource { node: conn.to_node });
        }
        if conn.to_port >= to_ports.inputs {
            return Err(PatchError::PortOutOfRange {
                node: conn.to_node,
                port: conn.to_port,
            });
        }
    }

    // Exposed parameters name a real parameter on an existing node.
    for exposed in &desc.exposed_params {
        if exposed.node >= node_count {
            return Err(PatchError::UnknownNode(exposed.node));
        }
        let names = desc.nodes[exposed.node].kind.param_names();
        if !names.iter().any(|name| *name == exposed.param) {
            return Err(PatchError::UnknownParam {
                node: exposed.node,
                param: exposed.param.clone(),
            });
        }
    }

    // Exposed triggers name a node that actually has a trigger input.
    for exposed in &desc.exposed_triggers {
        if exposed.node >= node_count {
            return Err(PatchError::UnknownNode(exposed.node));
        }
        if desc.nodes[exposed.node].kind.gate_names().is_empty() {
            return Err(PatchError::UnknownTrigger { node: exposed.node });
        }
    }

    // Outputs address an existing node and output port.
    for output in &desc.outputs {
        if output.node >= node_count {
            return Err(PatchError::UnknownNode(output.node));
        }
        let ports = desc.nodes[output.node].kind.port_count();
        if output.port >= ports.outputs {
            return Err(PatchError::PortOutOfRange {
                node: output.node,
                port: output.port,
            });
        }
    }

    if desc.outputs.is_empty() {
        return Err(PatchError::NoOutput);
    }

    // Recurse into nested patches.
    for node in &desc.nodes {
        if let NodeKind::SubPatch(sub) = &node.kind {
            validate_at_depth(sub, depth + 1)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::builder::PatchBuilder;
    use crate::patch::library::oscillator::OscWaveform;
    use crate::patch::node_kind::NodeKind;

    fn simple_osc_patch() -> PatchDescription {
        let mut b = PatchBuilder::new();
        let osc = b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        b.add_output(osc, 0);
        b.build()
    }

    #[test]
    fn valid_patch_passes() {
        assert!(validate(&simple_osc_patch()).is_ok());
    }

    #[test]
    fn missing_output_fails() {
        let mut b = PatchBuilder::new();
        b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        let desc = b.build();
        assert_eq!(validate(&desc), Err(PatchError::NoOutput));
    }

    #[test]
    fn connection_into_source_fails() {
        let mut b = PatchBuilder::new();
        let osc = b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        let other = b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc2",
        );
        b.connect(osc, 0, other, 0);
        b.add_output(osc, 0);
        let desc = b.build();
        assert_eq!(
            validate(&desc),
            Err(PatchError::ConnectionIntoSource { node: other })
        );
    }

    #[test]
    fn bad_exposed_param_fails() {
        let mut b = PatchBuilder::new();
        let g = b.add_node(NodeKind::Gain, "g");
        b.expose_param("x", g, "nonexistent");
        b.add_output(g, 0);
        let desc = b.build();
        assert!(matches!(
            validate(&desc),
            Err(PatchError::UnknownParam { .. })
        ));
    }
}
