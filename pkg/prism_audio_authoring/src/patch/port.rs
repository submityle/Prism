//! Port descriptors for Patch primitive nodes.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Describes the fixed input/output port shape of each design section 11
//! primitive. The compiler uses [`PortCount`] to allocate matching mono buffers
//! in the runtime `AudioGraph` and the validator uses it to range-check every
//! connection endpoint. [`SignalKind`] records the semantic role of a port so
//! tooling can distinguish audio, control, and trigger wires.

/// The semantic role carried by a Patch port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SignalKind {
    /// A full-rate audio signal.
    Audio,
    /// A control-rate scalar signal (treated as audio-rate internally).
    Control,
    /// A discrete trigger/gate edge.
    Trigger,
}

/// The number of input and output ports a primitive exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PortCount {
    /// Number of input ports.
    pub inputs: usize,
    /// Number of output ports.
    pub outputs: usize,
}

impl PortCount {
    /// Builds a [`PortCount`] from explicit input and output counts.
    #[must_use]
    pub const fn new(inputs: usize, outputs: usize) -> Self {
        Self { inputs, outputs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_count_stores_fields() {
        let pc = PortCount::new(2, 1);
        assert_eq!(pc.inputs, 2);
        assert_eq!(pc.outputs, 1);
    }

    #[test]
    fn signal_kinds_are_distinct() {
        assert_ne!(SignalKind::Audio, SignalKind::Trigger);
        assert_ne!(SignalKind::Control, SignalKind::Trigger);
    }
}
