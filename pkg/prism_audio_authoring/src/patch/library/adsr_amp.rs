//! Gate-driven amplitude envelope primitive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! An envelope primitive of the design section 11 node library, bridging to the
//! section 12 modulation system: it drives a `crate::modulation::envelope`
//! generator from a shared `crate::patch::trigger::GateCell` and multiplies its
//! mono input by the envelope level. Note-on edges (counter changes) retrigger
//! the attack; the sustain gate controls release, giving sample-accurate
//! triggered amplitude shaping.

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};

use crate::modulation::envelope::{Envelope, EnvelopeConfig};
use crate::patch::trigger::GateCell;

/// A single-input, single-output node applying a gated `AHDSR` envelope to its
/// input signal.
#[derive(Debug)]
pub struct AdsrAmpNode {
    envelope: Envelope,
    gate: GateCell,
    last_edges: u32,
    last_gate: bool,
}

impl AdsrAmpNode {
    /// Builds an envelope amplifier at `sample_rate` with `config`, driven by
    /// `gate`.
    #[must_use]
    pub fn new(sample_rate: u32, config: EnvelopeConfig, gate: GateCell) -> Self {
        Self {
            envelope: Envelope::new(sample_rate, config),
            last_edges: gate.edge_count(),
            last_gate: gate.gate(),
            gate,
        }
    }
}

impl AudioNode for AdsrAmpNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        // Resolve trigger/gate state once per block (control-rate gating).
        let edges = self.gate.edge_count();
        let gate_open = self.gate.gate();
        if edges != self.last_edges {
            self.last_edges = edges;
            self.envelope.retrigger();
        } else if gate_open != self.last_gate {
            self.envelope.set_gate(gate_open);
        }
        self.last_gate = gate_open;

        let (input, output) = io.io(0, 0);
        let src = input.channel(0);
        let dst = output.channel_mut(0);
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s * self.envelope.next_sample();
        }
    }

    fn reset(&mut self) {
        use crate::modulation::source::Modulator;
        self.envelope.reset();
        self.last_edges = self.gate.edge_count();
        self.last_gate = self.gate.gate();
    }
}
