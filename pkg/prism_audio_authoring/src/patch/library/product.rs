//! Two-input multiplier primitive (ring modulation / amplitude control).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! A math primitive of the design section 11 node library. Multiplying two mono
//! signals realizes ring modulation and voltage-controlled amplitude, used for
//! tremolo, amplitude envelopes applied as a signal, and metallic timbres.

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};

/// A node computing the per-sample product of two mono inputs.
#[derive(Debug, Default)]
pub struct ProductNode;

impl ProductNode {
    /// Builds a product node.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl AudioNode for ProductNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (inputs, outputs) = io.split();
        let a = inputs[0].channel(0);
        let b = inputs[1].channel(0);
        let dst = outputs[0].channel_mut(0);
        for (d, (x, y)) in dst.iter_mut().zip(a.iter().zip(b)) {
            *d = *x * *y;
        }
    }
}
