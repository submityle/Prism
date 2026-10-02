//! Multi-input mixer primitive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! A math primitive of the design section 11 node library. It sums a
//! configurable number of mono inputs into one output, the building block for
//! additive and layered synthesis within a patch.

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};

/// A node that sums `inputs` mono ports into a single mono output.
#[derive(Debug)]
pub struct SumNode {
    inputs: usize,
}

impl SumNode {
    /// Builds a summing node with `inputs` input ports (at least one).
    #[must_use]
    pub fn new(inputs: usize) -> Self {
        Self {
            inputs: inputs.max(1),
        }
    }
}

impl AudioNode for SumNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (inputs, outputs) = io.split();
        let dst = outputs[0].channel_mut(0);
        for d in dst.iter_mut() {
            *d = 0.0;
        }
        for input in inputs.iter().take(self.inputs) {
            let src = input.channel(0);
            for (d, s) in dst.iter_mut().zip(src) {
                *d += *s;
            }
        }
    }
}
