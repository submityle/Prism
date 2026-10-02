//! Constant/DC source primitive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! A math/logic primitive of the design section 11 node library. It emits the
//! value of its single exposed parameter, smoothed across the block via
//! `prism_audio_core::param::Smoothed` so parameter automation does not click.

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::param::{Ramp, Smoothed};

use crate::patch::param::ParamCell;

/// A single-output node emitting a smoothed constant drawn from a [`ParamCell`].
#[derive(Debug)]
pub struct ConstantNode {
    value: ParamCell,
    smoothed: Smoothed,
}

impl ConstantNode {
    /// Builds a constant source reading `value`.
    #[must_use]
    pub fn new(value: ParamCell) -> Self {
        let initial = value.get();
        Self {
            value,
            smoothed: Smoothed::new(initial),
        }
    }
}

impl AudioNode for ConstantNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let frames = out.active_frames();
        let target = self.value.get();
        self.smoothed
            .set_target(target, Ramp::Linear { samples: frames.max(1) as u32 });
        let dst = out.channel_mut(0);
        for d in dst.iter_mut() {
            *d = self.smoothed.next_sample();
        }
    }

    fn reset(&mut self) {
        self.smoothed = Smoothed::new(self.value.get());
    }
}
