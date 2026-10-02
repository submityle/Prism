//! Smoothed gain primitive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! A math primitive of the design section 11 node library: it multiplies its
//! single mono input by an exposed, smoothed `gain` parameter. Smoothing uses
//! `prism_audio_core::param::Smoothed` to avoid zipper noise on automation.

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::param::{Ramp, Smoothed};

use crate::patch::param::ParamCell;

/// A single-input, single-output smoothed gain stage.
#[derive(Debug)]
pub struct GainNode {
    gain: ParamCell,
    smoothed: Smoothed,
}

impl GainNode {
    /// Builds a gain node reading the `gain` parameter.
    #[must_use]
    pub fn new(gain: ParamCell) -> Self {
        let g = gain.get();
        Self {
            gain,
            smoothed: Smoothed::new(g),
        }
    }
}

impl AudioNode for GainNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let frames = output.active_frames();
        self.smoothed
            .set_target(self.gain.get(), Ramp::Linear { samples: frames.max(1) as u32 });
        let src = input.channel(0);
        let dst = output.channel_mut(0);
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s * self.smoothed.next_sample();
        }
    }

    fn reset(&mut self) {
        self.smoothed = Smoothed::new(self.gain.get());
    }
}
