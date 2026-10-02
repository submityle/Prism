//! Seeded white-noise source primitive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The stochastic source of the design section 11 node library. It draws from
//! the shared seeded generator `crate::rng::Rng`, so synthesized textures stay
//! reproducible for a given seed (the "deterministic synthesis" requirement).
//! Amplitude is an exposed parameter.

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::param::{Ramp, Smoothed};

use crate::patch::param::ParamCell;
use crate::rng::Rng;

/// A zero-input, single-output seeded white-noise source.
#[derive(Debug)]
pub struct NoiseNode {
    rng: Rng,
    seed: u64,
    amplitude: ParamCell,
    amp_smoothed: Smoothed,
}

impl NoiseNode {
    /// Builds a noise source seeded with `seed`, scaled by `amplitude`.
    #[must_use]
    pub fn new(seed: u64, amplitude: ParamCell) -> Self {
        let amp = amplitude.get();
        Self {
            rng: Rng::new(seed),
            seed,
            amplitude,
            amp_smoothed: Smoothed::new(amp),
        }
    }
}

impl AudioNode for NoiseNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let frames = out.active_frames();
        self.amp_smoothed
            .set_target(self.amplitude.get(), Ramp::Linear { samples: frames.max(1) as u32 });
        let dst = out.channel_mut(0);
        for d in dst.iter_mut() {
            let amp = self.amp_smoothed.next_sample();
            *d = self.rng.next_bipolar() * amp;
        }
    }

    fn reset(&mut self) {
        self.rng = Rng::new(self.seed);
        self.amp_smoothed = Smoothed::new(self.amplitude.get());
    }
}
