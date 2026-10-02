//! One-pole low-pass filter primitive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The filter primitive of the design section 11 node library. It is a standard
//! first-order low-pass whose coefficient is derived from an exposed
//! `cutoff_hz` parameter using `bevy_math::ops`, keeping the response
//! deterministic across platforms.

use bevy_math::ops;
use core::f32::consts::TAU;

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::Sample;

use crate::patch::param::ParamCell;

/// A single-input, single-output first-order low-pass filter.
#[derive(Debug)]
pub struct OnePoleLowpassNode {
    sample_rate: u32,
    cutoff: ParamCell,
    state: Sample,
}

impl OnePoleLowpassNode {
    /// Builds a low-pass reading the `cutoff_hz` parameter at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, cutoff: ParamCell) -> Self {
        Self {
            sample_rate,
            cutoff,
            state: 0.0,
        }
    }

    /// Computes the one-pole smoothing coefficient for a cutoff in Hz.
    #[inline]
    fn coefficient(&self, cutoff_hz: Sample) -> Sample {
        let nyquist = self.sample_rate as Sample * 0.5;
        let fc = cutoff_hz.clamp(0.0, nyquist);
        if fc <= 0.0 {
            return 0.0;
        }
        let x = TAU * fc / self.sample_rate as Sample;
        // Standard one-pole coefficient a = exp(-2*pi*fc/fs); y += (1-a)(x-y).
        ops::exp(-x)
    }
}

impl AudioNode for OnePoleLowpassNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let a = self.coefficient(self.cutoff.get());
        let one_minus_a = 1.0 - a;
        let src = input.channel(0);
        let dst = output.channel_mut(0);
        for (d, s) in dst.iter_mut().zip(src) {
            self.state += one_minus_a * (*s - self.state);
            *d = self.state;
        }
    }

    fn reset(&mut self) {
        self.state = 0.0;
    }
}
