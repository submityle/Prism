//! Deterministic band-unlimited oscillator primitive.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! The oscillator primitive of the design section 11 node library. It realizes
//! the "deterministic synthesis" requirement: the phase accumulator advances by
//! a fixed per-block increment and all trigonometry routes through
//! `bevy_math::ops`, so a given parameter stream is reproducible sample for
//! sample. Frequency and amplitude are exposed parameters read from
//! [`ParamCell`]s.

use bevy_math::ops;
use core::f32::consts::TAU;

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::param::{Ramp, Smoothed};
use prism_audio_core::Sample;

use crate::patch::param::ParamCell;

/// The waveform an [`OscillatorNode`] generates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum OscWaveform {
    /// Pure sine wave.
    Sine,
    /// Rising ramp sawtooth in `[-1, 1]`.
    Sawtooth,
    /// Bipolar square wave.
    Square,
    /// Symmetric triangle wave.
    Triangle,
}

impl OscWaveform {
    /// Evaluates the waveform at a normalized phase in `[0, 1)`.
    #[inline]
    #[must_use]
    fn eval(self, phase: Sample) -> Sample {
        match self {
            OscWaveform::Sine => ops::sin(TAU * phase),
            OscWaveform::Sawtooth => 2.0 * phase - 1.0,
            OscWaveform::Square => {
                if phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            OscWaveform::Triangle => 1.0 - 4.0 * ops::abs(phase - 0.5),
        }
    }
}

/// A zero-input, single-output oscillator with exposed frequency and amplitude.
#[derive(Debug)]
pub struct OscillatorNode {
    waveform: OscWaveform,
    sample_rate: u32,
    phase: Sample,
    frequency: ParamCell,
    amplitude: ParamCell,
    amp_smoothed: Smoothed,
}

impl OscillatorNode {
    /// Builds an oscillator at `sample_rate` reading `frequency` and
    /// `amplitude` parameters.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        waveform: OscWaveform,
        frequency: ParamCell,
        amplitude: ParamCell,
    ) -> Self {
        let amp = amplitude.get();
        Self {
            waveform,
            sample_rate,
            phase: 0.0,
            frequency,
            amplitude,
            amp_smoothed: Smoothed::new(amp),
        }
    }
}

impl AudioNode for OscillatorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let frames = out.active_frames();
        let freq = self.frequency.get().max(0.0);
        let increment = freq / self.sample_rate as Sample;
        self.amp_smoothed
            .set_target(self.amplitude.get(), Ramp::Linear { samples: frames.max(1) as u32 });
        let dst = out.channel_mut(0);
        for d in dst.iter_mut() {
            let amp = self.amp_smoothed.next_sample();
            *d = self.waveform.eval(self.phase) * amp;
            self.phase += increment;
            self.phase -= ops::floor(self.phase);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.amp_smoothed = Smoothed::new(self.amplitude.get());
    }
}
