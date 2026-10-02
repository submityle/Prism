//! Modulation-source adapter around the core low-frequency oscillator.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the `LfoModulator` of design section 12 by wrapping the existing
//! `prism_audio_core::modulation::Lfo` primitive (this crate deliberately does
//! not reimplement the oscillator). It implements `super::source::Modulator` so
//! the shared `Lfo` can drive the modulation matrix at control rate.

use prism_audio_core::modulation::{Lfo, LfoWaveform};
use prism_audio_core::Sample;

use super::source::{ModContext, Modulator};

/// A control-rate modulation source built on the core phase-accumulator
/// [`Lfo`].
///
/// Each control tick advances the oscillator by the block's frame count and
/// returns the resulting bipolar value in `[-1, 1]`. An optional unipolar
/// remap is available for parameters that expect a `[0, 1]` range.
#[derive(Debug, Clone, Copy)]
pub struct LfoModulator {
    lfo: Lfo,
    value: Sample,
    unipolar: bool,
}

impl LfoModulator {
    /// Creates an LFO source at `frequency_hz` with the given `waveform`.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        frequency_hz: Sample,
        waveform: LfoWaveform,
        unipolar: bool,
    ) -> Self {
        let lfo = Lfo::new(sample_rate, frequency_hz, waveform);
        let raw = lfo.value();
        Self {
            lfo,
            value: Self::shape(raw, unipolar),
            unipolar,
        }
    }

    /// Sets the oscillation frequency in Hz.
    #[inline]
    pub fn set_frequency(&mut self, sample_rate: u32, frequency_hz: Sample) {
        self.lfo.set_frequency(sample_rate, frequency_hz);
    }

    /// Sets the output waveform shape.
    #[inline]
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.lfo.set_waveform(waveform);
    }

    /// Offsets the phase into `[0, 1)`, useful to spread stacked LFOs.
    #[inline]
    pub fn set_phase(&mut self, phase01: Sample) {
        self.lfo.set_phase(phase01);
    }

    /// Maps a bipolar sample to the configured output range.
    #[inline]
    fn shape(raw: Sample, unipolar: bool) -> Sample {
        if unipolar {
            raw * 0.5 + 0.5
        } else {
            raw
        }
    }
}

impl Modulator for LfoModulator {
    fn tick(&mut self, ctx: &ModContext) -> Sample {
        let mut raw = self.lfo.value();
        for _ in 0..ctx.frames {
            raw = self.lfo.next_sample();
        }
        self.value = Self::shape(raw, self.unipolar);
        self.value
    }

    fn value(&self) -> Sample {
        self.value
    }

    fn reset(&mut self) {
        self.lfo.reset();
        self.value = Self::shape(self.lfo.value(), self.unipolar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    #[test]
    fn bipolar_output_is_bounded() {
        let mut m = LfoModulator::new(SR, 100.0, LfoWaveform::Sine, false);
        let ctx = ModContext::new(SR, 16);
        for _ in 0..1000 {
            let v = m.tick(&ctx);
            assert!((-1.0..=1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn unipolar_output_is_bounded() {
        let mut m = LfoModulator::new(SR, 100.0, LfoWaveform::Sine, true);
        let ctx = ModContext::new(SR, 16);
        for _ in 0..1000 {
            let v = m.tick(&ctx);
            assert!((0.0..=1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn reset_restores_phase() {
        let mut m = LfoModulator::new(SR, 10.0, LfoWaveform::Sawtooth, false);
        let ctx = ModContext::new(SR, 64);
        for _ in 0..10 {
            m.tick(&ctx);
        }
        m.reset();
        let after = m.value();
        let fresh = LfoModulator::new(SR, 10.0, LfoWaveform::Sawtooth, false).value();
        assert!((after - fresh).abs() < 1.0e-6);
    }
}
