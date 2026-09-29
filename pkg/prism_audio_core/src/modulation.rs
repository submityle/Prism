//! Control-rate modulation sources.
//!
//! A low-frequency oscillator (LFO) drives time-varying parameters such as the
//! delay time of a chorus/flanger or the all-pass coefficient of a phaser. It
//! runs at the audio sample rate but at sub-audio frequencies, so its output is
//! read once per sample and used as a control signal rather than mixed into the
//! audio itself.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::math::Sample;

/// Shape of an [`Lfo`]'s output waveform (all bounded to `[-1, 1]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum LfoWaveform {
    /// Smooth sinusoid; the most common modulation shape.
    Sine,
    /// Symmetric triangle: linear up then linear down.
    Triangle,
    /// Rising ramp from `-1` to `1`, then a discontinuous reset.
    Sawtooth,
    /// Bipolar square wave: `+1` for the first half of the cycle, `-1` after.
    Square,
}

/// A phase-accumulator low-frequency oscillator producing a bipolar control
/// signal in `[-1, 1]`.
///
/// The whole struct is `Copy` and allocation-free, so it can be embedded
/// directly inside real-time [`AudioNode`](crate::graph::AudioNode)s.
#[derive(Debug, Clone, Copy)]
pub struct Lfo {
    /// Normalized phase in `[0, 1)`.
    phase: Sample,
    /// Per-sample phase increment (`frequency / sample_rate`).
    increment: Sample,
    /// Output waveform shape.
    waveform: LfoWaveform,
}

impl Lfo {
    /// Creates an LFO at `frequency_hz` running at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, frequency_hz: Sample, waveform: LfoWaveform) -> Self {
        let mut lfo = Self {
            phase: 0.0,
            increment: 0.0,
            waveform,
        };
        lfo.set_frequency(sample_rate, frequency_hz);
        lfo
    }

    /// Sets the oscillation frequency (clamped to be non-negative).
    #[inline]
    pub fn set_frequency(&mut self, sample_rate: u32, frequency_hz: Sample) {
        let sr = (sample_rate.max(1)) as Sample;
        self.increment = frequency_hz.max(0.0) / sr;
    }

    /// Sets the output waveform shape.
    #[inline]
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.waveform = waveform;
    }

    /// Offsets the phase to `phase01` (wrapped into `[0, 1)`); useful to spread
    /// several LFOs across a cycle for multi-voice chorus.
    #[inline]
    pub fn set_phase(&mut self, phase01: Sample) {
        let p = phase01 - ops::floor(phase01);
        self.phase = p;
    }

    /// Resets the phase to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Evaluates the current phase without advancing.
    #[inline]
    #[must_use]
    pub fn value(&self) -> Sample {
        match self.waveform {
            LfoWaveform::Sine => ops::sin(TAU * self.phase),
            // Triangle: peaks at phase 0/1, trough at 0.5.
            LfoWaveform::Triangle => 4.0 * (self.phase - 0.5).abs() - 1.0,
            LfoWaveform::Sawtooth => 2.0 * self.phase - 1.0,
            LfoWaveform::Square => {
                if self.phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }

    /// Returns the current sample and advances the phase by one step.
    #[inline]
    pub fn next_sample(&mut self) -> Sample {
        let v = self.value();
        self.phase += self.increment;
        if self.phase >= 1.0 {
            self.phase -= ops::floor(self.phase);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_starts_at_zero_and_is_bounded() {
        let mut lfo = Lfo::new(48_000, 100.0, LfoWaveform::Sine);
        assert!(lfo.value().abs() < 1e-6);
        for _ in 0..48_000 {
            let v = lfo.next_sample();
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn sawtooth_ramps_between_bounds() {
        let mut lfo = Lfo::new(8, 1.0, LfoWaveform::Sawtooth);
        // 1 Hz at 8 Hz sample rate -> 8 steps per cycle.
        let first = lfo.next_sample();
        assert!((first + 1.0).abs() < 1e-6, "{first}");
        for _ in 0..3 {
            lfo.next_sample();
        }
        // Near mid-cycle the ramp crosses ~0.
        assert!(lfo.value().abs() < 1e-6);
    }

    #[test]
    fn square_is_bipolar() {
        let mut lfo = Lfo::new(8, 1.0, LfoWaveform::Square);
        assert_eq!(lfo.next_sample(), 1.0);
        for _ in 0..3 {
            lfo.next_sample();
        }
        assert_eq!(lfo.value(), -1.0);
    }

    #[test]
    fn phase_offset_wraps() {
        let mut lfo = Lfo::new(48_000, 1.0, LfoWaveform::Sawtooth);
        lfo.set_phase(1.25);
        // 1.25 wraps to 0.25 -> sawtooth value 2*0.25 - 1 = -0.5.
        assert!((lfo.value() + 0.5).abs() < 1e-6);
    }

    #[test]
    fn frequency_zero_is_static() {
        let mut lfo = Lfo::new(48_000, 0.0, LfoWaveform::Sine);
        let a = lfo.next_sample();
        let b = lfo.next_sample();
        assert_eq!(a, b);
    }
}
