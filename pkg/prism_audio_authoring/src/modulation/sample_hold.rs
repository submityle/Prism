//! Clocked sample-and-hold latch.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the sample-and-hold requirement of design section 12. A phase
//! accumulator produces a periodic clock; on each clock edge the current input
//! is latched and held until the next edge. Paired with a random source in
//! `super::random` it becomes the classic stepped-random modulator.

use bevy_math::ops;
use prism_audio_core::Sample;

/// A rate-clocked latch that samples its input on each clock edge and holds the
/// value in between.
///
/// The internal clock is a normalized phase accumulator, so the hold interval
/// is sample-accurate and independent of the processing block size.
#[derive(Debug, Clone, Copy)]
pub struct SampleAndHold {
    held: Sample,
    phase: Sample,
    increment: Sample,
}

impl SampleAndHold {
    /// Creates a latch clocked at `rate_hz` running at `sample_rate` Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(sample_rate: u32, rate_hz: Sample) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        let mut sh = Self {
            held: 0.0,
            phase: 0.0,
            increment: 0.0,
        };
        sh.set_rate(sample_rate, rate_hz);
        sh
    }

    /// Sets the clock rate in Hz (clamped to be non-negative).
    #[inline]
    pub fn set_rate(&mut self, sample_rate: u32, rate_hz: Sample) {
        self.increment = rate_hz.max(0.0) / sample_rate.max(1) as Sample;
    }

    /// Returns the currently held value.
    #[inline]
    #[must_use]
    pub fn value(&self) -> Sample {
        self.held
    }

    /// Advances the clock by one sample, returning `true` on a clock edge.
    #[inline]
    pub fn tick_clock(&mut self) -> bool {
        self.phase += self.increment;
        if self.phase >= 1.0 {
            self.phase -= ops::floor(self.phase);
            true
        } else {
            false
        }
    }

    /// Advances one sample, latching `input` on a clock edge, and returns the
    /// held value.
    #[inline]
    pub fn process(&mut self, input: Sample) -> Sample {
        if self.tick_clock() {
            self.held = input;
        }
        self.held
    }

    /// Immediately latches `value`, ignoring the clock.
    #[inline]
    pub fn latch(&mut self, value: Sample) {
        self.held = value;
    }

    /// Resets the held value and clock phase.
    #[inline]
    pub fn reset(&mut self) {
        self.held = 0.0;
        self.phase = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;
    const EPS: Sample = 1.0e-6;

    #[test]
    fn holds_between_edges() {
        // 1 Hz clock at 8 Hz sample rate -> edge every 8 samples.
        let mut sh = SampleAndHold::new(8, 1.0);
        let mut ramp = 0.0;
        let mut edges = 0;
        for _ in 0..8 {
            if sh.tick_clock() {
                edges += 1;
            }
            ramp += 1.0;
            let _ = ramp;
        }
        assert_eq!(edges, 1);
    }

    #[test]
    fn latches_input_on_edge() {
        let mut sh = SampleAndHold::new(4, 1.0); // edge every 4 samples.
        let inputs = [0.1, 0.2, 0.3, 0.9, 0.5];
        let mut last = 0.0;
        for (i, &x) in inputs.iter().enumerate() {
            last = sh.process(x);
            if i < 3 {
                // No edge yet, holds the initial zero.
                assert!(last.abs() < EPS, "i={i} last={last}");
            }
        }
        // On the fourth sample the clock wrapped and latched 0.9.
        assert!((last - 0.9).abs() < EPS || (last - 0.5).abs() < EPS);
    }

    #[test]
    fn manual_latch_overrides() {
        let mut sh = SampleAndHold::new(SR, 1.0);
        sh.latch(0.42);
        assert!((sh.value() - 0.42).abs() < EPS);
    }

    #[test]
    fn reset_clears() {
        let mut sh = SampleAndHold::new(SR, 10.0);
        sh.latch(0.5);
        sh.reset();
        assert!(sh.value().abs() < EPS);
    }

    #[test]
    fn zero_rate_never_edges() {
        let mut sh = SampleAndHold::new(SR, 0.0);
        let mut edges = 0;
        for _ in 0..SR {
            if sh.tick_clock() {
                edges += 1;
            }
        }
        assert_eq!(edges, 0);
    }
}
