//! A single granular voice: one short windowed tone grain.
//!
//! Granular synthesis builds a texture from a dense cloud of very short
//! "grains". Because this engine is fully synthetic (no sampled source) each
//! grain is a short sinusoid at the grain's pitch, multiplied by a raised-cosine
//! (Hann) window so it fades in and out smoothly and never clicks at its edges.
//! Each grain carries pre-computed equal-power pan gains so a cloud can be
//! scattered across the stereo field. A grain is a tiny, `Copy`, allocation-free
//! state machine: it is triggered with its parameters, ticked once per sample
//! until its length elapses, and then reported inactive so a pool can reclaim
//! it.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The windowed-grain
//! formulation and the Hann window are standard, publicly documented granular
//! and spectral DSP.
//!
//! # Relationship
//! The atomic unit of design section 47.4; pooled by
//! [`crate::granular::pool::GrainPool`] and scheduled by
//! [`crate::granular::engine::GranularEngine`].

use bevy_math::ops;

use crate::dsp::TWO_PI;
use prism_audio_core::math::{equal_power_pan, Sample};

/// A single windowed-sinusoid grain.
#[derive(Clone, Copy, Debug)]
pub struct Grain {
    phase: Sample,
    phase_inc: Sample,
    pos: u32,
    length: u32,
    amplitude: Sample,
    gain_l: Sample,
    gain_r: Sample,
    active: bool,
}

impl Default for Grain {
    #[inline]
    fn default() -> Self {
        Self {
            phase: 0.0,
            phase_inc: 0.0,
            pos: 0,
            length: 0,
            amplitude: 0.0,
            gain_l: 0.0,
            gain_r: 0.0,
            active: false,
        }
    }
}

impl Grain {
    /// Creates an inactive grain.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// (Re)triggers the grain with its parameters.
    ///
    /// `freq_hz` is the grain tone frequency, `length_samples` its duration (and
    /// window span), `amplitude` its peak level, and `pan` its stereo position
    /// in `[-1, 1]`. A zero length leaves the grain inactive.
    pub fn trigger(
        &mut self,
        freq_hz: Sample,
        length_samples: u32,
        amplitude: Sample,
        pan: Sample,
        sample_rate: u32,
    ) {
        if length_samples == 0 || !freq_hz.is_finite() || !amplitude.is_finite() {
            self.active = false;
            return;
        }
        let fs = sample_rate.max(1) as Sample;
        let freq = freq_hz.clamp(1.0, 0.49 * fs);
        self.phase = 0.0;
        self.phase_inc = TWO_PI * freq / fs;
        self.pos = 0;
        self.length = length_samples;
        self.amplitude = amplitude;
        let (l, r) = equal_power_pan(pan.clamp(-1.0, 1.0));
        self.gain_l = l;
        self.gain_r = r;
        self.active = true;
    }

    /// Advances the grain one sample and returns its stereo contribution.
    ///
    /// Returns `(0.0, 0.0)` when the grain is inactive. The grain deactivates
    /// itself after its last sample.
    #[inline]
    pub fn tick(&mut self) -> (Sample, Sample) {
        if !self.active {
            return (0.0, 0.0);
        }
        let t = self.pos as Sample / self.length as Sample;
        // Hann window: zero at both ends, smooth, no clicks.
        let window = 0.5 - 0.5 * ops::cos(TWO_PI * t);
        let mono = ops::sin(self.phase) * self.amplitude * window;
        self.phase += self.phase_inc;
        if self.phase >= TWO_PI {
            self.phase -= TWO_PI;
        }
        self.pos += 1;
        if self.pos >= self.length {
            self.active = false;
        }
        (mono * self.gain_l, mono * self.gain_r)
    }

    /// Returns `true` while the grain is still sounding.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Returns the grain's completion fraction in `[0, 1]` (used for voice
    /// stealing: the most-complete grain is cheapest to steal).
    #[inline]
    #[must_use]
    pub fn progress(&self) -> Sample {
        if self.length == 0 {
            1.0
        } else {
            self.pos as Sample / self.length as Sample
        }
    }

    /// Deactivates the grain.
    #[inline]
    pub fn silence(&mut self) {
        self.active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grain_runs_then_stops() {
        let mut g = Grain::new();
        g.trigger(440.0, 128, 1.0, 0.0, 48_000);
        assert!(g.is_active());
        let mut peak = 0.0f32;
        for _ in 0..128 {
            let (l, _r) = g.tick();
            peak = peak.max(l.abs());
        }
        assert!(peak > 0.1, "peak={peak}");
        assert!(!g.is_active());
        assert_eq!(g.tick(), (0.0, 0.0));
    }

    #[test]
    fn window_starts_and_ends_near_zero() {
        let mut g = Grain::new();
        g.trigger(100.0, 256, 1.0, 0.0, 48_000);
        let (first, _) = g.tick();
        assert!(first.abs() < 1e-3, "first={first}");
    }

    #[test]
    fn pan_splits_channels() {
        let mut left = Grain::new();
        let mut right = Grain::new();
        left.trigger(440.0, 256, 1.0, -1.0, 48_000);
        right.trigger(440.0, 256, 1.0, 1.0, 48_000);
        let mut lsum = 0.0f32;
        let mut rsum = 0.0f32;
        for _ in 0..256 {
            let (ll, lr) = left.tick();
            let (rl, rr) = right.tick();
            lsum += ll.abs() - lr.abs();
            rsum += rr.abs() - rl.abs();
        }
        assert!(lsum > 0.0, "lsum={lsum}");
        assert!(rsum > 0.0, "rsum={rsum}");
    }

    #[test]
    fn zero_length_stays_inactive() {
        let mut g = Grain::new();
        g.trigger(440.0, 0, 1.0, 0.0, 48_000);
        assert!(!g.is_active());
    }
}
