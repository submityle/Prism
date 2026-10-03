//! A single modal resonator: one exponentially decaying sinusoid.
//!
//! Each resonant mode of a struck object is, acoustically, a sinusoid at the
//! mode frequency whose amplitude decays exponentially. The digital equivalent
//! is a two-pole resonator whose complex-conjugate poles sit at radius `r`
//! (the per-sample decay) and angle `omega` (the mode frequency). This type is
//! that resonator in Direct Form, stepped one sample at a time so a parallel
//! bank can be summed per sample. Feeding it a unit impulse produces the mode's
//! decaying sinusoid; feeding it a continuous signal (friction noise) colours
//! that signal with the mode's resonance.
//!
//! The feed-forward gain is set to `gain * sin(omega)` so the impulse-response
//! envelope peaks at approximately `gain` regardless of frequency, giving the
//! caller a predictable per-mode level.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The two-pole
//! resonator and its pole-radius/decay relation are standard, publicly
//! documented classic DSP.
//!
//! # Relationship
//! The atomic unit of design section 47.2; aggregated by
//! [`crate::modal::bank::ModalBank`]. Uses the half-life-to-radius helper from
//! [`crate::dsp`] rather than restating it.

use bevy_math::ops;

use crate::dsp::{clamp_frequency, half_life_to_radius, TWO_PI};
use prism_audio_core::math::{flush_denormal, Sample};

/// A two-pole resonator realising one decaying-sinusoid mode.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ModeResonator {
    /// First feedback coefficient, `2 * r * cos(omega)`.
    a1: Sample,
    /// Second feedback coefficient, `r^2`.
    a2: Sample,
    /// Feed-forward gain, `gain * sin(omega)`.
    b0: Sample,
    /// Previous output `y[n-1]`.
    y1: Sample,
    /// Output before last `y[n-2]`.
    y2: Sample,
    /// Impulse energy queued to inject on the next tick.
    pending: Sample,
}

impl Default for ModeResonator {
    #[inline]
    fn default() -> Self {
        Self {
            a1: 0.0,
            a2: 0.0,
            b0: 0.0,
            y1: 0.0,
            y2: 0.0,
            pending: 0.0,
        }
    }
}

impl ModeResonator {
    /// Creates a resonator tuned to `freq_hz` with the given amplitude
    /// `half_life_s` and output `gain` at `sample_rate`.
    #[inline]
    #[must_use]
    pub fn new(freq_hz: Sample, half_life_s: Sample, gain: Sample, sample_rate: u32) -> Self {
        let mut r = Self::default();
        r.set_params(freq_hz, half_life_s, gain, sample_rate);
        r
    }

    /// Retunes the resonator in place, preserving its ringing state so a
    /// parameter glide does not click.
    #[inline]
    pub fn set_params(
        &mut self,
        freq_hz: Sample,
        half_life_s: Sample,
        gain: Sample,
        sample_rate: u32,
    ) {
        let freq = clamp_frequency(freq_hz, sample_rate);
        let fs = sample_rate.max(1) as Sample;
        let omega = TWO_PI * freq / fs;
        let (sin, cos) = ops::sin_cos(omega);
        let r = half_life_to_radius(half_life_s, sample_rate);
        self.a1 = 2.0 * r * cos;
        self.a2 = r * r;
        let g = if gain.is_finite() { gain } else { 0.0 };
        self.b0 = g * sin;
    }

    /// Queues an excitation impulse of the given `energy` for the next tick.
    #[inline]
    pub fn excite(&mut self, energy: Sample) {
        if energy.is_finite() {
            self.pending += energy;
        }
    }

    /// Advances the resonator one sample, mixing in continuous `input` plus any
    /// queued impulse, and returns the output sample.
    #[inline]
    pub fn tick(&mut self, input: Sample) -> Sample {
        let drive = if input.is_finite() { input } else { 0.0 } + self.pending;
        self.pending = 0.0;
        let y = self.b0 * drive + self.a1 * self.y1 - self.a2 * self.y2;
        let y = flush_denormal(y);
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    /// Clears the ringing state and any queued impulse to silence.
    #[inline]
    pub fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
        self.pending = 0.0;
    }

    /// Returns `true` while the resonator still holds audible energy.
    #[inline]
    #[must_use]
    pub fn is_ringing(&self, threshold: Sample) -> bool {
        ops::abs(self.y1) > threshold || ops::abs(self.y2) > threshold || self.pending != 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn impulse_response(res: &mut ModeResonator, n: usize) -> Vec<Sample> {
        res.excite(1.0);
        (0..n).map(|_| res.tick(0.0)).collect()
    }

    #[test]
    fn rings_then_decays() {
        let mut r = ModeResonator::new(440.0, 0.3, 1.0, 48_000);
        let y = impulse_response(&mut r, 48_000);
        let early = y[..1000].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let late = y[40_000..].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(early > 0.1, "early={early}");
        assert!(late < early, "late={late} early={early}");
    }

    #[test]
    fn higher_half_life_rings_longer() {
        let mut short = ModeResonator::new(440.0, 0.1, 1.0, 48_000);
        let mut long = ModeResonator::new(440.0, 1.0, 1.0, 48_000);
        let ys = impulse_response(&mut short, 24_000);
        let yl = impulse_response(&mut long, 24_000);
        let tail_s = ys[20_000..].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let tail_l = yl[20_000..].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(tail_l > tail_s, "tail_l={tail_l} tail_s={tail_s}");
    }

    #[test]
    fn output_is_finite_and_bounded() {
        let mut r = ModeResonator::new(1_000.0, 2.0, 1.0, 48_000);
        r.excite(4.0);
        for _ in 0..96_000 {
            let y = r.tick(0.0);
            assert!(y.is_finite());
            assert!(y.abs() < 100.0);
        }
    }

    #[test]
    fn dead_mode_does_not_ring() {
        let mut r = ModeResonator::new(440.0, 0.0, 1.0, 48_000);
        let y = impulse_response(&mut r, 100);
        let tail = y[10..].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(tail < 1e-3, "tail={tail}");
    }

    #[test]
    fn reset_silences() {
        let mut r = ModeResonator::new(440.0, 1.0, 1.0, 48_000);
        r.excite(1.0);
        for _ in 0..100 {
            r.tick(0.0);
        }
        r.reset();
        assert!(!r.is_ringing(1e-6));
        assert_eq!(r.tick(0.0), 0.0);
    }
}
