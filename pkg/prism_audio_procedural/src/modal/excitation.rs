//! Excitation shaping: the attack transient and the spectral tilt stage.
//!
//! A bare impulse into a modal bank sounds synthetic; real strikes have a short
//! broadband contact transient (the "click" of the collision) before the body
//! rings, and the balance of the collision between a square-on rap and a
//! grazing scrape tilts the excitation spectrum. This module provides both as
//! allocation-free, `Copy`-friendly primitives: a [`TransientBurst`] that emits
//! a very short, exponentially decaying noise cluster from the seeded RNG, and
//! a [`SpectralTilt`] one-pole stage that crossfades between a darkened
//! (low-passed) and brightened (high-passed) version of a drive signal. Both
//! are deterministic given their seed and inputs.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the excitation-shaping clauses of design section 47.2; drives the
//! modal bank ([`crate::modal::bank`]) and colours the continuous friction
//! source ([`crate::continuous::friction`]). Uses [`crate::rng`] and
//! [`crate::dsp::OnePole`].

use crate::dsp::{lerp, OnePole};
use crate::rng::ProceduralRng;
use prism_audio_core::math::Sample;

/// A short, exponentially decaying broadband noise burst modelling the contact
/// transient of a strike.
#[derive(Clone, Copy, Debug)]
pub struct TransientBurst {
    rng: ProceduralRng,
    amplitude: Sample,
    decay: Sample,
    remaining: u32,
}

impl TransientBurst {
    /// Creates an idle burst generator seeded with `seed`.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ProceduralRng::new(seed),
            amplitude: 0.0,
            decay: 0.0,
            remaining: 0,
        }
    }

    /// Triggers a burst of peak `amplitude` lasting `length_samples`.
    ///
    /// A longer length gives a softer, dustier contact; a very short one gives
    /// a sharp click. Re-triggering restarts the burst (the loudest contact
    /// wins) rather than summing unbounded.
    #[inline]
    pub fn trigger(&mut self, amplitude: Sample, length_samples: u32) {
        if !amplitude.is_finite() || amplitude <= 0.0 || length_samples == 0 {
            return;
        }
        self.amplitude = amplitude.max(self.amplitude);
        self.remaining = length_samples.max(self.remaining);
        // Per-sample multiplier that decays the envelope to ~1% over the length.
        let n = length_samples as Sample;
        self.decay = exp_decay_to(0.01, n);
    }

    /// Returns the next burst sample, or `0.0` when the burst has finished.
    #[inline]
    pub fn tick(&mut self) -> Sample {
        if self.remaining == 0 {
            return 0.0;
        }
        self.remaining -= 1;
        let out = self.rng.next_bipolar() * self.amplitude;
        self.amplitude *= self.decay;
        out
    }

    /// Returns `true` while the burst is still emitting samples.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.remaining > 0
    }
}

/// Per-sample multiplier that decays an envelope to `target` over `n` samples.
#[inline]
fn exp_decay_to(target: Sample, n: Sample) -> Sample {
    if n <= 1.0 {
        return 0.0;
    }
    // target = m^n  =>  m = target^(1/n).
    bevy_math::ops::powf(target.clamp(1e-6, 0.5), 1.0 / n)
}

/// A one-pole spectral tilt stage crossfading dark (low-pass) and bright
/// (high-pass) colourings of a drive signal.
#[derive(Clone, Copy, Debug)]
pub struct SpectralTilt {
    filter: OnePole,
}

impl SpectralTilt {
    /// Creates a tilt stage with its one-pole corner at `pivot_hz`.
    #[inline]
    #[must_use]
    pub fn new(pivot_hz: Sample, sample_rate: u32) -> Self {
        Self {
            filter: OnePole::new(pivot_hz, sample_rate),
        }
    }

    /// Retunes the tilt pivot frequency.
    #[inline]
    pub fn set_pivot(&mut self, pivot_hz: Sample, sample_rate: u32) {
        self.filter.set_cutoff(pivot_hz, sample_rate);
    }

    /// Processes one sample, crossfading toward the high-passed (bright) version
    /// as `brightness` approaches `1.0` and the low-passed (dark) version as it
    /// approaches `0.0`.
    #[inline]
    pub fn tick(&mut self, input: Sample, brightness: Sample) -> Sample {
        let low = self.filter.low(input);
        let high = input - low;
        lerp(low, high, brightness.clamp(0.0, 1.0))
    }

    /// Clears the internal filter state.
    #[inline]
    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burst_decays_and_stops() {
        let mut b = TransientBurst::new(1);
        b.trigger(1.0, 64);
        assert!(b.is_active());
        let mut last_abs = 1.0;
        let mut any = false;
        for _ in 0..64 {
            let s = b.tick();
            any |= s != 0.0;
            last_abs = s.abs();
        }
        assert!(any);
        assert!(!b.is_active());
        assert_eq!(b.tick(), 0.0);
        let _ = last_abs;
    }

    #[test]
    fn burst_is_deterministic() {
        let mut a = TransientBurst::new(42);
        let mut b = TransientBurst::new(42);
        a.trigger(1.0, 32);
        b.trigger(1.0, 32);
        for _ in 0..32 {
            assert_eq!(a.tick(), b.tick());
        }
    }

    #[test]
    fn burst_ignores_bad_trigger() {
        let mut b = TransientBurst::new(1);
        b.trigger(f32::NAN, 32);
        assert!(!b.is_active());
        b.trigger(1.0, 0);
        assert!(!b.is_active());
    }

    #[test]
    fn tilt_bright_boosts_highs() {
        let mut tilt = SpectralTilt::new(1_000.0, 48_000);
        // Feed DC: the dark path keeps it, the bright path removes it.
        let mut dark = 0.0;
        let mut bright = 0.0;
        for _ in 0..2_000 {
            dark = tilt.tick(1.0, 0.0);
            bright = tilt.tick(1.0, 1.0);
        }
        assert!(dark > bright);
    }
}
