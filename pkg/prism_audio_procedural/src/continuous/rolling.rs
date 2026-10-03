//! Rolling contact: a deterministic contact-pulse train exciting the modes.
//!
//! A body rolling across a surface strikes it as a rapid train of tiny contact
//! impulses, one per surface asperity it rides over. The audible pitch and
//! density of that train scale with how fast the body travels and how many
//! grains per metre the surface presents (its rolling density). This generator
//! produces exactly that: a phase accumulator advances at a pulse rate of
//! `speed * density` grains per second and emits a short impulse each time it
//! wraps. The impulse amplitude carries a deterministic, roughness-scaled
//! jitter so the train is lively rather than metronomic, yet an identical seed
//! and input sequence reproduce it exactly. The impulses are meant to be summed
//! into the object's modal bank, which turns each bare impulse into a short
//! resonant "tick".
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the rolling clause of design section 47.3; its pulse density
//! comes from [`crate::material::FrictionTemplate`], its jitter from
//! [`crate::rng`], and its output excites the modal bank through
//! [`crate::continuous::ContactVoice`].

use crate::rng::ProceduralRng;
use prism_audio_core::math::Sample;
use prism_audio_core::param::{Ramp, Smoothed};

/// A deterministic rolling contact-pulse generator.
#[derive(Clone, Debug)]
pub struct RollingGenerator {
    rng: ProceduralRng,
    phase: Sample,
    rate_hz: Smoothed,
    amplitude: Smoothed,
    roughness: Sample,
    sample_rate: u32,
    smoothing: Ramp,
}

impl RollingGenerator {
    /// Creates a rolling generator seeded with `seed` at `sample_rate`, idle.
    #[must_use]
    pub fn new(seed: u64, sample_rate: u32) -> Self {
        let sample_rate = sample_rate.max(1);
        Self {
            rng: ProceduralRng::new(seed),
            phase: 0.0,
            rate_hz: Smoothed::new(0.0),
            amplitude: Smoothed::new(0.0),
            roughness: 0.5,
            sample_rate,
            smoothing: Ramp::linear_seconds(0.01, sample_rate),
        }
    }

    /// Updates the pulse train from the physical contact quantities.
    ///
    /// The pulse rate is `tangential_speed * density` grains per second (clamped
    /// below half the sample rate so a pulse cannot land every sample), the
    /// amplitude tracks normal pressure, and `roughness` sets the amplitude
    /// jitter spread.
    pub fn update(
        &mut self,
        tangential_speed: Sample,
        normal_pressure: Sample,
        roughness: Sample,
        density: Sample,
    ) {
        let speed = if tangential_speed.is_finite() {
            tangential_speed.max(0.0)
        } else {
            0.0
        };
        let pressure = if normal_pressure.is_finite() {
            normal_pressure.max(0.0)
        } else {
            0.0
        };
        self.roughness = if roughness.is_finite() {
            roughness.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let density = if density.is_finite() {
            density.max(0.0)
        } else {
            0.0
        };

        let nyquist = 0.5 * self.sample_rate as Sample;
        let rate = (speed * density).clamp(0.0, nyquist);
        self.rate_hz.set_target(rate, self.smoothing);
        // Louder, denser contact when pressed harder; bounded.
        let target = 0.25 + 0.75 * (pressure / (pressure + 1.0));
        self.amplitude.set_target(target, self.smoothing);
    }

    /// Immediately targets silence; the amplitude still glides to zero.
    #[inline]
    pub fn silence(&mut self) {
        self.amplitude.set_target(0.0, self.smoothing);
        self.rate_hz.set_target(0.0, self.smoothing);
    }

    /// Returns the next pulse-train sample: a jittered impulse on a grain
    /// boundary, otherwise `0.0`.
    #[inline]
    pub fn tick(&mut self) -> Sample {
        let rate = self.rate_hz.next_sample();
        let amp = self.amplitude.next_sample();
        let fs = self.sample_rate as Sample;
        self.phase += rate / fs;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
            if self.phase >= 1.0 {
                // Extremely high rate: clamp to one pulse per sample.
                self.phase = 0.0;
            }
            let jitter = 1.0 + self.roughness * self.rng.next_bipolar() * 0.6;
            amp * jitter.max(0.0)
        } else {
            0.0
        }
    }

    /// Returns the current smoothed amplitude, for liveness checks.
    #[inline]
    #[must_use]
    pub fn current_amplitude(&self) -> Sample {
        self.amplitude.current()
    }

    /// Resets the phase and silences immediately.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.amplitude.set_target(0.0, Ramp::Immediate);
        self.rate_hz.set_target(0.0, Ramp::Immediate);
        let _ = self.amplitude.next_sample();
        let _ = self.rate_hz.next_sample();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_pulses(roller: &mut RollingGenerator, n: usize) -> usize {
        let mut pulses = 0;
        for _ in 0..n {
            if roller.tick() != 0.0 {
                pulses += 1;
            }
        }
        pulses
    }

    #[test]
    fn faster_rolls_denser() {
        let mut slow = RollingGenerator::new(1, 48_000);
        let mut fast = RollingGenerator::new(1, 48_000);
        slow.update(1.0, 1.0, 0.5, 60.0);
        fast.update(5.0, 1.0, 0.5, 60.0);
        // Warm the smoothers.
        for _ in 0..2_000 {
            slow.tick();
            fast.tick();
        }
        let slow_n = count_pulses(&mut slow, 48_000);
        let fast_n = count_pulses(&mut fast, 48_000);
        assert!(fast_n > slow_n, "slow={slow_n} fast={fast_n}");
    }

    #[test]
    fn stationary_is_silent() {
        let mut roller = RollingGenerator::new(2, 48_000);
        roller.update(0.0, 1.0, 0.5, 60.0);
        let n = count_pulses(&mut roller, 48_000);
        assert_eq!(n, 0);
    }

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = RollingGenerator::new(5, 48_000);
        let mut b = RollingGenerator::new(5, 48_000);
        a.update(3.0, 1.0, 0.7, 80.0);
        b.update(3.0, 1.0, 0.7, 80.0);
        for _ in 0..4_000 {
            assert_eq!(a.tick(), b.tick());
        }
    }

    #[test]
    fn output_is_finite() {
        let mut roller = RollingGenerator::new(3, 48_000);
        roller.update(f32::NAN, f32::INFINITY, 2.0, f32::NAN);
        for _ in 0..2_000 {
            assert!(roller.tick().is_finite());
        }
    }
}
