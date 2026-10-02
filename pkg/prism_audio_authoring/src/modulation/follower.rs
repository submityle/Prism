//! Peak/`RMS`-style envelope follower for adaptive modulation.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the `EnvelopeFollowerModulator` of design section 12: it tracks
//! the level of a driving signal (for example a control bus carrying a music or
//! dialog bus level) so other parameters can duck or pump in response. Feeds
//! control buses in `super::control_bus`.

use bevy_math::ops;
use prism_audio_core::Sample;

/// A rectifying one-pole envelope follower with independent attack and release
/// time constants.
///
/// The follower rectifies its input and smooths it asymmetrically: fast when
/// the level is rising (attack) and slow when it is falling (release), the
/// standard shape for ducking and pump effects.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeFollower {
    level: Sample,
    attack_coef: Sample,
    release_coef: Sample,
}

impl EnvelopeFollower {
    /// Creates a follower with `attack_seconds` and `release_seconds` time
    /// constants at `sample_rate` Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(sample_rate: u32, attack_seconds: Sample, release_seconds: Sample) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        let mut follower = Self {
            level: 0.0,
            attack_coef: 0.0,
            release_coef: 0.0,
        };
        follower.set_times(sample_rate, attack_seconds, release_seconds);
        follower
    }

    /// Recomputes the attack/release coefficients for new time constants.
    pub fn set_times(
        &mut self,
        sample_rate: u32,
        attack_seconds: Sample,
        release_seconds: Sample,
    ) {
        self.attack_coef = one_pole_coef(sample_rate, attack_seconds);
        self.release_coef = one_pole_coef(sample_rate, release_seconds);
    }

    /// Returns the current tracked level.
    #[inline]
    #[must_use]
    pub fn level(&self) -> Sample {
        self.level
    }

    /// Processes one input sample and returns the updated tracked level.
    #[inline]
    pub fn process(&mut self, input: Sample) -> Sample {
        let rectified = input.abs();
        let coef = if rectified > self.level {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.level = rectified + (self.level - rectified) * coef;
        self.level
    }

    /// Resets the tracked level to silence.
    #[inline]
    pub fn reset(&mut self) {
        self.level = 0.0;
    }
}

/// Computes a one-pole smoothing coefficient `exp(-1 / (tau * sr))`.
#[inline]
fn one_pole_coef(sample_rate: u32, seconds: Sample) -> Sample {
    let tau = seconds.max(0.0);
    if tau <= 0.0 {
        return 0.0;
    }
    ops::exp(-1.0 / (tau * sample_rate as Sample))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    #[test]
    fn rises_toward_input_level() {
        let mut f = EnvelopeFollower::new(SR, 0.001, 0.1);
        for _ in 0..SR / 10 {
            f.process(1.0);
        }
        assert!(f.level() > 0.9, "level={}", f.level());
    }

    #[test]
    fn falls_after_input_stops() {
        let mut f = EnvelopeFollower::new(SR, 0.001, 0.01);
        for _ in 0..SR / 10 {
            f.process(1.0);
        }
        for _ in 0..SR / 10 {
            f.process(0.0);
        }
        assert!(f.level() < 0.1, "level={}", f.level());
    }

    #[test]
    fn reset_zeroes_level() {
        let mut f = EnvelopeFollower::new(SR, 0.01, 0.01);
        f.process(1.0);
        f.reset();
        assert!(f.level().abs() < 1.0e-9);
    }

    #[test]
    fn rectifies_negative_input() {
        let mut f = EnvelopeFollower::new(SR, 0.001, 0.1);
        for _ in 0..SR / 10 {
            f.process(-1.0);
        }
        assert!(f.level() > 0.9, "level={}", f.level());
    }
}
