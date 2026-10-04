//! Asymmetric attack/release envelope for category activity.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! CRIWARE, or Google Resonance Audio source or derived code; no AI/ML. The
//! one-pole attack/release follower is a classical DSP primitive implemented
//! independently here.
//!
//! # Relationship
//! Provides the attack/recovery ballistics named by design section 46.8. The
//! compiler (see [`super::compiler`]) builds one envelope per
//! [`super::rule::DuckingRule`]; the runtime (see [`super::runtime`]) feeds each
//! envelope the trigger category's raw activity and writes the smoothed result
//! onto a design section 12 modulation input bus. Smoothing is deterministic
//! and block-rate so it stays golden-reproducible (design section 24).

use prism_audio_core::Sample;

/// A one-pole follower with independent attack and release time constants.
///
/// Raw activity (nominally `0` idle, `1` active, but any non-negative level is
/// accepted) is low-pass filtered toward its target. Rising input uses the
/// attack coefficient; falling input uses the release coefficient, giving the
/// fast-engage/slow-recover feel of a mixing duck.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ActivityEnvelope {
    /// Sample rate in Hz used to translate time constants into coefficients.
    sample_rate: u32,
    /// Attack time constant in seconds (rising input).
    attack_seconds: Sample,
    /// Release time constant in seconds (falling input).
    release_seconds: Sample,
    /// Current smoothed level.
    level: Sample,
}

impl ActivityEnvelope {
    /// Builds an envelope at `sample_rate` with the given attack and release
    /// times in seconds, starting fully idle.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(sample_rate: u32, attack_seconds: Sample, release_seconds: Sample) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        Self {
            sample_rate,
            attack_seconds: attack_seconds.max(0.0),
            release_seconds: release_seconds.max(0.0),
            level: 0.0,
        }
    }

    /// Returns the current smoothed level without advancing time.
    #[must_use]
    pub fn level(&self) -> Sample {
        self.level
    }

    /// Resets the smoothed level to fully idle.
    pub fn reset(&mut self) {
        self.level = 0.0;
    }

    /// Advances the envelope over `frames` toward `raw` and returns the new
    /// level.
    ///
    /// The per-block pole coefficient is `exp(-frames / (tau * sample_rate))`
    /// where `tau` is the attack time while `raw` is above the current level
    /// and the release time otherwise. A zero time constant engages instantly.
    pub fn process_block(&mut self, raw: Sample, frames: u32) -> Sample {
        let tau = if raw > self.level {
            self.attack_seconds
        } else {
            self.release_seconds
        };
        let coef = block_coefficient(tau, frames, self.sample_rate);
        self.level = raw + coef * (self.level - raw);
        self.level
    }

    /// Updates the attack and release times, clamping them to be non-negative.
    pub fn set_times(&mut self, attack_seconds: Sample, release_seconds: Sample) {
        self.attack_seconds = attack_seconds.max(0.0);
        self.release_seconds = release_seconds.max(0.0);
    }
}

/// Computes the one-pole block coefficient `exp(-frames / (tau * fs))`.
///
/// Returns `0.0` (instant tracking) when `tau` is zero so a zero time constant
/// snaps immediately. The coefficient is in `[0, 1)`: smaller means faster
/// tracking.
#[must_use]
fn block_coefficient(tau_seconds: Sample, frames: u32, sample_rate: u32) -> Sample {
    if tau_seconds <= 0.0 {
        return 0.0;
    }
    let frames = frames as Sample;
    let fs = sample_rate as Sample;
    bevy_math::ops::exp(-frames / (tau_seconds * fs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn zero_attack_snaps_to_target() {
        let mut env = ActivityEnvelope::new(48_000, 0.0, 0.1);
        assert!(close(env.process_block(1.0, 64), 1.0));
    }

    #[test]
    fn attack_rises_monotonically_toward_one() {
        let mut env = ActivityEnvelope::new(48_000, 0.05, 0.2);
        let a = env.process_block(1.0, 256);
        let b = env.process_block(1.0, 256);
        let c = env.process_block(1.0, 256);
        assert!(a > 0.0 && a < 1.0);
        assert!(b > a && c > b);
        assert!(c < 1.0);
    }

    #[test]
    fn release_falls_back_toward_zero() {
        let mut env = ActivityEnvelope::new(48_000, 0.0, 0.1);
        env.process_block(1.0, 64); // snap to 1.0
        let a = env.process_block(0.0, 256);
        let b = env.process_block(0.0, 256);
        assert!(a < 1.0 && a > 0.0);
        assert!(b < a);
    }

    #[test]
    fn attack_is_faster_than_release_for_same_distance() {
        let mut fast = ActivityEnvelope::new(48_000, 0.01, 0.5);
        let mut slow = ActivityEnvelope::new(48_000, 0.5, 0.01);
        let f = fast.process_block(1.0, 256);
        let s = slow.process_block(1.0, 256);
        // The short-attack envelope engages further in one block.
        assert!(f > s);
    }
}
