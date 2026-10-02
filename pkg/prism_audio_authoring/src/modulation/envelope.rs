//! Sample-accurate `AHDSR` / `ADSR` envelope generator.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the triggered-envelope requirement of design section 12
//! (`AdsrModulator`), extended to the attack-hold-decay-sustain-release form.
//! Segment shapes reuse `super::curve::Curve`; the generator also implements
//! `super::source::Modulator` so it can drive the modulation matrix.

use prism_audio_core::Sample;

use super::curve::Curve;
use super::source::{ModContext, Modulator};

/// The piecewise stage an [`Envelope`] is currently traversing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum EnvelopeStage {
    /// Fully released and silent; awaiting a gate.
    Idle,
    /// Rising from the current level toward unity.
    Attack,
    /// Held at unity for the configured hold time.
    Hold,
    /// Falling from unity toward the sustain level.
    Decay,
    /// Held at the sustain level while the gate is open.
    Sustain,
    /// Falling from the current level toward silence.
    Release,
}

/// Timing and shape configuration for an [`Envelope`].
///
/// All times are in seconds and are clamped to be non-negative. A zero-length
/// stage is skipped instantaneously, so setting `hold_seconds` to zero yields a
/// classic `ADSR` while a positive value yields an `AHDSR`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvelopeConfig {
    /// Attack time from the current level to unity, in seconds.
    pub attack_seconds: Sample,
    /// Hold time at unity, in seconds.
    pub hold_seconds: Sample,
    /// Decay time from unity to the sustain level, in seconds.
    pub decay_seconds: Sample,
    /// Steady-state level held while the gate is open, in `[0, 1]`.
    pub sustain_level: Sample,
    /// Release time from the current level to silence, in seconds.
    pub release_seconds: Sample,
    /// Shape applied across the attack segment.
    pub attack_curve: Curve,
    /// Shape applied across the decay segment.
    pub decay_curve: Curve,
    /// Shape applied across the release segment.
    pub release_curve: Curve,
}

impl Default for EnvelopeConfig {
    fn default() -> Self {
        Self {
            attack_seconds: 0.005,
            hold_seconds: 0.0,
            decay_seconds: 0.1,
            sustain_level: 0.7,
            release_seconds: 0.2,
            attack_curve: Curve::Linear,
            decay_curve: Curve::Exponential { curvature: -3.0 },
            release_curve: Curve::Exponential { curvature: -3.0 },
        }
    }
}

/// A gate-driven envelope generator evaluated one sample at a time.
///
/// The output level is always in `[0, 1]`. Opening the gate restarts the attack
/// from the current level (supporting legato retriggers); closing it begins the
/// release from wherever the level currently sits.
#[derive(Debug, Clone)]
pub struct Envelope {
    config: EnvelopeConfig,
    sample_rate: u32,
    stage: EnvelopeStage,
    level: Sample,
    gate: bool,
    /// Samples elapsed within the active ramping stage.
    stage_pos: u32,
    /// Total samples in the active ramping stage.
    stage_len: u32,
    /// Level captured when the active ramping stage began.
    stage_start: Sample,
}

impl Envelope {
    /// Creates an idle envelope for `config` at `sample_rate` Hz.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(sample_rate: u32, config: EnvelopeConfig) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        Self {
            config,
            sample_rate,
            stage: EnvelopeStage::Idle,
            level: 0.0,
            gate: false,
            stage_pos: 0,
            stage_len: 0,
            stage_start: 0.0,
        }
    }

    /// Returns the current stage.
    #[inline]
    #[must_use]
    pub fn stage(&self) -> EnvelopeStage {
        self.stage
    }

    /// Returns the instantaneous output level in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn level(&self) -> Sample {
        self.level
    }

    /// Returns `true` while the envelope is producing a non-idle signal.
    #[inline]
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.stage != EnvelopeStage::Idle
    }

    /// Replaces the configuration without disturbing the active stage level.
    ///
    /// The change takes effect the next time a stage (re)starts, so a running
    /// release is not retimed mid-flight.
    pub fn set_config(&mut self, config: EnvelopeConfig) {
        self.config = config;
    }

    /// Opens or closes the gate.
    ///
    /// A rising edge begins the attack; a falling edge begins the release. A
    /// level already matching the requested gate state is a no-op.
    pub fn set_gate(&mut self, gate: bool) {
        if gate && !self.gate {
            self.gate = true;
            self.begin_attack();
        } else if !gate && self.gate {
            self.gate = false;
            self.begin_release();
        }
    }

    /// Forces a fresh attack from the current level, as for a legato retrigger.
    pub fn retrigger(&mut self) {
        self.gate = true;
        self.begin_attack();
    }

    fn seconds_to_samples(&self, seconds: Sample) -> u32 {
        let n = seconds.max(0.0) * self.sample_rate as Sample;
        // Round to the nearest sample; keep as an integer count.
        (n + 0.5) as u32
    }

    fn begin_attack(&mut self) {
        self.stage = EnvelopeStage::Attack;
        self.stage_pos = 0;
        self.stage_len = self.seconds_to_samples(self.config.attack_seconds);
        self.stage_start = self.level;
    }

    fn begin_hold(&mut self) {
        self.level = 1.0;
        let hold = self.seconds_to_samples(self.config.hold_seconds);
        if hold == 0 {
            self.begin_decay();
        } else {
            self.stage = EnvelopeStage::Hold;
            self.stage_pos = 0;
            self.stage_len = hold;
        }
    }

    fn begin_decay(&mut self) {
        self.stage = EnvelopeStage::Decay;
        self.stage_pos = 0;
        self.stage_len = self.seconds_to_samples(self.config.decay_seconds);
        self.stage_start = self.level;
    }

    fn begin_sustain(&mut self) {
        self.stage = EnvelopeStage::Sustain;
        self.level = self.config.sustain_level.clamp(0.0, 1.0);
        self.stage_pos = 0;
        self.stage_len = 0;
    }

    fn begin_release(&mut self) {
        self.stage = EnvelopeStage::Release;
        self.stage_pos = 0;
        self.stage_len = self.seconds_to_samples(self.config.release_seconds);
        self.stage_start = self.level;
    }

    /// Advances one sample and returns the new output level.
    pub fn next_sample(&mut self) -> Sample {
        match self.stage {
            EnvelopeStage::Idle => {
                self.level = 0.0;
            }
            EnvelopeStage::Attack => {
                if self.stage_len == 0 {
                    self.level = 1.0;
                    self.begin_hold();
                } else {
                    let phase = self.stage_pos as Sample / self.stage_len as Sample;
                    let shaped = self.config.attack_curve.map(phase);
                    self.level = self.stage_start + (1.0 - self.stage_start) * shaped;
                    self.stage_pos += 1;
                    if self.stage_pos >= self.stage_len {
                        self.begin_hold();
                    }
                }
            }
            EnvelopeStage::Hold => {
                self.level = 1.0;
                self.stage_pos += 1;
                if self.stage_pos >= self.stage_len {
                    self.begin_decay();
                }
            }
            EnvelopeStage::Decay => {
                let sustain = self.config.sustain_level.clamp(0.0, 1.0);
                if self.stage_len == 0 {
                    self.begin_sustain();
                } else {
                    let phase = self.stage_pos as Sample / self.stage_len as Sample;
                    let shaped = self.config.decay_curve.map(phase);
                    self.level = self.stage_start + (sustain - self.stage_start) * shaped;
                    self.stage_pos += 1;
                    if self.stage_pos >= self.stage_len {
                        self.begin_sustain();
                    }
                }
            }
            EnvelopeStage::Sustain => {
                self.level = self.config.sustain_level.clamp(0.0, 1.0);
            }
            EnvelopeStage::Release => {
                if self.stage_len == 0 {
                    self.level = 0.0;
                    self.stage = EnvelopeStage::Idle;
                } else {
                    let phase = self.stage_pos as Sample / self.stage_len as Sample;
                    let shaped = self.config.release_curve.map(phase);
                    self.level = self.stage_start * (1.0 - shaped);
                    self.stage_pos += 1;
                    if self.stage_pos >= self.stage_len {
                        self.level = 0.0;
                        self.stage = EnvelopeStage::Idle;
                    }
                }
            }
        }
        self.level
    }
}

impl Modulator for Envelope {
    fn tick(&mut self, ctx: &ModContext) -> Sample {
        for _ in 0..ctx.frames {
            self.next_sample();
        }
        self.level
    }

    fn value(&self) -> Sample {
        self.level
    }

    fn reset(&mut self) {
        self.stage = EnvelopeStage::Idle;
        self.level = 0.0;
        self.gate = false;
        self.stage_pos = 0;
        self.stage_len = 0;
        self.stage_start = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;
    const EPS: Sample = 1.0e-3;

    fn test_config() -> EnvelopeConfig {
        EnvelopeConfig {
            attack_seconds: 0.01,
            hold_seconds: 0.0,
            decay_seconds: 0.02,
            sustain_level: 0.5,
            release_seconds: 0.02,
            attack_curve: Curve::Linear,
            decay_curve: Curve::Linear,
            release_curve: Curve::Linear,
        }
    }

    #[test]
    fn idle_is_silent() {
        let mut env = Envelope::new(SR, test_config());
        for _ in 0..100 {
            assert!(env.next_sample().abs() < EPS);
        }
    }

    #[test]
    fn attack_rises_to_unity() {
        let mut env = Envelope::new(SR, test_config());
        env.set_gate(true);
        let attack = (0.01 * SR as Sample) as usize;
        for _ in 0..attack {
            env.next_sample();
        }
        assert!((env.level() - 1.0).abs() < 2.0e-2, "level={}", env.level());
    }

    #[test]
    fn decays_to_sustain_and_holds() {
        let mut env = Envelope::new(SR, test_config());
        env.set_gate(true);
        // Run through attack + decay.
        let n = (0.05 * SR as Sample) as usize;
        for _ in 0..n {
            env.next_sample();
        }
        assert_eq!(env.stage(), EnvelopeStage::Sustain);
        assert!((env.level() - 0.5).abs() < EPS, "level={}", env.level());
    }

    #[test]
    fn release_falls_to_zero() {
        let mut env = Envelope::new(SR, test_config());
        env.set_gate(true);
        for _ in 0..(0.05 * SR as Sample) as usize {
            env.next_sample();
        }
        env.set_gate(false);
        for _ in 0..(0.05 * SR as Sample) as usize {
            env.next_sample();
        }
        assert_eq!(env.stage(), EnvelopeStage::Idle);
        assert!(env.level().abs() < EPS);
    }

    #[test]
    fn hold_stage_delays_decay() {
        let mut cfg = test_config();
        cfg.hold_seconds = 0.02;
        let mut env = Envelope::new(SR, cfg);
        env.set_gate(true);
        let attack = (0.01 * SR as Sample) as usize;
        for _ in 0..attack + 2 {
            env.next_sample();
        }
        assert_eq!(env.stage(), EnvelopeStage::Hold);
        assert!((env.level() - 1.0).abs() < EPS);
    }

    #[test]
    fn reset_returns_to_idle() {
        let mut env = Envelope::new(SR, test_config());
        env.set_gate(true);
        for _ in 0..50 {
            env.next_sample();
        }
        env.reset();
        assert_eq!(env.stage(), EnvelopeStage::Idle);
        assert!(env.level().abs() < EPS);
    }

    #[test]
    fn tick_advances_by_frames() {
        let mut env = Envelope::new(SR, test_config());
        env.set_gate(true);
        let ctx = ModContext::new(SR, 64);
        let a = env.tick(&ctx);
        assert!(a > 0.0);
        assert!(env.is_active());
    }
}
