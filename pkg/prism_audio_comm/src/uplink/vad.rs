//! Voice activity detection (VAD) for the uplink chain.
//!
//! The detector decides, per block, whether the signal contains speech so the
//! transport can gate silence (not sending comfort frames saves bandwidth) and
//! downstream perception/voice-management layers can react. It combines three
//! classic cues: short-term energy relative to an adaptively tracked noise
//! floor, the zero-crossing rate (which rises for unvoiced fricatives that
//! carry little energy), and a hangover counter that keeps the gate open
//! briefly after speech stops to avoid clipping word tails. No machine
//! learning is involved.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the VAD stage of design section 45.2 (uplink pre-processing).
//! Built on the sample scalar and deterministic math of `prism_audio_core`; it
//! is the final stage of [`crate::uplink::UplinkChain`] and gates the encoder.

use bevy_math::ops;
use prism_audio_core::math::{flush_denormal, Sample};

/// Tuning parameters for [`VoiceActivityDetector`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VadConfig {
    /// How far, in decibels, the block power must sit above the noise floor to
    /// count as speech on the energy cue alone.
    pub energy_margin_db: Sample,
    /// Zero-crossing rate (crossings per sample) above which the block is
    /// treated as a possible unvoiced consonant.
    pub zcr_high: Sample,
    /// Decibel margin above the floor required to confirm the zero-crossing
    /// cue, so broadband hiss alone does not trip it.
    pub zcr_energy_margin_db: Sample,
    /// Noise-floor adaptation coefficient in `[0, 1)` applied on non-speech
    /// blocks; closer to one adapts more slowly.
    pub floor_adapt: Sample,
    /// Number of non-speech blocks the gate stays open after speech ends.
    pub hangover_blocks: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_margin_db: 9.0,
            zcr_high: 0.25,
            zcr_energy_margin_db: 3.0,
            floor_adapt: 0.95,
            hangover_blocks: 8,
        }
    }
}

/// The result of analysing one block.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VadDecision {
    /// Whether the block is considered speech (including the hangover tail).
    pub is_speech: bool,
    /// Block power in decibels relative to full scale.
    pub energy_db: Sample,
    /// Zero-crossing rate in crossings per sample.
    pub zero_crossing_rate: Sample,
}

/// An energy / zero-crossing voice activity detector with hangover.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceActivityDetector {
    config: VadConfig,
    /// Adaptive noise-floor power estimate.
    floor_power: Sample,
    /// Remaining hangover blocks.
    hangover: usize,
    /// Whether a floor estimate has been seeded yet.
    initialized: bool,
}

impl VoiceActivityDetector {
    /// Creates a detector with the given configuration.
    #[must_use]
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            floor_power: 0.0,
            hangover: 0,
            initialized: false,
        }
    }

    /// Returns the active configuration.
    #[must_use]
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Returns the current noise-floor estimate in decibels.
    #[must_use]
    pub fn noise_floor_db(&self) -> Sample {
        power_to_db(self.floor_power)
    }

    /// Clears the adaptive floor and hangover state.
    pub fn reset(&mut self) {
        self.floor_power = 0.0;
        self.hangover = 0;
        self.initialized = false;
    }

    /// Analyses `block` and returns the detection result.
    ///
    /// The block is read only; VAD never alters the audio (gating is applied by
    /// the chain or transport based on [`VadDecision::is_speech`]).
    pub fn analyze(&mut self, block: &[Sample]) -> VadDecision {
        if block.is_empty() {
            return VadDecision {
                is_speech: false,
                energy_db: f32::NEG_INFINITY,
                zero_crossing_rate: 0.0,
            };
        }

        let mut power = 0.0;
        let mut crossings = 0usize;
        let mut prev = block[0];
        for &s in block {
            power += s * s;
            if (prev >= 0.0) != (s >= 0.0) {
                crossings += 1;
            }
            prev = s;
        }
        power /= block.len() as Sample;
        let zcr = crossings as Sample / block.len() as Sample;

        if !self.initialized {
            self.floor_power = power;
            self.initialized = true;
        }

        let floor = self.floor_power.max(1.0e-12);
        let ratio_db = 10.0 * ops::log10((power + 1.0e-12) / floor);
        let energy_active = ratio_db > self.config.energy_margin_db;
        let zcr_active = zcr > self.config.zcr_high && ratio_db > self.config.zcr_energy_margin_db;
        let active = energy_active || zcr_active;

        let is_speech = if active {
            self.hangover = self.config.hangover_blocks;
            true
        } else if self.hangover > 0 {
            self.hangover -= 1;
            true
        } else {
            false
        };

        // Adapt the floor only on genuinely non-speech blocks.
        if !active && !is_speech {
            let a = self.config.floor_adapt;
            self.floor_power = flush_denormal(a * self.floor_power + (1.0 - a) * power);
        } else if !active {
            // During hangover, let the floor creep slightly toward the signal
            // so a long trailing silence is eventually learned.
            let a = 0.5 * self.config.floor_adapt + 0.5;
            self.floor_power = flush_denormal(a * self.floor_power + (1.0 - a) * power);
        }

        VadDecision {
            is_speech,
            energy_db: power_to_db(power),
            zero_crossing_rate: zcr,
        }
    }
}

/// Converts a linear power value to decibels relative to full scale.
#[inline]
fn power_to_db(power: Sample) -> Sample {
    if power <= 1.0e-12 {
        f32::NEG_INFINITY
    } else {
        10.0 * ops::log10(power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::CommRng;
    use core::f32::consts::PI;

    #[test]
    fn silence_is_not_speech() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());
        let mut rng = CommRng::new(3);
        let mut last = true;
        for _ in 0..40 {
            let block: Vec<Sample> = (0..256).map(|_| rng.next_bipolar() * 0.001).collect();
            last = vad.analyze(&block).is_speech;
        }
        assert!(!last, "low-level noise should settle to non-speech");
    }

    #[test]
    fn loud_tone_is_speech() {
        let mut vad = VoiceActivityDetector::new(VadConfig::default());
        let mut rng = CommRng::new(3);
        // Learn a quiet floor first.
        for _ in 0..40 {
            let block: Vec<Sample> = (0..256).map(|_| rng.next_bipolar() * 0.001).collect();
            vad.analyze(&block);
        }
        let sr = 48_000.0;
        let block: Vec<Sample> = (0..256)
            .map(|i| 0.3 * ops::sin(2.0 * PI * 300.0 * i as Sample / sr))
            .collect();
        assert!(vad.analyze(&block).is_speech);
    }

    #[test]
    fn hangover_holds_gate_open() {
        let mut vad = VoiceActivityDetector::new(VadConfig {
            hangover_blocks: 5,
            ..VadConfig::default()
        });
        let mut rng = CommRng::new(9);
        for _ in 0..40 {
            let block: Vec<Sample> = (0..256).map(|_| rng.next_bipolar() * 0.001).collect();
            vad.analyze(&block);
        }
        let sr = 48_000.0;
        let speech: Vec<Sample> = (0..256)
            .map(|i| 0.3 * ops::sin(2.0 * PI * 300.0 * i as Sample / sr))
            .collect();
        assert!(vad.analyze(&speech).is_speech);
        // The first silent block right after speech is still gated open.
        let quiet: Vec<Sample> = (0..256).map(|_| rng.next_bipolar() * 0.001).collect();
        assert!(vad.analyze(&quiet).is_speech);
    }
}
