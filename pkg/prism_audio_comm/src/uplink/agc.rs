//! Automatic gain control (AGC) for the uplink chain.
//!
//! Microphone levels vary widely with distance, gain staging, and talker
//! loudness. The AGC drives the signal toward a target root-mean-square (RMS)
//! level with asymmetric attack/release ballistics (fast to pull back loud
//! bursts, slow to raise quiet passages) and then guarantees the output never
//! exceeds a peak ceiling with a classic feedforward peak limiter plus a final
//! hard safety clamp. It is a lightweight online relative of the loudness
//! normalisation in design section 13, implemented with ordinary one-pole
//! envelope followers; there is no machine learning.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the AGC stage of design section 45.2 (uplink pre-processing).
//! Built on the sample scalar and deterministic math of `prism_audio_core`; it
//! runs after [`crate::uplink::noise_suppress`] and before
//! [`crate::uplink::vad`] inside [`crate::uplink::UplinkChain`].

use bevy_math::ops;
use prism_audio_core::math::{flush_denormal, Sample};

/// Tuning parameters for [`AutomaticGainControl`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AgcConfig {
    /// Sample rate in hertz, used to convert time constants to coefficients.
    pub sample_rate: Sample,
    /// Target RMS level as a linear amplitude in `(0, 1]`.
    pub target_rms: Sample,
    /// Maximum make-up gain as a linear multiplier.
    pub max_gain: Sample,
    /// Minimum gain as a linear multiplier (allows attenuation of loud input).
    pub min_gain: Sample,
    /// Attack time in seconds for reducing gain on rising level.
    pub attack_seconds: Sample,
    /// Release time in seconds for raising gain on falling level.
    pub release_seconds: Sample,
    /// RMS detector averaging time in seconds.
    pub rms_seconds: Sample,
    /// Peak ceiling as a linear amplitude in `(0, 1]`; the limiter keeps the
    /// output at or below this magnitude.
    pub peak_ceiling: Sample,
}

impl Default for AgcConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            target_rms: 0.2,
            max_gain: 16.0,
            min_gain: 0.1,
            attack_seconds: 0.010,
            release_seconds: 0.200,
            rms_seconds: 0.050,
            peak_ceiling: 0.98,
        }
    }
}

/// Converts a time constant in seconds to a one-pole smoothing coefficient.
#[inline]
fn time_to_coeff(seconds: Sample, sample_rate: Sample) -> Sample {
    if seconds <= 0.0 {
        0.0
    } else {
        ops::exp(-1.0 / (seconds * sample_rate))
    }
}

/// A target-RMS automatic gain control with a built-in peak limiter.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AutomaticGainControl {
    config: AgcConfig,
    attack_coeff: Sample,
    release_coeff: Sample,
    rms_coeff: Sample,
    limiter_release_coeff: Sample,
    /// Smoothed signal power for the RMS detector.
    power_env: Sample,
    /// Current make-up gain.
    gain: Sample,
    /// Current limiter gain (<= 1).
    limiter_gain: Sample,
}

impl AutomaticGainControl {
    /// Creates an AGC with the given configuration.
    #[must_use]
    pub fn new(config: AgcConfig) -> Self {
        let sr = config.sample_rate.max(1.0);
        Self {
            attack_coeff: time_to_coeff(config.attack_seconds, sr),
            release_coeff: time_to_coeff(config.release_seconds, sr),
            rms_coeff: time_to_coeff(config.rms_seconds, sr),
            // The limiter attacks instantly (feedforward) and releases over the
            // AGC release time so gain recovery is smooth.
            limiter_release_coeff: time_to_coeff(config.release_seconds, sr),
            config,
            power_env: 0.0,
            gain: 1.0,
            limiter_gain: 1.0,
        }
    }

    /// Returns the active configuration.
    #[must_use]
    pub fn config(&self) -> &AgcConfig {
        &self.config
    }

    /// Returns the current make-up gain as a linear multiplier.
    #[must_use]
    pub fn current_gain(&self) -> Sample {
        self.gain
    }

    /// Resets the detector and gains to unity.
    pub fn reset(&mut self) {
        self.power_env = 0.0;
        self.gain = 1.0;
        self.limiter_gain = 1.0;
    }

    /// Processes one sample and returns the gained, limited output.
    #[inline]
    #[must_use]
    pub fn process_sample(&mut self, x: Sample) -> Sample {
        // RMS detector (one-pole on instantaneous power).
        self.power_env =
            flush_denormal(self.rms_coeff * self.power_env + (1.0 - self.rms_coeff) * x * x);
        let rms = ops::sqrt(self.power_env);

        // Desired gain to reach the target RMS, bounded by the gain range.
        let desired = (self.config.target_rms / (rms + 1.0e-9))
            .clamp(self.config.min_gain, self.config.max_gain);

        // Asymmetric ballistics: attack when turning the gain down.
        let coeff = if desired < self.gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.gain = flush_denormal(coeff * self.gain + (1.0 - coeff) * desired);

        let pre = x * self.gain;

        // Feedforward peak limiter: instant attack, smoothed release.
        let peak = ops::abs(pre);
        let target_lim = if peak > self.config.peak_ceiling {
            self.config.peak_ceiling / peak
        } else {
            1.0
        };
        if target_lim < self.limiter_gain {
            self.limiter_gain = target_lim;
        } else {
            let c = self.limiter_release_coeff;
            self.limiter_gain = c * self.limiter_gain + (1.0 - c) * target_lim;
        }
        let limited = pre * self.limiter_gain;

        // Final safety clamp guarantees the ceiling is never exceeded.
        limited.clamp(-self.config.peak_ceiling, self.config.peak_ceiling)
    }

    /// Processes `block` in place.
    pub fn process_block(&mut self, block: &mut [Sample]) {
        for sample in block.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn rms(block: &[Sample]) -> Sample {
        ops::sqrt(block.iter().map(|&v| v * v).sum::<Sample>() / block.len() as Sample)
    }

    #[test]
    fn raises_quiet_signal_toward_target() {
        let cfg = AgcConfig::default();
        let target = cfg.target_rms;
        let mut agc = AutomaticGainControl::new(cfg);
        let sr = 48_000.0;
        let freq = 300.0;
        let mut last = Vec::new();
        for b in 0..400 {
            let mut block: Vec<Sample> = (0..256)
                .map(|i| {
                    let n = (b * 256 + i) as Sample;
                    0.02 * ops::sin(2.0 * PI * freq * n / sr)
                })
                .collect();
            agc.process_block(&mut block);
            last = block;
        }
        let out_rms = rms(&last);
        assert!(
            (out_rms - target).abs() < target * 0.25,
            "out_rms={out_rms} target={target}"
        );
    }

    #[test]
    fn limits_peaks_below_ceiling() {
        let cfg = AgcConfig::default();
        let ceiling = cfg.peak_ceiling;
        let mut agc = AutomaticGainControl::new(cfg);
        let sr = 48_000.0;
        for b in 0..200 {
            let mut block: Vec<Sample> = (0..256)
                .map(|i| {
                    let n = (b * 256 + i) as Sample;
                    0.9 * ops::sin(2.0 * PI * 500.0 * n / sr)
                })
                .collect();
            agc.process_block(&mut block);
            for &s in &block {
                assert!(ops::abs(s) <= ceiling + 1e-4, "sample {s} exceeds ceiling");
            }
        }
    }

    #[test]
    fn reset_returns_to_unity() {
        let mut agc = AutomaticGainControl::new(AgcConfig::default());
        let mut block = [0.01; 256];
        agc.process_block(&mut block);
        agc.reset();
        assert!((agc.current_gain() - 1.0).abs() < 1e-9);
    }
}
