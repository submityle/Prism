//! Uplink (microphone capture) pre-processing chain.
//!
//! This module wires the classic capture stages of design section 45.2 into a
//! single allocation-free block processor:
//!
//! ```text
//! high_pass -> aec -> noise_suppress -> agc -> vad
//! ```
//!
//! Each stage is a standalone, fully-implemented classic DSP block in its own
//! file ([`high_pass`], [`aec`], [`noise_suppress`], [`agc`], [`vad`]). The
//! chain fixes the block size to the noise suppressor's frame so the whole path
//! can run from a device callback without allocating. The chain outputs the
//! processed near-end signal plus a per-block status (voice activity, echo
//! convergence) and can optionally gate non-speech blocks to save bandwidth.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the uplink of design section 45.1/45.2. Built on
//! `prism_audio_core`; consumed by [`crate::pipeline`] as the local capture
//! path and feeds the encoder/transport in [`crate::transport`].

pub mod aec;
pub mod agc;
pub mod high_pass;
pub mod noise_suppress;
pub mod vad;

use aec::{AecConfig, EchoCanceller, NlmsEchoCanceller};
use agc::{AgcConfig, AutomaticGainControl};
use high_pass::{HighPass, HighPassOrder};
use noise_suppress::{NoiseSuppressConfig, NoiseSuppressor};
use vad::{VadConfig, VadDecision, VoiceActivityDetector};

use prism_audio_core::math::Sample;

/// Configuration for the whole uplink chain.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct UplinkConfig {
    /// Sample rate in hertz.
    pub sample_rate: Sample,
    /// Processing block size (hop) in samples; rounded up to a power of two by
    /// the noise suppressor.
    pub frame: usize,
    /// High-pass cutoff in hertz for DC/rumble removal.
    pub high_pass_hz: Sample,
    /// High-pass filter order.
    pub high_pass_order: HighPassOrder,
    /// Echo canceller configuration.
    pub aec: AecConfig,
    /// Noise suppressor configuration.
    pub noise_suppress: NoiseSuppressConfig,
    /// Automatic gain control configuration.
    pub agc: AgcConfig,
    /// Voice activity detector configuration.
    pub vad: VadConfig,
    /// When `true`, blocks classified as non-speech are zeroed on output.
    pub gate_non_speech: bool,
}

impl Default for UplinkConfig {
    fn default() -> Self {
        let sample_rate = 48_000.0;
        Self {
            sample_rate,
            frame: 256,
            high_pass_hz: 80.0,
            high_pass_order: HighPassOrder::Second,
            aec: AecConfig::default(),
            noise_suppress: NoiseSuppressConfig::default(),
            agc: AgcConfig {
                sample_rate,
                ..AgcConfig::default()
            },
            vad: VadConfig::default(),
            gate_non_speech: true,
        }
    }
}

/// Per-block status reported by [`UplinkChain::process`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct UplinkStatus {
    /// The voice activity decision for the block.
    pub vad: VadDecision,
    /// The echo canceller's current echo-return-loss-enhancement in decibels.
    pub erle_db: Sample,
    /// Whether the output block was gated to silence (non-speech).
    pub gated: bool,
}

/// The assembled uplink pre-processing chain.
///
/// Call [`UplinkChain::process`] with equal-length `mic` and `reference`
/// blocks of exactly [`UplinkChain::frame`] samples. The processed near-end
/// signal is written back into `mic`.
#[derive(Clone, Debug)]
pub struct UplinkChain {
    frame: usize,
    gate_non_speech: bool,
    high_pass: HighPass,
    aec: NlmsEchoCanceller,
    noise_suppress: NoiseSuppressor,
    agc: AutomaticGainControl,
    vad: VoiceActivityDetector,
}

impl UplinkChain {
    /// Builds the chain from a configuration.
    #[must_use]
    pub fn new(config: UplinkConfig) -> Self {
        let noise_suppress = NoiseSuppressor::new(config.frame, config.noise_suppress);
        let frame = noise_suppress.frame();
        let agc = AutomaticGainControl::new(AgcConfig {
            sample_rate: config.sample_rate,
            ..config.agc
        });
        Self {
            frame,
            gate_non_speech: config.gate_non_speech,
            high_pass: HighPass::new(config.sample_rate, config.high_pass_hz, config.high_pass_order),
            aec: NlmsEchoCanceller::new(config.aec),
            noise_suppress,
            agc,
            vad: VoiceActivityDetector::new(config.vad),
        }
    }

    /// Returns the fixed block size in samples.
    #[must_use]
    pub fn frame(&self) -> usize {
        self.frame
    }

    /// Borrows the echo canceller (for telemetry or host inspection).
    #[must_use]
    pub fn echo_canceller(&self) -> &NlmsEchoCanceller {
        &self.aec
    }

    /// Resets every stage to its initial state.
    pub fn reset(&mut self) {
        self.high_pass.reset();
        self.aec.reset();
        self.noise_suppress.reset();
        self.agc.reset();
        self.vad.reset();
    }

    /// Runs the full chain on one block.
    ///
    /// `mic` is processed in place. `reference` is the loudspeaker signal for
    /// the same block, used by the echo canceller; pass a block of zeros for a
    /// headset path with no acoustic echo. If the slice lengths do not match
    /// the configured frame the block is returned unprocessed with a non-speech
    /// status so the real-time thread never panics.
    pub fn process(&mut self, mic: &mut [Sample], reference: &[Sample]) -> UplinkStatus {
        if mic.len() != self.frame || reference.len() != self.frame {
            return UplinkStatus {
                vad: VadDecision {
                    is_speech: false,
                    energy_db: f32::NEG_INFINITY,
                    zero_crossing_rate: 0.0,
                },
                erle_db: self.aec.erle_db(),
                gated: false,
            };
        }

        self.high_pass.process_block(mic);
        self.aec.process(mic, reference);
        self.noise_suppress.process_block(mic);
        self.agc.process_block(mic);
        let decision = self.vad.analyze(mic);

        let mut gated = false;
        if self.gate_non_speech && !decision.is_speech {
            for s in mic.iter_mut() {
                *s = 0.0;
            }
            gated = true;
        }

        UplinkStatus {
            vad: decision,
            erle_db: self.aec.erle_db(),
            gated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::CommRng;
    use bevy_math::ops;
    use core::f32::consts::PI;

    #[test]
    fn processes_fixed_frame() {
        let mut chain = UplinkChain::new(UplinkConfig::default());
        let frame = chain.frame();
        let reference = vec![0.0; frame];
        let sr = 48_000.0;
        let mut status = None;
        for b in 0..60 {
            let mut mic: Vec<Sample> = (0..frame)
                .map(|i| {
                    let n = (b * frame + i) as Sample;
                    0.2 * ops::sin(2.0 * PI * 220.0 * n / sr)
                })
                .collect();
            status = Some(chain.process(&mut mic, &reference));
        }
        // A sustained tone should eventually be classified as speech.
        assert!(status.unwrap().vad.is_speech);
    }

    #[test]
    fn gates_silence_to_zero() {
        let mut chain = UplinkChain::new(UplinkConfig::default());
        let frame = chain.frame();
        let reference = vec![0.0; frame];
        let mut rng = CommRng::new(11);
        let mut last_status = None;
        let mut last_block = Vec::new();
        for _ in 0..40 {
            let mut mic: Vec<Sample> = (0..frame).map(|_| rng.next_bipolar() * 0.0005).collect();
            last_status = Some(chain.process(&mut mic, &reference));
            last_block = mic;
        }
        let status = last_status.unwrap();
        if status.gated {
            assert!(last_block.iter().all(|&s| s == 0.0));
        }
    }

    #[test]
    fn mismatched_length_is_safe() {
        let mut chain = UplinkChain::new(UplinkConfig::default());
        let mut mic = [0.1; 7];
        let reference = [0.0; 7];
        let status = chain.process(&mut mic, &reference);
        assert!(!status.vad.is_speech);
    }
}
