//! Acoustic echo cancellation (AEC) for the uplink chain.
//!
//! When a participant runs loudspeakers instead of a headset, the far-end
//! voice played locally leaks back into the microphone and is transmitted to
//! the far end as an echo of their own voice. This module cancels that echo
//! with a classic normalised least-mean-squares (NLMS) adaptive FIR filter
//! that models the loudspeaker-room-microphone impulse response from the known
//! reference (loudspeaker) signal, in the structural spirit of the WebRTC AEC3
//! design. There is no machine learning anywhere in the path.
//!
//! Three classic components work together:
//!
//! * the NLMS adaptive filter, which predicts and subtracts the linear echo;
//! * a Geigel double-talk detector (DTD) that freezes adaptation when the
//!   near-end talker is active, so the filter is not corrupted by near speech;
//! * a Wiener-style residual echo suppressor that attenuates the non-linear
//!   and tail-leakage echo the linear filter cannot remove.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the AEC stage of design section 45.2 (uplink pre-processing).
//! Built on the sample scalar and deterministic math of `prism_audio_core`; it
//! runs after [`crate::uplink::high_pass`] and before
//! [`crate::uplink::noise_suppress`] inside [`crate::uplink::UplinkChain`].

use bevy_math::ops;
use prism_audio_core::math::{flush_denormal, Sample};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Abstraction over an echo canceller so hosts can swap in a platform or
/// hardware implementation while reusing the rest of the uplink chain.
pub trait EchoCanceller {
    /// Cancels echo in `mic` in place using the aligned loudspeaker
    /// `reference` block.
    ///
    /// Both slices must have the same length; the reference is the signal that
    /// was (or will be) rendered to the local loudspeakers for this block.
    fn process(&mut self, mic: &mut [Sample], reference: &[Sample]);

    /// Clears all adaptive state so the canceller re-converges from scratch.
    fn reset(&mut self);

    /// Returns the current echo-return-loss-enhancement estimate in decibels.
    ///
    /// Higher is better; a well-converged canceller on a linear echo path
    /// reaches tens of decibels. Returns `0.0` before any echo is observed.
    fn erle_db(&self) -> Sample;
}

/// Tuning parameters for [`NlmsEchoCanceller`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AecConfig {
    /// Adaptive filter length in taps. Covers the modelled echo tail; at
    /// 48 kHz, 1024 taps spans roughly 21 ms.
    pub taps: usize,
    /// NLMS step size in `(0, 2)`. Larger converges faster but is noisier;
    /// `0.5` is a robust default.
    pub step_size: Sample,
    /// NLMS regularisation added to the reference energy to avoid division by
    /// near-zero during silence.
    pub regularization: Sample,
    /// Geigel double-talk threshold in `(0, 1)`. The detector declares
    /// double-talk when the microphone magnitude exceeds this fraction of the
    /// recent reference peak.
    pub dtd_threshold: Sample,
    /// Number of samples adaptation stays frozen after a double-talk trigger.
    pub dtd_hangover: usize,
    /// Residual leakage fraction in `[0, 1)`: the assumed share of echo power
    /// that survives the linear filter and must be suppressed by the post
    /// filter.
    pub residual_leak: Sample,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            taps: 1024,
            step_size: 0.5,
            regularization: 1.0e-3,
            dtd_threshold: 0.5,
            dtd_hangover: 240,
            residual_leak: 0.1,
        }
    }
}

/// A time-domain NLMS echo canceller with double-talk detection and a
/// Wiener-style residual suppressor.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NlmsEchoCanceller {
    config: AecConfig,
    /// Adaptive filter taps; `weights[k]` multiplies the reference delayed by
    /// `k` samples.
    weights: Vec<Sample>,
    /// Circular history of the reference signal, length equal to `taps`.
    history: Vec<Sample>,
    /// Write cursor into `history`.
    pos: usize,
    /// Running sum of squares of the samples currently in `history`.
    energy: Sample,
    /// Decaying estimate of the reference peak magnitude for the detector.
    ref_peak: Sample,
    /// Remaining frozen samples after a double-talk trigger.
    hangover: usize,
    /// Smoothed microphone power for the ERLE metric.
    mic_power: Sample,
    /// Smoothed residual (output) power for the ERLE metric.
    err_power: Sample,
    /// Smoothed echo-estimate power feeding the residual suppressor.
    echo_power: Sample,
    /// Smoothed residual-error power feeding the residual suppressor.
    post_err_power: Sample,
}

impl NlmsEchoCanceller {
    /// Creates a canceller with the given configuration.
    ///
    /// The tap count is forced to at least one so the filter is always valid.
    #[must_use]
    pub fn new(config: AecConfig) -> Self {
        let taps = config.taps.max(1);
        Self {
            config: AecConfig { taps, ..config },
            weights: vec![0.0; taps],
            history: vec![0.0; taps],
            pos: 0,
            energy: 0.0,
            ref_peak: 0.0,
            hangover: 0,
            mic_power: 0.0,
            err_power: 0.0,
            echo_power: 0.0,
            post_err_power: 0.0,
        }
    }

    /// Returns the active configuration.
    #[must_use]
    pub fn config(&self) -> &AecConfig {
        &self.config
    }

    /// Returns `true` while adaptation is frozen by the double-talk detector.
    #[must_use]
    pub fn in_double_talk(&self) -> bool {
        self.hangover > 0
    }

    /// Pushes one reference sample into the circular history and updates the
    /// running energy and the decaying reference peak.
    #[inline]
    fn push_reference(&mut self, x: Sample) {
        let taps = self.config.taps;
        let old = self.history[self.pos];
        self.energy += x * x - old * old;
        if self.energy < 0.0 {
            self.energy = 0.0;
        }
        self.history[self.pos] = x;
        self.pos = if self.pos + 1 == taps { 0 } else { self.pos + 1 };
        let mag = ops::abs(x);
        // Sliding-peak estimate: instantaneous rise, slow exponential decay.
        self.ref_peak = if mag > self.ref_peak {
            mag
        } else {
            self.ref_peak * 0.999
        };
    }

    /// Computes the filtered echo estimate as the dot product of the taps with
    /// the reference history (most-recent sample first).
    #[inline]
    fn estimate_echo(&self) -> Sample {
        let taps = self.config.taps;
        let mut acc = 0.0;
        // history[pos-1] is the newest sample; weights[0] aligns to it.
        let mut idx = if self.pos == 0 { taps - 1 } else { self.pos - 1 };
        for &w in &self.weights {
            acc += w * self.history[idx];
            idx = if idx == 0 { taps - 1 } else { idx - 1 };
        }
        acc
    }

    /// Applies one NLMS gradient update scaled by the normalised error.
    #[inline]
    fn adapt(&mut self, error: Sample) {
        let taps = self.config.taps;
        let norm = self.energy + self.config.regularization;
        let scale = self.config.step_size * error / norm;
        let mut idx = if self.pos == 0 { taps - 1 } else { self.pos - 1 };
        for w in &mut self.weights {
            *w = flush_denormal(*w + scale * self.history[idx]);
            idx = if idx == 0 { taps - 1 } else { idx - 1 };
        }
    }
}

impl EchoCanceller for NlmsEchoCanceller {
    fn process(&mut self, mic: &mut [Sample], reference: &[Sample]) {
        let n = mic.len().min(reference.len());
        // Power smoothing coefficient (one-pole, about 10 ms at 48 kHz).
        let alpha = 0.995;
        for i in 0..n {
            let near = mic[i];
            self.push_reference(reference[i]);
            let echo = self.estimate_echo();
            let error = near - echo;

            // Geigel double-talk detector: near-end dominates the reference.
            let near_mag = ops::abs(near);
            let triggered = near_mag > self.config.dtd_threshold * self.ref_peak
                && self.ref_peak > 1.0e-4;
            if triggered {
                self.hangover = self.config.dtd_hangover;
            } else if self.hangover > 0 {
                self.hangover -= 1;
            }

            if self.hangover == 0 {
                self.adapt(error);
            }

            // Residual (post) suppressor: a Wiener gain derived from the
            // assumed surviving echo power relative to the error power.
            self.echo_power = flush_denormal(alpha * self.echo_power + (1.0 - alpha) * echo * echo);
            self.post_err_power =
                flush_denormal(alpha * self.post_err_power + (1.0 - alpha) * error * error);
            let residual = self.config.residual_leak * self.echo_power;
            let denom = self.post_err_power + residual + 1.0e-12;
            let gain = (self.post_err_power / denom).clamp(0.0, 1.0);
            let out = error * gain;

            // ERLE metrics on the pre/post powers.
            self.mic_power = flush_denormal(alpha * self.mic_power + (1.0 - alpha) * near * near);
            self.err_power = flush_denormal(alpha * self.err_power + (1.0 - alpha) * out * out);

            mic[i] = out;
        }
    }

    fn reset(&mut self) {
        for w in &mut self.weights {
            *w = 0.0;
        }
        for h in &mut self.history {
            *h = 0.0;
        }
        self.pos = 0;
        self.energy = 0.0;
        self.ref_peak = 0.0;
        self.hangover = 0;
        self.mic_power = 0.0;
        self.err_power = 0.0;
        self.echo_power = 0.0;
        self.post_err_power = 0.0;
    }

    fn erle_db(&self) -> Sample {
        // No microphone energy observed yet: nothing to report.
        if self.mic_power <= 1.0e-12 {
            return 0.0;
        }
        // Floor the residual so a near-perfectly cancelled echo yields a large
        // positive ERLE rather than collapsing to zero on underflow.
        let err = if self.err_power > 1.0e-12 {
            self.err_power
        } else {
            1.0e-12
        };
        let ratio = self.mic_power / err;
        if ratio <= 1.0 {
            0.0
        } else {
            10.0 * ops::log10(ratio)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::CommRng;

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn power(block: &[Sample]) -> Sample {
        if block.is_empty() {
            return 0.0;
        }
        block.iter().map(|&v| v * v).sum::<Sample>() / block.len() as Sample
    }

    // A short, fixed room impulse response for the synthetic echo path. A
    // real loudspeaker-room-microphone path is passive and attenuated (well
    // below unity), which is the regime the Geigel double-talk detector
    // assumes; a hot near-unity path would make any sample-wise detector
    // mistake the echo itself for near-end speech.
    const ECHO_IR: [Sample; 6] = [0.18, -0.09, 0.045, 0.024, -0.012, 0.006];

    fn convolve_echo(reference: &[Sample], tail: &mut [Sample; 6]) -> Vec<Sample> {
        let mut out = Vec::with_capacity(reference.len());
        for &x in reference {
            let mut acc = ECHO_IR[0] * x;
            for k in 1..ECHO_IR.len() {
                acc += ECHO_IR[k] * tail[k - 1];
            }
            // Shift the delay line.
            for k in (1..tail.len()).rev() {
                tail[k] = tail[k - 1];
            }
            tail[0] = x;
            out.push(acc);
        }
        out
    }

    #[test]
    fn cancels_linear_echo() {
        let mut aec = NlmsEchoCanceller::new(AecConfig {
            taps: 64,
            step_size: 0.7,
            ..AecConfig::default()
        });
        let mut rng = CommRng::new(1);
        let mut tail = [0.0; 6];

        // Converge over several blocks of white-noise reference, echo only.
        let mut last_out_power = 1.0;
        let mut mic_in_power = 1.0;
        for _ in 0..200 {
            let reference: Vec<Sample> = (0..256).map(|_| rng.next_bipolar() * 0.5).collect();
            let echo = convolve_echo(&reference, &mut tail);
            let mut mic = echo.clone();
            aec.process(&mut mic, &reference);
            last_out_power = power(&mic);
            mic_in_power = power(&echo);
        }
        // The residual should be far below the microphone echo power.
        assert!(
            last_out_power < mic_in_power * 0.05,
            "residual power {last_out_power} vs echo {mic_in_power}"
        );
        assert!(aec.erle_db() > 10.0, "erle={}", aec.erle_db());
    }

    #[test]
    fn double_talk_freezes_adaptation() {
        let mut aec = NlmsEchoCanceller::new(AecConfig {
            taps: 32,
            step_size: 0.7,
            ..AecConfig::default()
        });
        let mut rng = CommRng::new(7);
        let mut tail = [0.0; 6];
        // Converge first on echo only.
        for _ in 0..100 {
            let reference: Vec<Sample> = (0..256).map(|_| rng.next_bipolar() * 0.5).collect();
            let echo = convolve_echo(&reference, &mut tail);
            let mut mic = echo.clone();
            aec.process(&mut mic, &reference);
        }
        let converged: Vec<Sample> = aec.weights.clone();

        // Now inject strong near-end speech; adaptation must freeze. The
        // sine carries a phase offset so the near-end talker is already loud at
        // the first sample of the block: a Geigel detector is instantaneous and
        // cannot freeze on a near-end component that is still near a zero
        // crossing, so a stimulus that began exactly at phase zero (aligned to
        // the block boundary) would leak one adaptation step through before the
        // detector could react. Real double-talk is not phase-aligned to block
        // boundaries, and this models a talker who is already active.
        let reference: Vec<Sample> = (0..256).map(|_| rng.next_bipolar() * 0.5).collect();
        let echo = convolve_echo(&reference, &mut tail);
        let mut mic: Vec<Sample> = echo
            .iter()
            .enumerate()
            .map(|(i, &e)| e + 0.9 * ops::sin(i as Sample * 0.3 + 1.5))
            .collect();
        aec.process(&mut mic, &reference);
        assert!(aec.in_double_talk());

        // Weights should be essentially unchanged during double-talk.
        let mut max_delta: Sample = 0.0;
        for (a, b) in converged.iter().zip(aec.weights.iter()) {
            max_delta = max_delta.max(ops::abs(a - b));
        }
        assert!(max_delta < 1e-2, "weights drifted by {max_delta}");
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut aec = NlmsEchoCanceller::new(AecConfig {
            taps: 16,
            ..AecConfig::default()
        });
        let mut mic = [0.3; 128];
        let reference = [0.2; 128];
        aec.process(&mut mic, &reference);
        aec.reset();
        assert!(aec.weights.iter().all(|&w| w == 0.0));
        assert_eq!(aec.erle_db(), 0.0);
    }
}
