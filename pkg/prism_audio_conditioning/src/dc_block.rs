//! Bake-time DC-offset removal and zero-phase subsonic high-pass conditioning.
//!
//! A surprising amount of recorded or synthesized material arrives with a
//! static DC bias or inaudible sub-sonic rumble: an asymmetric waveform that
//! wastes headroom, biases loudness and true-peak measurement, degrades
//! downstream lossy encoding, and can thump a loudspeaker's woofer. AAA
//! conditioning pipelines therefore scrub both before an asset is measured and
//! shipped. This stage is the offline *source hygiene* step: it runs once at
//! bake time, before the analysis stages, so loudness, loop detection, and
//! transient extraction all describe the cleaned program the engine will
//! actually play.
//!
//! Two independent, config-gated actions are offered:
//!
//! - **Exact DC-offset removal** subtracts each channel's mean. For a one-time
//!   offline pass this is strictly better than a causal DC-blocker: it removes
//!   the `0 Hz` component exactly, with no settling transient and no phase or
//!   amplitude error anywhere in the audible band.
//! - **Zero-phase sub-sonic high-pass** attenuates infrasonic rumble below a
//!   configurable cutoff with a second-order Butterworth response applied
//!   forward and backward (`filtfilt`-style). Because the pass is run in both
//!   directions the result has exactly zero phase and therefore cannot smear
//!   transients or shift the loop points and markers that later stages detect;
//!   the magnitude response is squared, giving an effective fourth-order
//!   (`-24 dB/octave`) slope. Odd (point-symmetric) reflection padding on both
//!   ends keeps the filter's warm-up and cool-down from leaking an edge
//!   transient into the delivered program.
//!
//! The stage is a pure, deterministic function and defaults to disabled, so the
//! delivered program is byte-identical to its input unless a caller opts in.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. Mean
//! subtraction, the RBJ-cookbook Butterworth high-pass biquad, zero-phase
//! forward-backward filtering, and odd-reflection edge padding are classic
//! public DSP techniques; only the ideas are borrowed, not any implementation.
//!
//! # Relationship
//! The source-hygiene leg of design section 51, run by [`crate::pipeline`]
//! immediately after resampling and before the analysis stages. It is the
//! offline, zero-phase dual of the causal runtime high-pass / DC-blocker used
//! in the communication uplink (section 45.2) and the frequency-dependent
//! shaping of section 13; here latency and causality do not matter, so the
//! stronger zero-phase treatment is preferred.

use bevy_math::ops;

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::pcm::ConditionedPcm;

/// Number of cutoff-period cycles reflected onto each end of a channel before
/// the zero-phase high-pass runs.
///
/// Three cycles let the second-order Butterworth sections settle fully from
/// rest, so stripping the padding afterward leaves no edge transient in the
/// delivered program.
const EDGE_PAD_CYCLES: Sample = 3.0;

/// Configuration for the DC-offset / sub-sonic high-pass conditioning stage.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DcBlockConfig {
    /// Whether to subtract each channel's mean (exact DC-offset removal).
    pub remove_dc_offset: bool,
    /// Sub-sonic high-pass cutoff, in Hz. A value of `0` (or any non-positive
    /// value) disables the high-pass; the mean-subtraction step is controlled
    /// independently by [`remove_dc_offset`](Self::remove_dc_offset).
    pub highpass_cutoff_hz: Sample,
}

impl Default for DcBlockConfig {
    fn default() -> Self {
        Self {
            remove_dc_offset: false,
            highpass_cutoff_hz: 0.0,
        }
    }
}

impl DcBlockConfig {
    /// Returns whether either action (DC removal or high-pass) is active.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.remove_dc_offset || self.highpass_cutoff_hz > 0.0
    }
}

/// A DC-conditioned program plus the DC offset that was removed per channel.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DcBlocked {
    /// The program after DC removal and/or sub-sonic high-pass.
    pub pcm: ConditionedPcm,
    /// The mean (DC) value subtracted from each channel by the mean-removal
    /// step, in channel order. Entries are `0` when
    /// [`DcBlockConfig::remove_dc_offset`] is disabled (the high-pass does not
    /// contribute to this report).
    pub removed_dc: Vec<Sample>,
}

/// Normalized second-order biquad coefficients (`a0` folded into the rest).
#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    /// Feed-forward coefficient on `x[n]`.
    b0: Sample,
    /// Feed-forward coefficient on `x[n-1]`.
    b1: Sample,
    /// Feed-forward coefficient on `x[n-2]`.
    b2: Sample,
    /// Feedback coefficient on `y[n-1]`.
    a1: Sample,
    /// Feedback coefficient on `y[n-2]`.
    a2: Sample,
}

/// Builds a second-order Butterworth high-pass (`Q = 1/sqrt(2)`, RBJ cookbook)
/// for `cutoff_hz` at `sample_rate`, clamping the cutoff into `(0, Nyquist)`.
fn butterworth_highpass(cutoff_hz: Sample, sample_rate: Sample) -> BiquadCoeffs {
    let nyquist = sample_rate * 0.5;
    let fc = cutoff_hz.clamp(1.0, nyquist * 0.999);
    let w0 = 2.0 * core::f32::consts::PI * fc / sample_rate;
    let (sin_w0, cos_w0) = ops::sin_cos(w0);
    let q = core::f32::consts::FRAC_1_SQRT_2;
    let alpha = sin_w0 / (2.0 * q);
    let a0 = 1.0 + alpha;
    let inv_a0 = 1.0 / a0;
    BiquadCoeffs {
        b0: ((1.0 + cos_w0) * 0.5) * inv_a0,
        b1: (-(1.0 + cos_w0)) * inv_a0,
        b2: ((1.0 + cos_w0) * 0.5) * inv_a0,
        a1: (-2.0 * cos_w0) * inv_a0,
        a2: (1.0 - alpha) * inv_a0,
    }
}

/// Runs `coeffs` over `input` once (causally) via a transposed Direct Form II
/// structure, starting from rest, returning the filtered signal.
fn run_biquad(coeffs: &BiquadCoeffs, input: &[Sample]) -> Vec<Sample> {
    let mut z1: Sample = 0.0;
    let mut z2: Sample = 0.0;
    let mut out = Vec::with_capacity(input.len());
    for &x in input {
        let y = coeffs.b0 * x + z1;
        z1 = coeffs.b1 * x - coeffs.a1 * y + z2;
        z2 = coeffs.b2 * x - coeffs.a2 * y;
        out.push(y);
    }
    out
}

/// Filters `samples` in place with zero phase: odd-reflection pad both ends,
/// run `coeffs` forward then backward, then strip the padding.
///
/// `pad` is the per-end reflection length; it is clamped so the reflection
/// never indexes past the signal. Signals shorter than two samples carry no
/// meaningful sub-sonic content and are left unchanged.
fn zero_phase_highpass(samples: &mut [Sample], coeffs: &BiquadCoeffs, pad: usize) {
    let n = samples.len();
    if n < 2 {
        return;
    }
    let pad = pad.min(n - 1);

    let first = samples[0];
    let last = samples[n - 1];
    let mut buf = Vec::with_capacity(n + 2 * pad);
    // Left odd reflection: 2*x[0] - x[pad], ..., 2*x[0] - x[1].
    for k in 0..pad {
        buf.push(2.0 * first - samples[pad - k]);
    }
    buf.extend_from_slice(samples);
    // Right odd reflection: 2*x[last] - x[n-2], ..., 2*x[last] - x[n-1-pad].
    for k in 0..pad {
        buf.push(2.0 * last - samples[n - 2 - k]);
    }

    // Forward pass, reverse, backward pass, reverse back -> zero net phase.
    let forward = run_biquad(coeffs, &buf);
    let mut reversed: Vec<Sample> = forward.into_iter().rev().collect();
    let backward = run_biquad(coeffs, &reversed);
    reversed = backward.into_iter().rev().collect();

    samples.copy_from_slice(&reversed[pad..pad + n]);
}

/// Removes a static DC offset and/or sub-sonic rumble from `pcm`.
///
/// When [`DcBlockConfig::remove_dc_offset`] is set, each channel's mean is
/// subtracted exactly (reported in [`DcBlocked::removed_dc`]). When
/// [`DcBlockConfig::highpass_cutoff_hz`] is positive, a zero-phase second-order
/// Butterworth high-pass is applied per channel afterward. With the stage
/// disabled (both actions off) the returned program is byte-identical to `pcm`
/// and every reported DC offset is `0`.
#[must_use]
pub fn apply(pcm: &ConditionedPcm, config: &DcBlockConfig) -> DcBlocked {
    let channel_count = pcm.channel_count();
    if !config.is_enabled() {
        return DcBlocked {
            pcm: pcm.clone(),
            removed_dc: (0..channel_count).map(|_| 0.0 as Sample).collect(),
        };
    }

    let mut out = pcm.clone();
    let mut removed_dc = Vec::with_capacity(channel_count);

    let highpass = if config.highpass_cutoff_hz > 0.0 {
        Some(butterworth_highpass(
            config.highpass_cutoff_hz,
            out.sample_rate() as Sample,
        ))
    } else {
        None
    };
    // Three cutoff cycles of warm-up, in samples, bounds the reflection pad.
    let pad = if config.highpass_cutoff_hz > 0.0 {
        (EDGE_PAD_CYCLES * out.sample_rate() as Sample / config.highpass_cutoff_hz) as usize
    } else {
        0
    };

    for ch in 0..channel_count {
        let Some(samples) = out.channel_mut(ch) else {
            removed_dc.push(0.0);
            continue;
        };

        // Exact DC removal: subtract the channel mean (accumulated in f64 so a
        // long channel does not lose precision before the subtraction).
        let mut dc: Sample = 0.0;
        if config.remove_dc_offset && !samples.is_empty() {
            let mut sum: f64 = 0.0;
            for &s in samples.iter() {
                sum += f64::from(s);
            }
            let mean = (sum / samples.len() as f64) as Sample;
            if mean.is_finite() && mean != 0.0 {
                dc = mean;
                for s in samples.iter_mut() {
                    *s -= mean;
                }
            }
        }
        removed_dc.push(dc);

        if let Some(coeffs) = highpass {
            zero_phase_highpass(samples, &coeffs, pad);
        }
    }

    DcBlocked {
        pcm: out,
        removed_dc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use core::f32::consts::TAU;
    use prism_audio_core::buffer::ChannelLayout;

    fn mono(samples: Vec<Sample>) -> ConditionedPcm {
        ConditionedPcm::new(48_000, ChannelLayout::Mono, vec![samples]).unwrap()
    }

    fn mean(samples: &[Sample]) -> Sample {
        let sum: f64 = samples.iter().map(|&s| f64::from(s)).sum();
        (sum / samples.len() as f64) as Sample
    }

    fn peak(samples: &[Sample]) -> Sample {
        samples.iter().fold(0.0, |m, &s| m.max(ops::abs(s)))
    }

    #[test]
    fn disabled_is_identity() {
        let pcm = mono(vec![0.5, -0.3, 0.2, 0.9, -0.1]);
        let out = apply(&pcm, &DcBlockConfig::default());
        assert_eq!(out.pcm, pcm);
        assert_eq!(out.removed_dc, vec![0.0]);
    }

    #[test]
    fn removes_constant_dc_offset() {
        // A sine riding on a +0.25 bias; mean subtraction should recenter it.
        let bias = 0.25;
        let channel: Vec<Sample> = (0..4_800)
            .map(|n| bias + 0.5 * ops::sin(TAU * 440.0 * n as Sample / 48_000.0))
            .collect();
        let pcm = mono(channel);
        let config = DcBlockConfig {
            remove_dc_offset: true,
            highpass_cutoff_hz: 0.0,
        };
        let out = apply(&pcm, &config);

        assert!((out.removed_dc[0] - bias).abs() < 1.0e-3, "{}", out.removed_dc[0]);
        assert!(mean(out.pcm.channel(0).unwrap()).abs() < 1.0e-4);
    }

    #[test]
    fn removes_dc_offset_is_deterministic() {
        let channel: Vec<Sample> = (0..512).map(|n| 0.1 + 0.01 * n as Sample).collect();
        let pcm = mono(channel);
        let config = DcBlockConfig {
            remove_dc_offset: true,
            highpass_cutoff_hz: 0.0,
        };
        let a = apply(&pcm, &config);
        let b = apply(&pcm, &config);
        assert_eq!(a, b);
    }

    #[test]
    fn highpass_attenuates_subsonic_tone() {
        // A 5 Hz tone sits well below a 30 Hz cutoff and should be crushed,
        // while its amplitude elsewhere is untouched by DC removal here.
        let sr = 48_000.0;
        let channel: Vec<Sample> = (0..48_000)
            .map(|n| 0.8 * ops::sin(TAU * 5.0 * n as Sample / sr))
            .collect();
        let pcm = mono(channel);
        let config = DcBlockConfig {
            remove_dc_offset: false,
            highpass_cutoff_hz: 30.0,
        };
        let out = apply(&pcm, &config);
        // The sub-sonic tone is strongly attenuated across the interior.
        let interior = &out.pcm.channel(0).unwrap()[8_000..40_000];
        assert!(peak(interior) < 0.2, "{}", peak(interior));
    }

    #[test]
    fn highpass_preserves_in_band_tone() {
        // A 1 kHz tone sits far above a 30 Hz cutoff and should pass with its
        // amplitude essentially intact (zero-phase: no attenuation in band).
        let sr = 48_000.0;
        let channel: Vec<Sample> = (0..48_000)
            .map(|n| 0.6 * ops::sin(TAU * 1_000.0 * n as Sample / sr))
            .collect();
        let pcm = mono(channel);
        let config = DcBlockConfig {
            remove_dc_offset: false,
            highpass_cutoff_hz: 30.0,
        };
        let out = apply(&pcm, &config);
        let interior = &out.pcm.channel(0).unwrap()[8_000..40_000];
        assert!((peak(interior) - 0.6).abs() < 0.02, "{}", peak(interior));
    }

    #[test]
    fn highpass_is_zero_phase() {
        // Zero-phase filtering must not shift a transient's position. Place a
        // unit impulse and confirm the output's energy peak stays put.
        let mut channel = vec![0.0 as Sample; 2_048];
        channel[1_024] = 1.0;
        let pcm = mono(channel);
        let config = DcBlockConfig {
            remove_dc_offset: false,
            highpass_cutoff_hz: 50.0,
        };
        let out = apply(&pcm, &config);
        let samples = out.pcm.channel(0).unwrap();
        let argmax = samples
            .iter()
            .enumerate()
            .max_by(|a, b| ops::abs(*a.1).partial_cmp(&ops::abs(*b.1)).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 1_024);
    }

    #[test]
    fn highpass_is_deterministic() {
        let sr = 48_000.0;
        let channel: Vec<Sample> = (0..1_024)
            .map(|n| 0.4 * ops::sin(TAU * 10.0 * n as Sample / sr))
            .collect();
        let pcm = mono(channel);
        let config = DcBlockConfig {
            remove_dc_offset: true,
            highpass_cutoff_hz: 25.0,
        };
        let a = apply(&pcm, &config);
        let b = apply(&pcm, &config);
        assert_eq!(a, b);
    }

    #[test]
    fn stereo_channels_handled_independently() {
        let left: Vec<Sample> = (0..1_000).map(|_| 0.3).collect();
        let right: Vec<Sample> = (0..1_000).map(|_| -0.4).collect();
        let pcm = ConditionedPcm::new(48_000, ChannelLayout::Stereo, vec![left, right]).unwrap();
        let config = DcBlockConfig {
            remove_dc_offset: true,
            highpass_cutoff_hz: 0.0,
        };
        let out = apply(&pcm, &config);
        assert!((out.removed_dc[0] - 0.3).abs() < 1.0e-5);
        assert!((out.removed_dc[1] - (-0.4)).abs() < 1.0e-5);
        // Both channels are recentered to zero mean.
        assert!(mean(out.pcm.channel(0).unwrap()).abs() < 1.0e-5);
        assert!(mean(out.pcm.channel(1).unwrap()).abs() < 1.0e-5);
    }

    #[test]
    fn short_signal_survives_highpass() {
        // A single-sample channel has no sub-sonic content; it must not panic
        // and must pass through the high-pass unchanged.
        let pcm = mono(vec![0.7]);
        let config = DcBlockConfig {
            remove_dc_offset: false,
            highpass_cutoff_hz: 30.0,
        };
        let out = apply(&pcm, &config);
        assert_eq!(out.pcm.channel(0).unwrap(), &[0.7]);
    }
}
