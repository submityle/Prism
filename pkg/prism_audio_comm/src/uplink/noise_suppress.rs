//! Classic single-channel spectral noise suppression for the uplink chain.
//!
//! Steady-state background noise (fans, hiss, room tone) is attenuated with a
//! short-time Fourier transform (STFT) analysis, a per-bin noise-floor estimate
//! tracked by minimum statistics, and a Wiener / spectral-subtraction gain with
//! an over-subtraction factor and a spectral floor. The transform uses a
//! weighted overlap-add (WOLA) structure with a square-root Hann window at 50%
//! overlap, which reconstructs transparently when every gain is unity. No
//! machine-learning denoiser (`RNNoise`, `DeepFilterNet`, and the like) is
//! used; this is purely the classical spectral approach.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the noise-suppression stage of design section 45.2 (uplink
//! pre-processing). Built on the FFT and sample scalar of `prism_audio_core`;
//! it runs after [`crate::uplink::aec`] and before [`crate::uplink::agc`]
//! inside [`crate::uplink::UplinkChain`].

use bevy_math::ops;
use core::f32::consts::PI;
use prism_audio_core::fft::Fft;
use prism_audio_core::math::{flush_denormal, Sample};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Tuning parameters for [`NoiseSuppressor`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NoiseSuppressConfig {
    /// Over-subtraction factor applied to the estimated noise power. Values
    /// above `1.0` suppress more aggressively at the cost of some distortion.
    pub over_subtraction: Sample,
    /// Spectral floor as a linear gain in `[0, 1]`: the minimum gain any bin is
    /// allowed to reach, preventing musical-noise holes and keeping a natural
    /// noise bed.
    pub spectral_floor: Sample,
    /// Smoothing coefficient in `[0, 1)` for the per-bin power estimate used by
    /// the noise tracker.
    pub power_smoothing: Sample,
    /// Length in frames of the sliding window over which the minimum-statistics
    /// tracker searches for the per-bin power minimum. Longer windows ride over
    /// longer speech bursts but adapt more slowly to a changing noise floor.
    pub noise_window_frames: usize,
    /// Bias compensation for the minimum-statistics estimate in `[1, 4]`. The
    /// minimum of a smoothed periodogram sits below the true mean noise power
    /// (Martin's minimum-statistics bias), so the tracked minimum is scaled up
    /// by this factor before it is used as the noise-power estimate.
    pub noise_bias: Sample,
}

impl Default for NoiseSuppressConfig {
    fn default() -> Self {
        Self {
            over_subtraction: 1.5,
            spectral_floor: 0.1,
            power_smoothing: 0.8,
            noise_window_frames: 24,
            noise_bias: 1.2,
        }
    }
}

/// STFT noise suppressor with a weighted overlap-add transform.
///
/// Each call to [`NoiseSuppressor::process_block`] consumes and returns exactly
/// `frame` samples. The suppressor introduces an algorithmic latency of one
/// frame because of the 50% overlap.
#[derive(Clone, Debug)]
pub struct NoiseSuppressor {
    config: NoiseSuppressConfig,
    frame: usize,
    fft_size: usize,
    fft: Fft,
    window: Vec<Sample>,
    analysis: Vec<Sample>,
    out_accum: Vec<Sample>,
    re: Vec<Sample>,
    im: Vec<Sample>,
    power: Vec<Sample>,
    /// Number of subwindows the sliding minimum window is divided into.
    subwindows: usize,
    /// Length in frames of each subwindow.
    subwindow_len: usize,
    /// Running per-bin minimum of the subwindow currently being filled.
    sub_min: Vec<Sample>,
    /// Ring of committed per-bin subwindow minima, `subwindows` frames deep.
    sub_mins: Vec<Sample>,
    /// Per-bin minimum across all committed subwindows (refreshed on commit).
    stored_min: Vec<Sample>,
    /// Index of the subwindow slot that will receive the next commit.
    sub_pos: usize,
    /// Frames filled into the current subwindow so far.
    sub_frame: usize,
}

impl NoiseSuppressor {
    /// Creates a suppressor for a hop size of `frame` samples.
    ///
    /// The STFT length is `2 * frame`; `frame` is rounded up to a power of two
    /// so the window satisfies the constant-overlap-add constraint and the FFT
    /// length stays a power of two.
    #[must_use]
    pub fn new(frame: usize, config: NoiseSuppressConfig) -> Self {
        let frame = frame.max(1).next_power_of_two();
        let fft_size = frame * 2;
        let fft = Fft::new(fft_size);
        let fft_size = fft.size();
        // Square-root Hann window: analysis and synthesis share it, so their
        // product is a Hann window and 50% overlap reconstructs to unity.
        let window: Vec<Sample> = (0..fft_size)
            .map(|n| {
                let hann = 0.5 - 0.5 * ops::cos(2.0 * PI * n as Sample / fft_size as Sample);
                ops::sqrt(hann)
            })
            .collect();
        let bins = fft_size / 2 + 1;
        // Minimum-statistics sliding window, split into subwindows so the
        // running minimum can be refreshed in constant amortised time.
        let subwindows = 4;
        let subwindow_len = (config.noise_window_frames / subwindows).max(1);
        Self {
            config,
            frame,
            fft_size,
            fft,
            window,
            analysis: vec![0.0; fft_size],
            out_accum: vec![0.0; fft_size],
            re: vec![0.0; fft_size],
            im: vec![0.0; fft_size],
            power: vec![0.0; bins],
            subwindows,
            subwindow_len,
            sub_min: vec![Sample::MAX; bins],
            sub_mins: vec![Sample::MAX; subwindows * bins],
            stored_min: vec![Sample::MAX; bins],
            sub_pos: 0,
            sub_frame: 0,
        }
    }

    /// Returns the hop size (block length) in samples.
    #[must_use]
    pub fn frame(&self) -> usize {
        self.frame
    }

    /// Returns the active configuration.
    #[must_use]
    pub fn config(&self) -> &NoiseSuppressConfig {
        &self.config
    }

    /// Clears all internal buffers and the learned noise floor.
    pub fn reset(&mut self) {
        for v in &mut self.analysis {
            *v = 0.0;
        }
        for v in &mut self.out_accum {
            *v = 0.0;
        }
        for v in &mut self.power {
            *v = 0.0;
        }
        for v in &mut self.sub_min {
            *v = Sample::MAX;
        }
        for v in &mut self.sub_mins {
            *v = Sample::MAX;
        }
        for v in &mut self.stored_min {
            *v = Sample::MAX;
        }
        self.sub_pos = 0;
        self.sub_frame = 0;
    }

    /// Suppresses noise in `block` in place.
    ///
    /// `block.len()` must equal [`NoiseSuppressor::frame`]; otherwise the block
    /// is passed through unchanged so the caller never panics on a size
    /// mismatch in the real-time thread.
    pub fn process_block(&mut self, block: &mut [Sample]) {
        if block.len() != self.frame {
            return;
        }
        let hop = self.frame;
        let n = self.fft_size;

        // Slide the analysis buffer left by one hop and append the new samples.
        self.analysis.copy_within(hop..n, 0);
        self.analysis[n - hop..n].copy_from_slice(block);

        // Windowed analysis into the FFT work buffers.
        for i in 0..n {
            self.re[i] = self.analysis[i] * self.window[i];
            self.im[i] = 0.0;
        }
        self.fft.forward(&mut self.re, &mut self.im);

        let bins = self.power.len();
        let beta = self.config.power_smoothing;
        let first_in_subwindow = self.sub_frame == 0;
        for k in 0..bins {
            let mag2 = self.re[k] * self.re[k] + self.im[k] * self.im[k];
            self.power[k] = flush_denormal(beta * self.power[k] + (1.0 - beta) * mag2);

            // Minimum statistics: track the minimum smoothed power within the
            // subwindow being filled, then take the floor as the minimum across
            // all committed subwindows and the current one. The window bounds
            // both how low the estimate can dip and how fast it recovers.
            if first_in_subwindow || self.power[k] < self.sub_min[k] {
                self.sub_min[k] = self.power[k];
            }
            let noise = if self.stored_min[k] < self.sub_min[k] {
                self.stored_min[k]
            } else {
                self.sub_min[k]
            };

            // Spectral-subtraction / Wiener gain with over-subtraction. The
            // tracked minimum sits below the true mean noise power, so it is
            // bias-compensated up before the over-subtraction factor is
            // applied.
            let noise_floor = self.config.noise_bias * noise;
            let signal = self.power[k] - self.config.over_subtraction * noise_floor;
            let denom = self.power[k] + 1.0e-12;
            let gain = (signal / denom).clamp(self.config.spectral_floor, 1.0);
            self.re[k] *= gain;
            self.im[k] *= gain;

            // Mirror the gain onto the conjugate-symmetric upper half.
            if k > 0 && k < bins - 1 {
                let mirror = n - k;
                self.re[mirror] *= gain;
                self.im[mirror] *= gain;
            }
        }

        // Advance the subwindow clock; on completion commit the running minima
        // into the ring and refresh the cross-subwindow minimum.
        self.sub_frame += 1;
        if self.sub_frame >= self.subwindow_len {
            let base = self.sub_pos * bins;
            self.sub_mins[base..base + bins].copy_from_slice(&self.sub_min);
            self.sub_pos = if self.sub_pos + 1 == self.subwindows {
                0
            } else {
                self.sub_pos + 1
            };
            self.sub_frame = 0;
            for k in 0..bins {
                let mut m = self.sub_mins[k];
                for w in 1..self.subwindows {
                    let val = self.sub_mins[w * bins + k];
                    if val < m {
                        m = val;
                    }
                }
                self.stored_min[k] = m;
            }
        }

        self.fft.inverse(&mut self.re, &mut self.im);

        // Synthesis window and overlap-add.
        for i in 0..n {
            self.out_accum[i] += self.re[i] * self.window[i];
        }
        block.copy_from_slice(&self.out_accum[..hop]);

        // Shift the accumulator left by one hop and zero the new tail.
        self.out_accum.copy_within(hop..n, 0);
        for v in &mut self.out_accum[n - hop..n] {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(block: &[Sample]) -> Sample {
        if block.is_empty() {
            return 0.0;
        }
        ops::sqrt(block.iter().map(|&v| v * v).sum::<Sample>() / block.len() as Sample)
    }

    #[test]
    fn unity_gain_reconstructs_sine() {
        // With the noise floor held at zero (spectral_floor = 1.0) the WOLA
        // transform must reconstruct the input, delayed by one frame.
        let frame = 256;
        let mut ns = NoiseSuppressor::new(
            frame,
            NoiseSuppressConfig {
                spectral_floor: 1.0,
                over_subtraction: 0.0,
                ..NoiseSuppressConfig::default()
            },
        );
        let sr = 48_000.0;
        let freq = 1_000.0;
        let mut input = Vec::new();
        let mut output = Vec::new();
        for b in 0..8 {
            let mut block: Vec<Sample> = (0..frame)
                .map(|i| {
                    let n = (b * frame + i) as Sample;
                    ops::sin(2.0 * PI * freq * n / sr)
                })
                .collect();
            input.extend_from_slice(&block);
            ns.process_block(&mut block);
            output.extend_from_slice(&block);
        }
        // Compare input[0..] against output[frame..] (one-frame latency).
        let compare = 4 * frame;
        let mut err = 0.0;
        for i in 0..compare {
            let d = input[i] - output[i + frame];
            err += d * d;
        }
        let rms_err = ops::sqrt(err / compare as Sample);
        assert!(rms_err < 1e-2, "reconstruction rms error {rms_err}");
    }

    #[test]
    fn suppresses_steady_noise() {
        let frame = 256;
        let mut ns = NoiseSuppressor::new(frame, NoiseSuppressConfig::default());
        let mut rng = crate::rng::CommRng::new(5);
        let mut tail_rms = 0.0;
        let mut first_rms = 0.0;
        for b in 0..60 {
            let mut block: Vec<Sample> =
                (0..frame).map(|_| rng.next_bipolar() * 0.3).collect();
            let before = rms(&block);
            ns.process_block(&mut block);
            if b == 2 {
                first_rms = before;
            }
            if b == 55 {
                tail_rms = rms(&block);
            }
        }
        // After the floor is learned, broadband noise is clearly attenuated.
        assert!(tail_rms < first_rms * 0.6, "tail={tail_rms} first={first_rms}");
    }

    #[test]
    fn wrong_block_size_is_passthrough() {
        let mut ns = NoiseSuppressor::new(128, NoiseSuppressConfig::default());
        let mut block = [0.5; 100];
        ns.process_block(&mut block);
        assert!(block.iter().all(|&v| (v - 0.5).abs() < 1e-9));
    }
}
