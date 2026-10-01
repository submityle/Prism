//! Frequency-domain spectral gate: a short-time Fourier transform (`STFT`)
//! downward expander that attenuates spectral bins whose magnitude falls below
//! a threshold, suppressing steady broadband noise while leaving louder tonal
//! and transient content intact.
//!
//! A time-domain [`GateNode`](crate::nodes::dynamics::gate::GateNode) opens or
//! closes the whole signal at once from its broadband level. This node instead
//! gates each frequency bin independently: hiss and hum between musical notes
//! are pulled down toward a floor while the notes themselves pass, which a
//! broadband gate cannot do.
//!
//! # Model
//!
//! The signal is processed with a weighted overlap-add (`OLA`) `STFT`. Each
//! analysis frame of `fft_size` samples is multiplied by a Hann analysis
//! window, transformed with a `radix-2` decimation-in-time (`DIT`) fast Fourier
//! transform (`FFT`), gated bin by bin, inverse-transformed, multiplied by a
//! matching Hann synthesis window, and overlap-added with a hop of
//! `fft_size / OVERLAP_FACTOR` (75 percent overlap). The squared Hann window at
//! this hop satisfies the constant-overlap-add (`COLA`) condition, so with all
//! gains held open the output reconstructs the input exactly (apart from the
//! processing latency).
//!
//! For each bin the single-sided magnitude is normalised by the window
//! coherent gain into a `dBFS`-referenced amplitude (a full-scale sinusoid
//! sitting on a bin reads back `0 dBFS`). A bin at or above `threshold_db`
//! targets unity gain; a bin below it targets the linear `reduction_db` floor.
//! The per-bin gain follows its target through a one-pole smoother clocked once
//! per hop, using the `attack_ms` time constant when opening and the
//! `release_ms` time constant when closing, which suppresses the "musical
//! noise" that instantaneous bin gating produces. The gain is applied to the
//! bin and its Hermitian mirror so the inverse transform stays real.
//!
//! # Real-time contract
//!
//! All ring, overlap-add, twiddle, window, and scratch buffers are allocated
//! once at construction. [`SpectralGateNode::process`] performs no allocation,
//! locking, or panic on the hot path; non-finite input samples are treated as
//! silence. The node reports a processing latency of `fft_size` frames via
//! [`SpectralGateNode::latency_frames`].
//!
//! # Provenance
//!
//! The weighted overlap-add `STFT`, the Hann window, the `radix-2` Cooley-Tukey
//! `FFT`, and spectral gating / downward spectral expansion are standard,
//! publicly documented classic DSP techniques found in any signal-processing
//! text (for example the overlap-add `STFT` described by Allen and Rabiner).
//! This is pure classic DSP with no AI or ML. This module contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio
//! source or derived code**; only the widely documented transform and window
//! formulas are used.

use alloc::{vec, vec::Vec};
use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};

/// Smallest permitted transform size, in samples.
pub const MIN_FFT_SIZE: usize = 64;

/// Default transform size, in samples.
pub const DEFAULT_FFT_SIZE: usize = 1024;

/// Overlap factor: the hop is `fft_size / OVERLAP_FACTOR` (75 percent overlap),
/// the standard Hann weighted overlap-add choice that satisfies `COLA`.
pub const OVERLAP_FACTOR: usize = 4;

/// Default gate threshold in `dBFS`.
pub const DEFAULT_THRESHOLD_DB: Sample = -60.0;

/// Default reduction floor in decibels applied to sub-threshold bins.
pub const DEFAULT_REDUCTION_DB: Sample = -80.0;

/// Default gate-opening (attack) time constant in milliseconds.
pub const DEFAULT_ATTACK_MS: Sample = 2.0;

/// Default gate-closing (release) time constant in milliseconds.
pub const DEFAULT_RELEASE_MS: Sample = 50.0;

/// Tunable spectral-gate parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectralGateParams {
    /// Threshold in `dBFS`: bins at or above it pass, bins below are reduced.
    pub threshold_db: Sample,
    /// Floor gain in decibels (at most `0`) applied to sub-threshold bins.
    pub reduction_db: Sample,
    /// Gate-opening time constant in milliseconds (`0` opens instantly).
    pub attack_ms: Sample,
    /// Gate-closing time constant in milliseconds (`0` closes instantly).
    pub release_ms: Sample,
}

impl Default for SpectralGateParams {
    fn default() -> Self {
        Self {
            threshold_db: DEFAULT_THRESHOLD_DB,
            reduction_db: DEFAULT_REDUCTION_DB,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
        }
    }
}

/// Reverses the low `bits` of `value` (bit-reversal permutation index).
fn reverse_low_bits(mut value: usize, bits: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    result
}

/// Rounds `requested` up to the next power of two, never below [`MIN_FFT_SIZE`].
fn power_of_two_at_least(requested: usize) -> usize {
    let mut size = MIN_FFT_SIZE;
    while size < requested {
        size <<= 1;
    }
    size
}

/// In-place iterative `radix-2` decimation-in-time transform. `tw_im` holds the
/// forward (`-sin`) twiddles; the inverse path negates them and scales by
/// `inv_size`.
#[expect(
    clippy::too_many_arguments,
    reason = "a free function avoids borrowing self while its scratch buffers are mutably held"
)]
fn transform(
    re: &mut [Sample],
    im: &mut [Sample],
    rev: &[usize],
    tw_re: &[Sample],
    tw_im: &[Sample],
    size: usize,
    inverse: bool,
    inv_size: Sample,
) {
    for (i, &j) in rev.iter().enumerate() {
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= size {
        let half = len / 2;
        let step = size / len;
        let mut base = 0;
        while base < size {
            for k in 0..half {
                let tw = k * step;
                let wr = tw_re[tw];
                let wi = if inverse { -tw_im[tw] } else { tw_im[tw] };
                let a = base + k;
                let b = base + k + half;
                let tr = wr * re[b] - wi * im[b];
                let ti = wr * im[b] + wi * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
            base += len;
        }
        len <<= 1;
    }
    if inverse {
        for value in re.iter_mut() {
            *value *= inv_size;
        }
        for value in im.iter_mut() {
            *value *= inv_size;
        }
    }
}

/// One-pole smoothing coefficient for a time constant of `ms` milliseconds at a
/// frame rate of `sample_rate / hop` frames per second. Returns `1` (instant)
/// for a non-positive or non-finite time constant.
fn frame_coeff(ms: Sample, sample_rate: u32, hop: usize) -> Sample {
    if !ms.is_finite() || ms <= 0.0 {
        return 1.0;
    }
    let frame_period = hop as Sample / sample_rate as Sample;
    let tau = ms * 1.0e-3;
    (1.0 - ops::exp(-frame_period / tau)).clamp(0.0, 1.0)
}

/// Frequency-domain spectral gate node (weighted overlap-add `STFT`).
#[derive(Clone, Debug)]
pub struct SpectralGateNode {
    size: usize,
    hop: usize,
    half: usize,
    fifo_latency: usize,
    channels: usize,
    // Precomputed, read-only tables.
    win: Vec<Sample>,
    tw_re: Vec<Sample>,
    tw_im: Vec<Sample>,
    rev: Vec<usize>,
    inv_size: Sample,
    inv_window_sum: Sample,
    two_inv_window_sum: Sample,
    ola_norm: Sample,
    // Derived gate controls.
    threshold_linear: Sample,
    floor_gain: Sample,
    attack_coeff: Sample,
    release_coeff: Sample,
    // Per-channel streaming state.
    in_fifo: Vec<Sample>,
    out_fifo: Vec<Sample>,
    out_accum: Vec<Sample>,
    gate_gain: Vec<Sample>,
    rover: usize,
    // Hot-path scratch (reused across channels and frames).
    re: Vec<Sample>,
    im: Vec<Sample>,
}

impl SpectralGateNode {
    /// Builds a spectral gate for `channels` channels at `sample_rate`.
    ///
    /// `requested_size` is rounded up to a power of two no smaller than
    /// [`MIN_FFT_SIZE`]; `channels` is clamped to at least one.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::spectral_gate::{
    ///     SpectralGateNode, SpectralGateParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = SpectralGateNode::new(48_000, 2, 1_024, SpectralGateParams::default());
    /// // Weighted overlap-add STFT: the reported latency is one full frame.
    /// assert_eq!(node.latency_frames(), 1_024);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        requested_size: usize,
        params: SpectralGateParams,
    ) -> Self {
        let channels = channels.max(1);
        let size = power_of_two_at_least(requested_size);
        let hop = (size / OVERLAP_FACTOR).max(1);
        let half = size / 2;
        let fifo_latency = size - hop;
        let bits = size.trailing_zeros();

        let mut win = vec![0.0; size];
        let mut window_sum = 0.0f32;
        for (n, slot) in win.iter_mut().enumerate() {
            let w = 0.5 - 0.5 * ops::cos(TAU * n as Sample / size as Sample);
            *slot = w;
            window_sum += w;
        }

        let mut tw_re = vec![0.0; size];
        let mut tw_im = vec![0.0; size];
        for t in 0..size {
            let angle = TAU * t as Sample / size as Sample;
            tw_re[t] = ops::cos(angle);
            tw_im[t] = -ops::sin(angle);
        }

        let mut rev = vec![0usize; size];
        for (i, slot) in rev.iter_mut().enumerate() {
            *slot = reverse_low_bits(i, bits);
        }

        // Constant-overlap-add denominator of the squared window at the hop.
        let base = half % hop;
        let mut overlap_sum = 0.0f32;
        let mut k = base;
        while k < size {
            overlap_sum += win[k] * win[k];
            k += hop;
        }
        let ola_norm = if overlap_sum > 0.0 { 1.0 / overlap_sum } else { 0.0 };

        let inv_window_sum = if window_sum > 0.0 { 1.0 / window_sum } else { 0.0 };

        Self {
            size,
            hop,
            half,
            fifo_latency,
            channels,
            win,
            tw_re,
            tw_im,
            rev,
            inv_size: 1.0 / size as Sample,
            inv_window_sum,
            two_inv_window_sum: 2.0 * inv_window_sum,
            ola_norm,
            threshold_linear: db_to_linear(params.threshold_db),
            floor_gain: db_to_linear(params.reduction_db.min(0.0)),
            attack_coeff: frame_coeff(params.attack_ms, sample_rate, hop),
            release_coeff: frame_coeff(params.release_ms, sample_rate, hop),
            in_fifo: vec![0.0; channels * size],
            out_fifo: vec![0.0; channels * size],
            out_accum: vec![0.0; channels * size],
            gate_gain: vec![1.0; channels * (half + 1)],
            rover: fifo_latency,
            re: vec![0.0; size],
            im: vec![0.0; size],
        }
    }

    /// Transform size in samples (a power of two).
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.size
    }

    /// Hop size in samples between successive analysis frames.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Replaces the gate parameters, recomputing the derived controls. The
    /// smoothed per-bin gains are preserved for a click-free transition.
    pub fn set_params(&mut self, sample_rate: u32, params: SpectralGateParams) {
        self.threshold_linear = db_to_linear(params.threshold_db);
        self.floor_gain = db_to_linear(params.reduction_db.min(0.0));
        self.attack_coeff = frame_coeff(params.attack_ms, sample_rate, self.hop);
        self.release_coeff = frame_coeff(params.release_ms, sample_rate, self.hop);
    }
}

impl AudioNode for SpectralGateNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        let size = self.size;
        let hop = self.hop;
        let half = self.half;
        let fifo_latency = self.fifo_latency;
        let inv_size = self.inv_size;
        let inv_window_sum = self.inv_window_sum;
        let two_inv_window_sum = self.two_inv_window_sum;
        let ola_norm = self.ola_norm;
        let threshold_linear = self.threshold_linear;
        let floor_gain = self.floor_gain;
        let attack_coeff = self.attack_coeff;
        let release_coeff = self.release_coeff;
        let bin_count = half + 1;

        // Disjoint field borrows: read-only tables plus mutable state buffers.
        let win = &self.win;
        let tw_re = &self.tw_re;
        let tw_im = &self.tw_im;
        let rev = &self.rev;
        let re = &mut self.re;
        let im = &mut self.im;
        let in_fifo = &mut self.in_fifo;
        let out_fifo = &mut self.out_fifo;
        let out_accum = &mut self.out_accum;
        let gate_gain = &mut self.gate_gain;

        let mut rover = self.rover;

        for i in 0..frames {
            for ch in 0..channels {
                let fifo_base = ch * size;
                let x = input.channel(ch)[i];
                let x = if x.is_finite() { x } else { 0.0 };
                in_fifo[fifo_base + rover] = x;
                let y = out_fifo[fifo_base + (rover - fifo_latency)];
                let y = if y.is_finite() { flush_denormal(y) } else { 0.0 };
                output.channel_mut(ch)[i] = y;
            }

            rover += 1;
            if rover < size {
                continue;
            }
            rover = fifo_latency;

            for ch in 0..channels {
                let fifo_base = ch * size;
                let gain_base = ch * bin_count;

                // Analysis window into the complex scratch.
                for n in 0..size {
                    re[n] = win[n] * in_fifo[fifo_base + n];
                    im[n] = 0.0;
                }

                transform(re, im, rev, tw_re, tw_im, size, false, inv_size);

                // Per-bin spectral gate.
                for bin in 0..bin_count {
                    let mag = ops::sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
                    let norm = if bin == 0 || bin == half {
                        inv_window_sum
                    } else {
                        two_inv_window_sum
                    };
                    let amplitude = mag * norm;
                    let target = if amplitude >= threshold_linear {
                        1.0
                    } else {
                        floor_gain
                    };
                    let previous = gate_gain[gain_base + bin];
                    let coeff = if target > previous {
                        attack_coeff
                    } else {
                        release_coeff
                    };
                    let gain = previous + coeff * (target - previous);
                    gate_gain[gain_base + bin] = gain;

                    re[bin] *= gain;
                    im[bin] *= gain;
                    if bin > 0 && bin < half {
                        let mirror = size - bin;
                        re[mirror] *= gain;
                        im[mirror] *= gain;
                    }
                }

                transform(re, im, rev, tw_re, tw_im, size, true, inv_size);

                // Synthesis window and overlap-add.
                for n in 0..size {
                    out_accum[fifo_base + n] += win[n] * re[n] * ola_norm;
                }

                // Publish one hop of finished output, then advance the buffers.
                out_fifo[fifo_base..fifo_base + hop]
                    .copy_from_slice(&out_accum[fifo_base..fifo_base + hop]);
                for n in 0..(size - hop) {
                    out_accum[fifo_base + n] = out_accum[fifo_base + n + hop];
                }
                for n in (size - hop)..size {
                    out_accum[fifo_base + n] = 0.0;
                }
                for n in 0..(size - hop) {
                    in_fifo[fifo_base + n] = in_fifo[fifo_base + n + hop];
                }
            }
        }

        self.rover = rover;

        // Pass surplus channels (beyond the processed set) through untouched.
        for ch in channels..available {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }
    }

    fn reset(&mut self) {
        for value in &mut self.in_fifo {
            *value = 0.0;
        }
        for value in &mut self.out_fifo {
            *value = 0.0;
        }
        for value in &mut self.out_accum {
            *value = 0.0;
        }
        for value in &mut self.gate_gain {
            *value = 1.0;
        }
        self.rover = self.fifo_latency;
    }

    fn latency_frames(&self) -> u32 {
        self.size as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    /// Streams `signal` through the node one block and returns the output.
    fn run_mono(node: &mut SpectralGateNode, signal: &[Sample]) -> Vec<Sample> {
        let len = signal.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        input.channel_mut(0).copy_from_slice(signal);
        let output = AudioBuffer::new(ChannelLayout::Mono, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0).to_vec()
    }

    fn sine(freq: Sample, amp: Sample, len: usize) -> Vec<Sample> {
        (0..len)
            .map(|n| amp * ops::sin(TAU * freq * n as Sample / SR as Sample))
            .collect()
    }

    fn rms(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        ops::sqrt((sum / samples.len() as f64) as Sample)
    }

    #[test]
    fn requested_size_rounds_up_to_power_of_two() {
        let node = SpectralGateNode::new(SR, 1, 1_000, SpectralGateParams::default());
        assert_eq!(node.fft_size(), 1_024);
    }

    #[test]
    fn min_fft_size_enforced() {
        let node = SpectralGateNode::new(SR, 1, 8, SpectralGateParams::default());
        assert_eq!(node.fft_size(), MIN_FFT_SIZE);
    }

    #[test]
    fn hop_is_quarter_of_size() {
        let node = SpectralGateNode::new(SR, 1, 1_024, SpectralGateParams::default());
        assert_eq!(node.hop(), 1_024 / OVERLAP_FACTOR);
    }

    #[test]
    fn latency_is_one_full_frame() {
        // The weighted overlap-add pipeline delays the signal by a full frame.
        let node = SpectralGateNode::new(SR, 1, 1_024, SpectralGateParams::default());
        assert_eq!(node.latency_frames(), 1_024);
    }

    #[test]
    fn open_gate_reconstructs_input_delayed() {
        // Threshold far below the signal: every bin passes, so weighted
        // overlap-add must reconstruct the input delayed by the latency.
        let params = SpectralGateParams {
            threshold_db: -200.0,
            ..Default::default()
        };
        let mut node = SpectralGateNode::new(SR, 1, 512, params);
        let latency = node.latency_frames() as usize;
        let input = sine(1_000.0, 0.5, 8_192);
        let output = run_mono(&mut node, &input);

        // Compare a steady-state span (after the pipeline primes) against the
        // delayed input.
        let start = latency + node.fft_size();
        let end = input.len() - 1;
        let mut max_err = 0.0f32;
        for n in start..end {
            max_err = max_err.max((output[n] - input[n - latency]).abs());
        }
        assert!(max_err < 5.0e-3, "max reconstruction error {max_err}");
    }

    #[test]
    fn quiet_tone_is_attenuated() {
        // A tone well below threshold should be pushed toward the floor.
        let params = SpectralGateParams {
            threshold_db: -40.0,
            reduction_db: -80.0,
            attack_ms: 0.0,
            release_ms: 0.0,
        };
        let mut node = SpectralGateNode::new(SR, 1, 512, params);
        let input = sine(1_000.0, 1.0e-3, 8_192);
        let output = run_mono(&mut node, &input);
        let tail = &output[4_096..];
        let input_rms = rms(&input[4_096..]);
        let output_rms = rms(tail);
        assert!(
            output_rms < input_rms * 0.1,
            "output_rms {output_rms} input_rms {input_rms}"
        );
    }

    #[test]
    fn loud_tone_passes() {
        let params = SpectralGateParams {
            threshold_db: -40.0,
            reduction_db: -80.0,
            attack_ms: 0.0,
            release_ms: 0.0,
        };
        let mut node = SpectralGateNode::new(SR, 1, 512, params);
        let input = sine(1_000.0, 0.5, 8_192);
        let output = run_mono(&mut node, &input);
        let input_rms = rms(&input[4_096..]);
        let output_rms = rms(&output[4_096..]);
        assert!(
            (output_rms - input_rms).abs() < input_rms * 0.1,
            "output_rms {output_rms} input_rms {input_rms}"
        );
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut node = SpectralGateNode::new(SR, 1, 256, SpectralGateParams::default());
        let mut input = sine(1_000.0, 0.5, 4_096);
        input[10] = Sample::NAN;
        input[20] = Sample::INFINITY;
        input[30] = Sample::NEG_INFINITY;
        let output = run_mono(&mut node, &input);
        assert!(output.iter().all(|&y| y.is_finite()), "non-finite output");
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = SpectralGateNode::new(SR, 1, 256, SpectralGateParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn channels_gate_independently() {
        let params = SpectralGateParams {
            threshold_db: -40.0,
            reduction_db: -80.0,
            attack_ms: 0.0,
            release_ms: 0.0,
        };
        let mut node = SpectralGateNode::new(SR, 2, 512, params);
        let len = 8_192;
        let loud = sine(1_000.0, 0.5, len);
        let quiet = sine(1_000.0, 1.0e-3, len);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.channel_mut(0).copy_from_slice(&loud);
        input.channel_mut(1).copy_from_slice(&quiet);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);

        let left_rms = rms(&outputs[0].channel(0)[4_096..]);
        let right_rms = rms(&outputs[0].channel(1)[4_096..]);
        assert!(left_rms > 0.3, "left_rms {left_rms}");
        assert!(right_rms < 1.0e-4, "right_rms {right_rms}");
    }

    #[test]
    fn surplus_channels_pass_through() {
        // Node configured for one channel, fed a stereo buffer: the second
        // channel must be copied through unchanged.
        let mut node = SpectralGateNode::new(SR, 1, 256, SpectralGateParams::default());
        let len = 1_024;
        let data = sine(1_000.0, 0.5, len);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.channel_mut(1).copy_from_slice(&data);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert_eq!(outputs[0].channel(1), data.as_slice());
    }

    #[test]
    fn reset_clears_state() {
        // A reset node must reproduce a fresh node bit-for-bit on the same input,
        // proving every streaming buffer was cleared.
        let mut node = SpectralGateNode::new(SR, 1, 256, SpectralGateParams::default());
        let warmup = sine(1_000.0, 0.5, 4_096);
        let _ = run_mono(&mut node, &warmup);
        node.reset();
        let probe = sine(1_000.0, 0.5, 4_096);
        let after_reset = run_mono(&mut node, &probe);

        let mut fresh = SpectralGateNode::new(SR, 1, 256, SpectralGateParams::default());
        let baseline = run_mono(&mut fresh, &probe);

        let mut max_err = 0.0f32;
        for (a, b) in after_reset.iter().zip(baseline.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1.0e-6, "state leaked past reset: max_err {max_err}");
    }

    #[test]
    fn default_params_are_expected() {
        let p = SpectralGateParams::default();
        assert_eq!(p.threshold_db, DEFAULT_THRESHOLD_DB);
        assert_eq!(p.reduction_db, DEFAULT_REDUCTION_DB);
        assert_eq!(p.attack_ms, DEFAULT_ATTACK_MS);
        assert_eq!(p.release_ms, DEFAULT_RELEASE_MS);
    }

    #[test]
    fn frame_coeff_edges() {
        assert_eq!(frame_coeff(0.0, SR, 256), 1.0);
        assert_eq!(frame_coeff(-5.0, SR, 256), 1.0);
        let c = frame_coeff(50.0, SR, 256);
        assert!((0.0..=1.0).contains(&c), "coeff {c}");
    }

    #[test]
    fn set_params_updates_controls() {
        let mut node = SpectralGateNode::new(SR, 1, 256, SpectralGateParams::default());
        node.set_params(
            SR,
            SpectralGateParams {
                threshold_db: 0.0,
                reduction_db: -60.0,
                attack_ms: 1.0,
                release_ms: 10.0,
            },
        );
        assert!((node.threshold_linear - 1.0).abs() < 1.0e-6);
    }
}
