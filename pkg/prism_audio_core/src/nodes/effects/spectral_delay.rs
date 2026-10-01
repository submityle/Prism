//! Frequency-domain spectral delay: a short-time Fourier transform (`STFT`)
//! effect that delays each frequency bin by its own, frequency-dependent time
//! with an optional per-bin feedback loop, so that different parts of the
//! spectrum arrive at different moments and smear into a diffuse, iridescent
//! tail.
//!
//! # Relationship
//!
//! This node is deliberately distinct from the crate's other delay and
//! spectral effects:
//!
//! * [`DelayNode`](crate::nodes::effects::delay::DelayNode) and
//!   [`MultiTapDelayNode`](crate::nodes::effects::multi_tap_delay::MultiTapDelayNode)
//!   delay the whole broadband signal (or fixed taps of it) by one time in the
//!   time domain; every frequency is delayed equally.
//! * [`SpectralFreezeNode`](crate::nodes::effects::spectral_freeze::SpectralFreezeNode)
//!   latches and sustains a captured magnitude spectrum.
//! * [`SpectralGateNode`](crate::nodes::effects::spectral_gate::SpectralGateNode)
//!   attenuates bins below a threshold.
//! * [`PitchShifterNode`](crate::nodes::effects::pitch_shifter::PitchShifterNode)
//!   is a phase-vocoder transposer.
//!
//! This node is the only one that gives each frequency bin an **independent
//! delay time plus an independent band-limited feedback loop**, a frequency
//! smear that no time-domain delay can produce.
//!
//! # Model
//!
//! The signal is processed with a weighted overlap-add (`OLA`) `STFT`. Each
//! analysis frame of `fft_size` samples is multiplied by a Hann analysis
//! window, transformed with the crate's shared `radix-2` decimation-in-time
//! (`DIT`) fast Fourier transform ([`Fft`](crate::fft::Fft)), processed bin by
//! bin, inverse-transformed, multiplied by a matching Hann synthesis window,
//! and overlap-added with a hop of `fft_size / OVERLAP_FACTOR` (75 percent
//! overlap). The squared Hann window at this hop satisfies the
//! constant-overlap-add (`COLA`) condition, so with the dry / wet mix fully dry
//! the output reconstructs the input exactly (apart from the processing
//! latency).
//!
//! Each bin owns a short complex ring of past `STFT` frames clocked once per
//! hop. The per-bin delay in hops is linearly interpolated across the spectrum
//! from `low_delay_ms` at `DC` to `high_delay_ms` at Nyquist (either endpoint
//! may be the larger). For a bin with complex value `X` and delayed value `D`
//! read from its ring, the value written back into the ring is
//! `X + feedback * D`, and the emitted bin is `(1 - mix) * X + mix * D`. The
//! delayed value and the write are applied to the bin and mirrored (as the
//! complex conjugate) onto its Hermitian partner so the inverse transform stays
//! real. With `feedback` bounded below one the per-bin comb is stable.
//!
//! # Real-time contract
//!
//! Every ring, overlap-add, window, delay-table, and scratch buffer, together
//! with the shared [`Fft`](crate::fft::Fft) plan (its twiddle and bit-reversal
//! tables), is allocated once at construction. [`SpectralDelayNode::process`]
//! performs no allocation, locking, or panic on the hot path; non-finite input
//! samples are treated as silence and ring writes are denormal-flushed. The
//! node reports a processing latency of `fft_size` frames via
//! [`SpectralDelayNode::latency_frames`].
//!
//! # Provenance
//!
//! The weighted overlap-add `STFT`, the Hann window, the `radix-2` Cooley-Tukey
//! `FFT` (factored into the crate's shared [`Fft`](crate::fft::Fft) primitive),
//! and per-bin spectral delay with feedback are standard, publicly documented
//! classic DSP techniques found in any signal-processing text (for example the
//! overlap-add `STFT` described by Allen and Rabiner, and the spectral-delay
//! structures described in the audio-effects literature such as Zoelzer's
//! "DAFX"). This is pure classic DSP with no AI or ML. This module contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, or Web Audio source or derived code**; only the widely documented
//! transform, window, and delay formulas are used.

use alloc::{vec, vec::Vec};
use bevy_math::ops;
use core::f32::consts::TAU;

use crate::fft::Fft;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};

/// Smallest permitted transform size, in samples.
pub const MIN_SPECTRAL_DELAY_FFT_SIZE: usize = 64;

/// Default transform size, in samples.
pub const DEFAULT_SPECTRAL_DELAY_FFT_SIZE: usize = 1_024;

/// Overlap factor: the hop is `fft_size / OVERLAP_FACTOR` (75 percent overlap),
/// the standard Hann weighted overlap-add choice that satisfies `COLA`.
pub const SPECTRAL_DELAY_OVERLAP_FACTOR: usize = 4;

/// Largest per-bin delay, in milliseconds, that a bin may be assigned. This
/// bounds the complex ring allocated per bin.
pub const MAX_DELAY_MS: Sample = 500.0;

/// Largest permitted per-bin feedback magnitude; kept below one so each bin's
/// comb filter stays stable.
pub const MAX_SPECTRAL_DELAY_FEEDBACK: Sample = 0.95;

/// Default delay assigned to the `DC` end of the spectrum, in milliseconds.
pub const DEFAULT_LOW_DELAY_MS: Sample = 0.0;

/// Default delay assigned to the Nyquist end of the spectrum, in milliseconds.
pub const DEFAULT_HIGH_DELAY_MS: Sample = 120.0;

/// Default per-bin feedback.
pub const DEFAULT_FEEDBACK: Sample = 0.35;

/// Default dry / wet mix.
pub const DEFAULT_MIX: Sample = 0.5;

/// Tunable spectral-delay parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectralDelayParams {
    /// Delay in milliseconds assigned to the `DC` bin (lowest frequency).
    pub low_delay_ms: Sample,
    /// Delay in milliseconds assigned to the Nyquist bin (highest frequency).
    pub high_delay_ms: Sample,
    /// Per-bin feedback in `[0, MAX_SPECTRAL_DELAY_FEEDBACK]`.
    pub feedback: Sample,
    /// Dry / wet mix in `[0, 1]`: `0` is fully dry, `1` is fully delayed.
    pub mix: Sample,
}

impl Default for SpectralDelayParams {
    fn default() -> Self {
        Self {
            low_delay_ms: DEFAULT_LOW_DELAY_MS,
            high_delay_ms: DEFAULT_HIGH_DELAY_MS,
            feedback: DEFAULT_FEEDBACK,
            mix: DEFAULT_MIX,
        }
    }
}

impl SpectralDelayParams {
    /// Clamps every field into its valid range, replacing non-finite values
    /// with the defaults.
    #[must_use]
    fn sanitised(self) -> Self {
        let d = Self::default();
        let low = if self.low_delay_ms.is_finite() {
            self.low_delay_ms.clamp(0.0, MAX_DELAY_MS)
        } else {
            d.low_delay_ms
        };
        let high = if self.high_delay_ms.is_finite() {
            self.high_delay_ms.clamp(0.0, MAX_DELAY_MS)
        } else {
            d.high_delay_ms
        };
        let feedback = if self.feedback.is_finite() {
            self.feedback.clamp(0.0, MAX_SPECTRAL_DELAY_FEEDBACK)
        } else {
            d.feedback
        };
        let mix = if self.mix.is_finite() {
            self.mix.clamp(0.0, 1.0)
        } else {
            d.mix
        };
        Self {
            low_delay_ms: low,
            high_delay_ms: high,
            feedback,
            mix,
        }
    }
}

/// Frequency-domain spectral-delay node (weighted overlap-add `STFT`).
#[derive(Clone, Debug)]
pub struct SpectralDelayNode {
    size: usize,
    hop: usize,
    half: usize,
    bin_count: usize,
    fifo_latency: usize,
    channels: usize,
    ring_len: usize,
    max_cap: usize,
    // Precomputed, read-only tables.
    win: Vec<Sample>,
    fft: Fft,
    ola_norm: Sample,
    // Derived delay controls.
    delay_frames: Vec<usize>,
    feedback: Sample,
    mix: Sample,
    // Per-channel streaming state.
    in_fifo: Vec<Sample>,
    out_fifo: Vec<Sample>,
    out_accum: Vec<Sample>,
    ring_re: Vec<Sample>,
    ring_im: Vec<Sample>,
    write_idx: Vec<usize>,
    rover: usize,
    // Hot-path scratch (reused across channels and frames).
    re: Vec<Sample>,
    im: Vec<Sample>,
}

/// Fills `delay_frames` with the per-bin delay, in hops, linearly interpolated
/// from `low_delay_ms` at `DC` to `high_delay_ms` at Nyquist and clamped to
/// `max_cap`. This runs off the hot path (construction and parameter changes).
fn fill_delay_table(
    delay_frames: &mut [usize],
    low_delay_ms: Sample,
    high_delay_ms: Sample,
    half: usize,
    sample_rate: u32,
    hop: usize,
    max_cap: usize,
) {
    let hop_period_ms = 1_000.0 * hop as Sample / sample_rate as Sample;
    for (bin, slot) in delay_frames.iter_mut().enumerate() {
        let t = if half > 0 {
            bin as Sample / half as Sample
        } else {
            0.0
        };
        let ms = lerp(low_delay_ms, high_delay_ms, t);
        let frames = ops::round(ms / hop_period_ms);
        let frames = if frames.is_finite() { frames.max(0.0) } else { 0.0 };
        *slot = (frames as usize).min(max_cap);
    }
}

impl SpectralDelayNode {
    /// Builds a spectral delay for `channels` channels at `sample_rate`.
    ///
    /// `requested_size` is rounded up to a power of two no smaller than
    /// [`MIN_SPECTRAL_DELAY_FFT_SIZE`]; `channels` is clamped to at least one.
    /// `params` is clamped into range.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::spectral_delay::{
    ///     SpectralDelayNode, SpectralDelayParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = SpectralDelayNode::new(48_000, 2, 1_024, SpectralDelayParams::default());
    /// // Weighted overlap-add STFT: the reported latency is one full frame.
    /// assert_eq!(node.latency_frames(), 1_024);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        requested_size: usize,
        params: SpectralDelayParams,
    ) -> Self {
        let channels = channels.max(1);
        let fft = Fft::new(requested_size.max(MIN_SPECTRAL_DELAY_FFT_SIZE));
        let size = fft.size();
        let hop = (size / SPECTRAL_DELAY_OVERLAP_FACTOR).max(1);
        let half = size / 2;
        let bin_count = half + 1;
        let fifo_latency = size - hop;

        // Longest ring (in hops) any bin can address, derived from MAX_DELAY_MS
        // and the hop rate (independent of the transform size).
        let max_cap = (MAX_DELAY_MS * sample_rate as Sample / 1_000.0 / hop as Sample) as usize + 1;
        let ring_len = max_cap + 1;

        let mut win = vec![0.0; size];
        for (n, slot) in win.iter_mut().enumerate() {
            *slot = 0.5 - 0.5 * ops::cos(TAU * n as Sample / size as Sample);
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

        let params = params.sanitised();
        let mut delay_frames = vec![0usize; bin_count];
        fill_delay_table(
            &mut delay_frames,
            params.low_delay_ms,
            params.high_delay_ms,
            half,
            sample_rate,
            hop,
            max_cap,
        );

        Self {
            size,
            hop,
            half,
            bin_count,
            fifo_latency,
            channels,
            ring_len,
            max_cap,
            win,
            fft,
            ola_norm,
            delay_frames,
            feedback: params.feedback,
            mix: params.mix,
            in_fifo: vec![0.0; channels * size],
            out_fifo: vec![0.0; channels * size],
            out_accum: vec![0.0; channels * size],
            ring_re: vec![0.0; channels * bin_count * ring_len],
            ring_im: vec![0.0; channels * bin_count * ring_len],
            write_idx: vec![0usize; channels],
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

    /// Replaces the delay parameters, recomputing the per-bin delay table. The
    /// streaming ring state is preserved for a click-free transition.
    pub fn set_params(&mut self, sample_rate: u32, params: SpectralDelayParams) {
        let params = params.sanitised();
        fill_delay_table(
            &mut self.delay_frames,
            params.low_delay_ms,
            params.high_delay_ms,
            self.half,
            sample_rate,
            self.hop,
            self.max_cap,
        );
        self.feedback = params.feedback;
        self.mix = params.mix;
    }
}

impl AudioNode for SpectralDelayNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        let size = self.size;
        let hop = self.hop;
        let half = self.half;
        let bin_count = self.bin_count;
        let fifo_latency = self.fifo_latency;
        let ring_len = self.ring_len;
        let ola_norm = self.ola_norm;
        let feedback = self.feedback;
        let mix = self.mix;
        let dry = 1.0 - mix;

        // Disjoint field borrows: read-only tables plus mutable state buffers.
        let win = &self.win;
        let fft = &self.fft;
        let delay_frames = &self.delay_frames;
        let re = &mut self.re;
        let im = &mut self.im;
        let in_fifo = &mut self.in_fifo;
        let out_fifo = &mut self.out_fifo;
        let out_accum = &mut self.out_accum;
        let ring_re = &mut self.ring_re;
        let ring_im = &mut self.ring_im;
        let write_idx = &mut self.write_idx;

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

            #[expect(
                clippy::needless_range_loop,
                reason = "ch also indexes fifo_base, ring_base, and the input and output channels"
            )]
            for ch in 0..channels {
                let fifo_base = ch * size;

                // Analysis window into the complex scratch.
                for n in 0..size {
                    re[n] = win[n] * in_fifo[fifo_base + n];
                    im[n] = 0.0;
                }

                fft.forward(re, im);

                // Per-bin spectral delay with feedback.
                let w = write_idx[ch];
                for bin in 0..bin_count {
                    let d = delay_frames[bin];
                    let xr = re[bin];
                    let xi = im[bin];
                    let ring_base = (ch * bin_count + bin) * ring_len;
                    let (dr, di) = if d == 0 {
                        (xr, xi)
                    } else {
                        let read = (w + ring_len - d) % ring_len;
                        (ring_re[ring_base + read], ring_im[ring_base + read])
                    };

                    // Write X + feedback * delayed back into the ring.
                    ring_re[ring_base + w] = flush_denormal(xr + feedback * dr);
                    ring_im[ring_base + w] = flush_denormal(xi + feedback * di);

                    // Emit the dry / wet blend.
                    let or = dry * xr + mix * dr;
                    let oi = dry * xi + mix * di;
                    re[bin] = or;
                    im[bin] = oi;
                    if bin > 0 && bin < half {
                        let mirror = size - bin;
                        re[mirror] = or;
                        im[mirror] = -oi;
                    }
                }
                write_idx[ch] = (w + 1) % ring_len;

                fft.inverse(re, im);

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
        for value in &mut self.ring_re {
            *value = 0.0;
        }
        for value in &mut self.ring_im {
            *value = 0.0;
        }
        for value in &mut self.write_idx {
            *value = 0;
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

    /// Streams `signal` through the node in one block and returns the output.
    fn run_mono(node: &mut SpectralDelayNode, signal: &[Sample]) -> Vec<Sample> {
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

    fn argmax_abs(samples: &[Sample]) -> usize {
        let mut best = 0usize;
        let mut best_v = 0.0f32;
        for (i, &x) in samples.iter().enumerate() {
            let a = x.abs();
            if a > best_v {
                best_v = a;
                best = i;
            }
        }
        best
    }

    #[test]
    fn requested_size_rounds_up_to_power_of_two() {
        let node = SpectralDelayNode::new(SR, 1, 1_000, SpectralDelayParams::default());
        assert_eq!(node.fft_size(), 1_024);
    }

    #[test]
    fn min_fft_size_enforced() {
        let node = SpectralDelayNode::new(SR, 1, 8, SpectralDelayParams::default());
        assert_eq!(node.fft_size(), MIN_SPECTRAL_DELAY_FFT_SIZE);
    }

    #[test]
    fn hop_is_quarter_of_size() {
        let node = SpectralDelayNode::new(SR, 1, 1_024, SpectralDelayParams::default());
        assert_eq!(node.hop(), 1_024 / SPECTRAL_DELAY_OVERLAP_FACTOR);
    }

    #[test]
    fn latency_is_one_full_frame() {
        let node = SpectralDelayNode::new(SR, 1, 1_024, SpectralDelayParams::default());
        assert_eq!(node.latency_frames(), 1_024);
    }

    #[test]
    fn dry_mix_reconstructs_input_delayed() {
        // mix = 0 bypasses every bin, so the weighted overlap-add must
        // reconstruct the input delayed by the pipeline latency.
        let params = SpectralDelayParams {
            low_delay_ms: 50.0,
            high_delay_ms: 200.0,
            feedback: 0.5,
            mix: 0.0,
        };
        let mut node = SpectralDelayNode::new(SR, 1, 512, params);
        let latency = node.latency_frames() as usize;
        let input = sine(1_000.0, 0.5, 8_192);
        let output = run_mono(&mut node, &input);

        let start = latency + node.fft_size();
        let end = input.len() - 1;
        let mut max_err = 0.0f32;
        for n in start..end {
            max_err = max_err.max((output[n] - input[n - latency]).abs());
        }
        assert!(max_err < 5.0e-3, "max reconstruction error {max_err}");
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = SpectralDelayNode::new(SR, 1, 256, SpectralDelayParams::default());
        let output = run_mono(&mut node, &vec![0.0; 4_096]);
        assert!(output.iter().all(|&y| y == 0.0), "non-silent output");
    }

    #[test]
    fn tone_output_is_finite() {
        let mut node = SpectralDelayNode::new(SR, 1, 512, SpectralDelayParams::default());
        let input = sine(1_000.0, 0.5, 8_192);
        let output = run_mono(&mut node, &input);
        assert!(output.iter().all(|&y| y.is_finite()), "non-finite output");
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut node = SpectralDelayNode::new(SR, 1, 256, SpectralDelayParams::default());
        let mut input = sine(1_000.0, 0.5, 4_096);
        input[10] = Sample::NAN;
        input[20] = Sample::INFINITY;
        input[30] = Sample::NEG_INFINITY;
        let output = run_mono(&mut node, &input);
        assert!(output.iter().all(|&y| y.is_finite()), "non-finite output");
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = SpectralDelayNode::new(SR, 1, 256, SpectralDelayParams::default());
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
    fn extreme_params_do_not_panic() {
        let params = SpectralDelayParams {
            low_delay_ms: 10_000.0,
            high_delay_ms: -10.0,
            feedback: 5.0,
            mix: 2.0,
        };
        let mut node = SpectralDelayNode::new(SR, 1, 256, params);
        let input = sine(500.0, 0.9, 4_096);
        let output = run_mono(&mut node, &input);
        assert!(output.iter().all(|&y| y.is_finite()), "non-finite output");
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let params = SpectralDelayParams {
            low_delay_ms: Sample::NAN,
            high_delay_ms: Sample::INFINITY,
            feedback: Sample::NAN,
            mix: Sample::NEG_INFINITY,
        };
        let s = params.sanitised();
        let d = SpectralDelayParams::default();
        assert_eq!(s.low_delay_ms, d.low_delay_ms);
        assert_eq!(s.high_delay_ms, d.high_delay_ms);
        assert_eq!(s.feedback, d.feedback);
        assert_eq!(s.mix, d.mix);
    }

    #[test]
    fn feedback_is_clamped() {
        let s = SpectralDelayParams {
            feedback: 10.0,
            ..Default::default()
        }
        .sanitised();
        assert!(s.feedback <= MAX_SPECTRAL_DELAY_FEEDBACK);
    }

    #[test]
    fn delay_table_increases_from_low_to_high() {
        // low < high: the per-bin delay must be monotonically non-decreasing
        // from DC to Nyquist.
        let params = SpectralDelayParams {
            low_delay_ms: 0.0,
            high_delay_ms: 400.0,
            feedback: 0.0,
            mix: 1.0,
        };
        let node = SpectralDelayNode::new(SR, 1, 512, params);
        for w in node.delay_frames.windows(2) {
            assert!(w[1] >= w[0], "delay table not monotone: {w:?}");
        }
        assert!(*node.delay_frames.last().unwrap() > node.delay_frames[0]);
    }

    #[test]
    fn uniform_delay_shifts_the_signal() {
        // mix = 1, feedback = 0, low == high == D: every bin is delayed by the
        // same whole number of hops, so the reconstruction is the input shifted
        // by d * hop samples relative to a zero-delay reference.
        let size = 512;
        let hop = size / SPECTRAL_DELAY_OVERLAP_FACTOR;
        let d: usize = 4;
        let delay_ms = 1_000.0 * (d * hop) as Sample / SR as Sample;

        let shifted_params = SpectralDelayParams {
            low_delay_ms: delay_ms,
            high_delay_ms: delay_ms,
            feedback: 0.0,
            mix: 1.0,
        };
        let zero_params = SpectralDelayParams {
            low_delay_ms: 0.0,
            high_delay_ms: 0.0,
            feedback: 0.0,
            mix: 1.0,
        };

        let mut impulse = vec![0.0; 8_192];
        impulse[2_048] = 1.0;

        let mut shifted_node = SpectralDelayNode::new(SR, 1, size, shifted_params);
        // Confirm the table resolved to the intended whole-hop delay.
        assert_eq!(shifted_node.delay_frames[0], d);
        let shifted = run_mono(&mut shifted_node, &impulse);

        let mut zero_node = SpectralDelayNode::new(SR, 1, size, zero_params);
        let zero = run_mono(&mut zero_node, &impulse);

        let diff = argmax_abs(&shifted) as i64 - argmax_abs(&zero) as i64;
        assert_eq!(diff, (d * hop) as i64, "peak shift mismatch");
    }

    #[test]
    fn high_feedback_impulse_stays_bounded() {
        let params = SpectralDelayParams {
            low_delay_ms: 20.0,
            high_delay_ms: 60.0,
            feedback: MAX_SPECTRAL_DELAY_FEEDBACK,
            mix: 1.0,
        };
        let mut node = SpectralDelayNode::new(SR, 1, 256, params);
        let mut input = vec![0.0; 48_000];
        input[128] = 1.0;
        let output = run_mono(&mut node, &input);
        assert!(output.iter().all(|&y| y.is_finite()), "non-finite output");
        let peak = output.iter().fold(0.0f32, |m, &y| m.max(y.abs()));
        assert!(peak < 1_000.0, "feedback diverged, peak {peak}");
    }

    #[test]
    fn channels_delay_independently() {
        let params = SpectralDelayParams {
            low_delay_ms: 40.0,
            high_delay_ms: 40.0,
            feedback: 0.0,
            mix: 1.0,
        };
        let mut node = SpectralDelayNode::new(SR, 2, 512, params);
        let len = 8_192;
        let mut left = vec![0.0; len];
        left[2_048] = 1.0;
        let right = vec![0.0; len];
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.channel_mut(0).copy_from_slice(&left);
        input.channel_mut(1).copy_from_slice(&right);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);

        let left_peak = outputs[0]
            .channel(0)
            .iter()
            .fold(0.0f32, |m, &y| m.max(y.abs()));
        let right_peak = outputs[0]
            .channel(1)
            .iter()
            .fold(0.0f32, |m, &y| m.max(y.abs()));
        assert!(left_peak > 0.1, "left_peak {left_peak}");
        assert!(right_peak < 1.0e-6, "right_peak {right_peak}");
    }

    #[test]
    fn surplus_channels_pass_through() {
        let mut node = SpectralDelayNode::new(SR, 1, 256, SpectralDelayParams::default());
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
        let mut node = SpectralDelayNode::new(SR, 1, 256, SpectralDelayParams::default());
        let warmup = sine(1_000.0, 0.5, 4_096);
        let _ = run_mono(&mut node, &warmup);
        node.reset();
        let probe = sine(1_000.0, 0.5, 4_096);
        let after_reset = run_mono(&mut node, &probe);

        let mut fresh = SpectralDelayNode::new(SR, 1, 256, SpectralDelayParams::default());
        let baseline = run_mono(&mut fresh, &probe);

        let mut max_err = 0.0f32;
        for (a, b) in after_reset.iter().zip(baseline.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1.0e-6, "state leaked past reset: max_err {max_err}");
    }

    #[test]
    fn set_params_updates_delay_table() {
        let mut node = SpectralDelayNode::new(SR, 1, 512, SpectralDelayParams::default());
        node.set_params(
            SR,
            SpectralDelayParams {
                low_delay_ms: 300.0,
                high_delay_ms: 0.0,
                feedback: 0.1,
                mix: 0.25,
            },
        );
        // low > high now: the table must be monotonically non-increasing.
        for w in node.delay_frames.windows(2) {
            assert!(w[1] <= w[0], "table not reversed: {w:?}");
        }
        assert_eq!(node.mix, 0.25);
        assert!((node.feedback - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn default_params_are_expected() {
        let p = SpectralDelayParams::default();
        assert_eq!(p.low_delay_ms, DEFAULT_LOW_DELAY_MS);
        assert_eq!(p.high_delay_ms, DEFAULT_HIGH_DELAY_MS);
        assert_eq!(p.feedback, DEFAULT_FEEDBACK);
        assert_eq!(p.mix, DEFAULT_MIX);
    }
}
