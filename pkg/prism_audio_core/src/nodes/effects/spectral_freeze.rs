//! Frequency-domain spectral freeze (infinite spectral sustain).
//!
//! A [`SpectralFreezeNode`] runs a weighted overlap-add short-time Fourier
//! transform (`STFT`) on its input. While unfrozen it reconstructs the input
//! transparently (a pure latency delay). When the freeze control is engaged it
//! captures the current magnitude spectrum and sustains it indefinitely:
//! each synthesis hop advances every bin's phase by its own centre-frequency
//! increment, so the held timbre keeps ringing as a steady, infinitely
//! sustained pad rather than an audible loop of the captured block.
//!
//! An optional `diffusion` control sprinkles a small, deterministic per-hop
//! phase perturbation across the bins. This breaks up the static, metallic
//! quality of a perfectly coherent freeze and gives the sustain a gently
//! shimmering, evolving character without changing its spectral envelope.
//!
//! Engaging and releasing the freeze crossfades between the live
//! reconstruction and the held spectrum over [`FREEZE_RAMP_SECONDS`], so the
//! transition is click-free.
//!
//! # Provenance
//!
//! Implemented from first principles as a textbook weighted overlap-add STFT
//! whose synthesis magnitudes are latched and whose synthesis phases advance
//! by each bin's nominal frequency. No source code or derivative code from UE,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or Web
//! Audio was consulted or copied; only the well-known public concept of
//! spectral freezing informs the design. There is no AI or machine learning of
//! any kind: the optional phase diffusion is a deterministic classic
//! `xorshift64` pseudo-random generator seeded from the parameters.
//!
//! # Relationship
//!
//! The STFT framing (Hann analysis / synthesis window, hop of
//! `fft_size / OVERLAP_FACTOR`, constant-overlap-add normalization) mirrors the
//! structure used by [`pitch_shifter`](crate::nodes::effects::pitch_shifter)
//! and [`spectral_gate`](crate::nodes::effects::spectral_gate), and shares the
//! same radix-2 [`Fft`](crate::fft::Fft) plan rather than carrying a private
//! copy. Unlike the pitch shifter it performs no instantaneous-frequency
//! estimation or bin remapping, and unlike the spectral gate it latches and
//! holds magnitudes instead of attenuating them. It is distinct from the
//! time-domain [`granular`](crate::nodes::effects::granular) grain cloud, which
//! sustains by re-triggering recorded fragments rather than holding a spectrum.
//!
//! # Real-time contract
//!
//! The window, the shared [`Fft`](crate::fft::Fft) plan, and every per-channel
//! FIFO, accumulator, and spectral latch are sized and allocated in
//! [`SpectralFreezeNode::new`]. `process` only reads inputs, advances cursors,
//! and writes outputs; it flushes denormals on every emitted sample, performs
//! no allocation or locking, and is panic-free.

use alloc::{vec, vec::Vec};

use bevy_math::ops;

use core::f32::consts::{PI, TAU};

use crate::fft::{Fft, power_of_two_at_least};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Analysis/synthesis overlap factor; the hop is `fft_size / OVERLAP_FACTOR`.
pub const OVERLAP_FACTOR: usize = 4;

/// Default transform size in samples (before rounding up to a power of two).
pub const DEFAULT_FFT_SIZE: usize = 2048;

/// Smallest transform size the node will use, in samples.
pub const MIN_FREEZE_FFT_SIZE: usize = 64;

/// Crossfade time between live and frozen output, in seconds.
pub const FREEZE_RAMP_SECONDS: Sample = 0.05;

/// Largest allowed per-hop phase diffusion depth.
pub const MAX_DIFFUSION: Sample = 1.0;

/// Default pseudo-random seed (a golden-ratio-derived odd constant).
pub const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Reciprocal of 2^24, used to map a 24-bit random word onto `[0, 1)`.
const INV_RNG_SCALE: Sample = 1.0 / 16_777_216.0;

/// Construction and automation parameters for a [`SpectralFreezeNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectralFreezeParams {
    /// Whether the spectrum starts latched (frozen) at construction.
    pub frozen: bool,
    /// Per-hop phase diffusion depth in `[0, MAX_DIFFUSION]` (0 = coherent).
    pub diffusion: Sample,
    /// Seed for the deterministic phase-diffusion generator.
    pub seed: u64,
}

impl Default for SpectralFreezeParams {
    fn default() -> Self {
        Self {
            frozen: false,
            diffusion: 0.0,
            seed: DEFAULT_SEED,
        }
    }
}

impl SpectralFreezeParams {
    /// Returns a copy with every field clamped to its valid range and any
    /// non-finite value replaced by the default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frozen: self.frozen,
            diffusion: if self.diffusion.is_finite() {
                self.diffusion.clamp(0.0, MAX_DIFFUSION)
            } else {
                0.0
            },
            seed: self.seed,
        }
    }
}

/// Deterministic `xorshift64`-star pseudo-random generator for phase diffusion.
#[derive(Debug, Clone)]
struct FreezeRng {
    state: u64,
}

impl FreezeRng {
    /// Creates a generator from `seed`, forcing a non-zero odd state.
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advances the generator and returns the next 64-bit word.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns a uniform sample in `[-1, 1)`.
    fn next_bipolar(&mut self) -> Sample {
        let unit = ((self.next_u64() >> 40) as u32 as Sample) * INV_RNG_SCALE;
        unit * 2.0 - 1.0
    }
}

/// A frequency-domain spectral freeze (input port 0 -> output port 0).
///
/// While unfrozen the node reconstructs its input with a fixed latency of
/// [`SpectralFreezeNode::fft_size`] samples. Engaging the freeze latches the
/// current magnitude spectrum and sustains it; the live and held spectra are
/// crossfaded in the frequency domain so toggling is click-free.
#[derive(Debug, Clone)]
pub struct SpectralFreezeNode {
    // Geometry.
    size: usize,
    hop: usize,
    half: usize,
    bins: usize,
    fifo_latency: usize,
    channels: usize,
    // Precomputed, read-only tables.
    fft: Fft,
    win: Vec<Sample>,
    ola_norm: Sample,
    expct: Sample,
    // Control.
    frozen: bool,
    diffusion: Sample,
    freeze_mix: Sample,
    freeze_target: Sample,
    freeze_step: Sample,
    seed: u64,
    rng: FreezeRng,
    // Per-channel streaming state.
    in_fifo: Vec<Sample>,
    out_fifo: Vec<Sample>,
    out_accum: Vec<Sample>,
    frozen_mag: Vec<Sample>,
    frozen_phase: Vec<Sample>,
    rover: usize,
    // Hot-path scratch (reused across channels and frames).
    re: Vec<Sample>,
    im: Vec<Sample>,
}

impl SpectralFreezeNode {
    /// Builds a spectral freeze for `channels` channels at `sample_rate`.
    ///
    /// `requested_size` is rounded up to a power of two no smaller than
    /// [`MIN_FREEZE_FFT_SIZE`]; the hop is `fft_size / OVERLAP_FACTOR`.
    /// `channels` is clamped to at least one. Parameters are sanitized before
    /// use.
    ///
    /// # Example
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::spectral_freeze::{
    ///     SpectralFreezeNode, SpectralFreezeParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = SpectralFreezeNode::new(48_000, 2, 1_024, SpectralFreezeParams::default());
    /// assert_eq!(node.latency_frames(), 1_024);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        requested_size: usize,
        params: SpectralFreezeParams,
    ) -> Self {
        let p = params.sanitised();
        let channels = channels.max(1);
        let size = power_of_two_at_least(requested_size.max(MIN_FREEZE_FFT_SIZE));
        let hop = (size / OVERLAP_FACTOR).max(1);
        let half = size / 2;
        let bins = half + 1;
        let fifo_latency = size - hop;

        let mut win = vec![0.0; size];
        for (n, slot) in win.iter_mut().enumerate() {
            *slot = 0.5 - 0.5 * ops::cos(TAU * n as Sample / size as Sample);
        }

        // Constant-overlap-add denominator of the squared window at the hop so
        // that the unfrozen pipeline reconstructs the input at unity gain.
        let base = half % hop;
        let mut overlap_sum = 0.0f32;
        let mut k = base;
        while k < size {
            overlap_sum += win[k] * win[k];
            k += hop;
        }
        let ola_norm = if overlap_sum > 0.0 { 1.0 / overlap_sum } else { 0.0 };

        let sr = sample_rate.max(1) as Sample;
        let expct = TAU * hop as Sample / size as Sample;
        let ramp_frames = (FREEZE_RAMP_SECONDS * sr).max(1.0);
        let freeze_step = (hop as Sample / ramp_frames).clamp(1.0e-4, 1.0);
        let freeze_target = if p.frozen { 1.0 } else { 0.0 };

        Self {
            size,
            hop,
            half,
            bins,
            fifo_latency,
            channels,
            fft: Fft::new(size),
            win,
            ola_norm,
            expct,
            frozen: p.frozen,
            diffusion: p.diffusion,
            freeze_mix: freeze_target,
            freeze_target,
            freeze_step,
            seed: p.seed,
            rng: FreezeRng::new(p.seed),
            in_fifo: vec![0.0; channels * size],
            out_fifo: vec![0.0; channels * size],
            out_accum: vec![0.0; channels * size],
            frozen_mag: vec![0.0; channels * bins],
            frozen_phase: vec![0.0; channels * bins],
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

    /// Configured channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Whether the spectrum is currently latched.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Current per-hop phase diffusion depth.
    #[must_use]
    pub fn diffusion(&self) -> Sample {
        self.diffusion
    }

    /// Engages or releases the spectral freeze (click-free crossfade).
    pub fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
        self.freeze_target = if frozen { 1.0 } else { 0.0 };
    }

    /// Sets the per-hop phase diffusion depth (clamped to `[0, MAX_DIFFUSION]`).
    pub fn set_diffusion(&mut self, diffusion: Sample) {
        self.diffusion = if diffusion.is_finite() {
            diffusion.clamp(0.0, MAX_DIFFUSION)
        } else {
            0.0
        };
    }

    /// Transforms one buffered frame for channel `ch` and overlap-adds the
    /// (possibly frozen) reconstruction into that channel's accumulator.
    fn process_frame(&mut self, ch: usize) {
        let fifo_base = ch * self.size;
        let bin_base = ch * self.bins;

        // Advance the live/frozen crossfade once per hop.
        if self.freeze_mix < self.freeze_target {
            self.freeze_mix = (self.freeze_mix + self.freeze_step).min(self.freeze_target);
        } else if self.freeze_mix > self.freeze_target {
            self.freeze_mix = (self.freeze_mix - self.freeze_step).max(self.freeze_target);
        }
        let mix = self.freeze_mix;
        let frozen = self.frozen;
        let diffusion = self.diffusion;

        // Analysis window into the complex scratch, then forward transform.
        for n in 0..self.size {
            self.re[n] = self.win[n] * self.in_fifo[fifo_base + n];
            self.im[n] = 0.0;
        }
        self.fft.forward(&mut self.re, &mut self.im);

        // For each bin, blend the live spectrum with the held spectrum.
        for k in 0..self.bins {
            let live_re = self.re[k];
            let live_im = self.im[k];

            if frozen {
                // Hold the latched magnitude; advance phase by the bin's own
                // nominal increment, with optional deterministic diffusion.
                let mut phase = self.frozen_phase[bin_base + k] + k as Sample * self.expct;
                if diffusion > 0.0 {
                    phase += diffusion * self.rng.next_bipolar() * PI;
                }
                self.frozen_phase[bin_base + k] = phase;
            } else {
                // Track the live spectrum so a future freeze latches instantly.
                let mag = ops::sqrt(live_re * live_re + live_im * live_im);
                self.frozen_mag[bin_base + k] = mag;
                self.frozen_phase[bin_base + k] = ops::atan2(live_im, live_re);
            }

            let mag = self.frozen_mag[bin_base + k];
            let (sin_p, cos_p) = ops::sin_cos(self.frozen_phase[bin_base + k]);
            let held_re = mag * cos_p;
            let held_im = mag * sin_p;

            self.re[k] = live_re + mix * (held_re - live_re);
            self.im[k] = live_im + mix * (held_im - live_im);
        }

        // Hermitian symmetry for a real inverse transform.
        self.im[0] = 0.0;
        if self.half < self.size {
            self.im[self.half] = 0.0;
        }
        for j in 1..self.half {
            self.re[self.size - j] = self.re[j];
            self.im[self.size - j] = -self.im[j];
        }

        self.fft.inverse(&mut self.re, &mut self.im);

        // Synthesis window and overlap-add.
        for n in 0..self.size {
            self.out_accum[fifo_base + n] += self.win[n] * self.re[n] * self.ola_norm;
        }
    }
}

impl AudioNode for SpectralFreezeNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());
        if channels == 0 || frames == 0 {
            for ch in 0..available {
                let src = input.channel(ch);
                let dst = output.channel_mut(ch);
                let n = frames.min(src.len()).min(dst.len());
                dst[..n].copy_from_slice(&src[..n]);
            }
            return;
        }

        let size = self.size;
        let hop = self.hop;
        let fifo_latency = self.fifo_latency;
        let mut rover = self.rover;

        for i in 0..frames {
            for ch in 0..channels {
                let fifo_base = ch * size;
                let x = input.channel(ch)[i];
                let x = if x.is_finite() { x } else { 0.0 };
                self.in_fifo[fifo_base + rover] = x;
                let y = self.out_fifo[fifo_base + (rover - fifo_latency)];
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
                self.process_frame(ch);

                // Publish one hop of finished output, then slide the buffers.
                self.out_fifo[fifo_base..fifo_base + hop]
                    .copy_from_slice(&self.out_accum[fifo_base..fifo_base + hop]);
                for n in 0..(size - hop) {
                    self.out_accum[fifo_base + n] = self.out_accum[fifo_base + n + hop];
                }
                for n in (size - hop)..size {
                    self.out_accum[fifo_base + n] = 0.0;
                }
                for n in 0..(size - hop) {
                    self.in_fifo[fifo_base + n] = self.in_fifo[fifo_base + n + hop];
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
        for value in &mut self.frozen_mag {
            *value = 0.0;
        }
        for value in &mut self.frozen_phase {
            *value = 0.0;
        }
        self.rng = FreezeRng::new(self.seed);
        self.freeze_mix = self.freeze_target;
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

    /// Single-frequency DFT magnitude (Goertzel), normalised by length.
    fn goertzel(signal: &[Sample], freq: Sample) -> Sample {
        let n = signal.len();
        if n == 0 {
            return 0.0;
        }
        let w = TAU * freq / SR as Sample;
        let (sin_w, cos_w) = ops::sin_cos(w);
        let coeff = 2.0 * cos_w;
        let mut s_prev = 0.0f32;
        let mut s_prev2 = 0.0f32;
        for &x in signal {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        let real = s_prev - s_prev2 * cos_w;
        let imag = s_prev2 * sin_w;
        ops::sqrt(real * real + imag * imag) / n as Sample
    }

    fn run_mono(node: &mut SpectralFreezeNode, signal: &[Sample]) -> Vec<Sample> {
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

    fn run_stereo(
        node: &mut SpectralFreezeNode,
        left: &[Sample],
        right: &[Sample],
    ) -> (Vec<Sample>, Vec<Sample>) {
        let len = left.len();
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.channel_mut(0).copy_from_slice(left);
        input.channel_mut(1).copy_from_slice(right);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        (
            outputs[0].channel(0).to_vec(),
            outputs[0].channel(1).to_vec(),
        )
    }

    /// Captures the frozen sustain tail: track a tone, latch, then feed silence.
    fn frozen_tail(diffusion: Sample, tail_len: usize) -> Vec<Sample> {
        let params = SpectralFreezeParams {
            frozen: false,
            diffusion,
            seed: DEFAULT_SEED,
        };
        let mut node = SpectralFreezeNode::new(SR, 1, 1_024, params);
        let tone = sine(1_000.0, 0.5, 8_192);
        let _ = run_mono(&mut node, &tone);
        node.set_frozen(true);
        let _ = run_mono(&mut node, &tone);
        run_mono(&mut node, &vec![0.0; tail_len])
    }

    #[test]
    fn requested_size_rounds_up_to_power_of_two() {
        let node = SpectralFreezeNode::new(SR, 1, 1_000, SpectralFreezeParams::default());
        assert_eq!(node.fft_size(), 1_024);
    }

    #[test]
    fn small_request_hits_minimum() {
        let node = SpectralFreezeNode::new(SR, 1, 1, SpectralFreezeParams::default());
        assert_eq!(node.fft_size(), MIN_FREEZE_FFT_SIZE);
    }

    #[test]
    fn hop_is_fraction_of_size() {
        let node = SpectralFreezeNode::new(SR, 1, 1_024, SpectralFreezeParams::default());
        assert_eq!(node.hop(), 1_024 / OVERLAP_FACTOR);
    }

    #[test]
    fn latency_is_one_full_frame() {
        let node = SpectralFreezeNode::new(SR, 1, 1_024, SpectralFreezeParams::default());
        assert_eq!(node.latency_frames(), 1_024);
        assert_eq!(node.latency_frames() as usize, node.fft_size());
    }

    #[test]
    fn params_sanitised_clamps() {
        let p = SpectralFreezeParams {
            frozen: true,
            diffusion: 9.0,
            seed: 7,
        }
        .sanitised();
        assert!((p.diffusion - MAX_DIFFUSION).abs() < 1e-6);
        assert!(p.frozen);
        assert_eq!(p.seed, 7);
        let p = SpectralFreezeParams {
            frozen: false,
            diffusion: Sample::NAN,
            seed: 1,
        }
        .sanitised();
        assert!(p.diffusion.abs() < 1e-9);
    }

    #[test]
    fn set_diffusion_clamps_and_rejects_non_finite() {
        let mut node = SpectralFreezeNode::new(SR, 1, 256, SpectralFreezeParams::default());
        node.set_diffusion(100.0);
        assert!((node.diffusion() - MAX_DIFFUSION).abs() < 1e-6);
        node.set_diffusion(-1.0);
        assert!(node.diffusion().abs() < 1e-9);
        node.set_diffusion(Sample::INFINITY);
        assert!(node.diffusion().abs() < 1e-9);
    }

    #[test]
    fn set_frozen_updates_state() {
        let mut node = SpectralFreezeNode::new(SR, 1, 256, SpectralFreezeParams::default());
        assert!(!node.is_frozen());
        node.set_frozen(true);
        assert!(node.is_frozen());
        node.set_frozen(false);
        assert!(!node.is_frozen());
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = SpectralFreezeNode::new(SR, 1, 512, SpectralFreezeParams::default());
        let out = run_mono(&mut node, &vec![0.0; 4_096]);
        assert!(out.iter().all(|&y| y.abs() < 1e-6), "silence produced output");
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = SpectralFreezeNode::new(SR, 1, 512, SpectralFreezeParams::default());
        let mut signal = sine(1_000.0, 0.5, 4_096);
        signal[100] = Sample::NAN;
        signal[2_000] = Sample::INFINITY;
        let out = run_mono(&mut node, &signal);
        assert!(out.iter().all(|&y| y.is_finite()), "non-finite leaked");
    }

    #[test]
    fn zero_frames_safe() {
        let mut node = SpectralFreezeNode::new(SR, 1, 512, SpectralFreezeParams::default());
        let input = AudioBuffer::new(ChannelLayout::Mono, 1);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn unfrozen_reconstructs_input() {
        let mut node = SpectralFreezeNode::new(SR, 1, 1_024, SpectralFreezeParams::default());
        let signal = sine(1_000.0, 0.5, 8_192);
        let out = run_mono(&mut node, &signal);
        let tail = &out[4_096..];
        let m1000 = goertzel(tail, 1_000.0);
        let m2000 = goertzel(tail, 2_000.0);
        assert!(m1000 > m2000 * 4.0, "fundamental not dominant: {m1000} vs {m2000}");
        let level = rms(tail) / rms(&signal[4_096..]);
        assert!((0.7..1.3).contains(&level), "unity reconstruction drifted: {level}");
    }

    #[test]
    fn frozen_sustains_after_input_stops() {
        let tail = frozen_tail(0.0, 8_192);
        // After the input has gone silent, the latched spectrum keeps ringing.
        let energy = rms(&tail[2_048..]);
        assert!(energy > 0.05, "frozen spectrum decayed to silence: {energy}");
    }

    #[test]
    fn frozen_single_tone_sustains_frequency() {
        let tail = frozen_tail(0.0, 8_192);
        let back = &tail[2_048..];
        let m1000 = goertzel(back, 1_000.0);
        let m2000 = goertzel(back, 2_000.0);
        assert!(m1000 > m2000 * 4.0, "sustained tone off frequency: {m1000} vs {m2000}");
    }

    #[test]
    fn diffusion_decorrelates_frozen_output() {
        let coherent = frozen_tail(0.0, 8_192);
        let diffuse = frozen_tail(0.7, 8_192);
        let a = &coherent[4_096..];
        let b = &diffuse[4_096..];
        let diff: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from((x - y).abs()))
            .sum();
        let mean = (diff / a.len() as f64) as Sample;
        assert!(mean > 1e-3, "diffusion did not change the sustain: {mean}");
    }

    #[test]
    fn toggle_is_click_free() {
        let mut node = SpectralFreezeNode::new(SR, 1, 1_024, SpectralFreezeParams::default());
        let signal = sine(1_000.0, 0.5, 4_096);
        let _ = run_mono(&mut node, &signal);
        node.set_frozen(true);
        let out = run_mono(&mut node, &signal);
        // The crossfade bounds the output; no gross discontinuity may appear.
        assert!(out.iter().all(|&y| y.is_finite() && y.abs() < 2.0), "freeze toggle spiked");
        let max_step = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_step < 0.5, "sample-to-sample jump too large: {max_step}");
    }

    #[test]
    fn stereo_identical_channels_match() {
        let mut node = SpectralFreezeNode::new(SR, 2, 512, SpectralFreezeParams::default());
        let signal = sine(800.0, 0.4, 6_144);
        let (l, r) = run_stereo(&mut node, &signal, &signal);
        for (a, b) in l.iter().zip(&r) {
            assert!((a - b).abs() < 1e-6, "stereo channels diverged: {a} vs {b}");
        }
    }

    #[test]
    fn surplus_channels_pass_through() {
        let mut node = SpectralFreezeNode::new(SR, 1, 512, SpectralFreezeParams::default());
        let signal = sine(1_000.0, 0.5, 2_048);
        let (_l, r) = run_stereo(&mut node, &signal, &signal);
        assert_eq!(r.len(), signal.len());
        for (y, x) in r.iter().zip(&signal) {
            assert!((y - x).abs() < 1e-6, "surplus channel altered");
        }
    }

    #[test]
    fn reset_clears_tail() {
        let mut node = SpectralFreezeNode::new(SR, 1, 512, SpectralFreezeParams::default());
        let _ = run_mono(&mut node, &sine(1_000.0, 0.6, 4_096));
        node.reset();
        let out = run_mono(&mut node, &vec![0.0; 2_048]);
        assert!(out.iter().all(|&y| y.abs() < 1e-6), "reset left a tail");
    }
}
