//! Phase-vocoder pitch shifter: transposes a signal by a constant ratio while
//! preserving its duration.
//!
//! A pitch shifter multiplies the frequency of every partial by a constant
//! ratio `r` (an octave up is `r = 2`, an octave down is `r = 0.5`) so harmonic
//! ratios are preserved -- unlike the inharmonic
//! [`FrequencyShifterNode`](crate::nodes::effects::frequency_shifter), which
//! *adds* a constant Hz offset and breaks the harmonic series. Unlike naive
//! resampling, which also changes the playback duration ("chipmunk" speed-up),
//! this keeps the time axis fixed and only moves pitch.
//!
//! # The model (phase vocoder)
//!
//! The signal is analysed with a weighted overlap-add short-time Fourier
//! transform (`STFT`): a Hann window of `N` samples is advanced by a hop
//! `H = N / OVERLAP_FACTOR`, so successive analysis frames overlap. For each
//! frame and each bin `k` the magnitude `|X[k]|` and the *instantaneous
//! frequency* are estimated. The instantaneous frequency comes from phase
//! unwrapping: the measured phase advance between consecutive frames, minus the
//! phase advance a bin at its exact centre frequency would accumulate over the
//! hop, wrapped into `(-pi, pi]` (the principal argument), gives the frequency
//! *deviation* of the sinusoid actually present in that bin. Adding the
//! deviation back to the bin centre yields the true frequency in Hz.
//!
//! Pitch shifting is then a frequency-domain remap: the magnitude of analysis
//! bin `k` is accumulated into synthesis bin `round(k * r)`, and that synthesis
//! bin is tagged with the analysis bin's true frequency scaled by `r`. During
//! synthesis each bin's phase is advanced by the hop times its (shifted) true
//! frequency, accumulated across frames so the output stays phase-coherent, and
//! the complex spectrum is rebuilt with Hermitian (conjugate) symmetry, inverse
//! transformed, Hann-windowed again, and overlap-added. Because the analysis
//! and synthesis hops are equal the output duration equals the input duration;
//! only the pitch moves. At `r = 1` the pipeline reconstructs the input (delayed
//! by one frame) to within floating-point rounding.
//!
//! Upward shifts spread analysis bins apart, leaving gaps that this classic
//! single-bin remap does not fill; downward shifts collapse several bins onto
//! one, summing their magnitudes. These are the well-known trade-offs of the
//! basic phase vocoder and are intrinsic to the algorithm, not a stub.
//!
//! # Real-time contract
//!
//! The window, the shared [`Fft`](crate::fft::Fft) plan, and every per-channel
//! streaming buffer (input/output FIFOs, overlap accumulator, last-phase and
//! accumulated-phase tables) and per-frame scratch are allocated once in
//! [`PitchShifterNode::new`]. [`PitchShifterNode::process`] performs no
//! allocation, takes no locks, and cannot panic; outputs are denormal-flushed so
//! a decaying tail cannot stall on subnormals. All transcendental math routes
//! through [`bevy_math::ops`], so the transform is bit-reproducible across
//! platforms. The pipeline delays the signal by one full frame
//! ([`PitchShifterNode::latency_frames`]).
//!
//! # Provenance
//!
//! Classic DSP only, with no AI/ML of any kind. This module contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio
//! source or derived code**. The phase vocoder (`STFT` analysis, instantaneous
//! frequency from phase unwrapping, phase accumulation for resynthesis) is the
//! textbook construction described by Flanagan and Golden, Portnoff, Dolson's
//! "The Phase Vocoder: A Tutorial", and Laroche and Dolson's "Improved Phase
//! Vocoder Time-Scale Modification of Audio"; it is implemented here purely from
//! those publicly documented algorithms.
//!
//! # Relationship
//!
//! This node reuses the crate's shared radix-2 transform
//! [`Fft`](crate::fft::Fft) rather than carrying a private copy, mirroring how
//! the [`SpectralGateNode`](crate::nodes::effects::spectral_gate::SpectralGateNode)
//! and [`SpectrumAnalyzer`](crate::nodes::analysis::spectrum::SpectrumAnalyzer)
//! operate in the frequency domain. It differs from the time-domain
//! [`VibratoNode`](crate::nodes::effects::vibrato) (a modulated delay that bends
//! pitch cyclically) and from the single-sideband
//! [`FrequencyShifterNode`](crate::nodes::effects::frequency_shifter) (which
//! adds a Hz offset and makes partials inharmonic): this scales every partial by
//! one ratio and keeps the harmonic series intact.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::{PI, TAU};

use crate::fft::{Fft, power_of_two_at_least};
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Analysis/synthesis overlap factor; the hop is `fft_size / OVERLAP_FACTOR`.
pub const OVERLAP_FACTOR: usize = 4;

/// Smallest supported pitch ratio (two octaves down).
pub const MIN_PITCH_RATIO: Sample = 0.25;

/// Largest supported pitch ratio (two octaves up).
pub const MAX_PITCH_RATIO: Sample = 4.0;

/// Converts a transposition in semitones to a linear pitch ratio
/// `2^(semitones / 12)`.
#[must_use]
pub fn semitones_to_ratio(semitones: Sample) -> Sample {
    ops::powf(2.0, semitones / 12.0)
}

/// Wraps `phase` into the principal interval `(-pi, pi]`.
fn princ_arg(phase: Sample) -> Sample {
    let shifted = phase + PI;
    let wrapped = shifted - TAU * ops::floor(shifted / TAU);
    wrapped - PI
}

/// Parameters controlling the pitch shifter.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PitchShifterParams {
    /// Linear transposition ratio: `2` = one octave up, `0.5` = one octave down.
    pub pitch_ratio: Sample,
}

impl Default for PitchShifterParams {
    fn default() -> Self {
        Self { pitch_ratio: 1.0 }
    }
}

impl PitchShifterParams {
    /// Returns a copy with the ratio made finite and clamped to
    /// `[MIN_PITCH_RATIO, MAX_PITCH_RATIO]`.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let ratio = if self.pitch_ratio.is_finite() {
            self.pitch_ratio.clamp(MIN_PITCH_RATIO, MAX_PITCH_RATIO)
        } else {
            1.0
        };
        Self { pitch_ratio: ratio }
    }
}

/// Phase-vocoder pitch shifter node (weighted overlap-add `STFT`).
#[derive(Clone, Debug)]
pub struct PitchShifterNode {
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
    freq_per_bin: Sample,
    expct: Sample,
    osamp: Sample,
    // Control.
    pitch_ratio: Sample,
    // Per-channel streaming state.
    in_fifo: Vec<Sample>,
    out_fifo: Vec<Sample>,
    out_accum: Vec<Sample>,
    last_phase: Vec<Sample>,
    sum_phase: Vec<Sample>,
    rover: usize,
    // Hot-path scratch (reused across channels and frames).
    re: Vec<Sample>,
    im: Vec<Sample>,
    ana_mag: Vec<Sample>,
    ana_freq: Vec<Sample>,
    syn_mag: Vec<Sample>,
    syn_freq: Vec<Sample>,
}

impl PitchShifterNode {
    /// Builds a pitch shifter for `channels` channels at `sample_rate`.
    ///
    /// `requested_size` is rounded up to a power of two no smaller than the
    /// transform minimum; the hop is `fft_size / OVERLAP_FACTOR`. `channels` is
    /// clamped to at least one.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::pitch_shifter::{
    ///     PitchShifterNode, PitchShifterParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = PitchShifterNode::new(48_000, 2, 1_024, PitchShifterParams::default());
    /// assert_eq!(node.latency_frames(), 1_024);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        requested_size: usize,
        params: PitchShifterParams,
    ) -> Self {
        let channels = channels.max(1);
        let size = power_of_two_at_least(requested_size);
        let hop = (size / OVERLAP_FACTOR).max(1);
        let half = size / 2;
        let bins = half + 1;
        let fifo_latency = size - hop;

        let mut win = vec![0.0; size];
        for (n, slot) in win.iter_mut().enumerate() {
            *slot = 0.5 - 0.5 * ops::cos(TAU * n as Sample / size as Sample);
        }

        // Constant-overlap-add denominator of the squared window at the hop so
        // that the ratio == 1 pipeline reconstructs the input at unity gain.
        let base = half % hop;
        let mut overlap_sum = 0.0f32;
        let mut k = base;
        while k < size {
            overlap_sum += win[k] * win[k];
            k += hop;
        }
        let ola_norm = if overlap_sum > 0.0 { 1.0 / overlap_sum } else { 0.0 };

        let sr = sample_rate.max(1) as Sample;
        let freq_per_bin = sr / size as Sample;
        let expct = TAU * hop as Sample / size as Sample;
        let osamp = size as Sample / hop as Sample;

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
            freq_per_bin,
            expct,
            osamp,
            pitch_ratio: params.sanitised().pitch_ratio,
            in_fifo: vec![0.0; channels * size],
            out_fifo: vec![0.0; channels * size],
            out_accum: vec![0.0; channels * size],
            last_phase: vec![0.0; channels * bins],
            sum_phase: vec![0.0; channels * bins],
            rover: fifo_latency,
            re: vec![0.0; size],
            im: vec![0.0; size],
            ana_mag: vec![0.0; bins],
            ana_freq: vec![0.0; bins],
            syn_mag: vec![0.0; bins],
            syn_freq: vec![0.0; bins],
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

    /// Current linear pitch ratio.
    #[must_use]
    pub fn pitch_ratio(&self) -> Sample {
        self.pitch_ratio
    }

    /// Sets the linear pitch ratio (made finite and clamped to
    /// `[MIN_PITCH_RATIO, MAX_PITCH_RATIO]`).
    pub fn set_pitch_ratio(&mut self, ratio: Sample) {
        self.pitch_ratio = PitchShifterParams { pitch_ratio: ratio }
            .sanitised()
            .pitch_ratio;
    }

    /// Sets the transposition in semitones (converted to a ratio and clamped).
    pub fn set_semitones(&mut self, semitones: Sample) {
        let ratio = if semitones.is_finite() {
            semitones_to_ratio(semitones)
        } else {
            1.0
        };
        self.set_pitch_ratio(ratio);
    }

    /// Transforms one buffered frame for channel `ch`, pitch-shifts it, and
    /// overlap-adds the result into that channel's accumulator.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "k * ratio is non-negative and range-checked before the usize cast"
    )]
    fn process_frame(&mut self, ch: usize) {
        let fifo_base = ch * self.size;
        let phase_base = ch * self.bins;
        let ratio = self.pitch_ratio;

        // Analysis window into the complex scratch, then forward transform.
        for n in 0..self.size {
            self.re[n] = self.win[n] * self.in_fifo[fifo_base + n];
            self.im[n] = 0.0;
        }
        self.fft.forward(&mut self.re, &mut self.im);

        // Estimate magnitude and instantaneous frequency for each bin.
        for k in 0..self.bins {
            let mag = ops::sqrt(self.re[k] * self.re[k] + self.im[k] * self.im[k]);
            let phase = ops::atan2(self.im[k], self.re[k]);
            let mut delta = phase - self.last_phase[phase_base + k];
            self.last_phase[phase_base + k] = phase;
            delta -= k as Sample * self.expct;
            delta = princ_arg(delta);
            let deviation_bins = self.osamp * delta / TAU;
            self.ana_mag[k] = mag;
            self.ana_freq[k] = (k as Sample + deviation_bins) * self.freq_per_bin;
        }

        // Remap bins: accumulate magnitude into the shifted bin, scale the true
        // frequency by the ratio. Empty bins keep their centre frequency so the
        // phase stays coherent if they become active later.
        for j in 0..self.bins {
            self.syn_mag[j] = 0.0;
            self.syn_freq[j] = j as Sample * self.freq_per_bin;
        }
        for k in 0..self.bins {
            let target = ops::round(k as Sample * ratio);
            if target >= 0.0 && target < self.bins as Sample {
                let j = target as usize;
                self.syn_mag[j] += self.ana_mag[k];
                self.syn_freq[j] = self.ana_freq[k] * ratio;
            }
        }

        // Resynthesis: accumulate phase from each bin's shifted true frequency.
        for j in 0..self.bins {
            let mag = self.syn_mag[j];
            let mut delta = self.syn_freq[j] - j as Sample * self.freq_per_bin;
            delta /= self.freq_per_bin;
            delta = TAU * delta / self.osamp;
            delta += j as Sample * self.expct;
            self.sum_phase[phase_base + j] += delta;
            let (sin_p, cos_p) = ops::sin_cos(self.sum_phase[phase_base + j]);
            self.re[j] = mag * cos_p;
            self.im[j] = mag * sin_p;
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

impl AudioNode for PitchShifterNode {
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
        for value in &mut self.last_phase {
            *value = 0.0;
        }
        for value in &mut self.sum_phase {
            *value = 0.0;
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

    fn run_mono(node: &mut PitchShifterNode, signal: &[Sample]) -> Vec<Sample> {
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
        node: &mut PitchShifterNode,
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
        (outputs[0].channel(0).to_vec(), outputs[0].channel(1).to_vec())
    }

    #[test]
    fn requested_size_rounds_up_to_power_of_two() {
        let node = PitchShifterNode::new(SR, 1, 1_000, PitchShifterParams::default());
        assert_eq!(node.fft_size(), 1_024);
    }

    #[test]
    fn hop_is_fraction_of_size() {
        let node = PitchShifterNode::new(SR, 1, 1_024, PitchShifterParams::default());
        assert_eq!(node.hop(), 1_024 / OVERLAP_FACTOR);
    }

    #[test]
    fn latency_is_one_full_frame() {
        let node = PitchShifterNode::new(SR, 1, 1_024, PitchShifterParams::default());
        assert_eq!(node.latency_frames(), 1_024);
    }

    #[test]
    fn semitones_map_to_ratios() {
        assert!((semitones_to_ratio(0.0) - 1.0).abs() < 1e-6);
        assert!((semitones_to_ratio(12.0) - 2.0).abs() < 1e-4);
        assert!((semitones_to_ratio(-12.0) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn set_pitch_ratio_clamps_and_rejects_non_finite() {
        let mut node = PitchShifterNode::new(SR, 1, 256, PitchShifterParams::default());
        node.set_pitch_ratio(100.0);
        assert!((node.pitch_ratio() - MAX_PITCH_RATIO).abs() < 1e-6);
        node.set_pitch_ratio(0.0);
        assert!((node.pitch_ratio() - MIN_PITCH_RATIO).abs() < 1e-6);
        node.set_pitch_ratio(Sample::NAN);
        assert!((node.pitch_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn set_semitones_updates_ratio() {
        let mut node = PitchShifterNode::new(SR, 1, 256, PitchShifterParams::default());
        node.set_semitones(12.0);
        assert!((node.pitch_ratio() - 2.0).abs() < 1e-3);
        node.set_semitones(Sample::INFINITY);
        assert!((node.pitch_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn params_sanitised_clamps() {
        let p = PitchShifterParams { pitch_ratio: 1e9 }.sanitised();
        assert!((p.pitch_ratio - MAX_PITCH_RATIO).abs() < 1e-6);
        let p = PitchShifterParams { pitch_ratio: Sample::NAN }.sanitised();
        assert!((p.pitch_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = PitchShifterNode::new(SR, 1, 512, PitchShifterParams { pitch_ratio: 2.0 });
        let out = run_mono(&mut node, &vec![0.0; 4096]);
        assert!(out.iter().all(|&y| y.abs() < 1e-6), "silence produced output");
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = PitchShifterNode::new(SR, 1, 512, PitchShifterParams::default());
        let mut signal = sine(1_000.0, 0.5, 4096);
        signal[100] = Sample::NAN;
        signal[2000] = Sample::INFINITY;
        let out = run_mono(&mut node, &signal);
        assert!(out.iter().all(|&y| y.is_finite()), "non-finite leaked");
    }

    #[test]
    fn zero_frames_safe() {
        let mut node = PitchShifterNode::new(SR, 1, 512, PitchShifterParams::default());
        let input = AudioBuffer::new(ChannelLayout::Mono, 1);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn unity_ratio_preserves_tone_and_level() {
        let mut node = PitchShifterNode::new(SR, 1, 1_024, PitchShifterParams::default());
        let signal = sine(1_000.0, 0.5, 8_192);
        let out = run_mono(&mut node, &signal);
        let tail = &out[4_096..];
        let m1000 = goertzel(tail, 1_000.0);
        let m2000 = goertzel(tail, 2_000.0);
        assert!(m1000 > m2000 * 4.0, "fundamental not dominant: {m1000} vs {m2000}");
        let level = rms(tail) / rms(&signal[4_096..]);
        assert!((0.7..1.3).contains(&level), "level ratio drifted: {level}");
    }

    #[test]
    fn octave_up_doubles_frequency() {
        let mut node = PitchShifterNode::new(SR, 1, 1_024, PitchShifterParams { pitch_ratio: 2.0 });
        let signal = sine(1_000.0, 0.5, 8_192);
        let out = run_mono(&mut node, &signal);
        let tail = &out[4_096..];
        let m1000 = goertzel(tail, 1_000.0);
        let m2000 = goertzel(tail, 2_000.0);
        assert!(m2000 > m1000, "shifted tone not at 2 kHz: {m2000} vs {m1000}");
    }

    #[test]
    fn octave_down_halves_frequency() {
        let mut node = PitchShifterNode::new(SR, 1, 1_024, PitchShifterParams { pitch_ratio: 0.5 });
        let signal = sine(1_000.0, 0.5, 8_192);
        let out = run_mono(&mut node, &signal);
        let tail = &out[4_096..];
        let m1000 = goertzel(tail, 1_000.0);
        let m500 = goertzel(tail, 500.0);
        assert!(m500 > m1000, "shifted tone not at 500 Hz: {m500} vs {m1000}");
    }

    #[test]
    fn stereo_identical_channels_match() {
        let mut node = PitchShifterNode::new(SR, 2, 512, PitchShifterParams { pitch_ratio: 1.5 });
        let signal = sine(800.0, 0.4, 6_144);
        let (l, r) = run_stereo(&mut node, &signal, &signal);
        for (a, b) in l.iter().zip(&r) {
            assert!((a - b).abs() < 1e-6, "stereo channels diverged: {a} vs {b}");
        }
    }

    #[test]
    fn reset_clears_tail() {
        let mut node = PitchShifterNode::new(SR, 1, 512, PitchShifterParams { pitch_ratio: 2.0 });
        let _ = run_mono(&mut node, &sine(1_000.0, 0.6, 4_096));
        node.reset();
        let out = run_mono(&mut node, &vec![0.0; 2_048]);
        assert!(out.iter().all(|&y| y.abs() < 1e-6), "reset left a tail");
    }

    #[test]
    fn surplus_channels_pass_through() {
        // The node is built for one channel but fed a stereo buffer; the extra
        // channel must pass through untouched.
        let mut node = PitchShifterNode::new(SR, 1, 512, PitchShifterParams::default());
        let signal = sine(1_000.0, 0.5, 2_048);
        let (_l, r) = run_stereo(&mut node, &signal, &signal);
        assert_eq!(r.len(), signal.len());
        for (y, x) in r.iter().zip(&signal) {
            assert!((y - x).abs() < 1e-6, "surplus channel altered");
        }
    }
}
