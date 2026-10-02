//! Frequency-domain spectral compressor: a short-time Fourier transform
//! (`STFT`) dynamics processor that applies a feed-forward compression law to
//! each spectral bin independently, taming loud regions of the spectrum while
//! leaving quieter regions untouched.
//!
//! A time-domain [`CompressorNode`](crate::nodes::dynamics::compressor) reduces
//! the whole signal from a single broadband level detector: when a loud bass
//! note triggers gain reduction, the entire spectrum ducks with it. This node
//! instead compresses every frequency bin on its own detector, so a loud band
//! is attenuated without pumping the rest of the spectrum -- a per-bin
//! dynamics control a broadband compressor cannot provide.
//!
//! # Model
//!
//! The signal is processed with a weighted overlap-add (`OLA`) `STFT`. Each
//! analysis frame of `fft_size` samples is multiplied by a Hann analysis
//! window, transformed with the crate's shared `radix-2` decimation-in-time
//! (`DIT`) fast Fourier transform ([`Fft`](crate::fft::Fft)), compressed bin by
//! bin, inverse-transformed, multiplied by a matching Hann synthesis window,
//! and overlap-added with a hop of `fft_size / COMPRESSOR_OVERLAP_FACTOR`
//! (75 percent overlap). The squared Hann window at this hop satisfies the
//! constant-overlap-add (`COLA`) condition, so with every bin held at unity the
//! output reconstructs the input exactly (apart from the processing latency).
//!
//! For each bin the single-sided magnitude is normalised by the window
//! coherent gain into a `dBFS`-referenced amplitude (a full-scale sinusoid
//! sitting on a bin reads back `0 dBFS`) and converted to decibels. A standard
//! feed-forward compressor computes the target gain: below `threshold_db` the
//! bin passes at unity, above it the excess is scaled by `1 / ratio`, and a
//! quadratic soft knee of width `knee_db` smooths the transition. A constant
//! `makeup_db` gain is then folded in. The per-bin gain follows its target
//! through a one-pole smoother clocked once per hop, using the `attack_ms` time
//! constant while gain reduction increases and the `release_ms` time constant
//! while it recovers, which suppresses the "musical noise" that instantaneous
//! per-bin dynamics produce. A dry / wet `mix` blends the compressed spectrum
//! against the original; because the dry path is the same bin scaled by one,
//! the blend reduces to a single effective multiplier applied to the bin and
//! its Hermitian mirror, so the inverse transform stays real.
//!
//! # Real-time contract
//!
//! All ring, overlap-add, window, gain, and scratch buffers, together with the
//! shared [`Fft`](crate::fft::Fft) plan (its twiddle and bit-reversal tables),
//! are allocated once at construction. [`SpectralCompressorNode::process`]
//! performs no allocation, locking, or panic on the hot path; non-finite input
//! samples are treated as silence. The node reports a processing latency of
//! `fft_size` frames via [`SpectralCompressorNode::latency_frames`].
//!
//! # Relationship
//!
//! This node differs from the sibling [`spectral_gate`](crate::nodes::effects::spectral_gate)
//! node, which applies hard / soft per-bin *gating* (a bin is either open or
//! pulled to a fixed floor): here each bin receives a *continuous*,
//! ratio-based gain reduction that grows smoothly with how far it exceeds the
//! threshold. It also differs from the time-domain
//! [`CompressorNode`](crate::nodes::dynamics::compressor), which detects one
//! broadband level and compresses the entire signal together; this node runs an
//! independent compressor per frequency bin.
//!
//! # Provenance
//!
//! The weighted overlap-add `STFT`, the Hann window, the `radix-2` Cooley-Tukey
//! `FFT` (factored into the crate's shared [`Fft`](crate::fft::Fft) primitive),
//! and the standard feed-forward (downward) compressor law with a quadratic
//! soft knee are classic, publicly documented DSP techniques found in any
//! signal-processing text (for example the overlap-add `STFT` described by
//! Allen and Rabiner, the Hann window, and the standard dynamic-range
//! compression gain computer). This is pure classic DSP with no AI or ML. This
//! module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Web Audio, or STK source or derived code**; only the
//! widely documented transform, window, and compressor formulas are used.

use alloc::{vec, vec::Vec};
use bevy_math::ops;
use core::f32::consts::TAU;

use crate::fft::Fft;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{db_to_linear, flush_denormal, linear_to_db, Sample};

/// Smallest permitted transform size, in samples.
pub const MIN_COMPRESSOR_FFT_SIZE: usize = 64;

/// Default transform size, in samples.
pub const DEFAULT_COMPRESSOR_FFT_SIZE: usize = 1024;

/// Overlap factor: the hop is `fft_size / COMPRESSOR_OVERLAP_FACTOR`
/// (75 percent overlap), the standard Hann weighted overlap-add choice that
/// satisfies `COLA`.
pub const COMPRESSOR_OVERLAP_FACTOR: usize = 4;

/// Default compression threshold in `dBFS`.
pub const DEFAULT_COMPRESSOR_THRESHOLD_DB: Sample = -24.0;

/// Default compression ratio (input : output above the threshold).
pub const DEFAULT_COMPRESSOR_RATIO: Sample = 4.0;

/// Default soft-knee width in decibels.
pub const DEFAULT_COMPRESSOR_KNEE_DB: Sample = 6.0;

/// Default makeup gain in decibels.
pub const DEFAULT_COMPRESSOR_MAKEUP_DB: Sample = 0.0;

/// Default attack (gain-reduction) time constant in milliseconds.
pub const DEFAULT_COMPRESSOR_ATTACK_MS: Sample = 5.0;

/// Default release (recovery) time constant in milliseconds.
pub const DEFAULT_COMPRESSOR_RELEASE_MS: Sample = 80.0;

/// Default dry / wet mix (`1` is fully compressed).
pub const DEFAULT_COMPRESSOR_MIX: Sample = 1.0;

/// Largest permitted compression ratio.
pub const MAX_COMPRESSOR_RATIO: Sample = 100.0;

/// Largest permitted soft-knee width in decibels.
pub const MAX_COMPRESSOR_KNEE_DB: Sample = 48.0;

/// Largest permitted makeup-gain magnitude in decibels (clamp bound).
pub const MAX_COMPRESSOR_MAKEUP_DB: Sample = 48.0;

/// Largest permitted attack / release time constant in milliseconds.
pub const MAX_COMPRESSOR_TIME_MS: Sample = 10_000.0;

/// Smallest permitted threshold in `dBFS` (clamp bound).
pub const MIN_COMPRESSOR_THRESHOLD_DB: Sample = -120.0;

/// Largest permitted threshold in `dBFS` (clamp bound).
pub const MAX_COMPRESSOR_THRESHOLD_DB: Sample = 24.0;

/// Tunable spectral-compressor parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectralCompressorParams {
    /// Threshold in `dBFS`: bins below it pass, bins above are compressed.
    pub threshold_db: Sample,
    /// Compression ratio (at least `1`); `1` is no compression.
    pub ratio: Sample,
    /// Soft-knee width in decibels (at least `0`); `0` is a hard knee.
    pub knee_db: Sample,
    /// Makeup gain in decibels applied to every bin after compression.
    pub makeup_db: Sample,
    /// Attack (gain-reduction) time constant in milliseconds (`0` is instant).
    pub attack_ms: Sample,
    /// Release (recovery) time constant in milliseconds (`0` is instant).
    pub release_ms: Sample,
    /// Dry / wet mix in `[0, 1]`: `0` is fully dry, `1` is fully compressed.
    pub mix: Sample,
}

impl Default for SpectralCompressorParams {
    fn default() -> Self {
        Self {
            threshold_db: DEFAULT_COMPRESSOR_THRESHOLD_DB,
            ratio: DEFAULT_COMPRESSOR_RATIO,
            knee_db: DEFAULT_COMPRESSOR_KNEE_DB,
            makeup_db: DEFAULT_COMPRESSOR_MAKEUP_DB,
            attack_ms: DEFAULT_COMPRESSOR_ATTACK_MS,
            release_ms: DEFAULT_COMPRESSOR_RELEASE_MS,
            mix: DEFAULT_COMPRESSOR_MIX,
        }
    }
}

impl SpectralCompressorParams {
    /// Clamps every field into its valid range, replacing non-finite values
    /// with the defaults.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let threshold_db = if self.threshold_db.is_finite() {
            self.threshold_db
                .clamp(MIN_COMPRESSOR_THRESHOLD_DB, MAX_COMPRESSOR_THRESHOLD_DB)
        } else {
            d.threshold_db
        };
        let ratio = if self.ratio.is_finite() {
            self.ratio.clamp(1.0, MAX_COMPRESSOR_RATIO)
        } else {
            d.ratio
        };
        let knee_db = if self.knee_db.is_finite() {
            self.knee_db.clamp(0.0, MAX_COMPRESSOR_KNEE_DB)
        } else {
            d.knee_db
        };
        let makeup_db = if self.makeup_db.is_finite() {
            self.makeup_db
                .clamp(-MAX_COMPRESSOR_MAKEUP_DB, MAX_COMPRESSOR_MAKEUP_DB)
        } else {
            d.makeup_db
        };
        let attack_ms = if self.attack_ms.is_finite() {
            self.attack_ms.clamp(0.0, MAX_COMPRESSOR_TIME_MS)
        } else {
            d.attack_ms
        };
        let release_ms = if self.release_ms.is_finite() {
            self.release_ms.clamp(0.0, MAX_COMPRESSOR_TIME_MS)
        } else {
            d.release_ms
        };
        let mix = if self.mix.is_finite() {
            self.mix.clamp(0.0, 1.0)
        } else {
            d.mix
        };
        Self {
            threshold_db,
            ratio,
            knee_db,
            makeup_db,
            attack_ms,
            release_ms,
            mix,
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

/// Feed-forward (downward) compressor gain computer, in decibels.
///
/// Given a bin level `level_db`, returns the gain reduction (always `<= 0`) to
/// apply before makeup gain. `slope` is `1 / ratio`; `knee_db` is the width of
/// the quadratic soft knee (`0` collapses to a hard knee). The three regions
/// below the knee, inside the knee, and above the knee join continuously.
fn compress_gain_db(level_db: Sample, threshold_db: Sample, slope: Sample, knee_db: Sample) -> Sample {
    let over = level_db - threshold_db;
    let half_knee = knee_db * 0.5;
    if knee_db > 0.0 && over > -half_knee && over < half_knee {
        // Inside the knee: quadratic interpolation of the slope.
        let k = over + half_knee;
        (slope - 1.0) * k * k / (2.0 * knee_db)
    } else if over > 0.0 {
        // Above the knee: linear gain reduction.
        (slope - 1.0) * over
    } else {
        // Below the threshold (and knee): unity gain.
        0.0
    }
}

/// Frequency-domain spectral compressor node (weighted overlap-add `STFT`).
#[derive(Clone, Debug)]
pub struct SpectralCompressorNode {
    size: usize,
    hop: usize,
    half: usize,
    fifo_latency: usize,
    channels: usize,
    sample_rate: u32,
    // Precomputed, read-only tables.
    win: Vec<Sample>,
    fft: Fft,
    inv_window_sum: Sample,
    two_inv_window_sum: Sample,
    ola_norm: Sample,
    // Raw (sanitised) parameters, kept for getters.
    threshold_db: Sample,
    ratio: Sample,
    knee_db: Sample,
    makeup_db: Sample,
    attack_ms: Sample,
    release_ms: Sample,
    mix: Sample,
    // Derived compressor controls.
    slope: Sample,
    dry: Sample,
    attack_coeff: Sample,
    release_coeff: Sample,
    // Per-channel streaming state.
    in_fifo: Vec<Sample>,
    out_fifo: Vec<Sample>,
    out_accum: Vec<Sample>,
    comp_gain: Vec<Sample>,
    rover: usize,
    // Hot-path scratch (reused across channels and frames).
    re: Vec<Sample>,
    im: Vec<Sample>,
}

impl SpectralCompressorNode {
    /// Builds a spectral compressor for `channels` channels at `sample_rate`.
    ///
    /// `requested_size` is rounded up to a power of two no smaller than
    /// [`MIN_COMPRESSOR_FFT_SIZE`]; `channels` is clamped to at least one.
    /// `params` is sanitised into range.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::spectral_compressor::{
    ///     SpectralCompressorNode, SpectralCompressorParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node =
    ///     SpectralCompressorNode::new(48_000, 2, 1_024, SpectralCompressorParams::default());
    /// // Weighted overlap-add STFT: the reported latency is one full frame.
    /// assert_eq!(node.latency_frames(), 1_024);
    /// ```
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        requested_size: usize,
        params: SpectralCompressorParams,
    ) -> Self {
        let channels = channels.max(1);
        let fft = Fft::new(requested_size.max(MIN_COMPRESSOR_FFT_SIZE));
        let size = fft.size();
        let hop = (size / COMPRESSOR_OVERLAP_FACTOR).max(1);
        let half = size / 2;
        let fifo_latency = size - hop;

        let mut win = vec![0.0; size];
        let mut window_sum = 0.0f32;
        for (n, slot) in win.iter_mut().enumerate() {
            let w = 0.5 - 0.5 * ops::cos(TAU * n as Sample / size as Sample);
            *slot = w;
            window_sum += w;
        }

        // Constant-overlap-add denominator of the squared window at the hop.
        let base = half % hop;
        let mut overlap_sum = 0.0f32;
        let mut k = base;
        while k < size {
            overlap_sum += win[k] * win[k];
            k += hop;
        }
        let ola_norm = if overlap_sum > 0.0 {
            1.0 / overlap_sum
        } else {
            0.0
        };

        let inv_window_sum = if window_sum > 0.0 {
            1.0 / window_sum
        } else {
            0.0
        };

        let p = params.sanitised();

        Self {
            size,
            hop,
            half,
            fifo_latency,
            channels,
            sample_rate,
            win,
            fft,
            inv_window_sum,
            two_inv_window_sum: 2.0 * inv_window_sum,
            ola_norm,
            threshold_db: p.threshold_db,
            ratio: p.ratio,
            knee_db: p.knee_db,
            makeup_db: p.makeup_db,
            attack_ms: p.attack_ms,
            release_ms: p.release_ms,
            mix: p.mix,
            slope: 1.0 / p.ratio,
            dry: 1.0 - p.mix,
            attack_coeff: frame_coeff(p.attack_ms, sample_rate, hop),
            release_coeff: frame_coeff(p.release_ms, sample_rate, hop),
            in_fifo: vec![0.0; channels * size],
            out_fifo: vec![0.0; channels * size],
            out_accum: vec![0.0; channels * size],
            comp_gain: vec![1.0; channels * (half + 1)],
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

    /// Current threshold in `dBFS`.
    #[must_use]
    pub fn threshold_db(&self) -> Sample {
        self.threshold_db
    }

    /// Current compression ratio.
    #[must_use]
    pub fn ratio(&self) -> Sample {
        self.ratio
    }

    /// Current soft-knee width in decibels.
    #[must_use]
    pub fn knee_db(&self) -> Sample {
        self.knee_db
    }

    /// Current makeup gain in decibels.
    #[must_use]
    pub fn makeup_db(&self) -> Sample {
        self.makeup_db
    }

    /// Current attack time constant in milliseconds.
    #[must_use]
    pub fn attack_ms(&self) -> Sample {
        self.attack_ms
    }

    /// Current release time constant in milliseconds.
    #[must_use]
    pub fn release_ms(&self) -> Sample {
        self.release_ms
    }

    /// Current dry / wet mix.
    #[must_use]
    pub fn mix(&self) -> Sample {
        self.mix
    }

    /// Replaces every parameter at once, sanitising into range and recomputing
    /// the derived controls. The smoothed per-bin gains are preserved for a
    /// click-free transition.
    pub fn set_params(&mut self, sample_rate: u32, params: SpectralCompressorParams) {
        let p = params.sanitised();
        self.sample_rate = sample_rate;
        self.threshold_db = p.threshold_db;
        self.ratio = p.ratio;
        self.knee_db = p.knee_db;
        self.makeup_db = p.makeup_db;
        self.attack_ms = p.attack_ms;
        self.release_ms = p.release_ms;
        self.mix = p.mix;
        self.slope = 1.0 / p.ratio;
        self.dry = 1.0 - p.mix;
        self.attack_coeff = frame_coeff(p.attack_ms, sample_rate, self.hop);
        self.release_coeff = frame_coeff(p.release_ms, sample_rate, self.hop);
    }

    /// Sets the threshold in `dBFS`, replacing a non-finite value with the
    /// default and clamping into range.
    pub fn set_threshold_db(&mut self, threshold_db: Sample) {
        self.threshold_db = if threshold_db.is_finite() {
            threshold_db.clamp(MIN_COMPRESSOR_THRESHOLD_DB, MAX_COMPRESSOR_THRESHOLD_DB)
        } else {
            DEFAULT_COMPRESSOR_THRESHOLD_DB
        };
    }

    /// Sets the compression ratio, replacing a non-finite value with the
    /// default and clamping into `[1, MAX_COMPRESSOR_RATIO]`.
    pub fn set_ratio(&mut self, ratio: Sample) {
        self.ratio = if ratio.is_finite() {
            ratio.clamp(1.0, MAX_COMPRESSOR_RATIO)
        } else {
            DEFAULT_COMPRESSOR_RATIO
        };
        self.slope = 1.0 / self.ratio;
    }

    /// Sets the soft-knee width in decibels, replacing a non-finite value with
    /// the default and clamping into `[0, MAX_COMPRESSOR_KNEE_DB]`.
    pub fn set_knee_db(&mut self, knee_db: Sample) {
        self.knee_db = if knee_db.is_finite() {
            knee_db.clamp(0.0, MAX_COMPRESSOR_KNEE_DB)
        } else {
            DEFAULT_COMPRESSOR_KNEE_DB
        };
    }

    /// Sets the makeup gain in decibels, replacing a non-finite value with the
    /// default and clamping its magnitude to [`MAX_COMPRESSOR_MAKEUP_DB`].
    pub fn set_makeup_db(&mut self, makeup_db: Sample) {
        self.makeup_db = if makeup_db.is_finite() {
            makeup_db.clamp(-MAX_COMPRESSOR_MAKEUP_DB, MAX_COMPRESSOR_MAKEUP_DB)
        } else {
            DEFAULT_COMPRESSOR_MAKEUP_DB
        };
    }

    /// Sets the attack time constant in milliseconds, replacing a non-finite
    /// value with the default and clamping into range.
    pub fn set_attack_ms(&mut self, attack_ms: Sample) {
        self.attack_ms = if attack_ms.is_finite() {
            attack_ms.clamp(0.0, MAX_COMPRESSOR_TIME_MS)
        } else {
            DEFAULT_COMPRESSOR_ATTACK_MS
        };
        self.attack_coeff = frame_coeff(self.attack_ms, self.sample_rate, self.hop);
    }

    /// Sets the release time constant in milliseconds, replacing a non-finite
    /// value with the default and clamping into range.
    pub fn set_release_ms(&mut self, release_ms: Sample) {
        self.release_ms = if release_ms.is_finite() {
            release_ms.clamp(0.0, MAX_COMPRESSOR_TIME_MS)
        } else {
            DEFAULT_COMPRESSOR_RELEASE_MS
        };
        self.release_coeff = frame_coeff(self.release_ms, self.sample_rate, self.hop);
    }

    /// Sets the dry / wet mix, replacing a non-finite value with the default and
    /// clamping into `[0, 1]`.
    pub fn set_mix(&mut self, mix: Sample) {
        self.mix = if mix.is_finite() {
            mix.clamp(0.0, 1.0)
        } else {
            DEFAULT_COMPRESSOR_MIX
        };
        self.dry = 1.0 - self.mix;
    }
}

impl AudioNode for SpectralCompressorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        let size = self.size;
        let hop = self.hop;
        let half = self.half;
        let fifo_latency = self.fifo_latency;
        let inv_window_sum = self.inv_window_sum;
        let two_inv_window_sum = self.two_inv_window_sum;
        let ola_norm = self.ola_norm;
        let threshold_db = self.threshold_db;
        let slope = self.slope;
        let knee_db = self.knee_db;
        let makeup_db = self.makeup_db;
        let mix = self.mix;
        let dry = self.dry;
        let attack_coeff = self.attack_coeff;
        let release_coeff = self.release_coeff;
        let bin_count = half + 1;

        // Disjoint field borrows: read-only tables plus mutable state buffers.
        let win = &self.win;
        let fft = &self.fft;
        let re = &mut self.re;
        let im = &mut self.im;
        let in_fifo = &mut self.in_fifo;
        let out_fifo = &mut self.out_fifo;
        let out_accum = &mut self.out_accum;
        let comp_gain = &mut self.comp_gain;

        let mut rover = self.rover;

        for i in 0..frames {
            for ch in 0..channels {
                let fifo_base = ch * size;
                let x = input.channel(ch)[i];
                let x = if x.is_finite() { x } else { 0.0 };
                in_fifo[fifo_base + rover] = x;
                let y = out_fifo[fifo_base + (rover - fifo_latency)];
                let y = if y.is_finite() {
                    flush_denormal(y)
                } else {
                    0.0
                };
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

                fft.forward(re, im);

                // Per-bin spectral compression.
                for bin in 0..bin_count {
                    let re_b = re[bin];
                    let im_b = im[bin];
                    let mag = ops::sqrt(re_b * re_b + im_b * im_b);
                    let norm = if bin == 0 || bin == half {
                        inv_window_sum
                    } else {
                        two_inv_window_sum
                    };
                    let amplitude = mag * norm;
                    let level_db = linear_to_db(amplitude);
                    let reduction_db = compress_gain_db(level_db, threshold_db, slope, knee_db);
                    let target = db_to_linear(reduction_db + makeup_db);

                    let previous = comp_gain[gain_base + bin];
                    // Gain decreasing means more reduction: that is the attack.
                    let coeff = if target < previous {
                        attack_coeff
                    } else {
                        release_coeff
                    };
                    let gain = previous + coeff * (target - previous);
                    comp_gain[gain_base + bin] = gain;

                    // Dry / wet blend folds into one effective multiplier
                    // because the dry bin is the same bin scaled by one.
                    let applied = dry + mix * gain;
                    re[bin] *= applied;
                    im[bin] *= applied;
                    if bin > 0 && bin < half {
                        let mirror = size - bin;
                        re[mirror] *= applied;
                        im[mirror] *= applied;
                    }
                }

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
        for value in &mut self.comp_gain {
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
    fn run_mono(node: &mut SpectralCompressorNode, signal: &[Sample]) -> Vec<Sample> {
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
        let node = SpectralCompressorNode::new(SR, 1, 1_000, SpectralCompressorParams::default());
        assert_eq!(node.fft_size(), 1_024);
    }

    #[test]
    fn min_fft_size_enforced() {
        let node = SpectralCompressorNode::new(SR, 1, 8, SpectralCompressorParams::default());
        assert_eq!(node.fft_size(), MIN_COMPRESSOR_FFT_SIZE);
    }

    #[test]
    fn hop_is_quarter_of_size() {
        let node = SpectralCompressorNode::new(SR, 1, 1_024, SpectralCompressorParams::default());
        assert_eq!(node.hop(), 1_024 / COMPRESSOR_OVERLAP_FACTOR);
    }

    #[test]
    fn latency_is_one_full_frame() {
        // The weighted overlap-add pipeline delays the signal by a full frame.
        let node = SpectralCompressorNode::new(SR, 1, 1_024, SpectralCompressorParams::default());
        assert_eq!(node.latency_frames(), 1_024);
    }

    #[test]
    fn unity_ratio_reconstructs_input_delayed() {
        // ratio = 1 (and no makeup) leaves every bin at unity, so the weighted
        // overlap-add must reconstruct the input delayed by the latency.
        let params = SpectralCompressorParams {
            ratio: 1.0,
            knee_db: 0.0,
            makeup_db: 0.0,
            mix: 1.0,
            ..Default::default()
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
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
    fn dry_mix_reconstructs_input_delayed() {
        // mix = 0 bypasses every bin even under aggressive compression, so the
        // weighted overlap-add must reconstruct the delayed input.
        let params = SpectralCompressorParams {
            threshold_db: -60.0,
            ratio: 20.0,
            knee_db: 0.0,
            makeup_db: 24.0,
            mix: 0.0,
            ..Default::default()
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
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
    fn strong_compression_reduces_loud_tone() {
        // A loud tone well above a low threshold at a high ratio must lose
        // energy relative to the input.
        let params = SpectralCompressorParams {
            threshold_db: -40.0,
            ratio: 8.0,
            knee_db: 0.0,
            makeup_db: 0.0,
            attack_ms: 0.0,
            release_ms: 0.0,
            mix: 1.0,
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
        let input = sine(1_000.0, 0.5, 8_192);
        let output = run_mono(&mut node, &input);
        let input_rms = rms(&input[4_096..]);
        let output_rms = rms(&output[4_096..]);
        assert!(
            output_rms < input_rms * 0.5,
            "output_rms {output_rms} input_rms {input_rms}"
        );
    }

    #[test]
    fn ratio_one_is_transparent() {
        // ratio = 1 passes the signal through at essentially unity energy.
        let params = SpectralCompressorParams {
            threshold_db: -60.0,
            ratio: 1.0,
            knee_db: 0.0,
            makeup_db: 0.0,
            mix: 1.0,
            ..Default::default()
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
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
    fn makeup_gain_boosts_output() {
        // ratio = 1 means no compression, so +6 dB makeup should roughly double
        // the output amplitude.
        let params = SpectralCompressorParams {
            threshold_db: -60.0,
            ratio: 1.0,
            knee_db: 0.0,
            makeup_db: 6.0,
            attack_ms: 0.0,
            release_ms: 0.0,
            mix: 1.0,
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
        let input = sine(1_000.0, 0.25, 8_192);
        let output = run_mono(&mut node, &input);
        let input_rms = rms(&input[4_096..]);
        let output_rms = rms(&output[4_096..]);
        let ratio = output_rms / input_rms;
        assert!((ratio - 1.995).abs() < 0.1, "gain ratio {ratio}");
    }

    #[test]
    fn mix_blends_dry_and_wet() {
        // At mix = 0.5 with +6 dB makeup and ratio 1 the effective multiplier is
        // 0.5 + 0.5 * 1.995, strictly between the dry and fully wet levels.
        let params = SpectralCompressorParams {
            threshold_db: -60.0,
            ratio: 1.0,
            knee_db: 0.0,
            makeup_db: 6.0,
            attack_ms: 0.0,
            release_ms: 0.0,
            mix: 0.5,
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
        let input = sine(1_000.0, 0.25, 8_192);
        let output = run_mono(&mut node, &input);
        let input_rms = rms(&input[4_096..]);
        let output_rms = rms(&output[4_096..]);
        // Effective multiplier 0.5 + 0.5 * 1.995 = ~1.497.
        let ratio = output_rms / input_rms;
        assert!((ratio - 1.497).abs() < 0.1, "mixed gain ratio {ratio}");
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());
        let input = vec![0.0f32; 4_096];
        let output = run_mono(&mut node, &input);
        assert!(output.iter().all(|&y| y == 0.0), "non-silent output");
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut node = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());
        let mut input = sine(1_000.0, 0.5, 4_096);
        input[10] = Sample::NAN;
        input[20] = Sample::INFINITY;
        input[30] = Sample::NEG_INFINITY;
        let output = run_mono(&mut node, &input);
        assert!(output.iter().all(|&y| y.is_finite()), "non-finite output");
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());
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
    fn channels_compress_independently() {
        let params = SpectralCompressorParams {
            threshold_db: -40.0,
            ratio: 20.0,
            knee_db: 0.0,
            makeup_db: 0.0,
            attack_ms: 0.0,
            release_ms: 0.0,
            mix: 1.0,
        };
        let mut node = SpectralCompressorNode::new(SR, 2, 512, params);
        let len = 8_192;
        let loud = sine(1_000.0, 0.8, len);
        let quiet = sine(1_000.0, 1.0e-3, len);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.channel_mut(0).copy_from_slice(&loud);
        input.channel_mut(1).copy_from_slice(&quiet);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);

        // The loud channel is compressed down; the quiet channel is below
        // threshold and passes through essentially unchanged.
        let left_in = rms(&loud[4_096..]);
        let left_out = rms(&outputs[0].channel(0)[4_096..]);
        let right_in = rms(&quiet[4_096..]);
        let right_out = rms(&outputs[0].channel(1)[4_096..]);
        assert!(left_out < left_in * 0.5, "left_out {left_out}");
        assert!(
            (right_out - right_in).abs() < right_in * 0.2 + 1.0e-6,
            "right_out {right_out} right_in {right_in}"
        );
    }

    #[test]
    fn surplus_channels_pass_through() {
        // Node configured for one channel, fed a stereo buffer: the second
        // channel must be copied through unchanged.
        let mut node = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());
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
        // A reset node must reproduce a fresh node bit-for-bit on the same
        // input, proving every streaming buffer was cleared.
        let mut node = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());
        let warmup = sine(1_000.0, 0.5, 4_096);
        let _ = run_mono(&mut node, &warmup);
        node.reset();
        let probe = sine(1_000.0, 0.5, 4_096);
        let after_reset = run_mono(&mut node, &probe);

        let mut fresh = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());
        let baseline = run_mono(&mut fresh, &probe);

        let mut max_err = 0.0f32;
        for (a, b) in after_reset.iter().zip(baseline.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1.0e-6, "state leaked past reset: max_err {max_err}");
    }

    #[test]
    fn deterministic_repeat() {
        let params = SpectralCompressorParams::default();
        let input = sine(880.0, 0.6, 6_000);
        let mut a = SpectralCompressorNode::new(SR, 1, 512, params);
        let mut b = SpectralCompressorNode::new(SR, 1, 512, params);
        let out_a = run_mono(&mut a, &input);
        let out_b = run_mono(&mut b, &input);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn default_params_are_expected() {
        let p = SpectralCompressorParams::default();
        assert_eq!(p.threshold_db, DEFAULT_COMPRESSOR_THRESHOLD_DB);
        assert_eq!(p.ratio, DEFAULT_COMPRESSOR_RATIO);
        assert_eq!(p.knee_db, DEFAULT_COMPRESSOR_KNEE_DB);
        assert_eq!(p.makeup_db, DEFAULT_COMPRESSOR_MAKEUP_DB);
        assert_eq!(p.attack_ms, DEFAULT_COMPRESSOR_ATTACK_MS);
        assert_eq!(p.release_ms, DEFAULT_COMPRESSOR_RELEASE_MS);
        assert_eq!(p.mix, DEFAULT_COMPRESSOR_MIX);
    }

    #[test]
    fn params_sanitised_clamps_and_replaces() {
        let dirty = SpectralCompressorParams {
            threshold_db: Sample::NAN,
            ratio: 0.1,
            knee_db: -5.0,
            makeup_db: 1_000.0,
            attack_ms: Sample::INFINITY,
            release_ms: -3.0,
            mix: 5.0,
        };
        let s = dirty.sanitised();
        assert_eq!(s.threshold_db, DEFAULT_COMPRESSOR_THRESHOLD_DB);
        assert_eq!(s.ratio, 1.0);
        assert_eq!(s.knee_db, 0.0);
        assert_eq!(s.makeup_db, MAX_COMPRESSOR_MAKEUP_DB);
        assert_eq!(s.attack_ms, DEFAULT_COMPRESSOR_ATTACK_MS);
        assert_eq!(s.release_ms, 0.0);
        assert_eq!(s.mix, 1.0);
    }

    #[test]
    fn frame_coeff_edges() {
        assert_eq!(frame_coeff(0.0, SR, 256), 1.0);
        assert_eq!(frame_coeff(-5.0, SR, 256), 1.0);
        let c = frame_coeff(50.0, SR, 256);
        assert!((0.0..=1.0).contains(&c), "coeff {c}");
    }

    #[test]
    fn compress_gain_db_regions() {
        let slope = 0.5; // ratio 2.
        // Below threshold: unity (zero reduction).
        assert_eq!(compress_gain_db(-50.0, -20.0, slope, 0.0), 0.0);
        // Above knee (hard knee): linear reduction.
        let g = compress_gain_db(0.0, -20.0, slope, 0.0);
        assert!((g - (-10.0)).abs() < 1.0e-4, "hard-knee gain {g}");
        // Soft knee is continuous with the linear region at its upper edge.
        let knee = 6.0;
        let at_upper = compress_gain_db(-20.0 + knee * 0.5, -20.0, slope, knee);
        let linear = (slope - 1.0) * (knee * 0.5);
        assert!((at_upper - linear).abs() < 1.0e-4, "knee seam {at_upper}");
        // Reduction is never positive.
        assert!(compress_gain_db(10.0, -20.0, slope, knee) <= 0.0);
    }

    #[test]
    fn set_params_updates_controls() {
        let mut node = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());
        node.set_params(
            SR,
            SpectralCompressorParams {
                threshold_db: -12.0,
                ratio: 2.0,
                knee_db: 3.0,
                makeup_db: 2.0,
                attack_ms: 1.0,
                release_ms: 10.0,
                mix: 0.75,
            },
        );
        assert_eq!(node.threshold_db(), -12.0);
        assert_eq!(node.ratio(), 2.0);
        assert!((node.slope - 0.5).abs() < 1.0e-6);
        assert!((node.dry - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn setters_sanitise_non_finite_and_clamp() {
        let mut node = SpectralCompressorNode::new(SR, 1, 256, SpectralCompressorParams::default());

        node.set_threshold_db(Sample::NAN);
        assert_eq!(node.threshold_db(), DEFAULT_COMPRESSOR_THRESHOLD_DB);
        node.set_threshold_db(1_000.0);
        assert_eq!(node.threshold_db(), MAX_COMPRESSOR_THRESHOLD_DB);

        node.set_ratio(0.1);
        assert_eq!(node.ratio(), 1.0);
        node.set_ratio(Sample::INFINITY);
        assert_eq!(node.ratio(), DEFAULT_COMPRESSOR_RATIO);
        node.set_ratio(1_000.0);
        assert_eq!(node.ratio(), MAX_COMPRESSOR_RATIO);

        node.set_knee_db(-1.0);
        assert_eq!(node.knee_db(), 0.0);
        node.set_knee_db(Sample::NAN);
        assert_eq!(node.knee_db(), DEFAULT_COMPRESSOR_KNEE_DB);

        node.set_makeup_db(1_000.0);
        assert_eq!(node.makeup_db(), MAX_COMPRESSOR_MAKEUP_DB);
        node.set_makeup_db(Sample::NEG_INFINITY);
        assert_eq!(node.makeup_db(), DEFAULT_COMPRESSOR_MAKEUP_DB);

        node.set_attack_ms(-5.0);
        assert_eq!(node.attack_ms(), 0.0);
        node.set_release_ms(Sample::NAN);
        assert_eq!(node.release_ms(), DEFAULT_COMPRESSOR_RELEASE_MS);

        node.set_mix(5.0);
        assert_eq!(node.mix(), 1.0);
        node.set_mix(Sample::NAN);
        assert_eq!(node.mix(), DEFAULT_COMPRESSOR_MIX);
    }

    #[test]
    fn getters_report_configuration() {
        let node = SpectralCompressorNode::new(SR, 1, 1_024, SpectralCompressorParams::default());
        assert_eq!(node.fft_size(), 1_024);
        assert_eq!(node.hop(), 1_024 / COMPRESSOR_OVERLAP_FACTOR);
        assert_eq!(node.threshold_db(), DEFAULT_COMPRESSOR_THRESHOLD_DB);
        assert_eq!(node.ratio(), DEFAULT_COMPRESSOR_RATIO);
        assert_eq!(node.knee_db(), DEFAULT_COMPRESSOR_KNEE_DB);
        assert_eq!(node.makeup_db(), DEFAULT_COMPRESSOR_MAKEUP_DB);
        assert_eq!(node.attack_ms(), DEFAULT_COMPRESSOR_ATTACK_MS);
        assert_eq!(node.release_ms(), DEFAULT_COMPRESSOR_RELEASE_MS);
        assert_eq!(node.mix(), DEFAULT_COMPRESSOR_MIX);
    }

    #[test]
    fn new_sanitises_bad_params() {
        let node = SpectralCompressorNode::new(
            SR,
            1,
            256,
            SpectralCompressorParams {
                ratio: 0.0,
                mix: 9.0,
                ..Default::default()
            },
        );
        assert_eq!(node.ratio(), 1.0);
        assert_eq!(node.mix(), 1.0);
    }

    #[test]
    fn long_run_is_finite_and_bounded() {
        let params = SpectralCompressorParams {
            threshold_db: -30.0,
            ratio: 6.0,
            knee_db: 6.0,
            makeup_db: 6.0,
            mix: 1.0,
            ..Default::default()
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
        let mut input = sine(1_000.0, 0.9, 48_000);
        for (n, s) in input.iter_mut().enumerate() {
            *s += 0.3 * ops::sin(TAU * 3_000.0 * n as Sample / SR as Sample);
        }
        let output = run_mono(&mut node, &input);
        assert!(
            output.iter().all(|&y| y.is_finite() && y.abs() < 10.0),
            "non-finite or runaway output"
        );
    }

    #[test]
    fn high_threshold_is_near_identity() {
        // Threshold above the signal level: nothing is compressed, so with unity
        // makeup and full wet the output reconstructs the delayed input.
        let params = SpectralCompressorParams {
            threshold_db: MAX_COMPRESSOR_THRESHOLD_DB,
            ratio: 8.0,
            knee_db: 0.0,
            makeup_db: 0.0,
            mix: 1.0,
            ..Default::default()
        };
        let mut node = SpectralCompressorNode::new(SR, 1, 512, params);
        let latency = node.latency_frames() as usize;
        let input = sine(1_000.0, 0.3, 8_192);
        let output = run_mono(&mut node, &input);
        let start = latency + node.fft_size();
        let end = input.len() - 1;
        let mut max_err = 0.0f32;
        for n in start..end {
            max_err = max_err.max((output[n] - input[n - latency]).abs());
        }
        assert!(max_err < 5.0e-3, "max reconstruction error {max_err}");
    }
}
