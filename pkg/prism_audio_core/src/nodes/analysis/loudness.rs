//! `ITU-R` `BS.1770-4` / `EBU` `R128` loudness and true-peak metering.
//!
//! Modern game and broadcast engines normalize mixes to a target *loudness*
//! rather than a peak level so that dialogue, music, and effects sit at a
//! perceptually consistent level across content. This module implements the
//! measurement side of that workflow as a real-time-safe analysis tap:
//!
//! - **`K-weighting`** -- a two-stage filter (a high-shelf "head" pre-filter
//!   followed by an `RLB` high-pass) that approximates the frequency response
//!   of human loudness perception, from `ITU-R` `BS.1770`.
//! - **Momentary** loudness over a sliding 400 ms window and **short-term**
//!   loudness over a sliding 3 s window, both in `LUFS` (loudness units
//!   relative to full scale, equivalently `LKFS`).
//! - **Integrated** (programme) loudness with the two-stage gate from
//!   `BS.1770`: an absolute gate at -70 `LUFS` and a relative gate at -10 `LU`
//!   below the ungated mean.
//! - **Loudness range** (`LRA`) per `EBU` Tech 3342: the span between the 10th
//!   and 95th percentiles of the gated short-term distribution (relative gate
//!   -20 `LU`).
//! - **True-peak** level in `dBTP`, estimated with a 4x oversampling polyphase
//!   interpolator so inter-sample peaks that a sample-peak meter misses are
//!   caught, per `BS.1770` Annex 2.
//!
//! # Loudness formula
//!
//! For a measurement window the mean square of each `K-weighted` channel is
//! weighted by a channel gain `G_i` (unity for front channels, 1.41 for
//! surround per `BS.1770`), summed, and mapped to loudness via
//! `L = -0.691 + 10 * log10(sum_i G_i * z_i)`, where `z_i` is the per-channel
//! mean square. The -0.691 `LU` offset calibrates a 0 `dBFS` 997 Hz tone to
//! read 0 `LKFS`.
//!
//! # Real-time contract
//!
//! All state (filter memory, the sliding-window ring, the two gating
//! histograms, and the true-peak interpolator) is allocated once in
//! [`LoudnessMeter::new`]. The per-sample hot path ([`LoudnessMeter::feed_sample`]
//! / [`LoudnessMeter::advance_frame`] and [`LoudnessMeterNode::process`]) does
//! no allocation, takes no locks, and cannot panic. The reporting queries
//! ([`LoudnessMeter::integrated_lufs`] / [`LoudnessMeter::loudness_range_lu`])
//! each perform one bounded scan over the fixed histogram and are intended for
//! control-rate polling rather than per-sample use.
//!
//! Integrated loudness and `LRA` are accumulated into fixed-size histograms
//! (0.1 `LU` bins from -70 to +5 `LUFS`), so a measurement of any duration uses
//! bounded memory -- the standard approach for an unbounded-length meter.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The algorithm is
//! implemented from the publicly published standards `ITU-R` `BS.1770-4`
//! (loudness and true-peak), `EBU` `R128`, and `EBU` Tech 3341 / 3342 (meter
//! and loudness-range specifications). It is pure classic DSP with no AI/ML.
//!
//! # Relationship
//!
//! The `K-weighting` biquad coefficients are designed with the shared
//! [`BiquadCoeffs::design`](crate::nodes::biquad::BiquadCoeffs::design) RBJ
//! cookbook routine (reused, not re-implemented). Channel weights are derived
//! from [`ChannelLayout`](crate::buffer::ChannelLayout). This is a measurement
//! complement to the processing dynamics nodes
//! ([`LimiterNode`](crate::nodes::dynamics::LimiterNode) and friends): it
//! reports loudness/true-peak but does not alter the signal.

use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::PI;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, linear_to_db};
use crate::nodes::biquad::{BiquadCoeffs, BiquadKind};

/// The `LUFS` calibration offset (`-0.691` `LU`) from `ITU-R` `BS.1770`.
pub const LUFS_OFFSET: Sample = -0.691;

/// Number of 100 ms sub-blocks that make up a momentary (400 ms) window.
pub const MOMENTARY_SUBBLOCKS: usize = 4;

/// Number of 100 ms sub-blocks that make up a short-term (3 s) window.
pub const SHORT_TERM_SUBBLOCKS: usize = 30;

/// Absolute gate for integrated loudness and `LRA`, in `LUFS`.
pub const ABSOLUTE_GATE_LUFS: Sample = -70.0;

/// Relative gate for integrated loudness, in `LU` below the ungated mean.
pub const RELATIVE_GATE_LU: Sample = -10.0;

/// Relative gate for the loudness range, in `LU` below the ungated mean.
pub const LRA_RELATIVE_GATE_LU: Sample = -20.0;

/// Channel weight applied to surround channels per `BS.1770` (+1.5 dB).
pub const SURROUND_WEIGHT: Sample = 1.41;

/// Integer oversampling factor used by the true-peak estimator.
pub const OVERSAMPLE_FACTOR: usize = 4;

/// Taps per polyphase branch of the true-peak interpolator.
const TP_TAPS_PER_PHASE: usize = 12;

/// Lowest loudness bin edge of the gating histograms, in `LUFS`.
const HIST_MIN_LUFS: Sample = -70.0;

/// Width of each gating-histogram bin, in `LU`.
const HIST_STEP_LU: Sample = 0.1;

/// Number of histogram bins spanning `-70.0 ..= +5.0` `LUFS` at 0.1 `LU`.
const HIST_BINS: usize = 751;

/// Pre-filter (head/high-shelf) corner frequency in Hz (`BS.1770`).
const PREFILTER_FC: Sample = 1681.9745;

/// Pre-filter quality factor (`BS.1770`).
const PREFILTER_Q: Sample = 0.707_175_25;

/// Pre-filter shelf gain in dB (`BS.1770`).
const PREFILTER_GAIN_DB: Sample = 3.999_843_9;

/// `RLB` high-pass corner frequency in Hz (`BS.1770`).
const RLB_FC: Sample = 38.135_47;

/// `RLB` high-pass quality factor (`BS.1770`).
const RLB_Q: Sample = 0.500_327_05;

/// Returns the `BS.1770` loudness weight for channel `ch` of `layout`.
///
/// Front channels (L/R/C) weigh 1.0, surround channels weigh
/// [`SURROUND_WEIGHT`], and the LFE channel is excluded (weight 0). The
/// ambisonic first-order layout has no standardized loudness weighting, so only
/// its omnidirectional `W` component is measured (a documented pragmatic
/// fallback).
#[must_use]
fn channel_weight(layout: ChannelLayout, ch: usize) -> Sample {
    match layout {
        ChannelLayout::Mono | ChannelLayout::Stereo => 1.0,
        // L, R, Ls, Rs
        ChannelLayout::Quad => {
            if ch < 2 {
                1.0
            } else {
                SURROUND_WEIGHT
            }
        }
        // 5.1: L, R, C, LFE, Ls, Rs -- 7.1: L, R, C, LFE, Lss, Rss, Lrs, Rrs.
        // Both weight front L/R/C at 1.0, exclude the LFE, and lift surrounds.
        ChannelLayout::Surround5_1 | ChannelLayout::Surround7_1 => match ch {
            0..=2 => 1.0,
            3 => 0.0,
            _ => SURROUND_WEIGHT,
        },
        // Only the omnidirectional W component carries a defined weight.
        ChannelLayout::AmbisonicFoa => {
            if ch == 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Converts a block loudness in `LUFS` back to its linear mean-square energy.
#[inline]
#[must_use]
fn energy_of(loudness: Sample) -> Sample {
    ops::powf(10.0, (loudness - LUFS_OFFSET) * 0.1)
}

/// Maps a weighted sum-of-squares accumulated over `n_subblocks` of `step`
/// samples to a loudness value, or `None` for silence.
#[inline]
#[must_use]
fn loudness_from(weighted_ss: Sample, n_subblocks: usize, step: usize) -> Option<Sample> {
    if n_subblocks == 0 || step == 0 {
        return None;
    }
    let samples = (n_subblocks * step) as Sample;
    let z = weighted_ss / samples;
    if z > 0.0 && z.is_finite() {
        Some(LUFS_OFFSET + 10.0 * ops::log10(z))
    } else {
        None
    }
}

/// Two-stage `BS.1770` `K-weighting` filter with per-channel state.
///
/// Stage one is a high-shelf head pre-filter; stage two is the `RLB`
/// high-pass. Both are built from the shared RBJ cookbook designer so the
/// coefficients adapt to any sample rate.
#[derive(Debug, Clone)]
pub struct KWeighting {
    pre: BiquadCoeffs,
    rlb: BiquadCoeffs,
    /// Per-channel Direct Form I state for the pre-filter: `[x1, x2, y1, y2]`.
    pre_state: Vec<[Sample; 4]>,
    /// Per-channel Direct Form I state for the `RLB` high-pass.
    rlb_state: Vec<[Sample; 4]>,
}

impl KWeighting {
    /// Designs a `K-weighting` filter for `channels` channels at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        let ch = channels.max(1);
        Self {
            pre: BiquadCoeffs::design(
                BiquadKind::HighShelf,
                sample_rate,
                PREFILTER_FC,
                PREFILTER_Q,
                PREFILTER_GAIN_DB,
            ),
            rlb: BiquadCoeffs::design(BiquadKind::HighPass, sample_rate, RLB_FC, RLB_Q, 0.0),
            pre_state: alloc::vec![[0.0; 4]; ch],
            rlb_state: alloc::vec![[0.0; 4]; ch],
        }
    }

    /// Returns the number of channels this filter carries state for.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.pre_state.len()
    }

    /// Clears all per-channel filter memory.
    pub fn reset(&mut self) {
        for s in &mut self.pre_state {
            *s = [0.0; 4];
        }
        for s in &mut self.rlb_state {
            *s = [0.0; 4];
        }
    }

    /// One Direct Form I biquad step with denormal flushing.
    #[inline]
    fn df1(c: &BiquadCoeffs, st: &mut [Sample; 4], x0: Sample) -> Sample {
        let y0 = c.b0 * x0 + c.b1 * st[0] + c.b2 * st[1] - c.a1 * st[2] - c.a2 * st[3];
        let y0 = flush_denormal(y0);
        st[1] = st[0];
        st[0] = x0;
        st[3] = st[2];
        st[2] = y0;
        y0
    }

    /// Filters one sample of channel `ch`. Non-finite input is treated as
    /// silence so the recursive filter never poisons its state with `NaN`.
    #[inline]
    #[must_use]
    pub fn tick(&mut self, ch: usize, x: Sample) -> Sample {
        if ch >= self.pre_state.len() {
            return 0.0;
        }
        let x = if x.is_finite() { x } else { 0.0 };
        let a = Self::df1(&self.pre, &mut self.pre_state[ch], x);
        Self::df1(&self.rlb, &mut self.rlb_state[ch], a)
    }
}

/// A 4x oversampling true-peak estimator (one polyphase interpolator shared
/// across channels, with per-channel input history).
#[derive(Debug, Clone)]
pub struct TruePeakMeter {
    /// Polyphase branches: `phases[p][k]` is tap `k` of oversampling phase `p`.
    phases: Vec<[Sample; TP_TAPS_PER_PHASE]>,
    /// Per-channel history ring (`history[ch][0]` is the newest input sample).
    history: Vec<[Sample; TP_TAPS_PER_PHASE]>,
}

impl TruePeakMeter {
    /// Builds a true-peak estimator for `channels` channels.
    ///
    /// The interpolation prototype is a Hann-windowed sinc low-pass with its
    /// cutoff at the input Nyquist, split into [`OVERSAMPLE_FACTOR`] polyphase
    /// branches each normalized to unity DC gain.
    #[must_use]
    pub fn new(channels: usize) -> Self {
        let proto_len = OVERSAMPLE_FACTOR * TP_TAPS_PER_PHASE;
        let center = (proto_len - 1) as Sample * 0.5;
        let factor_f = OVERSAMPLE_FACTOR as Sample;
        let span = (proto_len - 1) as Sample;
        let mut phases = alloc::vec![[0.0; TP_TAPS_PER_PHASE]; OVERSAMPLE_FACTOR];
        for (p, branch) in phases.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (k, coeff) in branch.iter_mut().enumerate() {
                let i = p + k * OVERSAMPLE_FACTOR;
                let t = (i as Sample - center) / factor_f;
                let sinc = if t.abs() < 1.0e-6 {
                    1.0
                } else {
                    ops::sin(PI * t) / (PI * t)
                };
                let hann = 0.5 - 0.5 * ops::cos(2.0 * PI * (i as Sample) / span);
                let h = sinc * hann;
                *coeff = h;
                sum += h;
            }
            let norm = if sum.abs() > 1.0e-12 { 1.0 / sum } else { 1.0 };
            for coeff in branch.iter_mut() {
                *coeff *= norm;
            }
        }
        Self {
            phases,
            history: alloc::vec![[0.0; TP_TAPS_PER_PHASE]; channels.max(1)],
        }
    }

    /// Returns the number of channels this estimator tracks.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.history.len()
    }

    /// Clears all per-channel history.
    pub fn reset(&mut self) {
        for h in &mut self.history {
            *h = [0.0; TP_TAPS_PER_PHASE];
        }
    }

    /// Pushes one input sample for channel `ch` and returns the largest
    /// absolute value among the [`OVERSAMPLE_FACTOR`] reconstructed
    /// sub-samples (the local true-peak estimate).
    #[inline]
    #[must_use]
    pub fn process(&mut self, ch: usize, x: Sample) -> Sample {
        if ch >= self.history.len() {
            return 0.0;
        }
        let x = if x.is_finite() { x } else { 0.0 };
        {
            let hist = &mut self.history[ch];
            let mut k = TP_TAPS_PER_PHASE - 1;
            while k > 0 {
                hist[k] = hist[k - 1];
                k -= 1;
            }
            hist[0] = x;
        }
        let hist = &self.history[ch];
        let mut peak = 0.0;
        for branch in &self.phases {
            let mut acc = 0.0;
            for (coeff, sample) in branch.iter().zip(hist.iter()) {
                acc += coeff * sample;
            }
            let a = acc.abs();
            if a > peak {
                peak = a;
            }
        }
        peak
    }
}

/// A snapshot of every loudness statistic the meter tracks.
///
/// Fields that have no data yet (for example integrated loudness before any
/// gated block exists) report [`f32::NEG_INFINITY`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessMeasurement {
    /// Sliding 400 ms loudness in `LUFS`.
    pub momentary_lufs: Sample,
    /// Sliding 3 s loudness in `LUFS`.
    pub short_term_lufs: Sample,
    /// Gated programme loudness in `LUFS`.
    pub integrated_lufs: Sample,
    /// Loudness range in `LU` (0 when under-determined).
    pub loudness_range_lu: Sample,
    /// Maximum true-peak level in `dBTP`.
    pub true_peak_dbtp: Sample,
}

/// A real-time-safe `ITU-R` `BS.1770` / `EBU` `R128` loudness meter.
///
/// Drive it sample by sample with [`feed_sample`](Self::feed_sample) (once per
/// channel) followed by [`advance_frame`](Self::advance_frame) (once per
/// frame), then read the statistics with [`measurement`](Self::measurement).
/// [`LoudnessMeterNode`] wraps this as a pass-through graph node.
///
/// ```
/// use prism_audio_core::nodes::analysis::loudness::LoudnessMeter;
/// use prism_audio_core::buffer::ChannelLayout;
/// let sr = 48_000u32;
/// let mut meter = LoudnessMeter::new(sr, ChannelLayout::Stereo);
/// // One second of a -23 dBFS 1 kHz tone in both channels.
/// let amp = 0.070_794_57_f32; // 10^(-23/20)
/// for n in 0..sr {
///     let phase = core::f32::consts::TAU * 1_000.0 * (n as f32) / (sr as f32);
///     let x = amp * phase.sin();
///     meter.feed_sample(0, x);
///     meter.feed_sample(1, x);
///     meter.advance_frame();
/// }
/// let m = meter.measurement();
/// assert!(m.momentary_lufs.is_finite());
/// assert!((m.momentary_lufs - (-23.0)).abs() < 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct LoudnessMeter {
    channels: usize,
    weights: Vec<Sample>,
    step: usize,
    kweight: KWeighting,
    truepeak: TruePeakMeter,
    /// Ring of per-sub-block weighted sum-of-squares (newest written at
    /// `write_idx - 1`).
    ring: Vec<Sample>,
    write_idx: usize,
    filled: usize,
    finalized: u64,
    cur_ss: Sample,
    cur_count: usize,
    integ_hist: Vec<u32>,
    lra_hist: Vec<u32>,
    true_peak_max: Sample,
}

impl LoudnessMeter {
    /// Builds a meter for `layout` at `sample_rate`.
    ///
    /// The 100 ms sub-block length is `sample_rate / 10` samples (exact for the
    /// common broadcast rates; approximate for rates not divisible by ten).
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout) -> Self {
        let channels = layout.channel_count();
        let weights = (0..channels).map(|ch| channel_weight(layout, ch)).collect();
        let step = ((sample_rate.max(1) as usize) / 10).max(1);
        Self {
            channels,
            weights,
            step,
            kweight: KWeighting::new(sample_rate, channels),
            truepeak: TruePeakMeter::new(channels),
            ring: alloc::vec![0.0; SHORT_TERM_SUBBLOCKS],
            write_idx: 0,
            filled: 0,
            finalized: 0,
            cur_ss: 0.0,
            cur_count: 0,
            integ_hist: alloc::vec![0u32; HIST_BINS],
            lra_hist: alloc::vec![0u32; HIST_BINS],
            true_peak_max: 0.0,
        }
    }

    /// Returns the number of channels the meter measures.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Feeds one input sample for channel `ch` of the current frame.
    #[inline]
    pub fn feed_sample(&mut self, ch: usize, x: Sample) {
        if ch >= self.channels {
            return;
        }
        let k = self.kweight.tick(ch, x);
        self.cur_ss += self.weights[ch] * k * k;
        let tp = self.truepeak.process(ch, x);
        if tp > self.true_peak_max {
            self.true_peak_max = tp;
        }
    }

    /// Marks the end of a frame, finalizing a sub-block when 100 ms elapse.
    #[inline]
    pub fn advance_frame(&mut self) {
        self.cur_count += 1;
        if self.cur_count >= self.step {
            self.finalize_subblock();
        }
    }

    /// Clears every measurement and filter state.
    pub fn reset(&mut self) {
        self.kweight.reset();
        self.truepeak.reset();
        for v in &mut self.ring {
            *v = 0.0;
        }
        self.write_idx = 0;
        self.filled = 0;
        self.finalized = 0;
        self.cur_ss = 0.0;
        self.cur_count = 0;
        for b in &mut self.integ_hist {
            *b = 0;
        }
        for b in &mut self.lra_hist {
            *b = 0;
        }
        self.true_peak_max = 0.0;
    }

    /// Sums the weighted sum-of-squares of the most recent `n` sub-blocks,
    /// returning the sum and the number of blocks actually available.
    #[inline]
    fn sum_last(&self, n: usize) -> (Sample, usize) {
        let n = n.min(self.filled);
        let mut s = 0.0;
        for i in 0..n {
            let idx = (self.write_idx + SHORT_TERM_SUBBLOCKS - 1 - i) % SHORT_TERM_SUBBLOCKS;
            s += self.ring[idx];
        }
        (s, n)
    }

    /// Converts a loudness value to a histogram bin, rejecting values below the
    /// absolute gate (and `NaN`).
    #[inline]
    fn bin_index(loudness: Sample) -> Option<usize> {
        if loudness.is_nan() || loudness < HIST_MIN_LUFS {
            return None;
        }
        let scaled =
            ((loudness - HIST_MIN_LUFS) / HIST_STEP_LU).clamp(0.0, (HIST_BINS - 1) as Sample);
        Some(ops::round(scaled) as usize)
    }

    /// Finalizes the in-progress sub-block and feeds the overlapping gating
    /// windows.
    fn finalize_subblock(&mut self) {
        self.ring[self.write_idx] = self.cur_ss;
        self.write_idx = (self.write_idx + 1) % SHORT_TERM_SUBBLOCKS;
        if self.filled < SHORT_TERM_SUBBLOCKS {
            self.filled += 1;
        }
        self.finalized += 1;
        self.cur_ss = 0.0;
        self.cur_count = 0;

        if self.finalized >= MOMENTARY_SUBBLOCKS as u64 {
            let (s, n) = self.sum_last(MOMENTARY_SUBBLOCKS);
            if let Some(l) = loudness_from(s, n, self.step)
                && let Some(bin) = Self::bin_index(l)
            {
                self.integ_hist[bin] = self.integ_hist[bin].saturating_add(1);
            }
        }
        if self.finalized >= SHORT_TERM_SUBBLOCKS as u64 {
            let (s, n) = self.sum_last(SHORT_TERM_SUBBLOCKS);
            if let Some(l) = loudness_from(s, n, self.step)
                && let Some(bin) = Self::bin_index(l)
            {
                self.lra_hist[bin] = self.lra_hist[bin].saturating_add(1);
            }
        }
    }

    /// The sliding 400 ms momentary loudness in `LUFS`.
    #[must_use]
    pub fn momentary_lufs(&self) -> Sample {
        let (s, n) = self.sum_last(MOMENTARY_SUBBLOCKS);
        loudness_from(s, n, self.step).unwrap_or(f32::NEG_INFINITY)
    }

    /// The sliding 3 s short-term loudness in `LUFS`.
    #[must_use]
    pub fn short_term_lufs(&self) -> Sample {
        let (s, n) = self.sum_last(SHORT_TERM_SUBBLOCKS);
        loudness_from(s, n, self.step).unwrap_or(f32::NEG_INFINITY)
    }

    /// The gated integrated (programme) loudness in `LUFS`.
    #[must_use]
    pub fn integrated_lufs(&self) -> Sample {
        let (count_abs, energy_abs) = Self::histogram_energy(&self.integ_hist, HIST_MIN_LUFS);
        if count_abs == 0 {
            return f32::NEG_INFINITY;
        }
        let mean_abs = energy_abs / (count_abs as Sample);
        let relative_gate = LUFS_OFFSET + 10.0 * ops::log10(mean_abs) + RELATIVE_GATE_LU;
        let (count_rel, energy_rel) = Self::histogram_energy(&self.integ_hist, relative_gate);
        if count_rel == 0 {
            return f32::NEG_INFINITY;
        }
        LUFS_OFFSET + 10.0 * ops::log10(energy_rel / (count_rel as Sample))
    }

    /// The loudness range (`LRA`) in `LU`.
    #[must_use]
    pub fn loudness_range_lu(&self) -> Sample {
        let (count_abs, energy_abs) = Self::histogram_energy(&self.lra_hist, HIST_MIN_LUFS);
        if count_abs == 0 {
            return 0.0;
        }
        let mean_abs = energy_abs / (count_abs as Sample);
        let relative_gate = LUFS_OFFSET + 10.0 * ops::log10(mean_abs) + LRA_RELATIVE_GATE_LU;

        let mut total = 0u32;
        for (i, &cnt) in self.lra_hist.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            if Self::bin_loudness(i) >= relative_gate {
                total += cnt;
            }
        }
        if total == 0 {
            return 0.0;
        }

        let p10_target = 0.10 * (total as Sample);
        let p95_target = 0.95 * (total as Sample);
        let mut cum = 0u32;
        let mut p10 = HIST_MIN_LUFS;
        let mut p95 = HIST_MIN_LUFS;
        let mut got10 = false;
        let mut got95 = false;
        for (i, &cnt) in self.lra_hist.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let l = Self::bin_loudness(i);
            if l < relative_gate {
                continue;
            }
            cum += cnt;
            if !got10 && (cum as Sample) >= p10_target {
                p10 = l;
                got10 = true;
            }
            if !got95 && (cum as Sample) >= p95_target {
                p95 = l;
                got95 = true;
            }
        }
        (p95 - p10).max(0.0)
    }

    /// The maximum true-peak level seen so far in `dBTP`.
    #[must_use]
    pub fn true_peak_dbtp(&self) -> Sample {
        linear_to_db(self.true_peak_max)
    }

    /// A full snapshot of every statistic.
    #[must_use]
    pub fn measurement(&self) -> LoudnessMeasurement {
        LoudnessMeasurement {
            momentary_lufs: self.momentary_lufs(),
            short_term_lufs: self.short_term_lufs(),
            integrated_lufs: self.integrated_lufs(),
            loudness_range_lu: self.loudness_range_lu(),
            true_peak_dbtp: self.true_peak_dbtp(),
        }
    }

    /// The center loudness (in `LUFS`) of histogram bin `i`.
    #[inline]
    fn bin_loudness(i: usize) -> Sample {
        HIST_MIN_LUFS + (i as Sample) * HIST_STEP_LU
    }

    /// Returns the total count and summed linear energy of all histogram bins
    /// at or above the loudness `gate`.
    fn histogram_energy(hist: &[u32], gate: Sample) -> (u32, Sample) {
        let mut count = 0u32;
        let mut energy = 0.0;
        for (i, &cnt) in hist.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let l = Self::bin_loudness(i);
            if l < gate {
                continue;
            }
            count += cnt;
            energy += (cnt as Sample) * energy_of(l);
        }
        (count, energy)
    }
}

/// A pass-through graph node that measures the loudness of its single input.
///
/// The input is copied to the output unchanged (input port 0 -> output port 0)
/// while every sample is fed to an internal [`LoudnessMeter`]. Poll
/// [`measurement`](Self::measurement) off the audio thread for the current
/// statistics.
#[derive(Debug, Clone)]
pub struct LoudnessMeterNode {
    meter: LoudnessMeter,
}

impl LoudnessMeterNode {
    /// Builds a metering node for `layout` at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout) -> Self {
        Self {
            meter: LoudnessMeter::new(sample_rate, layout),
        }
    }

    /// Immutable access to the underlying meter (to read statistics).
    #[inline]
    #[must_use]
    pub fn meter(&self) -> &LoudnessMeter {
        &self.meter
    }

    /// A full snapshot of every statistic.
    #[inline]
    #[must_use]
    pub fn measurement(&self) -> LoudnessMeasurement {
        self.meter.measurement()
    }
}

impl AudioNode for LoudnessMeterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let frames = output.active_frames();

        // Pass the signal through unchanged.
        let copy_channels = out_channels.min(input.channels());
        for ch in 0..copy_channels {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }

        // Measure every frame across the metered channels.
        let metered = self.meter.channels().min(input.channels());
        for n in 0..frames {
            for ch in 0..metered {
                let x = input.channel(ch)[n];
                self.meter.feed_sample(ch, x);
            }
            self.meter.advance_frame();
        }
    }

    fn reset(&mut self) {
        self.meter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::AudioBuffer;
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    fn feed_tone(meter: &mut LoudnessMeter, freq: Sample, amp: Sample, seconds: Sample) {
        let channels = meter.channels();
        let total = (seconds * SR as Sample) as usize;
        for n in 0..total {
            let phase = core::f32::consts::TAU * freq * (n as Sample) / (SR as Sample);
            let x = amp * ops::sin(phase);
            for ch in 0..channels {
                meter.feed_sample(ch, x);
            }
            meter.advance_frame();
        }
    }

    #[test]
    fn silence_reads_negative_infinity() {
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        for _ in 0..SR {
            meter.feed_sample(0, 0.0);
            meter.feed_sample(1, 0.0);
            meter.advance_frame();
        }
        let m = meter.measurement();
        assert_eq!(m.momentary_lufs, f32::NEG_INFINITY);
        assert_eq!(m.integrated_lufs, f32::NEG_INFINITY);
        assert_eq!(m.true_peak_dbtp, f32::NEG_INFINITY);
    }

    #[test]
    fn reference_tone_reads_near_minus_23() {
        // EBU Tech 3341 calibration: a -23 dBFS 1 kHz stereo tone -> -23 LUFS.
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let amp = 0.070_794_57; // 10^(-23/20)
        feed_tone(&mut meter, 1_000.0, amp, 1.0);
        let m = meter.measurement();
        assert!((m.momentary_lufs - (-23.0)).abs() < 1.0, "{}", m.momentary_lufs);
        assert!((m.integrated_lufs - (-23.0)).abs() < 1.0, "{}", m.integrated_lufs);
    }

    #[test]
    fn louder_input_reads_higher() {
        let mut quiet = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let mut loud = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut quiet, 1_000.0, 0.05, 1.0);
        feed_tone(&mut loud, 1_000.0, 0.5, 1.0);
        assert!(loud.momentary_lufs() > quiet.momentary_lufs());
    }

    #[test]
    fn ten_db_increase_raises_about_ten_lu() {
        let mut a = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let mut b = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let amp = 0.1;
        feed_tone(&mut a, 1_000.0, amp, 1.0);
        // +10 dB is a linear factor of ~3.1623.
        feed_tone(&mut b, 1_000.0, amp * 3.162_277_6, 1.0);
        let delta = b.momentary_lufs() - a.momentary_lufs();
        assert!((delta - 10.0).abs() < 0.5, "{delta}");
    }

    #[test]
    fn low_frequency_is_attenuated_by_k_weighting() {
        let mut low = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let mut mid = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut low, 40.0, 0.5, 1.0);
        feed_tone(&mut mid, 1_000.0, 0.5, 1.0);
        // The RLB high-pass strongly attenuates 40 Hz, so it reads quieter.
        assert!(low.momentary_lufs() < mid.momentary_lufs() - 5.0);
    }

    #[test]
    fn high_frequency_shelf_boost() {
        // The head pre-filter boosts highs, so a high tone reads a touch louder
        // than a mid tone of equal amplitude.
        let mut high = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let mut mid = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut high, 10_000.0, 0.25, 1.0);
        feed_tone(&mut mid, 1_000.0, 0.25, 1.0);
        assert!(high.momentary_lufs() > mid.momentary_lufs());
    }

    #[test]
    fn absolute_gate_excludes_silence() {
        // Loud tone then a long silence: integrated should track the loud part,
        // not be dragged toward -inf by the gated-out silence.
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut meter, 1_000.0, 0.2, 2.0);
        for _ in 0..(SR * 3) {
            meter.feed_sample(0, 0.0);
            meter.feed_sample(1, 0.0);
            meter.advance_frame();
        }
        let integ = meter.integrated_lufs();
        assert!(integ.is_finite());
        assert!(integ > -40.0, "{integ}");
    }

    #[test]
    fn integrated_tracks_steady_tone() {
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut meter, 1_000.0, 0.1, 2.0);
        let integ = meter.integrated_lufs();
        let mom = meter.momentary_lufs();
        assert!((integ - mom).abs() < 1.0, "integ {integ} mom {mom}");
    }

    #[test]
    fn true_peak_of_full_scale_is_near_zero() {
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Mono);
        // A full-scale square-ish alternation drives the interpolator hard.
        for n in 0..2_000 {
            let x = if n % 2 == 0 { 1.0 } else { -1.0 };
            meter.feed_sample(0, x);
            meter.advance_frame();
        }
        let tp = meter.true_peak_dbtp();
        // Inter-sample peaks of an alternating signal exceed 0 dBTP.
        assert!(tp >= 0.0, "{tp}");
        assert!(tp < 6.0, "{tp}");
    }

    #[test]
    fn true_peak_detects_inter_sample_overshoot() {
        // A sampled sine whose peaks fall between samples reads a true-peak
        // above its sample peak.
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Mono);
        let sample_peak = {
            let mut p: Sample = 0.0;
            for n in 0..SR {
                let phase =
                    core::f32::consts::TAU * 11_000.0 * (n as Sample) / (SR as Sample) + 0.4;
                let x = 0.9 * ops::sin(phase);
                p = p.max(x.abs());
                meter.feed_sample(0, x);
                meter.advance_frame();
            }
            p
        };
        let tp_lin = crate::math::db_to_linear(meter.true_peak_dbtp());
        assert!(tp_lin > sample_peak, "tp {tp_lin} sample {sample_peak}");
    }

    #[test]
    fn surround_weighting_adds_energy() {
        // The same tone in 5.1 (with weighted surrounds) is louder than stereo.
        let mut stereo = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let mut surround = LoudnessMeter::new(SR, ChannelLayout::Surround5_1);
        feed_tone(&mut stereo, 1_000.0, 0.2, 1.0);
        feed_tone(&mut surround, 1_000.0, 0.2, 1.0);
        assert!(surround.momentary_lufs() > stereo.momentary_lufs());
    }

    #[test]
    fn lfe_channel_is_excluded() {
        assert_eq!(channel_weight(ChannelLayout::Surround5_1, 3), 0.0);
        assert_eq!(channel_weight(ChannelLayout::Surround7_1, 3), 0.0);
    }

    #[test]
    fn surround_channels_use_surround_weight() {
        assert_eq!(channel_weight(ChannelLayout::Surround5_1, 4), SURROUND_WEIGHT);
        assert_eq!(channel_weight(ChannelLayout::Quad, 2), SURROUND_WEIGHT);
        assert_eq!(channel_weight(ChannelLayout::Stereo, 0), 1.0);
    }

    #[test]
    fn loudness_range_is_nonnegative_and_sensible() {
        // Alternate loud and quiet 1 s sections -> a measurable range.
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        for _ in 0..4 {
            feed_tone(&mut meter, 1_000.0, 0.3, 1.0);
            feed_tone(&mut meter, 1_000.0, 0.03, 1.0);
        }
        let lra = meter.loudness_range_lu();
        assert!(lra >= 0.0);
        assert!(lra > 2.0, "{lra}");
    }

    #[test]
    fn loudness_range_is_small_for_steady_tone() {
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut meter, 1_000.0, 0.2, 6.0);
        let lra = meter.loudness_range_lu();
        assert!(lra < 1.0, "{lra}");
    }

    #[test]
    fn reset_clears_state() {
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut meter, 1_000.0, 0.5, 1.0);
        assert!(meter.momentary_lufs().is_finite());
        meter.reset();
        assert_eq!(meter.momentary_lufs(), f32::NEG_INFINITY);
        assert_eq!(meter.integrated_lufs(), f32::NEG_INFINITY);
        assert_eq!(meter.true_peak_dbtp(), f32::NEG_INFINITY);
    }

    #[test]
    fn determinism_two_runs_match() {
        let mut a = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        let mut b = LoudnessMeter::new(SR, ChannelLayout::Stereo);
        feed_tone(&mut a, 997.0, 0.3, 1.5);
        feed_tone(&mut b, 997.0, 0.3, 1.5);
        assert_eq!(a.momentary_lufs(), b.momentary_lufs());
        assert_eq!(a.integrated_lufs(), b.integrated_lufs());
        assert_eq!(a.true_peak_dbtp(), b.true_peak_dbtp());
    }

    #[test]
    fn non_finite_input_does_not_poison() {
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Mono);
        for n in 0..SR {
            let x = if n % 100 == 0 { f32::NAN } else { 0.3 };
            meter.feed_sample(0, x);
            meter.advance_frame();
        }
        let m = meter.measurement();
        assert!(m.momentary_lufs.is_finite());
        assert!(m.true_peak_dbtp.is_finite());
    }

    #[test]
    fn out_of_range_channel_is_ignored() {
        let mut meter = LoudnessMeter::new(SR, ChannelLayout::Mono);
        feed_tone(&mut meter, 1_000.0, 0.2, 0.5);
        let before = meter.momentary_lufs();
        meter.feed_sample(5, 1.0); // no such channel
        assert_eq!(meter.momentary_lufs(), before);
    }

    #[test]
    fn node_passes_signal_through_unchanged() {
        let mut node = LoudnessMeterNode::new(SR, ChannelLayout::Stereo);
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 4,
            playhead: 0,
        };
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        input.channel_mut(0).copy_from_slice(&[0.1, -0.2, 0.3, -0.4]);
        input.channel_mut(1).copy_from_slice(&[0.5, -0.6, 0.7, -0.8]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        assert_eq!(outputs[0].channel(0), &[0.1, -0.2, 0.3, -0.4]);
        assert_eq!(outputs[0].channel(1), &[0.5, -0.6, 0.7, -0.8]);
    }

    #[test]
    fn node_measures_what_flows_through() {
        let mut node = LoudnessMeterNode::new(SR, ChannelLayout::Stereo);
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 480,
            playhead: 0,
        };
        // Drive 1 s of a 1 kHz tone through the node in 480-frame blocks.
        let block = 480usize;
        let blocks = SR as usize / block;
        for b in 0..blocks {
            let mut buf: Vec<Sample> = Vec::with_capacity(block);
            for i in 0..block {
                let n = b * block + i;
                let phase = core::f32::consts::TAU * 1_000.0 * (n as Sample) / (SR as Sample);
                buf.push(0.2 * ops::sin(phase));
            }
            let mut input = AudioBuffer::new(ChannelLayout::Stereo, block);
            input.channel_mut(0).copy_from_slice(&buf);
            input.channel_mut(1).copy_from_slice(&buf);
            let output = AudioBuffer::new(ChannelLayout::Stereo, block);
            let inputs = [input];
            let mut outputs = [output];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }
        assert!(node.measurement().momentary_lufs.is_finite());
    }

    #[test]
    fn bin_index_rejects_below_gate() {
        assert_eq!(LoudnessMeter::bin_index(-80.0), None);
        assert_eq!(LoudnessMeter::bin_index(f32::NAN), None);
        assert_eq!(LoudnessMeter::bin_index(-70.0), Some(0));
        assert_eq!(LoudnessMeter::bin_index(1_000.0), Some(HIST_BINS - 1));
    }
}
