//! Inter-sample (true-peak) brick-wall limiter per `ITU-R` `BS.1770` /
//! `EBU` `R128` true-peak estimation.
//!
//! A conventional sample-peak limiter only guarantees that the *stored* sample
//! values stay under a ceiling. That is not enough for delivery: when a
//! digital-to-analog converter reconstructs the continuous waveform from those
//! samples, the reconstructed signal can swing *between* samples to a level
//! higher than any individual sample -- an **inter-sample peak**, or true peak.
//! A file that reads `0 dBFS` by its samples can easily reconstruct to
//! `+1.5 dBTP`, which then clips the converter or a downstream lossy encoder.
//!
//! The standard way to see those inter-sample peaks is the one `BS.1770`
//! prescribes: **oversample** the signal (at least `4x`) and measure the peak
//! of the denser grid, which approaches the true continuous peak. This node
//! reuses the crate's shared polyphase
//! [`Oversampler`](crate::oversampler::Oversampler) at `4x` to estimate the
//! true peak, then applies a **look-ahead** gain reduction with
//! program-dependent attack/release so the reconstructed inter-sample peak
//! never exceeds a configurable ceiling (`-1 dBTP` by default, per `EBU`
//! `R128`).
//!
//! # Model
//!
//! Because waveform reconstruction is a *linear* operation, scaling the host
//! samples by a gain `g` scales the reconstructed inter-sample peak by the same
//! `g`. So if the estimated true peak of the current neighbourhood is `p` and
//! the ceiling is `c`, driving the gain down to `g = c / p` makes the
//! reconstructed peak land exactly on the ceiling -- provided the gain is held
//! steady across the span the reconstruction filter touches. The two classic
//! ingredients that make that hold:
//!
//! - **Look-ahead.** The signal is delayed while the detector reads the
//!   un-delayed future, so the gain can begin falling *before* the peak reaches
//!   the output and has fully bottomed out by the time it arrives.
//! - **Windowed maximum.** The required reduction is held at the maximum over
//!   the whole look-ahead window, so the entire neighbourhood around a peak is
//!   attenuated by the same amount. This smears the reduction across the span
//!   the reconstruction kernel spreads an impulse over, which is what keeps the
//!   *inter-sample* peak -- not just the sample peak -- under the ceiling.
//!
//! The windowed maximum is smoothed by a decoupled attack/release
//! [`GainBallistics`](crate::nodes::dynamics::detector::GainBallistics): the
//! attack is derived from the look-ahead length so the reduction converges
//! within the window, and the release is program-dependent and configurable. A
//! final host-rate clamp at the ceiling is kept purely as a backstop; the
//! guarantee comes from the gain law, not the clamp.
//!
//! # Real-time contract
//!
//! All storage -- the per-channel oversampler histories, the look-ahead delay
//! rings, and the windowed-maximum monotonic deque -- is allocated once in
//! [`TruePeakLimiter::new`]. [`TruePeakLimiterNode::process`] performs no
//! allocation, takes no locks, cannot panic, and sanitises non-finite input to
//! silence so a stray `NaN`/`Inf` cannot poison the state. All transcendental
//! math routes through [`bevy_math::ops`] via the shared
//! [`math`](crate::math) helpers, so the result is bit-reproducible.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The true-peak
//! oversampling convention is implemented from the publicly published standards
//! `ITU-R` `BS.1770-4` and `EBU` `R128`; the look-ahead/windowed-maximum
//! limiter is a textbook dynamics construction. It is pure classic DSP with no
//! AI/ML.
//!
//! # Relationship
//!
//! This node reuses the shared polyphase
//! [`Oversampler`](crate::oversampler::Oversampler) for its inter-sample
//! estimation -- the very same primitive the non-linear
//! [`waveshaper`](crate::nodes::effects::waveshaper) uses for anti-aliasing --
//! rather than carrying a private resampler, and it reuses the dynamics-family
//! [`GainBallistics`](crate::nodes::dynamics::detector::GainBallistics) for its
//! attack/release envelope. It is the true-peak counterpart to the sample-peak
//! [`LimiterNode`](crate::nodes::dynamics::limiter::LimiterNode): where that one
//! guards the stored samples, this one guards the reconstructed waveform. It is
//! the natural final safety stage after the
//! [`LoudnessNormalizerNode`](crate::nodes::mastering::loudness_normalizer::LoudnessNormalizerNode).

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::buffer::AudioBuffer;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{db_to_linear, flush_denormal, linear_to_db, Sample};
use crate::nodes::dynamics::detector::GainBallistics;
use crate::oversampler::{Oversampler, OversamplerState, DEFAULT_TAPS_PER_PHASE};
use crate::param::Smoothed;

/// Default true-peak delivery ceiling (`-1 dBTP`, per `EBU` `R128`).
pub const DEFAULT_CEILING_DBTP: Sample = -1.0;

/// Default look-ahead window, in milliseconds.
pub const DEFAULT_LOOKAHEAD_MS: Sample = 2.0;

/// Default release time, in milliseconds.
pub const DEFAULT_RELEASE_MS: Sample = 100.0;

/// Default oversampling factor used for inter-sample peak estimation (`4x`,
/// the `BS.1770` minimum).
pub const DEFAULT_OVERSAMPLE_FACTOR: usize = 4;

/// Construction parameters for a [`TruePeakLimiter`] / [`TruePeakLimiterNode`].
///
/// All fields are plain values and the struct is [`Copy`]; it is a pure
/// description of the limiter policy, not processing state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TruePeakLimiterParams {
    /// Output true-peak ceiling, in `dBTP`; the reconstructed inter-sample peak
    /// never exceeds this.
    pub ceiling_dbtp: Sample,
    /// Look-ahead time in milliseconds. The reduction is front-loaded over this
    /// window so it bottoms out as a transient reaches the output.
    pub lookahead_ms: Sample,
    /// Release time in milliseconds (program-dependent recovery).
    pub release_ms: Sample,
    /// Input drive gain in dB applied before limiting.
    pub input_gain_db: Sample,
    /// Oversampling factor used for inter-sample peak estimation (clamped to
    /// `>= 1`; `BS.1770` recommends `>= 4`).
    pub oversample_factor: usize,
}

impl Default for TruePeakLimiterParams {
    fn default() -> Self {
        Self {
            ceiling_dbtp: DEFAULT_CEILING_DBTP,
            lookahead_ms: DEFAULT_LOOKAHEAD_MS,
            release_ms: DEFAULT_RELEASE_MS,
            input_gain_db: 0.0,
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
        }
    }
}

/// A sliding-window maximum over the last `window` pushed values.
///
/// Implemented as a monotonic (non-increasing) deque of `(index, value)` pairs,
/// so each [`push`](Self::push) is amortised `O(1)` and allocation-free: the
/// backing [`VecDeque`] is pre-sized to the window in [`new`](Self::new) and
/// never holds more than `window` entries.
#[derive(Debug, Clone)]
struct WindowMax {
    window: u64,
    deque: VecDeque<(u64, Sample)>,
    counter: u64,
}

impl WindowMax {
    fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window: window as u64,
            // One extra slot of headroom so a transient push never forces a
            // reallocation before the out-of-window prune runs.
            deque: VecDeque::with_capacity(window + 1),
            counter: 0,
        }
    }

    /// Pushes `value` and returns the maximum over the current window.
    #[inline]
    fn push(&mut self, value: Sample) -> Sample {
        let n = self.counter;
        self.counter = self.counter.wrapping_add(1);

        while let Some(&(_, back)) = self.deque.back() {
            if back <= value {
                self.deque.pop_back();
            } else {
                break;
            }
        }
        self.deque.push_back((n, value));

        // Window is the inclusive index range `[n - (window - 1), n]`.
        let oldest = n.saturating_sub(self.window - 1);
        while let Some(&(i, _)) = self.deque.front() {
            if i < oldest {
                self.deque.pop_front();
            } else {
                break;
            }
        }

        self.deque.front().map_or(0.0, |&(_, v)| v)
    }

    #[inline]
    fn reset(&mut self) {
        self.deque.clear();
        self.counter = 0;
    }
}

/// The true-peak limiter DSP core (format-agnostic, operates on
/// [`AudioBuffer`]s).
///
/// Construct one with [`new`](Self::new); drive it with
/// [`process`](Self::process). The [`TruePeakLimiterNode`] wrapper adapts it to
/// the [`AudioNode`] graph interface and adds nothing but plumbing.
#[derive(Debug, Clone)]
pub struct TruePeakLimiter {
    channels: usize,
    /// Shared polyphase oversampler used only for peak estimation.
    oversampler: Oversampler,
    /// Per-channel oversampler histories feeding the inter-sample estimate.
    det_states: Vec<OversamplerState>,
    /// Per-channel look-ahead delay rings for the signal path.
    delays: Vec<Vec<Sample>>,
    /// Total signal delay in host frames (`look-ahead + oversampler latency`).
    delay_len: usize,
    /// Shared write cursor into the delay rings.
    write_pos: usize,
    /// Sliding maximum of the required reduction over the look-ahead window.
    window_max: WindowMax,
    /// Attack/release smoothing of the reduction (in dB).
    ballistics: GainBallistics,
    /// Smoothed input drive gain (linear).
    input_gain: Smoothed,
    /// Ceiling in `dBTP` (for the gain computer).
    ceiling_db: Sample,
    /// Ceiling in linear amplitude (for the backstop clamp).
    ceiling_lin: Sample,
}

impl TruePeakLimiter {
    /// Builds a true-peak limiter for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: TruePeakLimiterParams) -> Self {
        let channels = channels.max(1);
        let sr = sample_rate.max(1);

        let factor = params.oversample_factor.max(1);
        let oversampler = Oversampler::new(factor, DEFAULT_TAPS_PER_PHASE);
        let os_latency = oversampler.latency_frames() as usize;

        // Look-ahead length (host frames). At least one frame so the detector
        // always leads the output.
        let lookahead = (bevy_math::ops::round(
            params.lookahead_ms.max(0.0) * (sr as Sample) * 0.001,
        ) as usize)
            .max(1);
        // The oversampler's own group delay eats into the usable look-ahead, so
        // the signal is delayed by both to keep the detector ahead of output.
        let delay_len = lookahead + os_latency;

        // The windowed maximum spans the look-ahead's worth of requirements, so
        // the reduction for the sample now leaving the delay is the maximum of
        // its own requirement and every requirement up to `lookahead` ahead.
        let window_max = WindowMax::new(lookahead + 1);

        // Attack converges well within the look-ahead window (tau ~ L/6, so the
        // residual is below ~0.3% by the time a peak reaches the output), while
        // release is program-dependent.
        let attack_ms = (params.lookahead_ms / 6.0).max(0.02);
        let ballistics = GainBallistics::new(attack_ms, params.release_ms.max(0.0), sr);

        let mut det_states = Vec::with_capacity(channels);
        let mut delays = Vec::with_capacity(channels);
        for _ in 0..channels {
            det_states.push(oversampler.make_state());
            let mut ring = Vec::with_capacity(delay_len);
            ring.resize(delay_len, 0.0);
            delays.push(ring);
        }

        Self {
            channels,
            oversampler,
            det_states,
            delays,
            delay_len,
            write_pos: 0,
            window_max,
            ballistics,
            input_gain: Smoothed::new(db_to_linear(params.input_gain_db)),
            ceiling_db: params.ceiling_dbtp,
            ceiling_lin: db_to_linear(params.ceiling_dbtp),
        }
    }

    /// Returns the number of channels this limiter was built for.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the processing latency in host frames (look-ahead plus the
    /// oversampler's group delay).
    #[inline]
    #[must_use]
    pub fn latency_frames(&self) -> u32 {
        self.delay_len as u32
    }

    /// Returns the current smoothed gain reduction in decibels (`>= 0`).
    #[inline]
    #[must_use]
    pub fn gain_reduction_db(&self) -> Sample {
        self.ballistics.current_db()
    }

    /// Sanitises one input sample: non-finite values become silence so a stray
    /// `NaN`/`Inf` can never enter the state or the output.
    #[inline]
    fn clean(x: Sample) -> Sample {
        if x.is_finite() {
            x
        } else {
            0.0
        }
    }

    /// Limits one block, writing the true-peak-limited result to `output`.
    ///
    /// A single broadband gain is applied to every channel per frame so the
    /// stereo/surround image stays coherent, driven by the loudest channel's
    /// estimated inter-sample peak.
    pub fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let channels = output
            .channels()
            .min(input.channels())
            .min(self.channels);
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || channels == 0 {
            return;
        }

        for f in 0..frames {
            let drive = self.input_gain.next_sample();

            // Estimate the inter-sample peak of the loudest driven channel.
            let mut peak = 0.0;
            for ch in 0..channels {
                let x = Self::clean(input.channel(ch)[f]) * drive;
                let mut ch_peak = x.abs();
                self.oversampler
                    .process_sample(&mut self.det_states[ch], x, |v| {
                        let a = v.abs();
                        if a.is_finite() && a > ch_peak {
                            ch_peak = a;
                        }
                        v
                    });
                if ch_peak > peak {
                    peak = ch_peak;
                }
            }

            // Reduction (dB, >= 0) needed to bring the estimated true peak down
            // to the ceiling.
            let req_db = if peak > self.ceiling_lin && peak > 0.0 {
                let r = linear_to_db(peak) - self.ceiling_db;
                if r.is_finite() {
                    r.max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let target = self.window_max.push(req_db);
            let reduction = self.ballistics.process(target);
            let gain = db_to_linear(-reduction);

            let w = self.write_pos;
            for ch in 0..channels {
                let x = Self::clean(input.channel(ch)[f]) * drive;
                let delayed = self.delays[ch][w];
                self.delays[ch][w] = flush_denormal(x);
                // Gain law provides the true-peak guarantee; the clamp is a
                // host-rate backstop for residual numeric overshoot.
                let y = (delayed * gain).clamp(-self.ceiling_lin, self.ceiling_lin);
                output.channel_mut(ch)[f] = flush_denormal(y);
            }

            self.write_pos = if w + 1 == self.delay_len { 0 } else { w + 1 };
        }
    }

    /// Clears all internal state back to silence.
    pub fn reset(&mut self) {
        for s in &mut self.det_states {
            s.reset();
        }
        for ring in &mut self.delays {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        self.write_pos = 0;
        self.window_max.reset();
        self.ballistics.reset();
        self.input_gain = Smoothed::new(self.input_gain.target());
    }
}

/// A true-peak brick-wall limiter graph node (input port 0 -> output port 0).
///
/// Thin [`AudioNode`] adapter over the [`TruePeakLimiter`] DSP core; it adds no
/// DSP of its own.
#[derive(Debug, Clone)]
pub struct TruePeakLimiterNode {
    engine: TruePeakLimiter,
}

impl TruePeakLimiterNode {
    /// Creates a true-peak limiter node for `channels` channels at
    /// `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: TruePeakLimiterParams) -> Self {
        Self {
            engine: TruePeakLimiter::new(sample_rate, channels, params),
        }
    }

    /// Borrows the underlying limiter engine.
    #[must_use]
    pub fn engine(&self) -> &TruePeakLimiter {
        &self.engine
    }

    /// Mutably borrows the underlying limiter engine.
    pub fn engine_mut(&mut self) -> &mut TruePeakLimiter {
        &mut self.engine
    }

    /// Returns the current smoothed gain reduction in decibels (`>= 0`).
    #[inline]
    #[must_use]
    pub fn gain_reduction_db(&self) -> Sample {
        self.engine.gain_reduction_db()
    }
}

impl AudioNode for TruePeakLimiterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        self.engine.process(input, output);
    }

    fn reset(&mut self) {
        self.engine.reset();
    }

    fn latency_frames(&self) -> u32 {
        self.engine.latency_frames()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use crate::oversampler::Oversampler;
    use bevy_math::ops;
    use core::f32::consts::PI;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    /// Estimates the true peak (linear) of a mono slice by oversampling `8x`
    /// with the shared oversampler -- a denser, independent grid than the
    /// limiter's own `4x` detector, to avoid grading against its own ruler.
    fn measure_true_peak(samples: &[Sample]) -> Sample {
        let os = Oversampler::new(8, DEFAULT_TAPS_PER_PHASE);
        let mut state = os.make_state();
        let mut peak = 0.0f32;
        for &x in samples {
            os.process_sample(&mut state, x, |v| {
                let a = v.abs();
                if a > peak {
                    peak = a;
                }
                v
            });
        }
        peak
    }

    fn run_mono(node: &mut TruePeakLimiterNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = mono(frames);
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    #[test]
    fn half_nyquist_tone_stays_under_ceiling() {
        // A full-scale tone at half Nyquist is the classic worst case for
        // inter-sample peaks: its samples read ~0 dBFS but it reconstructs well
        // above that.
        let params = TruePeakLimiterParams::default();
        let ceiling = db_to_linear(params.ceiling_dbtp);
        let mut node = TruePeakLimiterNode::new(SR, 1, params);
        let n = 16_000;
        let mut input = mono(n);
        // Half Nyquist = SR / 4; phase offset maximises inter-sample excursion.
        let f = SR as Sample / 4.0;
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let t = i as Sample / SR as Sample;
            *s = ops::sin(2.0 * PI * f * t + PI * 0.25);
        }
        let out = run_mono(&mut node, &input);
        // Measure the settled region (skip the latency/attack transient).
        let lat = node.engine().latency_frames() as usize;
        let settled: Vec<Sample> = out.channel(0)[(lat + 1_000)..].to_vec();
        let tp = measure_true_peak(&settled);
        assert!(
            tp <= ceiling * 1.02,
            "true peak {} exceeded ceiling {}",
            linear_to_db(tp),
            params.ceiling_dbtp
        );
    }

    #[test]
    fn full_scale_impulses_stay_under_ceiling() {
        let params = TruePeakLimiterParams::default();
        let ceiling = db_to_linear(params.ceiling_dbtp);
        let mut node = TruePeakLimiterNode::new(SR, 1, params);
        let n = 8_000;
        let mut input = mono(n);
        // A train of alternating-sign full-scale impulses: maximal inter-sample
        // content between the spikes.
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 73 == 0 {
                if (i / 73) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
        }
        let out = run_mono(&mut node, &input);
        let tp = measure_true_peak(out.channel(0));
        assert!(
            tp <= ceiling * 1.05,
            "impulse true peak {} exceeded ceiling {}",
            linear_to_db(tp),
            params.ceiling_dbtp
        );
    }

    #[test]
    fn quiet_signal_is_passed_through() {
        // Well below the ceiling: no reduction, so the output equals the
        // latency-delayed input.
        let params = TruePeakLimiterParams::default();
        let mut node = TruePeakLimiterNode::new(SR, 1, params);
        let n = 4_096;
        let mut input = mono(n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = 0.1 * ops::sin(0.05 * i as Sample);
        }
        let out = run_mono(&mut node, &input);
        let lat = node.engine().latency_frames() as usize;
        for i in 0..(n - lat) {
            let o = out.channel(0)[i + lat];
            let x = input.channel(0)[i];
            assert!((o - x).abs() < 1e-3, "quiet drift at {i}: {o} vs {x}");
        }
        assert!(node.gain_reduction_db().abs() < 1e-6);
    }

    #[test]
    fn deterministic_across_runs() {
        let params = TruePeakLimiterParams::default();
        let n = 2_048;
        let mut input = mono(n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(0.21 * i as Sample) + 0.5 * ops::sin(0.37 * i as Sample);
        }
        let mut a = TruePeakLimiterNode::new(SR, 1, params);
        let mut b = TruePeakLimiterNode::new(SR, 1, params);
        let out_a = run_mono(&mut a, &input);
        let out_b = run_mono(&mut b, &input);
        for i in 0..n {
            assert_eq!(out_a.channel(0)[i], out_b.channel(0)[i], "nondeterministic at {i}");
        }
    }

    #[test]
    fn non_finite_input_produces_no_nan() {
        let params = TruePeakLimiterParams::default();
        let mut node = TruePeakLimiterNode::new(SR, 1, params);
        let n = 512;
        let mut input = mono(n);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = match i % 4 {
                0 => Sample::NAN,
                1 => Sample::INFINITY,
                2 => Sample::NEG_INFINITY,
                _ => 0.5,
            };
        }
        let out = run_mono(&mut node, &input);
        for &y in out.channel(0) {
            assert!(y.is_finite(), "non-finite output: {y}");
        }
    }

    #[test]
    fn reset_clears_state() {
        let params = TruePeakLimiterParams::default();
        let mut node = TruePeakLimiterNode::new(SR, 1, params);
        let n = 1_024;
        let mut input = mono(n);
        for s in input.channel_mut(0).iter_mut() {
            *s = 1.0;
        }
        let _ = run_mono(&mut node, &input);
        node.reset();
        assert!(node.gain_reduction_db().abs() < 1e-6);
        // After reset a quiet signal passes through unchanged (past latency).
        let mut quiet = mono(n);
        for (i, s) in quiet.channel_mut(0).iter_mut().enumerate() {
            *s = 0.05 * ops::sin(0.03 * i as Sample);
        }
        let out = run_mono(&mut node, &quiet);
        let lat = node.engine().latency_frames() as usize;
        for i in 0..(n - lat) {
            let o = out.channel(0)[i + lat];
            let x = quiet.channel(0)[i];
            assert!((o - x).abs() < 1e-3, "post-reset drift at {i}: {o} vs {x}");
        }
    }

    #[test]
    fn reports_lookahead_latency() {
        let node = TruePeakLimiterNode::new(SR, 2, TruePeakLimiterParams::default());
        let os = Oversampler::new(DEFAULT_OVERSAMPLE_FACTOR, DEFAULT_TAPS_PER_PHASE);
        let lookahead =
            (ops::round(DEFAULT_LOOKAHEAD_MS * SR as Sample * 0.001) as usize).max(1);
        let expected = (lookahead + os.latency_frames() as usize) as u32;
        assert_eq!(node.latency_frames(), expected);
    }

    #[test]
    fn stereo_gain_is_coherent() {
        // The same gain is applied to both channels, so a channel that is
        // identical to another stays identical at the output.
        let params = TruePeakLimiterParams::default();
        let mut node = TruePeakLimiterNode::new(SR, 2, params);
        let n = 4_000;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, n);
        for i in 0..n {
            let t = i as Sample / SR as Sample;
            let v = ops::sin(2.0 * PI * (SR as Sample / 4.0) * t);
            input.channel_mut(0)[i] = v;
            input.channel_mut(1)[i] = v;
        }
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, n);
        let inputs = [input];
        let mut outputs = [out.clone()];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(n), &mut io);
        out = outputs.into_iter().next().unwrap();
        for i in 0..n {
            assert_eq!(out.channel(0)[i], out.channel(1)[i], "image not coherent at {i}");
        }
    }

    #[test]
    fn window_max_tracks_sliding_maximum() {
        // Spot-check the helper against a brute-force sliding maximum.
        let mut wm = WindowMax::new(3);
        let data = [1.0f32, 3.0, 2.0, 5.0, 4.0, 0.0, 0.0, 6.0];
        let mut got = Vec::new();
        for &v in &data {
            got.push(wm.push(v));
        }
        let expected = [1.0f32, 3.0, 3.0, 5.0, 5.0, 5.0, 4.0, 6.0];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_eq!(g, e);
        }
    }
}
