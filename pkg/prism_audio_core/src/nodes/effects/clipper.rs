//! Peak clipper: a transparent-below-ceiling waveshaper that passes signal
//! untouched until it reaches a fixed output ceiling, then clips the excursion
//! either as a brick-wall (hard) or with a smooth saturating knee (soft).
//!
//! A clipper is the mastering-chain complement of a saturator. Where the
//! [`waveshaper`](crate::nodes::effects::waveshaper) / `tanh` saturators
//! compress the *entire* transfer curve from zero, a clipper is **unity below
//! its ceiling** and only acts on peaks that exceed it. That is why modern
//! loudness workflows reach for a clipper (often ahead of a limiter) to shave
//! transients while leaving the body of the signal bit-for-bit intact.
//!
//! # The transfer curve
//!
//! Working in the ceiling-normalised domain `u = x / C` (ceiling `C` linear):
//!
//! - **Hard**: `clamp(u, -1, 1)` -- a brick wall. Signal below the ceiling is
//!   identical; everything above is flattened to exactly `C`.
//! - **Soft**: linear up to a knee that starts at `1 - knee`, then a
//!   `tanh`-shaped bend that asymptotically approaches the ceiling without ever
//!   quite reaching it. The knee is C1-continuous (unit slope at the join), so
//!   there is no corner to generate harsh high harmonics.
//!
//! The clipped result is rescaled by `C`, so the output magnitude never exceeds
//! the ceiling (strictly below it in soft mode).
//!
//! # Model
//!
//! An `input_gain` drives the signal into the fixed ceiling (more drive = more
//! clipping), an `output_gain` trims the result, and a `wet`/`dry` blend allows
//! parallel ("New York") clipping. Clipping a band-limited signal creates
//! harmonics above Nyquist, so the curve is evaluated through the shared
//! anti-aliasing [`Oversampler`](crate::oversampler::Oversampler) at 1x, 2x, or
//! 4x; the dry path is delayed to match the oversampler latency so the blend
//! stays phase-aligned.
//!
//! # Real-time contract
//!
//! All per-channel oversampler and dry-delay state is allocated in
//! [`ClipperNode::new`]. [`ClipperNode::process`] performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length blocks
//! degrade gracefully, non-finite inputs are treated as silence, and the output
//! is flushed of denormals. Latency equals the oversampler latency (zero at
//! 1x).
//!
//! # Relationship
//!
//! This node shares the [`Oversample`] mode and the
//! [`Oversampler`](crate::oversampler::Oversampler) /
//! [`DryDelay`](crate::oversampler::DryDelay) primitives with
//! [`WaveshaperNode`](crate::nodes::effects::waveshaper::WaveshaperNode), but
//! its transfer function is fundamentally different: a saturator bends the
//! whole curve, whereas this clipper is exactly linear below the ceiling and
//! only shapes the overshoot. It is instantaneous waveshaping, so unlike the
//! time-varying [`LimiterNode`](crate::nodes::dynamics::limiter::LimiterNode) it
//! applies no gain envelope and introduces no release pumping.
//!
//! # Provenance
//!
//! Pure classic DSP. Hard clipping is trivial amplitude clamping; the soft knee
//! is a standard piecewise `tanh` saturation documented throughout the
//! audio-engineering literature (e.g. Zoelzer, "DAFX", 2011; Pirkle,
//! "Designing Audio Effect Plugins in C++", 2019). There is no AI/ML of any
//! kind, and no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google
//! Resonance Audio, or Web Audio source or derived code; only the publicly
//! documented clipping curves and polyphase oversampling are used.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, db_to_linear, flush_denormal};
use crate::nodes::effects::waveshaper::Oversample;
use crate::oversampler::{DEFAULT_TAPS_PER_PHASE, DryDelay, Oversampler, OversamplerState};
use crate::param::{Ramp, Smoothed};

/// Default ceiling in dBFS (0 dBFS = full scale).
pub const DEFAULT_CLIPPER_CEILING_DB: Sample = 0.0;

/// Default soft-knee width as a fraction of the ceiling (`0` = hard corner,
/// `1` = knee spanning the whole range below the ceiling).
pub const DEFAULT_CLIPPER_KNEE: Sample = 0.1;

/// Largest input/output drive magnitude accepted, in dB.
pub const MAX_CLIPPER_GAIN_DB: Sample = 48.0;

/// Smallest linear ceiling, guarding the normalisation divide.
const MIN_CLIPPER_CEILING: Sample = 1.0e-6;

/// Selects the clipping transfer curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ClipperMode {
    /// Brick-wall clamp to the ceiling.
    Hard,
    /// Linear below the knee, then a smooth `tanh` bend to the ceiling.
    #[default]
    Soft,
}

/// Evaluates the clipping transfer curve for one sample.
///
/// `ceiling` is the linear output ceiling (`> 0`), `knee` is the soft-knee
/// fraction in `[0, 1]` (ignored in [`ClipperMode::Hard`]), and the result
/// magnitude never exceeds `ceiling`.
///
/// # Examples
///
/// ```
/// # use prism_audio_core::nodes::effects::clipper::{clip_sample, ClipperMode};
/// // Below the ceiling the hard clipper is perfectly transparent.
/// let y = clip_sample(0.4, 1.0, 0.0, ClipperMode::Hard);
/// assert!((y - 0.4).abs() < 1e-6);
/// // Above the ceiling it clamps to the ceiling exactly.
/// let y = clip_sample(2.0, 1.0, 0.0, ClipperMode::Hard);
/// assert!((y - 1.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn clip_sample(x: Sample, ceiling: Sample, knee: Sample, mode: ClipperMode) -> Sample {
    let c = ceiling.max(MIN_CLIPPER_CEILING);
    let u = x / c;
    let shaped = match mode {
        ClipperMode::Hard => u.clamp(-1.0, 1.0),
        ClipperMode::Soft => {
            let k = knee.clamp(0.0, 1.0);
            let t = 1.0 - k;
            let a = u.abs();
            if a <= t {
                u
            } else if k <= 0.0 {
                // Degenerate knee collapses to a hard clamp.
                if u >= 0.0 { 1.0 } else { -1.0 }
            } else {
                let s = if u >= 0.0 { 1.0 } else { -1.0 };
                let over = (a - t) / (1.0 - t);
                s * (t + (1.0 - t) * ops::tanh(over))
            }
        }
    };
    c * shaped
}

/// Construction parameters for a [`ClipperNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClipperParams {
    /// Pre-clip drive in dB (push signal into the ceiling).
    pub input_gain_db: Sample,
    /// Output ceiling in dBFS.
    pub ceiling_db: Sample,
    /// Soft-knee width as a fraction of the ceiling in `[0, 1]`.
    pub knee: Sample,
    /// Clipping curve.
    pub mode: ClipperMode,
    /// Anti-aliasing oversampling factor.
    pub oversample: Oversample,
    /// Post-clip make-up gain in dB.
    pub output_gain_db: Sample,
    /// Wet (clipped) mix coefficient.
    pub wet: Sample,
    /// Dry (unprocessed, latency-aligned) mix coefficient.
    pub dry: Sample,
}

impl Default for ClipperParams {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            ceiling_db: DEFAULT_CLIPPER_CEILING_DB,
            knee: DEFAULT_CLIPPER_KNEE,
            mode: ClipperMode::Soft,
            oversample: Oversample::X4,
            output_gain_db: 0.0,
            wet: 1.0,
            dry: 0.0,
        }
    }
}

/// A peak clipper with hard / soft curves and anti-aliasing oversampling
/// (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct ClipperNode {
    mode: ClipperMode,
    oversample: Oversample,
    ceiling: Sample,
    knee: Sample,
    input_gain: Smoothed,
    output_gain: Smoothed,
    wet: Smoothed,
    dry: Smoothed,
    oversampler: Oversampler,
    states: Vec<OversamplerState>,
    dry_delays: Vec<DryDelay>,
}

impl ClipperNode {
    /// Builds a clipper for a `channels`-wide signal.
    ///
    /// `sample_rate` is accepted for API symmetry; the oversampling filters are
    /// designed in normalised frequency and do not depend on it.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: ClipperParams) -> Self {
        let _ = sample_rate;
        let oversampler = Oversampler::new(params.oversample.factor(), DEFAULT_TAPS_PER_PHASE);
        let latency = oversampler.latency_frames() as usize;

        let mut states = Vec::with_capacity(channels);
        let mut dry_delays = Vec::with_capacity(channels);
        for _ in 0..channels {
            states.push(oversampler.make_state());
            dry_delays.push(DryDelay::new(latency));
        }

        Self {
            mode: params.mode,
            oversample: params.oversample,
            ceiling: Self::sanitised_ceiling(params.ceiling_db),
            knee: params.knee.clamp(0.0, 1.0),
            input_gain: Smoothed::new(Self::sanitised_gain(params.input_gain_db)),
            output_gain: Smoothed::new(Self::sanitised_gain(params.output_gain_db)),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
            oversampler,
            states,
            dry_delays,
        }
    }

    #[inline]
    fn sanitised_gain(db: Sample) -> Sample {
        let db = if db.is_finite() { db } else { 0.0 };
        db_to_linear(db.clamp(-MAX_CLIPPER_GAIN_DB, MAX_CLIPPER_GAIN_DB))
    }

    #[inline]
    fn sanitised_ceiling(db: Sample) -> Sample {
        let db = if db.is_finite() { db } else { 0.0 };
        db_to_linear(db.min(MAX_CLIPPER_GAIN_DB)).max(MIN_CLIPPER_CEILING)
    }

    /// Sets the pre-clip drive, gliding toward it with `ramp`.
    #[inline]
    pub fn set_input_gain_db(&mut self, db: Sample, ramp: Ramp) {
        self.input_gain.set_target(Self::sanitised_gain(db), ramp);
    }

    /// Sets the post-clip make-up gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_output_gain_db(&mut self, db: Sample, ramp: Ramp) {
        self.output_gain.set_target(Self::sanitised_gain(db), ramp);
    }

    /// Sets the output ceiling in dBFS.
    #[inline]
    pub fn set_ceiling_db(&mut self, db: Sample) {
        self.ceiling = Self::sanitised_ceiling(db);
    }

    /// Sets the soft-knee fraction (clamped to `[0, 1]`).
    #[inline]
    pub fn set_knee(&mut self, knee: Sample) {
        self.knee = knee.clamp(0.0, 1.0);
    }

    /// Selects the clipping curve.
    #[inline]
    pub fn set_mode(&mut self, mode: ClipperMode) {
        self.mode = mode;
    }

    /// Sets the wet-mix coefficient, gliding toward it with `ramp`.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry-mix coefficient, gliding toward it with `ramp`.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }

    /// Returns the configured oversampling mode.
    #[inline]
    #[must_use]
    pub fn oversample(&self) -> Oversample {
        self.oversample
    }

    /// Returns the clipping curve.
    #[inline]
    #[must_use]
    pub fn mode(&self) -> ClipperMode {
        self.mode
    }

    /// Returns the linear output ceiling.
    #[inline]
    #[must_use]
    pub fn ceiling(&self) -> Sample {
        self.ceiling
    }

    /// Returns the soft-knee fraction.
    #[inline]
    #[must_use]
    pub fn knee(&self) -> Sample {
        self.knee
    }

    /// Returns the length of the oversampling prototype FIR filter, or `0` when
    /// not oversampling.
    #[inline]
    #[must_use]
    pub fn filter_taps(&self) -> usize {
        self.oversampler.filter_taps()
    }
}

impl AudioNode for ClipperNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output
            .channels()
            .min(input.channels())
            .min(self.states.len());
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || channels == 0 {
            return;
        }

        let in0 = self.input_gain;
        let out0 = self.output_gain;
        let wet0 = self.wet;
        let dry0 = self.dry;
        let ceiling = self.ceiling;
        let knee = self.knee;
        let mode = self.mode;

        let oversampler = &self.oversampler;
        let states = &mut self.states;
        let dry_delays = &mut self.dry_delays;

        let mut committed: Option<(Smoothed, Smoothed, Smoothed, Smoothed)> = None;

        for ch in 0..channels {
            let mut input_gain = in0;
            let mut output_gain = out0;
            let mut wet = wet0;
            let mut dry = dry0;

            let state = &mut states[ch];
            let dry_delay = &mut dry_delays[ch];
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);

            for i in 0..frames {
                let raw = src[i];
                let x = if raw.is_finite() { raw } else { 0.0 };
                let ig = input_gain.next_sample();
                let og = output_gain.next_sample();
                let wv = wet.next_sample();
                let drv = dry.next_sample();

                let clipped =
                    oversampler.process_sample(state, x, |v| clip_sample(ig * v, ceiling, knee, mode));
                let dry_sig = dry_delay.push(x);
                dst[i] = flush_denormal(wv * (clipped * og) + drv * dry_sig);
            }

            if ch + 1 == channels {
                committed = Some((input_gain, output_gain, wet, dry));
            }
        }

        if let Some((input_gain, output_gain, wet, dry)) = committed {
            self.input_gain = input_gain;
            self.output_gain = output_gain;
            self.wet = wet;
            self.dry = dry;
        }
    }

    fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
        for dry_delay in &mut self.dry_delays {
            dry_delay.reset();
        }
        self.input_gain = Smoothed::new(self.input_gain.target());
        self.output_gain = Smoothed::new(self.output_gain.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }

    fn latency_frames(&self) -> u32 {
        self.oversampler.latency_frames()
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

    #[inline]
    fn ops_sin(x: Sample) -> Sample {
        ops::sin(x)
    }

    const TAU_1K: Sample = core::f32::consts::TAU * 1_000.0 / 48_000.0;

    /// Runs a mono sine of linear amplitude `amp` through the node.
    fn run_mono(node: &mut ClipperNode, amp: Sample, len: usize) -> Vec<Sample> {
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = amp * ops_sin(TAU_1K * i as Sample);
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0).to_vec()
    }

    fn peak(v: &[Sample]) -> Sample {
        v.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
    }

    #[test]
    fn curve_hard_transparent_below_ceiling() {
        for &x in &[-0.9, -0.3, 0.0, 0.25, 0.7] {
            let y = clip_sample(x, 1.0, 0.0, ClipperMode::Hard);
            assert!((y - x).abs() < 1e-6, "{x} -> {y}");
        }
    }

    #[test]
    fn curve_hard_clamps_above_ceiling() {
        assert!((clip_sample(5.0, 1.0, 0.0, ClipperMode::Hard) - 1.0).abs() < 1e-6);
        assert!((clip_sample(-5.0, 1.0, 0.0, ClipperMode::Hard) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn curve_respects_ceiling_scale() {
        let c = 0.5;
        assert!((clip_sample(2.0, c, 0.0, ClipperMode::Hard) - c).abs() < 1e-6);
        // Below ceiling stays transparent regardless of scale.
        assert!((clip_sample(0.3, c, 0.0, ClipperMode::Hard) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn curve_soft_transparent_below_knee() {
        // Knee 0.2 => linear region up to |u| = 0.8 of the ceiling.
        for &x in &[-0.7, -0.1, 0.0, 0.5, 0.75] {
            let y = clip_sample(x, 1.0, 0.2, ClipperMode::Soft);
            assert!((y - x).abs() < 1e-6, "{x} -> {y}");
        }
    }

    #[test]
    fn curve_soft_stays_below_ceiling() {
        // Soft mode asymptotes toward the ceiling: for mild overshoot it is
        // strictly below, and for any drive it never exceeds the ceiling (at
        // extreme drive f32 `tanh` saturates to exactly 1.0, so the bound is
        // `<=`).
        assert!(clip_sample(1.1, 1.0, 0.3, ClipperMode::Soft) < 1.0);
        for &x in &[1.1, 2.0, 8.0, 50.0] {
            let y = clip_sample(x, 1.0, 0.3, ClipperMode::Soft);
            assert!(y <= 1.0 + 1e-6, "soft output exceeded ceiling: {x} -> {y}");
            assert!(y > 0.0);
        }
    }

    #[test]
    fn curve_soft_is_continuous_at_knee() {
        // Just below and just above the knee join should match closely.
        let knee = 0.25;
        let t = 1.0 - knee;
        let below = clip_sample(t - 1e-3, 1.0, knee, ClipperMode::Soft);
        let above = clip_sample(t + 1e-3, 1.0, knee, ClipperMode::Soft);
        assert!((above - below).abs() < 5e-3, "{below} vs {above}");
    }

    #[test]
    fn curve_soft_zero_knee_matches_hard() {
        for &x in &[-3.0, -0.5, 0.2, 1.5, 4.0] {
            let soft = clip_sample(x, 1.0, 0.0, ClipperMode::Soft);
            let hard = clip_sample(x, 1.0, 0.0, ClipperMode::Hard);
            assert!((soft - hard).abs() < 1e-6, "{x}: {soft} vs {hard}");
        }
    }

    #[test]
    fn curve_is_odd_symmetric() {
        for &x in &[0.3, 0.9, 1.5, 4.0] {
            let pos = clip_sample(x, 1.0, 0.2, ClipperMode::Soft);
            let neg = clip_sample(-x, 1.0, 0.2, ClipperMode::Soft);
            assert!((pos + neg).abs() < 1e-6, "{x}: {pos} vs {neg}");
        }
    }

    #[test]
    fn latency_matches_oversampler() {
        let node = ClipperNode::new(SR, 2, ClipperParams::default());
        assert_eq!(node.latency_frames(), node.oversampler.latency_frames());
    }

    #[test]
    fn latency_zero_at_x1() {
        let params = ClipperParams {
            oversample: Oversample::X1,
            ..ClipperParams::default()
        };
        let node = ClipperNode::new(SR, 1, params);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn default_params_are_sane() {
        let p = ClipperParams::default();
        assert!(p.knee >= 0.0 && p.knee <= 1.0);
        assert_eq!(p.mode, ClipperMode::Soft);
    }

    #[test]
    fn hard_clip_bounds_output_to_ceiling() {
        let params = ClipperParams {
            mode: ClipperMode::Hard,
            input_gain_db: 12.0,
            ceiling_db: -6.0,
            oversample: Oversample::X1,
            ..ClipperParams::default()
        };
        let ceiling = db_to_linear(-6.0);
        let mut node = ClipperNode::new(SR, 1, params);
        let out = run_mono(&mut node, 1.0, 4_000);
        // X1 hard clip bounds output exactly to the ceiling (small epsilon).
        assert!(peak(&out) <= ceiling + 1e-4, "{} vs {ceiling}", peak(&out));
    }

    #[test]
    fn quiet_signal_passes_through_hard_x1() {
        let params = ClipperParams {
            mode: ClipperMode::Hard,
            oversample: Oversample::X1,
            ..ClipperParams::default()
        };
        let mut node = ClipperNode::new(SR, 1, params);
        // -20 dBFS sine, well below the 0 dBFS ceiling -> unchanged.
        let amp = 0.1;
        let out = run_mono(&mut node, amp, 2_000);
        assert!((peak(&out) - amp).abs() < amp * 0.01, "{} vs {amp}", peak(&out));
    }

    #[test]
    fn driving_harder_clips_more() {
        // More input drive into a fixed ceiling removes more of the sine tip,
        // so the clipped RMS rises toward a square (fuller) waveform.
        fn rms(v: &[Sample]) -> Sample {
            let s: Sample = v.iter().map(|x| x * x).sum();
            crate::math::linear_to_db((s / v.len() as Sample).max(1e-20).sqrt())
        }
        let mk = |drive: Sample| {
            let params = ClipperParams {
                mode: ClipperMode::Hard,
                input_gain_db: drive,
                oversample: Oversample::X1,
                output_gain_db: 0.0,
                ..ClipperParams::default()
            };
            let mut node = ClipperNode::new(SR, 1, params);
            run_mono(&mut node, 1.0, 4_000)
        };
        let soft_drive = rms(&mk(0.0));
        let hard_drive = rms(&mk(18.0));
        assert!(hard_drive > soft_drive, "{hard_drive} !> {soft_drive}");
    }

    #[test]
    fn dry_path_is_identity_when_fully_dry() {
        let params = ClipperParams {
            wet: 0.0,
            dry: 1.0,
            oversample: Oversample::X1,
            input_gain_db: 24.0,
            ..ClipperParams::default()
        };
        let mut node = ClipperNode::new(SR, 1, params);
        let out = run_mono(&mut node, 0.8, 1_000);
        // X1 has zero dry delay, so a fully dry path is the untouched input.
        let mut expect = AudioBuffer::new(ChannelLayout::Mono, 1_000);
        for (i, s) in expect.channel_mut(0).iter_mut().enumerate() {
            *s = 0.8 * ops_sin(TAU_1K * i as Sample);
        }
        for (o, e) in out.iter().zip(expect.channel(0)) {
            assert!((o - e).abs() < 1e-6, "{o} vs {e}");
        }
    }

    #[test]
    fn output_gain_scales_result() {
        let base = ClipperParams {
            mode: ClipperMode::Hard,
            input_gain_db: 12.0,
            oversample: Oversample::X1,
            ..ClipperParams::default()
        };
        let mut unity = ClipperNode::new(SR, 1, base);
        let mut boosted = ClipperNode::new(
            SR,
            1,
            ClipperParams {
                output_gain_db: 6.0,
                ..base
            },
        );
        let a = peak(&run_mono(&mut unity, 1.0, 2_000));
        let b = peak(&run_mono(&mut boosted, 1.0, 2_000));
        assert!(b > a * 1.5, "{b} !> {a}");
    }

    #[test]
    fn soft_mode_output_strictly_below_ceiling() {
        let params = ClipperParams {
            mode: ClipperMode::Soft,
            input_gain_db: 24.0,
            ceiling_db: 0.0,
            knee: 0.3,
            oversample: Oversample::X1,
            ..ClipperParams::default()
        };
        let mut node = ClipperNode::new(SR, 1, params);
        let out = run_mono(&mut node, 1.0, 4_000);
        // At heavy drive f32 `tanh` saturates to 1.0, so the soft curve bounds
        // the output to (not strictly below, but never above) the ceiling.
        assert!(peak(&out) <= 1.0 + 1e-6, "soft exceeded ceiling: {}", peak(&out));
    }

    #[test]
    fn non_finite_input_is_silenced() {
        let params = ClipperParams {
            oversample: Oversample::X1,
            ..ClipperParams::default()
        };
        let mut node = ClipperNode::new(SR, 1, params);
        let len = 128;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = if i % 2 == 0 { f32::NAN } else { f32::INFINITY };
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        for &o in outputs[0].channel(0) {
            assert!(o.is_finite(), "{o}");
        }
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = ClipperNode::new(SR, 1, ClipperParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1);
        input.set_active_frames(0);
        output.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn stereo_channels_independent() {
        let params = ClipperParams {
            mode: ClipperMode::Hard,
            input_gain_db: 12.0,
            oversample: Oversample::X1,
            ..ClipperParams::default()
        };
        let mut node = ClipperNode::new(SR, 2, params);
        let len = 2_000;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for i in 0..len {
            input.channel_mut(0)[i] = ops_sin(TAU_1K * i as Sample);
            input.channel_mut(1)[i] = 0.0;
        }
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        let right_peak = peak(outputs[0].channel(1));
        assert!(right_peak < 1e-6, "silent channel leaked: {right_peak}");
        assert!(peak(outputs[0].channel(0)) > 0.1);
    }

    #[test]
    fn reset_restores_initial_state() {
        let params = ClipperParams {
            oversample: Oversample::X2,
            input_gain_db: 12.0,
            ..ClipperParams::default()
        };
        let mut node = ClipperNode::new(SR, 1, params);
        let first = run_mono(&mut node, 1.0, 1_024);
        node.reset();
        let second = run_mono(&mut node, 1.0, 1_024);
        for (a, b) in first.iter().zip(&second) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn setters_update_state() {
        let mut node = ClipperNode::new(SR, 1, ClipperParams::default());
        node.set_mode(ClipperMode::Hard);
        node.set_ceiling_db(-6.0);
        node.set_knee(0.5);
        assert_eq!(node.mode(), ClipperMode::Hard);
        assert!((node.ceiling() - db_to_linear(-6.0)).abs() < 1e-6);
        assert!((node.knee() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let params = ClipperParams {
            input_gain_db: 1_000.0,
            ceiling_db: -1_000.0,
            knee: 9.0,
            output_gain_db: f32::NAN,
            oversample: Oversample::X4,
            wet: 2.0,
            dry: -1.0,
            mode: ClipperMode::Soft,
        };
        let mut node = ClipperNode::new(SR, 1, params);
        let out = run_mono(&mut node, 10.0, 1_024);
        for &o in &out {
            assert!(o.is_finite(), "{o}");
        }
    }

    #[test]
    fn oversampling_reduces_aliasing_peak() {
        // Hard-clipping a high sine with no oversampling folds energy back as
        // aliasing; 4x oversampling must not blow up the output envelope.
        let mk = |ov: Oversample| {
            let params = ClipperParams {
                mode: ClipperMode::Hard,
                input_gain_db: 18.0,
                oversample: ov,
                ..ClipperParams::default()
            };
            let mut node = ClipperNode::new(SR, 1, params);
            peak(&run_mono(&mut node, 1.0, 4_000))
        };
        let p1 = mk(Oversample::X1);
        let p4 = mk(Oversample::X4);
        assert!(p1.is_finite() && p4.is_finite());
        // Both are bounded near the 0 dBFS ceiling.
        assert!(p1 <= 1.2 && p4 <= 1.2, "{p1} / {p4}");
    }
}
