//! Reverse delay: a creative echo that captures the input in fixed-length
//! segments and plays each captured segment back in reverse, so transients
//! sound as if they swell backwards into themselves.
//!
//! The node keeps two record buffers per channel. While one buffer is filled
//! forwards with the incoming audio, the other (just-filled) buffer is read
//! back-to-front, so every segment of input is time-reversed on output. When
//! the recording buffer fills, the two roles swap and the next segment is
//! captured. A raised-cosine (Hann) window is applied to the reversed playback
//! so each reversed grain fades in and out, keeping the segment boundaries
//! click-free. An optional feedback term folds the reversed output back into
//! the record buffer for cascading reverse repeats, and a dry / wet control
//! blends the effect against the untouched input.
//!
//! # Relationship
//!
//! Unlike the forward echoes of
//! [`DelayNode`](crate::nodes::effects::delay::DelayNode),
//! [`MultiTapDelayNode`](crate::nodes::effects::multi_tap_delay::MultiTapDelayNode),
//! and [`PingPongDelayNode`](crate::nodes::effects::ping_pong_delay::PingPongDelayNode)
//! -- which all read a delay line in the same time direction as it was written
//! -- this node reverses the time axis within each captured segment, a
//! transformation no fractional-delay read can produce. It shares none of the
//! per-sample delay-line interpolation of those nodes; instead it swaps whole
//! segment buffers and reads them backwards.
//!
//! # Real-time contract
//!
//! Both record buffers for every channel are allocated once at construction,
//! sized for the maximum segment length and channel count.
//! [`ReverseDelayNode::process`] performs no allocation, locking, or panic on
//! the hot path; non-finite input samples are treated as silence, and every
//! value written into a record buffer is denormal-flushed so the feedback path
//! cannot stall on denormals. The dry path passes straight through, so the node
//! reports no compensating latency ([`ReverseDelayNode::latency_frames`]
//! returns zero); the reversed repeat naturally trails the dry signal by up to
//! one segment.
//!
//! # Provenance
//!
//! Reverse (backwards) delay by segment capture and reversed read-out, the
//! Hann window used to taper reversed grains, and a scalar feedback term in the
//! record path are standard, publicly documented classic DSP techniques (for
//! example Zoelzer's DAFX delay / time-segment processing material and the
//! long-established tape-reversal and reverse-reverb studio techniques). This
//! is pure classic DSP with no AI or ML. This module contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or
//! Web Audio source or derived code**; only the widely documented segment
//! reversal and Hann-window formulas are used.

use alloc::{vec, vec::Vec};
use core::f32::consts::TAU;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Largest segment length, in milliseconds, the node can capture and reverse.
pub const MAX_REVERSE_SEGMENT_MS: Sample = 2_000.0;

/// Smallest segment length, in milliseconds, kept above a couple of frames so
/// the Hann window always spans at least a few samples.
pub const MIN_REVERSE_SEGMENT_MS: Sample = 10.0;

/// Largest stable feedback coefficient, kept below unity so cascaded reverse
/// repeats decay instead of building without bound.
pub const MAX_REVERSE_FEEDBACK: Sample = 0.95;

/// Default segment length in milliseconds.
pub const DEFAULT_REVERSE_SEGMENT_MS: Sample = 300.0;

/// Default feedback coefficient.
pub const DEFAULT_REVERSE_FEEDBACK: Sample = 0.3;

/// Default dry / wet mix.
pub const DEFAULT_REVERSE_MIX: Sample = 0.5;

/// Parameters shared by every channel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReverseDelayParams {
    /// Segment length in milliseconds: how much input is captured before it is
    /// played back reversed. Clamped to
    /// `[MIN_REVERSE_SEGMENT_MS, MAX_REVERSE_SEGMENT_MS]`.
    pub segment_ms: Sample,
    /// Feedback coefficient in `[0, MAX_REVERSE_FEEDBACK]`: how much of the
    /// reversed output is folded back into the record buffer.
    pub feedback: Sample,
    /// Dry / wet mix in `[0, 1]`: `0` is the dry input, `1` is the reversed
    /// echo only.
    pub mix: Sample,
}

impl Default for ReverseDelayParams {
    fn default() -> Self {
        Self {
            segment_ms: DEFAULT_REVERSE_SEGMENT_MS,
            feedback: DEFAULT_REVERSE_FEEDBACK,
            mix: DEFAULT_REVERSE_MIX,
        }
    }
}

impl ReverseDelayParams {
    /// Returns the parameters with every field clamped into range and any
    /// non-finite field replaced by its default.
    #[must_use]
    fn sanitised(self) -> Self {
        let d = Self::default();
        let clamp = |v: Sample, lo: Sample, hi: Sample, fallback: Sample| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        Self {
            segment_ms: clamp(
                self.segment_ms,
                MIN_REVERSE_SEGMENT_MS,
                MAX_REVERSE_SEGMENT_MS,
                d.segment_ms,
            ),
            feedback: clamp(self.feedback, 0.0, MAX_REVERSE_FEEDBACK, d.feedback),
            mix: clamp(self.mix, 0.0, 1.0, d.mix),
        }
    }
}

/// A segment-reversing delay (reverse echo).
#[derive(Clone, Debug)]
pub struct ReverseDelayNode {
    sample_rate: u32,
    channels: usize,
    /// Largest addressable segment length in frames.
    max_seg_frames: usize,
    /// Current segment length in frames (`>= 2`).
    seg_frames: usize,
    /// Two record buffers per channel; roles swap each segment.
    bufs: Vec<[Vec<Sample>; 2]>,
    /// Index (`0` or `1`) of the buffer currently being recorded into.
    recording: usize,
    /// Shared write / read position inside the current segment.
    pos: usize,
    feedback: Sample,
    wet: Sample,
    dry: Sample,
}

impl ReverseDelayNode {
    /// Builds a reverse delay for `channels` channels at `sample_rate` Hz.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::reverse_delay::{
    ///     ReverseDelayNode, ReverseDelayParams,
    /// };
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = ReverseDelayNode::new(48_000, 2, ReverseDelayParams::default());
    /// // The dry path is immediate, so no compensating latency is reported.
    /// assert_eq!(node.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: ReverseDelayParams) -> Self {
        let sample_rate = sample_rate.max(1);
        let channels = channels.max(1);
        let max_seg_frames =
            (ops::floor(MAX_REVERSE_SEGMENT_MS * sample_rate as Sample / 1_000.0) as usize + 1)
                .max(2);
        let bufs = (0..channels)
            .map(|_| [vec![0.0; max_seg_frames], vec![0.0; max_seg_frames]])
            .collect();
        let mut node = Self {
            sample_rate,
            channels,
            max_seg_frames,
            seg_frames: 2,
            bufs,
            recording: 0,
            pos: 0,
            feedback: 0.0,
            wet: 0.0,
            dry: 1.0,
        };
        node.set_params(params);
        node
    }

    /// Number of channels this node was built for.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Converts a segment time in milliseconds to a clamped length in frames.
    fn ms_to_frames(&self, ms: Sample) -> usize {
        let frames = ops::floor(ms * self.sample_rate as Sample / 1_000.0) as usize;
        frames.clamp(2, self.max_seg_frames)
    }

    /// Replaces the segment length, feedback, and mix. Record buffers are
    /// preserved; the segment position is clamped so a shrinking segment stays
    /// in bounds.
    pub fn set_params(&mut self, params: ReverseDelayParams) {
        let params = params.sanitised();
        self.seg_frames = self.ms_to_frames(params.segment_ms);
        if self.pos >= self.seg_frames {
            self.pos = 0;
        }
        self.feedback = params.feedback;
        self.wet = params.mix;
        self.dry = 1.0 - params.mix;
    }

    /// Raised-cosine (Hann) taper for position `pos` inside the segment.
    fn window(&self, pos: usize) -> Sample {
        let denom = (self.seg_frames - 1) as Sample;
        0.5 - 0.5 * ops::cos(TAU * pos as Sample / denom)
    }
}

impl AudioNode for ReverseDelayNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        let seg = self.seg_frames;
        let feedback = self.feedback;
        let wet = self.wet;
        let dry = self.dry;

        for f in 0..frames {
            let pos = self.pos;
            let w = self.window(pos);
            let play = 1 - self.recording;
            let rev = seg - 1 - pos;
            let rec = self.recording;

            for ch in 0..channels {
                let x_raw = input.channel(ch)[f];
                let x = if x_raw.is_finite() { x_raw } else { 0.0 };
                let played = self.bufs[ch][play][rev] * w;
                let write_val = x + feedback * played;
                self.bufs[ch][rec][pos] = flush_denormal(write_val);
                output.channel_mut(ch)[f] = dry * x + wet * played;
            }

            if pos + 1 == seg {
                self.pos = 0;
                self.recording = play;
            } else {
                self.pos = pos + 1;
            }
        }

        // Pass surplus channels (beyond the processed set) through untouched.
        for ch in channels..available {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }
    }

    fn reset(&mut self) {
        for pair in &mut self.bufs {
            for buf in pair.iter_mut() {
                for s in buf.iter_mut() {
                    *s = 0.0;
                }
            }
        }
        self.recording = 0;
        self.pos = 0;
    }

    fn latency_frames(&self) -> u32 {
        0
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

    fn run_mono(node: &mut ReverseDelayNode, signal: &[Sample]) -> Vec<Sample> {
        let len = signal.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        input.set_active_frames(len);
        input.channel_mut(0)[..len].copy_from_slice(signal);
        let output = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        let inputs = [input];
        let mut outputs = [output];
        outputs[0].set_active_frames(len);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0)[..len].to_vec()
    }

    fn run_stereo(
        node: &mut ReverseDelayNode,
        left: &[Sample],
        right: &[Sample],
    ) -> (Vec<Sample>, Vec<Sample>) {
        let len = left.len().min(right.len());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, len.max(1));
        input.set_active_frames(len);
        input.channel_mut(0)[..len].copy_from_slice(&left[..len]);
        input.channel_mut(1)[..len].copy_from_slice(&right[..len]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, len.max(1));
        let inputs = [input];
        let mut outputs = [output];
        outputs[0].set_active_frames(len);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        (
            outputs[0].channel(0)[..len].to_vec(),
            outputs[0].channel(1)[..len].to_vec(),
        )
    }

    fn rms(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: Sample = samples.iter().map(|&x| x * x).sum();
        ops::sqrt(sum / samples.len() as Sample)
    }

    #[test]
    fn latency_is_zero() {
        let node = ReverseDelayNode::new(SR, 2, ReverseDelayParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn channels_getter_reports_build_width() {
        let node = ReverseDelayNode::new(SR, 2, ReverseDelayParams::default());
        assert_eq!(node.channels(), 2);
    }

    #[test]
    fn default_params_are_in_range() {
        let p = ReverseDelayParams::default();
        assert!(p.segment_ms >= MIN_REVERSE_SEGMENT_MS && p.segment_ms <= MAX_REVERSE_SEGMENT_MS);
        assert!(p.feedback >= 0.0 && p.feedback <= MAX_REVERSE_FEEDBACK);
        assert!(p.mix >= 0.0 && p.mix <= 1.0);
    }

    #[test]
    fn window_is_zero_at_endpoints_and_peaks_mid() {
        let node = ReverseDelayNode::new(
            SR,
            1,
            ReverseDelayParams {
                segment_ms: 10.0,
                ..ReverseDelayParams::default()
            },
        );
        let seg = node.seg_frames;
        assert!(node.window(0).abs() < 1e-6);
        assert!(node.window(seg - 1).abs() < 1e-6);
        assert!(node.window((seg - 1) / 2) > 0.99);
    }

    #[test]
    fn min_segment_is_clamped() {
        let node = ReverseDelayNode::new(
            SR,
            1,
            ReverseDelayParams {
                segment_ms: 0.0,
                ..ReverseDelayParams::default()
            },
        );
        // 10 ms at 48 kHz is 480 frames; a sub-minimum request clamps up.
        assert_eq!(node.seg_frames, 480);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = ReverseDelayNode::new(SR, 2, ReverseDelayParams::default());
        let (l, r) = run_stereo(&mut node, &vec![0.0; 2_048], &vec![0.0; 2_048]);
        assert!(l.iter().all(|&x| x == 0.0));
        assert!(r.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn dry_passthrough_when_mix_zero() {
        let params = ReverseDelayParams {
            mix: 0.0,
            feedback: 0.6,
            ..ReverseDelayParams::default()
        };
        let mut node = ReverseDelayNode::new(SR, 2, params);
        let mut left = vec![0.0; 1_024];
        for (i, s) in left.iter_mut().enumerate() {
            *s = (i as Sample * 0.01).sin();
        }
        let right = vec![0.0; 1_024];
        let (lo, ro) = run_stereo(&mut node, &left, &right);
        for (o, i) in lo.iter().zip(left.iter()) {
            assert!((o - i).abs() < 1e-6, "left dry mismatch: {o} vs {i}");
        }
        for (o, i) in ro.iter().zip(right.iter()) {
            assert!((o - i).abs() < 1e-6, "right dry mismatch: {o} vs {i}");
        }
    }

    #[test]
    fn reverses_segment_energy_to_the_opposite_end() {
        // 10 ms at 48 kHz is a 480-frame segment.
        let params = ReverseDelayParams {
            segment_ms: 10.0,
            feedback: 0.0,
            mix: 1.0,
        };
        let mut node = ReverseDelayNode::new(SR, 1, params);
        let seg = node.seg_frames;
        // Energy only in the first quarter of segment 0.
        let mut sig = vec![0.0; 2 * seg];
        for s in sig.iter_mut().take(seg / 4) {
            *s = 1.0;
        }
        let out = run_mono(&mut node, &sig);
        // Playback (segment 1) reverses it: energy must land in the last
        // quarter, not the first.
        let seg1 = &out[seg..2 * seg];
        let first_q = rms(&seg1[..seg / 4]);
        let last_q = rms(&seg1[3 * seg / 4..]);
        assert!(
            last_q > first_q * 4.0,
            "reversed energy must sit at the opposite end: first={first_q} last={last_q}"
        );
    }

    #[test]
    fn feedback_decays_and_stays_finite() {
        let params = ReverseDelayParams {
            segment_ms: 10.0,
            feedback: MAX_REVERSE_FEEDBACK,
            mix: 1.0,
        };
        let mut node = ReverseDelayNode::new(SR, 1, params);
        let seg = node.seg_frames;
        let mut sig = vec![0.0; 12 * seg];
        for s in sig.iter_mut().take(seg) {
            *s = 1.0;
        }
        let out = run_mono(&mut node, &sig);
        assert!(out.iter().all(|x| x.is_finite()));
        let early = rms(&out[seg..3 * seg]);
        let late = rms(&out[9 * seg..]);
        assert!(late < early, "reverse tail must decay: early={early} late={late}");
    }

    #[test]
    fn non_finite_input_becomes_finite() {
        let mut node = ReverseDelayNode::new(SR, 2, ReverseDelayParams::default());
        let mut left = vec![0.0; 2_048];
        left[0] = Sample::NAN;
        left[1] = Sample::INFINITY;
        left[2] = Sample::NEG_INFINITY;
        let (lo, ro) = run_stereo(&mut node, &left, &vec![0.0; 2_048]);
        assert!(lo.iter().all(|x| x.is_finite()));
        assert!(ro.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = ReverseDelayNode::new(SR, 2, ReverseDelayParams::default());
        let input = AudioBuffer::new(ChannelLayout::Stereo, 1);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 1);
        let mut inputs = [input];
        inputs[0].set_active_frames(0);
        let mut outputs = [output.clone()];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        output = outputs[0].clone();
        assert_eq!(output.active_frames(), 0);
    }

    #[test]
    fn extreme_params_do_not_panic() {
        let params = ReverseDelayParams {
            segment_ms: 100_000.0,
            feedback: 10.0,
            mix: 2.0,
        };
        let mut node = ReverseDelayNode::new(SR, 2, params);
        let (lo, ro) = run_stereo(&mut node, &vec![0.5; 4_096], &vec![-0.5; 4_096]);
        assert!(lo.iter().all(|x| x.is_finite()));
        assert!(ro.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn non_finite_params_fall_back_to_default() {
        let params = ReverseDelayParams {
            segment_ms: Sample::NAN,
            feedback: Sample::INFINITY,
            mix: Sample::NEG_INFINITY,
        };
        let sane = params.sanitised();
        let d = ReverseDelayParams::default();
        assert_eq!(sane.segment_ms, d.segment_ms);
        assert_eq!(sane.feedback, d.feedback);
        assert_eq!(sane.mix, d.mix);
    }

    #[test]
    fn surplus_channels_pass_through() {
        let mut node = ReverseDelayNode::new(SR, 2, ReverseDelayParams::default());
        let len = 512;
        let mut input = AudioBuffer::new(ChannelLayout::Quad, len);
        for ch in 0..4 {
            for f in 0..len {
                input.channel_mut(ch)[f] = (ch as Sample + 1.0) * 0.1 * f as Sample;
            }
        }
        let output = AudioBuffer::new(ChannelLayout::Quad, len);
        let inputs = [input.clone()];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        for ch in 2..4 {
            for f in 0..len {
                assert_eq!(outputs[0].channel(ch)[f], inputs[0].channel(ch)[f]);
            }
        }
    }

    #[test]
    fn reset_restores_fresh_state() {
        let params = ReverseDelayParams {
            segment_ms: 10.0,
            feedback: 0.5,
            mix: 1.0,
        };
        let mut node = ReverseDelayNode::new(SR, 2, params);
        let seg = node.seg_frames;
        let mut sig = vec![0.0; 3 * seg];
        for s in sig.iter_mut().take(seg) {
            *s = 0.7;
        }
        let first = run_mono_pair(&mut node, &sig);
        node.reset();
        let second = run_mono_pair(&mut node, &sig);
        let mut max_err = 0.0_f32;
        for (a, b) in first.iter().zip(second.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-6, "reset must reproduce the fresh response: {max_err}");
    }

    // Helper: run a mono signal through a stereo node on channel 0 only.
    fn run_mono_pair(node: &mut ReverseDelayNode, signal: &[Sample]) -> Vec<Sample> {
        let right = vec![0.0; signal.len()];
        run_stereo(node, signal, &right).0
    }

    #[test]
    fn set_params_changes_segment_length() {
        let mut node = ReverseDelayNode::new(
            SR,
            1,
            ReverseDelayParams {
                segment_ms: 10.0,
                feedback: 0.0,
                mix: 1.0,
            },
        );
        assert_eq!(node.seg_frames, 480);
        node.set_params(ReverseDelayParams {
            segment_ms: 20.0,
            feedback: 0.0,
            mix: 1.0,
        });
        assert_eq!(node.seg_frames, 960);
        // Still produces only finite output after the change.
        let out = run_mono(&mut node, &vec![0.3; 4_096]);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mono_produces_reversed_energy() {
        let params = ReverseDelayParams {
            segment_ms: 10.0,
            feedback: 0.2,
            mix: 1.0,
        };
        let mut node = ReverseDelayNode::new(SR, 1, params);
        let seg = node.seg_frames;
        let mut sig = vec![0.0; 3 * seg];
        for s in sig.iter_mut().take(seg / 2) {
            *s = 1.0;
        }
        let out = run_mono(&mut node, &sig);
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(rms(&out[seg..2 * seg]) > 1e-3, "a mono reverse delay must echo back reversed");
    }
}
