//! Stutter / beat-repeat: a glitch effect that captures a fixed-length slice of
//! the incoming audio and replays that one slice several times in lockstep,
//! producing the stuttering, machine-gun repeat heard in electronic and
//! glitch-hop production.
//!
//! The node runs a free-running cycle made of `repeats + 1` equal slices. The
//! first slice of each cycle is the **capture** phase: the input is recorded
//! into a per-channel buffer and passed straight through. The following
//! `repeats` slices are the **replay** phase: the captured slice is read back
//! from the start, so the same grain is looped back-to-back. A short
//! raised-cosine fade is applied at both ends of every replayed grain so the
//! hard loop points stay click-free, and a dry / wet control blends the
//! stuttered grain against the live input during the replay slices.
//!
//! # Relationship
//!
//! Unlike the delay-line echoes of
//! [`DelayNode`](crate::nodes::effects::delay::DelayNode),
//! [`MultiTapDelayNode`](crate::nodes::effects::multi_tap_delay::MultiTapDelayNode),
//! [`PingPongDelayNode`](crate::nodes::effects::ping_pong_delay::PingPongDelayNode),
//! and [`ReverseDelayNode`](crate::nodes::effects::reverse_delay::ReverseDelayNode),
//! this node has no feedback delay line and no fractional read: it hard-switches
//! between recording one grain and looping that exact integer-length grain
//! forwards. Unlike [`GranularNode`](crate::nodes::effects::granular::GranularNode),
//! which overlaps many independently scheduled (often randomized) grains, the
//! stutter replays a single grain deterministically and in strict phase lock, so
//! each repeat is a bit-exact copy of the captured slice (up to the edge fade).
//!
//! # Real-time contract
//!
//! One record buffer per channel is allocated once in [`StutterNode::new`],
//! sized for the maximum slice length and channel count.
//! [`StutterNode::process`] performs no allocation, locking, or panic on the hot
//! path; non-finite input samples are treated as silence and every value written
//! into a record buffer is denormal-flushed. The capture slice passes the input
//! through unchanged, so the node reports no compensating latency
//! ([`StutterNode::latency_frames`] returns zero).
//!
//! # Provenance
//!
//! Slice capture plus phase-locked looped replay with a raised-cosine edge fade
//! is an elementary, widely documented classic DSP construction (for example
//! Zoelzer's DAFX time-segment / block-processing material and the long-standing
//! beat-repeat / stutter studio technique). This is pure classic DSP with no AI
//! or ML. This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, Google Resonance Audio, or Web Audio source or derived code**;
//! only the publicly documented capture-and-loop and raised-cosine fade formulas
//! are used.

use alloc::{vec, vec::Vec};
use core::f32::consts::PI;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Largest slice length, in milliseconds, the node can capture and repeat.
pub const MAX_STUTTER_SLICE_MS: Sample = 1_000.0;

/// Smallest slice length, in milliseconds, kept above a couple of frames so a
/// grain always spans enough samples for the edge fade.
pub const MIN_STUTTER_SLICE_MS: Sample = 10.0;

/// Largest number of replay slices per cycle.
pub const MAX_STUTTER_REPEATS: u32 = 16;

/// Largest edge-fade length in milliseconds.
pub const MAX_STUTTER_FADE_MS: Sample = 50.0;

/// Default slice length in milliseconds.
pub const DEFAULT_STUTTER_SLICE_MS: Sample = 125.0;

/// Default number of replay slices per cycle.
pub const DEFAULT_STUTTER_REPEATS: u32 = 3;

/// Default edge-fade length in milliseconds.
pub const DEFAULT_STUTTER_FADE_MS: Sample = 2.0;

/// Default dry / wet mix.
pub const DEFAULT_STUTTER_MIX: Sample = 1.0;

/// Parameters shared by every channel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StutterParams {
    /// Slice length in milliseconds: the length of the grain that is captured
    /// and then looped.
    pub slice_ms: Sample,
    /// Number of times the captured slice is replayed before a fresh slice is
    /// captured. Clamped into `[1, MAX_STUTTER_REPEATS]`.
    pub repeats: u32,
    /// Raised-cosine edge-fade length in milliseconds applied to both ends of
    /// each replayed grain to keep the loop points click-free.
    pub fade_ms: Sample,
    /// Dry / wet mix in `[0, 1]`: during the replay slices `0` is the live
    /// input and `1` is the stuttered grain only. The capture slice always
    /// passes the input through unchanged.
    pub mix: Sample,
}

impl Default for StutterParams {
    fn default() -> Self {
        Self {
            slice_ms: DEFAULT_STUTTER_SLICE_MS,
            repeats: DEFAULT_STUTTER_REPEATS,
            fade_ms: DEFAULT_STUTTER_FADE_MS,
            mix: DEFAULT_STUTTER_MIX,
        }
    }
}

impl StutterParams {
    /// Returns the parameters with every field clamped into range and any
    /// non-finite field replaced by its default.
    #[must_use]
    fn sanitised(self) -> Self {
        let d = Self::default();
        let clamp = |v: Sample, lo: Sample, hi: Sample, fallback: Sample| {
            if v.is_finite() { v.clamp(lo, hi) } else { fallback }
        };
        Self {
            slice_ms: clamp(self.slice_ms, MIN_STUTTER_SLICE_MS, MAX_STUTTER_SLICE_MS, d.slice_ms),
            repeats: self.repeats.clamp(1, MAX_STUTTER_REPEATS),
            fade_ms: clamp(self.fade_ms, 0.0, MAX_STUTTER_FADE_MS, d.fade_ms),
            mix: clamp(self.mix, 0.0, 1.0, d.mix),
        }
    }
}

/// A phase-locked slice capture / loop stutter effect.
#[derive(Clone, Debug)]
pub struct StutterNode {
    sample_rate: u32,
    channels: usize,
    /// Capacity (in frames) of every record buffer: the maximum slice length.
    max_slice_frames: usize,
    /// Active slice length in frames (the grain length actually used).
    seg_frames: usize,
    /// Number of replay slices per cycle (`>= 1`).
    repeats: usize,
    /// Edge-fade length in frames, always `<= seg_frames / 2`.
    fade_frames: usize,
    /// One record buffer per channel.
    bufs: Vec<Vec<Sample>>,
    /// Shared position within the current slice, in `[0, seg_frames)`.
    pos: usize,
    /// Which slice of the cycle is active: `0` captures, `1..=repeats` replay.
    slice_index: usize,
    wet: Sample,
    dry: Sample,
}

impl StutterNode {
    /// Builds a stutter effect for `channels` channels at `sample_rate` Hz.
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::stutter::{StutterNode, StutterParams};
    /// use prism_audio_core::graph::AudioNode;
    ///
    /// let node = StutterNode::new(48_000, 2, StutterParams::default());
    /// // The capture slice passes through, so the node adds no reported latency.
    /// assert_eq!(node.latency_frames(), 0);
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: StutterParams) -> Self {
        let sample_rate = sample_rate.max(1);
        let channels = channels.max(1);
        let max_slice_frames =
            ops::floor(MAX_STUTTER_SLICE_MS * sample_rate as Sample / 1_000.0) as usize + 1;
        let mut node = Self {
            sample_rate,
            channels,
            max_slice_frames,
            seg_frames: 1,
            repeats: 1,
            fade_frames: 0,
            bufs: vec![vec![0.0; max_slice_frames]; channels],
            pos: 0,
            slice_index: 0,
            wet: 1.0,
            dry: 0.0,
        };
        node.set_params(params);
        node
    }

    /// Number of channels this node was built for.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Replaces the slice length, repeat count, edge fade, and mix. The record
    /// buffer contents are preserved; the cycle restarts only if the new slice
    /// length no longer contains the current position.
    pub fn set_params(&mut self, params: StutterParams) {
        let params = params.sanitised();
        let frames = ops::floor(params.slice_ms * self.sample_rate as Sample / 1_000.0) as usize;
        self.seg_frames = frames.clamp(1, self.max_slice_frames);
        self.repeats = params.repeats as usize;
        let fade = ops::floor(params.fade_ms * self.sample_rate as Sample / 1_000.0) as usize;
        self.fade_frames = fade.min(self.seg_frames / 2);
        self.wet = params.mix;
        self.dry = 1.0 - params.mix;
        if self.pos >= self.seg_frames {
            self.pos = 0;
            self.slice_index = 0;
        }
        if self.slice_index > self.repeats {
            self.slice_index = 0;
            self.pos = 0;
        }
    }

    /// Raised-cosine edge-fade gain for a replayed grain at position `pos`.
    ///
    /// Both ends taper to zero so the hard loop points are click-free; the
    /// middle of the grain is unity. A zero fade length is a flat gain of one.
    fn fade(&self, pos: usize) -> Sample {
        let f = self.fade_frames;
        if f == 0 {
            return 1.0;
        }
        let seg = self.seg_frames;
        let ff = f as Sample;
        if pos < f {
            0.5 - 0.5 * ops::cos(PI * pos as Sample / ff)
        } else if seg - 1 - pos < f {
            let k = seg - 1 - pos;
            0.5 - 0.5 * ops::cos(PI * k as Sample / ff)
        } else {
            1.0
        }
    }
}

impl AudioNode for StutterNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let available = output.channels().min(input.channels());
        let channels = available.min(self.channels);
        let frames = output.active_frames().min(input.active_frames());

        let seg = self.seg_frames;
        let wet = self.wet;
        let dry = self.dry;

        for f in 0..frames {
            let pos = self.pos;
            let capturing = self.slice_index == 0;
            let env = if capturing { 1.0 } else { self.fade(pos) };

            for ch in 0..channels {
                let x_raw = input.channel(ch)[f];
                let x = if x_raw.is_finite() { x_raw } else { 0.0 };
                let grain = if capturing {
                    self.bufs[ch][pos] = flush_denormal(x);
                    x
                } else {
                    self.bufs[ch][pos] * env
                };
                output.channel_mut(ch)[f] = dry * x + wet * grain;
            }

            if pos + 1 == seg {
                self.pos = 0;
                if self.slice_index == self.repeats {
                    self.slice_index = 0;
                } else {
                    self.slice_index += 1;
                }
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
        for buf in &mut self.bufs {
            for s in buf.iter_mut() {
                *s = 0.0;
            }
        }
        self.pos = 0;
        self.slice_index = 0;
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

    fn run_mono(node: &mut StutterNode, signal: &[Sample]) -> Vec<Sample> {
        let len = signal.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        let mut output = AudioBuffer::new(ChannelLayout::Mono, len.max(1));
        input.set_active_frames(len);
        output.set_active_frames(len);
        input.channel_mut(0)[..len].copy_from_slice(signal);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        outputs[0].channel(0)[..len].to_vec()
    }

    fn rms(samples: &[Sample]) -> Sample {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: Sample = samples.iter().map(|&v| v * v).sum();
        ops::sqrt(sum / samples.len() as Sample)
    }

    #[test]
    fn reports_zero_latency() {
        let node = StutterNode::new(SR, 2, StutterParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn channels_getter_reports_build_width() {
        let node = StutterNode::new(SR, 2, StutterParams::default());
        assert_eq!(node.channels(), 2);
    }

    #[test]
    fn default_params_are_in_domain() {
        let d = StutterParams::default();
        assert_eq!(d, d.sanitised());
    }

    #[test]
    fn silence_in_silence_out() {
        let mut node = StutterNode::new(SR, 1, StutterParams::default());
        let out = run_mono(&mut node, &vec![0.0; 4_000]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dry_mix_passes_input_through_exactly() {
        // mix = 0 => fully dry on every slice, including replay slices.
        let params = StutterParams {
            slice_ms: 10.0,
            repeats: 4,
            fade_ms: 2.0,
            mix: 0.0,
        };
        let mut node = StutterNode::new(SR, 1, params);
        let sig: Vec<Sample> = (0..5_000).map(|i| ops::sin(i as Sample * 0.05)).collect();
        let out = run_mono(&mut node, &sig);
        for (o, s) in out.iter().zip(sig.iter()) {
            assert!((o - s).abs() < 1e-6);
        }
    }

    #[test]
    fn capture_slice_passes_input_through_even_at_full_wet() {
        // The first slice of the cycle is always transparent regardless of mix.
        let params = StutterParams {
            slice_ms: 10.0,
            repeats: 3,
            fade_ms: 0.0,
            mix: 1.0,
        };
        let mut node = StutterNode::new(SR, 1, params);
        let seg = node.seg_frames;
        let sig: Vec<Sample> = (0..seg).map(|i| ops::sin(i as Sample * 0.03)).collect();
        let out = run_mono(&mut node, &sig);
        for (o, s) in out.iter().zip(sig.iter()) {
            assert!((o - s).abs() < 1e-6);
        }
    }

    #[test]
    fn replay_slice_repeats_the_captured_grain() {
        // With fade off and full wet, the replay slice should output the grain
        // captured during the preceding capture slice, ignoring the live input.
        let params = StutterParams {
            slice_ms: 10.0,
            repeats: 1,
            fade_ms: 0.0,
            mix: 1.0,
        };
        let mut node = StutterNode::new(SR, 1, params);
        let seg = node.seg_frames;
        // Capture slice (frames 0..seg) carries a known grain; replay slice
        // (frames seg..2*seg) carries different live input that must be ignored.
        let mut sig = vec![0.0; 2 * seg];
        for (i, s) in sig.iter_mut().enumerate() {
            *s = if i < seg {
                ops::sin(i as Sample * 0.07)
            } else {
                -0.5
            };
        }
        let out = run_mono(&mut node, &sig);
        for k in 0..seg {
            assert!((out[seg + k] - sig[k]).abs() < 1e-6);
        }
    }

    #[test]
    fn edge_fade_tapers_grain_ends_to_zero() {
        let params = StutterParams {
            slice_ms: 10.0,
            repeats: 1,
            fade_ms: 2.0,
            mix: 1.0,
        };
        let mut node = StutterNode::new(SR, 1, params);
        let seg = node.seg_frames;
        let sig = vec![1.0; 2 * seg];
        let out = run_mono(&mut node, &sig);
        // First replayed sample is at the start of its fade -> near zero.
        assert!(out[seg].abs() < 1e-6);
        // Middle of the replayed grain is unity (grain is all ones).
        assert!((out[seg + seg / 2] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn minimum_slice_is_clamped() {
        let params = StutterParams {
            slice_ms: 0.0,
            ..StutterParams::default()
        };
        let node = StutterNode::new(SR, 1, params);
        let expected =
            ops::floor(MIN_STUTTER_SLICE_MS * SR as Sample / 1_000.0) as usize;
        assert_eq!(node.seg_frames, expected);
    }

    #[test]
    fn maximum_slice_is_clamped() {
        let params = StutterParams {
            slice_ms: 10_000.0,
            ..StutterParams::default()
        };
        let node = StutterNode::new(SR, 1, params);
        let expected =
            ops::floor(MAX_STUTTER_SLICE_MS * SR as Sample / 1_000.0) as usize;
        assert_eq!(node.seg_frames, expected);
    }

    #[test]
    fn repeats_are_clamped() {
        let params = StutterParams {
            repeats: 1_000,
            ..StutterParams::default()
        };
        let node = StutterNode::new(SR, 1, params);
        assert_eq!(node.repeats, MAX_STUTTER_REPEATS as usize);
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut node = StutterNode::new(SR, 1, StutterParams::default());
        let mut sig = vec![0.5; 4_000];
        sig[10] = Sample::INFINITY;
        sig[20] = Sample::NAN;
        sig[30] = Sample::NEG_INFINITY;
        let out = run_mono(&mut node, &sig);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = StutterNode::new(SR, 1, StutterParams::default());
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
    fn extreme_params_do_not_panic() {
        let params = StutterParams {
            slice_ms: Sample::INFINITY,
            repeats: u32::MAX,
            fade_ms: 1e9,
            mix: 42.0,
        };
        let mut node = StutterNode::new(SR, 2, params);
        let _ = run_mono(&mut node, &vec![0.3; 6_000]);
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let params = StutterParams {
            slice_ms: Sample::NAN,
            repeats: 2,
            fade_ms: Sample::INFINITY,
            mix: Sample::NEG_INFINITY,
        };
        let s = params.sanitised();
        let d = StutterParams::default();
        assert_eq!(s.slice_ms, d.slice_ms);
        assert_eq!(s.fade_ms, d.fade_ms);
        assert_eq!(s.mix, d.mix);
        assert_eq!(s.repeats, 2);
    }

    #[test]
    fn surplus_channels_pass_through() {
        // Build a 2-channel node but feed a 4-channel (Quad) buffer; channels
        // 2 and 3 must be copied through untouched.
        let mut node = StutterNode::new(SR, 2, StutterParams::default());
        let len = 2_000;
        let mut input = AudioBuffer::new(ChannelLayout::Quad, len);
        let mut output = AudioBuffer::new(ChannelLayout::Quad, len);
        input.set_active_frames(len);
        output.set_active_frames(len);
        for ch in 0..4 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                *s = ops::sin(i as Sample * 0.01 + ch as Sample);
            }
        }
        let expected2 = input.channel(2).to_vec();
        let expected3 = input.channel(3).to_vec();
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(len), &mut io);
        assert_eq!(outputs[0].channel(2), expected2.as_slice());
        assert_eq!(outputs[0].channel(3), expected3.as_slice());
    }

    #[test]
    fn reset_restores_fresh_state() {
        let params = StutterParams {
            slice_ms: 10.0,
            repeats: 2,
            fade_ms: 1.0,
            mix: 1.0,
        };
        let mut node = StutterNode::new(SR, 1, params);
        let sig: Vec<Sample> = (0..5_000).map(|i| ops::sin(i as Sample * 0.04)).collect();
        let first = run_mono(&mut node, &sig);
        node.reset();
        let second = run_mono(&mut node, &sig);
        let max_err = first
            .iter()
            .zip(second.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 1e-6);
    }

    #[test]
    fn set_params_changes_slice_length() {
        let mut node = StutterNode::new(SR, 1, StutterParams::default());
        let before = node.seg_frames;
        node.set_params(StutterParams {
            slice_ms: 20.0,
            ..StutterParams::default()
        });
        let after = node.seg_frames;
        let expected = ops::floor(20.0 * SR as Sample / 1_000.0) as usize;
        assert_eq!(after, expected);
        assert_ne!(before, after);
        // Still produces finite output after the change.
        let out = run_mono(&mut node, &vec![0.4; 6_000]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mono_produces_stuttered_repeat() {
        // A transient confined to the capture slice should reappear in the
        // replay slice (full wet, no fade).
        let params = StutterParams {
            slice_ms: 10.0,
            repeats: 1,
            fade_ms: 0.0,
            mix: 1.0,
        };
        let mut node = StutterNode::new(SR, 1, params);
        let seg = node.seg_frames;
        let mut sig = vec![0.0; 2 * seg];
        for (i, s) in sig.iter_mut().enumerate().take(seg / 2) {
            *s = ops::sin(i as Sample * 0.1);
        }
        let out = run_mono(&mut node, &sig);
        // The replay slice carries the captured energy even though the live
        // input during replay is silent.
        let replay = &out[seg..2 * seg];
        assert!(rms(replay) > 1e-3);
    }
}
