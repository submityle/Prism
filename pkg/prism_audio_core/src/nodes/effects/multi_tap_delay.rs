//! Multi-tap delay: a single shared delay line read by several independently
//! timed, gained, and panned taps.
//!
//! Where [`DelayNode`](crate::nodes::effects::delay::DelayNode) exposes exactly
//! one fractional read tap (the echo / slap-back primitive), a *multi-tap*
//! delay reads the same stored history at several arbitrary offsets at once.
//! Each tap has its own delay time, output gain, and stereo pan, so one node
//! can paint a rhythmic pattern of echoes, emulate a cluster of discrete early
//! reflections, or spread a sound across the stereo field. A single global
//! feedback coefficient recirculates the summed taps back into the line to
//! sustain the pattern.
//!
//! The delay line itself is **mono**: the input channels are summed to a single
//! history buffer, each tap reads one delayed value, and that value is panned
//! into the stereo output. The dry signal is mixed back in per channel, so the
//! unprocessed image is preserved while the wet taps form a coherent stereo
//! pattern (the standard architecture for a send / pattern delay). All storage
//! is allocated at construction, so
//! [`process`](crate::graph::AudioNode::process) performs no allocation,
//! locking, or panicking and is safe on the audio callback thread.
//!
//! # Provenance
//!
//! Built from first principles on top of this crate's own ring-buffer, linear
//! interpolation ([`lerp`]), and [`equal_power_pan`] primitives. It contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google
//! Resonance Audio source or derived code** -- only the textbook idea of
//! reading one delay line at multiple offsets.
//!
//! # Relationship
//!
//! Distinct from the other delay-family nodes:
//! - [`DelayNode`](crate::nodes::effects::delay::DelayNode) keeps one
//!   per-channel ring with a single fractional tap and feedback. This node
//!   keeps one shared mono ring read by up to [`MAX_TAPS`] simultaneous taps,
//!   each with independent time / gain / pan, for rhythmic or spatial patterns.
//! - [`ChorusNode`](crate::nodes::effects::chorus::ChorusNode) and
//!   [`FlangerNode`](crate::nodes::effects::flanger::FlangerNode) sweep their
//!   taps with an LFO for pitch-modulated thickening; a multi-tap delay holds
//!   its tap times fixed (smoothed only to avoid zipper noise) and never
//!   modulates them cyclically.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, equal_power_pan, flush_denormal, lerp};
use crate::param::{Ramp, Smoothed};

/// Maximum number of simultaneous taps a [`MultiTapDelayNode`] can read.
pub const MAX_TAPS: usize = 8;

/// Largest stable feedback coefficient, kept just below unity so a sustained
/// recirculation decays instead of building without bound.
pub const MAX_FEEDBACK: Sample = 0.999;

/// Default number of active taps when a [`MultiTapDelayParams`] is built with
/// [`MultiTapDelayParams::default`].
pub const DEFAULT_TAP_COUNT: usize = 3;

/// Default wet (processed) mix gain.
pub const DEFAULT_WET: Sample = 0.5;

/// Default dry (unprocessed) mix gain.
pub const DEFAULT_DRY: Sample = 1.0;

/// Default feedback coefficient.
pub const DEFAULT_FEEDBACK: Sample = 0.0;

/// Configuration for a single tap of a [`MultiTapDelayNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TapSpec {
    /// Delay time in seconds (clamped to `[0, max_delay]` at apply time).
    pub delay_seconds: Sample,
    /// Linear output gain applied to this tap before summing.
    pub gain: Sample,
    /// Stereo pan in `[-1, 1]` (`-1` = hard left, `0` = center, `1` = hard
    /// right); ignored for mono output.
    pub pan: Sample,
}

impl Default for TapSpec {
    #[inline]
    fn default() -> Self {
        Self {
            delay_seconds: 0.0,
            gain: 0.0,
            pan: 0.0,
        }
    }
}

/// Full parameter set describing every tap plus the global mix controls.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiTapDelayParams {
    /// Per-tap configuration; only the first `active_taps` entries are read.
    pub taps: [TapSpec; MAX_TAPS],
    /// Number of leading taps that contribute to the output (clamped to
    /// `[0, MAX_TAPS]`).
    pub active_taps: usize,
    /// Global feedback coefficient (clamped to `[0, MAX_FEEDBACK]`).
    pub feedback: Sample,
    /// Wet (processed) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed input) mix gain.
    pub dry: Sample,
}

impl Default for MultiTapDelayParams {
    #[inline]
    fn default() -> Self {
        let mut taps = [TapSpec::default(); MAX_TAPS];
        // A gently decaying triplet spread across the stereo field.
        taps[0] = TapSpec {
            delay_seconds: 0.125,
            gain: 0.8,
            pan: -0.6,
        };
        taps[1] = TapSpec {
            delay_seconds: 0.250,
            gain: 0.6,
            pan: 0.6,
        };
        taps[2] = TapSpec {
            delay_seconds: 0.375,
            gain: 0.45,
            pan: 0.0,
        };
        Self {
            taps,
            active_taps: DEFAULT_TAP_COUNT,
            feedback: DEFAULT_FEEDBACK,
            wet: DEFAULT_WET,
            dry: DEFAULT_DRY,
        }
    }
}

/// A mono delay line read by up to [`MAX_TAPS`] simultaneous taps
/// (input port 0 -> output port 0).
///
/// # Examples
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::multi_tap_delay::{
///     MultiTapDelayNode, MultiTapDelayParams, TapSpec,
/// };
///
/// // One tap five frames back, fully wet.
/// let mut taps = [TapSpec::default(); 8];
/// taps[0] = TapSpec { delay_seconds: 5.0 / 48_000.0, gain: 1.0, pan: 0.0 };
/// let params = MultiTapDelayParams { taps, active_taps: 1, feedback: 0.0, wet: 1.0, dry: 0.0 };
/// let mut node = MultiTapDelayNode::new(48_000, 32, params);
///
/// let mut input = AudioBuffer::new(ChannelLayout::Mono, 16);
/// input.channel_mut(0)[0] = 1.0;
/// let inputs = [input];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 16)];
/// let ctx = RenderContext { sample_rate: 48_000, frames: 16, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
/// assert!((outputs[0].channel(0)[5] - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct MultiTapDelayNode {
    /// Sample rate in Hz, used to convert tap delay times from seconds.
    sample_rate: u32,
    /// Shared mono history buffer of length `ring_len`.
    ring: Vec<Sample>,
    /// Ring length in frames (`max_delay + 2`).
    ring_len: usize,
    /// Write cursor into the ring.
    write_pos: usize,
    /// Maximum addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// Number of leading taps currently contributing.
    active_taps: usize,
    /// Smoothed delay time (frames) per tap.
    delays: [Smoothed; MAX_TAPS],
    /// Smoothed linear gain per tap.
    gains: [Smoothed; MAX_TAPS],
    /// Smoothed pan position per tap.
    pans: [Smoothed; MAX_TAPS],
    /// Smoothed global feedback coefficient.
    feedback: Smoothed,
    /// Smoothed wet mix gain.
    wet: Smoothed,
    /// Smoothed dry mix gain.
    dry: Smoothed,
}

impl MultiTapDelayNode {
    /// Builds a multi-tap delay running at `sample_rate` Hz with a maximum
    /// addressable delay of `max_delay_frames` (clamped to at least 1) frames,
    /// initialised from `params`.
    ///
    /// `channels` only influences how the input is summed to the mono line and
    /// how many output channels are written; the history buffer is always mono.
    /// All initial values start settled (no glide).
    #[must_use]
    pub fn new(sample_rate: u32, max_delay_frames: usize, params: MultiTapDelayParams) -> Self {
        let max = max_delay_frames.max(1);
        let ring_len = max + 2;
        let max_delay = max as Sample;
        let mut node = Self {
            sample_rate,
            ring: vec![0.0; ring_len],
            ring_len,
            write_pos: 0,
            max_delay,
            active_taps: 0,
            delays: core::array::from_fn(|_| Smoothed::new(0.0)),
            gains: core::array::from_fn(|_| Smoothed::new(0.0)),
            pans: core::array::from_fn(|_| Smoothed::new(0.0)),
            feedback: Smoothed::new(0.0),
            wet: Smoothed::new(0.0),
            dry: Smoothed::new(0.0),
        };
        node.apply_params(&params, Ramp::Immediate);
        node
    }

    /// Returns the maximum addressable delay in frames.
    #[inline]
    #[must_use]
    pub fn max_delay_frames(&self) -> Sample {
        self.max_delay
    }

    /// Returns the number of active taps.
    #[inline]
    #[must_use]
    pub fn active_taps(&self) -> usize {
        self.active_taps
    }

    /// Snapshots the current parameter targets back into a
    /// [`MultiTapDelayParams`].
    #[must_use]
    pub fn params(&self) -> MultiTapDelayParams {
        let inv_sr = if self.sample_rate == 0 {
            0.0
        } else {
            1.0 / self.sample_rate as Sample
        };
        let taps = core::array::from_fn(|t| TapSpec {
            delay_seconds: self.delays[t].target() * inv_sr,
            gain: self.gains[t].target(),
            pan: self.pans[t].target(),
        });
        MultiTapDelayParams {
            taps,
            active_taps: self.active_taps,
            feedback: self.feedback.target(),
            wet: self.wet.target(),
            dry: self.dry.target(),
        }
    }

    /// Retargets every parameter, gliding with `ramp` so changes stay
    /// click-free. Filter and ring state are preserved.
    pub fn set_params(&mut self, params: &MultiTapDelayParams, ramp: Ramp) {
        self.apply_params(params, ramp);
    }

    /// Shared parameter application used by both the constructor and
    /// [`set_params`](Self::set_params).
    fn apply_params(&mut self, params: &MultiTapDelayParams, ramp: Ramp) {
        self.active_taps = params.active_taps.min(MAX_TAPS);
        for (t, spec) in params.taps.iter().enumerate() {
            let frames = finite(spec.delay_seconds).max(0.0) * self.sample_rate as Sample;
            self.delays[t].set_target(frames.clamp(0.0, self.max_delay), ramp);
            self.gains[t].set_target(finite(spec.gain), ramp);
            self.pans[t].set_target(finite(spec.pan).clamp(-1.0, 1.0), ramp);
        }
        self.feedback
            .set_target(finite(params.feedback).clamp(0.0, MAX_FEEDBACK), ramp);
        self.wet.set_target(finite(params.wet), ramp);
        self.dry.set_target(finite(params.dry), ramp);
    }
}

/// Replaces a non-finite value with zero so a bad parameter cannot poison the
/// feedback loop.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

/// Computes the two neighbouring ring indices and the interpolation fraction
/// for a read `delay` frames behind the write head `w`.
#[inline]
fn tap_indices(w: usize, delay: Sample, len_i: isize) -> (usize, usize, Sample) {
    let read_pos = w as Sample - delay;
    let base = ops::floor(read_pos);
    let frac = read_pos - base;
    let base_i = base as isize;
    let i0 = base_i.rem_euclid(len_i) as usize;
    let i1 = (base_i + 1).rem_euclid(len_i) as usize;
    (i0, i1, frac)
}

impl AudioNode for MultiTapDelayNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || out_channels == 0 || in_channels == 0 {
            return;
        }
        let stereo = out_channels >= 2;
        let inv_in = 1.0 / in_channels as Sample;
        let ring_len = self.ring_len;
        let len_i = ring_len as isize;
        let active = self.active_taps;

        for f in 0..frames {
            // Advance the global controls once per frame.
            let feedback = self.feedback.next_sample();
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            // Mono-sum the input for the shared delay line.
            let mut mono_in = 0.0;
            for ch in 0..in_channels {
                mono_in += input.channel(ch)[f];
            }
            mono_in *= inv_in;

            let w = self.write_pos;

            // Read every active tap from the shared ring.
            let mut tap_sum = 0.0; // summed taps, pre-pan, drives feedback.
            let mut wet_l = 0.0;
            let mut wet_r = 0.0;
            for t in 0..active {
                let delay = self.delays[t].next_sample();
                let gain = self.gains[t].next_sample();
                let pan = self.pans[t].next_sample();
                let (i0, i1, frac) = tap_indices(w, delay, len_i);
                let delayed = lerp(self.ring[i0], self.ring[i1], frac);
                let contribution = gain * delayed;
                tap_sum += contribution;
                if stereo {
                    let (lg, rg) = equal_power_pan(pan);
                    wet_l += lg * contribution;
                    wet_r += rg * contribution;
                }
            }

            // Recirculate the summed taps into the mono line.
            self.ring[w] = flush_denormal(mono_in + feedback * tap_sum);

            // Mix dry (per channel) with the wet pattern.
            if stereo {
                output.channel_mut(0)[f] = dry * input.channel(0)[f] + wet * wet_l;
                output.channel_mut(1)[f] = dry * input.channel(1)[f] + wet * wet_r;
                for ch in 2..out_channels {
                    let dryin = if ch < in_channels {
                        input.channel(ch)[f]
                    } else {
                        0.0
                    };
                    output.channel_mut(ch)[f] = dry * dryin;
                }
            } else {
                output.channel_mut(0)[f] = dry * input.channel(0)[f] + wet * tap_sum;
            }

            self.write_pos = if w + 1 == ring_len { 0 } else { w + 1 };
        }
    }

    fn reset(&mut self) {
        for s in &mut self.ring {
            *s = 0.0;
        }
        self.write_pos = 0;
        for t in 0..MAX_TAPS {
            self.delays[t] = Smoothed::new(self.delays[t].target());
            self.gains[t] = Smoothed::new(self.gains[t].target());
            self.pans[t] = Smoothed::new(self.pans[t].target());
        }
        self.feedback = Smoothed::new(self.feedback.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
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

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    fn single_tap_params(delay_frames: Sample) -> MultiTapDelayParams {
        let mut taps = [TapSpec::default(); MAX_TAPS];
        taps[0] = TapSpec {
            delay_seconds: delay_frames / SR as Sample,
            gain: 1.0,
            pan: 0.0,
        };
        MultiTapDelayParams {
            taps,
            active_taps: 1,
            feedback: 0.0,
            wet: 1.0,
            dry: 0.0,
        }
    }

    fn run(node: &mut MultiTapDelayNode, input: AudioBuffer, out: AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let inputs = [input];
        let mut outputs = [out];
        outputs[0].set_active_frames(frames);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [out] = outputs;
        out
    }

    #[test]
    fn dry_passthrough_when_wet_zero() {
        let mut params = single_tap_params(4.0);
        params.wet = 0.0;
        params.dry = 1.0;
        let mut node = MultiTapDelayNode::new(SR, 32, params);
        let mut input = mono(8);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = i as Sample + 1.0;
        }
        let expected: Vec<Sample> = input.channel(0).to_vec();
        let out = run(&mut node, input, mono(8));
        assert_eq!(out.channel(0), expected.as_slice());
    }

    #[test]
    fn single_integer_tap_shifts_impulse() {
        let mut node = MultiTapDelayNode::new(SR, 32, single_tap_params(5.0));
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let out = run(&mut node, input, mono(16));
        assert!((out.channel(0)[5] - 1.0).abs() < 1e-6);
        for (i, &s) in out.channel(0).iter().enumerate() {
            if i != 5 {
                assert!(s.abs() < 1e-6, "unexpected energy at {i}: {s}");
            }
        }
    }

    #[test]
    fn two_taps_produce_two_echoes() {
        let mut taps = [TapSpec::default(); MAX_TAPS];
        taps[0] = TapSpec {
            delay_seconds: 3.0 / SR as Sample,
            gain: 1.0,
            pan: 0.0,
        };
        taps[1] = TapSpec {
            delay_seconds: 7.0 / SR as Sample,
            gain: 0.5,
            pan: 0.0,
        };
        let params = MultiTapDelayParams {
            taps,
            active_taps: 2,
            feedback: 0.0,
            wet: 1.0,
            dry: 0.0,
        };
        let mut node = MultiTapDelayNode::new(SR, 32, params);
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let out = run(&mut node, input, mono(16));
        assert!((out.channel(0)[3] - 1.0).abs() < 1e-6);
        assert!((out.channel(0)[7] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fractional_tap_interpolates() {
        // A 4.5-frame delay should split a unit impulse between frames 4 and 5.
        let mut node = MultiTapDelayNode::new(SR, 32, single_tap_params(4.5));
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let out = run(&mut node, input, mono(16));
        assert!((out.channel(0)[4] - 0.5).abs() < 1e-4);
        assert!((out.channel(0)[5] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn hard_left_pan_silences_right() {
        let mut taps = [TapSpec::default(); MAX_TAPS];
        taps[0] = TapSpec {
            delay_seconds: 4.0 / SR as Sample,
            gain: 1.0,
            pan: -1.0,
        };
        let params = MultiTapDelayParams {
            taps,
            active_taps: 1,
            feedback: 0.0,
            wet: 1.0,
            dry: 0.0,
        };
        let mut node = MultiTapDelayNode::new(SR, 32, params);
        let mut input = stereo(16);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        let out = run(&mut node, input, stereo(16));
        assert!(out.channel(0)[4].abs() > 0.5, "left tap should be audible");
        for &s in out.channel(1) {
            assert!(s.abs() < 1e-6, "right channel should be silent");
        }
    }

    #[test]
    fn center_pan_is_equal_power() {
        let mut node = MultiTapDelayNode::new(SR, 32, single_tap_params(4.0));
        let mut input = stereo(16);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = 1.0;
        let out = run(&mut node, input, stereo(16));
        let frac_1_sqrt_2 = core::f32::consts::FRAC_1_SQRT_2;
        assert!((out.channel(0)[4] - frac_1_sqrt_2).abs() < 1e-4);
        assert!((out.channel(1)[4] - frac_1_sqrt_2).abs() < 1e-4);
    }

    #[test]
    fn feedback_recirculates_pattern() {
        let mut taps = [TapSpec::default(); MAX_TAPS];
        taps[0] = TapSpec {
            delay_seconds: 4.0 / SR as Sample,
            gain: 1.0,
            pan: 0.0,
        };
        let params = MultiTapDelayParams {
            taps,
            active_taps: 1,
            feedback: 0.5,
            wet: 1.0,
            dry: 0.0,
        };
        let mut node = MultiTapDelayNode::new(SR, 64, params);
        let mut input = mono(32);
        input.channel_mut(0)[0] = 1.0;
        let out = run(&mut node, input, mono(32));
        // Echoes at 4, 8, 12... decaying by 0.5 each repeat.
        assert!((out.channel(0)[4] - 1.0).abs() < 1e-5);
        assert!((out.channel(0)[8] - 0.5).abs() < 1e-5);
        assert!((out.channel(0)[12] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn feedback_is_clamped_below_unity() {
        let mut params = single_tap_params(4.0);
        params.feedback = 10.0;
        let node = MultiTapDelayNode::new(SR, 32, params);
        assert!((node.params().feedback - MAX_FEEDBACK).abs() < 1e-6);
    }

    #[test]
    fn active_taps_clamped_to_max() {
        let mut params = MultiTapDelayParams::default();
        params.active_taps = 999;
        let node = MultiTapDelayNode::new(SR, 32, params);
        assert_eq!(node.active_taps(), MAX_TAPS);
    }

    #[test]
    fn reset_restores_silence() {
        let mut node = MultiTapDelayNode::new(SR, 32, single_tap_params(4.0));
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let _ = run(&mut node, input, mono(16));
        node.reset();
        let out = run(&mut node, mono(16), mono(16));
        for &s in out.channel(0) {
            assert!(s.abs() < 1e-9, "ring should be silent after reset");
        }
    }

    #[test]
    fn reset_is_bit_exact_reproducible() {
        let mut node = MultiTapDelayNode::new(SR, 64, MultiTapDelayParams::default());
        let mut input = stereo(32);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(i as Sample * 0.3);
        }
        for (i, s) in input.channel_mut(1).iter_mut().enumerate() {
            *s = ops::sin(i as Sample * 0.37);
        }
        let first = run(&mut node, input.clone(), stereo(32));
        node.reset();
        let second = run(&mut node, input, stereo(32));
        assert_eq!(first.channel(0), second.channel(0));
        assert_eq!(first.channel(1), second.channel(1));
    }

    #[test]
    fn params_round_trip() {
        let params = MultiTapDelayParams::default();
        let node = MultiTapDelayNode::new(SR, 48_000, params);
        let back = node.params();
        assert_eq!(back.active_taps, params.active_taps);
        for t in 0..params.active_taps {
            assert!((back.taps[t].delay_seconds - params.taps[t].delay_seconds).abs() < 1e-4);
            assert!((back.taps[t].gain - params.taps[t].gain).abs() < 1e-6);
            assert!((back.taps[t].pan - params.taps[t].pan).abs() < 1e-6);
        }
        assert!((back.wet - params.wet).abs() < 1e-6);
        assert!((back.dry - params.dry).abs() < 1e-6);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = MultiTapDelayNode::new(SR, 32, single_tap_params(4.0));
        let mut input = mono(4);
        input.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [mono(4)];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn non_finite_param_does_not_poison_output() {
        let mut params = single_tap_params(4.0);
        params.taps[0].gain = Sample::NAN;
        params.feedback = Sample::INFINITY;
        let mut node = MultiTapDelayNode::new(SR, 32, params);
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let out = run(&mut node, input, mono(16));
        for &s in out.channel(0) {
            assert!(s.is_finite(), "output must stay finite");
        }
    }

    #[test]
    fn set_params_updates_targets() {
        let mut node = MultiTapDelayNode::new(SR, 48_000, MultiTapDelayParams::default());
        let mut params = MultiTapDelayParams::default();
        params.wet = 0.25;
        params.active_taps = 2;
        node.set_params(&params, Ramp::Immediate);
        assert_eq!(node.active_taps(), 2);
        assert!((node.params().wet - 0.25).abs() < 1e-6);
    }

    #[test]
    fn silent_input_stays_silent() {
        let mut node = MultiTapDelayNode::new(SR, 32, MultiTapDelayParams::default());
        let out = run(&mut node, stereo(16), stereo(16));
        for ch in 0..2 {
            for &s in out.channel(ch) {
                assert!(s.abs() < 1e-9);
            }
        }
    }
}
