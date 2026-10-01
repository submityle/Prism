//! Haas stereo widener: broadens the stereo image with a short inter-channel
//! time delay (the precedence / Haas effect) instead of a side-gain boost.
//!
//! When two copies of a sound arrive within roughly 1-35 ms of each other the
//! auditory system fuses them into a single event but shifts and widens its
//! perceived location toward the leading ear -- the *precedence effect* first
//! quantified by Helmut Haas in 1949. This node exploits that by delaying only
//! the stereo *side* component by a few milliseconds, decorrelating the left
//! and right channels in time so the image feels wider without changing the
//! spectral balance the way a side-gain widener does.
//!
//! The signal is processed in the Mid-Side domain:
//!
//! ```text
//! M = (L + R) / 2        S = (L - R) / 2
//! S' = side_level * lerp(S, delay(S, t), width)
//! L  = M + S'            R  = M - S'
//! ```
//!
//! Because the delay is applied to the side only, the mid (and therefore the
//! mono sum `L + R = 2 * M`) is left perfectly time-aligned, so the widened
//! signal stays **mono-compatible**: summing to mono neither comb-filters nor
//! cancels. `width = 0` reproduces the input exactly (direct side only);
//! increasing `width` blends in the time-delayed side for progressively more
//! decorrelation. All storage is allocated at construction, so
//! [`process`](crate::graph::AudioNode::process) performs no allocation,
//! locking, or panicking and is safe on the audio callback thread.
//!
//! # Provenance
//!
//! Built from first principles on this crate's own ring-buffer, linear
//! interpolation ([`lerp`]), and the public Blumlein Mid-Side sum/difference
//! identities. The precedence / Haas effect is textbook psychoacoustics. This
//! module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! or Google Resonance Audio source or derived code**.
//!
//! # Relationship
//!
//! Distinct from the other width / delay nodes:
//! - [`StereoWidthNode`](crate::nodes::effects::stereo_width::StereoWidthNode)
//!   scales the side **amplitude** (a static gain); it never introduces a time
//!   difference. This node widens purely through a short **time delay** on the
//!   side (decorrelation / precedence) and keeps the side gain near unity.
//! - [`MidSideMatrixNode`](crate::nodes::effects::mid_side_matrix::MidSideMatrixNode)
//!   only encodes / decodes M/S with trim gains and adds no delay.
//! - [`ChorusNode`](crate::nodes::effects::chorus::ChorusNode),
//!   [`FlangerNode`](crate::nodes::effects::flanger::FlangerNode), and
//!   [`VibratoNode`](crate::nodes::effects::vibrato::VibratoNode) sweep their
//!   delay with an LFO for pitch modulation; the Haas delay is fixed (smoothed
//!   only to avoid zipper noise) and never cyclically modulated.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};
use crate::param::{Ramp, Smoothed};

/// Largest side-blend amount the node accepts.
///
/// `width` is clamped to `[0, MAX_WIDTH]`; `0` reproduces the input exactly and
/// `1` uses the fully time-delayed side. Values above `1` are not meaningful
/// for a blend, so the ceiling is `1`.
pub const MAX_WIDTH: Sample = 1.0;

/// Largest overall side level the node accepts (an intensity trim on the
/// widened side signal).
pub const MAX_SIDE_LEVEL: Sample = 2.0;

/// Default Haas delay in milliseconds (inside the fusion window).
pub const DEFAULT_DELAY_MS: Sample = 12.0;

/// Default side-blend amount.
pub const DEFAULT_WIDTH: Sample = 0.5;

/// Default overall side level.
pub const DEFAULT_SIDE_LEVEL: Sample = 1.0;

/// Construction / automation parameters for a [`HaasWidenerNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HaasWidenerParams {
    /// Haas delay applied to the side component, in milliseconds (clamped to
    /// `[0, max_delay]`).
    pub delay_ms: Sample,
    /// Blend between the direct side (`0`) and the time-delayed side (`1`),
    /// clamped to `[0, MAX_WIDTH]`.
    pub width: Sample,
    /// Overall level applied to the widened side signal, clamped to
    /// `[0, MAX_SIDE_LEVEL]`.
    pub side_level: Sample,
}

impl Default for HaasWidenerParams {
    #[inline]
    fn default() -> Self {
        Self {
            delay_ms: DEFAULT_DELAY_MS,
            width: DEFAULT_WIDTH,
            side_level: DEFAULT_SIDE_LEVEL,
        }
    }
}

/// A Haas (precedence) stereo widener (input port 0 -> output port 0).
///
/// # Examples
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::effects::haas_widener::{HaasWidenerNode, HaasWidenerParams};
///
/// let params = HaasWidenerParams { delay_ms: 10.0, width: 1.0, side_level: 1.0 };
/// let mut node = HaasWidenerNode::new(48_000, 2048, params);
///
/// // A hard-panned-left transient gains a delayed right-channel echo.
/// let mut input = AudioBuffer::new(ChannelLayout::Stereo, 1024);
/// input.channel_mut(0)[0] = 1.0; // left only
/// let inputs = [input];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, 1024)];
/// let ctx = RenderContext { sample_rate: 48_000, frames: 1024, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
/// // The mono sum is preserved: L + R still equals the original L + R.
/// let sum0 = outputs[0].channel(0)[0] + outputs[0].channel(1)[0];
/// assert!((sum0 - 1.0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone)]
pub struct HaasWidenerNode {
    /// Sample rate in Hz, used to convert the delay time from milliseconds.
    sample_rate: u32,
    /// Shared mono history buffer of the side signal, length `ring_len`.
    ring: Vec<Sample>,
    /// Ring length in frames (`max_delay + 2`).
    ring_len: usize,
    /// Write cursor into the ring.
    write_pos: usize,
    /// Maximum addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// Smoothed Haas delay in frames.
    delay: Smoothed,
    /// Smoothed direct/delayed side blend.
    width: Smoothed,
    /// Smoothed overall side level.
    side_level: Smoothed,
}

impl HaasWidenerNode {
    /// Builds a Haas widener running at `sample_rate` Hz with a maximum
    /// addressable side delay of `max_delay_frames` (clamped to at least 1)
    /// frames, initialised from `params`. Initial values start settled.
    #[must_use]
    pub fn new(sample_rate: u32, max_delay_frames: usize, params: HaasWidenerParams) -> Self {
        let max = max_delay_frames.max(1);
        let ring_len = max + 2;
        let max_delay = max as Sample;
        let mut node = Self {
            sample_rate,
            ring: vec![0.0; ring_len],
            ring_len,
            write_pos: 0,
            max_delay,
            delay: Smoothed::new(0.0),
            width: Smoothed::new(0.0),
            side_level: Smoothed::new(0.0),
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

    /// Snapshots the current parameter targets.
    #[must_use]
    pub fn params(&self) -> HaasWidenerParams {
        let ms_per_frame = if self.sample_rate == 0 {
            0.0
        } else {
            1000.0 / self.sample_rate as Sample
        };
        HaasWidenerParams {
            delay_ms: self.delay.target() * ms_per_frame,
            width: self.width.target(),
            side_level: self.side_level.target(),
        }
    }

    /// Retargets every parameter, gliding with `ramp` so changes stay
    /// click-free. Ring state is preserved.
    pub fn set_params(&mut self, params: &HaasWidenerParams, ramp: Ramp) {
        self.apply_params(params, ramp);
    }

    /// Shared parameter application used by both the constructor and
    /// [`set_params`](Self::set_params).
    fn apply_params(&mut self, params: &HaasWidenerParams, ramp: Ramp) {
        let frames = finite(params.delay_ms).max(0.0) * self.sample_rate as Sample / 1000.0;
        self.delay.set_target(frames.clamp(0.0, self.max_delay), ramp);
        self.width
            .set_target(finite(params.width).clamp(0.0, MAX_WIDTH), ramp);
        self.side_level
            .set_target(finite(params.side_level).clamp(0.0, MAX_SIDE_LEVEL), ramp);
    }
}

/// Replaces a non-finite value with zero so a bad parameter cannot poison the
/// side path.
#[inline]
fn finite(x: Sample) -> Sample {
    if x.is_finite() { x } else { 0.0 }
}

impl AudioNode for HaasWidenerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let in_channels = input.channels();
        let channels = out_channels.min(in_channels);
        let frames = output.active_frames().min(input.active_frames());
        if frames == 0 || channels == 0 {
            return;
        }

        // Haas widening is a stereo concept: with fewer than two channels we
        // simply pass the signal through.
        if channels < 2 {
            for ch in 0..channels {
                let input_ch = input.channel(ch);
                let dst = output.channel_mut(ch);
                dst[..frames].copy_from_slice(&input_ch[..frames]);
            }
            return;
        }

        let ring_len = self.ring_len;
        let len_i = ring_len as isize;

        for f in 0..frames {
            let delay = self.delay.next_sample();
            let width = self.width.next_sample();
            let side_level = self.side_level.next_sample();

            let l = input.channel(0)[f];
            let r = input.channel(1)[f];
            let mid = 0.5 * (l + r);
            let side = 0.5 * (l - r);

            let w = self.write_pos;
            // Fractional read `delay` frames behind the write head.
            let read_pos = w as Sample - delay;
            let base = ops::floor(read_pos);
            let frac = read_pos - base;
            let base_i = base as isize;
            let i0 = base_i.rem_euclid(len_i) as usize;
            let i1 = (base_i + 1).rem_euclid(len_i) as usize;
            let side_delayed = lerp(self.ring[i0], self.ring[i1], frac);

            // Store the (undelayed) side for future reads.
            self.ring[w] = flush_denormal(side);

            let side_out = side_level * lerp(side, side_delayed, width);
            output.channel_mut(0)[f] = mid + side_out;
            output.channel_mut(1)[f] = mid - side_out;

            // Any channels beyond the front stereo pair pass through untouched.
            for ch in 2..channels {
                output.channel_mut(ch)[f] = input.channel(ch)[f];
            }

            self.write_pos = if w + 1 == ring_len { 0 } else { w + 1 };
        }
    }

    fn reset(&mut self) {
        for s in &mut self.ring {
            *s = 0.0;
        }
        self.write_pos = 0;
        self.delay = Smoothed::new(self.delay.target());
        self.width = Smoothed::new(self.width.target());
        self.side_level = Smoothed::new(self.side_level.target());
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

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    fn run(node: &mut HaasWidenerNode, input: AudioBuffer, out: AudioBuffer) -> AudioBuffer {
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
    fn width_zero_is_bit_exact_bypass() {
        let params = HaasWidenerParams {
            delay_ms: 10.0,
            width: 0.0,
            side_level: 1.0,
        };
        let mut node = HaasWidenerNode::new(SR, 2048, params);
        let mut input = stereo(64);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(i as Sample * 0.21);
        }
        for (i, s) in input.channel_mut(1).iter_mut().enumerate() {
            *s = ops::sin(i as Sample * 0.33);
        }
        let expected0: Vec<Sample> = input.channel(0).to_vec();
        let expected1: Vec<Sample> = input.channel(1).to_vec();
        let out = run(&mut node, input, stereo(64));
        for f in 0..64 {
            assert!((out.channel(0)[f] - expected0[f]).abs() < 1e-6);
            assert!((out.channel(1)[f] - expected1[f]).abs() < 1e-6);
        }
    }

    #[test]
    fn mono_sum_is_preserved() {
        // Delaying only the side must leave L + R = 2 * M untouched.
        let params = HaasWidenerParams {
            delay_ms: 8.0,
            width: 1.0,
            side_level: 1.5,
        };
        let mut node = HaasWidenerNode::new(SR, 2048, params);
        let mut input = stereo(256);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(i as Sample * 0.17);
        }
        for (i, s) in input.channel_mut(1).iter_mut().enumerate() {
            *s = 0.5 * ops::sin(i as Sample * 0.29);
        }
        let in_sum: Vec<Sample> = (0..256)
            .map(|i| input.channel(0)[i] + input.channel(1)[i])
            .collect();
        let out = run(&mut node, input, stereo(256));
        for (i, &expected) in in_sum.iter().enumerate() {
            let out_sum = out.channel(0)[i] + out.channel(1)[i];
            assert!(
                (out_sum - expected).abs() < 1e-5,
                "mono sum changed at {i}: {out_sum} vs {expected}",
            );
        }
    }

    #[test]
    fn delayed_side_appears_in_output() {
        // A pure side impulse (L = +a, R = -a) should produce a delayed
        // echo in the side after `delay` frames when width = 1.
        let params = HaasWidenerParams {
            delay_ms: 1000.0 * 10.0 / SR as Sample, // exactly 10 frames
            width: 1.0,
            side_level: 1.0,
        };
        let mut node = HaasWidenerNode::new(SR, 2048, params);
        let mut input = stereo(64);
        input.channel_mut(0)[0] = 0.5;
        input.channel_mut(1)[0] = -0.5; // side = 0.5, mid = 0
        let out = run(&mut node, input, stereo(64));
        // Side at frame 10 should be ~0.5 -> L = +0.5, R = -0.5.
        let side10 = 0.5 * (out.channel(0)[10] - out.channel(1)[10]);
        assert!((side10 - 0.5).abs() < 1e-4, "side echo missing: {side10}");
    }

    #[test]
    fn fractional_delay_interpolates_side() {
        let params = HaasWidenerParams {
            delay_ms: 1000.0 * 10.5 / SR as Sample,
            width: 1.0,
            side_level: 1.0,
        };
        let mut node = HaasWidenerNode::new(SR, 2048, params);
        let mut input = stereo(64);
        input.channel_mut(0)[0] = 0.5;
        input.channel_mut(1)[0] = -0.5;
        let out = run(&mut node, input, stereo(64));
        let side10 = 0.5 * (out.channel(0)[10] - out.channel(1)[10]);
        let side11 = 0.5 * (out.channel(0)[11] - out.channel(1)[11]);
        assert!((side10 - 0.25).abs() < 1e-3);
        assert!((side11 - 0.25).abs() < 1e-3);
    }

    #[test]
    fn side_level_scales_output() {
        let params = HaasWidenerParams {
            delay_ms: 0.0,
            width: 0.0,
            side_level: 2.0,
        };
        let mut node = HaasWidenerNode::new(SR, 2048, params);
        let mut input = stereo(16);
        input.channel_mut(0)[0] = 1.0; // mid = 0.5, side = 0.5
        input.channel_mut(1)[0] = 0.0;
        let out = run(&mut node, input, stereo(16));
        // side_out = 2 * 0.5 = 1.0, mid = 0.5 -> L = 1.5, R = -0.5.
        assert!((out.channel(0)[0] - 1.5).abs() < 1e-5);
        assert!((out.channel(1)[0] + 0.5).abs() < 1e-5);
    }

    #[test]
    fn width_is_clamped() {
        let params = HaasWidenerParams {
            delay_ms: 10.0,
            width: 5.0,
            side_level: 1.0,
        };
        let node = HaasWidenerNode::new(SR, 2048, params);
        assert!((node.params().width - MAX_WIDTH).abs() < 1e-6);
    }

    #[test]
    fn side_level_is_clamped() {
        let params = HaasWidenerParams {
            delay_ms: 10.0,
            width: 1.0,
            side_level: 99.0,
        };
        let node = HaasWidenerNode::new(SR, 2048, params);
        assert!((node.params().side_level - MAX_SIDE_LEVEL).abs() < 1e-6);
    }

    #[test]
    fn mono_input_passes_through() {
        let mut node = HaasWidenerNode::new(SR, 2048, HaasWidenerParams::default());
        let mut input = mono(16);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = i as Sample + 1.0;
        }
        let expected: Vec<Sample> = input.channel(0).to_vec();
        let out = run(&mut node, input, mono(16));
        assert_eq!(out.channel(0), expected.as_slice());
    }

    #[test]
    fn reset_is_bit_exact_reproducible() {
        let mut node = HaasWidenerNode::new(SR, 2048, HaasWidenerParams::default());
        let mut input = stereo(128);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = ops::sin(i as Sample * 0.3);
        }
        for (i, s) in input.channel_mut(1).iter_mut().enumerate() {
            *s = ops::sin(i as Sample * 0.41);
        }
        let first = run(&mut node, input.clone(), stereo(128));
        node.reset();
        let second = run(&mut node, input, stereo(128));
        assert_eq!(first.channel(0), second.channel(0));
        assert_eq!(first.channel(1), second.channel(1));
    }

    #[test]
    fn params_round_trip() {
        let params = HaasWidenerParams {
            delay_ms: 15.0,
            width: 0.7,
            side_level: 1.2,
        };
        let node = HaasWidenerNode::new(SR, 4096, params);
        let back = node.params();
        assert!((back.delay_ms - params.delay_ms).abs() < 0.05);
        assert!((back.width - params.width).abs() < 1e-6);
        assert!((back.side_level - params.side_level).abs() < 1e-6);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = HaasWidenerNode::new(SR, 2048, HaasWidenerParams::default());
        let mut input = stereo(4);
        input.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [stereo(4)];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn non_finite_param_stays_finite() {
        let params = HaasWidenerParams {
            delay_ms: Sample::NAN,
            width: Sample::INFINITY,
            side_level: Sample::NAN,
        };
        let mut node = HaasWidenerNode::new(SR, 2048, params);
        let mut input = stereo(32);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[0] = -1.0;
        let out = run(&mut node, input, stereo(32));
        for ch in 0..2 {
            for &s in out.channel(ch) {
                assert!(s.is_finite());
            }
        }
    }

    #[test]
    fn silent_input_stays_silent() {
        let mut node = HaasWidenerNode::new(SR, 2048, HaasWidenerParams::default());
        let out = run(&mut node, stereo(32), stereo(32));
        for ch in 0..2 {
            for &s in out.channel(ch) {
                assert!(s.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn set_params_updates_targets() {
        let mut node = HaasWidenerNode::new(SR, 4096, HaasWidenerParams::default());
        let params = HaasWidenerParams {
            delay_ms: 20.0,
            width: 0.25,
            side_level: 0.8,
        };
        node.set_params(&params, Ramp::Immediate);
        let back = node.params();
        assert!((back.width - 0.25).abs() < 1e-6);
        assert!((back.side_level - 0.8).abs() < 1e-6);
    }
}
