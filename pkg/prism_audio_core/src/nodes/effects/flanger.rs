//! Flanger: a single short LFO-swept delay tap with feedback, producing the
//! classic sweeping comb-filter "jet plane" effect.
//!
//! A flanger differs from a chorus in two ways: the delay is much shorter
//! (fractions of a millisecond up to ~10 ms) and the delayed signal is fed
//! back into the line. The feedback deepens the comb notches, and sweeping the
//! delay with an LFO drags those notches across the spectrum.
//!
//! Storage is fixed at construction, so
//! [`FlangerNode::process`](crate::graph::AudioNode::process) is real-time safe.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};
use crate::modulation::{Lfo, LfoWaveform};
use crate::param::{Ramp, Smoothed};

/// Largest stable feedback magnitude (kept below unity for a decaying comb).
const MAX_FEEDBACK: Sample = 0.98;

/// Construction parameters for a [`FlangerNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FlangerParams {
    /// Center delay time in milliseconds (typically 1-5 ms).
    pub base_delay_ms: Sample,
    /// Peak LFO delay excursion in milliseconds (typically 0.5-4 ms).
    pub depth_ms: Sample,
    /// LFO rate in Hz (typically 0.1-2 Hz).
    pub rate_hz: Sample,
    /// Feedback coefficient in `[-0.98, 0.98]`; negative inverts the comb.
    pub feedback: Sample,
    /// Wet (processed) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed) mix gain.
    pub dry: Sample,
}

impl Default for FlangerParams {
    fn default() -> Self {
        Self {
            base_delay_ms: 2.0,
            depth_ms: 1.5,
            rate_hz: 0.4,
            feedback: 0.5,
            wet: 0.5,
            dry: 1.0,
        }
    }
}

/// A mono/multi-channel flanger (input port 0 -> output port 0).
///
/// The delayed tap is `delayed`, the ring stores `input + feedback * delayed`,
/// and the output is `dry * input + wet * delayed`. Depth, feedback, and
/// wet/dry gains are [`Smoothed`] for click-free automation.
#[derive(Debug, Clone)]
pub struct FlangerNode {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Ring length in frames, shared by every channel.
    ring_len: usize,
    /// One delay ring per channel.
    rings: Vec<Vec<Sample>>,
    /// Shared write cursor into every channel's ring.
    write_pos: usize,
    /// Largest addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// Shared modulation LFO.
    lfo: Lfo,
    /// Smoothed center delay in frames.
    base_delay: Smoothed,
    /// Smoothed modulation depth in frames.
    depth: Smoothed,
    /// Smoothed feedback coefficient.
    feedback: Smoothed,
    /// Smoothed wet mix gain.
    wet: Smoothed,
    /// Smoothed dry mix gain.
    dry: Smoothed,
}

/// Converts milliseconds to (fractional) frames at `sample_rate`.
#[inline]
fn ms_to_frames(ms: Sample, sample_rate: u32) -> Sample {
    ms.max(0.0) * (sample_rate as Sample) * 0.001
}

impl FlangerNode {
    /// Builds a flanger for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: FlangerParams) -> Self {
        let channels = channels.max(1);
        let base = ms_to_frames(params.base_delay_ms, sample_rate);
        let depth = ms_to_frames(params.depth_ms, sample_rate);
        let max_delay = (base + depth).max(1.0);
        let ring_len = (ops::round(max_delay) as usize) + 2;

        let mut rings = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut ring = Vec::with_capacity(ring_len);
            ring.resize(ring_len, 0.0);
            rings.push(ring);
        }

        Self {
            sample_rate,
            ring_len,
            rings,
            write_pos: 0,
            max_delay: (ring_len - 2) as Sample,
            lfo: Lfo::new(sample_rate, params.rate_hz, LfoWaveform::Sine),
            base_delay: Smoothed::new(base.clamp(1.0, (ring_len - 2) as Sample)),
            depth: Smoothed::new(depth),
            feedback: Smoothed::new(params.feedback.clamp(-MAX_FEEDBACK, MAX_FEEDBACK)),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
        }
    }

    /// Returns the number of channels processed.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.rings.len()
    }

    /// Sets the LFO rate in Hz.
    #[inline]
    pub fn set_rate_hz(&mut self, rate_hz: Sample) {
        self.lfo.set_frequency(self.sample_rate, rate_hz);
    }

    /// Sets the modulation depth in milliseconds.
    #[inline]
    pub fn set_depth_ms(&mut self, depth_ms: Sample, ramp: Ramp) {
        self.depth
            .set_target(ms_to_frames(depth_ms, self.sample_rate), ramp);
    }

    /// Sets the feedback coefficient, clamped to `[-0.98, 0.98]`.
    #[inline]
    pub fn set_feedback(&mut self, feedback: Sample, ramp: Ramp) {
        self.feedback
            .set_target(feedback.clamp(-MAX_FEEDBACK, MAX_FEEDBACK), ramp);
    }

    /// Sets the wet mix gain.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry mix gain.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }
}

impl AudioNode for FlangerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.rings.len());
        let frames = output.active_frames();
        let ring_len = self.ring_len;
        let len_i = ring_len as isize;

        for f in 0..frames {
            let base = self.base_delay.next_sample();
            let depth = self.depth.next_sample();
            let feedback = self.feedback.next_sample();
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            // One LFO tick per frame, shared across channels.
            let delay = (base + depth * self.lfo.next_sample()).clamp(1.0, self.max_delay);
            let w = self.write_pos;
            let read_pos = w as Sample - delay;
            let floor = ops::floor(read_pos);
            let frac = read_pos - floor;
            let base_i = floor as isize;
            let i0 = base_i.rem_euclid(len_i) as usize;
            let i1 = (base_i + 1).rem_euclid(len_i) as usize;

            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let ring = &mut self.rings[ch];
                let delayed = lerp(ring[i0], ring[i1], frac);
                output.channel_mut(ch)[f] = dry * x + wet * delayed;
                ring[w] = flush_denormal(x + feedback * delayed);
            }

            self.write_pos = if w + 1 == ring_len { 0 } else { w + 1 };
        }
    }

    fn reset(&mut self) {
        for ring in &mut self.rings {
            for s in ring.iter_mut() {
                *s = 0.0;
            }
        }
        self.write_pos = 0;
        self.lfo.reset();
        self.base_delay = Smoothed::new(self.base_delay.target());
        self.depth = Smoothed::new(self.depth.target());
        self.feedback = Smoothed::new(self.feedback.target());
        self.wet = Smoothed::new(self.wet.target());
        self.dry = Smoothed::new(self.dry.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        }
    }

    fn mono(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Mono, frames)
    }

    #[test]
    fn dry_only_is_passthrough() {
        let params = FlangerParams {
            wet: 0.0,
            dry: 1.0,
            feedback: 0.0,
            ..FlangerParams::default()
        };
        let mut node = FlangerNode::new(48_000, 1, params);
        let mut input = mono(64);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) * 0.01;
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(64)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(64), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    #[test]
    fn feedback_stays_bounded() {
        let params = FlangerParams {
            feedback: 0.98,
            depth_ms: 0.0,
            rate_hz: 0.0,
            wet: 1.0,
            dry: 1.0,
            base_delay_ms: 3.0,
        };
        let mut node = FlangerNode::new(48_000, 1, params);
        let mut input = mono(4096);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let phase = 2.0 * core::f32::consts::PI * 0.05 * i as Sample;
            *s = 0.5 * ops::sin(phase);
        }
        let inputs = [input];
        let mut outputs = [mono(4096)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(4096), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite() && y.abs() < 50.0, "runaway feedback: {y}");
        }
    }

    #[test]
    fn produces_delayed_echo() {
        let params = FlangerParams {
            feedback: 0.0,
            depth_ms: 0.0,
            rate_hz: 0.0,
            wet: 1.0,
            dry: 0.0,
            base_delay_ms: 2.0,
        };
        let mut node = FlangerNode::new(48_000, 1, params);
        let mut input = mono(512);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        let delay = (0.002 * 48_000.0) as usize; // 96 frames
        let out = outputs[0].channel(0);
        assert!((out[delay] - 1.0).abs() < 1e-3, "impulse at {delay}: {}", out[delay]);
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn reset_clears_tail() {
        let mut node = FlangerNode::new(48_000, 1, FlangerParams::default());
        let mut input = mono(256);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        node.reset();
        let silence = [mono(256)];
        let mut outputs2 = [mono(256)];
        let mut io2 = ProcessIo::new(&silence, &mut outputs2);
        node.process(&ctx(256), &mut io2);
        for &y in outputs2[0].channel(0) {
            assert!(y.abs() < 1e-9, "tail not cleared: {y}");
        }
    }
}
