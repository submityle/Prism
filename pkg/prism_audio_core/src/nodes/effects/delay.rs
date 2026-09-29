//! Fractional delay line with feedback and independent wet/dry mixing.
//!
//! A delay is the foundation of echo, slap-back, and (with modulation) chorus /
//! flanger effects. This node keeps one pre-allocated ring buffer per channel
//! and reads a *fractional* number of samples behind the write head using
//! linear interpolation, so the delay time can be automated smoothly (and
//! swept for modulation effects) without stepping between integer taps.
//!
//! All storage is allocated at construction, so
//! [`DelayNode::process`](crate::graph::AudioNode::process) performs no
//! allocation, locking, or panicking and is safe on the audio callback thread.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};
use crate::param::{Ramp, Smoothed};

/// Largest stable feedback coefficient. Kept just below unity so a sustained
/// feedback path decays instead of building to infinity.
const MAX_FEEDBACK: Sample = 0.999;

/// A per-channel fractional delay line (input port 0 -> output port 0).
///
/// The output is `dry * input + wet * delayed`, where `delayed` is the signal
/// read `delay` frames in the past (linearly interpolated for fractional
/// delays) and the delayed signal is fed back into the line scaled by
/// `feedback`. Delay time, feedback, and the wet/dry gains are all
/// [`Smoothed`] so automation stays click-free.
#[derive(Debug, Clone)]
pub struct DelayNode {
    /// Sample rate in Hz, used to convert delay times expressed in seconds.
    sample_rate: u32,
    /// Ring length in frames (`max_delay + 2`), shared by every channel.
    ring_len: usize,
    /// One ring buffer per channel; each holds exactly `ring_len` samples.
    rings: Vec<Vec<Sample>>,
    /// Shared write cursor into every channel's ring buffer.
    write_pos: usize,
    /// Maximum addressable delay in frames (`ring_len - 2`).
    max_delay: Sample,
    /// Smoothed delay time in frames.
    delay: Smoothed,
    /// Smoothed feedback coefficient in `[0, MAX_FEEDBACK]`.
    feedback: Smoothed,
    /// Smoothed wet (processed) mix gain.
    wet: Smoothed,
    /// Smoothed dry (unprocessed input) mix gain.
    dry: Smoothed,
}

impl DelayNode {
    /// Builds a delay for `channels` channels running at `sample_rate` Hz with
    /// a maximum delay of `max_delay_frames` (clamped to at least 1) frames.
    ///
    /// The initial `delay_frames`, `feedback`, `wet`, and `dry` values start
    /// settled (no glide). `delay_frames` is clamped to `[1, max_delay_frames]`
    /// and `feedback` to `[0, 0.999]`.
    #[must_use]
    pub fn new(
        sample_rate: u32,
        channels: usize,
        max_delay_frames: usize,
        delay_frames: Sample,
        feedback: Sample,
        wet: Sample,
        dry: Sample,
    ) -> Self {
        let max = max_delay_frames.max(1);
        let ring_len = max + 2;
        let mut rings = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut ring = Vec::with_capacity(ring_len);
            ring.resize(ring_len, 0.0);
            rings.push(ring);
        }
        let max_delay = max as Sample;
        Self {
            sample_rate,
            ring_len,
            rings,
            write_pos: 0,
            max_delay,
            delay: Smoothed::new(delay_frames.clamp(1.0, max_delay)),
            feedback: Smoothed::new(feedback.clamp(0.0, MAX_FEEDBACK)),
            wet: Smoothed::new(wet),
            dry: Smoothed::new(dry),
        }
    }

    /// Returns the number of channels this delay processes.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.rings.len()
    }

    /// Returns the maximum addressable delay in frames.
    #[inline]
    #[must_use]
    pub fn max_delay_frames(&self) -> Sample {
        self.max_delay
    }

    /// Sets the delay time in frames, clamped to `[1, max_delay]`, gliding with
    /// `ramp` so sweeps do not click.
    #[inline]
    pub fn set_delay_frames(&mut self, frames: Sample, ramp: Ramp) {
        self.delay.set_target(frames.clamp(1.0, self.max_delay), ramp);
    }

    /// Sets the delay time in seconds (converted with the sample rate).
    #[inline]
    pub fn set_delay_seconds(&mut self, seconds: Sample, ramp: Ramp) {
        let frames = seconds.max(0.0) * self.sample_rate as Sample;
        self.set_delay_frames(frames, ramp);
    }

    /// Sets the feedback coefficient, clamped to `[0, 0.999]`.
    #[inline]
    pub fn set_feedback(&mut self, feedback: Sample, ramp: Ramp) {
        self.feedback.set_target(feedback.clamp(0.0, MAX_FEEDBACK), ramp);
    }

    /// Sets the wet (processed) mix gain.
    #[inline]
    pub fn set_wet(&mut self, wet: Sample, ramp: Ramp) {
        self.wet.set_target(wet, ramp);
    }

    /// Sets the dry (unprocessed input) mix gain.
    #[inline]
    pub fn set_dry(&mut self, dry: Sample, ramp: Ramp) {
        self.dry.set_target(dry, ramp);
    }
}

impl AudioNode for DelayNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.rings.len());
        let frames = output.active_frames();
        let ring_len = self.ring_len;
        let len_i = ring_len as isize;

        for f in 0..frames {
            // Advance every control parameter once per frame, then apply the
            // same values to all channels so the stereo image stays coherent.
            let delay = self.delay.next_sample();
            let feedback = self.feedback.next_sample();
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            let w = self.write_pos;
            // Fractional read position `delay` frames behind the write head.
            let read_pos = w as Sample - delay;
            let base = ops::floor(read_pos);
            let frac = read_pos - base;
            let base_i = base as isize;
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
        self.delay = Smoothed::new(self.delay.target());
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
    fn dry_passthrough_when_wet_zero() {
        // dry=1, wet=0, feedback=0 -> output must equal input exactly.
        let mut node = DelayNode::new(48_000, 1, 16, 4.0, 0.0, 0.0, 1.0);
        let mut input = mono(8);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = i as Sample + 1.0;
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(8)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(8), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    #[test]
    fn integer_delay_shifts_impulse() {
        let mut node = DelayNode::new(48_000, 1, 16, 4.0, 0.0, 1.0, 0.0);
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(16)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(16), &mut io);
        let out = outputs[0].channel(0);
        assert!((out[4] - 1.0).abs() < 1e-6, "impulse should land at frame 4: {out:?}");
        for (i, &s) in out.iter().enumerate() {
            if i != 4 {
                assert!(s.abs() < 1e-6, "unexpected energy at frame {i}: {s}");
            }
        }
    }

    #[test]
    fn feedback_produces_decaying_echoes() {
        let mut node = DelayNode::new(48_000, 1, 16, 4.0, 0.5, 1.0, 0.0);
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(16)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(16), &mut io);
        let out = outputs[0].channel(0);
        // Echoes at frames 4, 8, 12 with 0.5^n amplitude.
        assert!((out[4] - 1.0).abs() < 1e-6, "{out:?}");
        assert!((out[8] - 0.5).abs() < 1e-6, "{out:?}");
        assert!((out[12] - 0.25).abs() < 1e-6, "{out:?}");
        for &s in out {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn fractional_delay_interpolates() {
        let mut node = DelayNode::new(48_000, 1, 16, 4.5, 0.0, 1.0, 0.0);
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(16)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(16), &mut io);
        let out = outputs[0].channel(0);
        // A 4.5-frame delay spreads the impulse evenly across frames 4 and 5.
        assert!(out[4] > 0.4 && out[4] < 0.6, "{out:?}");
        assert!(out[5] > 0.4 && out[5] < 0.6, "{out:?}");
    }

    #[test]
    fn reset_clears_tail() {
        let mut node = DelayNode::new(48_000, 1, 16, 4.0, 0.9, 1.0, 0.0);
        let mut input = mono(16);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(16)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(16), &mut io);
        node.reset();
        let silence = [mono(16)];
        let mut outputs2 = [mono(16)];
        let mut io2 = ProcessIo::new(&silence, &mut outputs2);
        node.process(&ctx(16), &mut io2);
        for &s in outputs2[0].channel(0) {
            assert!(s.abs() < 1e-9, "delay tail not cleared: {s}");
        }
    }

    #[test]
    fn stereo_channels_are_independent() {
        let mut node = DelayNode::new(48_000, 2, 16, 4.0, 0.0, 1.0, 0.0);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 16);
        input.channel_mut(0)[0] = 1.0;
        input.channel_mut(1)[2] = 1.0;
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, 16)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(16), &mut io);
        assert!((outputs[0].channel(0)[4] - 1.0).abs() < 1e-6);
        assert!((outputs[0].channel(1)[6] - 1.0).abs() < 1e-6);
    }
}
