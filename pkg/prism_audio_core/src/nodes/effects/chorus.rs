//! Multi-voice chorus: several LFO-modulated fractional delay taps summed with
//! the dry signal to create a shimmering, ensemble-like thickening.
//!
//! Each voice reads the input a few milliseconds in the past, with its delay
//! time swept by a sine LFO. The voices share one frequency but are spread
//! evenly around the LFO cycle so their pitch wobbles decorrelate, widening the
//! sound. Unlike a flanger there is no feedback, so the effect is subtle and
//! comb-free.
//!
//! Storage is fixed at construction, so
//! [`ChorusNode::process`](crate::graph::AudioNode::process) is real-time safe.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};
use crate::modulation::{Lfo, LfoWaveform};
use crate::param::{Ramp, Smoothed};

/// Number of detuned voices summed by the chorus.
const VOICES: usize = 3;

/// Construction parameters for a [`ChorusNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ChorusParams {
    /// Center delay time in milliseconds (typically 15-35 ms).
    pub base_delay_ms: Sample,
    /// Peak LFO delay excursion in milliseconds (typically 2-10 ms).
    pub depth_ms: Sample,
    /// LFO rate in Hz (typically 0.2-2 Hz).
    pub rate_hz: Sample,
    /// Wet (processed) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed) mix gain.
    pub dry: Sample,
}

impl Default for ChorusParams {
    fn default() -> Self {
        Self {
            base_delay_ms: 22.0,
            depth_ms: 6.0,
            rate_hz: 0.6,
            wet: 0.5,
            dry: 1.0,
        }
    }
}

/// A three-voice chorus (input port 0 -> output port 0).
///
/// The output is `dry * input + wet * (mean of the three modulated taps)`. The
/// base delay, modulation depth, and wet/dry gains are [`Smoothed`] so
/// parameter automation stays click-free.
#[derive(Debug, Clone)]
pub struct ChorusNode {
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
    /// Per-voice modulation LFOs (phase-spread around the cycle).
    lfos: [Lfo; VOICES],
    /// Smoothed center delay in frames.
    base_delay: Smoothed,
    /// Smoothed modulation depth in frames.
    depth: Smoothed,
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

impl ChorusNode {
    /// Builds a chorus for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: ChorusParams) -> Self {
        let channels = channels.max(1);
        let base = ms_to_frames(params.base_delay_ms, sample_rate);
        let depth = ms_to_frames(params.depth_ms, sample_rate);
        // The ring must hold the deepest possible tap plus interpolation slack.
        let max_delay = (base + depth).max(1.0);
        let ring_len = (ops::round(max_delay) as usize) + 2;

        let mut rings = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut ring = Vec::with_capacity(ring_len);
            ring.resize(ring_len, 0.0);
            rings.push(ring);
        }

        let mut lfos = [Lfo::new(sample_rate, params.rate_hz, LfoWaveform::Sine); VOICES];
        for (i, lfo) in lfos.iter_mut().enumerate() {
            lfo.set_phase(i as Sample / VOICES as Sample);
        }

        Self {
            sample_rate,
            ring_len,
            rings,
            write_pos: 0,
            max_delay: (ring_len - 2) as Sample,
            lfos,
            base_delay: Smoothed::new(base.clamp(1.0, (ring_len - 2) as Sample)),
            depth: Smoothed::new(depth),
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

    /// Sets the LFO rate in Hz for every voice.
    #[inline]
    pub fn set_rate_hz(&mut self, rate_hz: Sample) {
        for lfo in &mut self.lfos {
            lfo.set_frequency(self.sample_rate, rate_hz);
        }
    }

    /// Sets the modulation depth in milliseconds.
    #[inline]
    pub fn set_depth_ms(&mut self, depth_ms: Sample, ramp: Ramp) {
        self.depth
            .set_target(ms_to_frames(depth_ms, self.sample_rate), ramp);
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

    /// Reads channel `ch` at `delay` fractional frames behind the write head.
    #[inline]
    fn read_tap(&self, ch: usize, write_pos: usize, delay: Sample) -> Sample {
        let len_i = self.ring_len as isize;
        let read_pos = write_pos as Sample - delay;
        let base = ops::floor(read_pos);
        let frac = read_pos - base;
        let base_i = base as isize;
        let i0 = base_i.rem_euclid(len_i) as usize;
        let i1 = (base_i + 1).rem_euclid(len_i) as usize;
        let ring = &self.rings[ch];
        lerp(ring[i0], ring[i1], frac)
    }
}

impl AudioNode for ChorusNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.rings.len());
        let frames = output.active_frames();
        let ring_len = self.ring_len;
        let inv_voices = 1.0 / VOICES as Sample;

        for f in 0..frames {
            let base = self.base_delay.next_sample();
            let depth = self.depth.next_sample();
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            // Advance every voice LFO once per frame so all channels agree.
            let mut mod_delay = [0.0 as Sample; VOICES];
            for (v, lfo) in self.lfos.iter_mut().enumerate() {
                let d = base + depth * lfo.next_sample();
                mod_delay[v] = d.clamp(1.0, self.max_delay);
            }

            let w = self.write_pos;
            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let mut wet_sum = 0.0;
                for &d in &mod_delay {
                    wet_sum += self.read_tap(ch, w, d);
                }
                output.channel_mut(ch)[f] = dry * x + wet * wet_sum * inv_voices;
                self.rings[ch][w] = flush_denormal(x);
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
        for (i, lfo) in self.lfos.iter_mut().enumerate() {
            lfo.reset();
            lfo.set_phase(i as Sample / VOICES as Sample);
        }
        self.base_delay = Smoothed::new(self.base_delay.target());
        self.depth = Smoothed::new(self.depth.target());
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
        let params = ChorusParams {
            wet: 0.0,
            dry: 1.0,
            ..ChorusParams::default()
        };
        let mut node = ChorusNode::new(48_000, 1, params);
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
    fn output_is_finite_and_bounded() {
        let mut node = ChorusNode::new(48_000, 2, ChorusParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 512);
        for ch in 0..2 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                let phase = 2.0 * core::f32::consts::PI * 0.01 * i as Sample;
                *s = 0.7 * ops::sin(phase);
            }
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, 512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        for ch in 0..2 {
            for &y in outputs[0].channel(ch) {
                assert!(y.is_finite() && y.abs() < 4.0, "{y}");
            }
        }
    }

    #[test]
    fn wet_signal_is_delayed_copy() {
        // With wet only and a slow LFO, the wet output should be a delayed,
        // non-zero echo of an initial impulse (not silence, not the dry click).
        let params = ChorusParams {
            depth_ms: 0.0,
            rate_hz: 0.0,
            wet: 1.0,
            dry: 0.0,
            base_delay_ms: 10.0,
        };
        let mut node = ChorusNode::new(48_000, 1, params);
        let mut input = mono(1024);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(1024)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(1024), &mut io);
        let out = outputs[0].channel(0);
        assert!(out[0].abs() < 1e-6, "dry should be muted: {}", out[0]);
        let delay = (0.010 * 48_000.0) as usize; // 480 frames
        let energy: Sample = out[(delay - 2)..=(delay + 2)].iter().map(|v| v.abs()).sum();
        assert!(energy > 0.5, "expected delayed impulse near frame {delay}");
    }

    #[test]
    fn reset_clears_tail() {
        let mut node = ChorusNode::new(48_000, 1, ChorusParams::default());
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
