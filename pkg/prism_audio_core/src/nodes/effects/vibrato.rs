//! Vibrato: a single LFO-swept fractional delay that produces periodic pitch
//! modulation.
//!
//! A vibrato reads the input from a delay line whose read distance is swept by
//! a low-frequency oscillator. Because the read pointer accelerates and
//! decelerates relative to the write pointer, the resampled output rises and
//! falls in pitch, giving the characteristic "wavering" tone of a singer's
//! vibrato or a guitar tremolo arm. Unlike a [`ChorusNode`](super::chorus::ChorusNode)
//! it uses a single tap and is normally fully wet (no dry copy blended in), and
//! unlike a [`FlangerNode`](super::flanger::FlangerNode) it has no feedback, so
//! it produces no comb coloration - only pitch movement.
//!
//! The delay ring is sized at construction from the requested depth, so
//! [`VibratoNode::process`](crate::graph::AudioNode::process) performs no
//! allocation and is real-time safe.
//!
//! # Signal model
//!
//! The instantaneous read delay in frames is
//!
//! `d[n] = depth * (1 + lfo[n])`,
//!
//! swept between `0` and `2 * depth` by a bipolar LFO in `[-1, 1]`, and clamped
//! to `[1, max_delay]` so the fractional read always straddles two valid ring
//! samples. The output is a linear-interpolated tap blended with the dry input
//! by `mix`:
//!
//! `y[n] = (1 - mix) * x[n] + mix * tap(d[n])`.
//!
//! One LFO drives every channel from the same phase, so a mono source fed to
//! several channels stays phase-coherent.
//!
//! # Provenance
//!
//! This is the textbook modulated-delay-line vibrato described in U. Zoelzer,
//! "DAFX: Digital Audio Effects", and standard DSP literature: a fractional
//! delay swept by an LFO. It contains **no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented signal-processing model.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal, lerp};
use crate::modulation::{Lfo, LfoWaveform};
use crate::param::{Ramp, Smoothed};

/// Construction parameters for a [`VibratoNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VibratoParams {
    /// LFO rate in Hz (typically 4-8 Hz for musical vibrato).
    pub rate_hz: Sample,
    /// Peak delay excursion in milliseconds (deeper means wider pitch swing).
    pub depth_ms: Sample,
    /// Wet/dry blend in `[0, 1]`: `0` is the untouched input, `1` is fully
    /// pitch-modulated.
    pub mix: Sample,
    /// LFO waveform driving the sweep.
    pub waveform: LfoWaveform,
}

impl Default for VibratoParams {
    fn default() -> Self {
        Self {
            rate_hz: 5.0,
            depth_ms: 2.0,
            mix: 1.0,
            waveform: LfoWaveform::Sine,
        }
    }
}

/// Converts milliseconds to (fractional) frames at `sample_rate`.
#[inline]
fn ms_to_frames(ms: Sample, sample_rate: u32) -> Sample {
    ms.max(0.0) * (sample_rate as Sample) * 0.001
}

/// A single-tap LFO-swept fractional delay vibrato (input port 0 -> output
/// port 0).
///
/// The read delay sweeps around a center equal to the depth, so the tap never
/// runs ahead of the write head. The depth and wet/dry [`mix`](VibratoParams::mix)
/// are [`Smoothed`] so automation stays click-free.
#[derive(Debug, Clone)]
pub struct VibratoNode {
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
    /// Sweep LFO shared by all channels.
    lfo: Lfo,
    /// Smoothed modulation depth in frames.
    depth: Smoothed,
    /// Smoothed wet/dry blend in `[0, 1]`.
    mix: Smoothed,
}

impl VibratoNode {
    /// Builds a vibrato for `channels` channels at `sample_rate` Hz.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_core::nodes::effects::{VibratoNode, VibratoParams};
    ///
    /// let node = VibratoNode::new(48_000, 2, VibratoParams::default());
    /// assert_eq!(node.channels(), 2);
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: VibratoParams) -> Self {
        let channels = channels.max(1);
        let depth = ms_to_frames(params.depth_ms, sample_rate);
        // The ring must hold the deepest sweep (2 * depth) plus interpolation
        // slack.
        let span = (2.0 * depth).max(1.0);
        let ring_len = (ops::round(span) as usize) + 2;

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
            lfo: Lfo::new(sample_rate, params.rate_hz, params.waveform),
            depth: Smoothed::new(depth),
            mix: Smoothed::new(params.mix.clamp(0.0, 1.0)),
        }
    }

    /// Returns the number of channels processed.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.rings.len()
    }

    /// Sets the LFO sweep rate in Hz.
    #[inline]
    pub fn set_rate_hz(&mut self, rate_hz: Sample) {
        self.lfo.set_frequency(self.sample_rate, rate_hz);
    }

    /// Sets the LFO waveform driving the sweep.
    #[inline]
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        self.lfo.set_waveform(waveform);
    }

    /// Sets the modulation depth in milliseconds.
    ///
    /// The target is clamped so the swept read stays inside the ring allocated
    /// at construction.
    #[inline]
    pub fn set_depth_ms(&mut self, depth_ms: Sample, ramp: Ramp) {
        let frames = ms_to_frames(depth_ms, self.sample_rate);
        // Half the ring span is the largest center depth that keeps 2 * depth
        // addressable.
        let max_center = 0.5 * self.max_delay;
        self.depth.set_target(frames.min(max_center), ramp);
    }

    /// Sets the wet/dry blend in `[0, 1]`.
    #[inline]
    pub fn set_mix(&mut self, mix: Sample, ramp: Ramp) {
        self.mix.set_target(mix.clamp(0.0, 1.0), ramp);
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

impl AudioNode for VibratoNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.rings.len());
        let frames = output.active_frames();
        let ring_len = self.ring_len;

        for f in 0..frames {
            let depth = self.depth.next_sample();
            let mix = self.mix.next_sample();
            // Advance the shared LFO once per frame so every channel agrees.
            let sweep = self.lfo.next_sample();
            let delay = (depth * (1.0 + sweep)).clamp(1.0, self.max_delay);

            let w = self.write_pos;
            for ch in 0..channels {
                let raw = input.channel(ch)[f];
                let x = if raw.is_finite() { raw } else { 0.0 };
                let wet = self.read_tap(ch, w, delay);
                output.channel_mut(ch)[f] = (1.0 - mix) * x + mix * wet;
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
        self.lfo.reset();
        self.depth = Smoothed::new(self.depth.target());
        self.mix = Smoothed::new(self.mix.target());
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

    fn ramp() -> AudioBuffer {
        let mut b = mono(128);
        for (i, s) in b.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample) * 0.01;
        }
        b
    }

    #[test]
    fn mix_zero_is_passthrough() {
        let params = VibratoParams {
            mix: 0.0,
            ..VibratoParams::default()
        };
        let mut node = VibratoNode::new(48_000, 1, params);
        let input = ramp();
        let inputs = [input.clone()];
        let mut outputs = [mono(128)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(128), &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    #[test]
    fn output_is_finite_and_bounded() {
        let mut node = VibratoNode::new(48_000, 2, VibratoParams::default());
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
                assert!(y.is_finite() && y.abs() < 2.0, "{y}");
            }
        }
    }

    #[test]
    fn static_lfo_is_a_fixed_delay() {
        // Zero rate with a sine LFO keeps the sweep at 0, so the read delay is a
        // constant `depth` frames. An impulse reappears delayed by that amount.
        let params = VibratoParams {
            rate_hz: 0.0,
            depth_ms: 2.0, // 96 frames at 48 kHz
            mix: 1.0,
            waveform: LfoWaveform::Sine,
        };
        let mut node = VibratoNode::new(48_000, 1, params);
        let mut input = mono(1024);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(1024)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(1024), &mut io);
        let out = outputs[0].channel(0);
        assert!(out[0].abs() < 1e-6, "no dry click when fully wet: {}", out[0]);
        let delay = 96usize;
        let energy: Sample = out[(delay - 2)..=(delay + 2)].iter().map(|v| v.abs()).sum();
        assert!(energy > 0.5, "expected delayed impulse near frame {delay}");
    }

    #[test]
    fn mix_blends_dry_and_wet() {
        // A half mix on a static delay is exactly half dry plus half the
        // delayed impulse.
        let params = VibratoParams {
            rate_hz: 0.0,
            depth_ms: 2.0,
            mix: 0.5,
            waveform: LfoWaveform::Sine,
        };
        let mut node = VibratoNode::new(48_000, 1, params);
        let mut input = mono(1024);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(1024)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(1024), &mut io);
        let out = outputs[0].channel(0);
        // Dry half of the impulse survives at frame 0.
        assert!((out[0] - 0.5).abs() < 1e-4, "dry half missing: {}", out[0]);
        let delay = 96usize;
        let energy: Sample = out[(delay - 2)..=(delay + 2)].iter().map(|v| v.abs()).sum();
        assert!(energy > 0.25, "wet half missing near frame {delay}");
    }

    #[test]
    fn modulation_changes_the_signal() {
        // A genuine sweep on a steady tone must alter the samples relative to
        // the dry input (pitch wobble), unlike a flat pass-through.
        let params = VibratoParams {
            rate_hz: 6.0,
            depth_ms: 3.0,
            mix: 1.0,
            waveform: LfoWaveform::Sine,
        };
        let mut node = VibratoNode::new(48_000, 1, params);
        let mut input = mono(2048);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let phase = 2.0 * core::f32::consts::PI * 440.0 * (i as Sample) / 48_000.0;
            *s = ops::sin(phase);
        }
        let inputs = [input.clone()];
        let mut outputs = [mono(2048)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(2048), &mut io);
        let out = outputs[0].channel(0);
        // Compare well past the initial fill so the ring is primed.
        let mut diff = 0.0 as Sample;
        for i in 512..2048 {
            diff += (out[i] - input.channel(0)[i]).abs();
        }
        assert!(diff > 1.0, "modulated output too close to dry: {diff}");
    }

    #[test]
    fn channels_are_phase_coherent() {
        // One LFO drives every channel, so identical inputs give identical
        // outputs.
        let mut node = VibratoNode::new(48_000, 2, VibratoParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 512);
        for ch in 0..2 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                let phase = 2.0 * core::f32::consts::PI * 220.0 * (i as Sample) / 48_000.0;
                *s = ops::sin(phase);
            }
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, 512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        assert_eq!(outputs[0].channel(0), outputs[0].channel(1));
    }

    #[test]
    fn negative_depth_is_clamped_to_zero() {
        // A negative depth becomes zero frames, so the read clamps to a 1-frame
        // delay and the output is a near-copy (no panic, finite).
        let params = VibratoParams {
            depth_ms: -5.0,
            rate_hz: 5.0,
            mix: 1.0,
            waveform: LfoWaveform::Sine,
        };
        let mut node = VibratoNode::new(48_000, 1, params);
        let input = ramp();
        let inputs = [input];
        let mut outputs = [mono(128)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(128), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite(), "{y}");
        }
    }

    #[test]
    fn mix_is_clamped() {
        let mut node = VibratoNode::new(
            48_000,
            1,
            VibratoParams {
                mix: 4.0,
                ..VibratoParams::default()
            },
        );
        // A mix above one is clamped so a static delay never amplifies past the
        // wet tap.
        node.set_rate_hz(0.0);
        let mut input = mono(512);
        input.channel_mut(0)[0] = 1.0;
        let inputs = [input];
        let mut outputs = [mono(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(512), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite() && y.abs() <= 1.0 + 1e-4, "{y}");
        }
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut node = VibratoNode::new(48_000, 1, VibratoParams::default());
        let mut input = mono(256);
        input.channel_mut(0)[0] = Sample::NAN;
        input.channel_mut(0)[1] = Sample::INFINITY;
        let inputs = [input];
        let mut outputs = [mono(256)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(256), &mut io);
        for &y in outputs[0].channel(0) {
            assert!(y.is_finite(), "non-finite leaked: {y}");
        }
    }

    #[test]
    fn reset_reproduces_output() {
        let mut node = VibratoNode::new(48_000, 1, VibratoParams::default());
        let mut input = mono(256);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let phase = 2.0 * core::f32::consts::PI * 330.0 * (i as Sample) / 48_000.0;
            *s = ops::sin(phase);
        }
        let inputs = [input];

        let mut out_a = [mono(256)];
        let mut io_a = ProcessIo::new(&inputs, &mut out_a);
        node.process(&ctx(256), &mut io_a);

        node.reset();
        let mut out_b = [mono(256)];
        let mut io_b = ProcessIo::new(&inputs, &mut out_b);
        node.process(&ctx(256), &mut io_b);

        assert_eq!(out_a[0].channel(0), out_b[0].channel(0));
    }

    #[test]
    fn zero_frames_does_not_panic() {
        let mut node = VibratoNode::new(48_000, 1, VibratoParams::default());
        let inputs = [mono(1)];
        let mut outputs = [mono(1)];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
    }

    #[test]
    fn deeper_depth_widens_the_sweep() {
        // A deeper vibrato departs further from the dry tone than a shallow one.
        fn deviation(depth_ms: Sample) -> Sample {
            let params = VibratoParams {
                rate_hz: 6.0,
                depth_ms,
                mix: 1.0,
                waveform: LfoWaveform::Sine,
            };
            let mut node = VibratoNode::new(48_000, 1, params);
            let mut input = mono(2048);
            for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
                let phase = 2.0 * core::f32::consts::PI * 440.0 * (i as Sample) / 48_000.0;
                *s = ops::sin(phase);
            }
            let inputs = [input.clone()];
            let mut outputs = [mono(2048)];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx(2048), &mut io);
            let out = outputs[0].channel(0);
            let mut d = 0.0 as Sample;
            for i in 512..2048 {
                d += (out[i] - input.channel(0)[i]).abs();
            }
            d
        }
        assert!(deviation(4.0) > deviation(1.0));
    }
}
