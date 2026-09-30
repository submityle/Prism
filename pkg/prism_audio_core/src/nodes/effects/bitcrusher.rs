//! Bit-crusher: quantization and sample-rate reduction ("lo-fi" effect).
//!
//! A bit-crusher deliberately degrades a signal in two independent ways:
//!
//! - **Bit-depth reduction** rounds every sample onto a coarse amplitude grid,
//!   as if it were stored with only a handful of bits. The rounding error is a
//!   signal-dependent quantization distortion that adds gritty harmonics.
//! - **Sample-rate reduction** holds each captured sample for several output
//!   frames (a zero-order sample-and-hold decimator), as if the signal had been
//!   sampled at a lower rate. The stair-stepping folds high frequencies down
//!   into aliased partials.
//!
//! Both are the deliberately "wrong" operations a clean converter tries to
//! avoid; used on purpose they give the crunchy, degraded character of early
//! samplers and chiptune hardware. A wet/dry `mix` blends the crushed signal
//! back against the untouched input.
//!
//! # The model
//!
//! With `bits` effective bits the amplitude grid has `2^bits` codes spanning
//! `[-1, 1)` with a step of `2 / 2^bits`; a sample `x` maps to
//! `round(x / step) * step`, with the integer code clamped to the signed
//! two's-complement range `[-2^(bits-1), 2^(bits-1) - 1]` so the code count is
//! exactly `2^bits` (as in real converter hardware). A `downsample` factor
//! `d >= 1` captures a fresh
//! (quantized) sample once every `d` frames via a phase accumulator advanced by
//! `1 / d` each frame, holding the last capture in between. Fractional `bits`
//! and `d` are allowed for smooth, continuously-morphable degradation.
//!
//! # Real-time contract
//!
//! Per-channel hold state is allocated once in [`BitcrusherNode::new`].
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length blocks
//! degrade gracefully.
//!
//! # Provenance
//!
//! Bit-depth quantization (rounding to a coarse amplitude grid) and zero-order
//! sample-and-hold decimation are elementary lo-fi effects described in the
//! standard literature (e.g. Zoelzer, "DAFX: Digital Audio Effects"). This
//! module reuses only this crate's own [`Sample`] and denormal-flushing
//! primitives. It contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**; it is implemented
//! purely from that publicly documented theory.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Smallest effective bit depth (a two-level, hard 1-bit grid).
pub const MIN_BIT_DEPTH: Sample = 1.0;
/// Largest effective bit depth (finer than 24-bit audio is transparent).
pub const MAX_BIT_DEPTH: Sample = 24.0;

/// Configuration for a [`BitcrusherNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BitcrusherParams {
    /// Effective bit depth in `[MIN_BIT_DEPTH, MAX_BIT_DEPTH]`; fractional
    /// values are allowed. Higher is cleaner.
    pub bit_depth: Sample,
    /// Sample-rate reduction factor `>= 1`: capture one fresh sample every
    /// `downsample` frames. `1` disables decimation.
    pub downsample: Sample,
    /// Wet/dry blend in `[0, 1]`: `0` is the untouched input, `1` fully crushed.
    pub mix: Sample,
}

impl Default for BitcrusherParams {
    fn default() -> Self {
        Self {
            bit_depth: 8.0,
            downsample: 1.0,
            mix: 1.0,
        }
    }
}

/// A bit-crusher / decimator node.
#[derive(Debug)]
pub struct BitcrusherNode {
    /// Effective bit depth (clamped to `[MIN_BIT_DEPTH, MAX_BIT_DEPTH]`).
    bit_depth: Sample,
    /// Sample-rate reduction factor (clamped to `>= 1`).
    downsample: Sample,
    /// Wet/dry blend (clamped to `[0, 1]`).
    mix: Sample,
    /// Last captured (quantized) sample per channel.
    held: Vec<Sample>,
    /// Fractional decimation phase shared across channels; a capture fires when
    /// it reaches `1`.
    phase: Sample,
}

impl BitcrusherNode {
    /// Builds a bit-crusher for `layout`.
    #[must_use]
    pub fn new(layout: ChannelLayout, params: BitcrusherParams) -> Self {
        let channels = layout.channel_count();
        Self {
            bit_depth: params.bit_depth.clamp(MIN_BIT_DEPTH, MAX_BIT_DEPTH),
            downsample: params.downsample.max(1.0),
            mix: params.mix.clamp(0.0, 1.0),
            held: alloc::vec![0.0; channels],
            // Start "due" so the very first frame captures a sample.
            phase: 1.0,
        }
    }

    /// Sets the effective bit depth (clamped to `[MIN_BIT_DEPTH, MAX_BIT_DEPTH]`).
    pub fn set_bit_depth(&mut self, bit_depth: Sample) {
        self.bit_depth = bit_depth.clamp(MIN_BIT_DEPTH, MAX_BIT_DEPTH);
    }

    /// Sets the sample-rate reduction factor (clamped to `>= 1`).
    pub fn set_downsample(&mut self, downsample: Sample) {
        self.downsample = downsample.max(1.0);
    }

    /// Sets the wet/dry blend (clamped to `[0, 1]`).
    pub fn set_mix(&mut self, mix: Sample) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Rounds `x` onto the current bit-depth amplitude grid.
    ///
    /// A `b`-bit converter has exactly `2^b` codes. Mid-tread rounding over the
    /// closed range `[-1, 1]` would yield `2^b + 1` levels (both endpoints get
    /// their own code), so the integer code is clamped to the signed
    /// two's-complement range `[-2^(b-1), 2^(b-1) - 1]`, exactly mirroring real
    /// converter hardware and keeping the code count at `2^b`.
    #[inline]
    fn quantize(&self, x: Sample) -> Sample {
        let levels = ops::powf(2.0, self.bit_depth);
        let step = 2.0 / levels;
        let half = levels * 0.5;
        let code = ops::round(x / step).clamp(-half, half - 1.0);
        flush_denormal(code * step)
    }
}

impl AudioNode for BitcrusherNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels()).min(self.held.len());
        let frames = output.active_frames();
        if frames == 0 || channels == 0 {
            return;
        }

        let increment = 1.0 / self.downsample;
        let mix = self.mix;
        let dry = 1.0 - mix;

        for f in 0..frames {
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                for ch in 0..channels {
                    self.held[ch] = self.quantize(input.channel(ch)[f]);
                }
            }
            for ch in 0..channels {
                let wet = self.held[ch];
                output.channel_mut(ch)[f] = input.channel(ch)[f] * dry + wet * mix;
            }
            self.phase += increment;
        }
    }

    fn reset(&mut self) {
        for h in &mut self.held {
            *h = 0.0;
        }
        self.phase = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::AudioBuffer;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn ramp(layout: ChannelLayout, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for ch in 0..layout.channel_count() {
            let data = buf.channel_mut(ch);
            for (i, s) in data.iter_mut().enumerate() {
                // A slow bipolar ramp across [-1, 1).
                *s = -1.0 + 2.0 * (i as Sample) / (frames as Sample);
            }
        }
        buf
    }

    fn dc(layout: ChannelLayout, frames: usize, values: &[Sample]) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for ch in 0..layout.channel_count() {
            let v = values[ch % values.len()];
            for s in buf.channel_mut(ch) {
                *s = v;
            }
        }
        buf
    }

    fn run(node: &mut BitcrusherNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames());
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    #[test]
    fn transparent_at_full_bits_no_downsample_full_wet() {
        let input = ramp(ChannelLayout::Mono, 512);
        let params = BitcrusherParams {
            bit_depth: 24.0,
            downsample: 1.0,
            mix: 1.0,
        };
        let mut node = BitcrusherNode::new(ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        for i in 0..out.active_frames() {
            assert!(
                (out.channel(0)[i] - input.channel(0)[i]).abs() < 1e-3,
                "not transparent at 24 bits"
            );
        }
    }

    #[test]
    fn dry_mix_passes_input_unchanged() {
        let input = ramp(ChannelLayout::Mono, 256);
        let params = BitcrusherParams {
            bit_depth: 2.0,
            downsample: 8.0,
            mix: 0.0,
        };
        let mut node = BitcrusherNode::new(ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        for i in 0..out.active_frames() {
            assert!((out.channel(0)[i] - input.channel(0)[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn low_bit_depth_collapses_to_few_levels() {
        // 2 bits -> 4 levels. A full-scale ramp must land on at most 4 values.
        let input = ramp(ChannelLayout::Mono, 4096);
        let params = BitcrusherParams {
            bit_depth: 2.0,
            downsample: 1.0,
            mix: 1.0,
        };
        let mut node = BitcrusherNode::new(ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        let mut levels: Vec<Sample> = Vec::new();
        for &s in out.channel(0) {
            if !levels.iter().any(|&l| (l - s).abs() < 1e-4) {
                levels.push(s);
            }
        }
        assert!(levels.len() <= 4, "too many levels: {}", levels.len());
    }

    #[test]
    fn quantized_values_sit_on_the_grid() {
        let input = ramp(ChannelLayout::Mono, 1024);
        let bits = 3.0;
        let params = BitcrusherParams {
            bit_depth: bits,
            downsample: 1.0,
            mix: 1.0,
        };
        let mut node = BitcrusherNode::new(ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        let step = 2.0 / ops::powf(2.0, bits);
        for &s in out.channel(0) {
            let ratio = s / step;
            let nearest = ops::round(ratio);
            assert!((ratio - nearest).abs() < 1e-3, "off grid: {s}");
        }
    }

    #[test]
    fn downsample_holds_samples_in_groups() {
        // downsample = 4 -> the output changes only every 4th frame.
        let input = ramp(ChannelLayout::Mono, 64);
        let params = BitcrusherParams {
            bit_depth: 24.0,
            downsample: 4.0,
            mix: 1.0,
        };
        let mut node = BitcrusherNode::new(ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        for i in 0..out.active_frames() {
            let group_start = i - (i % 4);
            assert!(
                (out.channel(0)[i] - out.channel(0)[group_start]).abs() < 1e-6,
                "hold broken at frame {i}"
            );
        }
    }

    #[test]
    fn stereo_channels_update_synchronously() {
        // Distinct DC per channel; with downsample 4 both channels must capture
        // on the same frames and hold otherwise.
        let input = dc(ChannelLayout::Stereo, 32, &[0.6, -0.4]);
        let params = BitcrusherParams {
            bit_depth: 24.0,
            downsample: 4.0,
            mix: 1.0,
        };
        let mut node = BitcrusherNode::new(ChannelLayout::Stereo, params);
        let out = run(&mut node, &input);
        // DC input: every held value equals the (quantized) DC, so both channels
        // are constant and near the input values.
        for i in 0..out.active_frames() {
            assert!((out.channel(0)[i] - 0.6).abs() < 1e-2);
            assert!((out.channel(1)[i] + 0.4).abs() < 1e-2);
        }
    }

    #[test]
    fn params_are_clamped() {
        let mut node = BitcrusherNode::new(ChannelLayout::Mono, BitcrusherParams::default());
        node.set_bit_depth(0.1);
        node.set_downsample(0.01);
        node.set_mix(5.0);
        // A degenerate downsample must not stall or panic; a clamped 1-bit grid
        // still produces finite output.
        let input = ramp(ChannelLayout::Mono, 128);
        let out = run(&mut node, &input);
        for &s in out.channel(0) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn output_is_finite_across_settings() {
        for &(b, d, m) in &[(1.0, 1.0, 1.0), (6.0, 3.5, 0.5), (16.0, 12.0, 0.8)] {
            let input = ramp(ChannelLayout::Stereo, 300);
            let params = BitcrusherParams {
                bit_depth: b,
                downsample: d,
                mix: m,
            };
            let mut node = BitcrusherNode::new(ChannelLayout::Stereo, params);
            let out = run(&mut node, &input);
            for ch in 0..2 {
                for &s in out.channel(ch) {
                    assert!(s.is_finite(), "non-finite for {b}/{d}/{m}");
                }
            }
        }
    }

    #[test]
    fn reset_restores_deterministic_state() {
        let input = ramp(ChannelLayout::Mono, 200);
        let params = BitcrusherParams {
            bit_depth: 4.0,
            downsample: 3.0,
            mix: 1.0,
        };
        let mut node = BitcrusherNode::new(ChannelLayout::Mono, params);
        let first = run(&mut node, &input);
        node.reset();
        let second = run(&mut node, &input);
        for i in 0..first.active_frames() {
            assert!((first.channel(0)[i] - second.channel(0)[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_frame_block_does_not_panic() {
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, 64);
        buf.set_active_frames(0);
        let mut node = BitcrusherNode::new(ChannelLayout::Stereo, BitcrusherParams::default());
        let _ = run(&mut node, &buf);
    }
}
