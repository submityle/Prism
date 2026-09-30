//! Tremolo and auto-pan: low-frequency amplitude modulation.
//!
//! Tremolo periodically varies a signal's loudness; auto-pan periodically
//! moves it across the stereo field. Both are driven by the same control-rate
//! [`Lfo`](crate::modulation::Lfo), differing only in how the oscillator maps
//! to per-channel gain.
//!
//! # The model
//!
//! A single bipolar LFO value `m` in `[-1, 1]` is read once per sample.
//!
//! - [`TremoloMode::Amplitude`] converts it to a scalar gain
//!   `g = 1 - depth * (0.5 - 0.5 * m)`, which sweeps between `1 - depth` (at the
//!   trough) and `1` (at the crest) and is applied to every channel. An
//!   optional per-channel phase offset (`stereo_phase`) staggers the LFO across
//!   channels to produce the classic "harmonic"/rotary stereo shimmer.
//! - [`TremoloMode::AutoPan`] treats `m` as a pan position and derives an
//!   equal-power left/right gain pair with [`equal_power_pan`], panning the
//!   image side to side without changing its perceived loudness. It applies to
//!   the first two channels; other layouts fall back to amplitude tremolo.
//!
//! # Real-time contract
//!
//! One LFO per channel is allocated in [`TremoloNode::new`].
//! [`process`](crate::graph::AudioNode::process) performs no allocation, takes
//! no locks, and cannot panic: mismatched channel counts and zero-length blocks
//! degrade gracefully.
//!
//! # Provenance
//!
//! Amplitude tremolo (a gain multiplied by an LFO) and auto-pan (an LFO driving
//! an equal-power pan law) are elementary modulation effects described in every
//! audio-effects text (e.g. Zoelzer, "DAFX"; Reiss and `McPherson`, "Audio
//! Effects", 2014). This module reuses this crate's own [`Lfo`] and
//! [`equal_power_pan`] primitives. It contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**; it is implemented purely from that publicly documented theory.

use alloc::vec::Vec;

use crate::buffer::ChannelLayout;
use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, equal_power_pan};
use crate::modulation::{Lfo, LfoWaveform};

/// How the modulation LFO is mapped onto the channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TremoloMode {
    /// Modulate loudness: one gain applied to every channel (the default).
    #[default]
    Amplitude,
    /// Modulate stereo position with an equal-power pan law across the first
    /// two channels.
    AutoPan,
}

/// Configuration for a [`TremoloNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TremoloParams {
    /// Modulation rate (Hz).
    pub rate_hz: Sample,
    /// Modulation depth in `[0, 1]`: `0` is bypass, `1` is full modulation.
    pub depth: Sample,
    /// LFO waveform shape.
    pub waveform: LfoWaveform,
    /// Whether to modulate amplitude or stereo position.
    pub mode: TremoloMode,
    /// Phase offset (fraction of a cycle, `[0, 1)`) applied per channel in
    /// amplitude mode, so channel `c` starts at `c * stereo_phase`. Zero keeps
    /// all channels in lockstep; `0.5` puts a stereo pair in antiphase.
    pub stereo_phase: Sample,
}

impl Default for TremoloParams {
    fn default() -> Self {
        Self {
            rate_hz: 5.0,
            depth: 0.5,
            waveform: LfoWaveform::Sine,
            mode: TremoloMode::Amplitude,
            stereo_phase: 0.0,
        }
    }
}

/// A tremolo / auto-pan modulation node.
#[derive(Debug)]
pub struct TremoloNode {
    /// One LFO per channel; offsets encode `stereo_phase`.
    lfos: Vec<Lfo>,
    /// Modulation depth in `[0, 1]`.
    depth: Sample,
    /// Amplitude vs auto-pan mapping.
    mode: TremoloMode,
    /// Per-channel starting phase offset (fraction of a cycle).
    stereo_phase: Sample,
}

impl TremoloNode {
    /// Builds a tremolo for `layout` at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, params: TremoloParams) -> Self {
        let channels = layout.channel_count();
        let mut lfos: Vec<Lfo> = Vec::with_capacity(channels);
        let phase = params.stereo_phase - ops_floor(params.stereo_phase);
        for c in 0..channels {
            let mut lfo = Lfo::new(sample_rate, params.rate_hz, params.waveform);
            lfo.set_phase(phase * c as Sample);
            lfos.push(lfo);
        }
        Self {
            lfos,
            depth: params.depth.clamp(0.0, 1.0),
            mode: params.mode,
            stereo_phase: phase,
        }
    }

    /// Sets the modulation depth (clamped to `[0, 1]`).
    pub fn set_depth(&mut self, depth: Sample) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    /// Sets the modulation rate (Hz) for every channel.
    pub fn set_rate(&mut self, sample_rate: u32, rate_hz: Sample) {
        for lfo in &mut self.lfos {
            lfo.set_frequency(sample_rate, rate_hz);
        }
    }

    /// Sets the LFO waveform for every channel.
    pub fn set_waveform(&mut self, waveform: LfoWaveform) {
        for lfo in &mut self.lfos {
            lfo.set_waveform(waveform);
        }
    }

    /// Selects amplitude or auto-pan mapping.
    pub fn set_mode(&mut self, mode: TremoloMode) {
        self.mode = mode;
    }
}

/// Fractional-part helper avoiding an unused top-level `ops` import.
#[inline]
fn ops_floor(x: Sample) -> Sample {
    bevy_math::ops::floor(x)
}

impl AudioNode for TremoloNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(input.channels());
        let frames = output.active_frames();
        if frames == 0 || channels == 0 || self.lfos.is_empty() {
            return;
        }

        let auto_pan = self.mode == TremoloMode::AutoPan && channels >= 2;

        for f in 0..frames {
            if auto_pan {
                // Drive an equal-power pan from the first LFO; leave the rest
                // advancing in lockstep so a later mode switch stays coherent.
                let m = self.lfos[0].next_sample();
                for lfo in self.lfos.iter_mut().skip(1) {
                    let _ = lfo.next_sample();
                }
                let (lg, rg) = equal_power_pan(m);
                output.channel_mut(0)[f] = input.channel(0)[f] * lg;
                output.channel_mut(1)[f] = input.channel(1)[f] * rg;
                for ch in 2..channels {
                    output.channel_mut(ch)[f] = input.channel(ch)[f];
                }
            } else {
                for ch in 0..channels {
                    let m = self.lfos[ch].next_sample();
                    let gain = 1.0 - self.depth * (0.5 - 0.5 * m);
                    output.channel_mut(ch)[f] = input.channel(ch)[f] * gain;
                }
            }
        }
    }

    fn reset(&mut self) {
        let phase = self.stereo_phase;
        for (c, lfo) in self.lfos.iter_mut().enumerate() {
            lfo.reset();
            lfo.set_phase(phase * c as Sample);
        }
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

    fn dc(layout: ChannelLayout, frames: usize, value: Sample) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        buf.set_active_frames(frames);
        for ch in 0..layout.channel_count() {
            for s in buf.channel_mut(ch) {
                *s = value;
            }
        }
        buf
    }

    fn run(node: &mut TremoloNode, input: &AudioBuffer) -> AudioBuffer {
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
    fn zero_depth_is_bypass() {
        let input = dc(ChannelLayout::Mono, 2048, 1.0);
        let params = TremoloParams {
            depth: 0.0,
            ..TremoloParams::default()
        };
        let mut node = TremoloNode::new(SR, ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        for s in out.channel(0) {
            assert!((s - 1.0).abs() < 1e-6, "not bypass: {s}");
        }
    }

    #[test]
    fn amplitude_output_stays_within_gain_envelope() {
        // With depth d the gain lives in [1 - d, 1], so a unit DC input never
        // exceeds 1 nor drops below 1 - d.
        let depth = 0.5;
        let input = dc(ChannelLayout::Mono, 48_000, 1.0);
        let params = TremoloParams {
            rate_hz: 5.0,
            depth,
            ..TremoloParams::default()
        };
        let mut node = TremoloNode::new(SR, ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        let mut lo = Sample::INFINITY;
        let mut hi = Sample::NEG_INFINITY;
        for &s in out.channel(0) {
            lo = lo.min(s);
            hi = hi.max(s);
        }
        assert!(hi <= 1.0 + 1e-4, "gain exceeded 1: {hi}");
        assert!(lo >= 1.0 - depth - 1e-4, "gain below floor: {lo}");
        // A 5 Hz sweep over a full second must actually reach both extremes.
        assert!(hi > 1.0 - 1e-3, "crest not reached: {hi}");
        assert!(lo < 1.0 - depth + 0.05, "trough not reached: {lo}");
    }

    #[test]
    fn deeper_depth_modulates_more() {
        let input = dc(ChannelLayout::Mono, 48_000, 1.0);
        let range = |depth: Sample| {
            let params = TremoloParams {
                depth,
                ..TremoloParams::default()
            };
            let mut node = TremoloNode::new(SR, ChannelLayout::Mono, params);
            let out = run(&mut node, &input);
            let mut lo = Sample::INFINITY;
            let mut hi = Sample::NEG_INFINITY;
            for &s in out.channel(0) {
                lo = lo.min(s);
                hi = hi.max(s);
            }
            hi - lo
        };
        assert!(range(0.8) > range(0.3), "deeper depth should swing more");
    }

    #[test]
    fn auto_pan_conserves_power_and_moves_image() {
        // Feed equal DC to both channels; auto-pan should push energy left then
        // right while the summed power stays roughly constant.
        let input = dc(ChannelLayout::Stereo, 48_000, 1.0);
        let params = TremoloParams {
            rate_hz: 2.0,
            depth: 1.0,
            mode: TremoloMode::AutoPan,
            ..TremoloParams::default()
        };
        let mut node = TremoloNode::new(SR, ChannelLayout::Stereo, params);
        let out = run(&mut node, &input);
        let mut min_l = Sample::INFINITY;
        let mut max_l = Sample::NEG_INFINITY;
        for i in 0..out.active_frames() {
            let l = out.channel(0)[i];
            let r = out.channel(1)[i];
            min_l = min_l.min(l);
            max_l = max_l.max(l);
            // Equal-power law: l^2 + r^2 == 1 for unit input.
            assert!((l * l + r * r - 1.0).abs() < 1e-3, "power drift: {l} {r}");
        }
        assert!(max_l > 0.95, "did not pan hard left: {max_l}");
        assert!(min_l < 0.05, "did not pan hard right: {min_l}");
    }

    #[test]
    fn auto_pan_on_mono_falls_back_to_amplitude() {
        let input = dc(ChannelLayout::Mono, 1024, 1.0);
        let params = TremoloParams {
            mode: TremoloMode::AutoPan,
            depth: 0.5,
            ..TremoloParams::default()
        };
        let mut node = TremoloNode::new(SR, ChannelLayout::Mono, params);
        let out = run(&mut node, &input);
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s <= 1.0 + 1e-4 && s >= 0.5 - 1e-4, "outside envelope: {s}");
        }
    }

    #[test]
    fn stereo_phase_offset_decorrelates_channels() {
        // Antiphase LFOs make the two channel gains differ at any instant.
        let input = dc(ChannelLayout::Stereo, 4096, 1.0);
        let params = TremoloParams {
            rate_hz: 5.0,
            depth: 0.8,
            stereo_phase: 0.5,
            ..TremoloParams::default()
        };
        let mut node = TremoloNode::new(SR, ChannelLayout::Stereo, params);
        let out = run(&mut node, &input);
        let mut max_diff = 0.0;
        for i in 0..out.active_frames() {
            let d = (out.channel(0)[i] - out.channel(1)[i]).abs();
            max_diff = Sample::max(max_diff, d);
        }
        assert!(max_diff > 0.3, "channels not decorrelated: {max_diff}");
    }

    #[test]
    fn reset_restores_initial_phase() {
        let input = dc(ChannelLayout::Mono, 256, 1.0);
        let params = TremoloParams::default();
        let mut node = TremoloNode::new(SR, ChannelLayout::Mono, params);
        let first = run(&mut node, &input);
        node.reset();
        let second = run(&mut node, &input);
        for i in 0..first.active_frames() {
            assert!((first.channel(0)[i] - second.channel(0)[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn output_is_finite_for_all_waveforms() {
        for wf in [
            LfoWaveform::Sine,
            LfoWaveform::Triangle,
            LfoWaveform::Sawtooth,
            LfoWaveform::Square,
        ] {
            let input = dc(ChannelLayout::Stereo, 1024, 0.7);
            let params = TremoloParams {
                waveform: wf,
                depth: 1.0,
                ..TremoloParams::default()
            };
            let mut node = TremoloNode::new(SR, ChannelLayout::Stereo, params);
            let out = run(&mut node, &input);
            for ch in 0..2 {
                for &s in out.channel(ch) {
                    assert!(s.is_finite(), "non-finite for {wf:?}");
                }
            }
        }
    }

    #[test]
    fn zero_frame_block_does_not_panic() {
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, 64);
        buf.set_active_frames(0);
        let mut node = TremoloNode::new(SR, ChannelLayout::Stereo, TremoloParams::default());
        let _ = run(&mut node, &buf);
    }
}
