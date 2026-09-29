//! Phaser: a cascade of first-order all-pass stages whose coefficient is swept
//! by an LFO, sweeping a series of notches through the spectrum.
//!
//! Each stage is a first-order all-pass filter with unity magnitude response
//! but a frequency-dependent phase shift. Summing the phase-shifted signal with
//! the dry input creates notches wherever the phase reaches 180 degrees; an
//! LFO sweeps the all-pass coefficient (and therefore the notch frequencies)
//! for the characteristic "whoosh". Optional feedback sharpens the notches.
//!
//! Storage is fixed at construction, so
//! [`PhaserNode::process`](crate::graph::AudioNode::process) is real-time safe.

use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::modulation::{Lfo, LfoWaveform};
use crate::param::{Ramp, Smoothed};

/// Number of cascaded all-pass stages (each adds one swept notch pair).
const STAGES: usize = 6;

/// Minimum all-pass coefficient reached at the bottom of the LFO sweep.
const A_MIN: Sample = 0.1;
/// Maximum all-pass coefficient reached at the top of the LFO sweep.
const A_MAX: Sample = 0.95;
/// Largest stable feedback magnitude.
const MAX_FEEDBACK: Sample = 0.95;

/// Construction parameters for a [`PhaserNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PhaserParams {
    /// LFO rate in Hz (typically 0.1-2 Hz).
    pub rate_hz: Sample,
    /// Sweep depth in `[0, 1]`; scales how far the coefficient travels.
    pub depth: Sample,
    /// Feedback coefficient in `[-0.95, 0.95]`; sharpens the notches.
    pub feedback: Sample,
    /// Wet (processed) mix gain.
    pub wet: Sample,
    /// Dry (unprocessed) mix gain.
    pub dry: Sample,
}

impl Default for PhaserParams {
    fn default() -> Self {
        Self {
            rate_hz: 0.5,
            depth: 1.0,
            feedback: 0.5,
            wet: 0.5,
            dry: 1.0,
        }
    }
}

/// A multi-stage phaser (input port 0 -> output port 0).
///
/// Depth, feedback, and wet/dry gains are [`Smoothed`] so automation stays
/// click-free. The all-pass coefficient is swept per sample by the LFO.
#[derive(Debug, Clone)]
pub struct PhaserNode {
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Per-channel previous all-pass inputs, one slot per stage.
    x_prev: Vec<[Sample; STAGES]>,
    /// Per-channel previous all-pass outputs, one slot per stage.
    y_prev: Vec<[Sample; STAGES]>,
    /// Per-channel last chain output, used for the feedback path.
    fb_state: Vec<Sample>,
    /// Shared sweep LFO.
    lfo: Lfo,
    /// Smoothed sweep depth in `[0, 1]`.
    depth: Smoothed,
    /// Smoothed feedback coefficient.
    feedback: Smoothed,
    /// Smoothed wet mix gain.
    wet: Smoothed,
    /// Smoothed dry mix gain.
    dry: Smoothed,
}

impl PhaserNode {
    /// Builds a phaser for `channels` channels at `sample_rate` Hz.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, params: PhaserParams) -> Self {
        let channels = channels.max(1);
        Self {
            sample_rate,
            x_prev: {
                let mut v = Vec::with_capacity(channels);
                v.resize(channels, [0.0; STAGES]);
                v
            },
            y_prev: {
                let mut v = Vec::with_capacity(channels);
                v.resize(channels, [0.0; STAGES]);
                v
            },
            fb_state: {
                let mut v = Vec::with_capacity(channels);
                v.resize(channels, 0.0);
                v
            },
            lfo: Lfo::new(sample_rate, params.rate_hz, LfoWaveform::Sine),
            depth: Smoothed::new(params.depth.clamp(0.0, 1.0)),
            feedback: Smoothed::new(params.feedback.clamp(-MAX_FEEDBACK, MAX_FEEDBACK)),
            wet: Smoothed::new(params.wet),
            dry: Smoothed::new(params.dry),
        }
    }

    /// Returns the number of channels processed.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.fb_state.len()
    }

    /// Sets the LFO rate in Hz.
    #[inline]
    pub fn set_rate_hz(&mut self, rate_hz: Sample) {
        self.lfo.set_frequency(self.sample_rate, rate_hz);
    }

    /// Sets the sweep depth in `[0, 1]`.
    #[inline]
    pub fn set_depth(&mut self, depth: Sample, ramp: Ramp) {
        self.depth.set_target(depth.clamp(0.0, 1.0), ramp);
    }

    /// Sets the feedback coefficient, clamped to `[-0.95, 0.95]`.
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

impl AudioNode for PhaserNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels().min(self.fb_state.len());
        let frames = output.active_frames();
        let center = (A_MIN + A_MAX) * 0.5;
        let half_range = (A_MAX - A_MIN) * 0.5;

        for f in 0..frames {
            let depth = self.depth.next_sample();
            let feedback = self.feedback.next_sample();
            let wet = self.wet.next_sample();
            let dry = self.dry.next_sample();

            // Sweep the all-pass coefficient with one LFO tick per frame.
            let a = (center + half_range * depth * self.lfo.next_sample()).clamp(-0.999, 0.999);

            for ch in 0..channels {
                let x = input.channel(ch)[f];
                let mut sample = x + feedback * self.fb_state[ch];

                let xs = &mut self.x_prev[ch];
                let ys = &mut self.y_prev[ch];
                for s in 0..STAGES {
                    let xin = sample;
                    // First-order all-pass: y = a*xin + x[-1] - a*y[-1].
                    let y = a * xin + xs[s] - a * ys[s];
                    xs[s] = xin;
                    ys[s] = y;
                    sample = y;
                }

                self.fb_state[ch] = flush_denormal(sample);
                output.channel_mut(ch)[f] = dry * x + wet * sample;
            }
        }
    }

    fn reset(&mut self) {
        for slot in &mut self.x_prev {
            *slot = [0.0; STAGES];
        }
        for slot in &mut self.y_prev {
            *slot = [0.0; STAGES];
        }
        for s in &mut self.fb_state {
            *s = 0.0;
        }
        self.lfo.reset();
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
    use bevy_math::ops;

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
        let params = PhaserParams {
            wet: 0.0,
            dry: 1.0,
            ..PhaserParams::default()
        };
        let mut node = PhaserNode::new(48_000, 1, params);
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
    fn allpass_preserves_sine_amplitude() {
        // Static coefficient (rate=0, depth=0), no feedback, wet only:
        // an all-pass cascade is unity-magnitude, so a steady sine keeps its
        // peak amplitude (only its phase shifts).
        let params = PhaserParams {
            rate_hz: 0.0,
            depth: 0.0,
            feedback: 0.0,
            wet: 1.0,
            dry: 0.0,
        };
        let mut node = PhaserNode::new(48_000, 1, params);
        let mut input = mono(4096);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            let phase = 2.0 * core::f32::consts::PI * 0.02 * i as Sample;
            *s = 0.5 * ops::sin(phase);
        }
        let inputs = [input];
        let mut outputs = [mono(4096)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(4096), &mut io);
        // Examine the settled tail.
        let peak = outputs[0].channel(0)[2048..]
            .iter()
            .fold(0.0 as Sample, |m, &v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.05, "allpass changed amplitude: {peak}");
    }

    #[test]
    fn feedback_stays_bounded() {
        let mut node = PhaserNode::new(48_000, 2, PhaserParams::default());
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4096);
        for ch in 0..2 {
            for (i, s) in input.channel_mut(ch).iter_mut().enumerate() {
                let phase = 2.0 * core::f32::consts::PI * 0.03 * i as Sample;
                *s = 0.6 * ops::sin(phase);
            }
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, 4096)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(4096), &mut io);
        for ch in 0..2 {
            for &y in outputs[0].channel(ch) {
                assert!(y.is_finite() && y.abs() < 20.0, "runaway: {y}");
            }
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut node = PhaserNode::new(48_000, 1, PhaserParams::default());
        let mut input = mono(256);
        for s in input.channel_mut(0).iter_mut() {
            *s = 0.5;
        }
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
            assert!(y.abs() < 1e-9, "state not cleared: {y}");
        }
    }
}
