//! Click-free gain (level trim) node.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Applies a smoothed linear gain to every channel of its single input,
/// writing the result to its single output.
///
/// The gain is driven by a [`Smoothed`] value so that automation (volume
/// fades, ducking) never introduces zipper-noise clicks. Input and output must
/// share the same channel layout; the node transforms in place conceptually
/// (input port 0 -> output port 0).
#[derive(Debug, Clone)]
pub struct GainNode {
    gain: Smoothed,
}

impl GainNode {
    /// Creates a gain node settled at `initial_linear` (a linear multiplier,
    /// not decibels — use [`db_to_linear`](crate::math::db_to_linear) to
    /// convert).
    #[must_use]
    pub fn new(initial_linear: Sample) -> Self {
        Self {
            gain: Smoothed::new(initial_linear),
        }
    }

    /// Sets a new target gain, gliding toward it with `ramp`.
    #[inline]
    pub fn set_gain(&mut self, target_linear: Sample, ramp: Ramp) {
        self.gain.set_target(target_linear, ramp);
    }

    /// Returns the instantaneous (current) linear gain.
    #[inline]
    #[must_use]
    pub fn current_gain(&self) -> Sample {
        self.gain.current()
    }
}

impl AudioNode for GainNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output.channels();
        let frames = output.active_frames();

        // Advance the smoothed gain once per frame and apply the same value to
        // every channel so the stereo/surround image stays coherent.
        if self.gain.is_settled() {
            let g = self.gain.current();
            for ch in 0..channels {
                let src = input.channel(ch);
                let dst = output.channel_mut(ch);
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *s * g;
                }
            }
        } else {
            // We need the same per-frame gain sequence for each channel, so
            // snapshot the smoother state and replay it channel by channel.
            let start = self.gain;
            for ch in 0..channels {
                let mut g = start;
                let src = input.channel(ch);
                let dst = output.channel_mut(ch);
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = *s * g.next_sample();
                }
                if ch + 1 == channels {
                    self.gain = g;
                }
            }
            let _ = frames;
        }
    }

    fn reset(&mut self) {
        self.gain = Smoothed::new(self.gain.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use crate::graph::ProcessIo;

    fn ctx() -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames: 4,
            playhead: 0,
        }
    }

    #[test]
    fn settled_gain_scales() {
        let mut node = GainNode::new(0.5);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let output = AudioBuffer::new(ChannelLayout::Mono, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        assert_eq!(outputs[0].channel(0), &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn ramp_is_monotonic_and_reaches_target() {
        let mut node = GainNode::new(0.0);
        node.set_gain(1.0, Ramp::Linear { samples: 4 });
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let output = AudioBuffer::new(ChannelLayout::Mono, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        let out = outputs[0].channel(0);
        assert!(out[0] < out[1] && out[1] < out[2] && out[2] < out[3]);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }
}
