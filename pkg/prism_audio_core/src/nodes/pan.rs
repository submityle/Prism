//! Equal-power mono-to-stereo panner node.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, equal_power_pan};
use crate::param::{Ramp, Smoothed};

/// Pans a mono input (port 0) into a stereo output (port 0) using the
/// constant-power pan law, so perceived loudness stays constant as the source
/// sweeps across the stereo field.
///
/// The pan position is smoothed to avoid clicks during fast movement (e.g. a
/// projectile flying past the listener).
#[derive(Debug, Clone)]
pub struct StereoPanNode {
    pan: Smoothed,
}

impl StereoPanNode {
    /// Creates a panner at position `pan` in `[-1.0, 1.0]` (`-1` = hard left,
    /// `0` = centre, `1` = hard right).
    #[must_use]
    pub fn new(pan: Sample) -> Self {
        Self {
            pan: Smoothed::new(pan.clamp(-1.0, 1.0)),
        }
    }

    /// Sets a new target pan position, gliding toward it with `ramp`.
    #[inline]
    pub fn set_pan(&mut self, pan: Sample, ramp: Ramp) {
        self.pan.set_target(pan.clamp(-1.0, 1.0), ramp);
    }
}

impl AudioNode for StereoPanNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let mono = input.channel(0);
        let frames = mono.len();
        let (left, right) = output.channel_pair_mut(0, 1);
        for i in 0..frames {
            let p = self.pan.next_sample();
            let (lg, rg) = equal_power_pan(p);
            let x = mono[i];
            left[i] = x * lg;
            right[i] = x * rg;
        }
    }

    fn reset(&mut self) {
        self.pan = Smoothed::new(self.pan.target());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    fn ctx() -> RenderContext {
        RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 }
    }

    #[test]
    fn center_is_balanced() {
        let mut node = StereoPanNode::new(0.0);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0; 4]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        let l = outputs[0].channel(0)[0];
        let r = outputs[0].channel(1)[0];
        assert!((l - r).abs() < 1e-6);
        assert!((l * l + r * r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hard_left_silences_right() {
        let mut node = StereoPanNode::new(-1.0);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0; 4]);
        let output = AudioBuffer::new(ChannelLayout::Stereo, 4);
        let inputs = [input];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        assert!(outputs[0].channel(0)[0] > 0.99);
        assert!(outputs[0].channel(1)[0] < 1e-3);
    }
}
