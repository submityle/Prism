//! Explicit N-input summing (mixer) node.

use crate::graph::{AudioNode, ProcessIo, RenderContext};

/// Sums all of its input ports into its single output port.
///
/// The [`AudioGraph`](crate::graph::AudioGraph) already sums multiple edges
/// feeding one input port, but a `SumNode` is useful when a bus wants a fixed,
/// explicitly addressable set of input slots (e.g. a mixer strip with named
/// channels) whose count is part of the graph topology. Every input port and
/// the output port must share the same channel layout.
#[derive(Debug, Clone, Default)]
pub struct SumNode;

impl SumNode {
    /// Creates a summing node.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl AudioNode for SumNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let inputs = io.input_count();
        {
            let out = io.output(0);
            out.clear();
        }
        for p in 0..inputs {
            // Borrow the input, then the output, disjointly per iteration.
            let (input, output) = io.io(p, 0);
            output.add_scaled(input, 1.0);
        }
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
    fn sums_all_inputs() {
        let mut node = SumNode::new();
        let mut a = AudioBuffer::new(ChannelLayout::Mono, 4);
        let mut b = AudioBuffer::new(ChannelLayout::Mono, 4);
        a.channel_mut(0).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        b.channel_mut(0).copy_from_slice(&[0.5, 0.5, 0.5, 0.5]);
        let output = AudioBuffer::new(ChannelLayout::Mono, 4);
        let inputs = [a, b];
        let mut outputs = [output];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(), &mut io);
        assert_eq!(outputs[0].channel(0), &[1.5, 2.5, 3.5, 4.5]);
    }
}
