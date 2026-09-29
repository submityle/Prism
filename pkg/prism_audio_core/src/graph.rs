//! The unified real-time audio render graph.
//!
//! Every sound source, effect, bus, and spatializer in the engine is an
//! [`AudioNode`] living in a single [`AudioGraph`]. This mirrors the design of
//! Web Audio's `AudioNode` graph, UE's Submix tree, and Godot's bus chain:
//! rather than giving every playing sound its own isolated mixer, the whole
//! signal flow is one directed acyclic graph that is compiled once (off the
//! audio thread) into a deterministic, allocation-free processing plan.
//!
//! # Lifecycle
//!
//! 1. Build the topology with [`AudioGraph::add_node`] and
//!    [`AudioGraph::connect`] (may allocate; run off the audio thread).
//! 2. Call [`AudioGraph::compile`] to topologically sort the graph and
//!    pre-allocate every intermediate buffer.
//! 3. Call [`AudioGraph::process`] once per block on the audio thread. This is
//!    allocation-free, lock-free, and panic-free.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::buffer::{AudioBuffer, ChannelLayout};
use crate::math::Sample;

/// Stable identifier for a node within a single [`AudioGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// Reference to a specific input or output port of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortRef {
    /// The node the port belongs to.
    pub node: NodeId,
    /// Zero-based port index.
    pub port: usize,
}

impl PortRef {
    /// Convenience constructor.
    #[inline]
    #[must_use]
    pub fn new(node: NodeId, port: usize) -> Self {
        Self { node, port }
    }
}

/// Per-block context handed to every [`AudioNode::process`] call.
#[derive(Debug, Clone, Copy)]
pub struct RenderContext {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Number of frames to render this block.
    pub frames: usize,
    /// Global sample position at the start of this block.
    pub playhead: u64,
}

/// The input and output buffers a node may read from and write to during
/// [`AudioNode::process`].
///
/// Input buffers are already populated with the (summed) signal from upstream
/// nodes. Output buffers are **not** cleared automatically; a node must write
/// every active sample it intends to emit (sources typically overwrite, effects
/// typically transform their input into their output).
pub struct ProcessIo<'a> {
    inputs: &'a [AudioBuffer],
    outputs: &'a mut [AudioBuffer],
}

impl<'a> ProcessIo<'a> {
    /// Builds a process view over borrowed input and output port buffers.
    ///
    /// The graph builds this internally, but it is also the entry point for
    /// driving a node standalone (e.g. a pooled voice or a unit test) without
    /// a surrounding [`AudioGraph`].
    #[inline]
    #[must_use]
    pub fn new(inputs: &'a [AudioBuffer], outputs: &'a mut [AudioBuffer]) -> Self {
        Self { inputs, outputs }
    }

    /// Number of input ports.
    #[inline]
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Number of output ports.
    #[inline]
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Immutable access to input port `port`.
    ///
    /// # Panics
    ///
    /// Panics if `port` is out of range.
    #[inline]
    #[must_use]
    pub fn input(&self, port: usize) -> &AudioBuffer {
        &self.inputs[port]
    }

    /// Mutable access to output port `port`.
    ///
    /// # Panics
    ///
    /// Panics if `port` is out of range.
    #[inline]
    pub fn output(&mut self, port: usize) -> &mut AudioBuffer {
        &mut self.outputs[port]
    }

    /// Simultaneous access to input port `ip` and output port `op`.
    ///
    /// This is the common shape for effects that transform an input into an
    /// output without an intermediate copy.
    #[inline]
    pub fn io(&mut self, ip: usize, op: usize) -> (&AudioBuffer, &mut AudioBuffer) {
        (&self.inputs[ip], &mut self.outputs[op])
    }

    /// Simultaneous access to *all* input ports and *all* output ports.
    ///
    /// This is the shape side-chain nodes need: a ducker, for instance, reads
    /// its key on input port 1 while transforming input port 0 into output
    /// port 0. [`io`](Self::io) only exposes a single input/output pair, so
    /// multi-input effects use this split borrow instead.
    #[inline]
    pub fn split(&mut self) -> (&[AudioBuffer], &mut [AudioBuffer]) {
        (self.inputs, self.outputs)
    }
}

/// A single processing unit in the [`AudioGraph`].
///
/// Implementations must be **real-time safe**: [`AudioNode::process`] must not
/// allocate, lock, block, or panic. All internal state (filter memory, delay
/// lines, smoothed parameters) must be pre-allocated at construction.
pub trait AudioNode: Send {
    /// Processes one block, reading `io.input(..)` and writing `io.output(..)`.
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>);

    /// Resets all internal state to silence (filter memory, delay lines, ...).
    ///
    /// Called when the graph is (re)compiled or a voice is recycled. The
    /// default implementation does nothing.
    fn reset(&mut self) {}

    /// Reports the node's processing latency in frames, for delay
    /// compensation. Defaults to zero.
    #[must_use]
    fn latency_frames(&self) -> u32 {
        0
    }
}

/// Errors produced while validating or compiling a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphError {
    /// A referenced node id does not exist.
    UnknownNode(NodeId),
    /// A referenced port index is out of range for its node.
    PortOutOfRange {
        /// The node whose port was addressed.
        node: NodeId,
        /// The out-of-range port index.
        port: usize,
    },
    /// The layouts of a connected output and input port differ.
    LayoutMismatch {
        /// The source output port.
        from: PortRef,
        /// The destination input port.
        to: PortRef,
    },
    /// The graph contains a cycle and cannot be topologically ordered.
    Cycle,
    /// No master output node was designated before compilation.
    NoMaster,
}

impl core::fmt::Display for GraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphError::UnknownNode(id) => write!(f, "unknown node {id:?}"),
            GraphError::PortOutOfRange { node, port } => {
                write!(f, "port {port} out of range for {node:?}")
            }
            GraphError::LayoutMismatch { from, to } => {
                write!(f, "layout mismatch connecting {from:?} -> {to:?}")
            }
            GraphError::Cycle => write!(f, "graph contains a cycle"),
            GraphError::NoMaster => write!(f, "no master output node set"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GraphError {}

struct Edge {
    from: PortRef,
    to: PortRef,
    gain: Sample,
}

struct NodeEntry {
    node: Box<dyn AudioNode>,
    input_layouts: Vec<ChannelLayout>,
    output_layouts: Vec<ChannelLayout>,
}

/// A container of [`AudioNode`]s and the connections between them.
pub struct AudioGraph {
    nodes: Vec<NodeEntry>,
    edges: Vec<Edge>,
    master: Option<PortRef>,
    max_block: usize,
    sample_rate: u32,

    // Compiled state (populated by `compile`).
    order: Vec<usize>,
    inputs: Vec<Vec<AudioBuffer>>,
    outputs: Vec<Vec<AudioBuffer>>,
    compiled: bool,
}

impl AudioGraph {
    /// Creates an empty graph rendering at `sample_rate` Hz with blocks of at
    /// most `max_block` frames.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` or `max_block` is zero.
    #[must_use]
    pub fn new(sample_rate: u32, max_block: usize) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");
        assert!(max_block > 0, "max_block must be non-zero");
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            master: None,
            max_block,
            sample_rate,
            order: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            compiled: false,
        }
    }

    /// Returns the render sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Adds a node with the given input and output port layouts.
    ///
    /// Returns the id used to connect and address the node. Adding a node
    /// invalidates any previous compilation.
    pub fn add_node(
        &mut self,
        node: Box<dyn AudioNode>,
        input_layouts: Vec<ChannelLayout>,
        output_layouts: Vec<ChannelLayout>,
    ) -> NodeId {
        self.compiled = false;
        let id = NodeId(self.nodes.len());
        self.nodes.push(NodeEntry {
            node,
            input_layouts,
            output_layouts,
        });
        id
    }

    /// Connects an output port to an input port with unity gain.
    ///
    /// See [`AudioGraph::connect_with_gain`] for send-style connections.
    pub fn connect(&mut self, from: PortRef, to: PortRef) -> Result<(), GraphError> {
        self.connect_with_gain(from, to, 1.0)
    }

    /// Connects an output port to an input port, scaling the signal by `gain`.
    ///
    /// Multiple connections into the same input port are summed; a single
    /// output port may fan out to many inputs.
    pub fn connect_with_gain(
        &mut self,
        from: PortRef,
        to: PortRef,
        gain: Sample,
    ) -> Result<(), GraphError> {
        let from_layout = self.output_layout(from)?;
        let to_layout = self.input_layout(to)?;
        if from_layout != to_layout {
            return Err(GraphError::LayoutMismatch { from, to });
        }
        self.compiled = false;
        self.edges.push(Edge { from, to, gain });
        Ok(())
    }

    /// Designates which node/port produces the final master mix.
    pub fn set_master(&mut self, port: PortRef) -> Result<(), GraphError> {
        // Validate the port exists as an output.
        self.output_layout(port)?;
        self.master = Some(port);
        self.compiled = false;
        Ok(())
    }

    /// Returns a mutable reference to a previously added node, downcast-free.
    ///
    /// Intended for applying parameter changes between blocks. Returns `None`
    /// if the id is unknown.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut (dyn AudioNode + 'static)> {
        self.nodes.get_mut(id.0).map(|e| e.node.as_mut())
    }

    fn output_layout(&self, port: PortRef) -> Result<ChannelLayout, GraphError> {
        let entry = self
            .nodes
            .get(port.node.0)
            .ok_or(GraphError::UnknownNode(port.node))?;
        entry
            .output_layouts
            .get(port.port)
            .copied()
            .ok_or(GraphError::PortOutOfRange {
                node: port.node,
                port: port.port,
            })
    }

    fn input_layout(&self, port: PortRef) -> Result<ChannelLayout, GraphError> {
        let entry = self
            .nodes
            .get(port.node.0)
            .ok_or(GraphError::UnknownNode(port.node))?;
        entry
            .input_layouts
            .get(port.port)
            .copied()
            .ok_or(GraphError::PortOutOfRange {
                node: port.node,
                port: port.port,
            })
    }

    /// Topologically orders the graph and pre-allocates all buffers.
    ///
    /// Must be called after the topology changes and before
    /// [`AudioGraph::process`]. Returns [`GraphError::Cycle`] if the graph is
    /// not a DAG.
    pub fn compile(&mut self) -> Result<(), GraphError> {
        if self.master.is_none() {
            return Err(GraphError::NoMaster);
        }
        let n = self.nodes.len();

        // Kahn's algorithm on node-level adjacency.
        let mut indegree = alloc::vec![0usize; n];
        let mut adjacency: Vec<Vec<usize>> = alloc::vec![Vec::new(); n];
        for e in &self.edges {
            let (a, b) = (e.from.node.0, e.to.node.0);
            adjacency[a].push(b);
            indegree[b] += 1;
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head];
            head += 1;
            order.push(node);
            for &next in &adjacency[node] {
                indegree[next] -= 1;
                if indegree[next] == 0 {
                    queue.push(next);
                }
            }
        }
        if order.len() != n {
            return Err(GraphError::Cycle);
        }
        self.order = order;

        // Pre-allocate every input/output port buffer at full capacity.
        self.inputs = self
            .nodes
            .iter()
            .map(|e| {
                e.input_layouts
                    .iter()
                    .map(|&l| AudioBuffer::new(l, self.max_block))
                    .collect()
            })
            .collect();
        self.outputs = self
            .nodes
            .iter()
            .map(|e| {
                e.output_layouts
                    .iter()
                    .map(|&l| AudioBuffer::new(l, self.max_block))
                    .collect()
            })
            .collect();

        for entry in &mut self.nodes {
            entry.node.reset();
        }

        self.compiled = true;
        Ok(())
    }

    /// Renders one block into `master_out`.
    ///
    /// `frames` must not exceed the `max_block` given to [`AudioGraph::new`].
    /// This method is allocation-free and must only be called after a
    /// successful [`AudioGraph::compile`].
    ///
    /// # Panics
    ///
    /// Panics if the graph has not been compiled, if `frames` exceeds the
    /// configured maximum block size, or if `master_out` has too few frames of
    /// capacity.
    pub fn process(&mut self, frames: usize, playhead: u64, master_out: &mut AudioBuffer) {
        assert!(self.compiled, "AudioGraph::process called before compile");
        assert!(frames <= self.max_block, "frames exceeds max_block");
        assert!(
            master_out.capacity_frames() >= frames,
            "master_out capacity too small"
        );

        // Resize active window on every arena buffer.
        for ports in self.inputs.iter_mut().chain(self.outputs.iter_mut()) {
            for b in ports {
                b.set_active_frames(frames);
            }
        }
        master_out.set_active_frames(frames);

        let ctx = RenderContext {
            sample_rate: self.sample_rate,
            frames,
            playhead,
        };

        for idx in 0..self.order.len() {
            let i = self.order[idx];

            // Gather: clear this node's input ports, then sum upstream outputs.
            let in_count = self.inputs[i].len();
            for p in 0..in_count {
                self.inputs[i][p].clear();
            }
            for e in 0..self.edges.len() {
                let edge = &self.edges[e];
                if edge.to.node.0 != i {
                    continue;
                }
                let src = edge.from.node.0;
                let sport = edge.from.port;
                let dport = edge.to.port;
                let gain = edge.gain;
                // `inputs` and `outputs` are disjoint fields, so this dual
                // borrow is sound and allocation-free.
                let (dst_ports, src_ports) = (&mut self.inputs[i], &self.outputs[src]);
                dst_ports[dport].add_scaled(&src_ports[sport], gain);
            }

            // Process: hand the node its populated inputs and blank outputs.
            let mut io = ProcessIo {
                inputs: &self.inputs[i],
                outputs: &mut self.outputs[i],
            };
            self.nodes[i].node.process(&ctx, &mut io);
        }

        // Copy the designated master output into the caller's buffer.
        let master = self.master.expect("master validated at compile");
        master_out.copy_from(&self.outputs[master.node.0][master.port]);
    }

    /// Number of nodes in the graph.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is currently compiled and ready to process.
    #[inline]
    #[must_use]
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::ChannelLayout;

    /// A source node that emits a constant DC value on its single output.
    struct Dc(Sample);
    impl AudioNode for Dc {
        fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
            let out = io.output(0);
            for ch in 0..out.channels() {
                for s in out.channel_mut(ch) {
                    *s = self.0;
                }
            }
        }
    }

    /// A unity pass-through that copies input 0 to output 0.
    struct Passthrough;
    impl AudioNode for Passthrough {
        fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
            let (inp, out) = io.io(0, 0);
            out.copy_from(inp);
        }
    }

    #[test]
    fn two_sources_sum_at_a_mixer() {
        let mut g = AudioGraph::new(48_000, 64);
        let a = g.add_node(Box::new(Dc(0.25)), vec![], vec![ChannelLayout::Mono]);
        let b = g.add_node(Box::new(Dc(0.5)), vec![], vec![ChannelLayout::Mono]);
        let mix = g.add_node(
            Box::new(Passthrough),
            vec![ChannelLayout::Mono],
            vec![ChannelLayout::Mono],
        );
        g.connect(PortRef::new(a, 0), PortRef::new(mix, 0)).unwrap();
        g.connect(PortRef::new(b, 0), PortRef::new(mix, 0)).unwrap();
        g.set_master(PortRef::new(mix, 0)).unwrap();
        g.compile().unwrap();

        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        g.process(64, 0, &mut out);
        assert!((out.channel(0)[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn send_gain_scales_signal() {
        let mut g = AudioGraph::new(48_000, 16);
        let a = g.add_node(Box::new(Dc(1.0)), vec![], vec![ChannelLayout::Mono]);
        let mix = g.add_node(
            Box::new(Passthrough),
            vec![ChannelLayout::Mono],
            vec![ChannelLayout::Mono],
        );
        g.connect_with_gain(PortRef::new(a, 0), PortRef::new(mix, 0), 0.25)
            .unwrap();
        g.set_master(PortRef::new(mix, 0)).unwrap();
        g.compile().unwrap();
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 16);
        g.process(16, 0, &mut out);
        assert!((out.channel(0)[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn cycles_are_rejected() {
        let mut g = AudioGraph::new(48_000, 8);
        let a = g.add_node(
            Box::new(Passthrough),
            vec![ChannelLayout::Mono],
            vec![ChannelLayout::Mono],
        );
        let b = g.add_node(
            Box::new(Passthrough),
            vec![ChannelLayout::Mono],
            vec![ChannelLayout::Mono],
        );
        g.connect(PortRef::new(a, 0), PortRef::new(b, 0)).unwrap();
        g.connect(PortRef::new(b, 0), PortRef::new(a, 0)).unwrap();
        g.set_master(PortRef::new(b, 0)).unwrap();
        assert_eq!(g.compile(), Err(GraphError::Cycle));
    }

    #[test]
    fn layout_mismatch_is_rejected() {
        let mut g = AudioGraph::new(48_000, 8);
        let a = g.add_node(Box::new(Dc(1.0)), vec![], vec![ChannelLayout::Mono]);
        let b = g.add_node(
            Box::new(Passthrough),
            vec![ChannelLayout::Stereo],
            vec![ChannelLayout::Stereo],
        );
        let err = g.connect(PortRef::new(a, 0), PortRef::new(b, 0));
        assert!(matches!(err, Err(GraphError::LayoutMismatch { .. })));
    }
}
