//! The compiled, runnable product of a Patch description.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Holds the output of the design section 11 compiler: a built
//! `prism_audio_core::graph::AudioGraph` plus the host-facing parameter and
//! trigger handles and the designated output ports. [`CompiledPatch`] renders
//! standalone, while [`CompiledPatch::into_node`] yields a [`PatchNode`] that
//! implements `prism_audio_core::graph::AudioNode`, so a compiled patch becomes
//! an ordinary, sample-accurate node inside a larger runtime graph (the
//! "compile a Patch into a node" requirement).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::graph::{AudioGraph, AudioNode, PortRef, ProcessIo, RenderContext};
use prism_audio_core::{AudioBuffer, ChannelLayout};

use super::param::ParamHandle;
use super::trigger::TriggerHandle;

/// A compiled patch: a built graph plus its host-facing handles and outputs.
pub struct CompiledPatch {
    graph: AudioGraph,
    params: Vec<ParamHandle>,
    triggers: Vec<TriggerHandle>,
    outputs: Vec<PortRef>,
    sample_rate: u32,
    max_block: usize,
}

impl CompiledPatch {
    /// Assembles a compiled patch from its parts (constructed by the compiler).
    #[must_use]
    pub fn new(
        graph: AudioGraph,
        params: Vec<ParamHandle>,
        triggers: Vec<TriggerHandle>,
        outputs: Vec<PortRef>,
        sample_rate: u32,
        max_block: usize,
    ) -> Self {
        Self {
            graph,
            params,
            triggers,
            outputs,
            sample_rate,
            max_block,
        }
    }

    /// Returns the exposed parameter handle named `name`, if present.
    #[must_use]
    pub fn param(&self, name: &str) -> Option<&ParamHandle> {
        self.params.iter().find(|handle| handle.name() == name)
    }

    /// Returns the exposed trigger handle named `name`, if present.
    #[must_use]
    pub fn trigger(&self, name: &str) -> Option<&TriggerHandle> {
        self.triggers.iter().find(|handle| handle.name() == name)
    }

    /// Returns all exposed parameter handles.
    #[must_use]
    pub fn params(&self) -> &[ParamHandle] {
        &self.params
    }

    /// Returns all exposed trigger handles.
    #[must_use]
    pub fn triggers(&self) -> &[TriggerHandle] {
        &self.triggers
    }

    /// Returns the number of audio outputs the patch declares.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Returns the sample rate the patch was compiled for.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the largest block size the patch was compiled for.
    #[must_use]
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Borrows the underlying graph mutably (for advanced host integration).
    pub fn graph_mut(&mut self) -> &mut AudioGraph {
        &mut self.graph
    }

    /// Renders `frames` of the primary output into `master_out`.
    ///
    /// `master_out` must be a mono buffer with capacity for at least `frames`.
    pub fn process(&mut self, frames: usize, playhead: u64, master_out: &mut AudioBuffer) {
        self.graph.process(frames, playhead, master_out);
    }

    /// Converts the compiled patch into a runtime [`PatchNode`].
    ///
    /// Clone any handles you need from [`CompiledPatch::params`] or
    /// [`CompiledPatch::triggers`] before calling this, since the returned node
    /// owns the graph.
    #[must_use]
    pub fn into_node(self) -> PatchNode {
        PatchNode::new(self.graph, self.max_block)
    }
}

/// A compiled patch wrapped as a single `AudioNode` for a larger graph.
///
/// It exposes zero audio inputs and one mono audio output carrying the patch's
/// primary output. Parameters and triggers are driven through the handles taken
/// from the originating [`CompiledPatch`].
pub struct PatchNode {
    graph: AudioGraph,
    scratch: AudioBuffer,
}

impl PatchNode {
    /// Wraps `graph`, pre-allocating a mono scratch buffer of `max_block`.
    #[must_use]
    pub fn new(graph: AudioGraph, max_block: usize) -> Self {
        Self {
            graph,
            scratch: AudioBuffer::new(ChannelLayout::Mono, max_block.max(1)),
        }
    }
}

impl AudioNode for PatchNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let frames = out.active_frames();
        self.graph.process(frames, ctx.playhead, &mut self.scratch);
        let dst = out.channel_mut(0);
        let src = self.scratch.channel(0);
        dst.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::builder::PatchBuilder;
    use crate::patch::compiler::compile;
    use crate::patch::node_kind::NodeKind;

    #[test]
    fn standalone_render_produces_signal() {
        let mut b = PatchBuilder::new();
        let osc = b.add_node(
            NodeKind::Oscillator {
                waveform: crate::patch::library::oscillator::OscWaveform::Sine,
            },
            "osc",
        );
        b.set_param(osc, "frequency", 440.0);
        b.set_param(osc, "amplitude", 0.5);
        b.add_output(osc, 0);
        let desc = b.build();

        let mut compiled = compile(&desc, 48_000, 256).expect("compiles");
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 256);
        compiled.process(256, 0, &mut out);
        let peak = out
            .channel(0)
            .iter()
            .fold(0.0_f32, |acc, s| acc.max(s.abs()));
        assert!(peak > 0.1);
    }

    #[test]
    fn param_change_silences_output() {
        use crate::patch::library::oscillator::OscWaveform;
        let mut b = PatchBuilder::new();
        let osc = b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        b.expose_param("amp", osc, "amplitude");
        b.add_output(osc, 0);
        let desc = b.build();

        let mut compiled = compile(&desc, 48_000, 256).expect("compiles");
        let amp = compiled.param("amp").expect("amp handle").clone();
        amp.set(0.0);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 256);
        // Let the amplitude smoother settle to zero across several blocks.
        for _ in 0..8 {
            compiled.process(256, 0, &mut out);
        }
        let peak = out
            .channel(0)
            .iter()
            .fold(0.0_f32, |acc, s| acc.max(s.abs()));
        assert!(peak < 1.0e-3);
    }

    #[test]
    fn gate_envelope_opens_on_trigger() {
        use crate::modulation::envelope::EnvelopeConfig;
        use crate::patch::library::oscillator::OscWaveform;
        let mut b = PatchBuilder::new();
        let osc = b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        let env = b.add_node(
            NodeKind::AdsrAmp {
                config: EnvelopeConfig::default(),
            },
            "env",
        );
        b.connect(osc, 0, env, 0);
        b.expose_trigger("gate", env);
        b.add_output(env, 0);
        let desc = b.build();

        let mut compiled = compile(&desc, 48_000, 256).expect("compiles");
        let gate = compiled.trigger("gate").expect("gate handle").clone();
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 256);

        // Before any trigger the envelope is idle and output is silent.
        compiled.process(256, 0, &mut out);
        let peak_before = out
            .channel(0)
            .iter()
            .fold(0.0_f32, |acc, s| acc.max(s.abs()));
        assert!(peak_before < 1.0e-4);

        // After a note-on the envelope opens and the signal passes through.
        gate.trigger();
        let mut peak_after = 0.0_f32;
        for _ in 0..4 {
            compiled.process(256, 0, &mut out);
            let p = out
                .channel(0)
                .iter()
                .fold(0.0_f32, |acc, s| acc.max(s.abs()));
            peak_after = peak_after.max(p);
        }
        assert!(peak_after > 0.1);
    }

    #[test]
    fn wrapped_patch_node_renders_into_output() {
        use crate::patch::library::oscillator::OscWaveform;
        let mut b = PatchBuilder::new();
        let osc = b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        b.set_param(osc, "amplitude", 0.5);
        b.add_output(osc, 0);
        let desc = b.build();

        let compiled = compile(&desc, 48_000, 128).expect("compiles");
        let mut node = compiled.into_node();

        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 128)];
        outputs[0].set_active_frames(128);
        let ctx = RenderContext {
            sample_rate: 48_000,
            frames: 128,
            playhead: 0,
        };
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }
        let peak = outputs[0]
            .channel(0)
            .iter()
            .fold(0.0_f32, |acc, s| acc.max(s.abs()));
        assert!(peak > 0.1);
    }
}
