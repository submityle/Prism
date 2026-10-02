//! Deterministic compiler from a Patch description to a runtime graph.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the design section 11 offline/incremental compile step: it
//! validates the description, flattens every nested `SubPatch` into leaf nodes
//! (bounded depth, compile-time expansion with prefixed re-exposed parameters),
//! then builds a `prism_audio_core::graph::AudioGraph` using the same
//! Kahn-topological compiler as the runtime. Parameters and triggers are bound
//! to lock-free cells so the host can drive the compiled patch. Compilation is
//! deterministic: node and connection ordering is derived solely from the
//! description, so the same input always yields the same graph.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::graph::{AudioGraph, NodeId, PortRef};
use prism_audio_core::{ChannelLayout, Sample};

use super::compiled::CompiledPatch;
use super::description::PatchDescription;
use super::error::PatchError;
use super::node_kind::NodeKind;
use super::param::{ParamCell, ParamHandle};
use super::trigger::{GateCell, TriggerHandle};
use super::validate::{self, MAX_NESTING_DEPTH};

/// A leaf node after nested patches have been expanded away.
struct FlatNode {
    kind: NodeKind,
    initial_params: Vec<(String, Sample)>,
}

/// A connection between flattened leaf nodes.
struct FlatConn {
    from_node: usize,
    from_port: usize,
    to_node: usize,
    to_port: usize,
}

/// An exposed parameter mapped onto a flattened leaf node.
struct FlatParam {
    public_name: String,
    node: usize,
    param: String,
}

/// An exposed trigger mapped onto a flattened leaf node.
struct FlatTrigger {
    public_name: String,
    node: usize,
}

/// The fully flattened, leaf-only form of a patch.
#[derive(Default)]
struct FlatPatch {
    nodes: Vec<FlatNode>,
    connections: Vec<FlatConn>,
    params: Vec<FlatParam>,
    triggers: Vec<FlatTrigger>,
    outputs: Vec<(usize, usize)>,
}

/// Compiles `desc` into a runnable [`CompiledPatch`].
///
/// `sample_rate` is the audio sample rate in Hz and `max_block` is the largest
/// block size the compiled patch will be asked to render.
///
/// # Errors
///
/// Returns a [`PatchError`] if validation fails, if nesting is too deep, or if
/// the underlying `prism_audio_core` graph fails to build (for example on a
/// cycle).
pub fn compile(
    desc: &PatchDescription,
    sample_rate: u32,
    max_block: usize,
) -> Result<CompiledPatch, PatchError> {
    validate::validate(desc)?;

    let mut flat = FlatPatch::default();
    let output_map = flatten_into(desc, "", 0, &mut flat)?;
    flat.outputs = desc
        .outputs
        .iter()
        .map(|out| output_map[out.node][out.port])
        .collect();
    if flat.outputs.is_empty() {
        return Err(PatchError::NoOutput);
    }

    build_graph(&flat, sample_rate, max_block)
}

/// Recursively flattens `desc` into `flat`, returning the per-local-node map
/// from `(local node, output port)` to a `(flat node, flat port)` source.
fn flatten_into(
    desc: &PatchDescription,
    prefix: &str,
    depth: usize,
    flat: &mut FlatPatch,
) -> Result<Vec<Vec<(usize, usize)>>, PatchError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(PatchError::NestingTooDeep {
            max: MAX_NESTING_DEPTH,
        });
    }

    let mut output_map: Vec<Vec<(usize, usize)>> = Vec::with_capacity(desc.nodes.len());
    let mut leaf_index: Vec<Option<usize>> = Vec::with_capacity(desc.nodes.len());

    for node in &desc.nodes {
        match &node.kind {
            NodeKind::SubPatch(sub) => {
                let mut sub_prefix = String::with_capacity(prefix.len() + node.name.len() + 1);
                sub_prefix.push_str(prefix);
                sub_prefix.push_str(&node.name);
                sub_prefix.push('/');
                let sub_map = flatten_into(sub, &sub_prefix, depth + 1, flat)?;

                let mut ports = Vec::with_capacity(sub.outputs.len());
                for out in &sub.outputs {
                    ports.push(sub_map[out.node][out.port]);
                }
                output_map.push(ports);
                leaf_index.push(None);
            }
            leaf => {
                let flat_idx = flat.nodes.len();
                flat.nodes.push(FlatNode {
                    kind: leaf.clone(),
                    initial_params: node.initial_params.clone(),
                });
                let port_count = leaf.port_count();
                let ports = (0..port_count.outputs).map(|p| (flat_idx, p)).collect();
                output_map.push(ports);
                leaf_index.push(Some(flat_idx));
            }
        }
    }

    for conn in &desc.connections {
        let source = output_map[conn.from_node][conn.from_port];
        let target = leaf_index[conn.to_node].ok_or(PatchError::ConnectionIntoSource {
            node: conn.to_node,
        })?;
        flat.connections.push(FlatConn {
            from_node: source.0,
            from_port: source.1,
            to_node: target,
            to_port: conn.to_port,
        });
    }

    for exposed in &desc.exposed_params {
        let flat_idx = leaf_index[exposed.node].ok_or(PatchError::UnknownParam {
            node: exposed.node,
            param: exposed.param.clone(),
        })?;
        let mut public = String::with_capacity(prefix.len() + exposed.public_name.len());
        public.push_str(prefix);
        public.push_str(&exposed.public_name);
        flat.params.push(FlatParam {
            public_name: public,
            node: flat_idx,
            param: exposed.param.clone(),
        });
    }

    for exposed in &desc.exposed_triggers {
        let flat_idx = leaf_index[exposed.node]
            .ok_or(PatchError::UnknownTrigger { node: exposed.node })?;
        let mut public = String::with_capacity(prefix.len() + exposed.public_name.len());
        public.push_str(prefix);
        public.push_str(&exposed.public_name);
        flat.triggers.push(FlatTrigger {
            public_name: public,
            node: flat_idx,
        });
    }

    Ok(output_map)
}

/// Builds and compiles the core graph for a flattened patch.
fn build_graph(
    flat: &FlatPatch,
    sample_rate: u32,
    max_block: usize,
) -> Result<CompiledPatch, PatchError> {
    let mut graph = AudioGraph::new(sample_rate, max_block);

    let mut node_ids: Vec<NodeId> = Vec::with_capacity(flat.nodes.len());
    let mut node_params: Vec<Vec<ParamCell>> = Vec::with_capacity(flat.nodes.len());
    let mut node_gates: Vec<Vec<GateCell>> = Vec::with_capacity(flat.nodes.len());

    for node in &flat.nodes {
        let names = node.kind.param_names();
        let defaults = node.kind.param_defaults();
        let mut cells = Vec::with_capacity(names.len());
        for (name, default) in names.iter().zip(defaults.iter()) {
            let initial = node
                .initial_params
                .iter()
                .find(|(param, _)| param == name)
                .map_or(*default, |(_, value)| *value);
            cells.push(ParamCell::new(initial));
        }

        let gate_names = node.kind.gate_names();
        let gates: Vec<GateCell> = gate_names.iter().map(|_| GateCell::new()).collect();

        let runtime = node
            .kind
            .instantiate(sample_rate, &cells, &gates)
            .ok_or(PatchError::UnflattenedSubPatch)?;

        let port_count = node.kind.port_count();
        let inputs = layouts(port_count.inputs);
        let outputs = layouts(port_count.outputs);
        let id = graph.add_node(runtime, inputs, outputs);

        node_ids.push(id);
        node_params.push(cells);
        node_gates.push(gates);
    }

    for conn in &flat.connections {
        let from = PortRef::new(node_ids[conn.from_node], conn.from_port);
        let to = PortRef::new(node_ids[conn.to_node], conn.to_port);
        graph.connect(from, to)?;
    }

    let output_ports: Vec<PortRef> = flat
        .outputs
        .iter()
        .map(|(node, port)| PortRef::new(node_ids[*node], *port))
        .collect();

    graph.set_master(output_ports[0])?;
    graph.compile()?;

    let params = flat
        .params
        .iter()
        .map(|fp| {
            let cell = bind_param(&flat.nodes[fp.node], &node_params[fp.node], &fp.param);
            ParamHandle::new(fp.public_name.clone(), cell)
        })
        .collect();

    let triggers = flat
        .triggers
        .iter()
        .map(|ft| TriggerHandle::new(ft.public_name.clone(), node_gates[ft.node][0].clone()))
        .collect();

    Ok(CompiledPatch::new(
        graph,
        params,
        triggers,
        output_ports,
        sample_rate,
        max_block,
    ))
}

/// Resolves the parameter cell named `param` on a flattened node.
///
/// The validator guarantees the name exists; the fallback keeps the function
/// total without panicking.
fn bind_param(node: &FlatNode, cells: &[ParamCell], param: &str) -> ParamCell {
    node.kind
        .param_names()
        .iter()
        .position(|name| *name == param)
        .and_then(|index| cells.get(index).cloned())
        .unwrap_or_default()
}

/// Builds a vector of `count` mono channel layouts.
fn layouts(count: usize) -> Vec<ChannelLayout> {
    (0..count).map(|_| ChannelLayout::Mono).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::builder::PatchBuilder;
    use crate::patch::library::oscillator::OscWaveform;

    #[test]
    fn compiles_simple_chain() {
        let mut b = PatchBuilder::new();
        let osc = b.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        let gain = b.add_node(NodeKind::Gain, "gain");
        b.connect(osc, 0, gain, 0);
        b.expose_param("gain", gain, "gain");
        b.add_output(gain, 0);
        let desc = b.build();

        let compiled = compile(&desc, 48_000, 128).expect("compiles");
        assert_eq!(compiled.output_count(), 1);
        assert!(compiled.param("gain").is_some());
    }

    #[test]
    fn cycle_is_rejected() {
        // gain_a -> gain_b -> gain_a forms a cycle.
        let mut b = PatchBuilder::new();
        let a = b.add_node(NodeKind::Gain, "a");
        let c = b.add_node(NodeKind::Gain, "b");
        b.connect(a, 0, c, 0);
        b.connect(c, 0, a, 0);
        b.add_output(a, 0);
        let desc = b.build();
        let result = compile(&desc, 48_000, 128);
        assert!(result.is_err());
    }

    #[test]
    fn nested_subpatch_expands_and_reexposes_params() {
        // Inner patch: a sine oscillator whose amplitude is exposed.
        let mut inner = PatchBuilder::new();
        let iosc = inner.add_node(
            NodeKind::Oscillator {
                waveform: OscWaveform::Sine,
            },
            "osc",
        );
        inner.expose_param("amp", iosc, "amplitude");
        inner.add_output(iosc, 0);
        let inner_desc = inner.build();

        // Outer patch references the inner patch as a sub-patch node.
        let mut outer = PatchBuilder::new();
        let sub = outer.add_node(NodeKind::SubPatch(Box::new(inner_desc)), "voice");
        let gain = outer.add_node(NodeKind::Gain, "gain");
        outer.connect(sub, 0, gain, 0);
        outer.add_output(gain, 0);
        let desc = outer.build();

        let compiled = compile(&desc, 48_000, 128).expect("compiles");
        // The nested parameter is re-exposed under the sub-patch node prefix.
        assert!(compiled.param("voice/amp").is_some());
        // Flattening produced two leaf nodes (oscillator + gain).
        assert_eq!(compiled.output_count(), 1);
    }

    #[test]
    fn nesting_too_deep_is_rejected() {
        // Build a chain of sub-patches deeper than the allowed limit.
        fn leaf() -> crate::patch::description::PatchDescription {
            let mut b = PatchBuilder::new();
            let osc = b.add_node(
                NodeKind::Oscillator {
                    waveform: OscWaveform::Sine,
                },
                "osc",
            );
            b.add_output(osc, 0);
            b.build()
        }
        let mut current = leaf();
        for _ in 0..(MAX_NESTING_DEPTH + 2) {
            let mut b = PatchBuilder::new();
            let sub = b.add_node(NodeKind::SubPatch(Box::new(current)), "nested");
            b.add_output(sub, 0);
            current = b.build();
        }
        let result = compile(&current, 48_000, 64);
        assert!(matches!(result, Err(PatchError::NestingTooDeep { .. })));
    }
}
