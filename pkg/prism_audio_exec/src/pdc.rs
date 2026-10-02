//! Plugin delay compensation (PDC).
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. Each node has an
//! intrinsic processing latency; summing latencies along every path and aligning
//! the arrivals at each input is the standard digital-audio-workstation delay
//! compensation computation, here expressed over integer frame counts.
//!
//! # Relationship
//!
//! Without compensation, a node fed by two paths of unequal latency would phase
//! or flam. The planner computes, for every node, the latency seen at its inputs
//! and outputs, then the per-edge compensating delay that realigns a shorter path
//! to the longest one reaching the same consumer. The total graph latency is the
//! output latency of the master node; the host reports it so gameplay timing and
//! recording stay sample-aligned.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::graph_desc::{GraphDesc, NodeId};

/// Latency figures and per-edge compensation for a scheduled graph.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PdcPlan {
    input_latency: Vec<u32>,
    output_latency: Vec<u32>,
    edge_delay: Vec<u32>,
    total_latency: u32,
}

impl PdcPlan {
    /// Returns the latency, in frames, observed at each node's inputs: the
    /// maximum output latency among its upstream producers.
    #[must_use]
    pub fn input_latency(&self) -> &[u32] {
        &self.input_latency
    }

    /// Returns the latency at each node's outputs: its input latency plus its own
    /// processing latency.
    #[must_use]
    pub fn output_latency(&self) -> &[u32] {
        &self.output_latency
    }

    /// Returns the compensating delay to insert on each edge, indexed by the
    /// edge's position in [`GraphDesc::edges`]. The delay realigns a producer that
    /// finishes earlier than the slowest producer feeding the same consumer.
    #[must_use]
    pub fn edge_delay(&self) -> &[u32] {
        &self.edge_delay
    }

    /// Returns the output latency of a node.
    #[must_use]
    pub fn output_latency_of(&self, node: NodeId) -> u32 {
        self.output_latency[node.0]
    }

    /// Returns the end-to-end latency of the graph (master output latency).
    #[must_use]
    pub fn total_latency(&self) -> u32 {
        self.total_latency
    }
}

/// Computes delay compensation for `desc` given a topological `order`.
///
/// # Panics
///
/// Panics only if `desc` has no master set; callers should run
/// [`GraphDesc::validate`] (as [`crate::exec_plan::ExecPlan::compile`] does)
/// first, which rejects that case with a recoverable error.
#[must_use]
pub fn compute_pdc(desc: &GraphDesc, order: &[NodeId]) -> PdcPlan {
    let n = desc.node_count();
    let mut input_latency = vec![0_u32; n];
    let mut output_latency = vec![0_u32; n];

    // Propagate in topological order so every producer is finalized first.
    for &node in order {
        let mut in_lat = 0_u32;
        for edge in desc.edges() {
            if edge.to_node == node {
                let producer = output_latency[edge.from_node.0];
                if producer > in_lat {
                    in_lat = producer;
                }
            }
        }
        input_latency[node.0] = in_lat;
        output_latency[node.0] = in_lat + desc.nodes()[node.0].latency_frames;
    }

    // Per-edge compensation: delay the producer up to the consumer's input
    // latency so all paths into that consumer arrive aligned.
    let edge_delay = desc
        .edges()
        .iter()
        .map(|edge| input_latency[edge.to_node.0] - output_latency[edge.from_node.0])
        .collect();

    let total_latency = desc.master().map_or(0, |m| output_latency[m.0]);

    PdcPlan {
        input_latency,
        output_latency,
        edge_delay,
        total_latency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_desc::{EdgeDesc, NodeDesc};
    use crate::topo::topological_order;

    #[test]
    fn chain_sums_latencies() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1).with_latency(10));
        let b = g.add_node(NodeDesc::new(1, 1).with_latency(20));
        let c = g.add_node(NodeDesc::new(1, 1).with_latency(5));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, c, 0));
        g.set_master(c);
        let order = topological_order(&g).unwrap();
        let pdc = compute_pdc(&g, &order);
        assert_eq!(pdc.output_latency_of(a), 10);
        assert_eq!(pdc.output_latency_of(b), 30);
        assert_eq!(pdc.output_latency_of(c), 35);
        assert_eq!(pdc.total_latency(), 35);
        assert_eq!(pdc.edge_delay(), &[0, 0]);
    }

    #[test]
    fn unequal_paths_are_compensated() {
        // a -> b (latency 100) -> d, and a -> c (latency 0) -> d.
        // The c path must be delayed by 100 frames to match the b path at d.
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1).with_latency(100));
        let c = g.add_node(NodeDesc::new(1, 1));
        let d = g.add_node(NodeDesc::new(2, 1));
        let e_ab = EdgeDesc::new(a, 0, b, 0);
        let e_ac = EdgeDesc::new(a, 0, c, 0);
        let e_bd = EdgeDesc::new(b, 0, d, 0);
        let e_cd = EdgeDesc::new(c, 0, d, 1);
        g.connect(e_ab);
        g.connect(e_ac);
        g.connect(e_bd);
        g.connect(e_cd);
        g.set_master(d);
        let order = topological_order(&g).unwrap();
        let pdc = compute_pdc(&g, &order);
        assert_eq!(pdc.input_latency()[d.0], 100);
        // edges order: ab, ac, bd, cd
        assert_eq!(pdc.edge_delay(), &[0, 0, 0, 100]);
        assert_eq!(pdc.total_latency(), 100);
    }

    #[test]
    fn zero_latency_graph_needs_no_compensation() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.set_master(b);
        let order = topological_order(&g).unwrap();
        let pdc = compute_pdc(&g, &order);
        assert_eq!(pdc.total_latency(), 0);
        assert!(pdc.edge_delay().iter().all(|&d| d == 0));
    }
}
