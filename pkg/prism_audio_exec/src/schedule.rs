//! Deterministic parallel wavefront schedule.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. Nodes are grouped by
//! their longest-path level (see `levels`): every node in a wavefront is
//! independent of its siblings, so a worker pool can run them concurrently, and
//! wavefronts execute in order. Membership within a wavefront is sorted by node
//! index so a run is reproducible irrespective of worker timing.
//!
//! # Relationship
//!
//! This turns the structural analysis into an executable plan for a job-based
//! renderer. Islands (see `islands`) are orthogonal: nodes from different islands
//! that land on the same wavefront are simply more independent jobs. The schedule
//! deliberately contains no floating-point work, guaranteeing the parallel render
//! is bit-identical to the serial topological render.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::graph_desc::{GraphDesc, NodeId};
use crate::levels::assign_levels;

/// Ordered set of parallel wavefronts.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParallelSchedule {
    wavefronts: Vec<Vec<NodeId>>,
}

impl ParallelSchedule {
    /// Returns the wavefronts in execution order; nodes within a wavefront may
    /// run concurrently and are listed in ascending index order.
    #[must_use]
    pub fn wavefronts(&self) -> &[Vec<NodeId>] {
        &self.wavefronts
    }

    /// Returns the number of sequential wavefronts (the critical-path length).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.wavefronts.len()
    }

    /// Returns the widest wavefront size, i.e. the maximum concurrency the graph
    /// can exploit.
    #[must_use]
    pub fn max_width(&self) -> usize {
        self.wavefronts.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Returns the total number of scheduled nodes across all wavefronts.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.wavefronts.iter().map(Vec::len).sum()
    }
}

/// Builds a parallel schedule from a graph and a valid topological order.
#[must_use]
pub fn build_schedule(desc: &GraphDesc, order: &[NodeId]) -> ParallelSchedule {
    let levels = assign_levels(desc, order);
    let depth = levels.depth() as usize;
    let mut wavefronts: Vec<Vec<NodeId>> = (0..depth).map(|_| Vec::new()).collect();
    // Emit nodes in ascending index order so each wavefront is deterministically
    // ordered without a separate sort.
    for (node_idx, &lvl) in levels.levels().iter().enumerate() {
        wavefronts[lvl as usize].push(NodeId(node_idx));
    }
    ParallelSchedule { wavefronts }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_desc::{EdgeDesc, NodeDesc};
    use crate::topo::topological_order;

    #[test]
    fn chain_is_fully_serial() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, c, 0));
        g.set_master(c);
        let order = topological_order(&g).unwrap();
        let sched = build_schedule(&g, &order);
        assert_eq!(sched.depth(), 3);
        assert_eq!(sched.max_width(), 1);
        assert_eq!(sched.node_count(), 3);
    }

    #[test]
    fn independent_sources_share_first_wavefront() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(0, 1));
        let mix = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(a, 0, mix, 0));
        g.connect(EdgeDesc::new(b, 0, mix, 1));
        g.set_master(mix);
        let order = topological_order(&g).unwrap();
        let sched = build_schedule(&g, &order);
        assert_eq!(sched.depth(), 2);
        assert_eq!(sched.wavefronts()[0], vec![NodeId(0), NodeId(1)]);
        assert_eq!(sched.wavefronts()[1], vec![NodeId(2)]);
        assert_eq!(sched.max_width(), 2);
    }

    #[test]
    fn every_node_scheduled_exactly_once() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        let c = g.add_node(NodeDesc::new(1, 1));
        let d = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(a, 0, c, 0));
        g.connect(EdgeDesc::new(b, 0, d, 0));
        g.connect(EdgeDesc::new(c, 0, d, 1));
        g.set_master(d);
        let order = topological_order(&g).unwrap();
        let sched = build_schedule(&g, &order);
        assert_eq!(sched.node_count(), g.node_count());
    }
}
