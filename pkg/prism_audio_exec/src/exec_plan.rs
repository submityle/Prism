//! Assembled execution plan for a compiled audio graph.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. This module performs
//! no new algorithm; it validates the graph once and composes the independent
//! analyses (`topo`, `schedule`, `islands`, `liveness`, `pdc`) into a single
//! immutable artifact the runtime can execute directly.
//!
//! # Relationship
//!
//! An [`ExecPlan`] is to the audio renderer what a compiled render graph is to a
//! GPU frame: a one-time compilation output that fixes execution order, buffer
//! aliasing, parallel structure, and latency so the per-block hot path performs
//! no planning. A host converts its own graph into a [`GraphDesc`], compiles it
//! once, recompiles only on topology change, and reuses the plan every block.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::graph_desc::{ExecError, GraphDesc, NodeId};
use crate::islands::{IslandPartition, partition_islands};
use crate::liveness::{SlotAllocation, allocate_slots};
use crate::pdc::{PdcPlan, compute_pdc};
use crate::schedule::{ParallelSchedule, build_schedule};
use crate::topo::topological_order;

/// Immutable, ready-to-run plan for one compiled graph topology.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExecPlan {
    order: Vec<NodeId>,
    schedule: ParallelSchedule,
    islands: IslandPartition,
    allocation: SlotAllocation,
    pdc: PdcPlan,
    master: NodeId,
    master_slot: usize,
}

impl ExecPlan {
    /// Compiles a graph description into an execution plan.
    ///
    /// # Errors
    ///
    /// Propagates [`GraphDesc::validate`] failures (unknown node, port out of
    /// range, missing master) and returns [`ExecError::Cycle`] for a non-DAG.
    pub fn compile(desc: &GraphDesc) -> Result<Self, ExecError> {
        let order = topological_order(desc)?;
        let master = desc.master().ok_or(ExecError::NoMaster)?;

        let schedule = build_schedule(desc, &order);
        let islands = partition_islands(desc);
        let allocation = allocate_slots(desc, &order);
        let pdc = compute_pdc(desc, &order);

        // The master's first output buffer holds the final mix slot; a master
        // with no output ports (silent graph) reports slot 0 of an empty pool.
        let master_slot = allocation
            .buffer_of(master, 0)
            .map_or(0, |buffer| buffer.slot);

        Ok(Self {
            order,
            schedule,
            islands,
            allocation,
            pdc,
            master,
            master_slot,
        })
    }

    /// Returns the serial topological execution order.
    #[must_use]
    pub fn order(&self) -> &[NodeId] {
        &self.order
    }

    /// Returns the parallel wavefront schedule.
    #[must_use]
    pub fn schedule(&self) -> &ParallelSchedule {
        &self.schedule
    }

    /// Returns the island partition.
    #[must_use]
    pub fn islands(&self) -> &IslandPartition {
        &self.islands
    }

    /// Returns the buffer-slot allocation.
    #[must_use]
    pub fn allocation(&self) -> &SlotAllocation {
        &self.allocation
    }

    /// Returns the delay-compensation plan.
    #[must_use]
    pub fn pdc(&self) -> &PdcPlan {
        &self.pdc
    }

    /// Returns the master output node.
    #[must_use]
    pub fn master(&self) -> NodeId {
        self.master
    }

    /// Returns the physical slot holding the final rendered mix.
    #[must_use]
    pub fn master_slot(&self) -> usize {
        self.master_slot
    }

    /// Returns the number of physical buffer slots the runtime must allocate.
    #[must_use]
    pub fn buffer_pool_size(&self) -> usize {
        self.allocation.pool_size()
    }

    /// Returns the end-to-end graph latency in frames.
    #[must_use]
    pub fn total_latency(&self) -> u32 {
        self.pdc.total_latency()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_desc::{EdgeDesc, NodeDesc};

    fn mixer_graph() -> GraphDesc {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(0, 1).with_latency(32));
        let b = g.add_node(NodeDesc::new(0, 1));
        let gain = g.add_node(NodeDesc::new(1, 1).with_in_place(true));
        let mix = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(a, 0, gain, 0));
        g.connect(EdgeDesc::new(gain, 0, mix, 0));
        g.connect(EdgeDesc::new(b, 0, mix, 1));
        g.set_master(mix);
        g
    }

    #[test]
    fn compiles_consistent_plan() {
        let g = mixer_graph();
        let plan = ExecPlan::compile(&g).unwrap();
        assert_eq!(plan.order().len(), g.node_count());
        assert_eq!(plan.schedule().node_count(), g.node_count());
        assert_eq!(plan.master(), NodeId(3));
        assert!(plan.buffer_pool_size() >= 2);
        // Master mix slot is the one holding mix's output buffer.
        let mix_buf = plan.allocation().buffer_of(NodeId(3), 0).unwrap();
        assert_eq!(plan.master_slot(), mix_buf.slot);
        // a has 32 frames latency; the b path into the mixer is compensated.
        assert_eq!(plan.total_latency(), 32);
    }

    #[test]
    fn rejects_cycle() {
        let mut g = GraphDesc::new();
        let a = g.add_node(NodeDesc::new(1, 1));
        let b = g.add_node(NodeDesc::new(1, 1));
        g.connect(EdgeDesc::new(a, 0, b, 0));
        g.connect(EdgeDesc::new(b, 0, a, 0));
        g.set_master(b);
        assert_eq!(ExecPlan::compile(&g), Err(ExecError::Cycle));
    }

    #[test]
    fn rejects_missing_master() {
        let mut g = GraphDesc::new();
        g.add_node(NodeDesc::new(0, 1));
        assert_eq!(ExecPlan::compile(&g), Err(ExecError::NoMaster));
    }

    #[test]
    fn single_island_for_connected_graph() {
        let g = mixer_graph();
        let plan = ExecPlan::compile(&g).unwrap();
        assert_eq!(plan.islands().island_count(), 1);
    }
}
