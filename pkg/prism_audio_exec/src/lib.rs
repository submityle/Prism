//! Compiled audio-graph execution planning for Prism's next-generation audio
//! engine.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. Every algorithm here
//! is a classic, well-documented graph technique (Kahn ordering, longest-path
//! layering, union-find components, linear-scan register allocation, and
//! workstation delay compensation) specialized to a float-free audio graph
//! description.
//!
//! # Relationship
//!
//! This crate is the planner behind the runtime graph in `prism_audio_core`.
//! It consumes a lightweight [`GraphDesc`] (which a host derives from its own
//! graph) and emits an immutable [`ExecPlan`] that fixes serial order, parallel
//! wavefronts, island partitioning, aliased buffer-slot allocation, and plugin
//! delay compensation. The core engine executes the plan every block without
//! re-planning, and `prism_audio_rt` swaps a freshly compiled plan when topology
//! changes. Keeping the planner integer-only guarantees the parallel schedule is
//! bit-identical to the serial render, satisfying the engine's determinism and
//! networking requirements.
//!
//! # Stages
//!
//! - [`graph_desc`]: the structural graph input and its validation.
//! - [`topo`]: deterministic topological ordering and cycle detection.
//! - [`levels`]: longest-path depth layering for parallelism.
//! - [`islands`]: weakly-connected component partitioning.
//! - [`liveness`]: output-buffer liveness and linear-scan slot allocation with
//!   in-place aliasing.
//! - [`pdc`]: plugin delay compensation.
//! - [`schedule`]: deterministic parallel wavefront schedule.
//! - [`exec_plan`]: the assembled, ready-to-run plan.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod exec_plan;
pub mod graph_desc;
pub mod islands;
pub mod levels;
pub mod liveness;
pub mod pdc;
pub mod schedule;
pub mod topo;

pub use exec_plan::ExecPlan;
pub use graph_desc::{EdgeDesc, ExecError, GraphDesc, NodeDesc, NodeId};
pub use islands::{IslandPartition, partition_islands};
pub use levels::{LevelAssignment, assign_levels};
pub use liveness::{BufferId, BufferLiveness, SlotAllocation, allocate_slots};
pub use pdc::{PdcPlan, compute_pdc};
pub use schedule::{ParallelSchedule, build_schedule};
pub use topo::{position_map, topological_order};

#[cfg(test)]
mod tests {
    use crate::{ExecPlan, EdgeDesc, GraphDesc, NodeDesc};

    /// End-to-end smoke test: a realistic small submix graph compiles and every
    /// derived view agrees on node counts.
    #[test]
    fn end_to_end_submix_graph() {
        let mut g = GraphDesc::new();
        let voice_a = g.add_node(NodeDesc::new(0, 1));
        let voice_b = g.add_node(NodeDesc::new(0, 1));
        let reverb = g.add_node(NodeDesc::new(1, 1).with_latency(48));
        let submix = g.add_node(NodeDesc::new(2, 1).with_in_place(true));
        let master = g.add_node(NodeDesc::new(2, 1));
        g.connect(EdgeDesc::new(voice_a, 0, submix, 0));
        g.connect(EdgeDesc::new(voice_b, 0, reverb, 0));
        g.connect(EdgeDesc::new(reverb, 0, submix, 1));
        g.connect(EdgeDesc::new(submix, 0, master, 0));
        g.connect(EdgeDesc::new(voice_a, 0, master, 1));
        g.set_master(master);

        let plan = ExecPlan::compile(&g).unwrap();
        assert_eq!(plan.order().len(), 5);
        assert_eq!(plan.schedule().node_count(), 5);
        assert_eq!(plan.islands().island_count(), 1);
        assert_eq!(plan.total_latency(), 48);
        assert!(plan.buffer_pool_size() >= 1);
    }
}
