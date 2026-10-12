//! The mesh queue: the draw items a frame's phases will record.
//!
//! Bevy's render world turns the visible set into render-phase items (a
//! `queue_material_meshes`-style stage): each item is a fully specialized draw —
//! the pipeline to bind, the mesh, and the material. Prism's equivalent is
//! [`RenderQueues`]: a set of per-phase item lists the queue stage fills from
//! the visible/binned set, then sorts deterministically so a frame records the
//! same order every run.
//!
//! The queue is phased by [`RenderPhase`] (opaque / transparent / stylized /
//! custom) — the material-model-agnostic phases of the single execution path —
//! not by material model, so PBR / NPR / hybrid / custom draws share one queue
//! and differ only in the pipeline they specialize to.

use prism_ecs::prelude::Resource;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};
use prism_render_driver::RenderPipelineId;

use crate::sets::RenderPhase;

/// One queued draw: the pipeline to bind plus the mesh/material it draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PhaseItem {
    /// The specialized pipeline this draw binds.
    pub pipeline: RenderPipelineId,
    /// The geometry (mesh) to draw.
    pub geometry: GeometryHandle,
    /// The material whose bindings this draw uses.
    pub material: SceneMaterialHandle,
    /// The scene instance this draw came from.
    pub scene: SceneHandle,
    /// The caller's ordering key within the phase (e.g. a depth or bin key).
    pub order: u64,
}

/// The frame's queued draws, one list per [`RenderPhase`].
#[derive(Resource)]
pub struct RenderQueues {
    phases: [Vec<PhaseItem>; RenderPhase::COUNT],
}

impl Default for RenderQueues {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderQueues {
    /// Empty queues for every phase.
    #[must_use]
    pub fn new() -> Self {
        Self {
            phases: core::array::from_fn(|_| Vec::new()),
        }
    }

    /// Drops every queued item, ready for the next frame.
    pub fn clear(&mut self) {
        for phase in &mut self.phases {
            phase.clear();
        }
    }

    /// Appends `item` to `phase`'s list.
    pub fn push(&mut self, phase: RenderPhase, item: PhaseItem) {
        self.phases[phase.index()].push(item);
    }

    /// The items queued for `phase`, in queue order.
    #[must_use]
    pub fn phase(&self, phase: RenderPhase) -> &[PhaseItem] {
        &self.phases[phase.index()]
    }

    /// Sorts every phase deterministically.
    ///
    /// The primary key is [`PhaseItem::order`]; ties break on
    /// `(pipeline, geometry, material, scene)`, so the recorded order does not
    /// depend on the order the visible set happened to be walked in.
    pub fn sort(&mut self) {
        for phase in &mut self.phases {
            phase.sort_by_key(|item| {
                (
                    item.order,
                    item.pipeline,
                    item.geometry,
                    item.material,
                    item.scene,
                )
            });
        }
    }

    /// The total number of queued items across every phase.
    #[must_use]
    pub fn len(&self) -> usize {
        self.phases.iter().map(Vec::len).sum()
    }

    /// Whether no items are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
