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
use prism_render_driver::{RenderDevice, RenderPipelineDescriptor, RenderPipelineId, VertexBufferLayout};
use prism_render_material::MaterialRenderClass;

use crate::draw::{DrawFunctionId, DrawFunctions};
use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineSpecializer};
use crate::sets::RenderPhase;

/// A resolved visible draw ready to queue: the scene has already resolved the
/// work item's geometry layout and material into a variant + phase.
///
/// The queue specializes it to a pipeline (through
/// [`MeshPipelineSpecializer`]) and appends a [`PhaseItem`].
pub struct DrawRequest {
    /// The phase the draw belongs to (from
    /// [`phase_for_render_class`]).
    pub phase: RenderPhase,
    /// The mesh pipeline variant (material class + fixed-function state).
    pub variant: MeshPipelineKey,
    /// The mesh's vertex layouts, folded into the pipeline key.
    pub layout: Vec<VertexBufferLayout>,
    /// The geometry to draw.
    pub geometry: GeometryHandle,
    /// The material whose bindings the draw uses.
    pub material: SceneMaterialHandle,
    /// The scene instance the draw came from.
    pub scene: SceneHandle,
    /// The caller's ordering key within the phase.
    pub order: u64,
    /// The draw function that records the item's commands.
    pub draw: DrawFunctionId,
}

/// The phase a surface draw belongs to, from its material's blend family.
///
/// Selection is by [`MaterialRenderClass`] **only**. `illumination`
/// (Lit / Stylized / Unlit / Custom) is the *response* axis
/// (`prism_material_pipeline_refactor_zh.md` §3.5/§8.5), not a phase: a stylized
/// or custom surface draws in the same opaque/transparent phase as a lit one and
/// differs only in the response its shader dispatches to. The
/// [`RenderPhase::Stylized`] and [`RenderPhase::Custom`] phases host extra
/// *passes* (an NPR outline pass, a project-injected pass), not the surface
/// draws of those materials.
#[must_use]
pub fn phase_for_render_class(class: MaterialRenderClass) -> RenderPhase {
    match class {
        MaterialRenderClass::Opaque
        | MaterialRenderClass::OpaqueTwoSided
        | MaterialRenderClass::Masked
        | MaterialRenderClass::MaskedTwoSided
        | MaterialRenderClass::Hair
        | MaterialRenderClass::Water => RenderPhase::Opaque,
        MaterialRenderClass::Transmissive
        | MaterialRenderClass::Transparent
        | MaterialRenderClass::Additive
        | MaterialRenderClass::Decal => RenderPhase::Transparent,
        // A volume is not a surface mesh draw; it is a project-injected pass.
        MaterialRenderClass::Volume => RenderPhase::Custom,
    }
}

/// One queued draw: the pipeline to bind plus the mesh/material it draws.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PhaseItem {
    /// The specialized pipeline this draw binds.
    pub pipeline: RenderPipelineId,
    /// The draw function that records this item's commands.
    pub draw: DrawFunctionId,
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

    /// Records every queued item of `phase` into `context`.
    ///
    /// Items are recorded in queue order (call [`sort`](Self::sort) first to
    /// make that deterministic), each through its own [`DrawFunctionId`].
    pub fn record<C>(
        &self,
        phase: RenderPhase,
        context: &mut C,
        functions: &DrawFunctions<C>,
    ) {
        for item in self.phase(phase) {
            functions.call(item.draw, item, context);
        }
    }

    /// Specializes and queues each resolved request.
    ///
    /// `build` supplies the pipeline descriptor for a variant (it needs the
    /// renderer's shader and target state, which the queue does not own). The
    /// specializer dedups per `(variant, layout)`, so two meshes that share a
    /// variant and layout share one pipeline.
    pub fn fill<D: RenderDevice>(
        &mut self,
        device: &D,
        specializer: &mut MeshPipelineSpecializer,
        requests: impl IntoIterator<Item = DrawRequest>,
        build: impl Fn(&MeshPipelineKey) -> RenderPipelineDescriptor,
    ) {
        for request in requests {
            let descriptor = build(&request.variant);
            let pipeline = specializer.specialize(
                device,
                request.variant,
                &request.layout,
                &descriptor,
            );
            self.push(
                request.phase,
                PhaseItem {
                    pipeline,
                    draw: request.draw,
                    geometry: request.geometry,
                    material: request.material,
                    scene: request.scene,
                    order: request.order,
                },
            );
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
