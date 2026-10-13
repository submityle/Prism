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
use prism_render_driver::{
    BufferId, IndexFormat, RenderDevice, RenderPipelineDescriptor, RenderPipelineId,
    VertexBufferLayout,
};
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
    /// The resolved LOD level to draw.
    pub lod_level: u32,
    /// The geometry to draw.
    pub geometry: GeometryHandle,
    /// The vertex buffers bound at slots `0..vertices.len()`.
    pub vertices: Vec<BufferId>,
    /// The index buffer and format, or `None` for a non-indexed draw.
    pub indices: Option<(BufferId, IndexFormat)>,
    /// The first element (index or vertex) to draw.
    pub first: u32,
    /// The number of elements to draw.
    pub count: u32,
    /// The base vertex added to each index (`0` for non-indexed).
    pub base_vertex: i32,
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

/// The phase a bin belongs to, from the material permutation / bin class the
/// bin key carries.
///
/// A bin key stores the compact
/// [`SpecializationId::bin_class`](prism_render_material::SpecializationId::bin_class)
/// (`illumination` 2 bits | `render_class` 8 bits | live lobe set), so the
/// `render_class` — and therefore the phase — is recoverable without the
/// material record. The packed class is decoded in exactly one place
/// ([`SpecializationId::from_bin_class`]), the same inverse the bin path uses to
/// rebuild the pipeline key, so a phase can never disagree with the pipeline a
/// bin specializes to. Returns `None` for an unassigned/foreign class value.
///
/// [`SpecializationId::from_bin_class`]: prism_render_material::SpecializationId::from_bin_class
#[must_use]
pub fn phase_for_bin_class(bin_class: u32) -> Option<RenderPhase> {
    let specialization = prism_render_material::SpecializationId::from_bin_class(bin_class)?;
    MaterialRenderClass::from_u32(specialization.render_class_bits()).map(phase_for_render_class)
}

/// One queued draw: the pipeline to bind plus the mesh/material it draws.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PhaseItem {
    /// The specialized pipeline this draw binds.
    pub pipeline: RenderPipelineId,
    /// The draw function that records this item's commands.
    pub draw: DrawFunctionId,
    /// The geometry (mesh) to draw.
    pub geometry: GeometryHandle,
    /// The vertex buffers bound at slots `0..vertices.len()`.
    pub vertices: Vec<BufferId>,
    /// The index buffer and format, or `None` for a non-indexed draw.
    pub indices: Option<(BufferId, IndexFormat)>,
    /// The first element to draw.
    pub first: u32,
    /// The number of elements to draw.
    pub count: u32,
    /// The base vertex added to each index.
    pub base_vertex: i32,
    /// The indirect argument range, for the GPU-driven bin path.
    ///
    /// When set, the draw function issues one multi-draw from this range instead
    /// of the item's direct element range (the vertex/index buffers are still
    /// bound; the arguments supply counts, first index, and base vertex).
    pub indirect: Option<IndirectArgs>,
    /// The resolved LOD level this draw renders.
    pub lod_level: u32,
    /// The material whose bindings this draw uses.
    pub material: SceneMaterialHandle,
    /// The scene instance this draw came from.
    pub scene: SceneHandle,
    /// The caller's ordering key within the phase (e.g. a depth or bin key).
    pub order: u64,
}

/// A range of packed indirect argument records in a declared buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IndirectArgs {
    /// The argument buffer, declared in the frame graph (so it is synchronized).
    pub args: prism_render_graph::BufferHandle,
    /// The byte offset of the first record.
    pub offset: u64,
    /// The number of records to draw.
    pub count: u32,
    /// The byte stride between records (tightly packed indexed args).
    pub stride: u32,
}

/// The frame's queued draws, one list per [`RenderPhase`].
///
/// Cloning is cheap (the items are plain value structs), which is what lets a
/// pass closure own the queue it records from instead of borrowing it: a pass
/// outlives the borrow, and the frame graph must not hold a reference into the
/// render world.
#[derive(Clone, Resource)]
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
    pub fn record(
        &self,
        phase: RenderPhase,
        context: &mut dyn crate::MeshRecorder,
        functions: &DrawFunctions,
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
            let pipeline = specializer
                .specialize_with(device, request.variant, &request.layout, &build)
                .id();
            self.push(
                request.phase,
                PhaseItem {
                    pipeline,
                    draw: request.draw,
                    geometry: request.geometry,
                    vertices: request.vertices,
                    indices: request.indices,
                    first: request.first,
                    count: request.count,
                    base_vertex: request.base_vertex,
                    indirect: None,
                    lod_level: request.lod_level,
                    material: request.material,
                    scene: request.scene,
                    order: request.order,
                },
            );
        }
    }

    /// Resolves the frame's visible work and queues it.
    ///
    /// This is the queue stage end to end: for each visible
    /// [`GpuRenderWorkItem`](prism_render_visibility::GpuRenderWorkItem), the
    /// [`SceneRegistry`](crate::SceneRegistry) resolves it into a
    /// [`DrawRequest`](crate::DrawRequest), the [`MeshPipelineSpecializer`]
    /// specializes it, and the item lands in its phase. Returns the number of
    /// draws queued; a work item whose geometry/material is not resident is
    /// skipped.
    pub fn fill_from_visibility<D: RenderDevice, P: crate::MeshPipelineProvider>(
        &mut self,
        device: &D,
        specializer: &mut MeshPipelineSpecializer,
        registry: &crate::SceneRegistry,
        meshes: &crate::RenderAssets<crate::RenderMesh>,
        work: &[prism_render_visibility::GpuRenderWorkItem],
        depth_format: Option<prism_render_driver::TextureFormat>,
        sample_count: u32,
        draw: DrawFunctionId,
        provider: &P,
    ) -> usize {
        let mut queued = 0;
        for item in work {
            let Some(request) = registry.resolve(item, meshes, depth_format, sample_count, draw)
            else {
                continue;
            };
            let pipeline = specializer
                .specialize_with_provider(device, request.variant, &request.layout, provider)
                .id();
            self.push(
                request.phase,
                PhaseItem {
                    pipeline,
                    draw: request.draw,
                    geometry: request.geometry,
                    vertices: request.vertices,
                    indices: request.indices,
                    first: request.first,
                    count: request.count,
                    base_vertex: request.base_vertex,
                    indirect: None,
                    lod_level: request.lod_level,
                    material: request.material,
                    scene: request.scene,
                    order: request.order,
                },
            );
            queued += 1;
        }
        queued
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
