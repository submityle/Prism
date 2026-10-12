//! The engine-neutral scene registry and work-item resolution.
//!
//! The visibility stage produces [`GpuRenderWorkItem`]s — "scene instance
//! `S` draws geometry `G` at LOD `L` with material `M`" — but a work item
//! carries only handles and classes, not the data a pipeline needs. This module
//! owns that data on the render side (the geometry / material records and the
//! vertex-buffer-class table) and resolves a work item into a [`DrawRequest`]
//! the queue can specialize and enqueue.
//!
//! It is the Prism-side replacement for the renderer's geometry/material
//! registries; because the records are the engine-neutral
//! `prism_render_architecture` / `prism_render_material` ones, no ECS or graphics
//! API types leak into the resolution.

use alloc::collections::BTreeMap;

use prism_ecs::prelude::Resource;
use prism_render_architecture::geometry::{GeometryPrimitiveKind, GeometryRecord};
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
use prism_render_driver::{
    BufferId, IndexFormat, PrimitiveTopology, TextureFormat, VertexBufferLayout,
};
use prism_render_material::MaterialRecord;
use prism_render_visibility::{DrawBinCandidate, DrawBinKey, GpuRenderWorkItem};

use crate::draw::DrawFunctionId;
use crate::frame::RENDER_TARGET_FORMAT;
use crate::mesh_pipeline::MeshPipelineKey;
use crate::queue::{DrawRequest, phase_for_render_class};

/// The frame's visible draw work, published by the visibility stage.
///
/// A plain holder the visibility stage fills (from culling / binning) and the
/// queue stage drains into [`crate::RenderQueues`]. Keeping it a data resource
/// keeps the two stages decoupled.
#[derive(Resource, Default)]
pub struct VisibilityWorkList {
    items: Vec<GpuRenderWorkItem>,
}

impl VisibilityWorkList {
    /// An empty work list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the list's contents for this frame.
    pub fn set(&mut self, items: Vec<GpuRenderWorkItem>) {
        self.items = items;
    }

    /// Appends one work item.
    pub fn push(&mut self, item: GpuRenderWorkItem) {
        self.items.push(item);
    }

    /// The frame's work items.
    #[must_use]
    pub fn items(&self) -> &[GpuRenderWorkItem] {
        &self.items
    }

    /// Drops the frame's work items.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// The number of visible work items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether no draws are visible.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// The pipeline inputs a vertex-buffer class implies: its vertex layouts and
/// primitive topology.
///
/// A work item names a `vertex_buffer_class`; this is where that class maps to
/// the concrete mesh pipeline inputs. Two geometries sharing a class share a
/// vertex stage (and, if their variants match, a pipeline).
#[derive(Clone, Debug, PartialEq)]
pub struct MeshClass {
    /// The vertex buffer layouts the class's meshes use.
    pub layout: Vec<VertexBufferLayout>,
    /// The primitive topology the class draws with.
    pub topology: PrimitiveTopology,
    /// The GPU buffers a draw of this class binds.
    pub buffers: MeshBuffers,
}

impl MeshClass {
    /// A class with `layout` drawn as `topology` from `buffers`.
    #[must_use]
    pub fn new(
        layout: Vec<VertexBufferLayout>,
        topology: PrimitiveTopology,
        buffers: MeshBuffers,
    ) -> Self {
        Self {
            layout,
            topology,
            buffers,
        }
    }
}

/// The GPU buffers a mesh draw binds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MeshBuffers {
    /// The vertex buffer bound at slot `0`.
    pub vertices: BufferId,
    /// The index buffer and its format, or `None` for a non-indexed class.
    pub indices: Option<(BufferId, IndexFormat)>,
}

/// The render side's geometry / material / mesh-class tables.
#[derive(Resource, Default)]
pub struct SceneRegistry {
    geometries: BTreeMap<GeometryHandle, GeometryRecord>,
    materials: BTreeMap<SceneMaterialHandle, MaterialRecord>,
    mesh_classes: BTreeMap<u32, MeshClass>,
}

impl SceneRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a geometry record.
    pub fn insert_geometry(&mut self, record: GeometryRecord) {
        self.geometries.insert(record.handle, record);
    }

    /// Inserts or replaces a material record.
    pub fn insert_material(&mut self, record: MaterialRecord) {
        self.materials.insert(record.handle, record);
    }

    /// Inserts or replaces the pipeline inputs for a vertex-buffer class.
    pub fn insert_mesh_class(&mut self, class: u32, mesh_class: MeshClass) {
        self.mesh_classes.insert(class, mesh_class);
    }

    /// The geometry record for `handle`.
    #[must_use]
    pub fn geometry(&self, handle: GeometryHandle) -> Option<&GeometryRecord> {
        self.geometries.get(&handle)
    }

    /// The material record for `handle`.
    #[must_use]
    pub fn material(&self, handle: SceneMaterialHandle) -> Option<&MaterialRecord> {
        self.materials.get(&handle)
    }

    /// The pipeline inputs registered for a vertex-buffer class.
    #[must_use]
    pub fn mesh_class(&self, class: u32) -> Option<&MeshClass> {
        self.mesh_classes.get(&class)
    }

    /// The number of resident geometries.
    #[must_use]
    pub fn geometry_count(&self) -> usize {
        self.geometries.len()
    }

    /// The number of resident materials.
    #[must_use]
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Resolves a visibility work item into a queueable draw.
    ///
    /// Returns `None` when the geometry or material is not resident, the
    /// requested LOD has no resident level, or the geometry's vertex-buffer
    /// class has no registered mesh class — each a case the caller skips rather
    /// than a hard error.
    #[must_use]
    pub fn resolve(
        &self,
        item: &GpuRenderWorkItem,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        draw: DrawFunctionId,
    ) -> Option<DrawRequest> {
        let geometry = self.geometries.get(&item.geometry)?;
        let lod = geometry.resolve_lod(item.lod_or_cluster)?;
        let material = self.materials.get(&item.material)?;
        let mesh_class = self.mesh_classes.get(&geometry.vertex_buffer_class)?;

        // `illumination` is a response axis, not a phase; the phase comes from
        // the blend family, and the permutation identity keys the pipeline.
        let phase = phase_for_render_class(material.render_class);
        let specialization = material.specialization();
        let variant = MeshPipelineKey {
            class: specialization.bin_class(),
            color_format: Some(RENDER_TARGET_FORMAT),
            depth_format,
            sample_count,
            topology: mesh_class.topology,
            shader_id: specialization.0,
        };

        Some(DrawRequest {
            phase,
            variant,
            layout: mesh_class.layout.clone(),
            lod_level: lod.level,
            geometry: item.geometry,
            material: item.material,
            scene: item.scene,
            order: item.sort_key.0,
            draw,
            vertices: mesh_class.buffers.vertices,
            indices: mesh_class.buffers.indices,
            first: lod.first_element,
            count: lod.element_count,
            base_vertex: lod.base_vertex,
        })
    }

    /// The mesh pipeline variant a packed draw bin specializes to.
    ///
    /// A bin key carries only the compact permutation class
    /// ([`SpecializationId::bin_class`]), so this reconstructs the full key —
    /// which is what makes a material specialize to the **same** pipeline
    /// whether it is queued directly or through a bin — and folds in the
    /// vertex-buffer class's topology and layout class, exactly as
    /// [`resolve`](Self::resolve) does for a work item.
    ///
    /// Returns `None` when the class does not decode to a legal permutation, or
    /// when the bin's vertex-buffer class is not resident.
    ///
    /// [`SpecializationId::bin_class`]: prism_render_material::SpecializationId::bin_class
    #[must_use]
    pub fn bin_variant(
        &self,
        key: &DrawBinKey,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
    ) -> Option<MeshPipelineKey> {
        let specialization =
            prism_render_material::SpecializationId::from_bin_class(key.pipeline_class)?;
        let mesh_class = self.mesh_classes.get(&key.vertex_buffer_class)?;
        Some(MeshPipelineKey {
            // Re-derived from the reconstructed key (not copied from the bin),
            // so class and shader identity can never disagree.
            class: specialization.bin_class(),
            color_format: Some(RENDER_TARGET_FORMAT),
            depth_format,
            sample_count,
            topology: mesh_class.topology,
            shader_id: specialization.0,
        })
    }

    /// Derives the frame's draw-bin candidates from the visible work list.
    ///
    /// One candidate per drawable work item: the key names the geometry, LOD,
    /// and the material's permutation class, and the item's visibility stages
    /// route it to the early or late raster pass. An item whose geometry, LOD,
    /// material, or vertex-buffer class is not resident is dropped, which is the
    /// same set the direct path ([`resolve`](Self::resolve)) draws — so both
    /// paths agree on what is drawable and a bin never outlives its content.
    ///
    /// `indexed`/`primitive_kind` follow the *class's bound buffers*, not the
    /// geometry record's statement: the draw binds the class's index buffer (or
    /// none), and the bin's GPU header must describe the draw that will happen.
    #[must_use]
    pub fn bin_candidates(&self, work: &[GpuRenderWorkItem]) -> Vec<DrawBinCandidate> {
        let mut candidates = Vec::new();
        for item in work {
            let Some(geometry) = self.geometries.get(&item.geometry) else {
                continue;
            };
            if geometry.resolve_lod(item.lod_or_cluster).is_none() {
                continue;
            }
            let Some(material) = self.materials.get(&item.material) else {
                continue;
            };
            let Some(mesh_class) = self.mesh_classes.get(&geometry.vertex_buffer_class) else {
                continue;
            };
            let indexed = mesh_class.buffers.indices.is_some();
            candidates.push(DrawBinCandidate {
                scene: item.scene,
                key: DrawBinKey {
                    geometry: item.geometry,
                    lod_or_cluster: item.lod_or_cluster,
                    pipeline_class: material.specialization().bin_class(),
                    vertex_buffer_class: geometry.vertex_buffer_class,
                    index_buffer_class: geometry.index_buffer_class,
                    indexed,
                    primitive_kind: if indexed {
                        GeometryPrimitiveKind::Indexed
                    } else {
                        GeometryPrimitiveKind::NonIndexed
                    },
                    pass_mask: item.pass_mask.0,
                },
                visibility_stages: item.visibility_stages,
            });
        }
        candidates
    }
}
