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
use prism_render_architecture::geometry::GeometryRecord;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
use prism_render_driver::{PrimitiveTopology, TextureFormat, VertexBufferLayout};
use prism_render_material::MaterialRecord;
use prism_render_visibility::GpuRenderWorkItem;

use crate::draw::DrawFunctionId;
use crate::frame::RENDER_TARGET_FORMAT;
use crate::mesh_pipeline::MeshPipelineKey;
use crate::queue::{DrawRequest, phase_for_render_class};

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
}

impl MeshClass {
    /// A class with `layout` drawn as `topology`.
    #[must_use]
    pub fn new(layout: Vec<VertexBufferLayout>, topology: PrimitiveTopology) -> Self {
        Self { layout, topology }
    }
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
        })
    }
}
