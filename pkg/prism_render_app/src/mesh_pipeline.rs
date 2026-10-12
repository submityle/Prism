//! Mesh pipeline specialization.
//!
//! Bevy's `SpecializedMeshPipelines` turns a mesh + variant key into a concrete
//! pipeline, building (and caching) one per distinct vertex layout and variant.
//! Prism's equivalent is [`MeshPipelineSpecializer`]: the caller supplies the
//! variant part of the key ([`MeshPipelineKey`]) and the mesh's vertex layouts,
//! and the specializer composes the full [`PipelineKey`] — including the vertex
//! layout hash the base variant key omits — and specializes through the shared
//! [`crate::PipelineCache`], so a layout/variant pair compiles once and is
//! reused.

use prism_render_driver::{
    PrimitiveTopology, RenderDevice, RenderPipelineDescriptor, RenderPipelineId, TextureFormat,
    VertexBufferLayout,
};

use crate::pipeline_cache::{PipelineCache, PipelineKey, hash_vertex_layouts};

/// The variant part of a mesh pipeline key: everything but the vertex layout.
///
/// The vertex layout is supplied separately to
/// [`MeshPipelineSpecializer::specialize`] because it comes from the mesh, while
/// the variant (material class, target formats, samples, topology, shader) is
/// the renderer's. Two meshes with the same variant but different layouts
/// specialize to different pipelines.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshPipelineKey {
    /// The material permutation / bin class.
    pub class: u32,
    /// The single color-target format, or `None` for a depth-only pipeline.
    pub color_format: Option<TextureFormat>,
    /// The depth/stencil format, or `None` when there is no depth state.
    pub depth_format: Option<TextureFormat>,
    /// The MSAA sample count (`1` for no multisampling).
    pub sample_count: u32,
    /// The primitive topology.
    pub topology: PrimitiveTopology,
    /// A stable identity for the shader module and its entry points.
    pub shader_id: u64,
}

impl MeshPipelineKey {
    /// The full pipeline key for a mesh with `layout`.
    #[must_use]
    pub fn with_layout(self, layout: &[VertexBufferLayout]) -> PipelineKey {
        PipelineKey::new(
            self.class,
            self.color_format,
            self.depth_format,
            self.sample_count,
            hash_vertex_layouts(layout),
            self.topology,
            self.shader_id,
        )
    }
}

/// Specializes and caches a pipeline per `(variant, vertex layout)` pair.
#[derive(Default)]
pub struct MeshPipelineSpecializer {
    cache: PipelineCache,
}

impl MeshPipelineSpecializer {
    /// An empty specializer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: PipelineCache::new(),
        }
    }

    /// The number of distinct specialized pipelines.
    #[must_use]
    pub fn pipeline_count(&self) -> usize {
        self.cache.len()
    }

    /// Returns the pipeline for `(variant, layout)`, compiling `descriptor` on
    /// first use.
    pub fn specialize<D: RenderDevice>(
        &mut self,
        device: &D,
        variant: MeshPipelineKey,
        layout: &[VertexBufferLayout],
        descriptor: &RenderPipelineDescriptor,
    ) -> RenderPipelineId {
        self.cache
            .get_or_create(device, variant.with_layout(layout), descriptor)
    }

    /// Drops one reference to `(variant, layout)`, destroying the pipeline at
    /// zero. Returns whether it was evicted.
    pub fn release<D: RenderDevice>(
        &mut self,
        device: &D,
        variant: MeshPipelineKey,
        layout: &[VertexBufferLayout],
    ) -> bool {
        self.cache.release(device, &variant.with_layout(layout))
    }

    /// Destroys every specialized pipeline.
    pub fn clear<D: RenderDevice>(&mut self, device: &D) {
        self.cache.clear(device);
    }
}
