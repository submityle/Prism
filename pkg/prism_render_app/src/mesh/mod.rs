//! The GPU form of a [`Mesh`]: buffers the pipeline binds.
//!
//! [`RenderMesh`] is what a raster or queue system reaches for: the vertex
//! buffer (and index buffer, when the mesh is indexed), the counts, the vertex
//! layout, and the model-space bounds — everything a draw needs, none of it
//! requiring the authored [`Mesh`] again. It is prepared and released by the
//! render-asset pipeline (`register_render_assets::<I, RenderMesh>`), so a mesh
//! created or changed in the simulation reaches the GPU without the renderer
//! knowing anything about the mesh's authoring types.

pub mod deformation;
pub mod consumer;
mod prepare;

#[cfg(test)]
mod tests;

use prism_mesh::MeshBounds;
use prism_render_driver::{IndexFormat, PrimitiveTopology, VertexBufferLayout};

use crate::scene::MeshBuffers;

/// How a mesh's elements are addressed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMeshBufferInfo {
    /// The draw fetches indices from an index buffer.
    Indexed {
        /// The width of the indices in the buffer.
        index_format: IndexFormat,
        /// How many indices the draw covers.
        index_count: u32,
    },
    /// The draw fetches vertices directly.
    NonIndexed,
}

impl RenderMeshBufferInfo {
    /// The number of elements the draw covers (indices, or vertices).
    #[must_use]
    pub const fn element_count(self, vertex_count: u32) -> u32 {
        match self {
            Self::Indexed { index_count, .. } => index_count,
            Self::NonIndexed => vertex_count,
        }
    }

    /// The index buffer's format, when the mesh is indexed.
    #[must_use]
    pub const fn index_format(self) -> Option<IndexFormat> {
        match self {
            Self::Indexed { index_format, .. } => Some(index_format),
            Self::NonIndexed => None,
        }
    }
}

/// A mesh's GPU buffers and everything a draw needs to bind them.
pub struct RenderMesh {
    /// The vertex buffer, and the index buffer when the mesh is indexed.
    pub buffers: MeshBuffers,
    /// How the draw addresses the mesh's elements.
    pub buffer_info: RenderMeshBufferInfo,
    /// The number of vertices the vertex buffer holds.
    pub vertex_count: u32,
    /// The layouts the pipeline binds at vertex-buffer slots in order.
    pub layout: Vec<VertexBufferLayout>,
    /// The topology the draw uses.
    pub topology: PrimitiveTopology,
    /// The mesh's model-space bounds, when its positions gave them.
    pub bounds: Option<MeshBounds>,
    /// The vertex buffer's byte capacity, so a re-prepare knows whether it can
    /// reuse the buffer it already has.
    vertex_capacity: Vec<u64>,
    /// The index buffer's byte capacity (`0` when the mesh is not indexed).
    index_capacity: u64,
}

impl RenderMesh {
    /// The primitive topology the draw uses.
    #[must_use]
    pub const fn primitive_topology(&self) -> PrimitiveTopology {
        self.topology
    }

    /// Whether the draw fetches indices from an index buffer.
    #[must_use]
    pub const fn is_indexed(&self) -> bool {
        matches!(self.buffer_info, RenderMeshBufferInfo::Indexed { .. })
    }

    /// The number of elements the draw covers.
    #[must_use]
    pub const fn element_count(&self) -> u32 {
        self.buffer_info.element_count(self.vertex_count)
    }
}
