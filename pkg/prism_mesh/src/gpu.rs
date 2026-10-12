//! Uploading a [`Mesh`] into RHI buffers.

use crate::mesh::Mesh;
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, IndexFormat, RenderDevice, RenderQueue,
};

/// The smallest buffer the RHI backends accept; a zero-sized allocation is
/// invalid, so an empty mesh still gets a (unused) minimum-size buffer.
const MIN_BUFFER_SIZE: u64 = 4;

/// A [`Mesh`] uploaded into RHI buffers, plus the metadata a draw call needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GpuMesh {
    /// The vertex buffer.
    pub vertices: BufferId,
    /// The index buffer.
    pub indices: BufferId,
    /// The format of the index buffer.
    pub index_format: IndexFormat,
    /// Number of vertices in the vertex buffer.
    pub vertex_count: u32,
    /// Number of indices in the index buffer.
    pub index_count: u32,
}

impl GpuMesh {
    /// Releases both buffers back to `device`.
    pub fn destroy<D: RenderDevice>(&self, device: &D) {
        device.destroy_buffer(self.vertices);
        device.destroy_buffer(self.indices);
    }
}

impl Mesh {
    /// Uploads this mesh's vertex and index bytes into fresh RHI buffers.
    ///
    /// The vertex buffer is created with `VERTEX | COPY_DST | COPY_SRC` and the
    /// index buffer with `INDEX | COPY_DST | COPY_SRC`. `COPY_DST` is needed for
    /// the upload; `COPY_SRC` is included so headless/golden tooling (and the
    /// round-trip test) can read the uploaded bytes back. Data is written via
    /// [`RenderQueue::write_buffer`], so it is visible to any later submission
    /// on the same queue.
    #[must_use]
    pub fn upload<D: RenderDevice, Q: RenderQueue>(&self, device: &D, queue: &Q) -> GpuMesh {
        let vertex_bytes = self.vertex_bytes();
        let index_bytes = self.index_bytes();

        let vertices = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("mesh-vertices")),
            size: (vertex_bytes.len() as u64).max(MIN_BUFFER_SIZE),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indices = device.create_buffer(&BufferDescriptor {
            label: Some(String::from("mesh-indices")),
            size: (index_bytes.len() as u64).max(MIN_BUFFER_SIZE),
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        if !vertex_bytes.is_empty() {
            queue.write_buffer(vertices, 0, vertex_bytes);
        }
        if !index_bytes.is_empty() {
            queue.write_buffer(indices, 0, &index_bytes);
        }

        GpuMesh {
            vertices,
            indices,
            index_format: self.index_format(),
            vertex_count: self.vertex_count(),
            index_count: self.index_count(),
        }
    }
}
