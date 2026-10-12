//! The engine-neutral geometry container.

use crate::vertex::{Vertex, VertexLayout};
use prism_render_driver::{IndexFormat, VertexBufferLayout};

/// A mesh: packed vertex bytes, `u32` index data, and the layout describing the
/// vertices.
///
/// The bytes and the layout are kept together so a mesh is always
/// self-describing — its [`layout`](Mesh::layout) can be handed to the vertex
/// stage and its [`vertex_bytes`](Mesh::vertex_bytes) to a buffer with no other
/// context.
#[derive(Clone, PartialEq, Debug)]
pub struct Mesh {
    vertices: Vec<u8>,
    indices: Vec<u32>,
    layout: VertexLayout,
}

impl Mesh {
    /// Wraps already-packed vertex bytes and indices with their layout.
    ///
    /// `vertices` must be a whole number of
    /// [`stride`](VertexLayout::stride) bytes; this is debug-asserted rather
    /// than checked, because the constructors that produce bytes
    /// ([`from_vertices`](Mesh::from_vertices) and the primitives) guarantee it.
    #[must_use]
    pub fn new(vertices: Vec<u8>, indices: Vec<u32>, layout: VertexLayout) -> Self {
        debug_assert!(
            layout.stride() > 0,
            "a vertex layout must describe at least one attribute"
        );
        debug_assert_eq!(
            vertices.len() as u64 % layout.stride(),
            0,
            "vertex bytes must be a whole number of vertices"
        );
        Self {
            vertices,
            indices,
            layout,
        }
    }

    /// Packs `vertices` into bytes according to `layout` and pairs them with
    /// `indices`.
    #[must_use]
    pub fn from_vertices(vertices: &[Vertex], indices: Vec<u32>, layout: VertexLayout) -> Self {
        let mut bytes = Vec::with_capacity(vertices.len() * layout.stride() as usize);
        for vertex in vertices {
            vertex.encode(layout, &mut bytes);
        }
        Self::new(bytes, indices, layout)
    }

    /// The RHI vertex buffer layout matching the packed vertex bytes exactly.
    #[must_use]
    pub fn layout(&self) -> VertexBufferLayout {
        self.layout.to_rhi()
    }

    /// The attribute model: which attributes the vertices carry.
    #[must_use]
    pub fn vertex_layout(&self) -> VertexLayout {
        self.layout
    }

    /// The packed vertex bytes.
    #[must_use]
    pub fn vertex_bytes(&self) -> &[u8] {
        &self.vertices
    }

    /// The indices as little-endian `u32` bytes.
    #[must_use]
    pub fn index_bytes(&self) -> Vec<u8> {
        self.indices
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    /// The index data.
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The format the indices are stored in.
    #[must_use]
    pub fn index_format(&self) -> IndexFormat {
        IndexFormat::Uint32
    }

    /// The byte stride between consecutive vertices.
    #[must_use]
    pub fn stride(&self) -> u64 {
        self.layout.stride()
    }

    /// How many vertices the mesh holds.
    #[must_use]
    pub fn vertex_count(&self) -> u32 {
        u32::try_from(self.vertices.len() as u64 / self.stride()).expect("vertex count fits u32")
    }

    /// How many indices the mesh holds.
    #[must_use]
    pub fn index_count(&self) -> u32 {
        u32::try_from(self.indices.len()).expect("index count fits u32")
    }

    /// Whether the mesh has no vertices or no indices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// Decodes the vertex at `index`, or `None` if it is out of range.
    #[must_use]
    pub fn read_vertex(&self, index: usize) -> Option<Vertex> {
        let stride = self.stride() as usize;
        let start = index.checked_mul(stride)?;
        let bytes = self.vertices.get(start..start.checked_add(stride)?)?;
        Some(Vertex::decode(self.layout, bytes))
    }

    /// Iterates the mesh's vertices, decoding each one.
    pub fn vertices(&self) -> impl Iterator<Item = Vertex> + '_ {
        (0..self.vertex_count() as usize).map(|index| {
            self.read_vertex(index)
                .expect("index is within vertex_count")
        })
    }
}
