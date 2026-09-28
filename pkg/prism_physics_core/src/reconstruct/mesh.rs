//! The triangle-mesh output type for surface reconstruction.
//!
//! [`SurfaceMesh`] is a pure geometry buffer: interleaved parallel arrays of
//! vertex positions and normals plus a flat triangle index list. It carries no
//! materials, no topology adjacency, and no rendering state — downstream
//! systems decide how to upload or process it.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! plain indexed-triangle container.

use glam::Vec3;

/// An indexed triangle mesh: `positions[i]` and `normals[i]` describe vertex
/// `i`, and every three consecutive entries of `indices` form one triangle.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SurfaceMesh {
    /// Vertex positions in world space.
    pub positions: Vec<Vec3>,
    /// Per-vertex unit normals, index-aligned with [`SurfaceMesh::positions`].
    pub normals: Vec<Vec3>,
    /// Triangle indices into the vertex arrays (three per triangle).
    pub indices: Vec<u32>,
}

impl SurfaceMesh {
    /// Creates an empty mesh.
    #[must_use]
    pub const fn new() -> SurfaceMesh {
        SurfaceMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// The number of vertices.
    #[inline]
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// The number of triangles.
    #[inline]
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Whether the mesh holds no triangles.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Whether every index is within the vertex range and the index count is a
    /// multiple of three (a basic well-formedness check).
    #[must_use]
    pub fn indices_are_valid(&self) -> bool {
        if !self.indices.len().is_multiple_of(3) {
            return false;
        }
        let n = self.positions.len() as u32;
        self.indices.iter().all(|&i| i < n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mesh_is_empty() {
        let m = SurfaceMesh::new();
        assert!(m.is_empty());
        assert_eq!(m.triangle_count(), 0);
        assert_eq!(m.vertex_count(), 0);
        assert!(m.indices_are_valid());
    }

    #[test]
    fn validity_checks_index_range() {
        let mut m = SurfaceMesh::new();
        m.positions.push(Vec3::ZERO);
        m.positions.push(Vec3::X);
        m.positions.push(Vec3::Y);
        m.normals = vec![Vec3::Z; 3];
        m.indices = vec![0, 1, 2];
        assert!(m.indices_are_valid());
        m.indices = vec![0, 1, 3];
        assert!(!m.indices_are_valid());
        m.indices = vec![0, 1];
        assert!(!m.indices_are_valid());
    }
}
