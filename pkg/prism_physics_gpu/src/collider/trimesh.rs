//! A static triangle mesh collider and the per-triangle bounding boxes that
//! feed its broad phase.
//!
//! A [`Trimesh`] is the canonical level-geometry collider: an indexed set of
//! world-space triangles that stays fixed while dynamic bounding-sphere
//! particles collide against it. It owns no collision math of its own; it is a
//! thin, deterministic accessor layer that hands individual [`Triangle`]s to
//! the shared sphere-versus-triangle narrow phase and emits one [`Aabb`] per
//! triangle so the aggregation layer can build an `LBVH` over the mesh once and
//! reuse it across every query frame.
//!
//! # Representation
//!
//! * `vertices` is the shared vertex pool, each a world-space [`Vec3`].
//! * `indices` is one `[u32; 3]` per triangle, each entry an index into
//!   `vertices`. The winding is counter-clockwise viewed from the `+normal`
//!   side, matching [`Triangle`]'s convention so the narrow phase's face-normal
//!   fallback points consistently.
//!
//! Triangle `i` is `(vertices[indices[i][0]], vertices[indices[i][1]],
//! vertices[indices[i][2]])`. The accessors never reorder triangles, so a
//! triangle index is stable across [`triangle`](Trimesh::triangle),
//! [`triangle_aabbs`](Trimesh::triangle_aabbs), and any `LBVH` built from those
//! boxes once its `sorted_indices` are followed back to the original order.
//!
//! Provenance: an indexed triangle soup is textbook mesh representation. No
//! Unreal Engine source or derived code.

use glam::Vec3;

use crate::bvh::Aabb;
use crate::narrowphase::Triangle;

/// A static, indexed triangle-mesh collider.
///
/// Holds a shared vertex pool and one index triple per triangle. The mesh is
/// immutable collision geometry: construct it once from level data, build an
/// `LBVH` from [`triangle_aabbs`](Trimesh::triangle_aabbs), and query spheres
/// against it each frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Trimesh {
    /// Shared world-space vertex pool.
    vertices: Vec<Vec3>,
    /// One index triple per triangle, each entry indexing `vertices`.
    indices: Vec<[u32; 3]>,
}

impl Trimesh {
    /// Creates a mesh from a vertex pool and its index triples.
    ///
    /// The caller owns winding and index validity: every index must address a
    /// vertex in `vertices`. Indices are validated lazily on access
    /// ([`triangle`](Trimesh::triangle) panics on an out-of-range index via the
    /// usual slice bounds check), so a malformed mesh fails loudly at use.
    #[must_use]
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Trimesh {
        Trimesh { vertices, indices }
    }

    /// The shared vertex pool.
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// The index triples, one per triangle.
    #[must_use]
    pub fn indices(&self) -> &[[u32; 3]] {
        &self.indices
    }

    /// The number of triangles in the mesh.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Materialises triangle `i` as a [`Triangle`] of three world-space
    /// vertices.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range or any of its indices falls outside the
    /// vertex pool, surfacing a malformed mesh at the point of use.
    #[must_use]
    pub fn triangle(&self, i: usize) -> Triangle {
        let [ia, ib, ic] = self.indices[i];
        Triangle::new(
            self.vertices[ia as usize],
            self.vertices[ib as usize],
            self.vertices[ic as usize],
        )
    }

    /// The axis-aligned bounding box of triangle `i`.
    ///
    /// The box is the component-wise min/max of the three vertices, the tight
    /// bound an `LBVH` leaf wants.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`triangle`](Trimesh::triangle).
    #[must_use]
    pub fn triangle_aabb(&self, i: usize) -> Aabb {
        let tri = self.triangle(i);
        let min = tri.a.min(tri.b).min(tri.c);
        let max = tri.a.max(tri.b).max(tri.c);
        Aabb::new(min, max)
    }

    /// One tight [`Aabb`] per triangle, in triangle order.
    ///
    /// This is the exact input [`cpu_build_lbvh`](crate::cpu_build_lbvh) wants:
    /// building a hierarchy over these boxes yields a tree whose
    /// `sorted_indices` map leaf slots back to triangle indices.
    #[must_use]
    pub fn triangle_aabbs(&self) -> Vec<Aabb> {
        (0..self.triangle_count())
            .map(|i| self.triangle_aabb(i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-triangle quad in the z = 0 plane spanning the unit square.
    fn unit_quad() -> Trimesh {
        Trimesh::new(
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn triangle_count_matches_indices() {
        assert_eq!(unit_quad().triangle_count(), 2);
    }

    #[test]
    fn triangle_materialises_in_order() {
        let mesh = unit_quad();
        let t0 = mesh.triangle(0);
        assert_eq!(t0.a, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(t0.b, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(t0.c, Vec3::new(1.0, 1.0, 0.0));
        let t1 = mesh.triangle(1);
        assert_eq!(t1.a, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(t1.b, Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(t1.c, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn triangle_aabb_is_tight() {
        let mesh = unit_quad();
        let b0 = mesh.triangle_aabb(0);
        assert_eq!(b0.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(b0.max, Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn triangle_aabbs_match_per_triangle() {
        let mesh = unit_quad();
        let boxes = mesh.triangle_aabbs();
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0], mesh.triangle_aabb(0));
        assert_eq!(boxes[1], mesh.triangle_aabb(1));
    }

    #[test]
    fn accessors_expose_raw_pools() {
        let mesh = unit_quad();
        assert_eq!(mesh.vertices().len(), 4);
        assert_eq!(mesh.indices().len(), 2);
        assert_eq!(mesh.indices()[1], [0, 2, 3]);
    }
}
