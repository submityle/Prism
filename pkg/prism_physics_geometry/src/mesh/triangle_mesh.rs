//! Indexed triangle mesh with a built-in BVH for exact ray queries.
//!
//! [`TriangleMesh`] bridges the broad phase and the narrow phase: it stores a
//! vertex/index buffer, builds a [`DynamicBvh`] over per-triangle bounding
//! boxes, and answers exact ray queries by walking BVH leaves in ray-entry
//! order and refining each candidate with the Möller-Trumbore
//! [`ray_triangle`] test. Because leaves are visited nearest-box-first, the
//! traversal stops as soon as the next box lies beyond the closest confirmed
//! triangle hit, so the query cost scales with the hit depth rather than the
//! triangle count.
//!
//! This is a clean-room composition of publicly documented spatial-query
//! techniques and contains no Unreal Engine source or derived code.

use alloc::vec::Vec;
use glam::Vec3;

use crate::bounding::{Aabb, Ray};
use crate::bvh::DynamicBvh;
use crate::narrow::ray_triangle;

/// An exact ray/triangle-mesh intersection.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshRayHit {
    /// Index of the triangle that was hit.
    pub triangle: u32,
    /// Parametric distance along the ray to the hit point.
    pub t: f32,
    /// World-space hit position (`ray.at(t)`).
    pub point: Vec3,
    /// Barycentric weight of the triangle's second vertex.
    pub u: f32,
    /// Barycentric weight of the triangle's third vertex.
    pub v: f32,
    /// Geometric (face) unit normal, from the triangle winding.
    pub normal: Vec3,
}

/// An indexed triangle mesh accelerated by a dynamic BVH.
#[derive(Clone, Debug)]
pub struct TriangleMesh {
    vertices: Vec<Vec3>,
    indices: Vec<[u32; 3]>,
    bvh: DynamicBvh,
}

impl TriangleMesh {
    /// Builds a mesh from a vertex buffer and triangle index triples.
    ///
    /// Triangles whose indices fall outside the vertex buffer, or whose corners
    /// do not form a valid bounding box, are skipped so a malformed triangle
    /// cannot poison later queries. The BVH leaf payload stores the original
    /// triangle index.
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        let mut bvh = DynamicBvh::with_capacity(indices.len());
        let vertex_count = vertices.len() as u32;
        for (tri_index, tri) in indices.iter().enumerate() {
            let [ia, ib, ic] = *tri;
            if ia >= vertex_count || ib >= vertex_count || ic >= vertex_count {
                continue;
            }
            let corners = [
                vertices[ia as usize],
                vertices[ib as usize],
                vertices[ic as usize],
            ];
            if let Some(aabb) = Aabb::from_points(&corners) {
                bvh.insert(aabb, tri_index as u64);
            }
        }
        Self {
            vertices,
            indices,
            bvh,
        }
    }

    /// Returns the number of triangles in the mesh (including any that were
    /// skipped from the BVH because they were malformed).
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` when the mesh has no triangles.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns the three world-space corners of triangle `index`, if valid.
    pub fn triangle(&self, index: usize) -> Option<[Vec3; 3]> {
        let [ia, ib, ic] = *self.indices.get(index)?;
        let a = *self.vertices.get(ia as usize)?;
        let b = *self.vertices.get(ib as usize)?;
        let c = *self.vertices.get(ic as usize)?;
        Some([a, b, c])
    }

    /// Casts `ray` against the mesh and returns the nearest exact hit within
    /// `[0, ray.tmax]`, or `None` when the ray misses every triangle.
    pub fn ray_cast(&self, ray: &Ray) -> Option<MeshRayHit> {
        let mut best: Option<MeshRayHit> = None;
        self.bvh.ray_cast_ordered(ray, &mut |data, _aabb, box_entry| {
            // Leaves arrive in nearest-box-first order, so once a candidate box
            // starts beyond the closest confirmed hit nothing further can win.
            if best.as_ref().is_some_and(|h| box_entry > h.t) {
                return false;
            }
            let tri_index = data as usize;
            let [ia, ib, ic] = self.indices[tri_index];
            let a = self.vertices[ia as usize];
            let b = self.vertices[ib as usize];
            let c = self.vertices[ic as usize];
            if let Some(hit) = ray_triangle(ray, a, b, c) {
                let closer = match &best {
                    Some(h) => hit.t < h.t,
                    None => true,
                };
                if closer {
                    best = Some(MeshRayHit {
                        triangle: data as u32,
                        t: hit.t,
                        point: ray.at(hit.t),
                        u: hit.u,
                        v: hit.v,
                        normal: (b - a).cross(c - a).normalize_or_zero(),
                    });
                }
            }
            true
        });
        best
    }
}

#[cfg(test)]
mod tests {
    use super::TriangleMesh;
    use crate::bounding::Ray;
    use glam::Vec3;

    /// Two parallel quads (as triangle pairs) stacked along the ray so the
    /// nearest must win and the far one must be pruned.
    fn two_quads() -> TriangleMesh {
        // Quad A at z = 2, quad B at z = 6, both spanning the XY unit square.
        let vertices = alloc::vec![
            // Quad A (z = 2)
            Vec3::new(-1.0, -1.0, 2.0),
            Vec3::new(1.0, -1.0, 2.0),
            Vec3::new(1.0, 1.0, 2.0),
            Vec3::new(-1.0, 1.0, 2.0),
            // Quad B (z = 6)
            Vec3::new(-1.0, -1.0, 6.0),
            Vec3::new(1.0, -1.0, 6.0),
            Vec3::new(1.0, 1.0, 6.0),
            Vec3::new(-1.0, 1.0, 6.0),
        ];
        let indices = alloc::vec![
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ];
        TriangleMesh::new(vertices, indices)
    }

    #[test]
    fn ray_hits_nearest_quad() {
        let mesh = two_quads();
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = mesh.ray_cast(&ray).expect("hit");
        // Nearest quad is at z = 2.
        assert!((hit.t - 2.0).abs() < 1e-5, "t = {}", hit.t);
        assert!(hit.point.abs_diff_eq(Vec3::new(0.0, 0.0, 2.0), 1e-5));
        // Hit one of the two front triangles.
        assert!(hit.triangle < 2, "front triangle, got {}", hit.triangle);
        // Face normal is axis-aligned on Z.
        assert!(hit.normal.z.abs() > 0.99);
    }

    #[test]
    fn ray_misses_outside_bounds() {
        let mesh = two_quads();
        let ray = Ray::new(Vec3::new(5.0, 5.0, 0.0), Vec3::Z);
        assert!(mesh.ray_cast(&ray).is_none());
    }

    #[test]
    fn ray_respects_tmax() {
        let mesh = two_quads();
        // tmax short of the first quad.
        let ray = Ray::with_tmax(Vec3::ZERO, Vec3::Z, 1.0);
        assert!(mesh.ray_cast(&ray).is_none());
    }

    #[test]
    fn malformed_indices_are_skipped() {
        let vertices = alloc::vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        // Second triangle references a nonexistent vertex and must be ignored.
        let indices = alloc::vec![[0, 1, 2], [0, 1, 99]];
        let mesh = TriangleMesh::new(vertices, indices);
        assert_eq!(mesh.triangle_count(), 2);
        // A ray through the valid triangle still hits.
        let ray = Ray::new(Vec3::new(0.2, 0.2, -1.0), Vec3::Z);
        assert!(mesh.ray_cast(&ray).is_some());
    }

    #[test]
    fn empty_mesh_has_no_hits() {
        let mesh = TriangleMesh::new(alloc::vec![], alloc::vec![]);
        assert!(mesh.is_empty());
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        assert!(mesh.ray_cast(&ray).is_none());
    }
}
