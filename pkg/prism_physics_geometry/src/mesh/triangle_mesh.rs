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

use crate::bounding::{Aabb, BoundingSphere, Capsule, Ray};
use crate::bvh::DynamicBvh;
use crate::narrow::{
    closest_point_on_triangle, closest_point_segment_triangle, ray_triangle,
    sweep_capsule_triangle, sweep_sphere_triangle, triangle_aabb_overlap,
};

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

/// The closest point on a [`TriangleMesh`] to a query point.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshClosestPoint {
    /// Index of the triangle carrying the closest point.
    pub triangle: u32,
    /// World-space closest point on that triangle.
    pub point: Vec3,
    /// Euclidean distance from the query to `point`.
    pub distance: f32,
}

/// The result of sweeping a sphere against a [`TriangleMesh`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshSweepHit {
    /// Index of the triangle the swept sphere first touches.
    pub triangle: u32,
    /// Time of impact along the sphere-centre ray (a distance, since the ray
    /// direction is unit length).
    pub t: f32,
    /// Contact point on the triangle surface.
    pub point: Vec3,
    /// Unit surface normal at the contact, pointing toward the sphere centre.
    pub normal: Vec3,
}

/// The result of sweeping a capsule against a [`TriangleMesh`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshCapsuleSweepHit {
    /// Index of the triangle the swept capsule first touches.
    pub triangle: u32,
    /// Time of impact along the motion direction (a distance when the ray
    /// direction is unit length, as it always is for [`Ray`]).
    pub t: f32,
    /// Contact point on the triangle surface.
    pub point: Vec3,
    /// Unit surface normal at the contact, pointing toward the capsule axis.
    pub normal: Vec3,
}

/// A single sphere/triangle overlap reported by [`TriangleMesh::sphere_contacts`].
///
/// Each contact describes how to push the sphere out of one overlapping
/// triangle: move the centre along `normal` by `depth` and the sphere surface
/// just touches the triangle at `point`. A character controller or rigid-body
/// solver can accumulate these to resolve penetration against a mesh.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshSphereContact {
    /// Index of the overlapping triangle.
    pub triangle: u32,
    /// Closest point on that triangle to the sphere centre.
    pub point: Vec3,
    /// Unit contact normal pointing from `point` toward the sphere centre.
    ///
    /// When the centre lies exactly on the triangle the direction is
    /// ill-defined, so it falls back to the triangle's geometric face normal.
    pub normal: Vec3,
    /// Penetration depth: how far the sphere surface lies past `point`
    /// (`radius - distance`), always `>= 0` for a reported contact.
    pub depth: f32,
}

/// A single capsule/triangle overlap reported by
/// [`TriangleMesh::capsule_contacts`].
///
/// A capsule is a segment inflated by a radius, so each contact is the
/// closest segment/triangle pair promoted to a push-out: shifting the capsule
/// axis along `normal` by `depth` lifts its surface clear of the triangle.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MeshCapsuleContact {
    /// Index of the overlapping triangle.
    pub triangle: u32,
    /// Closest point on that triangle to the capsule's core segment.
    pub point: Vec3,
    /// Unit contact normal pointing from `point` toward the capsule axis.
    ///
    /// Falls back to the triangle's geometric face normal when the axis
    /// touches the triangle (degenerate direction).
    pub normal: Vec3,
    /// Penetration depth (`radius - distance`), always `>= 0`.
    pub depth: f32,
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

    /// Returns the closest point on the mesh to `p`, or [`None`] when the mesh
    /// is empty.
    ///
    /// The query performs branch-and-bound over the BVH: fat boxes are visited
    /// nearest-first and each candidate triangle is refined with an exact
    /// Voronoi-region closest-point test, so traversal stops as soon as the
    /// next box lies farther than the closest confirmed triangle point.
    pub fn closest_point(&self, p: Vec3) -> Option<MeshClosestPoint> {
        let mut result: Option<MeshClosestPoint> = None;
        self.bvh.nearest_leaf_refined(p, |data, _box| {
            let tri_index = data as usize;
            let [ia, ib, ic] = self.indices[tri_index];
            let a = self.vertices[ia as usize];
            let b = self.vertices[ib as usize];
            let c = self.vertices[ic as usize];
            let cp = closest_point_on_triangle(p, a, b, c);
            let d2 = (cp - p).length_squared();
            if result.is_none_or(|r| d2 < r.distance * r.distance) {
                result = Some(MeshClosestPoint {
                    triangle: data as u32,
                    point: cp,
                    distance: d2.sqrt(),
                });
            }
            Some(d2)
        });
        result
    }

    /// Returns the Euclidean distance from `p` to the nearest triangle, or
    /// [`None`] when the mesh is empty.
    pub fn distance(&self, p: Vec3) -> Option<f32> {
        self.closest_point(p).map(|hit| hit.distance)
    }

    /// Returns `true` when any triangle lies within `radius` of `center`.
    ///
    /// Candidate triangles are gathered from the BVH by fat-box/sphere overlap
    /// and confirmed with an exact closest-point distance test, so a box that
    /// overlaps the sphere but whose triangle does not is correctly rejected.
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        if radius < 0.0 {
            return false;
        }
        let r2 = radius * radius;
        let mut hit = false;
        self.bvh
            .query_sphere(BoundingSphere::new(center, radius), &mut |data| {
                if hit {
                    return;
                }
                let tri_index = data as usize;
                let [ia, ib, ic] = self.indices[tri_index];
                let a = self.vertices[ia as usize];
                let b = self.vertices[ib as usize];
                let c = self.vertices[ic as usize];
                let cp = closest_point_on_triangle(center, a, b, c);
                if (cp - center).length_squared() <= r2 {
                    hit = true;
                }
            });
        hit
    }

    /// Collects the indices of every triangle within `radius` of `center`.
    ///
    /// Returns an empty vector when `radius` is negative or no triangle is in
    /// range. Indices are reported in BVH traversal order, not sorted.
    pub fn overlap_sphere(&self, center: Vec3, radius: f32) -> Vec<u32> {
        let mut out = Vec::new();
        if radius < 0.0 {
            return out;
        }
        let r2 = radius * radius;
        self.bvh
            .query_sphere(BoundingSphere::new(center, radius), &mut |data| {
                let tri_index = data as usize;
                let [ia, ib, ic] = self.indices[tri_index];
                let a = self.vertices[ia as usize];
                let b = self.vertices[ib as usize];
                let c = self.vertices[ic as usize];
                let cp = closest_point_on_triangle(center, a, b, c);
                if (cp - center).length_squared() <= r2 {
                    out.push(data as u32);
                }
            });
        out
    }

    /// Collects a depenetration contact for every triangle the sphere overlaps.
    ///
    /// For each candidate triangle gathered from the BVH, the exact closest
    /// point to `center` is computed; when it lies within `radius` the triangle
    /// contributes a [`MeshSphereContact`] whose `normal` points from the
    /// surface toward the centre and whose `depth` is `radius - distance`. A
    /// solver can iterate these to push a sphere (or capsule end-cap) out of a
    /// mesh. Returns an empty vector when `radius` is negative or nothing
    /// overlaps. Contacts are reported in BVH traversal order, not sorted by
    /// depth. When the centre lies exactly on a triangle the contact normal
    /// falls back to that triangle's geometric face normal so the push-out
    /// direction stays well defined.
    pub fn sphere_contacts(&self, center: Vec3, radius: f32) -> Vec<MeshSphereContact> {
        let mut out = Vec::new();
        if radius < 0.0 {
            return out;
        }
        let r2 = radius * radius;
        self.bvh
            .query_sphere(BoundingSphere::new(center, radius), &mut |data| {
                let tri_index = data as usize;
                let [ia, ib, ic] = self.indices[tri_index];
                let a = self.vertices[ia as usize];
                let b = self.vertices[ib as usize];
                let c = self.vertices[ic as usize];
                let cp = closest_point_on_triangle(center, a, b, c);
                let gap = center - cp;
                let dist2 = gap.length_squared();
                if dist2 > r2 {
                    return;
                }
                let dist = dist2.sqrt();
                // Prefer the surface-to-centre direction; when the centre sits
                // on the triangle it is degenerate, so fall back to the face
                // normal (and finally +Y for a degenerate triangle).
                let mut normal = gap.normalize_or_zero();
                if normal == Vec3::ZERO {
                    normal = (b - a).cross(c - a).normalize_or_zero();
                    if normal == Vec3::ZERO {
                        normal = Vec3::Y;
                    }
                }
                out.push(MeshSphereContact {
                    triangle: data as u32,
                    point: cp,
                    normal,
                    depth: radius - dist,
                });
            });
        out
    }

    /// Collects a depenetration contact for every triangle the capsule overlaps.
    ///
    /// A capsule is its core segment inflated by `capsule.radius`. For each
    /// candidate triangle gathered from the capsule's fat AABB, the exact
    /// closest segment/triangle pair is computed; when the gap is within the
    /// radius the triangle contributes a [`MeshCapsuleContact`] whose `normal`
    /// points from the surface toward the capsule axis and whose `depth` is
    /// `radius - distance`. Returns an empty vector when the radius is negative
    /// or nothing overlaps. Contacts follow BVH traversal order. When the axis
    /// touches the triangle the normal falls back to the face normal so the
    /// push-out direction stays well defined.
    pub fn capsule_contacts(&self, capsule: &Capsule) -> Vec<MeshCapsuleContact> {
        let mut out = Vec::new();
        if capsule.radius < 0.0 {
            return out;
        }
        let r = capsule.radius;
        let r2 = r * r;
        self.bvh.query_aabb(capsule.aabb(), &mut |data| {
            let tri_index = data as usize;
            let [ia, ib, ic] = self.indices[tri_index];
            let a = self.vertices[ia as usize];
            let b = self.vertices[ib as usize];
            let c = self.vertices[ic as usize];
            let closest = closest_point_segment_triangle(capsule.a, capsule.b, a, b, c);
            if closest.distance_squared > r2 {
                return;
            }
            let dist = closest.distance_squared.sqrt();
            let gap = closest.on_segment - closest.on_triangle;
            let mut normal = gap.normalize_or_zero();
            if normal == Vec3::ZERO {
                normal = (b - a).cross(c - a).normalize_or_zero();
                if normal == Vec3::ZERO {
                    normal = Vec3::Y;
                }
            }
            out.push(MeshCapsuleContact {
                triangle: data as u32,
                point: closest.on_triangle,
                normal,
                depth: r - dist,
            });
        });
        out
    }

    /// Collects the indices of every triangle overlapping `aabb`.
    ///
    /// Candidate triangles are gathered from the BVH by fat-box overlap and
    /// confirmed with an exact triangle/box separating-axis test, so a box that
    /// overlaps the fat bounds but not the triangle itself is rejected. Indices
    /// are reported in BVH traversal order, not sorted.
    pub fn overlap_aabb(&self, aabb: &Aabb) -> Vec<u32> {
        let mut out = Vec::new();
        self.bvh.query_aabb(*aabb, &mut |data| {
            let tri_index = data as usize;
            let [ia, ib, ic] = self.indices[tri_index];
            let a = self.vertices[ia as usize];
            let b = self.vertices[ib as usize];
            let c = self.vertices[ic as usize];
            if triangle_aabb_overlap(a, b, c, aabb) {
                out.push(data as u32);
            }
        });
        out
    }

    /// Sweeps a sphere of `radius` whose centre follows `ray` against the mesh
    /// and returns the earliest contact within `[0, ray.tmax]`.
    ///
    /// The broad phase walks BVH leaves whose box, expanded by `radius`, is hit
    /// by the ray in near-to-far order; because each box entry is a lower bound
    /// on the true time of impact, traversal stops as soon as a candidate box
    /// starts beyond the closest confirmed contact. Each candidate triangle is
    /// refined with an exact moving-sphere/triangle test. Returns `None` when
    /// the swept sphere never touches the mesh.
    pub fn sphere_cast(&self, ray: &Ray, radius: f32) -> Option<MeshSweepHit> {
        let mut best: Option<MeshSweepHit> = None;
        self.bvh
            .ray_cast_ordered_expanded(ray, radius, &mut |data, _box, box_entry| {
                // Boxes arrive by increasing entry, a lower bound on the TOI, so
                // once a box starts beyond the best contact nothing can beat it.
                if best.is_some_and(|h| box_entry > h.t) {
                    return false;
                }
                let tri_index = data as usize;
                let [ia, ib, ic] = self.indices[tri_index];
                let a = self.vertices[ia as usize];
                let b = self.vertices[ib as usize];
                let c = self.vertices[ic as usize];
                if let Some(hit) = sweep_sphere_triangle(ray, radius, a, b, c)
                    && best.is_none_or(|h| hit.t < h.t)
                {
                    best = Some(MeshSweepHit {
                        triangle: data as u32,
                        t: hit.t,
                        point: hit.point,
                        normal: hit.normal,
                    });
                }
                true
            });
        best
    }

    /// Sweeps a `capsule` whose core segment translates along `ray` against the
    /// mesh and returns the earliest contact within `[0, ray.tmax]`.
    ///
    /// The broad phase gathers every triangle whose fat box overlaps the swept
    /// capsule bounds (the start box unioned with the box translated by
    /// `ray.dir * ray.tmax`); for an infinite `ray.tmax` it conservatively
    /// scans the whole mesh. Each candidate is refined with an exact
    /// moving-capsule/triangle conservative-advancement test, and the global
    /// minimum time of impact wins. Returns `None` when the swept capsule never
    /// touches the mesh.
    pub fn capsule_cast(&self, capsule: &Capsule, ray: &Ray) -> Option<MeshCapsuleSweepHit> {
        let r = capsule.radius.max(0.0);
        let base = capsule.aabb();
        let query_box = if ray.tmax.is_finite() {
            let off = ray.dir * ray.tmax;
            let moved = Aabb::new(base.min + off, base.max + off);
            base.merged(&moved)
        } else {
            Aabb::new(Vec3::splat(-f32::MAX), Vec3::splat(f32::MAX))
        };
        let mut best: Option<MeshCapsuleSweepHit> = None;
        self.bvh.query_aabb(query_box, &mut |data| {
            let tri_index = data as usize;
            let [ia, ib, ic] = self.indices[tri_index];
            let a = self.vertices[ia as usize];
            let b = self.vertices[ib as usize];
            let c = self.vertices[ic as usize];
            if let Some(hit) =
                sweep_capsule_triangle(capsule.a, capsule.b, r, ray.dir, ray.tmax, a, b, c)
                && best.is_none_or(|h| hit.t < h.t)
            {
                best = Some(MeshCapsuleSweepHit {
                    triangle: data as u32,
                    t: hit.t,
                    point: hit.point,
                    normal: hit.normal,
                });
            }
        });
        best
    }
}

#[cfg(test)]
mod tests {
    use super::TriangleMesh;
    use crate::bounding::{Capsule, Ray};
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
    fn closest_point_on_near_quad() {
        let mesh = two_quads();
        // Point just in front of quad A (z = 2) along the ray.
        let hit = mesh.closest_point(Vec3::new(0.0, 0.0, 1.0)).expect("closest");
        assert!(hit.point.abs_diff_eq(Vec3::new(0.0, 0.0, 2.0), 1e-5));
        assert!((hit.distance - 1.0).abs() < 1e-5, "distance = {}", hit.distance);
        assert!(hit.triangle < 2, "front triangle, got {}", hit.triangle);
    }

    #[test]
    fn closest_point_prefers_true_nearest_triangle() {
        let mesh = two_quads();
        // Closer to the far quad (z = 6): point at z = 5.
        let hit = mesh.closest_point(Vec3::new(0.0, 0.0, 5.0)).expect("closest");
        assert!(hit.point.abs_diff_eq(Vec3::new(0.0, 0.0, 6.0), 1e-5));
        assert!(hit.triangle >= 2, "back triangle, got {}", hit.triangle);
    }

    #[test]
    fn closest_point_clamps_to_edge() {
        let mesh = two_quads();
        // Query off the +x side of quad A projects onto its edge at x = 1.
        let hit = mesh.closest_point(Vec3::new(3.0, 0.0, 2.0)).expect("closest");
        assert!((hit.point.x - 1.0).abs() < 1e-5, "x = {}", hit.point.x);
        assert!((hit.distance - 2.0).abs() < 1e-5, "distance = {}", hit.distance);
    }

    #[test]
    fn distance_matches_closest_point() {
        let mesh = two_quads();
        let p = Vec3::new(0.0, 0.0, -3.0);
        let d = mesh.distance(p).expect("distance");
        assert!((d - 5.0).abs() < 1e-5, "distance = {d}");
    }

    #[test]
    fn sphere_overlap_detects_and_rejects() {
        let mesh = two_quads();
        // Sphere centred at z = 1 with radius 1.5 reaches quad A (z = 2).
        assert!(mesh.intersects_sphere(Vec3::new(0.0, 0.0, 1.0), 1.5));
        // Radius 0.5 falls short.
        assert!(!mesh.intersects_sphere(Vec3::new(0.0, 0.0, 1.0), 0.5));
        // Negative radius never intersects.
        assert!(!mesh.intersects_sphere(Vec3::ZERO, -1.0));
    }

    #[test]
    fn overlap_sphere_collects_front_quad_only() {
        let mesh = two_quads();
        // Reach quad A (z = 2) but not quad B (z = 6).
        let tris = mesh.overlap_sphere(Vec3::new(0.0, 0.0, 1.0), 1.5);
        assert_eq!(tris.len(), 2, "both front triangles: {tris:?}");
        assert!(tris.iter().all(|&t| t < 2));
    }

    #[test]
    fn overlap_aabb_selects_crossing_triangles() {
        let mesh = two_quads();
        // A thin box straddling quad A's plane (z = 2) over the unit square.
        let box_ = crate::bounding::Aabb::new(
            Vec3::new(-2.0, -2.0, 1.9),
            Vec3::new(2.0, 2.0, 2.1),
        );
        let tris = mesh.overlap_aabb(&box_);
        assert_eq!(tris.len(), 2, "both front triangles: {tris:?}");
        assert!(tris.iter().all(|&t| t < 2));
    }

    #[test]
    fn overlap_aabb_rejects_boxes_between_quads() {
        let mesh = two_quads();
        // Box in the gap between the quads (z in [3, 4]) touches neither.
        let box_ = crate::bounding::Aabb::new(
            Vec3::new(-2.0, -2.0, 3.0),
            Vec3::new(2.0, 2.0, 4.0),
        );
        assert!(mesh.overlap_aabb(&box_).is_empty());
    }

    #[test]
    fn closest_point_on_empty_mesh_is_none() {
        let mesh = TriangleMesh::new(alloc::vec![], alloc::vec![]);
        assert!(mesh.closest_point(Vec3::ZERO).is_none());
        assert!(mesh.distance(Vec3::ZERO).is_none());
        assert!(!mesh.intersects_sphere(Vec3::ZERO, 10.0));
        assert!(mesh.overlap_sphere(Vec3::ZERO, 10.0).is_empty());
    }

    #[test]
    fn sphere_cast_stops_at_front_quad() {
        let mesh = two_quads();
        // Radius-0.5 sphere falling +Z toward quad A at z = 2.
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = mesh.sphere_cast(&ray, 0.5).expect("sweep hit");
        // Surface at z = 2, sphere stops with centre at z = 1.5.
        assert!((hit.t - 1.5).abs() < 1e-3, "t = {}", hit.t);
        assert!(hit.triangle < 2, "front triangle, got {}", hit.triangle);
        assert!(hit.normal.z < -0.99, "normal faces the ray: {:?}", hit.normal);
    }

    #[test]
    fn sphere_cast_misses_when_offset_far() {
        let mesh = two_quads();
        // Centre path at x = 5 is well outside the unit quads for radius 0.5.
        let ray = Ray::new(Vec3::new(5.0, 0.0, 0.0), Vec3::Z);
        assert!(mesh.sphere_cast(&ray, 0.5).is_none());
    }

    #[test]
    fn sphere_cast_radius_catches_grazing_edge() {
        let mesh = two_quads();
        // Centre path at x = 1.4 misses the quad (half-width 1.0) by 0.4, but a
        // radius-0.5 sphere still clips the +x edge of quad A.
        let ray = Ray::new(Vec3::new(1.4, 0.0, 0.0), Vec3::Z);
        assert!(mesh.sphere_cast(&ray, 0.5).is_some());
        // A smaller radius-0.3 sphere stays clear.
        assert!(mesh.sphere_cast(&ray, 0.3).is_none());
    }

    #[test]
    fn sphere_cast_respects_tmax() {
        let mesh = two_quads();
        let ray = Ray::with_tmax(Vec3::ZERO, Vec3::Z, 1.0);
        // Contact needs t = 1.5 but tmax is 1.0.
        assert!(mesh.sphere_cast(&ray, 0.5).is_none());
    }

    #[test]
    fn sphere_contacts_on_face_reports_depth_and_normal() {
        let mesh = two_quads();
        // Sphere centre at z = 1 just in front of quad A (surface z = 2) with a
        // radius of 1.5 penetrates the face by 0.5.
        let contacts = mesh.sphere_contacts(Vec3::new(0.0, 0.0, 1.0), 1.5);
        assert_eq!(contacts.len(), 2, "both front triangles: {contacts:?}");
        for c in &contacts {
            assert!(c.triangle < 2, "front triangle, got {}", c.triangle);
            assert!(c.point.abs_diff_eq(Vec3::new(0.0, 0.0, 2.0), 1e-5));
            // Centre is on the -Z side of the quad, so the push-out points -Z.
            assert!(c.normal.z < -0.99, "normal toward centre: {:?}", c.normal);
            assert!((c.depth - 0.5).abs() < 1e-5, "depth = {}", c.depth);
        }
    }

    #[test]
    fn sphere_contacts_clamp_to_edge() {
        let mesh = two_quads();
        // Centre off the +x edge of quad A (edge at x = 1, z = 2): closest point
        // is the edge, distance = sqrt(0.4^2 + 0.2^2) ~= 0.4472 < radius 0.6.
        let contacts = mesh.sphere_contacts(Vec3::new(1.4, 0.0, 2.2), 0.6);
        assert!(!contacts.is_empty(), "edge contact expected");
        let c = contacts
            .iter()
            .min_by(|a, b| a.depth.total_cmp(&b.depth))
            .expect("contact");
        assert!((c.point.x - 1.0).abs() < 1e-5, "clamped x = {}", c.point.x);
        let (dx, dz) = (1.4f32 - 1.0, 0.2f32);
        let expected = 0.6 - (dx * dx + dz * dz).sqrt();
        assert!((c.depth - expected).abs() < 1e-4, "depth = {}", c.depth);
        // Normal points from the edge toward the centre (outward +x / +z-ish).
        assert!(c.normal.x > 0.0 && c.normal.z > 0.0, "normal = {:?}", c.normal);
    }

    #[test]
    fn sphere_contacts_centre_on_face_uses_face_normal() {
        let mesh = two_quads();
        // Centre exactly on quad A's plane: the surface-to-centre direction is
        // degenerate, so the normal must fall back to the face normal (+/-Z).
        let contacts = mesh.sphere_contacts(Vec3::new(0.0, 0.0, 2.0), 0.5);
        assert!(!contacts.is_empty());
        for c in &contacts {
            assert!((c.depth - 0.5).abs() < 1e-5, "full-radius depth: {}", c.depth);
            assert!(c.normal.z.abs() > 0.99, "face normal fallback: {:?}", c.normal);
        }
    }

    #[test]
    fn sphere_contacts_reject_when_out_of_range() {
        let mesh = two_quads();
        // Radius 0.5 at z = 1 falls short of quad A at z = 2.
        assert!(mesh.sphere_contacts(Vec3::new(0.0, 0.0, 1.0), 0.5).is_empty());
        // Negative radius never contacts.
        assert!(mesh.sphere_contacts(Vec3::ZERO, -1.0).is_empty());
        // Empty mesh never contacts.
        let empty = TriangleMesh::new(alloc::vec![], alloc::vec![]);
        assert!(empty.sphere_contacts(Vec3::ZERO, 10.0).is_empty());
    }

    #[test]
    fn capsule_contacts_axis_parallel_to_face() {
        let mesh = two_quads();
        // Capsule axis lies in the plane z = 1.4 (0.6 below quad A at z = 2),
        // spanning x in [-0.5, 0.5]; radius 1.0 reaches the face.
        let capsule = Capsule::new(
            Vec3::new(-0.5, 0.0, 1.4),
            Vec3::new(0.5, 0.0, 1.4),
            1.0,
        );
        let contacts = mesh.capsule_contacts(&capsule);
        assert_eq!(contacts.len(), 2, "both front triangles: {contacts:?}");
        for c in &contacts {
            assert!(c.triangle < 2, "front triangle, got {}", c.triangle);
            assert!(c.point.z > 1.99 && c.point.z < 2.01, "on face: {:?}", c.point);
            assert!(c.normal.z < -0.99, "push toward axis (-z): {:?}", c.normal);
            assert!((c.depth - 0.4).abs() < 1e-4, "depth = {}", c.depth);
        }
    }

    #[test]
    fn capsule_contacts_endpoint_reaches_face() {
        let mesh = two_quads();
        // Vertical capsule whose top end-cap pokes into quad A at z = 2.
        let capsule = Capsule::new(
            Vec3::new(0.0, 0.0, 1.6),
            Vec3::new(0.0, 0.0, 0.0),
            0.6,
        );
        let contacts = mesh.capsule_contacts(&capsule);
        assert!(!contacts.is_empty(), "end-cap should touch the face");
        for c in &contacts {
            assert!(c.triangle < 2);
            // Nearest axis point is the z = 1.6 end, gap 0.4 < radius 0.6.
            assert!((c.depth - 0.2).abs() < 1e-4, "depth = {}", c.depth);
        }
    }

    #[test]
    fn capsule_contacts_reject_out_of_range() {
        let mesh = two_quads();
        // Axis at z = 1 with radius 0.5 falls short of quad A at z = 2.
        let far = Capsule::new(Vec3::new(-1.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0), 0.5);
        assert!(mesh.capsule_contacts(&far).is_empty());
        // Negative radius never contacts.
        let neg = Capsule::new(Vec3::ZERO, Vec3::X, -1.0);
        assert!(mesh.capsule_contacts(&neg).is_empty());
        // Empty mesh never contacts.
        let empty = TriangleMesh::new(alloc::vec![], alloc::vec![]);
        let cap = Capsule::new(Vec3::ZERO, Vec3::X, 10.0);
        assert!(empty.capsule_contacts(&cap).is_empty());
    }

    #[test]
    fn capsule_contacts_axis_pierces_face() {
        let mesh = two_quads();
        // Axis crosses quad A (z = 2): penetration depth equals the full radius.
        let capsule = Capsule::new(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 3.0),
            0.4,
        );
        let contacts = mesh.capsule_contacts(&capsule);
        assert!(!contacts.is_empty());
        let pierced = contacts.iter().any(|c| (c.depth - 0.4).abs() < 1e-4);
        assert!(pierced, "piercing axis gives full-radius depth: {contacts:?}");
    }

    #[test]
    fn capsule_cast_stops_at_front_quad() {
        let mesh = two_quads();
        // Horizontal capsule (axis along X) at z = 0 swept +Z toward quad A at
        // z = 2. Radius 0.5 so the surface meets the face at t = 1.5.
        let capsule = Capsule::new(
            Vec3::new(-0.3, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
            0.5,
        );
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = mesh.capsule_cast(&capsule, &ray).expect("capsule sweep hit");
        assert!((hit.t - 1.5).abs() < 1e-2, "t = {}", hit.t);
        assert!(hit.triangle < 2, "front triangle, got {}", hit.triangle);
        assert!(hit.normal.z < -0.99, "normal faces the ray: {:?}", hit.normal);
    }

    #[test]
    fn capsule_cast_misses_when_offset_far() {
        let mesh = two_quads();
        // Axis path well off the +x side of the unit quads for radius 0.3.
        let capsule = Capsule::new(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.6, 0.0, 0.0),
            0.3,
        );
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::Z);
        assert!(mesh.capsule_cast(&capsule, &ray).is_none());
    }

    #[test]
    fn capsule_cast_respects_tmax() {
        let mesh = two_quads();
        let capsule = Capsule::new(
            Vec3::new(-0.3, 0.0, 0.0),
            Vec3::new(0.3, 0.0, 0.0),
            0.5,
        );
        // Contact needs t = 1.5 but tmax is 1.0.
        let ray = Ray::with_tmax(Vec3::ZERO, Vec3::Z, 1.0);
        assert!(mesh.capsule_cast(&capsule, &ray).is_none());
    }

    #[test]
    fn capsule_cast_empty_mesh_is_none() {
        let mesh = TriangleMesh::new(alloc::vec![], alloc::vec![]);
        let capsule = Capsule::new(Vec3::ZERO, Vec3::X, 0.5);
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        assert!(mesh.capsule_cast(&capsule, &ray).is_none());
    }

    #[test]
    fn empty_mesh_has_no_hits() {
        let mesh = TriangleMesh::new(alloc::vec![], alloc::vec![]);
        assert!(mesh.is_empty());
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        assert!(mesh.ray_cast(&ray).is_none());
    }
}
