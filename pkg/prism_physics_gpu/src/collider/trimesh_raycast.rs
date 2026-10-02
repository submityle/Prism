//! Ray-versus-triangle-mesh scene query: the nearest triangle a directed ray
//! pierces in a static [`Trimesh`], with the exact hit point, barycentric
//! weights, and the surface normal oriented back toward the ray origin.
//!
//! This is the mesh analogue of the convex scene-query family in
//! [`crate::narrowphase`]: where [`ray_cast_bvh`](crate::narrowphase::ray_cast_bvh)
//! casts a ray at a set of convex hulls, this casts a ray at the triangle soup
//! that `AAA` engines use for all static level geometry. It is the primitive
//! behind `PhysX` `PxMeshQuery::raycast`, Jolt `MeshShape::CastRay`, and Unreal
//! `Chaos` triangle-mesh line traces.
//!
//! # Two interchangeable paths
//!
//! * [`cpu_trimesh_raycast`] is the brute-force golden: it intersects the ray
//!   with every triangle and keeps the nearest. It is the independent oracle the
//!   accelerated path and the `GPU` twin are pinned to.
//! * [`cpu_trimesh_raycast_bvh`] descends the `LBVH` built from
//!   [`Trimesh::triangle_aabbs`] and prunes any subtree the ray enters only
//!   farther than the nearest triangle found so far, so it returns the identical
//!   hit while touching a fraction of the triangles.
//!
//! # Moller-Trumbore, shared across paths and the device
//!
//! Both paths and the `GPU` kernel intersect a single ray with a single triangle
//! with the Moller-Trumbore algorithm (1997): build the two edge vectors from
//! the first vertex, form the determinant against the ray direction, and read
//! the barycentric coordinates and the ray parameter straight out of the triple
//! products. The test is double-sided - a ray is accepted whether it strikes the
//! front (`+normal`) or back face - and a near-zero determinant rejects a ray
//! parallel to the triangle plane. The reported [`TrimeshRayHit::front_face`]
//! records which side was struck and the normal is flipped to always point back
//! toward the ray origin, matching the convex
//! [`RayCastHit`](crate::narrowphase::RayCastHit) normal convention.
//!
//! # Determinism
//!
//! The nearest hit wins; on an exact distance tie the lower triangle index wins,
//! because every path keeps a candidate only when it is *strictly* closer and
//! visits triangles in ascending index order. The result is therefore stable and
//! identical across the brute, `BVH`, and device paths.
//!
//! # Provenance
//!
//! Ray-triangle intersection per Moller and Trumbore, "Fast, Minimum Storage
//! Ray/Triangle Intersection" (1997); `LBVH` per Karras 2012; branch-free slab
//! box test per Williams et al. 2005. No Unreal Engine source or derived code.

use glam::Vec3;

use crate::bvh::{cpu_build_lbvh, Aabb, Lbvh};

use super::Trimesh;

/// A directed ray for a mesh query: an origin, a direction, and a maximum
/// travel distance.
///
/// The ray is the point set `origin + t * direction` for `t` in
/// `[0, max_distance]`. [`direction`](MeshRay::direction) should be a unit
/// vector so the reported [`TrimeshRayHit::distance`] is a world-space length;
/// a non-unit direction scales the reported distance by the direction's length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshRay {
    /// World-space starting point of the ray.
    pub origin: Vec3,
    /// Ray direction; should be unit length for a metric distance.
    pub direction: Vec3,
    /// Largest `t` a hit may lie at; triangles struck beyond it are rejected.
    pub max_distance: f32,
}

impl MeshRay {
    /// Creates a ray from its `origin`, `direction`, and `max_distance`.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3, max_distance: f32) -> MeshRay {
        MeshRay {
            origin,
            direction,
            max_distance,
        }
    }
}

/// A single triangle-mesh ray hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimeshRayHit {
    /// Zero-based index of the triangle that was struck, addressing the mesh in
    /// its original (unsorted) triangle order.
    pub triangle: u32,
    /// Ray parameter `t` at the hit, a world-space length for a unit direction.
    pub distance: f32,
    /// World-space hit point `origin + direction * distance`.
    pub point: Vec3,
    /// Barycentric weights `(w_a, w_b, w_c)` of the hit on the triangle, each in
    /// `[0, 1]` and summing to one.
    pub bary: Vec3,
    /// Unit surface normal, flipped to point back toward the ray origin.
    pub normal: Vec3,
    /// Whether the ray struck the triangle's front (`+geometric-normal`) face.
    pub front_face: bool,
}

/// Determinant magnitude below which the ray is treated as parallel to the
/// triangle plane and the triangle is missed.
const PARALLEL_EPS: f32 = 1e-8;

/// The raw Moller-Trumbore solve for one ray against one triangle `(a, b, c)`.
///
/// Returns the ray parameter `t` and the barycentric edge coordinates `(u, v)`
/// (weights of `b` and `c`) when the ray strikes the triangle within
/// `[0, max_distance]`, or [`None`] on a miss. Double-sided: a near-zero
/// determinant (parallel ray) misses, but either face is otherwise accepted.
/// This is the one intersection kernel every path and the device twin share.
#[must_use]
pub(crate) fn ray_triangle_uvt(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, f32, f32)> {
    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = direction.cross(edge2);
    let det = edge1.dot(pvec);
    if det.abs() < PARALLEL_EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = origin - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(qvec) * inv_det;
    if t < 0.0 || t > max_distance {
        return None;
    }
    Some((t, u, v))
}

/// Builds the full [`TrimeshRayHit`] for triangle `index` from the shared
/// Moller-Trumbore outputs `(t, u, v)`.
///
/// The point is `origin + direction * t`; the barycentric weights are
/// `(1 - u - v, u, v)`; the geometric normal is `(b - a) x (c - a)`, and
/// [`front_face`](TrimeshRayHit::front_face) is set when the ray opposes that
/// normal. The stored [`normal`](TrimeshRayHit::normal) is the unit geometric
/// normal flipped to always point back toward the ray origin. `CPU` and `GPU`
/// paths both finalise through this one rule, so their hits are identical.
#[must_use]
pub(crate) fn finalize_hit(
    index: u32,
    direction: Vec3,
    origin: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    t: f32,
    u: f32,
    v: f32,
) -> TrimeshRayHit {
    let point = origin + direction * t;
    let bary = Vec3::new(1.0 - u - v, u, v);
    let geom = (b - a).cross(c - a);
    let front_face = direction.dot(geom) < 0.0;
    let unit = geom.normalize_or_zero();
    let normal = if front_face { unit } else { -unit };
    TrimeshRayHit {
        triangle: index,
        distance: t,
        point,
        bary,
        normal,
        front_face,
    }
}

/// Whether `candidate` should replace `best` under a total order on
/// `(distance, triangle)`: strictly nearer wins, and an exact distance tie
/// resolves to the lower triangle index. The index tie-break is what keeps the
/// traversal-order `BVH` walk and the `GPU` host reduction identical to the
/// index-order brute golden when two triangles share the struck edge or vertex.
/// Shared by the `CPU` and `GPU` nearest-hit reductions.
#[must_use]
#[expect(
    clippy::float_cmp,
    reason = "exact distance equality is the intended tie detector; the triangle index then gives a deterministic total order"
)]
pub(crate) fn closer_hit(candidate: &TrimeshRayHit, best: &Option<TrimeshRayHit>) -> bool {
    match best {
        Some(b) => {
            candidate.distance < b.distance
                || (candidate.distance == b.distance && candidate.triangle < b.triangle)
        }
        None => true,
    }
}

/// Casts `ray` at `mesh` by brute force, returning the nearest triangle hit.
///
/// Intersects the ray with every triangle and keeps the nearest; ties resolve to
/// the lowest triangle index. Returns [`None`] when the ray misses every
/// triangle or the mesh is empty. This is the independent golden the accelerated
/// and device paths are pinned to.
#[must_use]
pub fn cpu_trimesh_raycast(mesh: &Trimesh, ray: &MeshRay) -> Option<TrimeshRayHit> {
    let mut best: Option<TrimeshRayHit> = None;
    for i in 0..mesh.triangle_count() {
        let tri = mesh.triangle(i);
        let Some((t, u, v)) = ray_triangle_uvt(
            ray.origin,
            ray.direction,
            ray.max_distance,
            tri.a,
            tri.b,
            tri.c,
        ) else {
            continue;
        };
        let index = u32::try_from(i).unwrap_or(u32::MAX);
        let hit = finalize_hit(
            index,
            ray.direction,
            ray.origin,
            tri.a,
            tri.b,
            tri.c,
            t,
            u,
            v,
        );
        if closer_hit(&hit, &best) {
            best = Some(hit);
        }
    }
    best
}

/// The entry distance where `ray` enters `aabb`, or [`None`] on a miss.
///
/// Branch-free slab test of Williams et al. (2005): reciprocate the direction
/// per component (a zero component becomes a signed infinity, matching `WGSL`
/// `1.0 / 0.0`), map each slab to an entry and exit distance, and take the
/// largest entry and smallest exit across axes, clamping the entry to zero so an
/// origin inside the box enters at distance zero. The entry is a lower bound on
/// the ray parameter of any triangle contained in the box, so pruning a subtree
/// whose entry exceeds the best hit so far never discards a nearer triangle.
#[must_use]
fn aabb_ray_enter(ray: &MeshRay, aabb: &Aabb) -> Option<f32> {
    let inv = Vec3::ONE / ray.direction;
    let t0 = (aabb.min - ray.origin) * inv;
    let t1 = (aabb.max - ray.origin) * inv;
    let t_near = t0.min(t1).max_element().max(0.0);
    let t_far = t0.max(t1).min_element();
    if t_near <= t_far && t_near <= ray.max_distance {
        Some(t_near)
    } else {
        None
    }
}

/// The bounding box of an encoded `BVH` node: its leaf box or internal union.
#[must_use]
fn node_aabb(tree: &Lbvh, encoded: u32) -> Aabb {
    if tree.is_leaf(encoded) {
        tree.leaf_aabb[(encoded as usize) - tree.num_internal]
    } else {
        tree.internal_aabb[encoded as usize]
    }
}

/// Accelerated form of [`cpu_trimesh_raycast`]: descends the `LBVH` over the
/// mesh's per-triangle boxes and prunes any subtree the ray enters only farther
/// than the nearest triangle found so far.
///
/// `lbvh` must be the hierarchy built from `mesh.triangle_aabbs()`; its leaf
/// slots map back to original triangle indices through
/// [`Lbvh::sorted_indices`]. The result matches [`cpu_trimesh_raycast`] exactly,
/// including the lowest-index rule on a distance tie. Returns [`None`] when the
/// ray misses every triangle or the tree is empty.
#[must_use]
pub fn cpu_trimesh_raycast_bvh(mesh: &Trimesh, lbvh: &Lbvh, ray: &MeshRay) -> Option<TrimeshRayHit> {
    if lbvh.num_leaves == 0 {
        return None;
    }
    let mut best: Option<TrimeshRayHit> = None;
    let mut stack = vec![lbvh.root];
    while let Some(node) = stack.pop() {
        let Some(enter) = aabb_ray_enter(ray, &node_aabb(lbvh, node)) else {
            continue;
        };
        if let Some(b) = best
            && enter > b.distance
        {
            continue;
        }
        if lbvh.is_leaf(node) {
            let leaf = (node as usize) - lbvh.num_internal;
            let prim = lbvh.sorted_indices[leaf] as usize;
            let tri = mesh.triangle(prim);
            let Some((t, u, v)) = ray_triangle_uvt(
                ray.origin,
                ray.direction,
                ray.max_distance,
                tri.a,
                tri.b,
                tri.c,
            ) else {
                continue;
            };
            let index = u32::try_from(prim).unwrap_or(u32::MAX);
            let hit = finalize_hit(
                index,
                ray.direction,
                ray.origin,
                tri.a,
                tri.b,
                tri.c,
                t,
                u,
                v,
            );
            if closer_hit(&hit, &best) {
                best = Some(hit);
            }
        } else {
            stack.push(lbvh.left[node as usize]);
            stack.push(lbvh.right[node as usize]);
        }
    }
    best
}

/// Builds the mesh's `LBVH` and casts `ray` at it in one call, the convenience
/// form of [`cpu_trimesh_raycast_bvh`] for callers that do not cache the tree.
#[must_use]
pub fn cpu_trimesh_raycast_built(mesh: &Trimesh, ray: &MeshRay) -> Option<TrimeshRayHit> {
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    cpu_trimesh_raycast_bvh(mesh, &lbvh, ray)
}

#[cfg(test)]
mod tests {
    use super::{
        cpu_trimesh_raycast, cpu_trimesh_raycast_built, cpu_trimesh_raycast_bvh, MeshRay,
    };
    use crate::collider::Trimesh;
    use glam::Vec3;

    /// A unit quad in the `z = 0` plane spanning `[0, 1]^2`, two CCW triangles
    /// seen from `+z`.
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
    fn hits_front_face_from_plus_z() {
        let mesh = unit_quad();
        // Ray from +z aimed at the quad centre, travelling -z.
        let ray = MeshRay::new(Vec3::new(0.25, 0.25, 5.0), Vec3::new(0.0, 0.0, -1.0), 100.0);
        let hit = cpu_trimesh_raycast(&mesh, &ray).expect("hits the quad");
        assert_eq!(hit.triangle, 0, "lower quad triangle covers (0.25, 0.25)");
        assert!((hit.distance - 5.0).abs() < 1e-4, "distance was {}", hit.distance);
        assert!((hit.point - Vec3::new(0.25, 0.25, 0.0)).length() < 1e-4);
        assert!(hit.front_face, "ray comes from the +normal side");
        assert!((hit.normal - Vec3::Z).length() < 1e-4, "normal points back at origin");
    }

    #[test]
    fn hits_back_face_and_flips_normal() {
        let mesh = unit_quad();
        // Ray from -z travelling +z strikes the back face; normal must still
        // point back toward the origin, i.e. -z.
        let ray = MeshRay::new(Vec3::new(0.25, 0.25, -5.0), Vec3::new(0.0, 0.0, 1.0), 100.0);
        let hit = cpu_trimesh_raycast(&mesh, &ray).expect("hits the quad");
        assert!(!hit.front_face, "struck the back face");
        assert!((hit.normal - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
    }

    #[test]
    fn misses_when_aimed_away() {
        let mesh = unit_quad();
        let ray = MeshRay::new(Vec3::new(0.25, 0.25, 5.0), Vec3::new(0.0, 0.0, 1.0), 100.0);
        assert!(cpu_trimesh_raycast(&mesh, &ray).is_none(), "ray travels away from quad");
    }

    #[test]
    fn respects_max_distance() {
        let mesh = unit_quad();
        let ray = MeshRay::new(Vec3::new(0.25, 0.25, 5.0), Vec3::new(0.0, 0.0, -1.0), 2.0);
        assert!(cpu_trimesh_raycast(&mesh, &ray).is_none(), "quad is 5 away, max is 2");
    }

    #[test]
    fn bvh_matches_brute_on_nearest_of_many() {
        // A fan of parallel quads at increasing z; the nearest to a +z origin
        // travelling -z must win on both paths. The ray stays off the shared
        // diagonal edge (y < x), so the winning triangle is unambiguous.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for k in 0..8_u32 {
            let z = k as f32;
            let base = (vertices.len()) as u32;
            vertices.push(Vec3::new(-1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, 1.0, z));
            vertices.push(Vec3::new(-1.0, 1.0, z));
            indices.push([base, base + 1, base + 2]);
            indices.push([base, base + 2, base + 3]);
        }
        let mesh = Trimesh::new(vertices, indices);
        let ray = MeshRay::new(Vec3::new(0.3, 0.1, 20.0), Vec3::new(0.0, 0.0, -1.0), 100.0);
        let brute = cpu_trimesh_raycast(&mesh, &ray).expect("hits");
        let bvh = cpu_trimesh_raycast_built(&mesh, &ray).expect("hits");
        assert_eq!(brute.triangle, bvh.triangle, "same winning triangle");
        assert!((brute.distance - bvh.distance).abs() < 1e-5);
        // The nearest quad is at z = 7 (distance 13 from z = 20).
        assert!((brute.distance - 13.0).abs() < 1e-4, "distance was {}", brute.distance);
        // Quad k = 7 occupies triangles 14 and 15; the off-diagonal point lands
        // in the lower-right triangle (even index), so 14 must win.
        assert_eq!(brute.triangle, 14, "nearest lower-right triangle wins");
    }

    #[test]
    fn empty_mesh_misses() {
        let mesh = Trimesh::new(Vec::new(), Vec::new());
        let ray = MeshRay::new(Vec3::ZERO, Vec3::X, 100.0);
        assert!(cpu_trimesh_raycast(&mesh, &ray).is_none());
        let lbvh = crate::cpu_build_lbvh(&mesh.triangle_aabbs());
        assert!(cpu_trimesh_raycast_bvh(&mesh, &lbvh, &ray).is_none());
    }
}
