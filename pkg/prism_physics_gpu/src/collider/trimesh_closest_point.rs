//! Closest-point scene query against a static triangle mesh.
//!
//! Given a world-space query point and a [`Trimesh`], this reports the nearest
//! point on the mesh surface, the Euclidean distance to it, the triangle that
//! owns the nearest point, the barycentric weights of that point, and a unit
//! normal pointing from the surface back toward the query. It is the mesh
//! counterpart of the convex scene closest-point query and mirrors `PhysX`
//! `PxMeshQuery` / Jolt `MeshShape` point-distance lookups.
//!
//! Three code paths share one geometric kernel so they stay bit-for-bit
//! comparable:
//!
//! 1. [`cpu_trimesh_closest_point`] is the brute golden: it tests the query
//!    against every triangle and keeps the nearest.
//! 2. [`cpu_trimesh_closest_point_bvh`] walks a prebuilt [`Lbvh`] with a
//!    branch-and-bound prune, skipping any subtree whose axis-aligned bound is
//!    already farther than the best surface point found so far.
//! 3. [`cpu_trimesh_closest_point_built`] builds the `LBVH` from the mesh's
//!    per-triangle bounds and runs the pruned walk, as a convenience wrapper.
//!
//! A matching `GPU` kernel in `collider_trimesh_closest_point.wgsl` evaluates
//! every triangle in parallel and the host reduces with the identical rule, so
//! the real-device result equals the brute golden.
//!
//! The per-triangle closest point uses Ericson's Voronoi-region method
//! (*Real-Time Collision Detection*, 2005, section 5.1.5), the same routine the
//! sphere-triangle narrow phase relies on. No Unreal Engine source or derived
//! code.

use glam::Vec3;

use super::Trimesh;
use crate::bvh::{cpu_build_lbvh, Aabb, Lbvh};

/// Squared length below which the query is treated as lying on the surface, so
/// the separating direction is taken from the triangle's geometric normal
/// rather than an unstable near-zero difference.
const ON_SURFACE_EPS2: f32 = 1e-12;

/// A single closest-point result against a triangle mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimeshClosestHit {
    /// Zero-based index of the triangle that owns the nearest surface point,
    /// addressing the mesh in its original (unsorted) triangle order.
    pub triangle: u32,
    /// World-space nearest point on the mesh surface.
    pub point: Vec3,
    /// Euclidean distance from the query to [`point`](Self::point).
    pub distance: f32,
    /// Barycentric weights `(w_a, w_b, w_c)` of the nearest point on its
    /// triangle, each in `[0, 1]` and summing to one.
    pub bary: Vec3,
    /// Unit normal pointing from the surface back toward the query point; it
    /// falls back to the triangle's geometric normal when the query lies on the
    /// surface.
    pub normal: Vec3,
}

/// Returns the point on triangle `(a, b, c)` closest to `p` using Ericson's
/// Voronoi-region cascade. Kept private to this module so the `CPU` paths and
/// the device kernel reduce against the identical arithmetic.
pub(crate) fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Returns the barycentric weights `(w_a, w_b, w_c)` of `q` with respect to
/// triangle `(a, b, c)`, assuming `q` lies in the triangle's plane (as the
/// closest point always does). Follows Ericson's projection form.
fn barycentric(q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = q - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < f32::EPSILON {
        // Degenerate triangle: collapse everything onto vertex A.
        return Vec3::new(1.0, 0.0, 0.0);
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    Vec3::new(1.0 - v - w, v, w)
}

/// Builds the full [`TrimeshClosestHit`] for triangle `index` from the query
/// point `p`, the nearest surface point `q`, and the triangle vertices. Shared
/// by the `CPU` paths and the `GPU` host reduction so every path agrees.
pub(crate) fn finalize_hit(
    index: u32,
    p: Vec3,
    q: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> TrimeshClosestHit {
    let diff = p - q;
    let d2 = diff.dot(diff);
    let normal = if d2 > ON_SURFACE_EPS2 {
        diff * (1.0 / d2.sqrt())
    } else {
        let geom = (b - a).cross(c - a);
        let len = geom.length();
        if len > 0.0 {
            geom * (1.0 / len)
        } else {
            Vec3::ZERO
        }
    };
    TrimeshClosestHit {
        triangle: index,
        point: q,
        distance: d2.sqrt(),
        bary: barycentric(q, a, b, c),
        normal,
    }
}

/// Returns `true` when `cand` should replace `best`: strictly nearer, or an
/// exact distance tie broken toward the lower triangle index. Shared by every
/// path so ties resolve identically.
pub(crate) fn closer_hit(cand: &TrimeshClosestHit, best: &Option<TrimeshClosestHit>) -> bool {
    match best {
        None => true,
        Some(b) => {
            cand.distance < b.distance
                || (cand.distance == b.distance && cand.triangle < b.triangle)
        }
    }
}

/// Brute-force golden: returns the mesh surface point nearest to `point`, or
/// [`None`] when the mesh has no triangles.
#[must_use]
pub fn cpu_trimesh_closest_point(mesh: &Trimesh, point: Vec3) -> Option<TrimeshClosestHit> {
    let mut best: Option<TrimeshClosestHit> = None;
    for i in 0..mesh.triangle_count() {
        let tri = mesh.triangle(i);
        let q = closest_point_on_triangle(point, tri.a, tri.b, tri.c);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "triangle counts fit in u32 for any device-addressable mesh"
        )]
        let cand = finalize_hit(i as u32, point, q, tri.a, tri.b, tri.c);
        if closer_hit(&cand, &best) {
            best = Some(cand);
        }
    }
    best
}

/// Squared distance from `point` to the axis-aligned box `aabb`, zero when the
/// point is inside. Used as the branch-and-bound lower bound.
fn aabb_distance2(point: Vec3, aabb: &Aabb) -> f32 {
    let dx = (aabb.min.x - point.x).max(0.0).max(point.x - aabb.max.x);
    let dy = (aabb.min.y - point.y).max(0.0).max(point.y - aabb.max.y);
    let dz = (aabb.min.z - point.z).max(0.0).max(point.z - aabb.max.z);
    dx * dx + dy * dy + dz * dz
}

/// Returns the bound stored for `encoded` node of `lbvh`, selecting the leaf or
/// internal array the same way the raycast walk does.
fn node_aabb(lbvh: &Lbvh, encoded: u32) -> Aabb {
    if lbvh.is_leaf(encoded) {
        let slot = (encoded as usize) - lbvh.num_internal;
        lbvh.leaf_aabb[slot]
    } else {
        lbvh.internal_aabb[encoded as usize]
    }
}

/// Pruned closest-point walk over a prebuilt [`Lbvh`] of the mesh's
/// per-triangle bounds. Equivalent to [`cpu_trimesh_closest_point`] but skips
/// subtrees whose bound is already farther than the best point found so far.
#[must_use]
pub fn cpu_trimesh_closest_point_bvh(
    mesh: &Trimesh,
    lbvh: &Lbvh,
    point: Vec3,
) -> Option<TrimeshClosestHit> {
    if lbvh.num_leaves == 0 {
        return None;
    }
    let mut best: Option<TrimeshClosestHit> = None;
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    stack.push(lbvh.root);
    while let Some(encoded) = stack.pop() {
        // Prune: skip when the node bound is no nearer than the current best.
        if let Some(b) = best
            && aabb_distance2(point, &node_aabb(lbvh, encoded)) > b.distance * b.distance
        {
            continue;
        }
        if lbvh.is_leaf(encoded) {
            let slot = (encoded as usize) - lbvh.num_internal;
            let tri_index = lbvh.sorted_indices[slot];
            let tri = mesh.triangle(tri_index as usize);
            let q = closest_point_on_triangle(point, tri.a, tri.b, tri.c);
            let cand = finalize_hit(tri_index, point, q, tri.a, tri.b, tri.c);
            if closer_hit(&cand, &best) {
                best = Some(cand);
            }
        } else {
            stack.push(lbvh.left[encoded as usize]);
            stack.push(lbvh.right[encoded as usize]);
        }
    }
    best
}

/// Convenience wrapper: builds the `LBVH` from the mesh's per-triangle bounds
/// and runs [`cpu_trimesh_closest_point_bvh`].
#[must_use]
pub fn cpu_trimesh_closest_point_built(mesh: &Trimesh, point: Vec3) -> Option<TrimeshClosestHit> {
    if mesh.triangle_count() == 0 {
        return None;
    }
    let lbvh = cpu_build_lbvh(&mesh.triangle_aabbs());
    cpu_trimesh_closest_point_bvh(mesh, &lbvh, point)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit quad in the `z = 0` plane split into two triangles sharing the
    /// `(0, 0)`-to-`(1, 1)` diagonal.
    fn unit_quad() -> Trimesh {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        Trimesh::new(vertices, indices)
    }

    #[test]
    fn face_projection_from_above() {
        let mesh = unit_quad();
        // A point above the lower-right triangle projects straight down.
        let hit = cpu_trimesh_closest_point(&mesh, Vec3::new(0.6, 0.2, 3.0)).expect("has a surface");
        assert_eq!(hit.triangle, 0, "lower-right triangle owns (0.6, 0.2)");
        assert!((hit.distance - 3.0).abs() < 1e-5, "distance was {}", hit.distance);
        assert!((hit.point - Vec3::new(0.6, 0.2, 0.0)).length() < 1e-5);
        assert!((hit.normal - Vec3::Z).length() < 1e-5, "normal points up at the query");
    }

    #[test]
    fn clamps_to_edge_outside_face() {
        let mesh = unit_quad();
        // A point beyond the +x edge clamps onto that edge, not the interior.
        let hit = cpu_trimesh_closest_point(&mesh, Vec3::new(2.0, 0.5, 0.0)).expect("has a surface");
        assert!((hit.point - Vec3::new(1.0, 0.5, 0.0)).length() < 1e-5, "point was {:?}", hit.point);
        assert!((hit.distance - 1.0).abs() < 1e-5, "distance was {}", hit.distance);
        assert!((hit.normal - Vec3::X).length() < 1e-5, "normal points out along +x");
    }

    #[test]
    fn clamps_to_shared_vertex() {
        let mesh = unit_quad();
        // Beyond the far corner the nearest feature is the (1, 1) vertex.
        let hit = cpu_trimesh_closest_point(&mesh, Vec3::new(3.0, 3.0, 0.0)).expect("has a surface");
        assert!((hit.point - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-5, "point was {:?}", hit.point);
    }

    #[test]
    fn bvh_matches_brute_on_nearest_of_many() {
        // A stack of quads at increasing z; a query near z = 10 must snap to the
        // nearest quad on both the brute and the pruned path.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for k in 0..8_u32 {
            let z = k as f32;
            let base = vertices.len() as u32;
            vertices.push(Vec3::new(-1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, -1.0, z));
            vertices.push(Vec3::new(1.0, 1.0, z));
            vertices.push(Vec3::new(-1.0, 1.0, z));
            indices.push([base, base + 1, base + 2]);
            indices.push([base, base + 2, base + 3]);
        }
        let mesh = Trimesh::new(vertices, indices);
        // Off the shared diagonal (y < x) so the owning triangle is unambiguous.
        let point = Vec3::new(0.3, 0.1, 10.0);
        let brute = cpu_trimesh_closest_point(&mesh, point).expect("has a surface");
        let bvh = cpu_trimesh_closest_point_built(&mesh, point).expect("has a surface");
        assert_eq!(brute, bvh, "pruned walk must equal brute golden");
        // Nearest quad is at z = 7 (distance 3 from z = 10).
        assert!((brute.distance - 3.0).abs() < 1e-5, "distance was {}", brute.distance);
        assert_eq!(brute.triangle, 14, "nearest lower-right triangle wins");
    }

    #[test]
    fn empty_mesh_has_no_surface() {
        let mesh = Trimesh::new(Vec::new(), Vec::new());
        assert!(cpu_trimesh_closest_point(&mesh, Vec3::ZERO).is_none());
        assert!(cpu_trimesh_closest_point_built(&mesh, Vec3::ZERO).is_none());
    }
}
