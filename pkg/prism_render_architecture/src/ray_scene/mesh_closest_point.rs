//! Closest point on a triangle mesh surface for the `CPU` golden path.
//!
//! Snapping a probe to the nearest surface point is the primitive behind
//! signed-distance baking, collision response, cloth/particle projection,
//! click-to-surface picking, and proximity queries. `AAA` toolchains lean on
//! Christer Ericson's region-based point/triangle test (Real-Time Collision
//! Detection): the query point is classified against the triangle's vertex,
//! edge, and face Voronoi regions with a handful of dot products, so the exact
//! closest point and its barycentric coordinates fall out without any
//! bran-heavy projection or iterative solve.
//!
//! This module brute-forces that test across every triangle and keeps the
//! global minimum. All arithmetic accumulates in `f64`; the only non-polynomial
//! operation is a single square root to turn the minimum squared distance into
//! a distance (permitted by the `CPU` golden-path numeric policy), so results
//! are reproducible across platforms.
//!
//! [`closest_point_on_mesh`] returns [`MeshClosestPoint`] (surface point,
//! distance, triangle index, barycentric weights), or [`None`] for a mesh with
//! no triangles.

use super::triangle_mesh::TriangleMesh;

/// The nearest point found on a mesh surface for a given query point.
///
/// Produced by [`closest_point_on_mesh`]. The barycentric weights sum to one
/// and reconstruct [`MeshClosestPoint::point`] from the owning triangle's
/// vertices, which is convenient for interpolating per-vertex attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshClosestPoint {
    /// Closest surface point in the mesh's local space.
    point: [f32; 3],
    /// Euclidean distance from the query point to [`MeshClosestPoint::point`].
    distance: f32,
    /// Index of the triangle (into the mesh index buffer) that owns the point.
    triangle: u32,
    /// Barycentric weights `(u, v, w)` for the triangle's three vertices.
    barycentric: [f32; 3],
}

impl MeshClosestPoint {
    /// Closest surface point in the mesh's local space.
    pub fn point(&self) -> [f32; 3] {
        self.point
    }

    /// Euclidean distance from the query point to the surface point.
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// Index of the triangle that owns the closest point.
    pub fn triangle(&self) -> u32 {
        self.triangle
    }

    /// Barycentric weights `(u, v, w)` for the triangle's three vertices.
    pub fn barycentric(&self) -> [f32; 3] {
        self.barycentric
    }
}

/// A point/barycentric pair returned by the per-triangle region test.
struct TrianglePoint {
    /// Closest point on the triangle, in `f64`.
    point: [f64; 3],
    /// Barycentric weights `(u, v, w)` of that point.
    barycentric: [f64; 3],
}

/// Dot product of two 3-vectors in `f64`.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Vector subtraction `a - b` in `f64`.
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Returns `base + t * dir` componentwise.
fn mul_add(base: [f64; 3], dir: [f64; 3], t: f64) -> [f64; 3] {
    [base[0] + dir[0] * t, base[1] + dir[1] * t, base[2] + dir[2] * t]
}

/// Closest point on triangle `(a, b, c)` to `p`, with barycentric weights, via
/// Ericson's Voronoi-region classification. Handles degenerate triangles
/// gracefully because every branch is a comparison, never a division by a
/// quantity that can be zero in that branch.
fn closest_point_on_triangle(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> TrianglePoint {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    // Vertex region A.
    if d1 <= 0.0 && d2 <= 0.0 {
        return TrianglePoint { point: a, barycentric: [1.0, 0.0, 0.0] };
    }

    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    // Vertex region B.
    if d3 >= 0.0 && d4 <= d3 {
        return TrianglePoint { point: b, barycentric: [0.0, 1.0, 0.0] };
    }

    // Edge region AB.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return TrianglePoint {
            point: mul_add(a, ab, v),
            barycentric: [1.0 - v, v, 0.0],
        };
    }

    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    // Vertex region C.
    if d6 >= 0.0 && d5 <= d6 {
        return TrianglePoint { point: c, barycentric: [0.0, 0.0, 1.0] };
    }

    // Edge region AC.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return TrianglePoint {
            point: mul_add(a, ac, w),
            barycentric: [1.0 - w, 0.0, w],
        };
    }

    // Edge region BC.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub(c, b);
        return TrianglePoint {
            point: mul_add(b, bc, w),
            barycentric: [0.0, 1.0 - w, w],
        };
    }

    // Face region: project inside the triangle.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let point = mul_add(mul_add(a, ab, v), ac, w);
    TrianglePoint { point, barycentric: [1.0 - v - w, v, w] }
}

/// Finds the closest point on the mesh surface to `query`.
///
/// Iterates every triangle in index order, keeping the one with the smallest
/// squared distance (ties resolve to the lower triangle index for
/// determinism). Returns [`None`] when the mesh has no triangles.
pub fn closest_point_on_mesh(
    mesh: &TriangleMesh,
    query: [f32; 3],
) -> Option<MeshClosestPoint> {
    let positions = mesh.positions();
    let indices = mesh.indices();
    if indices.is_empty() {
        return None;
    }

    let p = [f64::from(query[0]), f64::from(query[1]), f64::from(query[2])];
    let vertex = |i: u32| -> [f64; 3] {
        let v = positions[i as usize];
        [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
    };

    let mut best_dist_sq = f64::INFINITY;
    let mut best: Option<(usize, TrianglePoint)> = None;
    for (face, tri) in indices.iter().enumerate() {
        let a = vertex(tri[0]);
        let b = vertex(tri[1]);
        let c = vertex(tri[2]);
        let candidate = closest_point_on_triangle(p, a, b, c);
        let diff = sub(candidate.point, p);
        let dist_sq = dot(diff, diff);
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best = Some((face, candidate));
        }
    }

    let (face, candidate) = best?;
    let point = candidate.point;
    Some(MeshClosestPoint {
        point: [point[0] as f32, point[1] as f32, point[2] as f32],
        distance: best_dist_sq.sqrt() as f32,
        triangle: face as u32,
        barycentric: [
            candidate.barycentric[0] as f32,
            candidate.barycentric[1] as f32,
            candidate.barycentric[2] as f32,
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// Unit right triangle in the z=0 plane: a=(0,0,0), b=(1,0,0), c=(0,1,0).
    fn unit_triangle() -> TriangleMesh {
        mesh(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], vec![[0, 1, 2]])
    }

    #[test]
    fn empty_mesh_has_no_closest_point() {
        let m = mesh(Vec::new(), Vec::new());
        assert!(closest_point_on_mesh(&m, [0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn point_above_face_projects_straight_down() {
        let m = unit_triangle();
        // Query hovers over the triangle interior at height 3.
        let r = closest_point_on_mesh(&m, [0.25, 0.25, 3.0]).unwrap();
        assert!((r.point()[0] - 0.25).abs() < 1e-5);
        assert!((r.point()[1] - 0.25).abs() < 1e-5);
        assert!(r.point()[2].abs() < 1e-5);
        assert!((r.distance() - 3.0).abs() < 1e-5);
        assert_eq!(r.triangle(), 0);
        // Barycentric weights sum to one.
        let bc = r.barycentric();
        assert!((bc[0] + bc[1] + bc[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn point_past_a_vertex_snaps_to_that_vertex() {
        let m = unit_triangle();
        // Far out past vertex A along the negative diagonal.
        let r = closest_point_on_mesh(&m, [-2.0, -2.0, 0.0]).unwrap();
        assert!((r.point()[0]).abs() < 1e-5);
        assert!((r.point()[1]).abs() < 1e-5);
        assert_eq!(r.barycentric(), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn point_beside_an_edge_snaps_onto_the_edge() {
        let m = unit_triangle();
        // Out along +x past the AB edge midpoint, below the hypotenuse region.
        let r = closest_point_on_mesh(&m, [0.5, -1.0, 0.0]).unwrap();
        assert!((r.point()[0] - 0.5).abs() < 1e-5, "x {}", r.point()[0]);
        assert!(r.point()[1].abs() < 1e-5, "y {}", r.point()[1]);
        // On edge AB, weight on C is zero.
        assert!(r.barycentric()[2].abs() < 1e-5);
    }

    #[test]
    fn reconstructs_point_from_barycentric_weights() {
        let m = unit_triangle();
        let r = closest_point_on_mesh(&m, [0.1, 0.2, 1.5]).unwrap();
        let bc = r.barycentric();
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0];
        let c = [0.0f32, 1.0, 0.0];
        let recon = [
            bc[0] * a[0] + bc[1] * b[0] + bc[2] * c[0],
            bc[0] * a[1] + bc[1] * b[1] + bc[2] * c[1],
            bc[0] * a[2] + bc[1] * b[2] + bc[2] * c[2],
        ];
        let got = r.point();
        for (axis, &value) in recon.iter().enumerate() {
            assert!((value - got[axis]).abs() < 1e-5, "axis {axis}");
        }
    }

    #[test]
    fn picks_nearest_of_two_triangles() {
        // Triangle 0 at z=0, triangle 1 at z=10; query near z=0 picks face 0.
        let m = mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 10.0],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );
        let near = closest_point_on_mesh(&m, [0.2, 0.2, 1.0]).unwrap();
        assert_eq!(near.triangle(), 0);
        let far = closest_point_on_mesh(&m, [0.2, 0.2, 9.0]).unwrap();
        assert_eq!(far.triangle(), 1);
    }
}
