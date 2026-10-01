//! Triangle/triangle intersection and mesh self-intersection detection for the
//! `CPU` golden path.
//!
//! Self-intersecting geometry breaks boolean operations, shell extraction,
//! collision baking, and watertight exports, so `AAA` mesh-validation tools
//! flag it early. The core query is Tomas Möller's "A Fast Triangle-Triangle
//! Intersection Test": two triangles are tested by signed distances to each
//! other's plane, and when they straddle, their intersections with the line of
//! the two planes are reduced to a 1-D interval-overlap check. The coplanar
//! case falls back to 2-D edge/edge crossing plus point-in-triangle containment
//! on the dominant plane axis. Every step is dot/cross/compare arithmetic — no
//! transcendental function and no square root.
//!
//! [`triangles_intersect`] is the primitive pair test; [`mesh_self_intersections`]
//! brute-forces it across all face pairs, skipping topologically adjacent faces
//! (those sharing a vertex index) so merely-touching neighbours are not
//! reported, and returns the colliding `(face_a, face_b)` pairs with `a < b`.

use super::triangle_mesh::TriangleMesh;

/// Dot product of two 3-vectors in `f64`.
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Vector subtraction `a - b`.
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a × b`.
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns `true` when all three values share a strict sign (all positive or
/// all negative), i.e. the triangle lies entirely on one side of a plane.
fn same_strict_sign(a: f64, b: f64, c: f64) -> bool {
    (a > 0.0 && b > 0.0 && c > 0.0) || (a < 0.0 && b < 0.0 && c < 0.0)
}

/// 2-D orientation sign of the triangle `(a, b, c)`: positive for CCW.
fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Returns `true` when 2-D point `p` lies inside or on triangle `(a, b, c)`.
fn point_in_triangle_2d(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let d1 = orient2d(a, b, p);
    let d2 = orient2d(b, c, p);
    let d3 = orient2d(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Returns `true` when 2-D segments `p1p2` and `q1q2` cross or touch.
fn segments_intersect_2d(p1: [f64; 2], p2: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> bool {
    let d1 = orient2d(q1, q2, p1);
    let d2 = orient2d(q1, q2, p2);
    let d3 = orient2d(p1, p2, q1);
    let d4 = orient2d(p1, p2, q2);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    // Collinear/touching endpoints.
    let on_segment = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> bool {
        c[0] >= a[0].min(b[0])
            && c[0] <= a[0].max(b[0])
            && c[1] >= a[1].min(b[1])
            && c[1] <= a[1].max(b[1])
    };
    (d1 == 0.0 && on_segment(q1, q2, p1))
        || (d2 == 0.0 && on_segment(q1, q2, p2))
        || (d3 == 0.0 && on_segment(p1, p2, q1))
        || (d4 == 0.0 && on_segment(p1, p2, q2))
}

/// Projects a 3-D point onto the two axes other than `drop_axis`.
fn project_2d(p: [f64; 3], drop_axis: usize) -> [f64; 2] {
    match drop_axis {
        0 => [p[1], p[2]],
        1 => [p[0], p[2]],
        _ => [p[0], p[1]],
    }
}

/// Coplanar triangle overlap: project both onto the plane's dominant axis and
/// test edge/edge crossings plus mutual vertex containment.
fn coplanar_overlap(
    normal: [f64; 3],
    t1: [[f64; 3]; 3],
    t2: [[f64; 3]; 3],
) -> bool {
    // Drop the axis with the largest normal magnitude for the most stable 2-D
    // projection.
    let abs = [normal[0].abs(), normal[1].abs(), normal[2].abs()];
    let drop_axis = if abs[0] >= abs[1] && abs[0] >= abs[2] {
        0
    } else if abs[1] >= abs[2] {
        1
    } else {
        2
    };
    let a = [
        project_2d(t1[0], drop_axis),
        project_2d(t1[1], drop_axis),
        project_2d(t1[2], drop_axis),
    ];
    let b = [
        project_2d(t2[0], drop_axis),
        project_2d(t2[1], drop_axis),
        project_2d(t2[2], drop_axis),
    ];
    // Any edge crossing means overlap.
    for i in 0..3 {
        let a0 = a[i];
        let a1 = a[(i + 1) % 3];
        for j in 0..3 {
            let b0 = b[j];
            let b1 = b[(j + 1) % 3];
            if segments_intersect_2d(a0, a1, b0, b1) {
                return true;
            }
        }
    }
    // Full containment (no edge crossing): one triangle inside the other.
    point_in_triangle_2d(a[0], b[0], b[1], b[2])
        || point_in_triangle_2d(b[0], a[0], a[1], a[2])
}

/// Selects the "lone" vertex — the one that sits by itself on one side of the
/// other triangle's plane — using Möller's full sign classification, which also
/// copes with vertices lying exactly on the plane (zero distance).
fn lone_vertex(d: [f64; 3]) -> usize {
    if d[0] * d[1] > 0.0 {
        2
    } else if d[0] * d[2] > 0.0 {
        1
    } else if d[1] * d[2] > 0.0 || d[0] != 0.0 {
        0
    } else if d[1] != 0.0 {
        1
    } else {
        2
    }
}

/// Computes the parametric interval `[min, max]` where a triangle's projection
/// onto the intersection line overlaps the other triangle's plane.
///
/// `d` holds the signed distances of the triangle's vertices to the other
/// plane; `p` holds the vertices projected onto the intersection-line axis. The
/// lone vertex (opposite side) anchors the two edge/plane crossing points.
fn triangle_interval(d: [f64; 3], p: [f64; 3]) -> [f64; 2] {
    let lone = lone_vertex(d);
    let others: Vec<usize> = (0..3).filter(|&k| k != lone).collect();
    let i0 = others[0];
    let i2 = others[1];
    let t0 = p[i0] + (p[lone] - p[i0]) * d[i0] / (d[i0] - d[lone]);
    let t1 = p[i2] + (p[lone] - p[i2]) * d[i2] / (d[i2] - d[lone]);
    if t0 <= t1 {
        [t0, t1]
    } else {
        [t1, t0]
    }
}

/// Returns `true` when triangles `t1` and `t2` intersect (share any point),
/// including coplanar overlap and edge/vertex touching.
pub fn triangles_intersect(t1: [[f64; 3]; 3], t2: [[f64; 3]; 3]) -> bool {
    // Plane of t2.
    let n2 = cross(sub(t2[1], t2[0]), sub(t2[2], t2[0]));
    let d2 = -dot(n2, t2[0]);
    let dv0 = dot(n2, t1[0]) + d2;
    let dv1 = dot(n2, t1[1]) + d2;
    let dv2 = dot(n2, t1[2]) + d2;
    if same_strict_sign(dv0, dv1, dv2) {
        return false;
    }

    // Plane of t1.
    let n1 = cross(sub(t1[1], t1[0]), sub(t1[2], t1[0]));
    let d1 = -dot(n1, t1[0]);
    let du0 = dot(n1, t2[0]) + d1;
    let du1 = dot(n1, t2[1]) + d1;
    let du2 = dot(n1, t2[2]) + d1;
    if same_strict_sign(du0, du1, du2) {
        return false;
    }

    // Coplanar when t2 lies entirely on t1's plane.
    if du0 == 0.0 && du1 == 0.0 && du2 == 0.0 {
        // Degenerate (zero-normal) triangles cannot define a coplanar overlap.
        if n1 == [0.0, 0.0, 0.0] {
            return false;
        }
        return coplanar_overlap(n1, t1, t2);
    }

    // Intersection-line direction; project onto its dominant axis.
    let dir = cross(n1, n2);
    let abs = [dir[0].abs(), dir[1].abs(), dir[2].abs()];
    let axis = if abs[0] >= abs[1] && abs[0] >= abs[2] {
        0
    } else if abs[1] >= abs[2] {
        1
    } else {
        2
    };
    let pv = [t1[0][axis], t1[1][axis], t1[2][axis]];
    let pu = [t2[0][axis], t2[1][axis], t2[2][axis]];

    let [lo1, hi1] = triangle_interval([dv0, dv1, dv2], pv);
    let [lo2, hi2] = triangle_interval([du0, du1, du2], pu);
    // Intervals overlap (inclusive) iff the triangles cross on the line.
    hi1 >= lo2 && hi2 >= lo1
}

/// Detects all self-intersecting triangle pairs in a mesh.
///
/// Brute-forces [`triangles_intersect`] over every unordered face pair, skipping
/// pairs that share at least one vertex index (topological neighbours that only
/// touch). Returns the colliding `(face_a, face_b)` pairs with `face_a < face_b`,
/// in ascending order.
pub fn mesh_self_intersections(mesh: &TriangleMesh) -> Vec<(u32, u32)> {
    let positions = mesh.positions();
    let indices = mesh.indices();
    let vertex = |i: u32| -> [f64; 3] {
        let v = positions[i as usize];
        [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
    };
    let tri = |t: [u32; 3]| -> [[f64; 3]; 3] { [vertex(t[0]), vertex(t[1]), vertex(t[2])] };

    let shares_vertex = |a: [u32; 3], b: [u32; 3]| -> bool {
        a.iter().any(|x| b.contains(x))
    };

    let mut hits = Vec::new();
    for (i, &face_i) in indices.iter().enumerate() {
        let ti = tri(face_i);
        for (offset, &face_j) in indices[i + 1..].iter().enumerate() {
            if shares_vertex(face_i, face_j) {
                continue;
            }
            if triangles_intersect(ti, tri(face_j)) {
                let j = i + 1 + offset;
                hits.push((i as u32, j as u32));
            }
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::triangle_mesh::TriangleMesh;

    /// Builds an index-only triangle mesh from positions (no normals/uvs).
    fn mesh(positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> TriangleMesh {
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    #[test]
    fn separated_triangles_do_not_intersect() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]];
        assert!(!triangles_intersect(a, b));
    }

    #[test]
    fn piercing_triangles_intersect() {
        // Triangle A in the z=0 plane; triangle B stands vertically and stabs
        // through A's interior.
        let a = [[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]];
        let b = [[0.3, 0.3, -1.0], [0.3, 0.3, 1.0], [0.6, 0.6, 1.0]];
        assert!(triangles_intersect(a, b));
    }

    #[test]
    fn coplanar_overlapping_triangles_intersect() {
        let a = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let b = [[0.5, 0.5, 0.0], [2.5, 0.5, 0.0], [0.5, 2.5, 0.0]];
        assert!(triangles_intersect(a, b));
    }

    #[test]
    fn coplanar_disjoint_triangles_do_not_intersect() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [5.0, 6.0, 0.0]];
        assert!(!triangles_intersect(a, b));
    }

    #[test]
    fn parallel_offset_triangles_do_not_intersect() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]];
        assert!(!triangles_intersect(a, b));
    }

    #[test]
    fn clean_mesh_reports_no_self_intersections() {
        // Two triangles forming a flat quad (share an edge, only touch).
        let m = mesh(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        assert!(mesh_self_intersections(&m).is_empty());
    }

    #[test]
    fn crossing_faces_are_detected() {
        // Face 0 flat in z=0; face 1 vertical, piercing face 0, sharing no
        // vertices with it.
        let m = mesh(
            vec![
                [-1.0, -1.0, 0.0],
                [3.0, -1.0, 0.0],
                [-1.0, 3.0, 0.0],
                [0.3, 0.3, -1.0],
                [0.3, 0.3, 1.0],
                [0.6, 0.6, 1.0],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );
        let hits = mesh_self_intersections(&m);
        assert_eq!(hits, vec![(0, 1)]);
    }

    #[test]
    fn vertex_sharing_faces_are_skipped() {
        // Two triangles sharing vertex 0; they only touch at that vertex and
        // are topological neighbours, so they must not be reported.
        let m = mesh(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            vec![[0, 1, 2], [0, 3, 4]],
        );
        assert!(mesh_self_intersections(&m).is_empty());
    }
}
