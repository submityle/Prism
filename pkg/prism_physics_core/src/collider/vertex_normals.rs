//! Robust triangle-mesh normals: per-face unit normals and angle-weighted
//! per-vertex normals.
//!
//! Vertex normals drive signed-distance sign recovery, contact-normal
//! smoothing and shading. The naive "average of incident face normals" is
//! mesh-density dependent: a vertex touched by many small slivers on one side
//! is dragged toward that side regardless of geometry. *Angle weighting*
//! (Thuermer & Wuethrich, 1998) weights each incident face by the interior
//! angle it subtends at the vertex, which is invariant to tessellation and is
//! the standard robust choice for geometry processing.
//!
//! Everything here is pure triangle-soup geometry with no coupling to the
//! collision pipeline, and nothing is derived from Unreal Engine source.

use glam::Vec3;

/// Shortest edge length below which a triangle is treated as degenerate and
/// contributes no normal.
const DEGENERATE_EPSILON: f32 = 1.0e-12;

/// Computes a unit face normal for every triangle.
///
/// Returns `None` when `vertices` or `indices` is empty. Triangles that
/// reference out-of-range vertices or that are degenerate (zero-area) yield a
/// zero vector in the corresponding slot, so the output length always matches
/// `indices.len()`. The winding is right-handed: `(b - a) x (c - a)`.
#[must_use]
pub fn face_normals(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<Vec<Vec3>> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let n = vertices.len();
    let mut normals = Vec::with_capacity(indices.len());
    for tri in indices {
        normals.push(face_normal(vertices, n, *tri).unwrap_or(Vec3::ZERO));
    }
    Some(normals)
}

/// Computes angle-weighted unit vertex normals.
///
/// Each incident triangle contributes its face normal scaled by the interior
/// angle it subtends at the shared vertex, which makes the result invariant to
/// tessellation density. Returns `None` when `vertices` or `indices` is empty.
/// Vertices with no incident (non-degenerate) triangle, and any vertex whose
/// accumulated normal cancels to zero, receive a zero vector. The output has
/// one entry per input vertex.
#[must_use]
pub fn vertex_normals(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<Vec<Vec3>> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let n = vertices.len();
    let mut accum = vec![Vec3::ZERO; n];
    for tri in indices {
        let Some(face) = face_normal(vertices, n, *tri) else {
            continue;
        };
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let p0 = vertices[i0];
        let p1 = vertices[i1];
        let p2 = vertices[i2];
        accum[i0] += face * corner_angle(p0, p1, p2);
        accum[i1] += face * corner_angle(p1, p2, p0);
        accum[i2] += face * corner_angle(p2, p0, p1);
    }
    for v in &mut accum {
        *v = normalize_or_zero(*v);
    }
    Some(accum)
}

/// Right-handed unit normal of one triangle, or `None` when it references an
/// out-of-range vertex or is degenerate.
fn face_normal(vertices: &[Vec3], n: usize, tri: [u32; 3]) -> Option<Vec3> {
    let i0 = tri[0] as usize;
    let i1 = tri[1] as usize;
    let i2 = tri[2] as usize;
    if i0 >= n || i1 >= n || i2 >= n {
        return None;
    }
    let a = vertices[i0];
    let b = vertices[i1];
    let c = vertices[i2];
    let normal = (b - a).cross(c - a);
    if normal.length_squared() <= DEGENERATE_EPSILON {
        return None;
    }
    Some(normal.normalize())
}

/// Interior angle (radians) at corner `apex` of the triangle `(apex, u, v)`.
fn corner_angle(apex: Vec3, u: Vec3, v: Vec3) -> f32 {
    let e0 = u - apex;
    let e1 = v - apex;
    let l0 = e0.length();
    let l1 = e1.length();
    if l0 <= DEGENERATE_EPSILON || l1 <= DEGENERATE_EPSILON {
        return 0.0;
    }
    let cos = (e0.dot(e1) / (l0 * l1)).clamp(-1.0, 1.0);
    // f64 acos: the f32 trig ops are disallowed in this workspace for
    // libm determinism, and the extra precision is harmless here.
    f64::from(cos).acos() as f32
}

/// Normalizes `v`, returning the zero vector when it is too short to have a
/// stable direction.
fn normalize_or_zero(v: Vec3) -> Vec3 {
    if v.length_squared() <= DEGENERATE_EPSILON {
        Vec3::ZERO
    } else {
        v.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin, outward wound, as a triangle soup
    /// with shared vertices.
    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let f = vec![
            // -z
            [0, 2, 1],
            [0, 3, 2],
            // +z
            [4, 5, 6],
            [4, 6, 7],
            // -y
            [0, 1, 5],
            [0, 5, 4],
            // +y
            [3, 7, 6],
            [3, 6, 2],
            // -x
            [0, 4, 7],
            [0, 7, 3],
            // +x
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    #[test]
    fn empty_input_is_rejected() {
        assert!(face_normals(&[], &[[0, 1, 2]]).is_none());
        assert!(face_normals(&[Vec3::ZERO], &[]).is_none());
        assert!(vertex_normals(&[], &[[0, 1, 2]]).is_none());
        assert!(vertex_normals(&[Vec3::ZERO], &[]).is_none());
    }

    #[test]
    fn single_triangle_face_normal_is_right_handed() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let normals = face_normals(&verts, &[[0, 1, 2]]).unwrap();
        assert_eq!(normals.len(), 1);
        assert!((normals[0] - Vec3::Z).length() < 1.0e-6);
    }

    #[test]
    fn degenerate_triangle_yields_zero_face_normal() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let normals = face_normals(&verts, &[[0, 1, 2]]).unwrap();
        assert_eq!(normals[0], Vec3::ZERO);
    }

    #[test]
    fn out_of_range_triangle_yields_zero_face_normal() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let normals = face_normals(&verts, &[[0, 1, 9]]).unwrap();
        assert_eq!(normals.len(), 1);
        assert_eq!(normals[0], Vec3::ZERO);
    }

    #[test]
    fn cube_corner_normals_point_diagonally_outward() {
        let (verts, faces) = unit_cube();
        let normals = vertex_normals(&verts, &faces).unwrap();
        assert_eq!(normals.len(), verts.len());
        for (i, &p) in verts.iter().enumerate() {
            // Each cube corner's outward direction is its position normalized.
            let expected = p.normalize();
            assert!(
                (normals[i] - expected).length() < 1.0e-5,
                "vertex {i}: got {:?} expected {expected:?}",
                normals[i]
            );
        }
    }

    #[test]
    fn vertex_normals_are_unit_or_zero() {
        let (verts, faces) = unit_cube();
        let normals = vertex_normals(&verts, &faces).unwrap();
        for nrm in normals {
            let len = nrm.length();
            assert!(len == 0.0 || (len - 1.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn isolated_vertex_gets_zero_normal() {
        // Three used vertices plus one unreferenced vertex.
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(5.0, 5.0, 5.0)];
        let normals = vertex_normals(&verts, &[[0, 1, 2]]).unwrap();
        assert_eq!(normals.len(), 4);
        assert_eq!(normals[3], Vec3::ZERO);
        assert!((normals[0] - Vec3::Z).length() < 1.0e-6);
    }

    #[test]
    fn angle_weighting_is_tessellation_invariant() {
        // A flat fan around the origin: splitting one wedge into two coplanar
        // triangles must not change the apex normal (still +Z), because angle
        // weighting depends on subtended angle, not triangle count.
        let apex = Vec3::ZERO;
        let ring = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
        ];
        let mid = Vec3::new(0.5, 0.5, 0.0).normalize();

        let verts_coarse = vec![apex, ring[0], ring[1], ring[2]];
        let coarse = vertex_normals(&verts_coarse, &[[0, 1, 2], [0, 2, 3]]).unwrap();

        let verts_fine = vec![apex, ring[0], mid, ring[1], ring[2]];
        let fine = vertex_normals(&verts_fine, &[[0, 1, 2], [0, 2, 3], [0, 3, 4]]).unwrap();

        assert!((coarse[0] - Vec3::Z).length() < 1.0e-6);
        assert!((fine[0] - Vec3::Z).length() < 1.0e-6);
        assert!((coarse[0] - fine[0]).length() < 1.0e-6);
    }
}
