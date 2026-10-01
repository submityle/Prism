//! Area-weighted smooth vertex-normal generation for the `CPU` golden path.
//!
//! Imported or procedurally welded [`TriangleMesh`]es often arrive without
//! shading normals, or with hard faceted ones. The standard fix is to
//! accumulate each triangle's geometric normal into its three vertices and
//! renormalize, giving a smooth per-vertex normal field. Weighting each
//! contribution by the triangle's **area** — which falls straight out of the
//! un-normalized cross product `‖e1 × e2‖ = 2·area` — makes large faces
//! dominate small slivers, matching the de-facto behaviour of `DCC` tools and
//! avoiding the bias that pure face-count averaging introduces near fans.
//!
//! Two entry points are provided: [`compute_smooth_normals`] returns the raw
//! `Vec<[f32; 3]>` aligned with the vertex pool, and
//! [`with_smooth_normals`] rebuilds the mesh with those normals installed
//! (preserving positions, `UV`s, and indices).
//!
//! Degenerate (zero-area) triangles contribute nothing; a vertex that remains
//! without any incident area falls back to a unit `+Z` normal so no downstream
//! normalization divides by zero. All math is linear plus a single `sqrt` per
//! normalization, honouring the golden-path ban on `f32` transcendental
//! functions.

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Computes area-weighted smooth per-vertex normals for `mesh`.
///
/// Returns one unit normal per vertex, aligned with
/// [`TriangleMesh::positions`]. The existing normal pool (if any) is ignored;
/// normals are derived purely from positions and connectivity.
#[must_use]
pub fn compute_smooth_normals(mesh: &TriangleMesh) -> Vec<[f32; 3]> {
    let positions = mesh.positions();
    let mut acc = vec![[0.0f32; 3]; positions.len()];

    for tri in mesh.indices() {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let e1 = sub(positions[i1], positions[i0]);
        let e2 = sub(positions[i2], positions[i0]);
        // Un-normalized face normal: its length is twice the triangle area, so
        // accumulating it directly yields the area weighting.
        let face = cross(e1, e2);
        acc[i0] = add(acc[i0], face);
        acc[i1] = add(acc[i1], face);
        acc[i2] = add(acc[i2], face);
    }

    acc.into_iter()
        .map(|n| normalize_or(n, [0.0, 0.0, 1.0]))
        .collect()
}

/// Rebuilds `mesh` with freshly computed area-weighted smooth normals,
/// preserving positions, `UV`s, and indices.
///
/// # Errors
///
/// Propagates [`TriangleMeshError`] from [`TriangleMesh::new`]; because the
/// position count and indices are reused verbatim and the normal count matches,
/// this does not fail in practice.
pub fn with_smooth_normals(mesh: &TriangleMesh) -> Result<TriangleMesh, TriangleMeshError> {
    let normals = compute_smooth_normals(mesh);
    TriangleMesh::new(
        mesh.positions().to_vec(),
        normals,
        mesh.uvs().to_vec(),
        mesh.indices().to_vec(),
    )
}

/// Normalizes `v`, returning `fallback` when `v` is too short to normalize.
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len2 = dot(v, v);
    if len2 > 1.0e-24 {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

/// Component-wise `a + b`.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Component-wise `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat XY quad (two triangles) with no normals supplied.
    fn flat_quad() -> TriangleMesh {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        TriangleMesh::new(positions, vec![], vec![], indices).expect("valid quad")
    }

    /// Approximate vector equality.
    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5
    }

    #[test]
    fn flat_quad_normals_point_up() {
        let normals = compute_smooth_normals(&flat_quad());
        assert_eq!(normals.len(), 4);
        for n in &normals {
            assert!(close(*n, [0.0, 0.0, 1.0]), "expected +Z, got {n:?}");
        }
    }

    #[test]
    fn normals_are_unit_length() {
        for n in compute_smooth_normals(&flat_quad()) {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal not unit: {len}");
        }
    }

    #[test]
    fn count_matches_vertices() {
        let mesh = flat_quad();
        assert_eq!(
            compute_smooth_normals(&mesh).len(),
            mesh.vertex_count()
        );
    }

    #[test]
    fn shared_edge_is_averaged() {
        // A ridge: two quads meeting along the x-axis at a 90° fold. The shared
        // edge vertices should average the two face normals (+Z and +Y) to a
        // unit 45° normal, while the outer vertices keep their single face
        // normal.
        let positions = vec![
            [0.0, 0.0, 0.0], // 0 shared edge
            [1.0, 0.0, 0.0], // 1 shared edge
            [1.0, 1.0, 0.0], // 2 top face (+Z)
            [0.0, 1.0, 0.0], // 3 top face (+Z)
            [1.0, 0.0, 1.0], // 4 side face (+Y)
            [0.0, 0.0, 1.0], // 5 side face (+Y)
        ];
        // Top quad in z=0 plane (normal +Z): 0,1,2,3
        // Side quad in y=0 plane (normal +Y): 0,1,4,5 wound so its normal is +Y
        let indices = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 4, 1],
            [0, 5, 4],
        ];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let normals = compute_smooth_normals(&mesh);
        // Shared-edge vertices 0 and 1 blend +Z and +Y → normalized (0, √½, √½).
        let s = core::f32::consts::FRAC_1_SQRT_2;
        assert!(close(normals[0], [0.0, s, s]), "vertex 0: {:?}", normals[0]);
        assert!(close(normals[1], [0.0, s, s]), "vertex 1: {:?}", normals[1]);
        // Outer top vertices keep +Z.
        assert!(close(normals[2], [0.0, 0.0, 1.0]), "vertex 2: {:?}", normals[2]);
        assert!(close(normals[3], [0.0, 0.0, 1.0]), "vertex 3: {:?}", normals[3]);
        // Outer side vertices keep +Y.
        assert!(close(normals[4], [0.0, 1.0, 0.0]), "vertex 4: {:?}", normals[4]);
    }

    #[test]
    fn larger_face_dominates_the_shared_vertex() {
        // Two triangles share vertex 0. The big triangle lies in z=0 (+Z normal)
        // and dwarfs the small one in y=0 (+Y normal), so the area weighting
        // pulls the shared normal toward +Z.
        let positions = vec![
            [0.0, 0.0, 0.0],  // 0 shared
            [10.0, 0.0, 0.0], // 1 big
            [0.0, 10.0, 0.0], // 2 big
            [0.1, 0.0, 0.1],  // 3 small
            [0.0, 0.0, 0.1],  // 4 small
        ];
        let indices = vec![[0, 1, 2], [0, 3, 4]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let n = compute_smooth_normals(&mesh)[0];
        assert!(n[2] > n[1], "big +Z face should dominate: {n:?}");
        assert!(n[2] > 0.9, "shared normal should lean strongly +Z: {n:?}");
    }

    #[test]
    fn degenerate_triangle_contributes_nothing() {
        // A zero-area (collinear) triangle plus one real triangle: the real one
        // alone determines the normals.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], // collinear with 0,1 → zero-area triangle
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [0, 1, 3]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let normals = compute_smooth_normals(&mesh);
        // Triangle 0,1,3 lies in z=0 with +Z normal; the collinear triangle adds
        // nothing, so every incident vertex is +Z (not NaN).
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal not unit: {n:?}");
        }
        assert!(close(normals[3], [0.0, 0.0, 1.0]), "vertex 3: {:?}", normals[3]);
    }

    #[test]
    fn isolated_vertex_falls_back() {
        // Vertex 3 is referenced by no triangle, so it has no incident area and
        // must take the +Z fallback rather than a zero/NaN normal.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0], // isolated
        ];
        let indices = vec![[0, 1, 2]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let normals = compute_smooth_normals(&mesh);
        assert!(close(normals[3], [0.0, 0.0, 1.0]), "isolated: {:?}", normals[3]);
    }

    #[test]
    fn with_smooth_normals_installs_them() {
        let mesh = flat_quad();
        let shaded = with_smooth_normals(&mesh).expect("rebuilt mesh");
        assert!(shaded.has_normals());
        assert_eq!(shaded.vertex_count(), mesh.vertex_count());
        assert_eq!(shaded.indices(), mesh.indices());
        for n in shaded.normals() {
            assert!(close(*n, [0.0, 0.0, 1.0]));
        }
    }

    #[test]
    fn with_smooth_normals_preserves_uvs() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mesh = TriangleMesh::new(positions, vec![], uvs.clone(), vec![[0, 1, 2]]).expect("mesh");
        let shaded = with_smooth_normals(&mesh).expect("rebuilt");
        assert_eq!(shaded.uvs(), uvs.as_slice());
    }
}
