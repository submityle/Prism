//! Per-vertex tangent-frame generation for the `CPU` golden path.
//!
//! Normal mapping, parallax, and anisotropic shading all need a per-vertex
//! **tangent frame** `(T, B, N)` consistent with the mesh's `UV` layout: `T`
//! points along increasing `u`, `B` along increasing `v`, and `N` is the shading
//! normal. This module computes that frame directly from a
//! [`TriangleMesh`]'s positions, normals, and texture coordinates using the
//! standard least-squares accumulation (Lengyel's method, as popularized by the
//! `MikkTSpace` convention):
//!
//! 1. For each triangle, solve the 2×2 `UV` system to recover the surface
//!    tangent and bitangent directions implied by its position and `UV` edges.
//! 2. Accumulate those directions into every incident vertex (area-weighted by
//!    the un-normalized edge products, so larger triangles count more).
//! 3. Per vertex, Gram-Schmidt the accumulated tangent against the shading
//!    normal and record a handedness sign `w ∈ {-1, +1}` so the bitangent is
//!    reconstructed in-shader as `w · (N × T)`.
//!
//! The result is a `Vec<[f32; 4]>` of `xyzw` tangents aligned with the vertex
//! pool, matching the ubiquitous glTF / `MikkTSpace` packing. Degenerate `UV`
//! triangles are skipped, and a vertex that still has no usable tangent (an
//! isolated or fully degenerate vertex) falls back to an arbitrary basis
//! orthogonal to its normal so downstream code never divides by zero.
//!
//! All math is linear plus a single `sqrt` per normalization, honouring the
//! golden-path ban on `f32` transcendental functions.

use super::triangle_mesh::TriangleMesh;

/// Why tangent generation could not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentError {
    /// The mesh carries no per-vertex shading normals.
    MissingNormals,
    /// The mesh carries no per-vertex texture coordinates.
    MissingUvs,
}

impl core::fmt::Display for TangentError {
    /// Formats a human-readable reason.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TangentError::MissingNormals => {
                f.write_str("tangent generation requires per-vertex normals")
            }
            TangentError::MissingUvs => {
                f.write_str("tangent generation requires per-vertex texture coordinates")
            }
        }
    }
}

impl std::error::Error for TangentError {}

/// Computes `MikkTSpace`-style per-vertex tangents for `mesh`.
///
/// Returns one `[tx, ty, tz, w]` per vertex (aligned with
/// [`TriangleMesh::positions`]): `(tx, ty, tz)` is the unit tangent and `w` the
/// handedness so the bitangent is `w · (N × T)`.
///
/// # Errors
///
/// Returns [`TangentError::MissingNormals`] or [`TangentError::MissingUvs`]
/// when the required attribute pool is absent.
pub fn compute_tangents(mesh: &TriangleMesh) -> Result<Vec<[f32; 4]>, TangentError> {
    if !mesh.has_normals() {
        return Err(TangentError::MissingNormals);
    }
    if !mesh.has_uvs() {
        return Err(TangentError::MissingUvs);
    }

    let positions = mesh.positions();
    let normals = mesh.normals();
    let uvs = mesh.uvs();
    let count = positions.len();

    let mut tan = vec![[0.0f32; 3]; count];
    let mut bitan = vec![[0.0f32; 3]; count];

    for tri in mesh.indices() {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let w0 = uvs[i0];
        let w1 = uvs[i1];
        let w2 = uvs[i2];

        let e1 = sub(p1, p0);
        let e2 = sub(p2, p0);
        let du1 = w1[0] - w0[0];
        let dv1 = w1[1] - w0[1];
        let du2 = w2[0] - w0[0];
        let dv2 = w2[1] - w0[1];

        // Determinant of the UV edge matrix; skip degenerate (collinear) UVs.
        let det = du1 * dv2 - du2 * dv1;
        if det.abs() <= 1.0e-20 {
            continue;
        }
        let r = 1.0 / det;

        // Un-normalized tangent/bitangent directions. Leaving them unscaled by
        // triangle size makes the accumulation naturally area-weighted.
        let t = [
            (e1[0] * dv2 - e2[0] * dv1) * r,
            (e1[1] * dv2 - e2[1] * dv1) * r,
            (e1[2] * dv2 - e2[2] * dv1) * r,
        ];
        let b = [
            (e2[0] * du1 - e1[0] * du2) * r,
            (e2[1] * du1 - e1[1] * du2) * r,
            (e2[2] * du1 - e1[2] * du2) * r,
        ];

        for &i in &[i0, i1, i2] {
            tan[i] = add(tan[i], t);
            bitan[i] = add(bitan[i], b);
        }
    }

    let mut out = Vec::with_capacity(count);
    for v in 0..count {
        let n = normalize_or(normals[v], [0.0, 0.0, 1.0]);
        let acc = tan[v];

        // Gram-Schmidt: remove the normal component from the accumulated
        // tangent, then normalize.
        let ndt = dot(n, acc);
        let ortho = [
            acc[0] - n[0] * ndt,
            acc[1] - n[1] * ndt,
            acc[2] - n[2] * ndt,
        ];
        let t = if dot(ortho, ortho) > 1.0e-16 {
            normalize_or(ortho, fallback_tangent(n))
        } else {
            fallback_tangent(n)
        };

        // Handedness: +1 when the accumulated bitangent agrees with N × T.
        let w = if dot(cross(n, t), bitan[v]) < 0.0 {
            -1.0
        } else {
            1.0
        };
        out.push([t[0], t[1], t[2], w]);
    }
    Ok(out)
}

/// Returns a deterministic unit tangent orthogonal to `n`.
///
/// Picks the world axis least aligned with `n` and projects it onto the plane,
/// so the basis is stable and never collapses regardless of `n`'s direction.
fn fallback_tangent(n: [f32; 3]) -> [f32; 3] {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    let axis = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let d = dot(axis, n);
    let ortho = [axis[0] - n[0] * d, axis[1] - n[1] * d, axis[2] - n[2] * d];
    normalize_or(ortho, [1.0, 0.0, 0.0])
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

    /// Builds a unit XY quad (two triangles) with +Z normals and a matching
    /// `UV` layout: `u` → +X, `v` → +Y.
    fn xy_quad() -> TriangleMesh {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        TriangleMesh::new(positions, normals, uvs, indices).expect("valid quad")
    }

    /// Approximate vector equality.
    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5
    }

    #[test]
    fn missing_normals_is_rejected() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mesh = TriangleMesh::new(positions, vec![], uvs, vec![[0, 1, 2]]).expect("mesh");
        assert_eq!(compute_tangents(&mesh), Err(TangentError::MissingNormals));
    }

    #[test]
    fn missing_uvs_is_rejected() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let mesh = TriangleMesh::new(positions, normals, vec![], vec![[0, 1, 2]]).expect("mesh");
        assert_eq!(compute_tangents(&mesh), Err(TangentError::MissingUvs));
    }

    #[test]
    fn axis_aligned_quad_tangent_is_plus_x() {
        let mesh = xy_quad();
        let tangents = compute_tangents(&mesh).expect("tangents");
        assert_eq!(tangents.len(), 4);
        for t in &tangents {
            assert!(
                close([t[0], t[1], t[2]], [1.0, 0.0, 0.0]),
                "expected +X tangent, got {t:?}"
            );
        }
    }

    #[test]
    fn tangents_are_unit_length() {
        let mesh = xy_quad();
        for t in compute_tangents(&mesh).expect("tangents") {
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "tangent not unit: {len}");
        }
    }

    #[test]
    fn tangents_are_orthogonal_to_normals() {
        let mesh = xy_quad();
        let normals = mesh.normals().to_vec();
        for (t, n) in compute_tangents(&mesh).expect("tangents").iter().zip(normals) {
            let d = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
            assert!(d.abs() < 1e-5, "tangent not orthogonal to normal: {d}");
        }
    }

    #[test]
    fn handedness_is_unit_signed() {
        let mesh = xy_quad();
        for t in compute_tangents(&mesh).expect("tangents") {
            assert!(t[3] == 1.0 || t[3] == -1.0, "handedness must be ±1: {}", t[3]);
        }
    }

    #[test]
    fn right_handed_uv_gives_positive_w() {
        // With u→+X, v→+Y, N=+Z, the bitangent N×T = +Z × +X = +Y agrees with
        // increasing v, so handedness is +1.
        let mesh = xy_quad();
        for t in compute_tangents(&mesh).expect("tangents") {
            assert_eq!(t[3], 1.0);
        }
    }

    #[test]
    fn mirrored_uv_flips_handedness() {
        // Flip the v axis of the UVs so the bitangent opposes N×T → w = -1.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let mesh = TriangleMesh::new(positions, normals, uvs, indices).expect("mesh");
        for t in compute_tangents(&mesh).expect("tangents") {
            assert_eq!(t[3], -1.0, "mirrored UV should flip handedness");
        }
    }

    #[test]
    fn degenerate_uv_triangle_is_skipped() {
        // All three UVs coincide: det = 0, so the triangle contributes nothing
        // and the fallback basis (orthogonal to the normal) is returned.
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let uvs = vec![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let mesh = TriangleMesh::new(positions, normals, uvs, vec![[0, 1, 2]]).expect("mesh");
        let tangents = compute_tangents(&mesh).expect("tangents");
        for t in &tangents {
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "fallback tangent must be unit");
            let d = t[0] * 0.0 + t[1] * 0.0 + t[2] * 1.0;
            assert!(d.abs() < 1e-5, "fallback tangent must be orthogonal to normal");
        }
    }

    #[test]
    fn rotated_uv_rotates_tangent() {
        // Swap u and v so the tangent now follows +Y instead of +X.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        // u increases with world +Y, v increases with world +X.
        let uvs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let mesh = TriangleMesh::new(positions, normals, uvs, indices).expect("mesh");
        for t in compute_tangents(&mesh).expect("tangents") {
            assert!(
                close([t[0], t[1], t[2]], [0.0, 1.0, 0.0]),
                "expected +Y tangent, got {t:?}"
            );
        }
    }

    #[test]
    fn tangent_count_matches_vertices() {
        let mesh = xy_quad();
        assert_eq!(
            compute_tangents(&mesh).expect("tangents").len(),
            mesh.vertex_count()
        );
    }
}
