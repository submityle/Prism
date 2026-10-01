//! Uniform midpoint (1-to-4) triangle subdivision for the `CPU` golden path.
//!
//! Many pipelines need to *refine* an existing triangle mesh — to add vertices
//! for per-vertex displacement, to raise shading-normal resolution, or to feed
//! a smoothing pass more degrees of freedom — without a parametric surface to
//! re-sample. The classic primitive-split operator does this: every triangle is
//! divided into four by inserting a new vertex at the midpoint of each of its
//! three edges and reconnecting them into a central triangle plus three corner
//! triangles.
//!
//! ```text
//!            v0                      v0
//!            /\                      /\
//!           /  \        ──▶        m2--m0
//!          /    \                  / \ / \
//!        v2------v1              v2--m1--v1
//! ```
//!
//! **Watertight by construction.** Each edge midpoint is keyed by its *sorted*
//! endpoint index pair, so the two triangles sharing an edge insert and reuse
//! the *same* midpoint vertex — no cracks, no T-junctions. Midpoint attributes
//! are the average of the edge endpoints: positions and `UV`s are averaged
//! directly, and normals are averaged then renormalized (falling back to `+Z`
//! if the two endpoint normals cancel). Applying `levels` iterations multiplies
//! the triangle count by `4^levels`.
//!
//! This is purely linear interpolation — no transcendental calls — so it stays
//! within the golden-path float policy. It performs *linear* refinement (new
//! vertices lie on the original faces); use it before a smoothing or
//! displacement pass when a curved limit surface is desired.

use std::collections::HashMap;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Maximum subdivision depth. Each level quadruples the triangle count, so the
/// cap keeps a single call from exhausting memory on a dense input.
pub const MAX_LEVELS: u32 = 7;

/// Subdivides `mesh` `levels` times with uniform 1-to-4 midpoint splitting,
/// sharing edge midpoints so the result stays watertight.
///
/// `levels` is clamped to [`MAX_LEVELS`]; `levels == 0` returns a clone of the
/// input. Normal and `UV` pools are refined only when the input carries them.
///
/// # Errors
///
/// Propagates [`TriangleMeshError`] from [`TriangleMesh::new`]; by construction
/// the generated pools are valid, so this does not fail in practice.
pub fn subdivide(mesh: &TriangleMesh, levels: u32) -> Result<TriangleMesh, TriangleMeshError> {
    let levels = levels.min(MAX_LEVELS);
    let mut current = mesh.clone();
    for _ in 0..levels {
        current = subdivide_once(&current)?;
    }
    Ok(current)
}

/// Performs a single 1-to-4 subdivision pass.
fn subdivide_once(mesh: &TriangleMesh) -> Result<TriangleMesh, TriangleMeshError> {
    let has_normals = mesh.has_normals();
    let has_uvs = mesh.has_uvs();

    let mut positions = mesh.positions().to_vec();
    let mut normals = if has_normals { mesh.normals().to_vec() } else { Vec::new() };
    let mut uvs = if has_uvs { mesh.uvs().to_vec() } else { Vec::new() };
    let mut indices: Vec<[u32; 3]> = Vec::with_capacity(mesh.indices().len() * 4);

    // Shared edge-midpoint cache keyed by the sorted endpoint index pair.
    let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();

    for tri in mesh.indices() {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        // Midpoints of edges AB, BC, CA (shared with the neighbour across each).
        let ab = edge_midpoint(a, b, &mut positions, &mut normals, &mut uvs, &mut midpoints, has_normals, has_uvs);
        let bc = edge_midpoint(b, c, &mut positions, &mut normals, &mut uvs, &mut midpoints, has_normals, has_uvs);
        let ca = edge_midpoint(c, a, &mut positions, &mut normals, &mut uvs, &mut midpoints, has_normals, has_uvs);
        // Four children: three corners plus the central triangle. Winding is
        // preserved so front faces stay front faces.
        indices.push([a, ab, ca]);
        indices.push([b, bc, ab]);
        indices.push([c, ca, bc]);
        indices.push([ab, bc, ca]);
    }

    TriangleMesh::new(positions, normals, uvs, indices)
}

/// Returns the shared midpoint index for edge `(i, j)`, creating it (and its
/// averaged attributes) the first time the edge is seen.
#[expect(
    clippy::too_many_arguments,
    reason = "threads the shared mutable attribute pools and midpoint cache through a single hot helper"
)]
fn edge_midpoint(
    i: u32,
    j: u32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    midpoints: &mut HashMap<(u32, u32), u32>,
    has_normals: bool,
    has_uvs: bool,
) -> u32 {
    let key = if i < j { (i, j) } else { (j, i) };
    if let Some(&existing) = midpoints.get(&key) {
        return existing;
    }
    let idx = positions.len() as u32;
    let pi = positions[i as usize];
    let pj = positions[j as usize];
    positions.push(midpoint3(pi, pj));
    if has_normals {
        let ni = normals[i as usize];
        let nj = normals[j as usize];
        normals.push(normalize_or(midpoint3(ni, nj), [0.0, 0.0, 1.0]));
    }
    if has_uvs {
        let ui = uvs[i as usize];
        let uj = uvs[j as usize];
        uvs.push([0.5 * (ui[0] + uj[0]), 0.5 * (ui[1] + uj[1])]);
    }
    midpoints.insert(key, idx);
    idx
}

/// Component-wise midpoint of two 3-vectors.
fn midpoint3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]
}

/// Normalizes `v`, returning `fallback` when `v` is too short to normalize.
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 > 1.0e-24 {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;

    /// A single triangle in the `z = 0` plane with no attributes.
    fn one_triangle() -> TriangleMesh {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        TriangleMesh::new(positions, vec![], vec![], vec![[0, 1, 2]]).expect("triangle")
    }

    /// Two triangles forming a shared-edge quad in the `z = 0` plane.
    fn quad() -> TriangleMesh {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        TriangleMesh::new(positions, vec![], vec![], indices).expect("quad")
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6 && (a[2] - b[2]).abs() < 1e-6
    }

    #[test]
    fn level_zero_is_a_clone() {
        let mesh = one_triangle();
        let out = subdivide(&mesh, 0).expect("subdivide");
        assert_eq!(out, mesh);
    }

    #[test]
    fn one_triangle_splits_into_four() {
        let out = subdivide(&one_triangle(), 1).expect("subdivide");
        assert_eq!(out.triangle_count(), 4);
        // Three original corners plus three edge midpoints.
        assert_eq!(out.vertex_count(), 6);
    }

    #[test]
    fn two_levels_quadruple_twice() {
        let out = subdivide(&one_triangle(), 2).expect("subdivide");
        assert_eq!(out.triangle_count(), 16);
    }

    #[test]
    fn quad_shares_diagonal_midpoint() {
        // 4 corners + midpoints of the 5 unique edges (AB,BC,CA shared diagonal,
        // CD,DA) = 9 vertices; 2 triangles become 8.
        let out = subdivide(&quad(), 1).expect("subdivide");
        assert_eq!(out.triangle_count(), 8);
        assert_eq!(out.vertex_count(), 9, "shared diagonal midpoint must be reused");
    }

    #[test]
    fn midpoint_positions_are_edge_averages() {
        let out = subdivide(&one_triangle(), 1).expect("subdivide");
        // The three new vertices (indices 3..6) must be the edge midpoints.
        let mids: Vec<[f32; 3]> = out.positions()[3..6].to_vec();
        let expected = [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0]];
        for e in &expected {
            assert!(mids.iter().any(|m| close(*m, *e)), "missing midpoint {e:?}");
        }
    }

    #[test]
    fn clamps_to_max_levels() {
        // Asking for more than MAX_LEVELS behaves like exactly MAX_LEVELS.
        let a = subdivide(&one_triangle(), MAX_LEVELS).expect("a");
        let b = subdivide(&one_triangle(), MAX_LEVELS + 5).expect("b");
        assert_eq!(a.triangle_count(), b.triangle_count());
    }

    #[test]
    fn normals_are_averaged_and_unit() {
        // Two differing normals on an edge average to a unit, interpolated one.
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let mesh = TriangleMesh::new(positions, normals, vec![], vec![[0, 1, 2]]).expect("mesh");
        let out = subdivide(&mesh, 1).expect("subdivide");
        assert!(out.has_normals());
        for n in out.normals() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "normal not unit: {n:?}");
        }
    }

    #[test]
    fn uvs_are_averaged() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mesh = TriangleMesh::new(positions, vec![], uvs, vec![[0, 1, 2]]).expect("mesh");
        let out = subdivide(&mesh, 1).expect("subdivide");
        assert!(out.has_uvs());
        // Midpoint of UV (0,0)-(1,0) must be (0.5, 0).
        assert!(out.uvs().iter().any(|uv| (uv[0] - 0.5).abs() < 1e-6 && uv[1].abs() < 1e-6));
    }

    #[test]
    fn empty_attribute_pools_stay_empty() {
        let out = subdivide(&one_triangle(), 2).expect("subdivide");
        assert!(!out.has_normals());
        assert!(!out.has_uvs());
    }

    #[test]
    fn subdivided_mesh_is_hittable() {
        // The refined flat quad still occupies the z=0 plane and is watertight.
        let out = subdivide(&quad(), 2).expect("subdivide");
        let bvh = crate::ray_scene::triangle_mesh::TriangleMeshBvh::build(out);
        let ray = Ray::infinite([0.53, 0.47, 10.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit the plane");
        assert!(close(hit.position, [0.53, 0.47, 0.0]));
    }
}
