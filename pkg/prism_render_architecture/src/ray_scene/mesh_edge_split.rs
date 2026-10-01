//! Adaptive edge-split refinement driven by a length budget, for the `CPU`
//! golden path.
//!
//! Uniform subdivision (see [`super::mesh_subdivision`]) quadruples *every*
//! triangle whether it needs it or not. Adaptive refinement instead inserts new
//! vertices only where triangles are too large — along edges that exceed a
//! target length — so detail concentrates where it matters (near the camera,
//! across a displacement feature) without flooding flat regions with triangles.
//!
//! ## Marked-edge templates
//!
//! The operator works in two phases. First every edge longer than the budget is
//! *marked* and given a shared midpoint vertex, keyed by its sorted endpoint
//! pair so the two triangles meeting on that edge insert the **same** vertex.
//! Second, each triangle is re-triangulated from a small template chosen by how
//! many of its three edges are marked:
//!
//! ```text
//!   0 marked → keep         1 marked → bisect (2)   3 marked → 1-to-4 (4)
//!        v2                       v2                      v2
//!        /\                       /\                      /\
//!       /  \                     /  \                   m2--m1
//!      /    \                   /    \                  / \ / \
//!    v0------v1               v0--m0--v1              v0--m0--v1
//!
//!   2 marked → split into a corner triangle plus a quad cut on its
//!   shorter internal diagonal (3 triangles).
//! ```
//!
//! Because marking is a property of the *edge*, both triangles sharing a marked
//! edge always split it, so the mesh stays watertight with no T-junctions. The
//! only per-triangle freedom — which diagonal cuts the 2-marked quad — is
//! entirely internal to that triangle and never affects a neighbour.
//!
//! Running several passes bisects an over-long edge repeatedly (`L → L/2 → …`),
//! converging geometrically toward the budget. Midpoint positions, normals, and
//! `UV`s are averaged (normals renormalized), matching the subdivision module.
//! The only non-arithmetic operations are the `sqrt` length comparisons, so the
//! module stays within the golden-path float policy.

use std::collections::HashMap;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Maximum number of refinement passes a single [`split_long_edges`] call will
/// run, bounding worst-case growth even for a tiny length budget.
pub const MAX_PASSES: u32 = 10;

/// Refines `mesh` by repeatedly splitting every edge longer than
/// `max_edge_length`, up to [`MAX_PASSES`] passes (also capped by `max_passes`).
///
/// Each pass marks over-long edges, inserts a shared midpoint per marked edge,
/// and re-triangulates every face from the marked-edge template, keeping the
/// mesh watertight. A non-positive or non-finite `max_edge_length`, a zero
/// `max_passes`, or a mesh with no over-long edge returns the input unchanged
/// (cloned). Midpoint positions/`UV`s are averaged and normals
/// averaged-then-renormalized.
///
/// # Errors
///
/// Propagates [`TriangleMeshError`] from rebuilding each pass; by construction
/// the generated pools are valid, so this does not fail in practice.
pub fn split_long_edges(
    mesh: &TriangleMesh,
    max_edge_length: f32,
    max_passes: u32,
) -> Result<TriangleMesh, TriangleMeshError> {
    if !max_edge_length.is_finite() || max_edge_length <= 0.0 || max_passes == 0 {
        return Ok(mesh.clone());
    }
    let budget_sq = max_edge_length * max_edge_length;
    let passes = max_passes.min(MAX_PASSES);

    let mut current = mesh.clone();
    for _ in 0..passes {
        let (next, split_any) = split_pass(&current, budget_sq)?;
        current = next;
        if !split_any {
            break;
        }
    }
    Ok(current)
}

/// Runs one marked-edge refinement pass. Returns the refined mesh and whether
/// any edge was split (so the caller can stop early once converged).
fn split_pass(
    mesh: &TriangleMesh,
    budget_sq: f32,
) -> Result<(TriangleMesh, bool), TriangleMeshError> {
    let has_normals = mesh.has_normals();
    let has_uvs = mesh.has_uvs();

    let mut positions = mesh.positions().to_vec();
    let mut normals = mesh.normals().to_vec();
    let mut uvs = mesh.uvs().to_vec();

    // Midpoint vertex per marked edge, keyed by the sorted endpoint pair.
    let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
    let mut split_any = false;

    // Pre-mark every over-long edge so neighbours agree on the split.
    for tri in mesh.indices() {
        let [a, b, c] = *tri;
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            if edge_len_sq(&positions, u, v) > budget_sq {
                let key = sorted_pair(u, v);
                if let std::collections::hash_map::Entry::Vacant(slot) = midpoints.entry(key) {
                    let new_index = positions.len() as u32;
                    push_midpoint(
                        &mut positions,
                        &mut normals,
                        &mut uvs,
                        has_normals,
                        has_uvs,
                        key.0,
                        key.1,
                    );
                    slot.insert(new_index);
                    split_any = true;
                }
            }
        }
    }

    if !split_any {
        return Ok((mesh.clone(), false));
    }

    let mut indices = Vec::with_capacity(mesh.indices().len());
    for tri in mesh.indices() {
        let [v0, v1, v2] = *tri;
        let mids = [
            midpoints.get(&sorted_pair(v0, v1)).copied(),
            midpoints.get(&sorted_pair(v1, v2)).copied(),
            midpoints.get(&sorted_pair(v2, v0)).copied(),
        ];
        split_triangle([v0, v1, v2], mids, &positions, &mut indices);
    }

    let mesh = TriangleMesh::new(positions, normals, uvs, indices)?;
    Ok((mesh, true))
}

/// Emits the refined triangles for one face given its three corner indices and
/// the optional midpoint of each edge (edge `i` joins corner `i` and corner
/// `(i + 1) % 3`), appending them to `out`.
fn split_triangle(
    corners: [u32; 3],
    mids: [Option<u32>; 3],
    positions: &[[f32; 3]],
    out: &mut Vec<[u32; 3]>,
) {
    let [v0, v1, v2] = corners;
    let marked = mids.iter().filter(|m| m.is_some()).count();
    match marked {
        0 => out.push([v0, v1, v2]),
        3 => {
            // 1-to-4, identical to uniform midpoint subdivision.
            let (m0, m1, m2) = (mids[0].unwrap(), mids[1].unwrap(), mids[2].unwrap());
            out.push([v0, m0, m2]);
            out.push([m0, v1, m1]);
            out.push([m2, m1, v2]);
            out.push([m0, m1, m2]);
        }
        1 => split_one(corners, mids, out),
        _ => split_two(corners, mids, positions, out),
    }
}

/// Handles the single-marked-edge case: bisect the triangle through the one
/// midpoint and the opposite corner.
fn split_one(corners: [u32; 3], mids: [Option<u32>; 3], out: &mut Vec<[u32; 3]>) {
    let [v0, v1, v2] = corners;
    if let Some(m) = mids[0] {
        // Edge (v0, v1) split; opposite corner v2.
        out.push([v0, m, v2]);
        out.push([m, v1, v2]);
    } else if let Some(m) = mids[1] {
        // Edge (v1, v2) split; opposite corner v0.
        out.push([v1, m, v0]);
        out.push([m, v2, v0]);
    } else if let Some(m) = mids[2] {
        // Edge (v2, v0) split; opposite corner v1.
        out.push([v2, m, v1]);
        out.push([m, v0, v1]);
    }
}

/// Handles the two-marked-edge case: a corner (apex) triangle plus a quad cut
/// along its shorter internal diagonal.
fn split_two(
    corners: [u32; 3],
    mids: [Option<u32>; 3],
    positions: &[[f32; 3]],
    out: &mut Vec<[u32; 3]>,
) {
    let [v0, v1, v2] = corners;
    // Identify the unmarked edge, then route to the apex layout for it.
    if mids[2].is_none() {
        // Edges 0 and 1 marked; apex v1; unmarked edge (v2, v0).
        let (m0, m1) = (mids[0].unwrap(), mids[1].unwrap());
        out.push([m0, v1, m1]);
        emit_quad(v0, m0, m1, v2, positions, out);
    } else if mids[0].is_none() {
        // Edges 1 and 2 marked; apex v2; unmarked edge (v0, v1).
        let (m1, m2) = (mids[1].unwrap(), mids[2].unwrap());
        out.push([m1, v2, m2]);
        emit_quad(v1, m1, m2, v0, positions, out);
    } else {
        // Edges 2 and 0 marked; apex v0; unmarked edge (v1, v2).
        let (m2, m0) = (mids[2].unwrap(), mids[0].unwrap());
        out.push([m2, v0, m0]);
        emit_quad(v2, m2, m0, v1, positions, out);
    }
}

/// Triangulates the counter-clockwise quad `[a, b, c, d]` along its shorter
/// diagonal, appending the two triangles to `out`.
fn emit_quad(
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    positions: &[[f32; 3]],
    out: &mut Vec<[u32; 3]>,
) {
    // Diagonal (a, c) vs (b, d); pick the shorter for better-shaped triangles.
    if edge_len_sq(positions, a, c) <= edge_len_sq(positions, b, d) {
        out.push([a, b, c]);
        out.push([a, c, d]);
    } else {
        out.push([a, b, d]);
        out.push([b, c, d]);
    }
}

/// Returns the sorted `(min, max)` endpoint pair used to key a shared edge.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Squared length of the edge between vertices `a` and `b`.
fn edge_len_sq(positions: &[[f32; 3]], a: u32, b: u32) -> f32 {
    let p = positions[a as usize];
    let q = positions[b as usize];
    let dx = q[0] - p[0];
    let dy = q[1] - p[1];
    let dz = q[2] - p[2];
    dx * dx + dy * dy + dz * dz
}

/// Appends the averaged midpoint of edge `(a, b)` to the attribute pools,
/// averaging positions and `UV`s and averaging-then-renormalizing normals.
fn push_midpoint(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    has_normals: bool,
    has_uvs: bool,
    a: u32,
    b: u32,
) {
    let pa = positions[a as usize];
    let pb = positions[b as usize];
    positions.push([
        0.5 * (pa[0] + pb[0]),
        0.5 * (pa[1] + pb[1]),
        0.5 * (pa[2] + pb[2]),
    ]);
    if has_normals {
        let na = normals[a as usize];
        let nb = normals[b as usize];
        let mut n = [na[0] + nb[0], na[1] + nb[1], na[2] + nb[2]];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            let inv = 1.0 / len;
            n = [n[0] * inv, n[1] * inv, n[2] * inv];
        } else {
            n = [0.0, 0.0, 1.0];
        }
        normals.push(n);
    }
    if has_uvs {
        let ta = uvs[a as usize];
        let tb = uvs[b as usize];
        uvs.push([0.5 * (ta[0] + tb[0]), 0.5 * (ta[1] + tb[1])]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;
    use crate::ray_scene::triangle_mesh::TriangleMeshBvh;

    /// One triangle in the z = 0 plane with legs of length 4.
    fn big_triangle() -> TriangleMesh {
        TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap()
    }

    /// A unit square (two triangles) sharing the diagonal (1, 2).
    fn unit_quad() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [1, 3, 2]],
        )
        .unwrap()
    }

    /// Longest edge length present in `mesh`.
    fn max_edge(mesh: &TriangleMesh) -> f32 {
        let mut m = 0.0f32;
        for tri in mesh.indices() {
            let [a, b, c] = *tri;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let len = edge_len_sq(mesh.positions(), u, v).sqrt();
                m = m.max(len);
            }
        }
        m
    }

    #[test]
    fn non_positive_budget_is_identity() {
        let mesh = big_triangle();
        let out = split_long_edges(&mesh, 0.0, 4).unwrap();
        assert_eq!(out.indices(), mesh.indices());
        let out = split_long_edges(&mesh, -1.0, 4).unwrap();
        assert_eq!(out.indices(), mesh.indices());
    }

    #[test]
    fn zero_passes_is_identity() {
        let mesh = big_triangle();
        let out = split_long_edges(&mesh, 0.1, 0).unwrap();
        assert_eq!(out.indices(), mesh.indices());
    }

    #[test]
    fn budget_above_all_edges_is_identity() {
        let mesh = big_triangle();
        let out = split_long_edges(&mesh, 100.0, 4).unwrap();
        assert_eq!(out.triangle_count(), 1);
    }

    #[test]
    fn single_pass_bisects_each_marked_edge() {
        // All three legs exceed 3.0, so the first pass is a full 1-to-4 split.
        let mesh = big_triangle();
        let out = split_long_edges(&mesh, 3.0, 1).unwrap();
        assert_eq!(out.triangle_count(), 4);
    }

    #[test]
    fn refinement_drives_edges_under_budget() {
        let mesh = big_triangle();
        let budget = 1.0;
        let out = split_long_edges(&mesh, budget, MAX_PASSES).unwrap();
        // Allow a tiny epsilon for f32 midpoint rounding.
        assert!(max_edge(&out) <= budget + 1.0e-4, "max {}", max_edge(&out));
        assert!(out.triangle_count() > 4);
    }

    #[test]
    fn shared_diagonal_stays_watertight() {
        // Split the quad hard, then confirm a ray still hits the refined,
        // crack-free surface at an off-seam interior point.
        let mesh = unit_quad();
        let out = split_long_edges(&mesh, 0.2, MAX_PASSES).unwrap();
        let bvh = TriangleMeshBvh::build(out);
        let ray = Ray::infinite([0.53, 0.47, 5.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray);
        assert!(hit.is_some(), "ray missed the refined quad");
        assert!(hit.unwrap().position[2].abs() < 1.0e-3);
    }

    #[test]
    fn refined_plane_stays_coplanar() {
        let mesh = unit_quad();
        let out = split_long_edges(&mesh, 0.3, MAX_PASSES).unwrap();
        for p in out.positions() {
            assert!(p[2].abs() < 1.0e-5);
        }
    }

    #[test]
    fn attributes_are_interpolated() {
        // Normals constant +Z, UVs equal to xy: midpoints must inherit them.
        let positions = vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let uvs = vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let mesh = TriangleMesh::new(positions, normals, uvs, vec![[0, 1, 2]]).unwrap();
        let out = split_long_edges(&mesh, 1.0, MAX_PASSES).unwrap();
        assert!(out.has_normals());
        assert!(out.has_uvs());
        assert_eq!(out.normals().len(), out.vertex_count());
        assert_eq!(out.uvs().len(), out.vertex_count());
        for n in out.normals() {
            assert!((n[2] - 1.0).abs() < 1.0e-5);
        }
        // UV == xy must hold for every (interpolated) vertex.
        for (p, t) in out.positions().iter().zip(out.uvs()) {
            assert!((p[0] - t[0]).abs() < 1.0e-4);
            assert!((p[1] - t[1]).abs() < 1.0e-4);
        }
    }

    #[test]
    fn two_marked_edges_make_three_triangles() {
        // A tall isosceles triangle: the short base stays under budget while
        // the two long sides are over it, so exactly the 2-marked template
        // fires (apex triangle plus a quad split into two → three triangles).
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 4.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        // Base = 1.0; the two sides = ~4.03. Budget 2.0 marks the sides only.
        let out = split_long_edges(&mesh, 2.0, 1).unwrap();
        assert_eq!(out.triangle_count(), 3);
    }
}
