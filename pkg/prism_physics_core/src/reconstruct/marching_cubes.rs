//! Iso-surface extraction from a [`ScalarField`] via marching tetrahedra.
//!
//! Every grid cell is split into six tetrahedra (the Freudenthal–Kuhn
//! decomposition, see [`crate::reconstruct::tables`]) and each tetrahedron is
//! marched independently. Emitted vertices are deduplicated by the *edge* they
//! lie on — identified by the ordered pair of global node indices at the edge
//! endpoints — so tetrahedra that share a face (within a cell or across
//! neighbouring cells) reuse the same vertex. Because the Kuhn tiling is a
//! single consistent simplicial complex, the result is a watertight,
//! edge-manifold triangle mesh: every interior edge is shared by exactly two
//! triangles.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! marching-tetrahedra extraction follows Doi & Koide 1991 and Bourke,
//! "Polygonising a scalar field using tetrahedrons"; vertex interpolation is
//! the standard linear edge crossing of Lorensen & Cline 1987.

use std::collections::HashMap;

use glam::Vec3;

use crate::math::scalar::Real;
use crate::reconstruct::field::ScalarField;
use crate::reconstruct::mesh::SurfaceMesh;
use crate::reconstruct::tables::{CUBE_CORNERS, CUBE_TETRAHEDRA, TET_EDGES, TET_TRIANGLES};

/// Extracts the iso-surface of `field` at the level `iso` as an indexed
/// triangle mesh.
///
/// A node is treated as *inside* the surface when its value is strictly
/// greater than `iso`. Vertices are placed by linear interpolation along the
/// tetrahedron edges that cross the level set, and their normals are derived
/// from the (outward) field gradient. An empty field yields an empty mesh.
#[must_use]
pub fn triangulate(field: &ScalarField, iso: Real) -> SurfaceMesh {
    let mut mesh = SurfaceMesh::new();
    if field.is_empty() || field.nx() < 2 || field.ny() < 2 || field.nz() < 2 {
        return mesh;
    }

    // Maps an edge (ordered pair of global node linear indices) to the emitted
    // vertex index, so shared edges reuse one vertex and the mesh stays
    // watertight.
    let mut vertex_cache: HashMap<(u32, u32), u32> = HashMap::new();

    for k in 0..field.nz() - 1 {
        for j in 0..field.ny() - 1 {
            for i in 0..field.nx() - 1 {
                march_cell(field, iso, (i, j, k), &mut vertex_cache, &mut mesh);
            }
        }
    }

    mesh
}

/// The global node coordinate of corner `corner` of cell `(i, j, k)`.
#[inline]
fn corner_node(cell: (usize, usize, usize), corner: usize) -> (usize, usize, usize) {
    let off = CUBE_CORNERS[corner];
    (
        cell.0 + off[0] as usize,
        cell.1 + off[1] as usize,
        cell.2 + off[2] as usize,
    )
}

/// Marches every tetrahedron of one cell, appending geometry to `mesh`.
fn march_cell(
    field: &ScalarField,
    iso: Real,
    cell: (usize, usize, usize),
    cache: &mut HashMap<(u32, u32), u32>,
    mesh: &mut SurfaceMesh,
) {
    for tet in &CUBE_TETRAHEDRA {
        // The four global nodes and field values of this tetrahedron.
        let mut nodes = [(0usize, 0usize, 0usize); 4];
        let mut vals = [0.0 as Real; 4];
        let mut mask = 0u8;
        for (local, &corner) in tet.iter().enumerate() {
            let n = corner_node(cell, corner);
            nodes[local] = n;
            let v = field.value(n.0, n.1, n.2);
            vals[local] = v;
            if v > iso {
                mask |= 1 << local;
            }
        }

        let tris = &TET_TRIANGLES[mask as usize];
        let mut t = 0;
        while t < tris.len() && tris[t] >= 0 {
            let e0 = tris[t] as usize;
            let e1 = tris[t + 1] as usize;
            let e2 = tris[t + 2] as usize;
            let i0 = edge_vertex(field, iso, &nodes, &vals, e0, cache, mesh);
            let i1 = edge_vertex(field, iso, &nodes, &vals, e1, cache, mesh);
            let i2 = edge_vertex(field, iso, &nodes, &vals, e2, cache, mesh);
            mesh.indices.push(i0);
            mesh.indices.push(i1);
            mesh.indices.push(i2);
            t += 3;
        }
    }
}

/// Returns the deduplicated mesh vertex index for the crossing on tetrahedron
/// edge `edge`, creating and caching a new vertex when first seen.
fn edge_vertex(
    field: &ScalarField,
    iso: Real,
    nodes: &[(usize, usize, usize); 4],
    vals: &[Real; 4],
    edge: usize,
    cache: &mut HashMap<(u32, u32), u32>,
    mesh: &mut SurfaceMesh,
) -> u32 {
    let a = TET_EDGES[edge][0];
    let b = TET_EDGES[edge][1];
    let na = nodes[a];
    let nb = nodes[b];
    let ga = field.node_idx(na.0, na.1, na.2) as u32;
    let gb = field.node_idx(nb.0, nb.1, nb.2) as u32;
    let key = if ga <= gb { (ga, gb) } else { (gb, ga) };

    if let Some(&idx) = cache.get(&key) {
        return idx;
    }

    let va = vals[a];
    let vb = vals[b];
    let denom = vb - va;
    // Guard against a degenerate (zero-length) crossing; fall back to the edge
    // midpoint when both endpoints share the iso value.
    let t = if denom.abs() > Real::EPSILON {
        ((iso - va) / denom).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let pa = field.node_position(na.0, na.1, na.2);
    let pb = field.node_position(nb.0, nb.1, nb.2);
    let position = pa + (pb - pa) * t;

    // Outward normal: the field decreases away from the particles, so the
    // negated gradient points outward. Fall back to a stable up-vector when the
    // gradient is degenerate.
    let grad = field.gradient(position);
    let normal = {
        let n = -grad;
        let len = n.length();
        if len > Real::EPSILON {
            n / len
        } else {
            Vec3::Y
        }
    };

    let idx = mesh.positions.len() as u32;
    mesh.positions.push(position);
    mesh.normals.push(normal);
    cache.insert(key, idx);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an analytic sphere field: value = radius − |p − centre| sampled
    /// on a regular grid, so the `iso = 0` surface is a sphere.
    fn sphere_field(centre: Vec3, radius: Real, dx: Real, n: usize, origin: Vec3) -> ScalarField {
        let mut f = ScalarField::zeros(n, n, n, dx, origin);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = f.node_position(i, j, k);
                    f.add_value(i, j, k, radius - (p - centre).length());
                }
            }
        }
        f
    }

    /// Counts how many triangles reference each undirected edge.
    fn edge_share_counts(mesh: &SurfaceMesh) -> HashMap<(u32, u32), u32> {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in mesh.indices.chunks_exact(3) {
            let e = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
            for &(a, b) in &e {
                let key = if a <= b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    #[test]
    fn empty_field_gives_empty_mesh() {
        let f = ScalarField::default();
        let m = triangulate(&f, 0.5);
        assert!(m.is_empty());
        assert_eq!(m.triangle_count(), 0);
    }

    #[test]
    fn sphere_reconstructs_closed_manifold() {
        let dx = 0.1;
        let n = 20;
        let origin = Vec3::splat(-1.0);
        let centre = Vec3::ZERO;
        let radius = 0.6;
        let f = sphere_field(centre, radius, dx, n, origin);
        let mesh = triangulate(&f, 0.0);

        assert!(mesh.triangle_count() > 0, "expected a non-empty surface");
        assert!(mesh.indices_are_valid(), "indices must be in range");

        // Watertight & edge-manifold: every edge shared by exactly two tris.
        let counts = edge_share_counts(&mesh);
        for (&(a, b), &c) in &counts {
            assert_eq!(c, 2, "edge ({a},{b}) shared by {c} triangles, expected 2");
        }

        // Normals are unit length.
        for nrm in &mesh.normals {
            assert!(
                (nrm.length() - 1.0).abs() < 1e-3,
                "normal not unit: {nrm:?}"
            );
        }
    }

    #[test]
    fn sphere_vertices_lie_near_the_isosurface() {
        let dx = 0.08;
        let n = 26;
        let origin = Vec3::splat(-1.0);
        let centre = Vec3::ZERO;
        let radius = 0.6;
        let f = sphere_field(centre, radius, dx, n, origin);
        let mesh = triangulate(&f, 0.0);
        assert!(mesh.triangle_count() > 0);

        // Every reconstructed vertex should sit close to the true sphere.
        let mut max_err = 0.0 as Real;
        for &p in &mesh.positions {
            let err = ((p - centre).length() - radius).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < dx, "max radial error {max_err} exceeds dx {dx}");
    }

    #[test]
    fn dense_particle_blob_is_manifold() {
        // A small dense cluster of particles skinned into a field.
        let mut pts = Vec::new();
        for k in 0..5 {
            for j in 0..5 {
                for i in 0..5 {
                    pts.push(Vec3::new(i as Real, j as Real, k as Real) * 0.05);
                }
            }
        }
        let field = ScalarField::from_particles(&pts, 0.12, 0.05, 2);
        let mesh = triangulate(&field, 0.5);
        assert!(mesh.triangle_count() > 0);
        assert!(mesh.indices_are_valid());
        let counts = edge_share_counts(&mesh);
        for &c in counts.values() {
            assert_eq!(c, 2, "non-manifold edge with share count {c}");
        }
    }
}
