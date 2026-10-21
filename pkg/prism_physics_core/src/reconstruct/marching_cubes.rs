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

/// Logical work/storage ceilings for one extraction. Allocator metadata and
/// spare vector/hash-table capacity are additional, bounded implementation costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceBudget {
    /// Maximum grid cells visited, including cells with no surface crossing.
    pub max_cells: usize,
    /// Maximum unique vertices (and edge-cache entries).
    pub max_vertices: usize,
    /// Maximum emitted triangles, each with three `u32` indices.
    pub max_triangles: usize,
}

/// A failed extraction returns no partial mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceBuildError {
    /// The isovalue, field values or reconstructed geometry are non-finite.
    InvalidField,
    /// The cell product cannot be represented, or exceeds the work ceiling.
    CellBudgetExceeded,
    /// Unique vertices exceed the ceiling or the `u32` index space.
    VertexBudgetExceeded,
    /// Triangles exceed the ceiling or the addressable index buffer length.
    TriangleBudgetExceeded,
    /// A vector or edge-cache allocation failed.
    AllocationFailed,
}

/// Extracts the iso-surface of `field` at the level `iso` as an indexed
/// triangle mesh.
///
/// A node is treated as *inside* the surface when its value is strictly
/// greater than `iso`. Vertices are placed by linear interpolation along the
/// tetrahedron edges that cross the level set, and their normals are derived
/// from the (outward) field gradient. An empty field yields an empty mesh.
#[must_use]
pub fn triangulate(field: &ScalarField, iso: Real) -> SurfaceMesh {
    triangulate_bounded(
        field,
        iso,
        SurfaceBudget {
            max_cells: usize::MAX,
            max_vertices: u32::MAX as usize,
            max_triangles: usize::MAX / 3,
        },
    )
    .unwrap_or_default()
}

/// Extracts a complete surface within explicit work and output budgets.
///
/// Cell work is checked before iteration; vertex/triangle limits and fallible
/// allocations are checked before appending. Budget rejection never publishes a
/// truncated or open mesh. Node identities use `usize`, independent of the mesh
/// index width, so large grids cannot alias edges through a narrowing cast.
pub fn triangulate_bounded(
    field: &ScalarField,
    iso: Real,
    budget: SurfaceBudget,
) -> Result<SurfaceMesh, SurfaceBuildError> {
    if !iso.is_finite() {
        return Err(SurfaceBuildError::InvalidField);
    }
    let mut mesh = SurfaceMesh::new();
    if field.is_empty() || field.nx() < 2 || field.ny() < 2 || field.nz() < 2 {
        return Ok(mesh);
    }
    let cells = (field.nx() - 1)
        .checked_mul(field.ny() - 1)
        .and_then(|n| n.checked_mul(field.nz() - 1))
        .ok_or(SurfaceBuildError::CellBudgetExceeded)?;
    if cells > budget.max_cells {
        return Err(SurfaceBuildError::CellBudgetExceeded);
    }
    if !field
        .node_position(field.nx() - 1, field.ny() - 1, field.nz() - 1)
        .is_finite()
    {
        return Err(SurfaceBuildError::InvalidField);
    }

    // Maps an edge (ordered pair of global node linear indices) to the emitted
    // vertex index, so shared edges reuse one vertex and the mesh stays
    // watertight.
    let mut vertex_cache: HashMap<(usize, usize), u32> = HashMap::new();

    for k in 0..field.nz() - 1 {
        for j in 0..field.ny() - 1 {
            for i in 0..field.nx() - 1 {
                march_cell(field, iso, (i, j, k), &mut vertex_cache, &mut mesh, budget)?;
            }
        }
    }

    Ok(mesh)
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
    cache: &mut HashMap<(usize, usize), u32>,
    mesh: &mut SurfaceMesh,
    budget: SurfaceBudget,
) -> Result<(), SurfaceBuildError> {
    let mut cube_nodes = [(0, 0, 0); 8];
    let mut cube_values = [0.0; 8];
    for corner in 0..8 {
        let node = corner_node(cell, corner);
        let value = field.value(node.0, node.1, node.2);
        if !value.is_finite() {
            return Err(SurfaceBuildError::InvalidField);
        }
        cube_nodes[corner] = node;
        cube_values[corner] = value;
    }
    for tet in &CUBE_TETRAHEDRA {
        // The four global nodes and field values of this tetrahedron.
        let mut nodes = [(0usize, 0usize, 0usize); 4];
        let mut vals = [0.0 as Real; 4];
        let mut mask = 0u8;
        for (local, &corner) in tet.iter().enumerate() {
            let n = cube_nodes[corner];
            nodes[local] = n;
            let v = cube_values[corner];
            vals[local] = v;
            if v > iso {
                mask |= 1 << local;
            }
        }

        let tris = &TET_TRIANGLES[mask as usize];
        let mut t = 0;
        while t < tris.len() && tris[t] >= 0 {
            if mesh.indices.len() / 3 >= budget.max_triangles
                || mesh.indices.len().checked_add(3).is_none()
            {
                return Err(SurfaceBuildError::TriangleBudgetExceeded);
            }
            mesh.indices
                .try_reserve(3)
                .map_err(|_| SurfaceBuildError::AllocationFailed)?;
            let e0 = tris[t] as usize;
            let e1 = tris[t + 1] as usize;
            let e2 = tris[t + 2] as usize;
            let i0 = edge_vertex(field, iso, &nodes, &vals, e0, cache, mesh, budget)?;
            let i1 = edge_vertex(field, iso, &nodes, &vals, e1, cache, mesh, budget)?;
            let i2 = edge_vertex(field, iso, &nodes, &vals, e2, cache, mesh, budget)?;
            mesh.indices.push(i0);
            mesh.indices.push(i1);
            mesh.indices.push(i2);
            t += 3;
        }
    }
    Ok(())
}

/// Returns the deduplicated mesh vertex index for the crossing on tetrahedron
/// edge `edge`, creating and caching a new vertex when first seen.
fn edge_vertex(
    field: &ScalarField,
    iso: Real,
    nodes: &[(usize, usize, usize); 4],
    vals: &[Real; 4],
    edge: usize,
    cache: &mut HashMap<(usize, usize), u32>,
    mesh: &mut SurfaceMesh,
    budget: SurfaceBudget,
) -> Result<u32, SurfaceBuildError> {
    let a = TET_EDGES[edge][0];
    let b = TET_EDGES[edge][1];
    let na = nodes[a];
    let nb = nodes[b];
    let ga = field.node_idx(na.0, na.1, na.2);
    let gb = field.node_idx(nb.0, nb.1, nb.2);
    let key = if ga <= gb { (ga, gb) } else { (gb, ga) };

    if let Some(&idx) = cache.get(&key) {
        return Ok(idx);
    }

    if mesh.positions.len() >= budget.max_vertices {
        return Err(SurfaceBuildError::VertexBudgetExceeded);
    }
    let idx =
        u32::try_from(mesh.positions.len()).map_err(|_| SurfaceBuildError::VertexBudgetExceeded)?;

    let va = vals[a];
    let vb = vals[b];
    let denom = vb - va;
    if !denom.is_finite() {
        return Err(SurfaceBuildError::InvalidField);
    }
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
    if !position.is_finite() || !grad.is_finite() {
        return Err(SurfaceBuildError::InvalidField);
    }
    let normal = {
        let n = -grad;
        // Scale before normalization so a finite large gradient cannot overflow
        // its squared length and silently produce a zero normal.
        let scale = n.abs().max_element();
        if scale > Real::EPSILON {
            (n / scale).normalize()
        } else {
            Vec3::Y
        }
    };

    mesh.positions
        .try_reserve(1)
        .map_err(|_| SurfaceBuildError::AllocationFailed)?;
    mesh.normals
        .try_reserve(1)
        .map_err(|_| SurfaceBuildError::AllocationFailed)?;
    cache
        .try_reserve(1)
        .map_err(|_| SurfaceBuildError::AllocationFailed)?;
    mesh.positions.push(position);
    mesh.normals.push(normal);
    cache.insert(key, idx);
    Ok(idx)
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

    #[test]
    fn bounded_surface_exact_limits_match_and_each_lower_limit_rejects() {
        let field = sphere_field(Vec3::ZERO, 0.6, 0.2, 11, Vec3::splat(-1.0));
        let reference = triangulate(&field, 0.0);
        assert!(!reference.is_empty());
        let budget = SurfaceBudget {
            max_cells: 1_000,
            max_vertices: reference.vertex_count(),
            max_triangles: reference.triangle_count(),
        };
        assert_eq!(triangulate_bounded(&field, 0.0, budget), Ok(reference));
        assert_eq!(
            triangulate_bounded(
                &field,
                0.0,
                SurfaceBudget {
                    max_cells: 999,
                    ..budget
                }
            ),
            Err(SurfaceBuildError::CellBudgetExceeded)
        );
        assert_eq!(
            triangulate_bounded(
                &field,
                0.0,
                SurfaceBudget {
                    max_vertices: budget.max_vertices - 1,
                    ..budget
                }
            ),
            Err(SurfaceBuildError::VertexBudgetExceeded)
        );
        assert_eq!(
            triangulate_bounded(
                &field,
                0.0,
                SurfaceBudget {
                    max_triangles: budget.max_triangles - 1,
                    ..budget
                }
            ),
            Err(SurfaceBuildError::TriangleBudgetExceeded)
        );
    }

    #[test]
    fn bounded_surface_empty_and_non_finite_inputs_are_explicit() {
        let budget = SurfaceBudget {
            max_cells: 1,
            max_vertices: 0,
            max_triangles: 0,
        };
        assert!(triangulate_bounded(&ScalarField::default(), 0.0, budget)
            .unwrap()
            .is_empty());
        assert_eq!(
            triangulate_bounded(&ScalarField::default(), Real::NAN, budget),
            Err(SurfaceBuildError::InvalidField)
        );
        let mut field = ScalarField::zeros(2, 2, 2, 0.1, Vec3::ZERO);
        // An all-outside cell still has to validate every field value.
        field.add_value(1, 1, 1, Real::NAN);
        assert_eq!(
            triangulate_bounded(&field, 0.1, budget),
            Err(SurfaceBuildError::InvalidField)
        );
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
