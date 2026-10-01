//! Uniform-grid vertex clustering for fast, aggressive mesh `LOD` on the `CPU`
//! golden path.
//!
//! Where [`super::mesh_welding`] merges only near-coincident vertices to repair
//! split meshes without changing their shape, *vertex clustering*
//! (Rossignac–Borrel style) is a decimation technique: it overlays a coarse
//! uniform grid on the mesh and collapses **every** vertex that falls in the
//! same cell onto a single representative, regardless of the surface detail
//! lost. Driving the cell size up trades fidelity for triangle count, making it
//! the cheapest way to generate a very low-detail proxy — the far-distance
//! impostor rung beneath edge-collapse ([`super::mesh_decimation`]) `QEM`
//! simplification.
//!
//! [`cluster_vertices`] snaps each vertex into the integer grid cell
//! `floor(position / cell_size)`, then for every occupied cell emits one
//! representative whose attributes are the **centroid** of its members:
//! positions and `UV`s are plain averages and normals are averaged then
//! renormalized. The index buffer is rewritten to the per-cell representatives;
//! any triangle whose three corners no longer reference three distinct cells
//! has collapsed to zero area and is dropped, and representatives left
//! unreferenced after that cull are compacted out so the result carries no dead
//! data. A cell size at or below the mesh's minimum vertex spacing leaves every
//! vertex in its own cell and reproduces the input (up to representative
//! re-indexing); an over-large cell size can fold the whole mesh into a single
//! cell, yielding an empty mesh.
//!
//! All math is linear — grid snapping via `floor`, attribute averaging, and a
//! distinct-corner comparison — honouring the golden-path ban on `f32`
//! transcendental functions. Centroid accumulation is carried in `f64` so large
//! clusters do not lose precision.

use std::collections::HashMap;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Errors returned by [`cluster_vertices`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClusterError {
    /// The supplied cell size was not a finite, strictly positive value.
    InvalidCellSize,
    /// Rebuilding the clustered [`TriangleMesh`] failed. Because the clustered
    /// pools are internally consistent by construction this does not occur in
    /// practice, but the underlying error is surfaced rather than panicked on.
    Rebuild(TriangleMeshError),
}

impl core::fmt::Display for ClusterError {
    /// Formats the error as a short human-readable diagnostic.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidCellSize => {
                write!(f, "cluster cell size must be a finite, strictly positive value")
            }
            Self::Rebuild(err) => write!(f, "failed to rebuild clustered mesh: {err}"),
        }
    }
}

impl std::error::Error for ClusterError {}

/// Integer grid-cell coordinate used as the clustering key, one component per
/// spatial axis.
type CellKey = (i64, i64, i64);

/// Running centroid accumulator for one occupied grid cell.
struct Cluster {
    /// Output index assigned to this cell's representative, in first-seen
    /// order.
    rep: u32,
    /// Summed vertex positions, accumulated in `f64` for precision.
    position: [f64; 3],
    /// Summed vertex normals (only meaningful when the mesh has normals).
    normal: [f64; 3],
    /// Summed vertex `UV`s (only meaningful when the mesh has `UV`s).
    uv: [f64; 2],
    /// Number of source vertices folded into this cell.
    count: u32,
}

/// Simplifies `mesh` by collapsing every vertex sharing a `cell_size` grid cell
/// onto a single centroid representative, producing an aggressive `LOD` proxy.
///
/// Representatives average member positions and `UV`s and average-then-
/// renormalize member normals. Triangles that collapse to a point or edge once
/// their corners share a cell are dropped, and unreferenced representatives are
/// compacted away. A cell size at or below the minimum vertex spacing returns
/// the input geometry unchanged (up to re-indexing); an over-large cell size
/// may return an empty mesh.
///
/// # Errors
///
/// Returns [`ClusterError::InvalidCellSize`] when `cell_size` is not finite or
/// not strictly positive, and [`ClusterError::Rebuild`] if the clustered pools
/// fail [`TriangleMesh`] validation (not expected by construction).
pub fn cluster_vertices(
    mesh: &TriangleMesh,
    cell_size: f32,
) -> Result<TriangleMesh, ClusterError> {
    if !cell_size.is_finite() || cell_size <= 0.0 {
        return Err(ClusterError::InvalidCellSize);
    }

    let has_normals = mesh.has_normals();
    let has_uvs = mesh.has_uvs();
    let inv_cell = 1.0_f64 / cell_size as f64;

    // Assign each source vertex to a grid cell, accumulating centroids.
    let mut clusters: HashMap<CellKey, Cluster> = HashMap::new();
    let mut vertex_rep = vec![0_u32; mesh.vertex_count()];
    let mut next_rep: u32 = 0;
    for (vertex, position) in mesh.positions().iter().enumerate() {
        let key = cell_key(position, inv_cell);
        let entry = clusters.entry(key).or_insert_with(|| {
            let rep = next_rep;
            next_rep += 1;
            Cluster {
                rep,
                position: [0.0; 3],
                normal: [0.0; 3],
                uv: [0.0; 2],
                count: 0,
            }
        });
        entry.position[0] += position[0] as f64;
        entry.position[1] += position[1] as f64;
        entry.position[2] += position[2] as f64;
        if has_normals {
            let n = mesh.normals()[vertex];
            entry.normal[0] += n[0] as f64;
            entry.normal[1] += n[1] as f64;
            entry.normal[2] += n[2] as f64;
        }
        if has_uvs {
            let t = mesh.uvs()[vertex];
            entry.uv[0] += t[0] as f64;
            entry.uv[1] += t[1] as f64;
        }
        entry.count += 1;
        vertex_rep[vertex] = entry.rep;
    }

    // Materialize representative attributes in first-seen (`rep`) order.
    let rep_count = next_rep as usize;
    let mut rep_position = vec![[0.0_f32; 3]; rep_count];
    let mut rep_normal = vec![[0.0_f32; 3]; rep_count];
    let mut rep_uv = vec![[0.0_f32; 2]; rep_count];
    for cluster in clusters.values() {
        let inv = 1.0_f64 / f64::from(cluster.count);
        let idx = cluster.rep as usize;
        rep_position[idx] = [
            (cluster.position[0] * inv) as f32,
            (cluster.position[1] * inv) as f32,
            (cluster.position[2] * inv) as f32,
        ];
        if has_normals {
            rep_normal[idx] = normalize_or_z([
                cluster.normal[0] * inv,
                cluster.normal[1] * inv,
                cluster.normal[2] * inv,
            ]);
        }
        if has_uvs {
            rep_uv[idx] = [(cluster.uv[0] * inv) as f32, (cluster.uv[1] * inv) as f32];
        }
    }

    // Rewrite triangles to representatives, dropping collapsed faces, and track
    // which representatives survive so unreferenced ones can be compacted out.
    let mut used = vec![false; rep_count];
    let mut clustered_tris: Vec<[u32; 3]> = Vec::with_capacity(mesh.indices().len());
    for tri in mesh.indices() {
        let [a, b, c] = *tri;
        let (ra, rb, rc) = (
            vertex_rep[a as usize],
            vertex_rep[b as usize],
            vertex_rep[c as usize],
        );
        if ra == rb || rb == rc || rc == ra {
            continue;
        }
        used[ra as usize] = true;
        used[rb as usize] = true;
        used[rc as usize] = true;
        clustered_tris.push([ra, rb, rc]);
    }

    // Compact surviving representatives into a dense output pool.
    let mut remap = vec![u32::MAX; rep_count];
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for (rep, &is_used) in used.iter().enumerate() {
        if !is_used {
            continue;
        }
        remap[rep] = positions.len() as u32;
        positions.push(rep_position[rep]);
        if has_normals {
            normals.push(rep_normal[rep]);
        }
        if has_uvs {
            uvs.push(rep_uv[rep]);
        }
    }

    let indices: Vec<[u32; 3]> = clustered_tris
        .into_iter()
        .map(|[a, b, c]| [remap[a as usize], remap[b as usize], remap[c as usize]])
        .collect();

    TriangleMesh::new(positions, normals, uvs, indices).map_err(ClusterError::Rebuild)
}

/// Snaps `position` into its integer grid cell given the reciprocal cell size.
fn cell_key(position: &[f32; 3], inv_cell: f64) -> CellKey {
    (
        (position[0] as f64 * inv_cell).floor() as i64,
        (position[1] as f64 * inv_cell).floor() as i64,
        (position[2] as f64 * inv_cell).floor() as i64,
    )
}

/// Normalizes an `f64` vector to a unit `f32` normal, falling back to `+Z` for
/// a degenerate (zero-length) accumulation.
fn normalize_or_z(v: [f64; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        [(v[0] * inv) as f32, (v[1] * inv) as f32, (v[2] * inv) as f32]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::traversal::Ray;
    use crate::ray_scene::triangle_mesh::TriangleMeshBvh;

    /// A unit square (two triangles) in the z = 0 plane sharing diagonal (1, 2).
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

    #[test]
    fn rejects_non_finite_or_non_positive_cell_size() {
        let mesh = unit_quad();
        assert_eq!(
            cluster_vertices(&mesh, 0.0).unwrap_err(),
            ClusterError::InvalidCellSize
        );
        assert_eq!(
            cluster_vertices(&mesh, -1.0).unwrap_err(),
            ClusterError::InvalidCellSize
        );
        assert_eq!(
            cluster_vertices(&mesh, f32::NAN).unwrap_err(),
            ClusterError::InvalidCellSize
        );
        assert_eq!(
            cluster_vertices(&mesh, f32::INFINITY).unwrap_err(),
            ClusterError::InvalidCellSize
        );
    }

    #[test]
    fn fine_cells_preserve_geometry() {
        let mesh = unit_quad();
        // Cell far smaller than the 1.0 spacing → every vertex its own cell.
        let out = cluster_vertices(&mesh, 0.01).unwrap();
        assert_eq!(out.vertex_count(), 4);
        assert_eq!(out.triangle_count(), 2);
    }

    #[test]
    fn coarse_cell_merges_dense_cluster() {
        // Four vertices packed within one 1.0 cell plus two far away forming a
        // triangle: the packed group collapses to a single representative.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 4, 5], [1, 4, 5], [2, 4, 5]],
        )
        .unwrap();
        let out = cluster_vertices(&mesh, 1.0).unwrap();
        // The four packed corners fold to one; the two far corners stay.
        assert_eq!(out.vertex_count(), 3);
        // All three source triangles shared that packed corner, so each is the
        // same (rep, far_a, far_b) triangle after clustering; they remain valid
        // (three distinct cells) but are duplicates — still three faces.
        assert_eq!(out.triangle_count(), 3);
    }

    #[test]
    fn collapsed_triangles_are_dropped() {
        // Two near-coincident corners share a cell, collapsing the only face.
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 5.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let out = cluster_vertices(&mesh, 1.0).unwrap();
        assert_eq!(out.triangle_count(), 0);
        // With no surviving triangle every representative is unreferenced.
        assert_eq!(out.vertex_count(), 0);
    }

    #[test]
    fn representative_is_cell_centroid() {
        // Two vertices in one cell → representative at their midpoint.
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [0.4, 0.2, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 2, 3], [1, 2, 3]],
        )
        .unwrap();
        let out = cluster_vertices(&mesh, 1.0).unwrap();
        // The merged representative is the centroid of (0,0,0) and (0.4,0.2,0).
        let rep = out.positions()[0];
        assert!((rep[0] - 0.2).abs() < 1e-6, "x = {}", rep[0]);
        assert!((rep[1] - 0.1).abs() < 1e-6, "y = {}", rep[1]);
        assert!(rep[2].abs() < 1e-6, "z = {}", rep[2]);
    }

    #[test]
    fn normals_are_averaged_and_renormalized() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            Vec::new(),
            vec![[0, 2, 3], [1, 2, 3]],
        )
        .unwrap();
        let out = cluster_vertices(&mesh, 1.0).unwrap();
        assert!(out.has_normals());
        let n = out.normals()[0];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6, "normal not unit: {len}");
        // Average of +X and +Y, renormalized → equal x/y components.
        assert!((n[0] - n[1]).abs() < 1e-6, "n = {n:?}");
    }

    #[test]
    fn uvs_are_averaged() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            Vec::new(),
            vec![[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            vec![[0, 2, 3], [1, 2, 3]],
        )
        .unwrap();
        let out = cluster_vertices(&mesh, 1.0).unwrap();
        assert!(out.has_uvs());
        let uv = out.uvs()[0];
        assert!((uv[0] - 0.5).abs() < 1e-6, "u = {}", uv[0]);
        assert!((uv[1] - 0.5).abs() < 1e-6, "v = {}", uv[1]);
    }

    #[test]
    fn over_coarse_cell_empties_mesh() {
        let mesh = unit_quad();
        // One cell swallows the whole 1x1 quad → every face collapses.
        let out = cluster_vertices(&mesh, 1000.0).unwrap();
        assert_eq!(out.triangle_count(), 0);
        assert_eq!(out.vertex_count(), 0);
    }

    #[test]
    fn attributeless_mesh_stays_attributeless() {
        let mesh = unit_quad();
        let out = cluster_vertices(&mesh, 0.01).unwrap();
        assert!(!out.has_normals());
        assert!(!out.has_uvs());
    }

    #[test]
    fn clustered_mesh_is_ray_traceable() {
        // Fine clustering preserves the quad; a ray through an off-vertex,
        // off-seam point must still hit it.
        let mesh = cluster_vertices(&unit_quad(), 0.01).unwrap();
        let bvh = TriangleMeshBvh::build(mesh);
        let ray = Ray::infinite([0.53, 0.47, 1.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("ray should hit clustered quad");
        assert!((hit.position[2]).abs() < 1e-6, "hit z = {}", hit.position[2]);
    }
}
