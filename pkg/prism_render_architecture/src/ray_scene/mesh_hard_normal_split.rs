//! Hard-normal (smoothing-group) vertex splitting for the `CPU` golden path.
//!
//! Smooth shading averages a vertex's incident face normals into one shared
//! normal, which rounds off every edge meeting at that vertex. For *hard*
//! edges — the sharp creases, open boundaries, and non-manifold junctions
//! classified by [`super::mesh_feature_edges`] — that averaging smears the
//! silhouette and leaks lighting across what should be a crisp break. The fix
//! used by every `AAA` asset pipeline is to **split** the shared vertex into
//! one copy per *smoothing group*: the maximal set of faces around the vertex
//! reachable from one another without crossing a hard edge. Each copy then
//! carries the area-weighted normal of just its own group, so a cube renders
//! with eight sharp corners instead of a balloon.
//!
//! [`split_hard_normals`] classifies edges with a cosine threshold (reusing
//! [`super::mesh_feature_edges::detect_feature_edges`]), then, for every
//! original vertex, unions its incident triangles across each *smooth* edge
//! touching that vertex. Faces separated by a crease, boundary, or
//! non-manifold edge land in different groups. Every resulting group becomes
//! one output vertex whose normal is the normalized sum of the raw face
//! cross-products of its triangles (area weighting, since the cross magnitude
//! is twice the triangle area). Positions and optional `UV`s are duplicated
//! from the source vertex; unreferenced vertices are dropped.
//!
//! The output mesh always carries normals even when the input did not. All
//! math is cross/dot plus one `sqrt` per group, honouring the golden-path ban
//! on `f32` transcendental functions.

use std::collections::HashMap;

use super::mesh_feature_edges::{detect_feature_edges, FeatureEdgeError};
use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Errors returned by [`split_hard_normals`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HardNormalError {
    /// `cos_threshold` was not finite or lay outside the valid cosine range
    /// `[-1, 1]`.
    InvalidThreshold,
    /// The split result failed to rebuild into a valid [`TriangleMesh`]. This
    /// should not occur for well-formed input and indicates an internal
    /// invariant violation.
    Rebuild(TriangleMeshError),
}

impl core::fmt::Display for HardNormalError {
    /// Formats the error as a short human-readable diagnostic.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidThreshold => {
                f.write_str("cos_threshold must be finite and within [-1, 1]")
            }
            Self::Rebuild(err) => write!(f, "failed to rebuild split mesh: {err}"),
        }
    }
}

impl std::error::Error for HardNormalError {}

impl From<FeatureEdgeError> for HardNormalError {
    /// Maps the feature-edge threshold error onto the matching hard-normal
    /// error variant.
    fn from(err: FeatureEdgeError) -> Self {
        match err {
            FeatureEdgeError::InvalidThreshold => Self::InvalidThreshold,
        }
    }
}

/// Result of splitting a mesh along its hard edges into smoothing groups.
#[derive(Clone, Debug)]
pub struct HardNormalSplit {
    /// The rebuilt mesh with per-smoothing-group vertices and normals.
    mesh: TriangleMesh,
    /// Count of original vertices that were split into more than one output
    /// vertex because two or more smoothing groups met there.
    split_vertices: usize,
}

impl HardNormalSplit {
    /// Returns the rebuilt mesh with hard-edge-aware per-corner normals.
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// Consumes the result, returning the rebuilt mesh.
    pub fn into_mesh(self) -> TriangleMesh {
        self.mesh
    }

    /// Returns the number of original vertices that were split into more than
    /// one output vertex (i.e. had multiple smoothing groups meeting there).
    pub fn split_vertices(&self) -> usize {
        self.split_vertices
    }
}

/// Splits `mesh` along its hard edges and recomputes an area-weighted normal
/// per smoothing group, treating a manifold edge as hard (a crease) when its
/// two unit face normals dot below `cos_threshold`, and always treating
/// boundary and non-manifold edges as hard.
///
/// Positions and optional `UV`s are duplicated from the source vertex for each
/// group; the output mesh always carries normals. Unreferenced input vertices
/// are dropped.
///
/// # Errors
///
/// Returns [`HardNormalError::InvalidThreshold`] when `cos_threshold` is not
/// finite or lies outside `[-1, 1]`, or [`HardNormalError::Rebuild`] if the
/// rebuilt topology is somehow invalid.
pub fn split_hard_normals(
    mesh: &TriangleMesh,
    cos_threshold: f32,
) -> Result<HardNormalSplit, HardNormalError> {
    let features = detect_feature_edges(mesh, cos_threshold)?;

    let positions = mesh.positions();
    let indices = mesh.indices();
    let vertex_count = mesh.vertex_count();

    // Precompute the raw (area-weighted) face normal of every triangle.
    let face_normals: Vec<[f32; 3]> =
        indices.iter().map(|tri| raw_face_normal(positions, tri)).collect();

    // For each vertex, record the triangles incident to it (in triangle order).
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
    for (triangle, tri) in indices.iter().enumerate() {
        for &corner in tri {
            incident[corner as usize].push(triangle);
        }
    }

    // Map each undirected edge to the triangles sharing it.
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (triangle, tri) in indices.iter().enumerate() {
        let [a, b, c] = *tri;
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            edge_faces.entry(sorted_pair(u, v)).or_default().push(triangle);
        }
    }

    // Build the new vertex set by grouping each vertex's incident triangles.
    let mut new_positions: Vec<[f32; 3]> = Vec::new();
    let mut new_normals: Vec<[f32; 3]> = Vec::new();
    let mut new_uvs: Vec<[f32; 2]> = Vec::new();
    let has_uvs = mesh.has_uvs();
    let uvs = mesh.uvs();
    // Maps (triangle, original vertex) to the output vertex index.
    let mut corner_remap: HashMap<(usize, u32), u32> = HashMap::new();
    let mut split_vertices = 0usize;

    for (vertex, tris) in incident.iter().enumerate() {
        if tris.is_empty() {
            continue;
        }
        let vertex_u32 = vertex as u32;

        // Local union-find over this vertex's incident triangles.
        let count = tris.len();
        let mut parent: Vec<usize> = (0..count).collect();
        let local_of: HashMap<usize, usize> =
            tris.iter().enumerate().map(|(local, &tri)| (tri, local)).collect();

        for (local, &triangle) in tris.iter().enumerate() {
            let [a, b, c] = indices[triangle];
            for &(u, v) in &[(a, b), (b, c), (c, a)] {
                // Only edges that actually touch this vertex can merge its
                // incident triangles.
                if u != vertex_u32 && v != vertex_u32 {
                    continue;
                }
                let edge = sorted_pair(u, v);
                // Hard edges (crease / boundary / non-manifold) never merge.
                if features.is_feature(edge.0, edge.1) {
                    continue;
                }
                if let Some(faces) = edge_faces.get(&edge) {
                    for &other in faces {
                        if let Some(&other_local) = local_of.get(&other) {
                            union(&mut parent, local, other_local);
                        }
                    }
                }
            }
        }

        // Assign one output vertex per group, in first-seen order.
        let mut root_to_new: HashMap<usize, u32> = HashMap::new();
        let mut group_count = 0usize;
        for (local, &triangle) in tris.iter().enumerate() {
            let root = find(&mut parent, local);
            let new_index = *root_to_new.entry(root).or_insert_with(|| {
                group_count += 1;
                let idx = new_positions.len() as u32;
                new_positions.push(positions[vertex]);
                new_normals.push([0.0, 0.0, 0.0]);
                if has_uvs {
                    new_uvs.push(uvs[vertex]);
                }
                idx
            });
            // Accumulate the area-weighted face normal into this group.
            let fn_ = face_normals[triangle];
            let acc = &mut new_normals[new_index as usize];
            acc[0] += fn_[0];
            acc[1] += fn_[1];
            acc[2] += fn_[2];
            corner_remap.insert((triangle, vertex_u32), new_index);
        }
        if group_count > 1 {
            split_vertices += 1;
        }
    }

    // Normalize the accumulated group normals.
    for normal in &mut new_normals {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 0.0 {
            let inv = 1.0 / len;
            normal[0] *= inv;
            normal[1] *= inv;
            normal[2] *= inv;
        }
    }

    // Remap every triangle corner onto its output vertex.
    let mut new_indices: Vec<[u32; 3]> = Vec::with_capacity(indices.len());
    for (triangle, tri) in indices.iter().enumerate() {
        let remap = |corner: u32| corner_remap[&(triangle, corner)];
        new_indices.push([remap(tri[0]), remap(tri[1]), remap(tri[2])]);
    }

    let uvs_out = if has_uvs { new_uvs } else { Vec::new() };
    let mesh = TriangleMesh::new(new_positions, new_normals, uvs_out, new_indices)
        .map_err(HardNormalError::Rebuild)?;

    Ok(HardNormalSplit { mesh, split_vertices })
}

/// Returns the sorted `(min, max)` endpoint pair keying a shared edge.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Returns the raw (un-normalized, area-weighted) geometric normal of a
/// triangle; its magnitude is twice the triangle's area.
fn raw_face_normal(positions: &[[f32; 3]], tri: &[u32; 3]) -> [f32; 3] {
    let p0 = positions[tri[0] as usize];
    let p1 = positions[tri[1] as usize];
    let p2 = positions[tri[2] as usize];
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ]
}

/// Returns the representative root of `x`, compressing the path along the way.
fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

/// Merges the sets containing `a` and `b`.
fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat unit square (two coplanar triangles) sharing diagonal (1, 2).
    fn flat_quad() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [2, 1, 3]],
        )
        .unwrap()
    }

    /// Two triangles folded at a 90 degree dihedral along their shared edge
    /// (0, 1).
    fn folded_pair() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2], [0, 1, 3]],
        )
        .unwrap()
    }

    #[test]
    fn rejects_invalid_threshold() {
        for bad in [f32::NAN, f32::INFINITY, 2.0, -2.0] {
            assert_eq!(
                split_hard_normals(&flat_quad(), bad).unwrap_err(),
                HardNormalError::InvalidThreshold
            );
        }
    }

    #[test]
    fn flat_mesh_is_not_split() {
        let split = split_hard_normals(&flat_quad(), 0.5).unwrap();
        // All four corners stay shared: no vertex is duplicated.
        assert_eq!(split.mesh().vertex_count(), 4);
        assert_eq!(split.split_vertices(), 0);
        // Every normal points along +Z.
        for n in split.mesh().normals() {
            assert!((n[0]).abs() < 1e-6);
            assert!((n[1]).abs() < 1e-6);
            assert!((n[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn output_always_has_normals() {
        // Input quad carries no normals; output must.
        let split = split_hard_normals(&flat_quad(), 0.5).unwrap();
        assert!(split.mesh().has_normals());
        assert_eq!(split.mesh().normals().len(), split.mesh().vertex_count());
    }

    #[test]
    fn sharp_fold_splits_shared_edge_vertices() {
        // 90 degree fold, strict threshold -> edge (0,1) is a crease.
        let split = split_hard_normals(&folded_pair(), 0.707).unwrap();
        // Vertices 0 and 1 each sit on the crease between two groups -> each
        // splits into two copies. Vertices 2 and 3 are unique to one face.
        // 4 original -> 2 (unchanged) + 2 * 2 (split) = 6 output vertices.
        assert_eq!(split.mesh().vertex_count(), 6);
        assert_eq!(split.split_vertices(), 2);
    }

    #[test]
    fn gentle_threshold_keeps_fold_welded() {
        // Permissive threshold: the 90 degree fold is smooth, so nothing
        // splits and the shared edge stays welded.
        let split = split_hard_normals(&folded_pair(), 0.0).unwrap();
        assert_eq!(split.mesh().vertex_count(), 4);
        assert_eq!(split.split_vertices(), 0);
    }

    #[test]
    fn split_normals_are_per_face_at_a_crease() {
        let split = split_hard_normals(&folded_pair(), 0.707).unwrap();
        let mesh = split.mesh();
        // Triangle [0,1,2] lies in the z=0 plane -> normal +/-Z.
        // Triangle [0,1,3] lies in the y=0 plane -> normal +/-Y.
        // After the split each triangle's corner normals match its own face,
        // not a blend of the two.
        for tri in mesh.indices() {
            let n0 = mesh.normals()[tri[0] as usize];
            let n1 = mesh.normals()[tri[1] as usize];
            let n2 = mesh.normals()[tri[2] as usize];
            // All three corner normals of a face are identical (flat face).
            for k in 0..3 {
                assert!((n0[k] - n1[k]).abs() < 1e-6);
                assert!((n0[k] - n2[k]).abs() < 1e-6);
            }
            // The face normal is axis-aligned (pure +/-Y or +/-Z), i.e. not a
            // 45 degree blend.
            let max_axis = n0[0].abs().max(n0[1].abs()).max(n0[2].abs());
            assert!(max_axis > 0.99);
        }
    }

    #[test]
    fn preserves_uvs_when_present() {
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            Vec::new(),
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            vec![[0, 1, 2], [0, 1, 3]],
        )
        .unwrap();
        let split = split_hard_normals(&mesh, 0.707).unwrap();
        assert!(split.mesh().has_uvs());
        assert_eq!(split.mesh().uvs().len(), split.mesh().vertex_count());
        // A split copy of vertex 0 keeps vertex 0's UV (0,0).
        let mut saw_origin_uv = 0;
        for uv in split.mesh().uvs() {
            if uv[0].abs() < 1e-6 && uv[1].abs() < 1e-6 {
                saw_origin_uv += 1;
            }
        }
        // Vertex 0 split into two copies -> its UV appears twice.
        assert_eq!(saw_origin_uv, 2);
    }

    #[test]
    fn triangle_count_is_preserved() {
        let split = split_hard_normals(&folded_pair(), 0.707).unwrap();
        assert_eq!(split.mesh().triangle_count(), 2);
    }

    #[test]
    fn unreferenced_vertices_are_dropped() {
        // Vertex 3 is never indexed.
        let mesh = TriangleMesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [9.0, 9.0, 9.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let split = split_hard_normals(&mesh, 0.5).unwrap();
        // Only the three referenced corners survive.
        assert_eq!(split.mesh().vertex_count(), 3);
    }

    #[test]
    fn empty_mesh_round_trips() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0]],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let split = split_hard_normals(&mesh, 0.5).unwrap();
        assert_eq!(split.mesh().vertex_count(), 0);
        assert_eq!(split.mesh().triangle_count(), 0);
        assert_eq!(split.split_vertices(), 0);
    }

    #[test]
    fn remapped_indices_are_in_range() {
        let split = split_hard_normals(&folded_pair(), 0.707).unwrap();
        let n = split.mesh().vertex_count() as u32;
        for tri in split.mesh().indices() {
            for &i in tri {
                assert!(i < n);
            }
        }
    }
}
