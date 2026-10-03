//! Boundary-surface extraction for tetrahedral meshes.
//!
//! A tetrahedral volume mesh (produced by [`crate::collider::tetrahedralize`])
//! encloses its surface implicitly: a triangular face is on the boundary iff it
//! is shared by exactly one tet, while interior faces are shared by two. This
//! module recovers that surface as an outward-oriented, closed triangle mesh.
//!
//! Orientation is derived per face from its *apex* (the fourth vertex of the
//! owning tet): the face normal is flipped so it points away from the apex,
//! which for a boundary face is the outward direction. This is robust to the
//! winding of the input tets. Nothing here is derived from Unreal Engine
//! source; it is the standard face-counting boundary extraction.

use glam::Vec3;
use std::collections::HashMap;

/// The extracted boundary surface of a tetrahedral mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetBoundary {
    /// Outward-oriented boundary triangles, referencing the original tet-mesh
    /// vertex indices.
    pub triangles: Vec<[u32; 3]>,
    /// Sorted, de-duplicated indices of the vertices that lie on the boundary.
    pub boundary_vertices: Vec<u32>,
}

impl TetBoundary {
    /// Number of boundary triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Number of distinct vertices on the boundary.
    #[must_use]
    pub fn boundary_vertex_count(&self) -> usize {
        self.boundary_vertices.len()
    }

    /// Returns `true` when no boundary triangles were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Builds a standalone surface mesh: positions limited to the boundary
    /// vertices, with triangles re-indexed into that compact array.
    ///
    /// `vertices` must be the same slice passed to [`extract_tet_boundary`].
    #[must_use]
    pub fn to_mesh(&self, vertices: &[Vec3]) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut remap: HashMap<u32, u32> = HashMap::with_capacity(self.boundary_vertices.len());
        let mut positions = Vec::with_capacity(self.boundary_vertices.len());
        for &old in &self.boundary_vertices {
            let new = positions.len() as u32;
            remap.insert(old, new);
            positions.push(vertices[old as usize]);
        }
        let tris = self
            .triangles
            .iter()
            .map(|t| [remap[&t[0]], remap[&t[1]], remap[&t[2]]])
            .collect();
        (positions, tris)
    }
}

/// The four `(face, apex)` splits of a tetrahedron, as `([3 face verts], apex)`.
fn tet_face_splits(t: [u32; 4]) -> [([u32; 3], u32); 4] {
    [
        ([t[1], t[2], t[3]], t[0]),
        ([t[0], t[2], t[3]], t[1]),
        ([t[0], t[1], t[3]], t[2]),
        ([t[0], t[1], t[2]], t[3]),
    ]
}

/// Orients `face` so its geometric normal points away from `apex`.
fn orient_face_away_from_apex(face: [u32; 3], apex: u32, vertices: &[Vec3]) -> [u32; 3] {
    let a = vertices[face[0] as usize];
    let b = vertices[face[1] as usize];
    let c = vertices[face[2] as usize];
    let d = vertices[apex as usize];
    let normal = (b - a).cross(c - a);
    // Positive dot => normal currently points toward the apex (inward for a
    // boundary face); flip the winding so it points outward.
    if normal.dot(d - a) > 0.0 {
        [face[0], face[2], face[1]]
    } else {
        face
    }
}

/// Extracts the outward-oriented boundary surface of a tetrahedral mesh.
///
/// A face shared by exactly one tet is a boundary face; faces shared by two
/// tets are interior and dropped. Each boundary triangle is wound so its normal
/// points away from the owning tet's apex (i.e. outward).
///
/// Returns `None` when `tets` is empty or any tet references a vertex index
/// outside `vertices`.
#[must_use]
pub fn extract_tet_boundary(vertices: &[Vec3], tets: &[[u32; 4]]) -> Option<TetBoundary> {
    if tets.is_empty() {
        return None;
    }
    let n = vertices.len();
    for t in tets {
        for &id in t {
            if (id as usize) >= n {
                return None;
            }
        }
    }

    // Key each geometric face by its sorted vertex triple; store how many tets
    // use it plus one owning `(face, apex)` for later orientation.
    let mut faces: HashMap<[u32; 3], (u32, [u32; 3], u32)> = HashMap::new();
    for &t in tets {
        for (face, apex) in tet_face_splits(t) {
            let mut key = face;
            key.sort_unstable();
            let entry = faces.entry(key).or_insert((0, face, apex));
            entry.0 += 1;
        }
    }

    // Deterministic output: visit boundary faces in sorted-key order.
    let mut keys: Vec<[u32; 3]> = faces
        .iter()
        .filter_map(|(k, v)| if v.0 == 1 { Some(*k) } else { None })
        .collect();
    keys.sort_unstable();

    let mut triangles = Vec::with_capacity(keys.len());
    let mut vertex_set: Vec<u32> = Vec::with_capacity(keys.len() * 3);
    for key in keys {
        let (_, face, apex) = faces[&key];
        let oriented = orient_face_away_from_apex(face, apex, vertices);
        vertex_set.extend_from_slice(&oriented);
        triangles.push(oriented);
    }
    vertex_set.sort_unstable();
    vertex_set.dedup();

    Some(TetBoundary {
        triangles,
        boundary_vertices: vertex_set,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::{tetrahedralize, TetMeshParams};

    fn unit_tet() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        (v, vec![[0u32, 1, 2, 3]])
    }

    fn cube_surface(h: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        let idx = vec![
            [0u32, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (verts, idx)
    }

    #[test]
    fn empty_tets_returns_none() {
        assert!(extract_tet_boundary(&[], &[]).is_none());
        let (v, _) = unit_tet();
        assert!(extract_tet_boundary(&v, &[]).is_none());
    }

    #[test]
    fn out_of_range_index_returns_none() {
        let (v, _) = unit_tet();
        assert!(extract_tet_boundary(&v, &[[0u32, 1, 2, 9]]).is_none());
    }

    #[test]
    fn single_tet_yields_four_outward_faces() {
        let (v, tets) = unit_tet();
        let b = extract_tet_boundary(&v, &tets).unwrap();
        assert_eq!(b.triangle_count(), 4);
        assert_eq!(b.boundary_vertex_count(), 4);

        // Every face normal must point away from the tet centroid (outward).
        let centroid = (v[0] + v[1] + v[2] + v[3]) / 4.0;
        for tri in &b.triangles {
            let a = v[tri[0] as usize];
            let bb = v[tri[1] as usize];
            let c = v[tri[2] as usize];
            let normal = (bb - a).cross(c - a);
            let face_c = (a + bb + c) / 3.0;
            assert!(
                normal.dot(face_c - centroid) > 0.0,
                "face normal points inward"
            );
        }
    }

    #[test]
    fn shared_face_is_dropped() {
        // Two tets sharing face (0,1,2): boundary = 4 + 4 - 2 = 6 triangles.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 0.7, 1.6),
            Vec3::new(1.0, 0.7, -1.6),
        ];
        let tets = vec![[0u32, 1, 2, 3], [0, 2, 1, 4]];
        let b = extract_tet_boundary(&v, &tets).unwrap();
        assert_eq!(b.triangle_count(), 6);
        // All five vertices lie on the boundary (shared-face verts appear on
        // other faces too).
        assert_eq!(b.boundary_vertex_count(), 5);
    }

    #[test]
    fn cube_boundary_is_closed_and_consistently_oriented() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let b = extract_tet_boundary(&mesh.vertices, &mesh.tets).unwrap();
        assert!(!b.is_empty());

        // Closed + consistently oriented: each directed edge appears once, and
        // its reverse also appears once (so every undirected edge is used by
        // exactly two triangles).
        let mut directed: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &b.triangles {
            for (a, c) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *directed.entry((a, c)).or_insert(0) += 1;
            }
        }
        for (&(a, c), &count) in &directed {
            assert_eq!(count, 1, "edge ({a},{c}) used more than once");
            assert_eq!(
                directed.get(&(c, a)).copied().unwrap_or(0),
                1,
                "edge ({a},{c}) has no opposite half-edge"
            );
        }
    }

    #[test]
    fn cube_boundary_encloses_positive_volume() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let b = extract_tet_boundary(&mesh.vertices, &mesh.tets).unwrap();
        // Signed volume via the divergence theorem; outward winding => positive.
        let mut vol6 = 0.0f32;
        for t in &b.triangles {
            let a = mesh.vertices[t[0] as usize];
            let bb = mesh.vertices[t[1] as usize];
            let c = mesh.vertices[t[2] as usize];
            vol6 += a.dot(bb.cross(c));
        }
        assert!(vol6 > 0.0, "boundary encloses negative/zero volume");
    }

    #[test]
    fn to_mesh_compacts_vertices_and_reindexes() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let b = extract_tet_boundary(&mesh.vertices, &mesh.tets).unwrap();
        let (pos, tris) = b.to_mesh(&mesh.vertices);
        assert_eq!(pos.len(), b.boundary_vertex_count());
        assert_eq!(tris.len(), b.triangle_count());
        for t in &tris {
            for &id in t {
                assert!((id as usize) < pos.len(), "reindexed id out of range");
            }
        }
    }

    #[test]
    fn extraction_is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = extract_tet_boundary(&mesh.vertices, &mesh.tets).unwrap();
        let b = extract_tet_boundary(&mesh.vertices, &mesh.tets).unwrap();
        assert_eq!(a, b);
    }
}
