//! Tetrahedron-to-tetrahedron adjacency (the volumetric dual graph).
//!
//! Two tets are neighbours when they share a triangular face. This module
//! builds, for every tet, the index of the neighbour across each of its four
//! faces (or `None` on the boundary). The face indexing is canonical: face `f`
//! is the triangle *opposite* local vertex `f`, matching the convention used by
//! [`crate::collider::tet_boundary`].
//!
//! The dual graph is the foundation for Delaunay-style face flips, graph
//! colouring for parallel Gauss-Seidel, and domain partitioning. A well-formed
//! tetrahedral volume is a 3-manifold, so every interior face is shared by
//! exactly two tets; a face shared by three or more indicates a non-manifold
//! input and is rejected. This is standard mesh connectivity; nothing here is
//! derived from Unreal Engine source.

use std::collections::HashMap;

/// Per-tet face adjacency for a tetrahedral mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetAdjacency {
    /// `neighbours[t][f]` is the tet adjacent to tet `t` across the face
    /// opposite local vertex `f`, or `None` when that face lies on the
    /// boundary.
    pub neighbours: Vec<[Option<u32>; 4]>,
}

impl TetAdjacency {
    /// Number of tets described.
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.neighbours.len()
    }

    /// Total number of boundary faces (faces with no neighbour).
    #[must_use]
    pub fn boundary_face_count(&self) -> usize {
        self.neighbours
            .iter()
            .flat_map(|n| n.iter())
            .filter(|f| f.is_none())
            .count()
    }

    /// Number of interior faces (each shared face counted once).
    #[must_use]
    pub fn interior_face_count(&self) -> usize {
        let paired: usize = self
            .neighbours
            .iter()
            .flat_map(|n| n.iter())
            .filter(|f| f.is_some())
            .count();
        paired / 2
    }

    /// Returns `true` when tet `t` has at least one boundary face.
    #[must_use]
    pub fn is_boundary_tet(&self, t: usize) -> bool {
        self.neighbours
            .get(t)
            .is_some_and(|n| n.iter().any(Option::is_none))
    }
}

/// The sorted vertex triple of the face opposite each local vertex of a tet.
///
/// Entry `f` is the face opposite local vertex `f`.
fn opposite_faces(t: [u32; 4]) -> [[u32; 3]; 4] {
    let sort3 = |mut a: [u32; 3]| {
        a.sort_unstable();
        a
    };
    [
        sort3([t[1], t[2], t[3]]),
        sort3([t[0], t[2], t[3]]),
        sort3([t[0], t[1], t[3]]),
        sort3([t[0], t[1], t[2]]),
    ]
}

/// Builds the face-adjacency (dual graph) of a tetrahedral mesh.
///
/// Returns `None` when `tets` is empty, any tet references a vertex index `>=
/// num_vertices`, or any face is shared by three or more tets (non-manifold).
#[must_use]
pub fn build_tet_adjacency(num_vertices: usize, tets: &[[u32; 4]]) -> Option<TetAdjacency> {
    if tets.is_empty() {
        return None;
    }
    for t in tets {
        for &id in t {
            if (id as usize) >= num_vertices {
                return None;
            }
        }
    }

    // Map each face key to the (tet, local-face) slots that use it.
    let mut face_users: HashMap<[u32; 3], Vec<(u32, u8)>> = HashMap::new();
    for (ti, &t) in tets.iter().enumerate() {
        for (f, key) in opposite_faces(t).into_iter().enumerate() {
            let slot = face_users.entry(key).or_default();
            if slot.len() >= 2 {
                // A third user of this face: the volume is non-manifold.
                return None;
            }
            slot.push((ti as u32, f as u8));
        }
    }

    let mut neighbours = vec![[None; 4]; tets.len()];
    for users in face_users.values() {
        if let [(ta, fa), (tb, fb)] = users[..] {
            neighbours[ta as usize][fa as usize] = Some(tb);
            neighbours[tb as usize][fb as usize] = Some(ta);
        }
    }

    Some(TetAdjacency { neighbours })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_boundary::extract_tet_boundary;
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};
    use glam::Vec3;

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
        assert!(build_tet_adjacency(0, &[]).is_none());
        assert!(build_tet_adjacency(4, &[]).is_none());
    }

    #[test]
    fn out_of_range_index_returns_none() {
        assert!(build_tet_adjacency(4, &[[0u32, 1, 2, 9]]).is_none());
    }

    #[test]
    fn single_tet_has_four_boundary_faces() {
        let adj = build_tet_adjacency(4, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(adj.tet_count(), 1);
        assert_eq!(adj.boundary_face_count(), 4);
        assert_eq!(adj.interior_face_count(), 0);
        assert!(adj.is_boundary_tet(0));
        assert_eq!(adj.neighbours[0], [None, None, None, None]);
    }

    #[test]
    fn two_tets_share_exactly_one_face() {
        // Shared face (0,1,2); apexes 3 and 4.
        let tets = vec![[0u32, 1, 2, 3], [0, 2, 1, 4]];
        let adj = build_tet_adjacency(5, &tets).unwrap();
        assert_eq!(adj.interior_face_count(), 1);
        assert_eq!(adj.boundary_face_count(), 6);

        // Tet 0's shared face is opposite vertex 3 -> local face 3.
        assert_eq!(adj.neighbours[0][3], Some(1));
        assert_eq!(adj.neighbours[0][0], None);
        // Tet 1's shared face (0,1,2) is opposite its apex 4 -> local face 3.
        assert_eq!(adj.neighbours[1][3], Some(0));
    }

    #[test]
    fn three_tets_on_one_face_is_non_manifold() {
        // Three tets all sharing face (0,1,2) with distinct apexes.
        let tets = vec![[0u32, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]];
        assert!(build_tet_adjacency(6, &tets).is_none());
    }

    #[test]
    fn adjacency_is_symmetric_and_matches_boundary_extraction() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let adj = build_tet_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();

        // Symmetry: every pointer has a matching back-pointer.
        for t in 0..adj.tet_count() {
            for f in 0..4 {
                if let Some(nb) = adj.neighbours[t][f] {
                    let back = adj.neighbours[nb as usize]
                        .iter()
                        .any(|x| *x == Some(t as u32));
                    assert!(back, "tet {t} face {f} -> {nb} has no back-pointer");
                }
            }
        }

        // The boundary-face count must equal the number of triangles the
        // independent boundary extractor produces.
        let boundary = extract_tet_boundary(&mesh.vertices, &mesh.tets).unwrap();
        assert_eq!(adj.boundary_face_count(), boundary.triangle_count());
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = build_tet_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        let b = build_tet_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        assert_eq!(a, b);
    }
}
