//! Cohesive-interface insertion by node splitting.
//!
//! Given a conforming tetrahedral mesh and a set of *interior* triangular
//! facets to open, this module performs the standard **cohesive insertion**
//! (node-splitting) preprocess: it duplicates the mesh vertices that straddle a
//! cut so the two sides of every opened facet can displace independently, then
//! emits one [`CohesiveInterface`] per opened facet wired to the duplicated
//! vertices. The result feeds directly into
//! [`assemble_cohesive_forces`](super::cohesive_zone_assembly::assemble_cohesive_forces).
//!
//! # Node splitting
//!
//! A vertex is *split* into as many copies as there are connected groups of
//! incident tets once the cut facets are treated as walls. Formally, two tets
//! that both touch a vertex `v` share the *same* copy of `v` exactly when they
//! are reachable from each other through faces that (a) contain `v` and (b) are
//! **not** cut. Each connected component becomes one copy: the first keeps the
//! original index, the rest are appended, and a `parent` map records the
//! original index of every copy so the caller can duplicate rest/current
//! positions with [`CohesiveMesh::expand_positions`].
//!
//! This handles the general case where a vertex borders several cut facets (so
//! it may split into three or more copies) and where cuts terminate inside the
//! mesh (a crack tip stays stitched because the tets around it remain connected
//! through the uncut faces beyond the tip).
//!
//! # Interface emission
//!
//! Every cut facet is interior, hence shared by exactly two tets `A` and `B`.
//! For each of the facet's three original vertices the builder looks up the
//! copy assigned to the `A`-side tet and the copy assigned to the `B`-side tet;
//! these paired copies form `side_a[i]` / `side_b[i]`, so the emitted interface
//! is correctly glued corner-to-corner.
//!
//! This module performs pure connectivity bookkeeping and holds no solver
//! state. Nothing here is derived from Unreal Engine source.

use crate::collider::cohesive_zone_assembly::CohesiveInterface;
use crate::collider::tet_adjacency::build_tet_adjacency;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// A tetrahedral mesh after cohesive insertion.
#[derive(Clone, Debug, PartialEq)]
pub struct CohesiveMesh {
    /// Number of vertices after splitting (`>= original`).
    pub vertex_count: usize,
    /// Remapped tet connectivity referencing the split vertices.
    pub tets: Vec<[u32; 4]>,
    /// One interface per opened facet, wired to the duplicated vertices.
    pub interfaces: Vec<CohesiveInterface>,
    /// `parent[new_id]` is the original vertex index each copy came from.
    pub parent: Vec<u32>,
}

impl CohesiveMesh {
    /// Expands an original per-vertex attribute array (e.g. positions) to the
    /// split vertex set by copying each parent's value to all of its copies.
    ///
    /// Returns `None` when `original.len()` does not match the pre-split vertex
    /// count implied by `parent` (its maximum parent index plus one).
    #[must_use]
    pub fn expand_positions(&self, original: &[Vec3]) -> Option<Vec<Vec3>> {
        let needed = self
            .parent
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1);
        if original.len() < needed {
            return None;
        }
        Some(self.parent.iter().map(|&p| original[p as usize]).collect())
    }

    /// Number of extra vertices created by splitting.
    #[must_use]
    pub fn added_vertex_count(&self) -> usize {
        self.vertex_count
            .saturating_sub(self.original_vertex_count())
    }

    /// The pre-split vertex count (highest parent index plus one).
    #[must_use]
    pub fn original_vertex_count(&self) -> usize {
        self.parent
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1)
    }
}

/// Sorted vertex triples of the four faces of a tet, face `f` opposite local
/// vertex `f` (matching [`build_tet_adjacency`]'s convention).
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

/// Disjoint-set (union–find) over a small, locally indexed set of tets.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Inserts cohesive interfaces along `cut_facets` by node splitting.
///
/// `cut_facets` entries are triples of original vertex indices in any order;
/// each must be an interior facet of the mesh (shared by exactly two tets).
/// Returns `None` when:
/// - `tets` is empty or references an out-of-range vertex,
/// - the mesh is non-manifold (a face shared by three or more tets), or
/// - any requested cut facet is not an interior facet of the mesh.
///
/// Passing an empty `cut_facets` yields the mesh unchanged with no interfaces.
#[must_use]
pub fn insert_cohesive_interfaces(
    vertex_count: usize,
    tets: &[[u32; 4]],
    cut_facets: &[[u32; 3]],
) -> Option<CohesiveMesh> {
    let adjacency = build_tet_adjacency(vertex_count, tets)?;

    // Canonical cut-facet set.
    let mut cut: HashSet<[u32; 3]> = HashSet::with_capacity(cut_facets.len());
    for &f in cut_facets {
        let mut s = f;
        s.sort_unstable();
        cut.insert(s);
    }

    // Validate every requested cut facet is an interior facet, and record the
    // two tets sharing it. interior_facets[sorted] = (tetA, tetB).
    let mut interior: HashMap<[u32; 3], (u32, u32)> = HashMap::new();
    for (ti, &t) in tets.iter().enumerate() {
        let faces = opposite_faces(t);
        for (f, key) in faces.into_iter().enumerate() {
            if let Some(nb) = adjacency.neighbours[ti][f] {
                // Record once, with the smaller tet id first for determinism.
                interior.entry(key).or_insert_with(|| {
                    let a = ti as u32;
                    if a < nb {
                        (a, nb)
                    } else {
                        (nb, a)
                    }
                });
            }
        }
    }
    for c in &cut {
        if !interior.contains_key(c) {
            return None;
        }
    }

    // --- Node splitting --------------------------------------------------
    // Build, per original vertex, the list of incident (tet, local-index).
    let mut incident: Vec<Vec<(u32, u8)>> = vec![Vec::new(); vertex_count];
    for (ti, &t) in tets.iter().enumerate() {
        for (lv, &v) in t.iter().enumerate() {
            incident[v as usize].push((ti as u32, lv as u8));
        }
    }

    let mut new_tets: Vec<[u32; 4]> = tets.to_vec();
    let mut parent: Vec<u32> = (0..vertex_count as u32).collect();

    for (v, inc) in incident.iter().enumerate() {
        if inc.is_empty() {
            continue;
        }
        // Local index of each incident tet within `inc`.
        let mut local_of: HashMap<u32, usize> = HashMap::with_capacity(inc.len());
        for (li, &(ti, _)) in inc.iter().enumerate() {
            local_of.insert(ti, li);
        }
        let mut uf = UnionFind::new(inc.len());
        // Connect incident tets across uncut faces that contain v.
        for (li, &(ti, _)) in inc.iter().enumerate() {
            let t = tets[ti as usize];
            let faces = opposite_faces(t);
            for (f, key) in faces.into_iter().enumerate() {
                // A face contains v iff v is one of its three vertices; the
                // face opposite local vertex f omits exactly t[f].
                if t[f] as usize == v {
                    continue; // this face does not contain v
                }
                if cut.contains(&key) {
                    continue; // the cut is a wall
                }
                if let Some(nb) = adjacency.neighbours[ti as usize][f]
                    && let Some(&lj) = local_of.get(&nb)
                {
                    uf.union(li, lj);
                }
            }
        }
        // Assign a copy id per component; first component keeps original `v`.
        let mut root_to_id: HashMap<usize, u32> = HashMap::new();
        let mut first_used = false;
        for (li, &(ti, lv)) in inc.iter().enumerate() {
            let root = uf.find(li);
            let id = if let Some(&existing) = root_to_id.get(&root) {
                existing
            } else {
                let new_id = if first_used {
                    let fresh = parent.len() as u32;
                    parent.push(v as u32);
                    fresh
                } else {
                    first_used = true;
                    v as u32
                };
                root_to_id.insert(root, new_id);
                new_id
            };
            new_tets[ti as usize][lv as usize] = id;
        }
    }

    // --- Interface emission ---------------------------------------------
    let mut interfaces = Vec::with_capacity(cut.len());
    for c in &cut {
        let (ta, tb) = interior[c];
        let side_a = map_face_corners(*c, &new_tets[ta as usize], tets[ta as usize])?;
        let side_b = map_face_corners(*c, &new_tets[tb as usize], tets[tb as usize])?;
        interfaces.push(CohesiveInterface::new(side_a, side_b));
    }

    Some(CohesiveMesh {
        vertex_count: parent.len(),
        tets: new_tets,
        interfaces,
        parent,
    })
}

/// Maps a facet's three original vertices (`face`, sorted) to their split copy
/// ids within one incident tet, preserving the facet's sorted corner order so
/// `side_a[i]` and `side_b[i]` refer to the same original vertex.
fn map_face_corners(face: [u32; 3], new_tet: &[u32; 4], orig_tet: [u32; 4]) -> Option<[u32; 3]> {
    let mut out = [0u32; 3];
    for (i, &ov) in face.iter().enumerate() {
        let local = orig_tet.iter().position(|&x| x == ov)?;
        out[i] = new_tet[local];
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two tets sharing the interior face {1,2,3}:
    //   tet0 = [0,1,2,3], tet1 = [1,2,3,4].
    fn two_tets() -> (usize, Vec<[u32; 4]>) {
        (5, vec![[0, 1, 2, 3], [1, 2, 3, 4]])
    }

    #[test]
    fn no_cut_leaves_mesh_unchanged() {
        let (nv, tets) = two_tets();
        let m = insert_cohesive_interfaces(nv, &tets, &[]).unwrap();
        assert_eq!(m.vertex_count, nv);
        assert_eq!(m.tets, tets);
        assert!(m.interfaces.is_empty());
        assert_eq!(m.added_vertex_count(), 0);
    }

    #[test]
    fn single_interior_cut_duplicates_shared_face() {
        let (nv, tets) = two_tets();
        let m = insert_cohesive_interfaces(nv, &tets, &[[1, 2, 3]]).unwrap();
        // The three shared vertices {1,2,3} each split into two copies.
        assert_eq!(m.added_vertex_count(), 3);
        assert_eq!(m.vertex_count, nv + 3);
        assert_eq!(m.interfaces.len(), 1);
        // The two tets must no longer share any vertex of the opened face.
        let a: HashSet<u32> = m.tets[0].iter().copied().collect();
        let b: HashSet<u32> = m.tets[1].iter().copied().collect();
        let shared: Vec<u32> = a.intersection(&b).copied().collect();
        assert!(
            shared.is_empty(),
            "opened face must fully separate, got {shared:?}"
        );
    }

    #[test]
    fn interface_corners_are_paired_by_parent() {
        let (nv, tets) = two_tets();
        let m = insert_cohesive_interfaces(nv, &tets, &[[1, 2, 3]]).unwrap();
        let iface = m.interfaces[0];
        for i in 0..3 {
            let pa = m.parent[iface.side_a[i] as usize];
            let pb = m.parent[iface.side_b[i] as usize];
            assert_eq!(pa, pb, "corner {i} must share an original parent");
        }
        // side_a / side_b must be distinct copies.
        for i in 0..3 {
            assert_ne!(iface.side_a[i], iface.side_b[i]);
        }
    }

    #[test]
    fn expand_positions_duplicates_parent_values() {
        let (nv, tets) = two_tets();
        let m = insert_cohesive_interfaces(nv, &tets, &[[1, 2, 3]]).unwrap();
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let expanded = m.expand_positions(&rest).unwrap();
        assert_eq!(expanded.len(), m.vertex_count);
        for (new_id, &p) in m.parent.iter().enumerate() {
            assert_eq!(expanded[new_id], rest[p as usize]);
        }
    }

    #[test]
    fn boundary_facet_cut_is_rejected() {
        let (nv, tets) = two_tets();
        // {0,1,2} is a boundary face of tet0 (no neighbour) ⇒ not interior.
        assert!(insert_cohesive_interfaces(nv, &tets, &[[0, 1, 2]]).is_none());
    }

    #[test]
    fn nonexistent_facet_cut_is_rejected() {
        let (nv, tets) = two_tets();
        assert!(insert_cohesive_interfaces(nv, &tets, &[[0, 2, 4]]).is_none());
    }

    #[test]
    fn out_of_range_vertex_is_rejected() {
        let tets = vec![[0u32, 1, 2, 9]];
        assert!(insert_cohesive_interfaces(4, &tets, &[]).is_none());
    }

    #[test]
    fn facet_order_is_normalized() {
        let (nv, tets) = two_tets();
        // Same facet given in a scrambled order must behave identically.
        let m1 = insert_cohesive_interfaces(nv, &tets, &[[1, 2, 3]]).unwrap();
        let m2 = insert_cohesive_interfaces(nv, &tets, &[[3, 1, 2]]).unwrap();
        assert_eq!(m1.vertex_count, m2.vertex_count);
        assert_eq!(m1.interfaces.len(), m2.interfaces.len());
    }

    // Three tets in a fan sharing a common edge, cutting one interior face
    // should split only the vertices on that face, keeping the rest stitched.
    #[test]
    fn partial_cut_keeps_untouched_vertices_single() {
        // tet0=[0,1,2,3], tet1=[1,2,3,4] share face {1,2,3}; cut it.
        let (nv, tets) = two_tets();
        let m = insert_cohesive_interfaces(nv, &tets, &[[1, 2, 3]]).unwrap();
        // Vertices 0 and 4 are each in only one tet ⇒ never split.
        assert_eq!(m.parent.iter().filter(|&&p| p == 0).count(), 1);
        assert_eq!(m.parent.iter().filter(|&&p| p == 4).count(), 1);
        // Vertices 1,2,3 each split into exactly two copies.
        for v in 1..=3u32 {
            assert_eq!(m.parent.iter().filter(|&&p| p == v).count(), 2, "v={v}");
        }
    }

    #[test]
    fn resulting_mesh_is_consistent_for_assembly() {
        use crate::collider::cohesive_zone::CohesiveModel;
        use crate::collider::cohesive_zone_assembly::{assemble_cohesive_forces, rest_states};
        let (nv, tets) = two_tets();
        let m = insert_cohesive_interfaces(nv, &tets, &[[1, 2, 3]]).unwrap();
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let rest_x = m.expand_positions(&rest).unwrap();
        let model = CohesiveModel::new(1.0e6, 1.0e3, 1.0, 1.0).unwrap();
        let mut states = rest_states(m.interfaces.len());
        // At rest the assembly must succeed and produce ~zero force.
        let out =
            assemble_cohesive_forces(&m.interfaces, &rest_x, &rest_x, &model, &mut states).unwrap();
        let total: Vec3 = out.forces.iter().copied().sum();
        assert!(total.length() < 1e-5);
        assert_eq!(out.forces.len(), m.vertex_count);
    }
}
