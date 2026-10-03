//! Reverse `Cuthill-McKee` reordering of a tetrahedral mesh's vertices.
//!
//! Finite-element assembly over a tetrahedral mesh produces a sparse stiffness
//! matrix whose nonzero pattern follows the vertex one-ring graph built by
//! [`crate::collider::tet_vertex_adjacency`]. The *bandwidth* of that matrix
//! (the largest index gap across any edge) controls both the fill-in of a
//! banded factorisation and the cache locality of sparse matrix-vector
//! products: neighbouring vertices that sit far apart in memory thrash the
//! cache. Relabelling the vertices to pull every edge's endpoints close
//! together shrinks the bandwidth.
//!
//! `Cuthill-McKee` is a breadth-first relabelling that visits the graph in
//! level sets, ordering each frontier by ascending degree so that
//! low-connectivity vertices are placed first. Reversing the resulting order
//! (`RCM`) typically reduces fill-in further and is the standard choice. This
//! module returns the permutation only; callers apply it to their own vertex
//! arrays, tet indices, and solver state.
//!
//! The permutation is a pure function of connectivity and is fully
//! deterministic: components are seeded in ascending index order, and each
//! breadth-first frontier is ordered by `(degree, index)`.
//!
//! `Cuthill-McKee` and its reverse are classical, publicly documented graph
//! bandwidth-reduction algorithms; nothing here is derived from Unreal Engine
//! source.

use super::tet_vertex_adjacency::{build_tet_vertex_adjacency, TetVertexAdjacency};

/// A vertex relabelling of a tetrahedral mesh.
///
/// The two arrays are mutual inverses and each is a permutation of
/// `0..vertex_count()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetVertexReorder {
    /// `old_to_new[old]` is the new position assigned to the vertex that was
    /// originally at index `old`.
    pub old_to_new: Vec<u32>,
    /// `new_to_old[new]` is the original index of the vertex now placed at
    /// position `new`.
    pub new_to_old: Vec<u32>,
}

impl TetVertexReorder {
    /// Number of vertices in the permutation.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.new_to_old.len()
    }

    /// The new position of the originally-indexed vertex `old`, or `None` when
    /// `old` is out of range.
    #[must_use]
    pub fn new_index(&self, old: usize) -> Option<u32> {
        self.old_to_new.get(old).copied()
    }

    /// The original index of the vertex now placed at position `new`, or `None`
    /// when `new` is out of range.
    #[must_use]
    pub fn old_index(&self, new: usize) -> Option<u32> {
        self.new_to_old.get(new).copied()
    }
}

/// Computes a reverse `Cuthill-McKee` relabelling of the given vertex graph.
///
/// Each connected component is seeded from its lowest unvisited index and
/// explored breadth-first, with every frontier ordered by ascending
/// `(degree, index)`. The concatenated `Cuthill-McKee` order is then reversed
/// to form the `RCM` permutation. Vertices that no tet references form
/// singleton components and still appear in the permutation.
#[must_use]
pub fn reorder_tet_vertices(adjacency: &TetVertexAdjacency) -> TetVertexReorder {
    let n = adjacency.vertex_count();
    let mut visited = vec![false; n];
    // The breadth-first queue is the Cuthill-McKee order itself; a head cursor
    // walks it so each vertex is appended before its neighbours are expanded.
    let mut cm_order: Vec<u32> = Vec::with_capacity(n);

    for seed in 0..n {
        if visited[seed] {
            continue;
        }
        visited[seed] = true;
        let component_start = cm_order.len();
        cm_order.push(seed as u32);
        let mut head = component_start;
        while head < cm_order.len() {
            let current = cm_order[head] as usize;
            head += 1;
            let mut frontier: Vec<u32> = adjacency
                .neighbours_of(current)
                .iter()
                .copied()
                .filter(|&w| !visited[w as usize])
                .collect();
            frontier.sort_by_key(|&w| (adjacency.degree(w as usize), w));
            for w in frontier {
                visited[w as usize] = true;
                cm_order.push(w);
            }
        }
    }

    // Reversing the Cuthill-McKee order yields the reverse variant.
    let mut new_to_old = cm_order;
    new_to_old.reverse();
    let mut old_to_new = vec![0u32; n];
    for (new_pos, &old) in new_to_old.iter().enumerate() {
        old_to_new[old as usize] = new_pos as u32;
    }

    TetVertexReorder {
        old_to_new,
        new_to_old,
    }
}

/// Builds the vertex graph of the mesh and reorders it in one call.
///
/// Returns `None` under the same conditions as
/// [`build_tet_vertex_adjacency`]: empty `tets` or an out-of-range vertex
/// index.
#[must_use]
pub fn reorder_tet_vertex_graph(
    num_vertices: usize,
    tets: &[[u32; 4]],
) -> Option<TetVertexReorder> {
    let adjacency = build_tet_vertex_adjacency(num_vertices, tets)?;
    Some(reorder_tet_vertices(&adjacency))
}

/// The bandwidth of the vertex graph under the relabelling `perm`.
///
/// `perm[old]` must give the new position of each vertex (the `old_to_new`
/// array). The bandwidth is the largest `|perm[u] - perm[w]|` over every edge
/// `(u, w)`; smaller is better. Returns zero when the graph is edgeless or when
/// `perm` does not cover every vertex.
#[must_use]
pub fn vertex_bandwidth(adjacency: &TetVertexAdjacency, perm: &[u32]) -> usize {
    let n = adjacency.vertex_count();
    if perm.len() != n {
        return 0;
    }
    let mut max_bandwidth = 0usize;
    for u in 0..n {
        let pu = perm[u];
        for &w in adjacency.neighbours_of(u) {
            let pw = perm[w as usize];
            let gap = pu.abs_diff(pw) as usize;
            if gap > max_bandwidth {
                max_bandwidth = gap;
            }
        }
    }
    max_bandwidth
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A linear chain of face-sharing tets: `tet[k] = [k, k+1, k+2, k+3]`.
    /// The vertex graph joins each vertex to those within three of it, so the
    /// natural order already has a small bandwidth while an interleaved
    /// relabelling has a large one.
    fn tet_chain(links: usize) -> (usize, Vec<[u32; 4]>) {
        let tets: Vec<[u32; 4]> = (0..links)
            .map(|k| {
                let k = k as u32;
                [k, k + 1, k + 2, k + 3]
            })
            .collect();
        (links + 3, tets)
    }

    fn is_permutation(perm: &[u32]) -> bool {
        let mut seen = vec![false; perm.len()];
        for &p in perm {
            let i = p as usize;
            if i >= seen.len() || seen[i] {
                return false;
            }
            seen[i] = true;
        }
        seen.iter().all(|&s| s)
    }

    #[test]
    fn permutation_is_valid_bijection_on_cube() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let order = reorder_tet_vertex_graph(mesh.vertices.len(), &mesh.tets).unwrap();
        let n = mesh.vertices.len();
        assert_eq!(order.vertex_count(), n);
        assert!(is_permutation(&order.old_to_new));
        assert!(is_permutation(&order.new_to_old));
        // The two arrays are exact inverses.
        for old in 0..n {
            let new = order.old_to_new[old] as usize;
            assert_eq!(order.new_to_old[new] as usize, old);
        }
    }

    #[test]
    fn single_tet_is_valid_permutation() {
        let order = reorder_tet_vertex_graph(4, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(order.vertex_count(), 4);
        assert!(is_permutation(&order.old_to_new));
        assert!(is_permutation(&order.new_to_old));
    }

    #[test]
    fn unreferenced_vertex_appears_in_permutation() {
        // Vertex 4 is a singleton component but must still be relabelled.
        let order = reorder_tet_vertex_graph(5, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(order.vertex_count(), 5);
        assert!(is_permutation(&order.old_to_new));
        assert!(order.new_index(4).is_some());
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = reorder_tet_vertex_graph(mesh.vertices.len(), &mesh.tets).unwrap();
        let b = reorder_tet_vertex_graph(mesh.vertices.len(), &mesh.tets).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_or_invalid_returns_none() {
        assert!(reorder_tet_vertex_graph(0, &[]).is_none());
        assert!(reorder_tet_vertex_graph(4, &[[0u32, 1, 2, 9]]).is_none());
    }

    #[test]
    fn reorder_beats_an_interleaved_relabelling() {
        let (nv, tets) = tet_chain(10);
        let adj = build_tet_vertex_adjacency(nv, &tets).unwrap();
        let order = reorder_tet_vertices(&adj);

        // Identity labelling of a chain already has a small bandwidth; RCM must
        // not do worse than it.
        let identity: Vec<u32> = (0..nv as u32).collect();
        let id_bw = vertex_bandwidth(&adj, &identity);
        let rcm_bw = vertex_bandwidth(&adj, &order.old_to_new);
        assert!(
            rcm_bw <= id_bw,
            "RCM bandwidth {rcm_bw} worse than identity {id_bw}"
        );

        // An even/odd split scatters each edge across the array, giving a large
        // bandwidth that RCM must beat.
        let evens = nv.div_ceil(2);
        let bad: Vec<u32> = (0..nv)
            .map(|i| {
                if i % 2 == 0 {
                    (i / 2) as u32
                } else {
                    (evens + i / 2) as u32
                }
            })
            .collect();
        assert!(is_permutation(&bad));
        let bad_bw = vertex_bandwidth(&adj, &bad);
        assert!(
            rcm_bw < bad_bw,
            "RCM bandwidth {rcm_bw} not better than interleaved {bad_bw}"
        );
    }

    #[test]
    fn bandwidth_rejects_mismatched_permutation() {
        let (nv, tets) = tet_chain(4);
        let adj = build_tet_vertex_adjacency(nv, &tets).unwrap();
        // A permutation of the wrong length is reported as zero bandwidth.
        assert_eq!(vertex_bandwidth(&adj, &[0, 1, 2]), 0);
    }
}
