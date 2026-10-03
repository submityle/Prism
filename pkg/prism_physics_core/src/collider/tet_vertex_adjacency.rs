//! Vertex one-ring adjacency of a tetrahedral mesh (the node graph).
//!
//! Where [`crate::collider::tet_adjacency`] builds the *element* dual graph
//! (tet-to-tet across shared faces), this module builds the *vertex* graph:
//! two vertices are neighbours when a tet edge joins them. Every tet
//! `[a, b, c, d]` contributes the six undirected edges `ab, ac, ad, bc, bd, cd`;
//! edges shared by several tets are merged.
//!
//! The vertex graph is the foundation for vertex-based parallel solvers
//! (vertex-block descent, Jacobi/Gauss-Seidel node sweeps, graph colouring over
//! nodes) and for finite-element stiffness-matrix sparsity. It mirrors the
//! crate's spring-based [`VbdColoring`](crate::vbd::coloring) input but is
//! derived purely from tet connectivity, so a volumetric mesh with no explicit
//! spring set can still be coloured or assembled.
//!
//! This is standard mesh connectivity; nothing here is derived from Unreal
//! Engine source.

use std::collections::HashSet;

/// Per-vertex one-ring adjacency for a tetrahedral mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetVertexAdjacency {
    /// `neighbours[v]` lists the vertices joined to vertex `v` by a tet edge,
    /// in ascending order with no duplicates and no self-reference. Vertices
    /// that no tet references have an empty list.
    pub neighbours: Vec<Vec<u32>>,
}

impl TetVertexAdjacency {
    /// Number of vertices in the graph (the `num_vertices` the graph was built
    /// against).
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.neighbours.len()
    }

    /// Number of distinct undirected edges (each counted once).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        let directed: usize = self.neighbours.iter().map(Vec::len).sum();
        directed / 2
    }

    /// The number of neighbours of vertex `v` (its degree), or zero when `v`
    /// is out of range.
    #[must_use]
    pub fn degree(&self, v: usize) -> usize {
        self.neighbours.get(v).map_or(0, Vec::len)
    }

    /// The neighbours of vertex `v`, or an empty slice when `v` is out of
    /// range.
    #[must_use]
    pub fn neighbours_of(&self, v: usize) -> &[u32] {
        self.neighbours.get(v).map_or(&[], Vec::as_slice)
    }

    /// The largest vertex degree in the graph (zero for an edgeless graph).
    #[must_use]
    pub fn max_degree(&self) -> usize {
        self.neighbours.iter().map(Vec::len).max().unwrap_or(0)
    }
}

/// The six undirected edges of a tet, each as a sorted `(low, high)` pair.
fn tet_edges(t: [u32; 4]) -> [(u32, u32); 6] {
    let edge = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    [
        edge(t[0], t[1]),
        edge(t[0], t[2]),
        edge(t[0], t[3]),
        edge(t[1], t[2]),
        edge(t[1], t[3]),
        edge(t[2], t[3]),
    ]
}

/// Builds the vertex one-ring adjacency of a tetrahedral mesh.
///
/// Returns `None` when `tets` is empty or any tet references a vertex index
/// `>= num_vertices`.
#[must_use]
pub fn build_tet_vertex_adjacency(
    num_vertices: usize,
    tets: &[[u32; 4]],
) -> Option<TetVertexAdjacency> {
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

    // Collect unique undirected edges, then expand into sorted adjacency lists.
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for &t in tets {
        for e in tet_edges(t) {
            // A tet never has a repeated vertex in a well-formed mesh, but guard
            // against a degenerate edge rather than inserting a self-loop.
            if e.0 != e.1 {
                edges.insert(e);
            }
        }
    }

    let mut neighbours = vec![Vec::new(); num_vertices];
    for (a, b) in edges {
        neighbours[a as usize].push(b);
        neighbours[b as usize].push(a);
    }
    for list in &mut neighbours {
        list.sort_unstable();
    }

    Some(TetVertexAdjacency { neighbours })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tetrahedralize::{tetrahedralize, TetMeshParams};
    use glam::Vec3;
    use std::collections::HashSet;

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
    fn single_tet_is_a_complete_graph_on_four() {
        let adj = build_tet_vertex_adjacency(4, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(adj.vertex_count(), 4);
        assert_eq!(adj.edge_count(), 6);
        for v in 0..4 {
            assert_eq!(adj.degree(v), 3);
            // Neighbours are exactly the other three vertices.
            let expected: Vec<u32> = (0..4u32).filter(|&x| x as usize != v).collect();
            assert_eq!(adj.neighbours_of(v), expected.as_slice());
        }
    }

    #[test]
    fn two_tets_share_an_edge_counted_once() {
        // Both tets use edge (0,1); apexes differ.
        let tets = vec![[0u32, 1, 2, 3], [0, 1, 4, 5]];
        let adj = build_tet_vertex_adjacency(6, &tets).unwrap();
        // Tet A edges: 01 02 03 12 13 23 ; Tet B: 01 04 05 14 15 45.
        // Union has 11 distinct edges (01 shared).
        assert_eq!(adj.edge_count(), 11);
        // Vertex 0 touches {1,2,3,4,5}.
        assert_eq!(adj.neighbours_of(0), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn adjacency_is_symmetric_sorted_and_simple() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let adj = build_tet_vertex_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        for u in 0..adj.vertex_count() {
            let ns = adj.neighbours_of(u);
            // Sorted, strictly increasing (no duplicates), no self-loop.
            for w in ns.windows(2) {
                assert!(w[0] < w[1], "neighbours of {u} not strictly sorted");
            }
            assert!(ns.iter().all(|&x| x as usize != u), "self-loop at {u}");
            // Symmetry.
            for &w in ns {
                assert!(
                    adj.neighbours_of(w as usize).contains(&(u as u32)),
                    "edge {u}-{w} missing back-reference"
                );
            }
        }
    }

    #[test]
    fn edge_count_matches_independent_edge_set() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let adj = build_tet_vertex_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();

        let mut reference: HashSet<(u32, u32)> = HashSet::new();
        for &t in &mesh.tets {
            let pairs = [
                (t[0], t[1]),
                (t[0], t[2]),
                (t[0], t[3]),
                (t[1], t[2]),
                (t[1], t[3]),
                (t[2], t[3]),
            ];
            for (a, b) in pairs {
                reference.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
        assert_eq!(adj.edge_count(), reference.len());
        assert!(adj.max_degree() > 0);
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = build_tet_vertex_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        let b = build_tet_vertex_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_or_out_of_range_returns_none() {
        assert!(build_tet_vertex_adjacency(0, &[]).is_none());
        assert!(build_tet_vertex_adjacency(4, &[]).is_none());
        assert!(build_tet_vertex_adjacency(4, &[[0u32, 1, 2, 9]]).is_none());
    }

    #[test]
    fn unreferenced_vertex_has_empty_ring() {
        // Vertex 4 is declared but no tet uses it.
        let adj = build_tet_vertex_adjacency(5, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(adj.vertex_count(), 5);
        assert_eq!(adj.degree(4), 0);
        assert!(adj.neighbours_of(4).is_empty());
    }
}
