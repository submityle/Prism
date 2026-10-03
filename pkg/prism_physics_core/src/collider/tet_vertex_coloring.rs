//! Greedy colouring of the tetrahedral vertex graph for parallel node sweeps.
//!
//! Vertex-based solvers (vertex-block descent, position-based node relaxation,
//! Jacobi/Gauss-Seidel over mesh nodes) update one vertex from the current
//! positions of its one-ring neighbours. Two vertices joined by a tet edge
//! therefore conflict and cannot be relaxed simultaneously. Partitioning the
//! vertices into colours such that no edge is monochromatic removes the hazard:
//! every vertex in a colour has its neighbours in other colours, so a whole
//! colour can be relaxed in parallel. Applying colours one after another is a
//! valid Gauss-Seidel sweep whose result is independent of the within-colour
//! order.
//!
//! The colouring is a deterministic greedy pass over the vertex graph produced
//! by [`crate::collider::tet_vertex_adjacency`]: vertices are visited in
//! ascending index order and each takes the lowest colour not used by an
//! already-coloured neighbour. This is the node-level analogue of the element
//! colouring in [`crate::collider::tet_coloring`] and mirrors the spring-driven
//! [`VbdColoring`](crate::vbd::coloring), but is derived purely from tet
//! connectivity.
//!
//! Greedy graph colouring for parallel Gauss-Seidel is a standard, publicly
//! documented technique; nothing here is derived from Unreal Engine source.

use super::tet_vertex_adjacency::{build_tet_vertex_adjacency, TetVertexAdjacency};

/// A colour assignment over the vertices of a tetrahedral mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TetVertexColoring {
    /// `colours[v]` is the colour assigned to vertex `v`. No two edge-adjacent
    /// vertices share a colour.
    pub colours: Vec<u32>,
}

impl TetVertexColoring {
    /// Number of vertices coloured.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.colours.len()
    }

    /// Number of distinct colours used (largest colour index plus one, or zero
    /// when there are no vertices).
    #[must_use]
    pub fn colour_count(&self) -> usize {
        self.colours
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1)
    }

    /// Groups the vertex indices by colour. `groups()[c]` lists every vertex
    /// assigned colour `c`, in ascending index order. The groups partition
    /// `0..vertex_count()`.
    #[must_use]
    pub fn groups(&self) -> Vec<Vec<u32>> {
        let mut groups = vec![Vec::new(); self.colour_count()];
        for (v, &c) in self.colours.iter().enumerate() {
            groups[c as usize].push(v as u32);
        }
        groups
    }
}

/// Colours the vertices of the given one-ring graph with a deterministic greedy
/// pass.
///
/// Edge-adjacent vertices never receive the same colour. Uses at most
/// `max_degree + 1` colours.
#[must_use]
pub fn colour_tet_vertices(adjacency: &TetVertexAdjacency) -> TetVertexColoring {
    let n = adjacency.vertex_count();
    let mut colours = vec![u32::MAX; n];
    let mut used: Vec<bool> = Vec::new();

    for v in 0..n {
        used.clear();
        for &nb in adjacency.neighbours_of(v) {
            let c = colours[nb as usize];
            if c != u32::MAX {
                let ci = c as usize;
                if ci >= used.len() {
                    used.resize(ci + 1, false);
                }
                used[ci] = true;
            }
        }

        let mut chosen = 0usize;
        while chosen < used.len() && used[chosen] {
            chosen += 1;
        }
        colours[v] = chosen as u32;
    }

    TetVertexColoring { colours }
}

/// Builds the vertex graph of the mesh and colours it in one call.
///
/// Returns `None` under the same conditions as
/// [`build_tet_vertex_adjacency`]: empty `tets` or an out-of-range vertex
/// index.
#[must_use]
pub fn colour_tet_vertex_graph(
    num_vertices: usize,
    tets: &[[u32; 4]],
) -> Option<TetVertexColoring> {
    let adjacency = build_tet_vertex_adjacency(num_vertices, tets)?;
    Some(colour_tet_vertices(&adjacency))
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

    #[test]
    fn single_tet_needs_four_colours() {
        // K4 is 4-chromatic: every vertex is adjacent to the other three.
        let col = colour_tet_vertex_graph(4, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(col.vertex_count(), 4);
        assert_eq!(col.colour_count(), 4);
        assert_eq!(col.colours, vec![0, 1, 2, 3]);
    }

    #[test]
    fn colouring_is_valid_on_cube() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let adj = build_tet_vertex_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        let col = colour_tet_vertices(&adj);
        for u in 0..adj.vertex_count() {
            for &w in adj.neighbours_of(u) {
                assert_ne!(
                    col.colours[u], col.colours[w as usize],
                    "edge {u}-{w} is monochromatic ({})",
                    col.colours[u]
                );
            }
        }
    }

    #[test]
    fn colour_count_bounded_by_max_degree_plus_one() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(12)).unwrap();
        let adj = build_tet_vertex_adjacency(mesh.vertices.len(), &mesh.tets).unwrap();
        let col = colour_tet_vertices(&adj);
        assert!(
            col.colour_count() <= adj.max_degree() + 1,
            "greedy used {} colours for max degree {}",
            col.colour_count(),
            adj.max_degree()
        );
    }

    #[test]
    fn groups_partition_all_vertices() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let col = colour_tet_vertex_graph(mesh.vertices.len(), &mesh.tets).unwrap();
        let groups = col.groups();
        let mut seen = vec![false; col.vertex_count()];
        let mut total = 0usize;
        for (c, g) in groups.iter().enumerate() {
            for &vtx in g {
                assert!(!seen[vtx as usize], "vertex {vtx} in two groups");
                seen[vtx as usize] = true;
                assert_eq!(col.colours[vtx as usize] as usize, c);
                total += 1;
            }
        }
        assert_eq!(total, col.vertex_count());
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn unreferenced_vertex_takes_colour_zero() {
        // Vertex 4 has an empty one-ring, so greedy assigns colour 0.
        let col = colour_tet_vertex_graph(5, &[[0u32, 1, 2, 3]]).unwrap();
        assert_eq!(col.colours[4], 0);
    }

    #[test]
    fn is_deterministic() {
        let (v, i) = cube_surface(1.0);
        let mesh = tetrahedralize(&v, &i, &TetMeshParams::new(10)).unwrap();
        let a = colour_tet_vertex_graph(mesh.vertices.len(), &mesh.tets).unwrap();
        let b = colour_tet_vertex_graph(mesh.vertices.len(), &mesh.tets).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn empty_or_invalid_returns_none() {
        assert!(colour_tet_vertex_graph(0, &[]).is_none());
        assert!(colour_tet_vertex_graph(4, &[[0u32, 1, 2, 9]]).is_none());
    }
}
