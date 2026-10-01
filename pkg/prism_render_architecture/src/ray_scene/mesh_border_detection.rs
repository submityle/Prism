//! Open-boundary and non-manifold edge detection for the `CPU` golden path.
//!
//! A triangle mesh is *watertight* when every edge is shared by exactly two
//! faces. Edges touched by a single face are **open boundaries** — the rim of a
//! hole or the border of an open patch — and edges touched by three or more
//! faces are **non-manifold**, a topological defect that breaks most geometry
//! processing. Knowing where these edges are is a prerequisite for hole
//! filling, `UV`-seam handling, boundary-aware smoothing/decimation, shell
//! thickening, and mesh-sanity validation.
//!
//! This module classifies every edge by its incident-face count and, for the
//! open boundaries, stitches them into **oriented loops**. Each triangle is
//! wound counter-clockwise, so its three directed edges `a→b→c→a` give every
//! boundary edge a direction; following those directions walks each hole rim as
//! an ordered vertex ring, which is exactly what a fan or ear-clip hole fill
//! consumes.
//!
//! Everything here is integer index bookkeeping over the connectivity — no
//! floating-point arithmetic at all — so it is trivially within the golden-path
//! float policy.

use std::collections::HashMap;

use super::triangle_mesh::TriangleMesh;

/// The boundary and non-manifold structure extracted from a mesh.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MeshBorders {
    /// Undirected open-boundary edges (incident to exactly one face), each
    /// stored as `[min, max]` vertex indices and de-duplicated.
    boundary_edges: Vec<[u32; 2]>,
    /// Oriented boundary loops: each inner `Vec` lists the vertices of one hole
    /// rim in winding order. A closed loop repeats neither endpoint; an open
    /// chain (from inconsistent winding or a dangling edge) is still returned.
    loops: Vec<Vec<u32>>,
    /// Undirected non-manifold edges (incident to three or more faces), each
    /// stored as `[min, max]` vertex indices.
    non_manifold_edges: Vec<[u32; 2]>,
}

impl MeshBorders {
    /// The de-duplicated open-boundary edges as `[min, max]` index pairs.
    #[must_use]
    pub fn boundary_edges(&self) -> &[[u32; 2]] {
        &self.boundary_edges
    }

    /// The oriented boundary loops; each entry is one hole rim in winding
    /// order.
    #[must_use]
    pub fn loops(&self) -> &[Vec<u32>] {
        &self.loops
    }

    /// The non-manifold edges (three or more incident faces).
    #[must_use]
    pub fn non_manifold_edges(&self) -> &[[u32; 2]] {
        &self.non_manifold_edges
    }

    /// `true` when the mesh has no open boundaries and no non-manifold edges —
    /// i.e. every edge is shared by exactly two faces.
    #[must_use]
    pub fn is_watertight(&self) -> bool {
        self.boundary_edges.is_empty() && self.non_manifold_edges.is_empty()
    }

    /// The number of distinct boundary loops (holes/open borders).
    #[must_use]
    pub fn loop_count(&self) -> usize {
        self.loops.len()
    }
}

/// Classifies every edge of `mesh` by incident-face count and assembles the
/// open boundaries into oriented loops.
///
/// Degenerate faces (a repeated vertex index) are skipped. Edge direction is
/// taken from each triangle's counter-clockwise winding, so boundary loops come
/// back in winding order ready for hole filling.
#[must_use]
pub fn detect_borders(mesh: &TriangleMesh) -> MeshBorders {
    // Undirected incident-face count per edge.
    let mut undirected: HashMap<(u32, u32), u32> = HashMap::new();
    // Directed edge presence, used to orient boundary loops.
    let mut directed: HashMap<(u32, u32), u32> = HashMap::new();

    for tri in mesh.indices() {
        let [a, b, c] = *tri;
        if a == b || b == c || a == c {
            continue;
        }
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            *undirected.entry(key).or_insert(0) += 1;
            *directed.entry((u, v)).or_insert(0) += 1;
        }
    }

    let mut boundary_edges = Vec::new();
    let mut non_manifold_edges = Vec::new();
    for (&(u, v), &count) in &undirected {
        match count {
            1 => boundary_edges.push([u, v]),
            n if n >= 3 => non_manifold_edges.push([u, v]),
            _ => {}
        }
    }
    boundary_edges.sort_unstable();
    non_manifold_edges.sort_unstable();

    let loops = assemble_loops(&boundary_edges, &directed);

    MeshBorders {
        boundary_edges,
        loops,
        non_manifold_edges,
    }
}

/// Walks the oriented boundary edges into loops. `directed` records which
/// directions actually occur in the winding; each undirected boundary edge is
/// emitted once in its occurring direction, and loops are traced by following
/// `start → end` links.
fn assemble_loops(
    boundary_edges: &[[u32; 2]],
    directed: &HashMap<(u32, u32), u32>,
) -> Vec<Vec<u32>> {
    // Build the directed boundary edge list and a start→successors adjacency.
    let mut successors: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut remaining = 0usize;
    for edge in boundary_edges {
        let [a, b] = *edge;
        // Pick whichever direction the winding actually produced.
        let (from, to) = if directed.contains_key(&(a, b)) {
            (a, b)
        } else {
            (b, a)
        };
        successors.entry(from).or_default().push(to);
        remaining += 1;
    }

    let mut loops = Vec::new();
    while remaining > 0 {
        // Find any vertex that still has an unused outgoing boundary edge.
        let Some((&start, _)) = successors.iter().find(|(_, outs)| !outs.is_empty()) else {
            break;
        };
        let mut chain = vec![start];
        let mut current = start;
        while let Some(outs) = successors.get_mut(&current) {
            let Some(next) = outs.pop() else {
                break;
            };
            remaining -= 1;
            if next == start {
                // Closed the loop; do not repeat the start vertex.
                break;
            }
            chain.push(next);
            current = next;
        }
        loops.push(chain);
    }

    loops.sort();
    loops
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single triangle: all three edges are open boundaries forming one loop.
    fn single_triangle() -> TriangleMesh {
        TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap()
    }

    /// A closed regular tetrahedron: four faces, every edge shared twice.
    fn tetrahedron() -> TriangleMesh {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // Outward-consistent winding.
        let indices = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    /// A flat `grid × grid` quad lattice (open disk) on `z = 0`.
    fn planar_grid(grid: u32) -> TriangleMesh {
        let stride = grid + 1;
        let mut positions = Vec::new();
        for j in 0..=grid {
            for i in 0..=grid {
                positions.push([i as f32, j as f32, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for j in 0..grid {
            for i in 0..grid {
                let a = j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.push([a, b, c]);
                indices.push([b, d, c]);
            }
        }
        TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap()
    }

    #[test]
    fn single_triangle_has_one_three_edge_loop() {
        let borders = detect_borders(&single_triangle());
        assert_eq!(borders.boundary_edges().len(), 3);
        assert_eq!(borders.loop_count(), 1);
        assert_eq!(borders.loops()[0].len(), 3);
        assert!(!borders.is_watertight());
        assert!(borders.non_manifold_edges().is_empty());
    }

    #[test]
    fn tetrahedron_is_watertight() {
        let borders = detect_borders(&tetrahedron());
        assert!(borders.boundary_edges().is_empty());
        assert!(borders.non_manifold_edges().is_empty());
        assert_eq!(borders.loop_count(), 0);
        assert!(borders.is_watertight());
    }

    #[test]
    fn planar_grid_has_single_rectangular_loop() {
        let grid = 4;
        let borders = detect_borders(&planar_grid(grid));
        // A grid-by-grid open quad has 4*grid boundary edges in one loop.
        assert_eq!(borders.boundary_edges().len() as u32, 4 * grid);
        assert_eq!(borders.loop_count(), 1);
        assert_eq!(borders.loops()[0].len() as u32, 4 * grid);
        assert!(!borders.is_watertight());
    }

    #[test]
    fn boundary_loop_is_connected_cycle() {
        // Every consecutive pair in the loop must be an actual boundary edge,
        // and the last must connect back to the first.
        let borders = detect_borders(&planar_grid(3));
        let loop0 = &borders.loops()[0];
        let edge_set: std::collections::HashSet<(u32, u32)> = borders
            .boundary_edges()
            .iter()
            .map(|e| (e[0], e[1]))
            .collect();
        let n = loop0.len();
        for k in 0..n {
            let a = loop0[k];
            let b = loop0[(k + 1) % n];
            let key = if a < b { (a, b) } else { (b, a) };
            assert!(edge_set.contains(&key), "missing edge {a}->{b}");
        }
    }

    #[test]
    fn two_holes_yield_two_loops() {
        // Two disjoint triangles → two independent 3-edge boundary loops.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap();
        let borders = detect_borders(&mesh);
        assert_eq!(borders.loop_count(), 2);
        assert_eq!(borders.boundary_edges().len(), 6);
    }

    #[test]
    fn non_manifold_edge_is_flagged() {
        // Three triangles fanning around a shared edge (0,1): that edge has
        // three incident faces and must be reported non-manifold.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        let mesh = TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap();
        let borders = detect_borders(&mesh);
        assert!(
            borders.non_manifold_edges().contains(&[0, 1]),
            "shared edge (0,1) not flagged: {:?}",
            borders.non_manifold_edges()
        );
        assert!(!borders.is_watertight());
    }

    #[test]
    fn degenerate_faces_are_ignored() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![[0, 1, 2], [0, 0, 1]];
        let mesh = TriangleMesh::new(positions, Vec::new(), Vec::new(), indices).unwrap();
        let borders = detect_borders(&mesh);
        // Only the valid triangle contributes: one 3-edge loop.
        assert_eq!(borders.loop_count(), 1);
        assert_eq!(borders.boundary_edges().len(), 3);
    }

    #[test]
    fn empty_mesh_has_no_borders() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let borders = detect_borders(&mesh);
        assert!(borders.is_watertight());
        assert_eq!(borders.loop_count(), 0);
    }
}
