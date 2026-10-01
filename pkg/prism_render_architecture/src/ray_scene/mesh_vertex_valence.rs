//! Vertex-valence and local topology statistics for the `CPU` golden path.
//!
//! The *valence* of a vertex is the number of distinct neighbours it is joined
//! to by an edge; its *triangle degree* is the number of faces that touch it.
//! These counts drive the quality heuristics every `AAA` geometry stage leans
//! on: remeshing aims for interior valence 6 (triangular) regularity,
//! subdivision schemes weight by valence, decimation protects high-valence or
//! boundary vertices, and tessellation balances fan sizes. A vertex is on the
//! surface *boundary* when any incident edge is used by a single triangle.
//!
//! [`vertex_valence`] walks the index buffer once, accumulating per-vertex
//! neighbour sets, incident-triangle counts, and per-edge face counts, then
//! derives valence, degree, boundary flags, and aggregate min / max / average
//! statistics. Unreferenced vertices report valence and degree `0`. The
//! computation is pure integer bookkeeping (the single average is `f64`),
//! touching no `f32` transcendental functions.

use std::collections::{HashMap, HashSet};

use super::triangle_mesh::TriangleMesh;

/// Per-vertex valence and local topology statistics for a [`TriangleMesh`].
#[derive(Clone, Debug)]
pub struct VertexValence {
    /// Number of distinct edge-neighbours of each vertex (its valence).
    valences: Vec<u32>,
    /// Number of triangles incident to each vertex (its triangle degree).
    degrees: Vec<u32>,
    /// Whether each vertex lies on an open boundary edge.
    boundary: Vec<bool>,
}

impl VertexValence {
    /// Returns the number of vertices described (matching the source mesh's
    /// vertex count).
    pub fn len(&self) -> usize {
        self.valences.len()
    }

    /// Returns whether the mesh had no vertices.
    pub fn is_empty(&self) -> bool {
        self.valences.is_empty()
    }

    /// Returns the valence (distinct edge-neighbour count) of vertex `v`, or
    /// `None` when `v` is out of range.
    pub fn valence(&self, v: u32) -> Option<u32> {
        self.valences.get(v as usize).copied()
    }

    /// Returns the triangle degree (incident-face count) of vertex `v`, or
    /// `None` when `v` is out of range.
    pub fn triangle_degree(&self, v: u32) -> Option<u32> {
        self.degrees.get(v as usize).copied()
    }

    /// Returns whether vertex `v` lies on an open boundary edge, or `None`
    /// when `v` is out of range.
    pub fn is_boundary(&self, v: u32) -> Option<bool> {
        self.boundary.get(v as usize).copied()
    }

    /// Returns the per-vertex valences.
    pub fn valences(&self) -> &[u32] {
        &self.valences
    }

    /// Returns the per-vertex triangle degrees.
    pub fn degrees(&self) -> &[u32] {
        &self.degrees
    }

    /// Returns the sorted indices of all boundary vertices.
    pub fn boundary_vertices(&self) -> Vec<u32> {
        self.boundary
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i as u32) } else { None })
            .collect()
    }

    /// Returns the minimum valence over referenced vertices (valence `> 0`), or
    /// `None` when no vertex is referenced.
    pub fn min_valence(&self) -> Option<u32> {
        self.valences.iter().copied().filter(|&v| v > 0).min()
    }

    /// Returns the maximum valence over all vertices, or `None` for an empty
    /// mesh.
    pub fn max_valence(&self) -> Option<u32> {
        self.valences.iter().copied().max()
    }

    /// Returns the average valence over referenced vertices (valence `> 0`), or
    /// `None` when no vertex is referenced.
    pub fn average_valence(&self) -> Option<f64> {
        let mut sum = 0u64;
        let mut count = 0u64;
        for &v in &self.valences {
            if v > 0 {
                sum += u64::from(v);
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum as f64 / count as f64)
        }
    }

    /// Returns the number of interior (non-boundary) vertices whose valence
    /// equals `target` — e.g. `target = 6` counts the regular vertices of a
    /// triangle mesh.
    pub fn regular_count(&self, target: u32) -> usize {
        self.valences
            .iter()
            .zip(&self.boundary)
            .filter(|&(&v, &b)| !b && v == target)
            .count()
    }
}

/// Computes per-vertex valence, triangle degree, boundary flags, and aggregate
/// statistics for `mesh` in a single pass over its index buffer.
pub fn vertex_valence(mesh: &TriangleMesh) -> VertexValence {
    let vertex_count = mesh.vertex_count();
    let indices = mesh.indices();

    let mut neighbours: Vec<HashSet<u32>> = vec![HashSet::new(); vertex_count];
    let mut degrees: Vec<u32> = vec![0; vertex_count];
    let mut edge_faces: HashMap<(u32, u32), u32> = HashMap::new();

    for tri in indices {
        let [a, b, c] = *tri;
        for &v in &[a, b, c] {
            degrees[v as usize] += 1;
        }
        for &(u, w) in &[(a, b), (b, c), (c, a)] {
            if u != w {
                neighbours[u as usize].insert(w);
                neighbours[w as usize].insert(u);
            }
            *edge_faces.entry(sorted_pair(u, w)).or_insert(0) += 1;
        }
    }

    let valences: Vec<u32> = neighbours.iter().map(|n| n.len() as u32).collect();

    let mut boundary = vec![false; vertex_count];
    for (&(u, w), &faces) in &edge_faces {
        if faces == 1 {
            boundary[u as usize] = true;
            boundary[w as usize] = true;
        }
    }

    VertexValence { valences, degrees, boundary }
}

/// Returns the sorted `(min, max)` endpoint pair keying a shared edge.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
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

    /// A closed regular octahedron: 6 vertices, 8 faces, every vertex valence
    /// 4 and interior (no boundary).
    fn octahedron() -> TriangleMesh {
        TriangleMesh::new(
            vec![
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            Vec::new(),
            Vec::new(),
            vec![
                [0, 2, 4],
                [2, 1, 4],
                [1, 3, 4],
                [3, 0, 4],
                [2, 0, 5],
                [1, 2, 5],
                [3, 1, 5],
                [0, 3, 5],
            ],
        )
        .unwrap()
    }

    #[test]
    fn quad_valences_and_degrees() {
        let vv = vertex_valence(&flat_quad());
        assert_eq!(vv.len(), 4);
        // Corner 0: neighbours {1, 2} -> valence 2, one triangle.
        assert_eq!(vv.valence(0), Some(2));
        assert_eq!(vv.triangle_degree(0), Some(1));
        // Shared-diagonal corner 1: neighbours {0, 2, 3} -> valence 3, two tris.
        assert_eq!(vv.valence(1), Some(3));
        assert_eq!(vv.triangle_degree(1), Some(2));
    }

    #[test]
    fn quad_boundary_detection() {
        let vv = vertex_valence(&flat_quad());
        // Every corner of a single quad touches a rim edge.
        for v in 0..4 {
            assert_eq!(vv.is_boundary(v), Some(true));
        }
        assert_eq!(vv.boundary_vertices(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn octahedron_is_closed_valence_four() {
        let vv = vertex_valence(&octahedron());
        assert_eq!(vv.len(), 6);
        for v in 0..6 {
            assert_eq!(vv.valence(v), Some(4));
            assert_eq!(vv.triangle_degree(v), Some(4));
            assert_eq!(vv.is_boundary(v), Some(false));
        }
        assert!(vv.boundary_vertices().is_empty());
    }

    #[test]
    fn min_max_and_average_over_referenced() {
        let vv = vertex_valence(&flat_quad());
        // Valences: v0=2, v1=3, v2=3, v3=2 -> min 2, max 3, avg 2.5.
        assert_eq!(vv.min_valence(), Some(2));
        assert_eq!(vv.max_valence(), Some(3));
        assert_eq!(vv.average_valence(), Some(2.5));
    }

    #[test]
    fn regular_count_targets_interior_valence() {
        let vv = vertex_valence(&octahedron());
        // All six interior vertices have valence 4.
        assert_eq!(vv.regular_count(4), 6);
        assert_eq!(vv.regular_count(6), 0);
    }

    #[test]
    fn quad_has_no_interior_regular_vertices() {
        let vv = vertex_valence(&flat_quad());
        // Every vertex is on the boundary, so none counts as interior-regular.
        assert_eq!(vv.regular_count(2), 0);
        assert_eq!(vv.regular_count(3), 0);
    }

    #[test]
    fn unreferenced_vertex_reports_zero() {
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
        let vv = vertex_valence(&mesh);
        assert_eq!(vv.valence(3), Some(0));
        assert_eq!(vv.triangle_degree(3), Some(0));
        assert_eq!(vv.is_boundary(3), Some(false));
        // Average ignores the unreferenced vertex (valence 0).
        assert_eq!(vv.average_valence(), Some(2.0));
        assert_eq!(vv.min_valence(), Some(2));
    }

    #[test]
    fn out_of_range_queries_return_none() {
        let vv = vertex_valence(&flat_quad());
        assert_eq!(vv.valence(99), None);
        assert_eq!(vv.triangle_degree(99), None);
        assert_eq!(vv.is_boundary(99), None);
    }

    #[test]
    fn empty_mesh_has_no_stats() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0]],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let vv = vertex_valence(&mesh);
        assert!(!vv.is_empty());
        assert_eq!(vv.len(), 1);
        // The single vertex is unreferenced.
        assert_eq!(vv.valence(0), Some(0));
        assert_eq!(vv.min_valence(), None);
        assert_eq!(vv.max_valence(), Some(0));
        assert_eq!(vv.average_valence(), None);
    }

    #[test]
    fn truly_empty_mesh_is_empty() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let vv = vertex_valence(&mesh);
        assert!(vv.is_empty());
        assert_eq!(vv.len(), 0);
        assert_eq!(vv.max_valence(), None);
    }
}
