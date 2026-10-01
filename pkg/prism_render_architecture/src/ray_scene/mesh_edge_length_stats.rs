//! Undirected edge-length statistics for the `CPU` golden path.
//!
//! Isotropic remeshing — the backbone of `AAA` geometry cleanup and adaptive
//! tessellation — is driven by a *target edge length*: edges longer than
//! `4/3 * target` are split, edges shorter than `4/5 * target` are collapsed,
//! and the mesh converges toward a uniform triangle size. Choosing and
//! monitoring that target needs the mesh's current edge-length distribution:
//! its min, max, mean, and how many edges fall on either side of a candidate
//! length.
//!
//! [`edge_length_stats`] gathers every undirected edge exactly once (so an
//! interior edge shared by two triangles is measured a single time), records
//! its Euclidean length, and exposes per-edge data in sorted endpoint order
//! plus aggregate statistics. Each length is one `sqrt` of a dot product; the
//! mean is accumulated in `f64` for stability. No `f32` transcendental
//! function is used.

use std::collections::HashSet;

use super::triangle_mesh::TriangleMesh;

/// Undirected edge-length statistics for a [`TriangleMesh`].
#[derive(Clone, Debug)]
pub struct EdgeLengthStats {
    /// Each unique undirected edge as `((min, max), length)`, sorted by
    /// endpoint pair.
    edges: Vec<((u32, u32), f32)>,
}

impl EdgeLengthStats {
    /// Returns the number of unique undirected edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns whether the mesh has no edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Returns the unique edges as `((min, max), length)`, sorted by endpoint
    /// pair.
    pub fn edges(&self) -> &[((u32, u32), f32)] {
        &self.edges
    }

    /// Returns the length of the undirected edge `{a, b}`, or `None` when no
    /// such edge exists.
    pub fn length(&self, a: u32, b: u32) -> Option<f32> {
        let key = sorted_pair(a, b);
        self.edges
            .binary_search_by(|&(edge, _)| edge.cmp(&key))
            .ok()
            .map(|i| self.edges[i].1)
    }

    /// Returns the shortest edge length, or `None` for a mesh with no edges.
    pub fn min_length(&self) -> Option<f32> {
        self.edges.iter().map(|&(_, l)| l).reduce(f32::min)
    }

    /// Returns the longest edge length, or `None` for a mesh with no edges.
    pub fn max_length(&self) -> Option<f32> {
        self.edges.iter().map(|&(_, l)| l).reduce(f32::max)
    }

    /// Returns the total length of all unique edges.
    pub fn total_length(&self) -> f32 {
        self.edges.iter().map(|&(_, l)| f64::from(l)).sum::<f64>() as f32
    }

    /// Returns the mean edge length, or `None` for a mesh with no edges.
    pub fn mean_length(&self) -> Option<f32> {
        if self.edges.is_empty() {
            return None;
        }
        let sum: f64 = self.edges.iter().map(|&(_, l)| f64::from(l)).sum();
        Some((sum / self.edges.len() as f64) as f32)
    }

    /// Returns how many edges are strictly shorter than `threshold` — the
    /// collapse candidates for a target of that length.
    pub fn count_shorter_than(&self, threshold: f32) -> usize {
        self.edges.iter().filter(|&&(_, l)| l < threshold).count()
    }

    /// Returns how many edges are strictly longer than `threshold` — the split
    /// candidates for a target of that length.
    pub fn count_longer_than(&self, threshold: f32) -> usize {
        self.edges.iter().filter(|&&(_, l)| l > threshold).count()
    }
}

/// Computes per-edge lengths and aggregate statistics for `mesh`, measuring
/// each undirected edge exactly once.
pub fn edge_length_stats(mesh: &TriangleMesh) -> EdgeLengthStats {
    let positions = mesh.positions();

    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut edges: Vec<((u32, u32), f32)> = Vec::new();
    for tri in mesh.indices() {
        let [a, b, c] = *tri;
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            let key = sorted_pair(u, v);
            if seen.insert(key) {
                let length = distance(positions[u as usize], positions[v as usize]);
                edges.push((key, length));
            }
        }
    }

    edges.sort_by_key(|a| a.0);
    EdgeLengthStats { edges }
}

/// Returns the sorted `(min, max)` endpoint pair keying a shared edge.
fn sorted_pair(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Returns the Euclidean distance between two points.
fn distance(p: [f32; 3], q: [f32; 3]) -> f32 {
    let dx = p[0] - q[0];
    let dy = p[1] - q[1];
    let dz = p[2] - q[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
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

    #[test]
    fn quad_has_five_unique_edges() {
        let stats = edge_length_stats(&flat_quad());
        // Four unit rim edges plus one shared diagonal of length sqrt(2).
        assert_eq!(stats.edge_count(), 5);
    }

    #[test]
    fn shared_edge_counted_once() {
        let stats = edge_length_stats(&flat_quad());
        // The diagonal (1, 2) is shared by both triangles but listed once.
        let diagonal = stats.edges().iter().filter(|&&(e, _)| e == (1, 2)).count();
        assert_eq!(diagonal, 1);
    }

    #[test]
    fn min_max_lengths() {
        let stats = edge_length_stats(&flat_quad());
        assert!((stats.min_length().unwrap() - 1.0).abs() < 1e-6);
        assert!((stats.max_length().unwrap() - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn length_lookup_by_endpoints() {
        let stats = edge_length_stats(&flat_quad());
        // Order-independent lookup of the diagonal.
        assert!((stats.length(1, 2).unwrap() - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!((stats.length(2, 1).unwrap() - 2.0_f32.sqrt()).abs() < 1e-6);
        // A rim edge.
        assert!((stats.length(0, 1).unwrap() - 1.0).abs() < 1e-6);
        // A non-existent edge.
        assert_eq!(stats.length(0, 3), None);
    }

    #[test]
    fn total_and_mean_length() {
        let stats = edge_length_stats(&flat_quad());
        let expected_total = 4.0 + 2.0_f32.sqrt();
        assert!((stats.total_length() - expected_total).abs() < 1e-5);
        let expected_mean = expected_total / 5.0;
        assert!((stats.mean_length().unwrap() - expected_mean).abs() < 1e-5);
    }

    #[test]
    fn split_and_collapse_candidate_counts() {
        let stats = edge_length_stats(&flat_quad());
        // Four edges of length 1 are shorter than 1.2.
        assert_eq!(stats.count_shorter_than(1.2), 4);
        // Only the sqrt(2) diagonal is longer than 1.2.
        assert_eq!(stats.count_longer_than(1.2), 1);
        // Nothing is shorter than the minimum.
        assert_eq!(stats.count_shorter_than(1.0), 0);
    }

    #[test]
    fn single_triangle_has_three_edges() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let stats = edge_length_stats(&mesh);
        assert_eq!(stats.edge_count(), 3);
        // 3-4-5 right triangle.
        assert!((stats.length(0, 1).unwrap() - 3.0).abs() < 1e-6);
        assert!((stats.length(0, 2).unwrap() - 4.0).abs() < 1e-6);
        assert!((stats.length(1, 2).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn degenerate_zero_length_edge() {
        // Two coincident vertices make a zero-length edge.
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            Vec::new(),
            Vec::new(),
            vec![[0, 1, 2]],
        )
        .unwrap();
        let stats = edge_length_stats(&mesh);
        assert_eq!(stats.length(0, 1), Some(0.0));
        assert_eq!(stats.min_length(), Some(0.0));
    }

    #[test]
    fn edges_are_sorted_by_endpoint() {
        let stats = edge_length_stats(&flat_quad());
        let keys: Vec<(u32, u32)> = stats.edges().iter().map(|&(e, _)| e).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn empty_mesh_has_no_edges() {
        let mesh = TriangleMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let stats = edge_length_stats(&mesh);
        assert!(stats.is_empty());
        assert_eq!(stats.edge_count(), 0);
        assert_eq!(stats.min_length(), None);
        assert_eq!(stats.max_length(), None);
        assert_eq!(stats.mean_length(), None);
        assert_eq!(stats.total_length(), 0.0);
    }
}
