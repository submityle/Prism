//! Shortest paths and distance fields along the edges of a triangle mesh.
//!
//! A cooker often needs surface distances rather than straight-line distances:
//! routing a seam, measuring how far apart two features are across the hull,
//! or growing a region outward from a set of seed vertices. The exact geodesic
//! distance on a polyhedron is expensive to compute, but the shortest path
//! *along mesh edges* is a close, cheap, and well-defined upper bound that is
//! sufficient for cooking decisions. This module builds the mesh's edge graph
//! once and answers both single-pair shortest-path and multi-source
//! distance-field queries over it with Dijkstra's algorithm.
//!
//! Edge weights are Euclidean edge lengths, so the graph distance equals the
//! true geodesic distance whenever the geodesic happens to run along edges
//! (and over-estimates it otherwise). Results are deterministic: ties in the
//! priority queue are broken by vertex index.
//!
//! This is pure triangle-soup geometry with no coupling to the collision
//! pipeline, and nothing here is derived from Unreal Engine source.

use alloc::collections::BinaryHeap;
use core::cmp::Ordering;

use glam::Vec3;

/// A shortest path along mesh edges, as a vertex sequence and its length.
#[derive(Clone, Debug, PartialEq)]
pub struct GeodesicPath {
    /// Vertex indices from the start to the goal, inclusive of both.
    pub vertices: Vec<u32>,
    /// Total Euclidean length of the path along its edges.
    pub length: f32,
}

impl GeodesicPath {
    /// Number of edges traversed (one less than the number of vertices).
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }
}

/// The undirected edge graph of a triangle mesh, with Euclidean edge weights.
///
/// Built once with [`MeshEdgeGraph::build`], it answers repeated
/// [`shortest_path`](MeshEdgeGraph::shortest_path) and
/// [`distance_field`](MeshEdgeGraph::distance_field) queries without rebuilding
/// the adjacency.
#[derive(Clone, Debug)]
pub struct MeshEdgeGraph {
    /// Per-vertex adjacency: `neighbours[v]` lists `(neighbour, edge_length)`,
    /// sorted by neighbour index and de-duplicated.
    neighbours: Vec<Vec<(u32, f32)>>,
}

/// A Dijkstra priority-queue entry ordered as a min-heap by distance, with the
/// vertex index as a deterministic tie-breaker.
#[derive(Clone, Copy, Debug)]
struct Frontier {
    distance: f32,
    vertex: u32,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.vertex == other.vertex
    }
}

impl Eq for Frontier {}

impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the distance comparison so `BinaryHeap` (a max-heap) pops the
        // smallest distance first; break ties on vertex index (also reversed)
        // for determinism.
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.vertex.cmp(&self.vertex))
    }
}

impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl MeshEdgeGraph {
    /// Builds the edge graph of a triangle mesh.
    ///
    /// Returns `None` when the mesh has no vertices or no triangles, or when a
    /// triangle references a vertex out of range.
    #[must_use]
    pub fn build(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<Self> {
        if vertices.is_empty() || indices.is_empty() {
            return None;
        }
        let n = vertices.len();
        let mut neighbours: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
        for tri in indices {
            for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let (ai, bi) = (a as usize, b as usize);
                if ai >= n || bi >= n {
                    return None;
                }
                if ai == bi {
                    continue;
                }
                let w = (vertices[ai] - vertices[bi]).length();
                neighbours[ai].push((b, w));
                neighbours[bi].push((a, w));
            }
        }
        for list in &mut neighbours {
            list.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)));
            list.dedup_by_key(|e| e.0);
        }
        Some(Self { neighbours })
    }

    /// Number of vertices in the graph.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.neighbours.len()
    }

    /// The shortest path along mesh edges from `start` to `goal`.
    ///
    /// Returns `None` when either index is out of range or when `goal` is not
    /// reachable from `start`. A zero-length path is returned when
    /// `start == goal`.
    #[must_use]
    pub fn shortest_path(&self, start: u32, goal: u32) -> Option<GeodesicPath> {
        let n = self.neighbours.len();
        let (s, g) = (start as usize, goal as usize);
        if s >= n || g >= n {
            return None;
        }
        if start == goal {
            return Some(GeodesicPath {
                vertices: vec![start],
                length: 0.0,
            });
        }

        let mut dist = vec![f32::INFINITY; n];
        let mut prev = vec![u32::MAX; n];
        let mut visited = vec![false; n];
        let mut heap = BinaryHeap::new();
        dist[s] = 0.0;
        heap.push(Frontier {
            distance: 0.0,
            vertex: start,
        });

        while let Some(Frontier { distance, vertex }) = heap.pop() {
            let v = vertex as usize;
            if visited[v] {
                continue;
            }
            visited[v] = true;
            if vertex == goal {
                break;
            }
            for &(nb, w) in &self.neighbours[v] {
                let nbi = nb as usize;
                if visited[nbi] {
                    continue;
                }
                let candidate = distance + w;
                if candidate < dist[nbi] {
                    dist[nbi] = candidate;
                    prev[nbi] = vertex;
                    heap.push(Frontier {
                        distance: candidate,
                        vertex: nb,
                    });
                }
            }
        }

        if !dist[g].is_finite() {
            return None;
        }

        // Walk the predecessor chain back from the goal and reverse it.
        let mut path = vec![goal];
        let mut cursor = goal;
        while cursor != start {
            let p = prev[cursor as usize];
            if p == u32::MAX {
                return None;
            }
            path.push(p);
            cursor = p;
        }
        path.reverse();
        Some(GeodesicPath {
            vertices: path,
            length: dist[g],
        })
    }

    /// The shortest edge-graph distance from the nearest of `sources` to every
    /// vertex, as a parallel array (`f32::INFINITY` where unreachable).
    ///
    /// Returns `None` when `sources` is empty or references a vertex out of
    /// range. This is a multi-source Dijkstra, so a vertex's value is its
    /// distance to the closest source.
    #[must_use]
    pub fn distance_field(&self, sources: &[u32]) -> Option<Vec<f32>> {
        let n = self.neighbours.len();
        if sources.is_empty() {
            return None;
        }
        let mut dist = vec![f32::INFINITY; n];
        let mut visited = vec![false; n];
        let mut heap = BinaryHeap::new();
        for &src in sources {
            let si = src as usize;
            if si >= n {
                return None;
            }
            if 0.0 < dist[si] {
                dist[si] = 0.0;
                heap.push(Frontier {
                    distance: 0.0,
                    vertex: src,
                });
            }
        }

        while let Some(Frontier { distance, vertex }) = heap.pop() {
            let v = vertex as usize;
            if visited[v] {
                continue;
            }
            visited[v] = true;
            for &(nb, w) in &self.neighbours[v] {
                let nbi = nb as usize;
                if visited[nbi] {
                    continue;
                }
                let candidate = distance + w;
                if candidate < dist[nbi] {
                    dist[nbi] = candidate;
                    heap.push(Frontier {
                        distance: candidate,
                        vertex: nb,
                    });
                }
            }
        }
        Some(dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit quad in the XY plane split into two triangles sharing the 0-2
    /// diagonal.
    fn quad() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        (verts, tris)
    }

    #[test]
    fn build_rejects_degenerate_input() {
        assert!(MeshEdgeGraph::build(&[], &[]).is_none());
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(MeshEdgeGraph::build(&verts, &[]).is_none());
        // Out-of-range triangle index.
        assert!(MeshEdgeGraph::build(&verts, &[[0, 1, 9]]).is_none());
    }

    #[test]
    fn shortest_path_takes_the_diagonal() {
        let (verts, tris) = quad();
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        let path = graph.shortest_path(0, 2).expect("0 reaches 2");
        // The diagonal edge 0-2 (length sqrt(2)) beats the two unit detours.
        assert_eq!(path.vertices, vec![0, 2]);
        assert!((path.length - 2.0_f32.sqrt()).abs() < 1e-6);
        assert_eq!(path.edge_count(), 1);
    }

    #[test]
    fn shortest_path_without_a_direct_edge_detours() {
        let (verts, tris) = quad();
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        // Vertices 1 and 3 share no edge, so the path is two unit edges.
        let path = graph.shortest_path(1, 3).expect("1 reaches 3");
        assert!((path.length - 2.0).abs() < 1e-6);
        assert_eq!(path.vertices.first(), Some(&1));
        assert_eq!(path.vertices.last(), Some(&3));
        assert_eq!(path.edge_count(), 2);
    }

    #[test]
    fn same_start_and_goal_is_zero_length() {
        let (verts, tris) = quad();
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        let path = graph.shortest_path(2, 2).expect("trivial path");
        assert_eq!(path.vertices, vec![2]);
        assert_eq!(path.length, 0.0);
        assert_eq!(path.edge_count(), 0);
    }

    #[test]
    fn out_of_range_endpoints_are_rejected() {
        let (verts, tris) = quad();
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        assert!(graph.shortest_path(0, 99).is_none());
        assert!(graph.shortest_path(99, 0).is_none());
    }

    #[test]
    fn disconnected_components_are_unreachable() {
        // Two triangles with disjoint vertex sets: 0-1-2 and 3-4-5.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(11.0, 0.0, 0.0),
            Vec3::new(10.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [3, 4, 5]];
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        assert!(graph.shortest_path(0, 4).is_none());
        assert!(graph.shortest_path(0, 1).is_some());
    }

    #[test]
    fn distance_field_is_multi_source() {
        let (verts, tris) = quad();
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        let field = graph.distance_field(&[0]).expect("field from vertex 0");
        assert_eq!(field[0], 0.0);
        assert!((field[1] - 1.0).abs() < 1e-6);
        assert!((field[2] - 2.0_f32.sqrt()).abs() < 1e-6);
        assert!((field[3] - 1.0).abs() < 1e-6);

        // With both 1 and 3 as sources, vertex 2 is one unit edge away from
        // each, closer than the sqrt(2) diagonal from vertex 0.
        let field2 = graph.distance_field(&[1, 3]).expect("field from 1 and 3");
        assert_eq!(field2[1], 0.0);
        assert_eq!(field2[3], 0.0);
        assert!((field2[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distance_field_rejects_bad_sources() {
        let (verts, tris) = quad();
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        assert!(graph.distance_field(&[]).is_none());
        assert!(graph.distance_field(&[99]).is_none());
    }

    #[test]
    fn queries_are_deterministic() {
        let (verts, tris) = quad();
        let graph = MeshEdgeGraph::build(&verts, &tris).expect("graph builds");
        let a = graph.shortest_path(1, 3).expect("run a");
        let b = graph.shortest_path(1, 3).expect("run b");
        assert_eq!(a, b);
        assert_eq!(graph.distance_field(&[0]), graph.distance_field(&[0]));
    }
}
