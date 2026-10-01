//! Vertex→triangle-corner adjacency (`CSR`) for the `GPU` cloth-pressure gather.
//!
//! The compliant pressure projection accumulates, at every vertex, the sum of
//! the per-triangle volume-gradient contributions of each triangle *corner*
//! that lands on it. On the `CPU` that is a scatter (`gradients[i] += ...`); a
//! race-free `GPU` pass cannot scatter without atomics, so it instead *gathers*
//! — one thread per vertex walks the list of corners incident to that vertex
//! and sums their contributions. This module builds that list once on the host
//! as a compressed-sparse-row structure so the kernel is a plain indexed read.
//!
//! Each incident corner is packed as `triangle * 4 + corner` (`corner` in
//! `0..3`), leaving the low two bits for the corner so the per-triangle pass and
//! the gather agree on which of a triangle's three gradient slots to read.
//!
//! # Provenance
//!
//! `CSR` adjacency and the scatter→gather reformulation are standard
//! data-parallel techniques. No Unreal Engine source or derived code.

use alloc::vec;
use alloc::vec::Vec;

/// Compressed-sparse-row adjacency mapping each particle to the triangle
/// corners that reference it.
///
/// `offsets` has `vertex_count + 1` entries (exclusive prefix sums); the corners
/// incident to vertex `v` are `corners[offsets[v]..offsets[v + 1]]`, each packed
/// as `triangle * 4 + corner` via [`pack_corner`]. Triangle indices that fall
/// outside `vertex_count` are dropped here exactly as the projection skips them,
/// so the gather never reads a missing vertex.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexTriangleAdjacency {
    /// Exclusive prefix sums; `vertex_count + 1` entries.
    pub offsets: Vec<u32>,
    /// Packed `triangle * 4 + corner` ids, grouped by vertex.
    pub corners: Vec<u32>,
}

impl VertexTriangleAdjacency {
    /// The number of vertices this adjacency was built for.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The total number of incident `(triangle, corner)` entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.corners.len()
    }

    /// The packed corner ids incident to `vertex`.
    ///
    /// # Panics
    ///
    /// Panics if `vertex >= self.vertex_count()`.
    #[must_use]
    pub fn corners_for(&self, vertex: usize) -> &[u32] {
        let start = self.offsets[vertex] as usize;
        let end = self.offsets[vertex + 1] as usize;
        &self.corners[start..end]
    }
}

/// Packs a `(triangle, corner)` pair into a single `u32` as `triangle * 4 + corner`.
#[must_use]
pub fn pack_corner(triangle: u32, corner: u32) -> u32 {
    triangle * 4 + corner
}

/// Unpacks a `triangle * 4 + corner` id back into `(triangle, corner)`.
#[must_use]
pub fn unpack_corner(packed: u32) -> (u32, u32) {
    (packed >> 2, packed & 0b11)
}

/// Builds the vertex→corner [`VertexTriangleAdjacency`] for `triangles` over a
/// particle array of `vertex_count` vertices.
///
/// Corners whose vertex index is `>= vertex_count` are dropped, mirroring the
/// projection's out-of-range skip, so a well-formed closed shell yields one
/// entry per triangle corner and every entry points at a live vertex. The build
/// is a deterministic counting sort: degree histogram, exclusive prefix sum,
/// then a stable scatter into per-vertex runs (corners within a vertex stay in
/// ascending triangle-then-corner order).
#[must_use]
pub fn build_vertex_triangle_adjacency(
    triangles: &[[u32; 3]],
    vertex_count: usize,
) -> VertexTriangleAdjacency {
    let mut offsets = vec![0u32; vertex_count + 1];
    if vertex_count == 0 {
        return VertexTriangleAdjacency {
            offsets,
            corners: Vec::new(),
        };
    }

    // Degree histogram (stored shifted by one so it becomes the prefix sum).
    for tri in triangles {
        for &v in tri {
            let vi = v as usize;
            if vi < vertex_count {
                offsets[vi + 1] += 1;
            }
        }
    }
    // Exclusive prefix sum into start offsets.
    for i in 0..vertex_count {
        offsets[i + 1] += offsets[i];
    }

    let total = offsets[vertex_count] as usize;
    let mut corners = vec![0u32; total];
    // Per-vertex write cursor, seeded at each vertex's run start.
    let mut cursor = offsets.clone();
    for (t, tri) in triangles.iter().enumerate() {
        for corner in 0u32..3 {
            let vi = tri[corner as usize] as usize;
            if vi < vertex_count {
                let slot = cursor[vi] as usize;
                corners[slot] = pack_corner(t as u32, corner);
                cursor[vi] += 1;
            }
        }
    }

    VertexTriangleAdjacency { offsets, corners }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trips() {
        for t in 0u32..1000 {
            for c in 0u32..3 {
                assert_eq!(unpack_corner(pack_corner(t, c)), (t, c));
            }
        }
    }

    #[test]
    fn single_triangle_lists_one_corner_per_vertex() {
        let adj = build_vertex_triangle_adjacency(&[[0, 1, 2]], 3);
        assert_eq!(adj.vertex_count(), 3);
        assert_eq!(adj.entry_count(), 3);
        assert_eq!(adj.corners_for(0), &[pack_corner(0, 0)]);
        assert_eq!(adj.corners_for(1), &[pack_corner(0, 1)]);
        assert_eq!(adj.corners_for(2), &[pack_corner(0, 2)]);
    }

    #[test]
    fn shared_vertex_accumulates_corners_in_triangle_order() {
        // Two triangles share vertex 0; its list must hold both corners.
        let adj = build_vertex_triangle_adjacency(&[[0, 1, 2], [0, 2, 3]], 4);
        assert_eq!(adj.corners_for(0), &[pack_corner(0, 0), pack_corner(1, 0)]);
        assert_eq!(adj.corners_for(2), &[pack_corner(0, 2), pack_corner(1, 1)]);
        assert_eq!(adj.corners_for(1), &[pack_corner(0, 1)]);
        assert_eq!(adj.corners_for(3), &[pack_corner(1, 2)]);
    }

    #[test]
    fn offsets_are_a_valid_prefix_sum() {
        let adj = build_vertex_triangle_adjacency(&[[0, 1, 2], [0, 2, 3], [1, 2, 3]], 4);
        assert_eq!(adj.offsets.len(), 5);
        assert_eq!(adj.offsets[0], 0);
        assert_eq!(*adj.offsets.last().unwrap() as usize, adj.entry_count());
        for w in adj.offsets.windows(2) {
            assert!(w[0] <= w[1], "offsets must be non-decreasing");
        }
        // Total corners == 3 triangles * 3 corners.
        assert_eq!(adj.entry_count(), 9);
    }

    #[test]
    fn every_entry_is_recovered_by_the_gather() {
        // Each (triangle, corner) must appear exactly once, under the vertex it
        // belongs to, so a gather reconstructs the full scatter.
        let triangles = [[0u32, 1, 2], [2, 1, 3], [0, 3, 1]];
        let adj = build_vertex_triangle_adjacency(&triangles, 4);
        let mut seen = alloc::vec![0u32; triangles.len() * 3];
        for v in 0..adj.vertex_count() {
            for &packed in adj.corners_for(v) {
                let (t, c) = unpack_corner(packed);
                assert_eq!(triangles[t as usize][c as usize] as usize, v);
                seen[(t * 3 + c) as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "each corner gathered once");
    }

    #[test]
    fn out_of_range_corners_are_dropped() {
        // Vertex index 9 is outside a 3-vertex array and must not appear.
        let adj = build_vertex_triangle_adjacency(&[[0, 1, 9]], 3);
        assert_eq!(adj.entry_count(), 2);
        assert_eq!(adj.corners_for(0), &[pack_corner(0, 0)]);
        assert_eq!(adj.corners_for(1), &[pack_corner(0, 1)]);
        assert!(adj.corners_for(2).is_empty());
    }

    #[test]
    fn empty_mesh_has_single_zero_offset_run() {
        let adj = build_vertex_triangle_adjacency(&[], 4);
        assert_eq!(adj.offsets, alloc::vec![0, 0, 0, 0, 0]);
        assert!(adj.corners.is_empty());

        let none = build_vertex_triangle_adjacency(&[], 0);
        assert_eq!(none.offsets, alloc::vec![0]);
        assert_eq!(none.vertex_count(), 0);
    }
}
