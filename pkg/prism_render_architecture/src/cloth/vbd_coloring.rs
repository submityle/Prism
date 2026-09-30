//! Vertex graph coloring for the parallel (GPU) VBD cloth sweep.
//!
//! The [`super::vbd`] Vertex Block Descent solver relaxes mesh vertices with a
//! Gauss-Seidel sweep: vertex `i` reads its neighbors' *current* positions, so
//! the natural-order CPU sweep in [`super::vbd::solve_cloth_vbd`] is inherently
//! sequential. A GPU dispatch cannot honor that ordering across the whole mesh,
//! but it can honor it *per color*: if the vertices are partitioned so that no
//! two vertices sharing a distance constraint land in the same color, then every
//! vertex in one color reads only vertices of *other* colors, and the whole
//! color can be relaxed in parallel with no read-after-write hazard. Applying
//! colors one after another (Gauss-Seidel across colors, Jacobi within a color)
//! reproduces a valid Gauss-Seidel sweep whose per-vertex result is independent
//! of the intra-color order — which is exactly what makes the GPU kernel and the
//! colored CPU reference in [`super::vbd::solve_cloth_vbd_colored`] agree
//! bit-for-bit.
//!
//! The adjacency used here mirrors [`super::vbd`] exactly: two vertices are
//! neighbors when some constraint names them as its two endpoints, and a
//! degenerate self-constraint (`a == b`) or an endpoint outside
//! `particle_count` is skipped. Keeping the skip rule identical guarantees the
//! coloring is a *proper* coloring of the same graph the solver actually sweeps,
//! so no two vertices that read each other ever share a color.
//!
//! The coloring is a deterministic greedy pass (ascending vertex index, lowest
//! free color), so a fixed constraint list always yields the same colors,
//! color-major order, and offsets — a hard requirement for reproducible GPU
//! dispatch and for the parity twin.

use alloc::vec;
use alloc::vec::Vec;

use super::Constraint;

/// Sentinel stored in the per-vertex color array until a vertex is assigned a
/// color by the greedy pass.
const UNCOLORED: u32 = u32::MAX;

/// A proper vertex coloring of the cloth constraint graph.
///
/// Produced by [`color_cloth_vertices`]. Besides the per-vertex color, it carries
/// a *color-major ordering* of the vertices (`order`) split into contiguous runs
/// by `offsets`, so a caller can iterate one color's vertices as
/// `order[offsets[c]..offsets[c + 1]]`. That layout is what both the colored CPU
/// reference and the GPU per-color dispatch consume.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexColoring {
    /// `colors[v]` is the color index assigned to vertex `v`. Every vertex in
    /// `0..particle_count` receives a color (isolated vertices get color `0`).
    colors: Vec<u32>,
    /// Number of distinct colors (serial GPU passes). `0` only when there are no
    /// vertices.
    color_count: u32,
    /// Vertices listed color-major: all of color `0` (ascending), then color `1`,
    /// and so on. Length equals `particle_count`.
    order: Vec<u32>,
    /// Prefix offsets into `order`; color `c` occupies `order[offsets[c]..offsets[c + 1]]`.
    /// Length is `color_count + 1` (or a single `0` when empty).
    offsets: Vec<u32>,
}

impl VertexColoring {
    /// The number of colors, i.e. the number of serial passes a colored sweep
    /// performs per Gauss-Seidel iteration.
    #[must_use]
    pub fn color_count(&self) -> u32 {
        self.color_count
    }

    /// Returns `true` when the coloring covers no vertices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// The color assigned to `vertex`, or `None` when the index is out of range.
    #[must_use]
    pub fn color_of(&self, vertex: usize) -> Option<u32> {
        self.colors.get(vertex).copied()
    }

    /// The per-vertex color array (length equals the vertex count).
    #[must_use]
    pub fn colors(&self) -> &[u32] {
        &self.colors
    }

    /// The color-major vertex order (length equals the vertex count).
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The prefix offsets into [`order`](Self::order); length is
    /// `color_count + 1` whenever there is at least one vertex.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// The half-open range into [`order`](Self::order) for `color`, or an empty
    /// range when the color index is out of bounds (never panics).
    #[must_use]
    pub fn color_range(&self, color: usize) -> core::ops::Range<usize> {
        if color + 1 < self.offsets.len() {
            let start = self.offsets[color] as usize;
            let end = self.offsets[color + 1] as usize;
            start..end
        } else {
            0..0
        }
    }

    /// The vertices belonging to `color` in ascending index order, or an empty
    /// slice when the color index is out of bounds.
    #[must_use]
    pub fn color_vertices(&self, color: usize) -> &[u32] {
        let range = self.color_range(color);
        &self.order[range]
    }
}

/// Builds the vertex-neighbor adjacency used for coloring.
///
/// `out[v]` lists every vertex that shares a constraint with `v`. The skip rule
/// (self-constraint or out-of-range endpoint) matches [`super::vbd`] so the
/// coloring is proper on exactly the graph the solver sweeps. Duplicate edges
/// (the same pair named by several constraints) may appear more than once; that
/// is harmless for greedy coloring, which only cares whether a neighbor color is
/// present.
fn neighbor_adjacency(constraints: &[Constraint], particle_count: usize) -> Vec<Vec<u32>> {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); particle_count];
    for constraint in constraints {
        let a = constraint.a as usize;
        let b = constraint.b as usize;
        if a == b || a >= particle_count || b >= particle_count {
            continue;
        }
        adjacency[a].push(constraint.b);
        adjacency[b].push(constraint.a);
    }
    adjacency
}

/// Greedily colors the cloth constraint graph so that no two vertices sharing a
/// distance constraint share a color.
///
/// The pass walks vertices in ascending index order and assigns each the lowest
/// color not already used by one of its already-colored neighbors, so an
/// isolated vertex always lands in color `0`. The result is deterministic for a
/// fixed constraint list. An empty vertex set yields an empty coloring with
/// `color_count == 0`.
///
/// The returned [`VertexColoring`] also carries the color-major ordering and
/// per-color offsets the colored CPU reference and the GPU dispatch consume.
#[must_use]
pub fn color_cloth_vertices(constraints: &[Constraint], particle_count: usize) -> VertexColoring {
    if particle_count == 0 {
        return VertexColoring {
            colors: Vec::new(),
            color_count: 0,
            order: Vec::new(),
            offsets: vec![0],
        };
    }

    let adjacency = neighbor_adjacency(constraints, particle_count);
    let mut colors = vec![UNCOLORED; particle_count];
    let mut max_color = 0u32;

    for v in 0..particle_count {
        // A vertex can need at most (degree + 1) colors, so a boolean scratch of
        // that size always covers the lowest free color.
        let mut forbidden = vec![false; adjacency[v].len() + 1];
        for &neighbor in &adjacency[v] {
            let c = colors[neighbor as usize];
            if c != UNCOLORED && (c as usize) < forbidden.len() {
                forbidden[c as usize] = true;
            }
        }
        let mut chosen = 0u32;
        while (chosen as usize) < forbidden.len() && forbidden[chosen as usize] {
            chosen += 1;
        }
        colors[v] = chosen;
        max_color = max_color.max(chosen);
    }

    let color_count = max_color + 1;

    // Stable-count the colors, then lay vertices out color-major (ascending
    // within each color) using a running cursor per color.
    let mut offsets = vec![0u32; color_count as usize + 1];
    for &c in &colors {
        offsets[c as usize + 1] += 1;
    }
    for i in 0..color_count as usize {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut order = vec![0u32; particle_count];
    for (v, &c) in colors.iter().enumerate() {
        let slot = cursor[c as usize];
        order[slot as usize] = v as u32;
        cursor[c as usize] = slot + 1;
    }

    VertexColoring {
        colors,
        color_count,
        order,
        offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::{Compliance, ConstraintKind};

    fn edge(a: u32, b: u32) -> Constraint {
        Constraint::new(a, b, 1.0, Compliance(1.0e-4), ConstraintKind::Stretch)
    }

    #[test]
    fn empty_particles_color_to_empty_coloring() {
        let coloring = color_cloth_vertices(&[], 0);
        assert!(coloring.is_empty());
        assert_eq!(coloring.color_count(), 0);
        assert_eq!(coloring.offsets(), &[0]);
        assert!(coloring.order().is_empty());
    }

    #[test]
    fn isolated_vertices_all_share_color_zero() {
        let coloring = color_cloth_vertices(&[], 4);
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.colors(), &[0, 0, 0, 0]);
        assert_eq!(coloring.color_vertices(0), &[0, 1, 2, 3]);
    }

    #[test]
    fn a_single_edge_needs_two_colors() {
        let coloring = color_cloth_vertices(&[edge(0, 1)], 2);
        assert_eq!(coloring.color_count(), 2);
        assert_ne!(coloring.color_of(0), coloring.color_of(1));
    }

    #[test]
    fn a_chain_alternates_two_colors() {
        // 0-1-2-3-4 path graph is 2-colorable.
        let cons = [edge(0, 1), edge(1, 2), edge(2, 3), edge(3, 4)];
        let coloring = color_cloth_vertices(&cons, 5);
        assert_eq!(coloring.color_count(), 2);
        assert_eq!(coloring.colors(), &[0, 1, 0, 1, 0]);
    }

    #[test]
    fn a_triangle_needs_three_colors() {
        let cons = [edge(0, 1), edge(1, 2), edge(0, 2)];
        let coloring = color_cloth_vertices(&cons, 3);
        assert_eq!(coloring.color_count(), 3);
        assert_eq!(coloring.colors(), &[0, 1, 2]);
    }

    #[test]
    fn coloring_is_a_proper_coloring() {
        // A small grid of quads: build stretch + shear edges and assert no
        // constraint ever joins two same-colored vertices.
        let mut cons = Vec::new();
        let (nx, nz) = (5usize, 4usize);
        let idx = |x: usize, z: usize| (z * nx + x) as u32;
        for z in 0..nz {
            for x in 0..nx {
                if x + 1 < nx {
                    cons.push(edge(idx(x, z), idx(x + 1, z)));
                }
                if z + 1 < nz {
                    cons.push(edge(idx(x, z), idx(x, z + 1)));
                }
                if x + 1 < nx && z + 1 < nz {
                    cons.push(edge(idx(x, z), idx(x + 1, z + 1)));
                    cons.push(edge(idx(x + 1, z), idx(x, z + 1)));
                }
            }
        }
        let coloring = color_cloth_vertices(&cons, nx * nz);
        for c in &cons {
            assert_ne!(
                coloring.color_of(c.a as usize),
                coloring.color_of(c.b as usize),
                "constraint {}-{} joined two same-colored vertices",
                c.a,
                c.b
            );
        }
        // Order is a permutation of every vertex exactly once.
        let mut seen = vec![false; nx * nz];
        for &v in coloring.order() {
            assert!(!seen[v as usize], "vertex {v} listed twice");
            seen[v as usize] = true;
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn degenerate_and_out_of_range_edges_are_ignored() {
        // Self edge and out-of-range endpoint must not create adjacency, so
        // both vertices stay independent (color 0).
        let cons = [edge(0, 0), edge(0, 9), edge(1, 1)];
        let coloring = color_cloth_vertices(&cons, 2);
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.colors(), &[0, 0]);
    }

    #[test]
    fn coloring_is_deterministic() {
        let cons = [edge(0, 1), edge(1, 2), edge(2, 3), edge(0, 3), edge(0, 2)];
        let first = color_cloth_vertices(&cons, 4);
        let second = color_cloth_vertices(&cons, 4);
        assert_eq!(first, second);
    }
}
