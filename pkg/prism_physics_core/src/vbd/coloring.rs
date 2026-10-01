//! Vertex graph colouring for the parallel (GPU) Vertex Block Descent sweep.
//!
//! The natural-order [`VbdSolver::step`](super::solver::VbdSolver::step) sweep
//! is Gauss-Seidel: vertex `i` reads its neighbours' *current* positions, so it
//! is inherently sequential and cannot be dispatched across a whole mesh in
//! parallel. It can, however, be dispatched *per colour*: if the vertices are
//! partitioned so that no two vertices sharing a spring land in the same colour,
//! then every vertex in one colour reads only vertices of *other* colours and
//! the whole colour relaxes in parallel with no read-after-write hazard.
//! Applying colours one after another (Gauss-Seidel across colours, Jacobi
//! within a colour) reproduces a valid Gauss-Seidel sweep whose per-vertex
//! result is independent of the intra-colour order. That is exactly what lets
//! the GPU kernel and the colour-ordered CPU reference
//! [`VbdSolver::step_colored`](super::solver::VbdSolver::step_colored) agree
//! bit-for-bit.
//!
//! The adjacency used here mirrors the solver exactly: two vertices are
//! neighbours when a spring names them as its two endpoints, and a degenerate
//! self-spring (`a == b`) or an endpoint outside `vertex_count` is skipped.
//! Keeping the skip rule identical guarantees the colouring is a *proper*
//! colouring of the same graph the solver sweeps.
//!
//! The colouring is a deterministic greedy pass (ascending vertex index, lowest
//! free colour), so a fixed spring list always yields the same colours,
//! colour-major order, and offsets — a hard requirement for reproducible GPU
//! dispatch and for the parity twin.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Greedy
//! graph colouring for parallel Gauss-Seidel is a standard, publicly documented
//! technique; the per-colour VBD schedule follows Chen et al., "Vertex Block
//! Descent" (2024).

use super::element::SpringSet;

/// Sentinel stored in the per-vertex colour array until a vertex is assigned a
/// colour by the greedy pass.
const UNCOLORED: u32 = u32::MAX;

/// A proper vertex colouring of a [`SpringSet`] graph.
///
/// Produced by [`color_springs`]. Besides the per-vertex colour it carries a
/// *colour-major ordering* of the vertices (`order`) split into contiguous runs
/// by `offsets`, so a caller iterates one colour's vertices as
/// `order[offsets[c]..offsets[c + 1]]`. That layout is what both the
/// colour-ordered CPU reference and the GPU per-colour dispatch consume.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VbdColoring {
    /// `colors[v]` is the colour index assigned to vertex `v`. Every vertex in
    /// `0..vertex_count` receives a colour (isolated vertices get colour `0`).
    colors: Vec<u32>,
    /// Number of distinct colours (serial GPU passes). `0` only when there are
    /// no vertices.
    color_count: u32,
    /// Vertices listed colour-major: all of colour `0` (ascending), then colour
    /// `1`, and so on. Length equals `vertex_count`.
    order: Vec<u32>,
    /// Prefix offsets into `order`; colour `c` occupies
    /// `order[offsets[c]..offsets[c + 1]]`. Length is `color_count + 1` (or a
    /// single `0` when empty).
    offsets: Vec<u32>,
}

impl VbdColoring {
    /// The number of colours, i.e. the number of serial passes a coloured sweep
    /// performs per Gauss-Seidel iteration.
    #[must_use]
    pub fn color_count(&self) -> u32 {
        self.color_count
    }

    /// Returns `true` when the colouring covers no vertices.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// The colour assigned to `vertex`, or `None` when the index is out of range.
    #[must_use]
    pub fn color_of(&self, vertex: usize) -> Option<u32> {
        self.colors.get(vertex).copied()
    }

    /// The per-vertex colour array (length equals the vertex count).
    #[must_use]
    pub fn colors(&self) -> &[u32] {
        &self.colors
    }

    /// The colour-major vertex order (length equals the vertex count).
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
    /// range when the colour index is out of bounds (never panics).
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
    /// slice when the colour index is out of bounds.
    #[must_use]
    pub fn color_vertices(&self, color: usize) -> &[u32] {
        let range = self.color_range(color);
        &self.order[range]
    }
}

/// Builds the vertex-neighbour adjacency used for colouring.
///
/// `out[v]` lists every vertex that shares a spring with `v`. The skip rule
/// (self-spring or out-of-range endpoint) matches the solver so the colouring
/// is proper on exactly the graph the solver sweeps. Duplicate edges (the same
/// pair named by several springs) may appear more than once; that is harmless
/// for greedy colouring, which only cares whether a neighbour colour is present.
fn neighbor_adjacency(springs: &SpringSet, vertex_count: usize) -> Vec<Vec<u32>> {
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
    for spring in &springs.springs {
        let a = spring.a.index();
        let b = spring.b.index();
        if a == b || a >= vertex_count || b >= vertex_count {
            continue;
        }
        adjacency[a].push(b as u32);
        adjacency[b].push(a as u32);
    }
    adjacency
}

/// Greedily colours a [`SpringSet`] graph so that no two vertices sharing a
/// spring share a colour.
///
/// The pass walks vertices in ascending index order and assigns each the lowest
/// colour not already used by one of its already-coloured neighbours, so an
/// isolated vertex always lands in colour `0`. The result is deterministic for a
/// fixed spring list. An empty vertex set yields an empty colouring with
/// `color_count == 0`.
///
/// The returned [`VbdColoring`] also carries the colour-major ordering and
/// per-colour offsets the colour-ordered CPU reference and the GPU dispatch
/// consume.
#[must_use]
pub fn color_springs(springs: &SpringSet, vertex_count: usize) -> VbdColoring {
    if vertex_count == 0 {
        return VbdColoring {
            colors: Vec::new(),
            color_count: 0,
            order: Vec::new(),
            offsets: vec![0],
        };
    }

    let adjacency = neighbor_adjacency(springs, vertex_count);
    let mut colors = vec![UNCOLORED; vertex_count];
    let mut max_color = 0u32;

    for v in 0..vertex_count {
        // A vertex can need at most (degree + 1) colours, so a boolean scratch of
        // that size always covers the lowest free colour.
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

    // Stable-count the colours, then lay vertices out colour-major (ascending
    // within each colour) using a running cursor per colour.
    let mut offsets = vec![0u32; color_count as usize + 1];
    for &c in &colors {
        offsets[c as usize + 1] += 1;
    }
    for i in 0..color_count as usize {
        offsets[i + 1] += offsets[i];
    }
    let mut cursor = offsets.clone();
    let mut order = vec![0u32; vertex_count];
    for (v, &c) in colors.iter().enumerate() {
        let slot = cursor[c as usize];
        order[slot as usize] = v as u32;
        cursor[c as usize] = slot + 1;
    }

    VbdColoring {
        colors,
        color_count,
        order,
        offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::particle::ParticleHandle;
    use crate::vbd::element::SpringElement;

    fn handle(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    fn spring(a: u32, b: u32) -> SpringElement {
        SpringElement::new(handle(a), handle(b), 1.0, 1000.0)
    }

    #[test]
    fn empty_vertices_color_to_empty_coloring() {
        let coloring = color_springs(&SpringSet::new(), 0);
        assert!(coloring.is_empty());
        assert_eq!(coloring.color_count(), 0);
        assert_eq!(coloring.offsets(), &[0]);
        assert!(coloring.order().is_empty());
    }

    #[test]
    fn isolated_vertices_all_get_color_zero() {
        let coloring = color_springs(&SpringSet::new(), 4);
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.colors(), &[0, 0, 0, 0]);
        assert_eq!(coloring.color_vertices(0), &[0, 1, 2, 3]);
    }

    #[test]
    fn single_edge_needs_two_colors() {
        let mut springs = SpringSet::new();
        springs.push(spring(0, 1));
        let coloring = color_springs(&springs, 2);
        assert_eq!(coloring.color_count(), 2);
        assert_ne!(coloring.color_of(0), coloring.color_of(1));
    }

    #[test]
    fn adjacent_vertices_never_share_a_color() {
        // A 3x3 grid of structural springs.
        let mut springs = SpringSet::new();
        let idx = |r: u32, c: u32| r * 3 + c;
        for r in 0..3 {
            for c in 0..3 {
                if c + 1 < 3 {
                    springs.push(spring(idx(r, c), idx(r, c + 1)));
                }
                if r + 1 < 3 {
                    springs.push(spring(idx(r, c), idx(r + 1, c)));
                }
            }
        }
        let coloring = color_springs(&springs, 9);
        for s in &springs.springs {
            assert_ne!(
                coloring.color_of(s.a.index()),
                coloring.color_of(s.b.index()),
                "spring endpoints shared a color"
            );
        }
        // Every vertex is placed exactly once, color-major.
        let mut seen = coloring.order().to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn self_spring_and_out_of_range_endpoints_are_skipped() {
        let mut springs = SpringSet::new();
        springs.push(spring(0, 0)); // degenerate self-spring
        springs.push(spring(1, 9)); // out-of-range endpoint
        let coloring = color_springs(&springs, 2);
        // Neither edge constrains the graph, so both vertices stay in color 0.
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.colors(), &[0, 0]);
    }

    #[test]
    fn coloring_is_deterministic() {
        let mut springs = SpringSet::new();
        for i in 0..5 {
            springs.push(spring(i, i + 1));
        }
        let a = color_springs(&springs, 6);
        let b = color_springs(&springs, 6);
        assert_eq!(a, b);
    }
}
