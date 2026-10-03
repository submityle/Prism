//! Oriented boundary-loop extraction for triangle meshes.
//!
//! The open edges of a mesh - those bordering exactly one triangle - chain into
//! closed loops that ring every hole and open shell. AAA cookers surface these
//! loops to drive hole filling, open-edge visualisation, UV-seam handling and
//! watertightness diagnostics, so this module extracts them as ordered,
//! consistently oriented vertex rings.
//!
//! A directed half-edge `(u, v)` is on the boundary when its reverse `(v, u)`
//! is absent from the mesh. Walking from each boundary half-edge to the next
//! outgoing one at its tip recovers the loops. The mesh is welded first
//! (reusing [`weld_mesh`](crate::collider::weld_mesh)) so split vertices do not
//! fragment a loop; results are indexed into the welded vertex list. This is
//! pure triangle-soup geometry with no coupling to the collision pipeline, and
//! nothing here is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::{HashMap, HashSet};

use crate::collider::weld::{weld_mesh, WeldParams};

/// Cross-product length below which a triangle is treated as degenerate.
const DEGENERATE_EPSILON: f32 = 1.0e-12;

/// Tuning for [`extract_boundary_loops`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryLoopParams {
    /// Position tolerance used to weld near-coincident vertices first.
    pub weld_epsilon: f32,
}

impl Default for BoundaryLoopParams {
    /// A `1e-5` weld tolerance.
    fn default() -> Self {
        Self {
            weld_epsilon: 1.0e-5,
        }
    }
}

/// Ordered, oriented boundary loops over the welded mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryLoops {
    /// Welded vertex positions the loop indices refer to.
    pub vertices: Vec<Vec3>,
    /// Each loop is a cyclic sequence of welded vertex indices; the closing
    /// edge runs from the last index back to the first.
    pub loops: Vec<Vec<u32>>,
}

impl BoundaryLoops {
    /// Number of distinct boundary loops.
    #[must_use]
    pub fn loop_count(&self) -> usize {
        self.loops.len()
    }

    /// Whether the mesh is closed (has no boundary edges).
    #[must_use]
    pub fn is_closed_mesh(&self) -> bool {
        self.loops.is_empty()
    }

    /// Total number of boundary edges across all loops.
    #[must_use]
    pub fn total_boundary_edges(&self) -> usize {
        self.loops.iter().map(Vec::len).sum()
    }

    /// Number of vertices in the welded mesh.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Whether the welded mesh is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// Extracts the oriented boundary loops of the welded mesh.
///
/// Returns `None` when the vertex or index slice is empty, `weld_epsilon` is
/// not finite and positive, or welding collapses the mesh to nothing. A closed
/// mesh yields `Some` with an empty loop list. Degenerate triangles are
/// skipped. The result is deterministic.
#[must_use]
pub fn extract_boundary_loops(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: BoundaryLoopParams,
) -> Option<BoundaryLoops> {
    if vertices.is_empty()
        || indices.is_empty()
        || !(params.weld_epsilon.is_finite() && params.weld_epsilon > 0.0)
    {
        return None;
    }

    let welded = weld_mesh(
        vertices,
        indices,
        WeldParams {
            position_epsilon: params.weld_epsilon,
            drop_duplicate_triangles: true,
        },
    )?;
    let verts = &welded.vertices;
    let tris = &welded.indices;
    if verts.is_empty() || tris.is_empty() {
        return None;
    }

    // Collect every directed half-edge of every non-degenerate triangle.
    let mut directed: HashSet<(u32, u32)> = HashSet::new();
    for tri in tris {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        let cross_len = (verts[b as usize] - verts[a as usize])
            .cross(verts[c as usize] - verts[a as usize])
            .length();
        if cross_len <= DEGENERATE_EPSILON {
            continue;
        }
        directed.insert((a, b));
        directed.insert((b, c));
        directed.insert((c, a));
    }

    // A half-edge is on the boundary when its reverse is absent. Index the
    // boundary half-edges by their start vertex so we can walk them.
    let mut outgoing: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(u, v) in &directed {
        if !directed.contains(&(v, u)) {
            outgoing.entry(u).or_default().push(v);
        }
    }
    if outgoing.is_empty() {
        // Closed mesh: no boundary.
        return Some(BoundaryLoops {
            vertices: welded.vertices,
            loops: Vec::new(),
        });
    }

    // Deterministic traversal: sort each adjacency list and visit start
    // vertices in ascending order, consuming half-edges as we walk.
    for ends in outgoing.values_mut() {
        ends.sort_unstable();
    }
    let mut starts: Vec<u32> = outgoing.keys().copied().collect();
    starts.sort_unstable();

    let mut loops: Vec<Vec<u32>> = Vec::new();
    for &start in &starts {
        while outgoing.get(&start).is_some_and(|e| !e.is_empty()) {
            let mut loop_vertices = vec![start];
            let mut current = start;
            // Walk outgoing boundary half-edges until we return to `start` or
            // run out (dangling chain on a non-manifold boundary).
            while let Some(next) = pop_outgoing(&mut outgoing, current) {
                if next == start {
                    break;
                }
                loop_vertices.push(next);
                current = next;
            }
            loops.push(loop_vertices);
        }
    }

    // Canonical ordering of loops for determinism, by their smallest vertex.
    loops.sort_by_key(|l| l.iter().copied().min().unwrap_or(u32::MAX));

    Some(BoundaryLoops {
        vertices: welded.vertices,
        loops,
    })
}

/// Removes and returns the next outgoing boundary vertex from `from`, if any.
fn pop_outgoing(outgoing: &mut HashMap<u32, Vec<u32>>, from: u32) -> Option<u32> {
    let ends = outgoing.get_mut(&from)?;
    if ends.is_empty() {
        return None;
    }
    Some(ends.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin, outward wound and closed.
    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let f = vec![
            [0, 2, 1],
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
        (v, f)
    }

    #[test]
    fn empty_or_invalid_input_is_rejected() {
        let (v, f) = unit_cube();
        let p = BoundaryLoopParams::default();
        assert!(extract_boundary_loops(&[], &f, p).is_none());
        assert!(extract_boundary_loops(&v, &[], p).is_none());
        assert!(extract_boundary_loops(&v, &f, BoundaryLoopParams { weld_epsilon: 0.0 }).is_none());
        assert!(
            extract_boundary_loops(&v, &f, BoundaryLoopParams { weld_epsilon: -1.0 }).is_none()
        );
    }

    #[test]
    fn closed_cube_has_no_boundary_loops() {
        let (v, f) = unit_cube();
        let report = extract_boundary_loops(&v, &f, BoundaryLoopParams::default()).unwrap();
        assert!(report.is_closed_mesh());
        assert_eq!(report.loop_count(), 0);
        assert_eq!(report.total_boundary_edges(), 0);
    }

    #[test]
    fn single_triangle_is_one_loop_of_three() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let report =
            extract_boundary_loops(&verts, &[[0, 1, 2]], BoundaryLoopParams::default()).unwrap();
        assert_eq!(report.loop_count(), 1);
        assert_eq!(report.loops[0].len(), 3);
        assert_eq!(report.total_boundary_edges(), 3);
        // The loop is a permutation of all three vertices.
        let mut sorted = report.loops[0].clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn open_quad_is_one_loop_of_four() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let report = extract_boundary_loops(&verts, &tris, BoundaryLoopParams::default()).unwrap();
        assert_eq!(report.loop_count(), 1);
        assert_eq!(report.loops[0].len(), 4);
        // Walking the loop traverses consecutive boundary edges.
        let l = &report.loops[0];
        let edges: HashSet<(u32, u32)> = l
            .iter()
            .enumerate()
            .map(|(i, &u)| {
                let v = l[(i + 1) % l.len()];
                (u, v)
            })
            .collect();
        // The interior diagonal (0,2)/(2,0) must not appear as a boundary edge.
        assert!(!edges.contains(&(0, 2)));
        assert!(!edges.contains(&(2, 0)));
    }

    #[test]
    fn two_disjoint_triangles_are_two_loops() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(5.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [3, 4, 5]];
        let report = extract_boundary_loops(&verts, &tris, BoundaryLoopParams::default()).unwrap();
        assert_eq!(report.loop_count(), 2);
        assert_eq!(report.total_boundary_edges(), 6);
    }

    #[test]
    fn report_is_deterministic() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let p = BoundaryLoopParams::default();
        let a = extract_boundary_loops(&verts, &tris, p).unwrap();
        let b = extract_boundary_loops(&verts, &tris, p).unwrap();
        assert_eq!(a, b);
    }
}
