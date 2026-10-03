//! Triangle-mesh welding and cleanup for collision cooking.
//!
//! Authoring tools routinely emit collision geometry with coincident
//! (duplicated) vertices, hairline cracks, degenerate "sliver" triangles and
//! repeated faces. AAA cooking pipelines (`PhysX` `cookTriangleMesh`'s clean
//! pass, `Jolt`'s mesh preparation) weld near-coincident vertices and drop
//! degenerate/duplicate triangles before building an acceleration structure,
//! both to shrink the data and to keep the narrow phase from tripping over
//! zero-area faces.
//!
//! This module performs that cleanup as a standalone, deterministic pass:
//!
//! - near-coincident vertices within [`WeldParams::position_epsilon`] are
//!   merged using a uniform spatial hash (so the cost is roughly linear in the
//!   vertex count rather than quadratic),
//! - triangles that collapse to a line or point after welding are discarded,
//!   and
//! - optionally, triangles that become duplicates (same vertex triple,
//!   ignoring winding) are collapsed to a single face.
//!
//! The result is a compact remapped mesh plus the counts of what was removed.
//! This is standard geometry cleanup; nothing here is derived from Unreal
//! Engine source.

use glam::Vec3;
use std::collections::HashMap;
use std::collections::HashSet;

/// Parameters controlling the welding pass.
#[derive(Clone, Copy, Debug)]
pub struct WeldParams {
    /// Vertices closer than this distance are merged into one. Must be finite
    /// and strictly positive.
    pub position_epsilon: f32,
    /// When set, triangles sharing the same (unordered) vertex triple are
    /// collapsed to the first occurrence.
    pub drop_duplicate_triangles: bool,
}

impl Default for WeldParams {
    fn default() -> Self {
        Self {
            position_epsilon: 1e-5,
            drop_duplicate_triangles: true,
        }
    }
}

/// A cleaned-up triangle mesh plus a tally of what the pass removed.
#[derive(Clone, Debug)]
pub struct WeldedMesh {
    /// Deduplicated vertex positions (representatives of each merge cluster).
    pub vertices: Vec<Vec3>,
    /// Triangles reindexed into [`WeldedMesh::vertices`].
    pub indices: Vec<[u32; 3]>,
    /// How many input vertices were merged away.
    pub removed_vertices: usize,
    /// How many input triangles were dropped (degenerate or duplicate).
    pub removed_triangles: usize,
}

/// Welds near-coincident vertices and removes degenerate/duplicate triangles.
///
/// Returns `None` when `vertices` is empty or `position_epsilon` is not finite
/// and strictly positive. Triangles that reference out-of-range vertices are
/// skipped and counted as removed.
#[must_use]
pub fn weld_mesh(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: WeldParams,
) -> Option<WeldedMesh> {
    if vertices.is_empty() || !params.position_epsilon.is_finite() || params.position_epsilon <= 0.0
    {
        return None;
    }

    let eps = params.position_epsilon;
    let eps_sq = eps * eps;
    let inv_cell = 1.0 / eps;

    // Spatial hash: grid cell -> indices of representatives placed in that cell.
    let mut buckets: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    let mut reps: Vec<Vec3> = Vec::new();
    // old vertex index -> representative (new) index.
    let mut remap: Vec<u32> = Vec::with_capacity(vertices.len());

    let cell_of = |p: Vec3| -> (i64, i64, i64) {
        (
            (p.x * inv_cell).floor() as i64,
            (p.y * inv_cell).floor() as i64,
            (p.z * inv_cell).floor() as i64,
        )
    };

    for &v in vertices {
        let (cx, cy, cz) = cell_of(v);
        let mut found: Option<u32> = None;
        // Search the 27 neighbouring cells so a cluster straddling a cell
        // boundary still merges.
        'search: for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(list) = buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &ri in list {
                            if (reps[ri as usize] - v).length_squared() <= eps_sq {
                                found = Some(ri);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }

        let new_index = match found {
            Some(ri) => ri,
            None => {
                let ri = reps.len() as u32;
                reps.push(v);
                buckets.entry((cx, cy, cz)).or_default().push(ri);
                ri
            }
        };
        remap.push(new_index);
    }

    let old_vertex_count = vertices.len();
    let new_vertex_count = reps.len();

    let mut out_indices: Vec<[u32; 3]> = Vec::with_capacity(indices.len());
    let mut seen: HashSet<[u32; 3]> = HashSet::new();
    let vcount = vertices.len() as u32;
    let mut kept = 0usize;

    for tri in indices {
        if tri[0] >= vcount || tri[1] >= vcount || tri[2] >= vcount {
            continue; // out-of-range: drop (counted as removed below).
        }
        let a = remap[tri[0] as usize];
        let b = remap[tri[1] as usize];
        let c = remap[tri[2] as usize];
        // Collapsed to a line or point after welding -> degenerate.
        if a == b || b == c || a == c {
            continue;
        }
        if params.drop_duplicate_triangles {
            let mut key = [a, b, c];
            key.sort_unstable();
            if !seen.insert(key) {
                continue;
            }
        }
        out_indices.push([a, b, c]);
        kept += 1;
    }

    let removed_triangles = indices.len() - kept;

    Some(WeldedMesh {
        vertices: reps,
        indices: out_indices,
        removed_vertices: old_vertex_count - new_vertex_count,
        removed_triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(weld_mesh(&[], &[], WeldParams::default()).is_none());
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(weld_mesh(
            &v,
            &[[0, 1, 2]],
            WeldParams {
                position_epsilon: 0.0,
                drop_duplicate_triangles: true
            }
        )
        .is_none());
        assert!(weld_mesh(
            &v,
            &[[0, 1, 2]],
            WeldParams {
                position_epsilon: f32::NAN,
                drop_duplicate_triangles: true
            }
        )
        .is_none());
    }

    #[test]
    fn merges_coincident_vertices_and_rewrites_indices() {
        // Two triangles sharing an edge, but the shared edge is authored as four
        // separate (pairwise-coincident) vertices.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            // Duplicates of 0 and 1 (within epsilon).
            Vec3::new(1e-7, 0.0, 0.0),
            Vec3::new(1.0, 1e-7, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let idx = vec![[0, 1, 2], [3, 4, 5]];
        let w = weld_mesh(&v, &idx, WeldParams::default()).expect("welds");
        // Vertices 0<-3 and 1<-4 merge: 6 -> 4 unique vertices.
        assert_eq!(w.vertices.len(), 4);
        assert_eq!(w.removed_vertices, 2);
        // Both triangles survive (non-degenerate) and are distinct.
        assert_eq!(w.indices.len(), 2);
        assert_eq!(w.removed_triangles, 0);
    }

    #[test]
    fn drops_degenerate_triangles() {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e-8, 0.0, 0.0), // merges into vertex 0
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        // First triangle collapses (0 and 1 weld -> two equal indices).
        let idx = vec![[0, 1, 2], [0, 2, 3]];
        let w = weld_mesh(&v, &idx, WeldParams::default()).expect("welds");
        assert_eq!(w.indices.len(), 1);
        assert_eq!(w.removed_triangles, 1);
        assert_eq!(w.vertices.len(), 3);
    }

    #[test]
    fn drops_duplicate_triangles_when_requested() {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        // Same triangle three times (one with rotated winding).
        let idx = vec![[0, 1, 2], [0, 1, 2], [1, 2, 0]];
        let w = weld_mesh(&v, &idx, WeldParams::default()).expect("welds");
        assert_eq!(w.indices.len(), 1);
        assert_eq!(w.removed_triangles, 2);

        // With deduplication disabled, every non-degenerate triangle is kept.
        let w2 = weld_mesh(
            &v,
            &idx,
            WeldParams {
                position_epsilon: 1e-5,
                drop_duplicate_triangles: false,
            },
        )
        .expect("welds");
        assert_eq!(w2.indices.len(), 3);
        assert_eq!(w2.removed_triangles, 0);
    }

    #[test]
    fn skips_out_of_range_triangles() {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let idx = vec![[0, 1, 2], [0, 1, 9]];
        let w = weld_mesh(&v, &idx, WeldParams::default()).expect("welds");
        assert_eq!(w.indices.len(), 1);
        assert_eq!(w.removed_triangles, 1);
    }

    #[test]
    fn cluster_straddling_cell_boundary_still_merges() {
        // Place two points on opposite sides of a grid-cell boundary but within
        // epsilon of each other; the 27-cell neighbour search must still merge.
        let eps = 0.01;
        let p = Vec3::new(eps, 0.0, 0.0); // sits right at a cell boundary
        let q = Vec3::new(eps + 0.5 * eps, 0.0, 0.0); // within epsilon of p
        let v = vec![p, q, Vec3::new(0.0, 1.0, 0.0)];
        let w = weld_mesh(
            &v,
            &[[0, 1, 2]],
            WeldParams {
                position_epsilon: eps,
                drop_duplicate_triangles: true,
            },
        )
        .expect("welds");
        assert_eq!(w.vertices.len(), 2, "p and q should merge");
    }

    #[test]
    fn leaves_a_clean_mesh_untouched() {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let idx = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        let w = weld_mesh(&v, &idx, WeldParams::default()).expect("welds");
        assert_eq!(w.vertices.len(), 4);
        assert_eq!(w.indices.len(), 4);
        assert_eq!(w.removed_vertices, 0);
        assert_eq!(w.removed_triangles, 0);
    }
}
