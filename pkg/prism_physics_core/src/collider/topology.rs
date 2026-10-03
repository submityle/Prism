//! Triangle-mesh topology analysis for collision cooking validation.
//!
//! Before building a signed-distance field, running convex decomposition or
//! trusting a ray-parity inside test, cookers want to know whether a mesh is
//! actually a clean, closed 2-manifold. AAA pipelines (`PhysX`, `Jolt`) surface
//! exactly these diagnostics: open edges (holes), non-manifold edges (shared by
//! more than two faces) and inconsistent winding all break the assumptions that
//! inside/outside queries and volume integrals rely on.
//!
//! This module computes those diagnostics from the mesh's edge-use structure:
//!
//! - every triangle contributes three directed edges; edges are keyed by their
//!   unordered vertex pair,
//! - an edge used by exactly one triangle is a *boundary* (hole) edge,
//! - an edge used by more than two triangles is *non-manifold*, and
//! - a shared edge whose two triangles traverse it in the *same* direction
//!   indicates inconsistent winding.
//!
//! Near-coincident vertices are welded first (reusing
//! [`weld_mesh`](crate::collider::weld_mesh)) so hairline cracks are not
//! misreported as holes. This is standard mesh validation; nothing here is
//! derived from Unreal Engine source.

use glam::Vec3;
use std::collections::{HashMap, HashSet};

use crate::collider::weld::{weld_mesh, WeldParams};

/// Diagnostics describing the connectivity quality of a triangle mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshTopology {
    /// Triangles considered (after welding, excluding degenerate faces).
    pub triangles: usize,
    /// Distinct unordered edges.
    pub edges: usize,
    /// Edges used by exactly one triangle (open boundary / holes).
    pub boundary_edges: usize,
    /// Edges used by more than two triangles (non-manifold).
    pub non_manifold_edges: usize,
    /// Edges shared by two triangles that traverse them with the same winding.
    pub inconsistent_edges: usize,
    /// Euler characteristic `V - E + F` of the welded mesh. For a single closed
    /// genus-0 surface this is `2`.
    pub euler_characteristic: i64,
}

impl MeshTopology {
    /// Whether no edge is shared by more than two triangles.
    #[must_use]
    pub fn is_manifold(&self) -> bool {
        self.non_manifold_edges == 0
    }

    /// Whether the surface is closed (watertight): manifold with no boundary
    /// edges.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.is_manifold() && self.boundary_edges == 0
    }

    /// Whether every shared edge is traversed in opposite directions by its two
    /// triangles (globally consistent winding).
    #[must_use]
    pub fn is_consistently_oriented(&self) -> bool {
        self.inconsistent_edges == 0
    }

    /// Whether the mesh is a clean closed, consistently wound 2-manifold: the
    /// precondition SDF cooking and volume integrals want.
    #[must_use]
    pub fn is_watertight_manifold(&self) -> bool {
        self.is_closed() && self.is_consistently_oriented()
    }
}

/// Per-edge accumulator: how many triangles touch it and the net winding
/// balance (sum of `+1` for the canonical direction, `-1` for the reverse).
#[derive(Clone, Copy, Default)]
struct EdgeUse {
    count: u32,
    balance: i32,
}

/// Analyses the topology of a welded triangle mesh.
///
/// Vertices closer than `weld_epsilon` are merged before analysis so cracks are
/// not misread as holes. Returns `None` when `vertices` or `indices` is empty,
/// when `weld_epsilon` is not finite and strictly positive, or when welding
/// leaves no non-degenerate triangle.
#[must_use]
pub fn analyze_topology(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    weld_epsilon: f32,
) -> Option<MeshTopology> {
    let welded = weld_mesh(
        vertices,
        indices,
        WeldParams {
            position_epsilon: weld_epsilon,
            drop_duplicate_triangles: false,
        },
    )?;

    if welded.indices.is_empty() {
        return None;
    }

    let mut edges: HashMap<(u32, u32), EdgeUse> = HashMap::new();
    let mut touched_vertices: HashSet<u32> = HashSet::new();
    let record = |a: u32, b: u32, edges: &mut HashMap<(u32, u32), EdgeUse>| {
        let (key, dir) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
        let e = edges.entry(key).or_default();
        e.count += 1;
        e.balance += dir;
    };

    for tri in &welded.indices {
        touched_vertices.insert(tri[0]);
        touched_vertices.insert(tri[1]);
        touched_vertices.insert(tri[2]);
        record(tri[0], tri[1], &mut edges);
        record(tri[1], tri[2], &mut edges);
        record(tri[2], tri[0], &mut edges);
    }

    let mut boundary_edges = 0usize;
    let mut non_manifold_edges = 0usize;
    let mut inconsistent_edges = 0usize;
    for e in edges.values() {
        match e.count {
            1 => boundary_edges += 1,
            2 => {
                // A clean manifold edge is traversed once each way (balance 0).
                if e.balance != 0 {
                    inconsistent_edges += 1;
                }
            }
            _ => non_manifold_edges += 1,
        }
    }

    let v = touched_vertices.len() as i64;
    let e = edges.len() as i64;
    let f = welded.indices.len() as i64;

    Some(MeshTopology {
        triangles: welded.indices.len(),
        edges: edges.len(),
        boundary_edges,
        non_manifold_edges,
        inconsistent_edges,
        euler_characteristic: v - e + f,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    /// Builds a closed, welded icosahedron subdivided `levels` times: a
    /// watertight, consistently wound 2-manifold.
    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let mut verts: Vec<Vec3> = vec![
            Vec3::new(-1.0, t, 0.0),
            Vec3::new(1.0, t, 0.0),
            Vec3::new(-1.0, -t, 0.0),
            Vec3::new(1.0, -t, 0.0),
            Vec3::new(0.0, -1.0, t),
            Vec3::new(0.0, 1.0, t),
            Vec3::new(0.0, -1.0, -t),
            Vec3::new(0.0, 1.0, -t),
            Vec3::new(t, 0.0, -1.0),
            Vec3::new(t, 0.0, 1.0),
            Vec3::new(-t, 0.0, -1.0),
            Vec3::new(-t, 0.0, 1.0),
        ];
        for v in &mut verts {
            *v = v.normalize();
        }
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut cache: StdHashMap<(u32, u32), u32> = StdHashMap::new();
            let mut next: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&m) = cache.get(&key) {
                    return m;
                }
                let m = verts.len() as u32;
                let p = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                verts.push(p);
                cache.insert(key, m);
                m
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                next.push([f[0], a, c]);
                next.push([f[1], b, a]);
                next.push([f[2], c, b]);
                next.push([a, b, c]);
            }
            faces = next;
        }
        (verts, faces)
    }

    #[test]
    fn closed_sphere_is_watertight_manifold() {
        let (v, t) = icosphere(2);
        let topo = analyze_topology(&v, &t, 1e-5).expect("analyses");
        assert!(topo.is_manifold());
        assert!(topo.is_closed());
        assert!(topo.is_consistently_oriented());
        assert!(topo.is_watertight_manifold());
        assert_eq!(topo.boundary_edges, 0);
        assert_eq!(topo.non_manifold_edges, 0);
        assert_eq!(topo.inconsistent_edges, 0);
        // A closed genus-0 surface has Euler characteristic 2.
        assert_eq!(topo.euler_characteristic, 2);
        // Closed triangle mesh: E = 3F/2.
        assert_eq!(topo.edges * 2, topo.triangles * 3);
    }

    #[test]
    fn single_triangle_has_three_boundary_edges() {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let t = vec![[0u32, 1, 2]];
        let topo = analyze_topology(&v, &t, 1e-5).expect("analyses");
        assert_eq!(topo.boundary_edges, 3);
        assert!(topo.is_manifold(), "boundary edges are still manifold");
        assert!(!topo.is_closed(), "an open sheet is not watertight");
        assert!(topo.is_consistently_oriented());
    }

    #[test]
    fn open_sphere_with_hole_is_not_closed() {
        let (v, mut t) = icosphere(1);
        // Punch a hole by removing one triangle: its three edges become
        // boundaries.
        t.pop();
        let topo = analyze_topology(&v, &t, 1e-5).expect("analyses");
        assert!(topo.is_manifold());
        assert!(!topo.is_closed());
        assert_eq!(topo.boundary_edges, 3);
        assert!(topo.is_consistently_oriented());
        // Removing one face from a closed surface drops Euler char to 1.
        assert_eq!(topo.euler_characteristic, 1);
    }

    #[test]
    fn fan_of_three_triangles_sharing_an_edge_is_non_manifold() {
        // Three triangles share the edge (0,1): a non-manifold "T"-junction.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let t = vec![[0u32, 1, 2], [0, 1, 3], [0, 1, 4]];
        let topo = analyze_topology(&v, &t, 1e-5).expect("analyses");
        assert!(!topo.is_manifold());
        assert_eq!(topo.non_manifold_edges, 1);
    }

    #[test]
    fn flipped_neighbour_is_inconsistently_oriented() {
        // Two triangles share edge (1,2). The first winds 0->1->2, the second
        // is deliberately wound so it also traverses 1->2 (same direction),
        // which is an orientation flip.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        // tri A: 0,1,2 -> directed edge (1,2)
        // tri B: 1,2,3 -> directed edge (1,2) as well (same direction) = flip
        let t = vec![[0u32, 1, 2], [1, 2, 3]];
        let topo = analyze_topology(&v, &t, 1e-5).expect("analyses");
        assert_eq!(topo.inconsistent_edges, 1);
        assert!(!topo.is_consistently_oriented());
        assert!(topo.is_manifold());
    }

    #[test]
    fn cracked_sphere_welds_closed() {
        // Duplicate every vertex so the two authored copies form a cracked mesh
        // referencing distinct indices; welding at a loose epsilon should fuse
        // them back into a closed manifold.
        let (base_v, base_t) = icosphere(1);
        let mut v: Vec<Vec3> = Vec::new();
        // Copy A sits exactly on the surface; copy B is nudged ~1e-4 away so
        // the seam only fuses once the weld tolerance exceeds that gap.
        for &p in &base_v {
            v.push(p);
            v.push(p + Vec3::new(1e-4, 0.0, 0.0));
        }
        // Half the faces use copy A, half use copy B, so the seam only closes
        // after welding.
        let t: Vec<[u32; 3]> = base_t
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let off = (i % 2) as u32;
                [f[0] * 2 + off, f[1] * 2 + off, f[2] * 2 + off]
            })
            .collect();

        let unwelded = analyze_topology(&v, &t, 1e-9).expect("analyses");
        assert!(
            !unwelded.is_closed(),
            "distinct index copies leave open seams"
        );

        let welded = analyze_topology(&v, &t, 1e-3).expect("analyses");
        assert!(welded.is_closed(), "welding fuses the seam closed");
        assert!(welded.is_watertight_manifold());
    }

    #[test]
    fn empty_or_invalid_input_is_rejected() {
        assert!(analyze_topology(&[], &[], 1e-5).is_none());
        let (v, t) = icosphere(0);
        assert!(analyze_topology(&v, &t, 0.0).is_none());
        assert!(analyze_topology(&v, &t, f32::NAN).is_none());
    }
}
