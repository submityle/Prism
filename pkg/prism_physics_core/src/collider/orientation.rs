//! Winding-consistency repair and outward reorientation for triangle meshes.
//!
//! [`crate::collider::analyze_topology`] can *detect* inconsistent winding
//! (via `MeshTopology::is_consistently_oriented`), but detection alone does not
//! fix a mesh whose triangles disagree about which side is "out". This module
//! supplies the repair step [`orient_outward`]:
//!
//! 1. Co-located vertices are welded into shared ids so a seam duplicating a
//!    vertex does not split the adjacency graph.
//! 2. A breadth-first traversal across manifold edges propagates a per-triangle
//!    flip bit so every connected component becomes internally consistent.
//! 3. Each component's signed volume (divergence theorem) decides whether the
//!    whole component faces inward; inward components are flipped so normals
//!    point outward.
//!
//! The final per-triangle flip is the exclusive-or of the consistency bit and
//! the per-component outward bit. This is textbook mesh sanitation; nothing
//! here is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::HashMap;

/// Result of reorienting a triangle mesh so every connected component is
/// consistently wound and faces outward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrientedMesh {
    /// Reindexed triangles. Vertex indices are unchanged; only the per-triangle
    /// vertex *order* (winding) may differ from the input.
    pub indices: Vec<[u32; 3]>,
    /// How many triangles had their winding reversed relative to the input.
    pub flipped: usize,
    /// Number of connected components discovered (degenerate triangles that
    /// collapse to a line or point each form their own singleton component).
    pub components: usize,
    /// `true` when some manifold edge forced contradictory flip bits, i.e. the
    /// surface cannot be made globally consistent (non-orientable like a
    /// Moebius strip, or corrupted by non-manifold stitching). The output is
    /// still the best consistent assignment reachable by the traversal.
    pub had_conflicts: bool,
}

impl OrientedMesh {
    /// Number of triangles in the reoriented mesh.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Whether the mesh carries no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Assigns a weld id to every vertex, merging positions that quantise to the
/// same cell so duplicated seam vertices share connectivity.
fn weld_ids(vertices: &[Vec3], eps: f32) -> Vec<u32> {
    let inv = f64::from(eps).recip();
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut ids: Vec<u32> = Vec::with_capacity(vertices.len());
    let mut next: u32 = 0;
    for v in vertices {
        let key = (
            (f64::from(v.x) * inv).round() as i64,
            (f64::from(v.y) * inv).round() as i64,
            (f64::from(v.z) * inv).round() as i64,
        );
        let id = *map.entry(key).or_insert_with(|| {
            let assigned = next;
            next += 1;
            assigned
        });
        ids.push(id);
    }
    ids
}

/// Scalar triple product `p0 . (p1 x p2)` accumulated in `f64` for a stable
/// signed-volume sum regardless of mesh scale.
fn triple_f64(p0: Vec3, p1: Vec3, p2: Vec3) -> f64 {
    let a = (f64::from(p0.x), f64::from(p0.y), f64::from(p0.z));
    let b = (f64::from(p1.x), f64::from(p1.y), f64::from(p1.z));
    let c = (f64::from(p2.x), f64::from(p2.y), f64::from(p2.z));
    let cross = (
        b.1 * c.2 - b.2 * c.1,
        b.2 * c.0 - b.0 * c.2,
        b.0 * c.1 - b.1 * c.0,
    );
    a.0 * cross.0 + a.1 * cross.1 + a.2 * cross.2
}

/// Reorients a triangle mesh so every connected component is consistently wound
/// and faces outward (normals point away from the enclosed volume).
///
/// Returns `None` when `vertices` or `indices` is empty. Vertex indices in the
/// output are untouched; only the winding of each triangle may change. Flat
/// components whose signed volume is numerically ambiguous keep their
/// consistency-repaired winding without an outward flip.
#[must_use]
pub fn orient_outward(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<OrientedMesh> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    // Weld epsilon tracks the mesh scale so hairline seams merge but genuinely
    // distinct vertices stay apart.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for v in vertices {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    let diag = (hi - lo).length();
    let eps = (diag * 1e-6).max(1e-9);
    let wid = weld_ids(vertices, eps);
    let vcount = vertices.len() as u32;

    let tri_count = indices.len();

    // Per triangle: the undirected weld-edge keys it touches plus the direction
    // (+1 when traversed low->high weld id, -1 otherwise). Degenerate triangles
    // (two welded corners coincide) contribute no edges and stay isolated.
    let mut tri_edges: Vec<Vec<((u32, u32), i32)>> = Vec::with_capacity(tri_count);
    let mut edge_map: HashMap<(u32, u32), Vec<(usize, i32)>> = HashMap::new();

    for (ti, tri) in indices.iter().enumerate() {
        // Guard against out-of-range indices: bail if any triangle references a
        // vertex we do not have.
        if tri[0] >= vcount || tri[1] >= vcount || tri[2] >= vcount {
            return None;
        }
        let w = [
            wid[tri[0] as usize],
            wid[tri[1] as usize],
            wid[tri[2] as usize],
        ];
        let degenerate = w[0] == w[1] || w[1] == w[2] || w[0] == w[2];
        let mut edges: Vec<((u32, u32), i32)> = Vec::new();
        if !degenerate {
            for &(u, v) in &[(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
                let (key, dir) = if u < v { ((u, v), 1) } else { ((v, u), -1) };
                edges.push((key, dir));
                edge_map.entry(key).or_default().push((ti, dir));
            }
        }
        tri_edges.push(edges);
    }

    // Breadth-first traversal: `sign[t]` is +1 to keep the input winding of
    // triangle `t`, -1 to reverse it for consistency with its seed.
    let mut sign: Vec<i32> = vec![0; tri_count];
    let mut component: Vec<usize> = vec![usize::MAX; tri_count];
    let mut had_conflicts = false;
    let mut components = 0usize;
    let mut stack: Vec<usize> = Vec::new();

    for seed in 0..tri_count {
        if sign[seed] != 0 {
            continue;
        }
        let comp = components;
        components += 1;
        sign[seed] = 1;
        component[seed] = comp;
        stack.push(seed);
        while let Some(t) = stack.pop() {
            let st = sign[t];
            for &(key, dir_t) in &tri_edges[t] {
                let Some(uses) = edge_map.get(&key) else {
                    continue;
                };
                // Only clean manifold edges (shared by exactly two triangles)
                // transmit orientation. Boundary and non-manifold edges stop
                // propagation.
                if uses.len() != 2 {
                    continue;
                }
                for &(other, dir_n) in uses {
                    if other == t {
                        continue;
                    }
                    // Consistency requires the two triangles traverse the shared
                    // edge in opposite effective directions:
                    //   sign[t]*dir_t = -sign[n]*dir_n
                    // => sign[n] = -dir_t*dir_n*sign[t].
                    let expected = -dir_t * dir_n * st;
                    if sign[other] == 0 {
                        sign[other] = expected;
                        component[other] = comp;
                        stack.push(other);
                    } else if sign[other] != expected {
                        had_conflicts = true;
                    }
                }
            }
        }
    }

    // Per-component signed volume under the consistency-repaired winding decides
    // whether the whole component faces inward and must be flipped outward.
    let d = f64::from(diag);
    let tiny = (d * d * d * 1e-9).max(1e-20);
    let mut comp_volume: Vec<f64> = vec![0.0; components];
    for (ti, tri) in indices.iter().enumerate() {
        if tri_edges[ti].is_empty() {
            // Degenerate triangle: zero area, no contribution.
            continue;
        }
        let p0 = vertices[tri[0] as usize];
        let p1 = vertices[tri[1] as usize];
        let p2 = vertices[tri[2] as usize];
        let vol = if sign[ti] < 0 {
            triple_f64(p0, p2, p1)
        } else {
            triple_f64(p0, p1, p2)
        };
        comp_volume[component[ti]] += vol;
    }
    let comp_flip: Vec<bool> = comp_volume.iter().map(|&vol| vol < -tiny).collect();

    // Final winding = consistency flip XOR per-component outward flip.
    let mut out_indices: Vec<[u32; 3]> = Vec::with_capacity(tri_count);
    let mut flipped = 0usize;
    for (ti, tri) in indices.iter().enumerate() {
        let consistency_flip = sign[ti] < 0;
        let vol_flip = component[ti] != usize::MAX && comp_flip[component[ti]];
        let flip = consistency_flip ^ vol_flip;
        if flip {
            out_indices.push([tri[0], tri[2], tri[1]]);
            flipped += 1;
        } else {
            out_indices.push([tri[0], tri[1], tri[2]]);
        }
    }

    Some(OrientedMesh {
        indices: out_indices,
        flipped,
        components,
        had_conflicts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::analyze_topology;
    use std::collections::HashMap as StdHashMap;

    fn cube_mesh(half: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let h = half;
        let verts = vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ];
        // Outward-facing winding (CCW seen from outside).
        let idx = vec![
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
        (verts, idx)
    }

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

    fn reverse_all(idx: &[[u32; 3]]) -> Vec<[u32; 3]> {
        idx.iter().map(|t| [t[0], t[2], t[1]]).collect()
    }

    fn mangle(idx: &[[u32; 3]]) -> Vec<[u32; 3]> {
        idx.iter()
            .enumerate()
            .map(|(i, t)| if i % 2 == 0 { [t[0], t[2], t[1]] } else { *t })
            .collect()
    }

    fn signed_volume(v: &[Vec3], idx: &[[u32; 3]]) -> f32 {
        let mut s = 0.0f32;
        for t in idx {
            let a = v[t[0] as usize];
            let b = v[t[1] as usize];
            let c = v[t[2] as usize];
            s += a.dot(b.cross(c));
        }
        s / 6.0
    }

    #[test]
    fn empty_input_returns_none() {
        let (v, i) = cube_mesh(1.0);
        assert!(orient_outward(&[], &i).is_none());
        assert!(orient_outward(&v, &[]).is_none());
    }

    #[test]
    fn fully_inward_cube_is_flipped_outward() {
        let (v, i) = cube_mesh(1.0);
        let inward = reverse_all(&i);
        assert!(signed_volume(&v, &inward) < 0.0);
        let out = orient_outward(&v, &inward).unwrap();
        assert_eq!(out.components, 1);
        assert!(!out.had_conflicts);
        assert!(signed_volume(&v, &out.indices) > 0.0);
        let topo = analyze_topology(&v, &out.indices, 1e-6).unwrap();
        assert!(topo.is_consistently_oriented());
        // Every face was pointing inward, so all 12 flip.
        assert_eq!(out.flipped, 12);
    }

    #[test]
    fn mangled_cube_becomes_consistent_outward() {
        let (v, i) = cube_mesh(1.0);
        let messy = mangle(&i);
        let out = orient_outward(&v, &messy).unwrap();
        assert_eq!(out.components, 1);
        assert!(!out.had_conflicts);
        assert!(signed_volume(&v, &out.indices) > 0.0);
        let topo = analyze_topology(&v, &out.indices, 1e-6).unwrap();
        assert!(topo.is_consistently_oriented());
    }

    #[test]
    fn already_correct_cube_needs_no_flip() {
        let (v, i) = cube_mesh(1.0);
        let out = orient_outward(&v, &i).unwrap();
        assert_eq!(out.components, 1);
        assert_eq!(out.flipped, 0);
        assert!(!out.had_conflicts);
        let topo = analyze_topology(&v, &out.indices, 1e-6).unwrap();
        assert!(topo.is_consistently_oriented());
    }

    #[test]
    fn mangled_sphere_becomes_consistent_outward() {
        let (v, i) = icosphere(2);
        let messy = mangle(&i);
        let out = orient_outward(&v, &messy).unwrap();
        assert_eq!(out.components, 1);
        assert!(!out.had_conflicts);
        assert!(signed_volume(&v, &out.indices) > 0.0);
        let topo = analyze_topology(&v, &out.indices, 1e-6).unwrap();
        assert!(topo.is_consistently_oriented());
    }

    #[test]
    fn orientation_is_deterministic() {
        let (v, i) = icosphere(2);
        let messy = mangle(&i);
        let a = orient_outward(&v, &messy).unwrap();
        let b = orient_outward(&v, &messy).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn signed_volume_magnitude_is_preserved() {
        let (v, i) = cube_mesh(1.3);
        let messy = mangle(&i);
        let before = signed_volume(&v, &i).abs();
        let out = orient_outward(&v, &messy).unwrap();
        let after = signed_volume(&v, &out.indices).abs();
        assert!((before - after).abs() < 1e-4);
    }

    #[test]
    fn two_separate_components_each_outward() {
        let (c0v, c0i) = cube_mesh(1.0);
        // Second cube translated far enough that welding keeps it distinct, and
        // wound inward so the repair must flip it.
        let offset = Vec3::new(5.0, 0.0, 0.0);
        let mut v: Vec<Vec3> = c0v.clone();
        v.extend(c0v.iter().map(|p| *p + offset));
        let base = c0v.len() as u32;
        let mut i: Vec<[u32; 3]> = c0i.clone();
        for t in reverse_all(&c0i) {
            i.push([t[0] + base, t[1] + base, t[2] + base]);
        }
        let out = orient_outward(&v, &i).unwrap();
        assert_eq!(out.components, 2);
        assert!(!out.had_conflicts);
        assert!(signed_volume(&v, &out.indices) > 0.0);
        let topo = analyze_topology(&v, &out.indices, 1e-6).unwrap();
        assert!(topo.is_consistently_oriented());
    }
}
