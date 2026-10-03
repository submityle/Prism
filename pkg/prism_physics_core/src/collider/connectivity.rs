//! Connected-component splitting for triangle-soup collision geometry.
//!
//! Authoring and procedural tools frequently hand the cooker a single index
//! buffer that actually contains several disjoint shells: pre-fractured debris,
//! a prop plus its loose bolts, or many instanced chunks baked into one mesh.
//! Convex decomposition, inertia estimation and per-chunk `BVH` builds all want
//! those shells separated first. AAA cookers (`PhysX`, `Jolt`) perform exactly
//! this island split before downstream processing.
//!
//! This module performs the split deterministically:
//!
//! - vertices that are near-coincident (within
//!   [`ConnectivityParams::weld_epsilon`]) are joined so hairline cracks between
//!   two index islands still count as one shell, using a uniform spatial hash so
//!   the cost stays roughly linear in the vertex count,
//! - triangles that share any (welded) vertex are joined with a union-find pass,
//!   and
//! - each resulting component is re-emitted against the *original* vertex
//!   positions with a compact local index buffer.
//!
//! Output ordering is deterministic: components are sorted by the smallest
//! original triangle index they contain, and within a component the triangles
//! keep their original relative order. This is standard mesh connectivity;
//! nothing here is derived from Unreal Engine source.

use glam::Vec3;
use std::collections::HashMap;

/// Parameters controlling how triangles are grouped into shells.
#[derive(Clone, Copy, Debug)]
pub struct ConnectivityParams {
    /// Vertices closer than this distance are treated as the same point when
    /// deciding connectivity, so two index islands separated only by a hairline
    /// crack still merge into one shell. Must be finite and non-negative; a
    /// value of `0.0` disables position welding and uses purely topological
    /// (shared-index) connectivity.
    pub weld_epsilon: f32,
}

impl Default for ConnectivityParams {
    fn default() -> Self {
        Self { weld_epsilon: 1e-5 }
    }
}

impl ConnectivityParams {
    /// Purely topological connectivity: triangles join only when they literally
    /// share a vertex index. No position welding is performed.
    #[must_use]
    pub fn exact() -> Self {
        Self { weld_epsilon: 0.0 }
    }

    /// Connectivity with position welding at the given tolerance, so shells
    /// separated by cracks narrower than `epsilon` still merge.
    #[must_use]
    pub fn welded(epsilon: f32) -> Self {
        Self {
            weld_epsilon: epsilon,
        }
    }
}

/// One connected shell extracted from a triangle soup.
///
/// The geometry is self-contained: `indices` reference `vertices`, which are a
/// compact copy of just the positions this shell uses, taken verbatim from the
/// input mesh.
#[derive(Clone, Debug)]
pub struct MeshComponent {
    /// Positions used by this shell, copied from the input (no welding applied
    /// to the emitted coordinates).
    pub vertices: Vec<Vec3>,
    /// Triangles reindexed into [`MeshComponent::vertices`].
    pub indices: Vec<[u32; 3]>,
}

impl MeshComponent {
    /// Number of triangles in this shell.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Number of distinct vertices this shell references.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Whether this shell carries no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// A disjoint-set (union-find) structure with path compression and union by
/// size, over a fixed number of elements.
struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count as u32).collect(),
            size: vec![1; count],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // Path halving: point each node at its grandparent as we climb.
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Attach the smaller tree under the larger one.
        let (small, large) = if self.size[ra as usize] < self.size[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = large;
        self.size[large as usize] += self.size[small as usize];
    }
}

/// Splits a triangle soup into its connected components (shells).
///
/// Two triangles belong to the same component when they share a vertex, either
/// by index or (when [`ConnectivityParams::weld_epsilon`] is positive) by
/// position. Each component is returned against a compact copy of the original
/// vertex positions it uses.
///
/// Returns `None` when `vertices` or `indices` is empty, when `weld_epsilon` is
/// not finite or is negative, or when no in-range, non-degenerate triangle
/// survives. Triangles that reference out-of-range vertices, or that repeat a
/// vertex index (zero-area), are skipped and do not appear in any component.
///
/// Components are ordered deterministically by the smallest original triangle
/// index they contain, and triangles keep their original relative order within
/// a component.
#[must_use]
pub fn split_connected_components(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: ConnectivityParams,
) -> Option<Vec<MeshComponent>> {
    if vertices.is_empty()
        || indices.is_empty()
        || !params.weld_epsilon.is_finite()
        || params.weld_epsilon < 0.0
    {
        return None;
    }

    let vertex_count = vertices.len();
    let mut uf = UnionFind::new(vertex_count);

    // Position welding: union near-coincident vertices via a uniform spatial
    // hash. Each vertex is unioned with the first already-placed vertex found
    // within `epsilon`; transitivity through the union-find handles clusters.
    if params.weld_epsilon > 0.0 {
        let eps = params.weld_epsilon;
        let eps_sq = eps * eps;
        let inv_cell = 1.0 / eps;
        let cell_of = |p: Vec3| -> (i64, i64, i64) {
            (
                (p.x * inv_cell).floor() as i64,
                (p.y * inv_cell).floor() as i64,
                (p.z * inv_cell).floor() as i64,
            )
        };
        let mut buckets: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        for (idx, &v) in vertices.iter().enumerate() {
            let idx = idx as u32;
            let (cx, cy, cz) = cell_of(v);
            'search: for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(list) = buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &other in list {
                                if (vertices[other as usize] - v).length_squared() <= eps_sq {
                                    uf.union(idx, other);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
            buckets.entry((cx, cy, cz)).or_default().push(idx);
        }
    }

    // Topological union: triangles that share any in-range vertex join. Only
    // triangles whose three indices are all in range contribute.
    let in_range = |t: &[u32; 3]| -> bool {
        (t[0] as usize) < vertex_count
            && (t[1] as usize) < vertex_count
            && (t[2] as usize) < vertex_count
    };
    for tri in indices {
        if !in_range(tri) {
            continue;
        }
        uf.union(tri[0], tri[1]);
        uf.union(tri[1], tri[2]);
    }

    // Group emittable triangles (in range and non-degenerate) by their
    // component root, recording the smallest original index per group for a
    // deterministic ordering.
    struct Group {
        min_tri: usize,
        tris: Vec<usize>,
    }
    let mut groups: HashMap<u32, Group> = HashMap::new();
    for (ti, tri) in indices.iter().enumerate() {
        if !in_range(tri) || tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
            continue;
        }
        let root = uf.find(tri[0]);
        let group = groups.entry(root).or_insert_with(|| Group {
            min_tri: ti,
            tris: Vec::new(),
        });
        group.tris.push(ti);
    }

    if groups.is_empty() {
        return None;
    }

    let mut ordered: Vec<Group> = groups.into_values().collect();
    ordered.sort_by_key(|g| g.min_tri);

    let mut components = Vec::with_capacity(ordered.len());
    for group in ordered {
        let mut local: HashMap<u32, u32> = HashMap::new();
        let mut verts: Vec<Vec3> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(group.tris.len());
        for &ti in &group.tris {
            let tri = indices[ti];
            let mut remapped = [0u32; 3];
            for (slot, &vi) in remapped.iter_mut().zip(tri.iter()) {
                let next = local.len() as u32;
                let local_index = *local.entry(vi).or_insert_with(|| {
                    verts.push(vertices[vi as usize]);
                    next
                });
                *slot = local_index;
            }
            tris.push(remapped);
        }
        components.push(MeshComponent {
            vertices: verts,
            indices: tris,
        });
    }

    Some(components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    /// Builds a closed, welded icosahedron subdivided `levels` times: a
    /// watertight 2-manifold unit sphere.
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

    /// Appends `mesh` to `(verts, tris)`, shifting the triangle indices and
    /// translating the vertices by `offset`.
    fn append_shifted(
        verts: &mut Vec<Vec3>,
        tris: &mut Vec<[u32; 3]>,
        mesh: &(Vec<Vec3>, Vec<[u32; 3]>),
        offset: Vec3,
    ) {
        let base = verts.len() as u32;
        for &v in &mesh.0 {
            verts.push(v + offset);
        }
        for t in &mesh.1 {
            tris.push([t[0] + base, t[1] + base, t[2] + base]);
        }
    }

    #[test]
    fn single_shell_is_one_component() {
        let (v, t) = icosphere(2);
        let parts = split_connected_components(&v, &t, ConnectivityParams::default())
            .expect("non-empty mesh splits");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].triangle_count(), t.len());
        assert_eq!(parts[0].vertex_count(), v.len());
    }

    #[test]
    fn two_separated_shells_are_two_components() {
        let sphere = icosphere(1);
        let mut v = Vec::new();
        let mut t = Vec::new();
        append_shifted(&mut v, &mut t, &sphere, Vec3::ZERO);
        append_shifted(&mut v, &mut t, &sphere, Vec3::new(10.0, 0.0, 0.0));
        let parts = split_connected_components(&v, &t, ConnectivityParams::default())
            .expect("two shells split");
        assert_eq!(parts.len(), 2);
        for p in &parts {
            assert_eq!(p.triangle_count(), sphere.1.len());
            assert_eq!(p.vertex_count(), sphere.0.len());
            // Each component must be independently valid (indices in range).
            let n = p.vertices.len() as u32;
            for tri in &p.indices {
                assert!(tri.iter().all(|&i| i < n));
            }
        }
    }

    #[test]
    fn three_shells_count_and_triangles_preserved() {
        let sphere = icosphere(1);
        let mut v = Vec::new();
        let mut t = Vec::new();
        append_shifted(&mut v, &mut t, &sphere, Vec3::new(-10.0, 0.0, 0.0));
        append_shifted(&mut v, &mut t, &sphere, Vec3::ZERO);
        append_shifted(&mut v, &mut t, &sphere, Vec3::new(10.0, 0.0, 0.0));
        let parts = split_connected_components(&v, &t, ConnectivityParams::exact())
            .expect("three shells split");
        assert_eq!(parts.len(), 3);
        let total: usize = parts.iter().map(MeshComponent::triangle_count).sum();
        assert_eq!(total, t.len());
    }

    #[test]
    fn crack_welding_merges_touching_shells() {
        // Two spheres whose nearest points nearly touch (gap < epsilon) but use
        // distinct vertex indices: exact mode keeps them split, welding joins.
        let sphere = icosphere(2);
        let mut v = Vec::new();
        let mut t = Vec::new();
        append_shifted(&mut v, &mut t, &sphere, Vec3::ZERO);
        // Diameter is 2 (unit sphere); place the second so the surfaces are
        // 1e-4 apart along +x: centre at 2 + 1e-4.
        append_shifted(&mut v, &mut t, &sphere, Vec3::new(2.0 + 1e-4, 0.0, 0.0));

        let exact =
            split_connected_components(&v, &t, ConnectivityParams::exact()).expect("exact split");
        assert_eq!(exact.len(), 2, "distinct indices stay split in exact mode");

        let welded = split_connected_components(&v, &t, ConnectivityParams::welded(1e-3))
            .expect("welded split");
        assert_eq!(
            welded.len(),
            1,
            "near-touching surfaces weld into one shell"
        );
    }

    #[test]
    fn ordering_is_deterministic_and_by_first_triangle() {
        let sphere = icosphere(1);
        let mut v = Vec::new();
        let mut t = Vec::new();
        // Component A is authored first, component B second.
        append_shifted(&mut v, &mut t, &sphere, Vec3::ZERO);
        let a_tris = t.len();
        append_shifted(&mut v, &mut t, &sphere, Vec3::new(10.0, 0.0, 0.0));

        let run1 = split_connected_components(&v, &t, ConnectivityParams::default()).unwrap();
        let run2 = split_connected_components(&v, &t, ConnectivityParams::default()).unwrap();
        assert_eq!(run1.len(), 2);
        // Determinism across runs.
        assert_eq!(run1[0].indices, run2[0].indices);
        assert_eq!(run1[1].indices, run2[1].indices);
        // First component owns the first-authored triangles.
        assert_eq!(run1[0].triangle_count(), a_tris);
    }

    #[test]
    fn out_of_range_and_degenerate_triangles_are_skipped() {
        let (mut v, mut t) = icosphere(1);
        let good = t.len();
        let n = v.len() as u32;
        // Out-of-range triangle.
        t.push([n, n + 1, n + 2]);
        // Degenerate (repeated index) triangle on an in-range vertex.
        t.push([0, 0, 1]);
        let _ = &mut v;
        let parts = split_connected_components(&v, &t, ConnectivityParams::default())
            .expect("valid geometry remains");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].triangle_count(), good);
    }

    #[test]
    fn empty_input_is_rejected() {
        assert!(split_connected_components(&[], &[], ConnectivityParams::default()).is_none());
        let (v, _) = icosphere(0);
        assert!(split_connected_components(&v, &[], ConnectivityParams::default()).is_none());
    }

    #[test]
    fn non_finite_epsilon_is_rejected() {
        let (v, t) = icosphere(0);
        assert!(split_connected_components(&v, &t, ConnectivityParams::welded(f32::NAN)).is_none());
        assert!(split_connected_components(&v, &t, ConnectivityParams::welded(-1.0)).is_none());
    }

    #[test]
    fn all_invalid_triangles_yield_none() {
        let (v, _) = icosphere(0);
        let n = v.len() as u32;
        let t = vec![[n, n + 1, n + 2], [0, 0, 0]];
        assert!(split_connected_components(&v, &t, ConnectivityParams::default()).is_none());
    }
}
