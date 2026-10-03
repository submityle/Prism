//! Quadric-error edge-collapse decimation for triangle-mesh colliders.
//!
//! A render mesh cooked straight into a collision shape usually carries far more
//! triangles than the solver needs: broad-phase refit, mid-phase BVH traversal
//! and narrow-phase triangle tests all scale with the triangle count, so every
//! engine ships a *collision LOD* that keeps the silhouette but spends a
//! fraction of the triangles (`PhysX` `PxTriangleMeshCookingResult`, `Jolt`
//! mesh reduction, Chaos `FMeshSimplifier`). [`decimate_mesh`] is that reducer.
//!
//! # Algorithm
//!
//! Classic Garland--Heckbert quadric-error-metric (QEM) decimation:
//!
//! - every vertex accumulates the [`Quadric`] of its incident triangle planes,
//!   plus a heavily weighted perpendicular plane on each open boundary edge so
//!   the silhouette is pinned,
//! - every edge is scored by the error of collapsing it to the quadric-optimal
//!   point and placed in a min-heap keyed on that cost (with an index tie-break
//!   so the result is bit-for-bit deterministic),
//! - the cheapest legal collapse is applied repeatedly until the triangle target
//!   or the error ceiling is hit. A collapse is legal only when it satisfies the
//!   edge *link condition* (so the mesh stays 2-manifold) and flips no incident
//!   triangle normal (so the surface does not fold through itself).
//!
//! Heap entries are invalidated lazily by a per-vertex version counter, so a
//! collapse only needs to re-score the edges around the merged vertex.
//!
//! Everything here is the standard QEM method (Garland & Heckbert, SIGGRAPH
//! 1997; link condition after Dey et al.); nothing is derived from Unreal
//! Engine source.

use super::quadric::Quadric;
use alloc::collections::BinaryHeap;
use glam::Vec3;
use std::collections::{HashMap, HashSet};

/// Weight applied to boundary-edge constraint planes. Large enough that open
/// edges are effectively frozen relative to interior error.
const BOUNDARY_WEIGHT: f32 = 1.0e3;
/// Smallest triangle count a closed mesh can be reduced to (a tetrahedron).
const MIN_TRIANGLES: usize = 4;

/// How many triangles [`decimate_mesh`] should aim to leave behind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecimateTarget {
    /// Stop at (approximately) this many triangles.
    TriangleCount(usize),
    /// Keep this fraction of the input triangles, clamped to `(0, 1]`.
    Ratio(f32),
}

/// Parameters controlling a [`decimate_mesh`] run.
#[derive(Clone, Copy, Debug)]
pub struct DecimateParams {
    /// Triangle budget for the result.
    pub target: DecimateTarget,
    /// Collapses whose quadric error exceeds this ceiling are never applied,
    /// even if the triangle target has not been reached. Use `f32::INFINITY` to
    /// decimate purely to the triangle budget.
    pub max_error: f32,
}

impl DecimateParams {
    /// Decimate down to `triangles`, ignoring the error ceiling.
    #[must_use]
    pub fn to_triangles(triangles: usize) -> Self {
        Self {
            target: DecimateTarget::TriangleCount(triangles),
            max_error: f32::INFINITY,
        }
    }

    /// Keep `ratio` of the input triangles, ignoring the error ceiling.
    #[must_use]
    pub fn to_ratio(ratio: f32) -> Self {
        Self {
            target: DecimateTarget::Ratio(ratio),
            max_error: f32::INFINITY,
        }
    }
}

/// A decimated triangle mesh: a fresh, compacted vertex array and the triangle
/// indices referencing it.
#[derive(Clone, Debug, PartialEq)]
pub struct DecimatedMesh {
    /// Surviving vertex positions, re-indexed from zero.
    pub vertices: Vec<Vec3>,
    /// Triangle indices (triples) into [`DecimatedMesh::vertices`].
    pub indices: Vec<u32>,
}

impl DecimatedMesh {
    /// Number of triangles in the decimated mesh.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of vertices in the decimated mesh.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

/// Reduces a triangle soup to a lower-triangle-count collision proxy using
/// quadric-error edge collapses.
///
/// Returns `None` when the input is empty, the index buffer is not a whole
/// number of triangles, or an index is out of range. A mesh already at or below
/// the target is returned compacted but otherwise unchanged.
#[must_use]
pub fn decimate_mesh(
    vertices: &[Vec3],
    indices: &[u32],
    params: DecimateParams,
) -> Option<DecimatedMesh> {
    if vertices.is_empty() || indices.is_empty() || !indices.len().is_multiple_of(3) {
        return None;
    }
    let vcount = vertices.len() as u32;
    if indices.iter().any(|&i| i >= vcount) {
        return None;
    }

    let mut dec = Decimator::build(vertices, indices);
    let input_tris = dec.alive_faces;
    if input_tris == 0 {
        return None;
    }

    let target = resolve_target(params.target, input_tris);
    dec.run(target, params.max_error);
    Some(dec.compact())
}

/// Resolves the requested target into an absolute triangle count, clamped to a
/// closed mesh's minimum.
fn resolve_target(target: DecimateTarget, input_tris: usize) -> usize {
    let raw = match target {
        DecimateTarget::TriangleCount(n) => n,
        DecimateTarget::Ratio(r) => {
            let r = r.clamp(0.0, 1.0);
            (input_tris as f32 * r).round() as usize
        }
    };
    raw.clamp(MIN_TRIANGLES.min(input_tris), input_tris)
}

/// Canonical (ordered) undirected edge key.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// A pending collapse of edge `(u, v)` scored by `cost`, valid only while the
/// stored per-vertex versions still match.
#[derive(Clone, Copy, Debug)]
struct Collapse {
    cost: f32,
    u: u32,
    v: u32,
    ver_u: u32,
    ver_v: u32,
}

impl PartialEq for Collapse {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits() && self.u == other.u && self.v == other.v
    }
}
impl Eq for Collapse {}

impl Ord for Collapse {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap on (cost, u, v): a smaller cost (then smaller indices) must
        // compare as `Greater` so `BinaryHeap::pop` yields it first.
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.u.cmp(&self.u))
            .then_with(|| other.v.cmp(&self.v))
    }
}
impl PartialOrd for Collapse {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Mutable working state for one decimation run.
struct Decimator {
    pos: Vec<Vec3>,
    quad: Vec<Quadric>,
    vert_alive: Vec<bool>,
    version: Vec<u32>,
    vert_faces: Vec<Vec<u32>>,
    faces: Vec<[u32; 3]>,
    face_alive: Vec<bool>,
    alive_faces: usize,
    heap: BinaryHeap<Collapse>,
}

impl Decimator {
    fn build(vertices: &[Vec3], indices: &[u32]) -> Self {
        let n = vertices.len();
        let mut faces: Vec<[u32; 3]> = Vec::with_capacity(indices.len() / 3);
        let mut face_alive: Vec<bool> = Vec::with_capacity(indices.len() / 3);
        for tri in indices.chunks_exact(3) {
            let f = [tri[0], tri[1], tri[2]];
            let degenerate = f[0] == f[1]
                || f[1] == f[2]
                || f[0] == f[2]
                || triangle_normal(
                    vertices[f[0] as usize],
                    vertices[f[1] as usize],
                    vertices[f[2] as usize],
                )
                .is_none();
            faces.push(f);
            face_alive.push(!degenerate);
        }

        let mut vert_faces: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut quad = vec![Quadric::ZERO; n];
        for (fi, f) in faces.iter().enumerate() {
            if !face_alive[fi] {
                continue;
            }
            let (a, b, c) = (f[0] as usize, f[1] as usize, f[2] as usize);
            let fq = Quadric::from_triangle(vertices[a], vertices[b], vertices[c]);
            quad[a] += fq;
            quad[b] += fq;
            quad[c] += fq;
            vert_faces[a].push(fi as u32);
            vert_faces[b].push(fi as u32);
            vert_faces[c].push(fi as u32);
        }

        add_boundary_constraints(vertices, &faces, &face_alive, &mut quad);

        let alive_faces = face_alive.iter().filter(|&&a| a).count();
        let mut dec = Self {
            pos: vertices.to_vec(),
            quad,
            vert_alive: vec![true; n],
            version: vec![0; n],
            vert_faces,
            faces,
            face_alive,
            alive_faces,
            heap: BinaryHeap::new(),
        };
        dec.seed_heap();
        dec
    }

    /// Scores every unique surviving edge once into the heap.
    fn seed_heap(&mut self) {
        let mut edges: HashSet<(u32, u32)> = HashSet::new();
        for (fi, f) in self.faces.iter().enumerate() {
            if !self.face_alive[fi] {
                continue;
            }
            edges.insert(edge_key(f[0], f[1]));
            edges.insert(edge_key(f[1], f[2]));
            edges.insert(edge_key(f[2], f[0]));
        }
        for (u, v) in edges {
            self.push_edge(u, v);
        }
    }

    /// Scores edge `(u, v)` and pushes it onto the heap with current versions.
    fn push_edge(&mut self, u: u32, v: u32) {
        let (cost, _) = self.evaluate(u, v);
        self.heap.push(Collapse {
            cost,
            u,
            v,
            ver_u: self.version[u as usize],
            ver_v: self.version[v as usize],
        });
    }

    /// Combined quadric of `(u, v)`, the position its collapse should use, and
    /// the resulting error.
    fn evaluate(&self, u: u32, v: u32) -> (f32, Vec3) {
        let q = self.quad[u as usize].add(&self.quad[v as usize]);
        let target = q.optimal_point().unwrap_or_else(|| {
            let a = self.pos[u as usize];
            let b = self.pos[v as usize];
            let mid = (a + b) * 0.5;
            [a, b, mid]
                .into_iter()
                .min_by(|&p0, &p1| q.error(p0).total_cmp(&q.error(p1)))
                .unwrap_or(mid)
        });
        (q.error(target), target)
    }

    /// Neighbour vertices of `u` reachable through one alive incident face.
    fn neighbors(&self, u: u32) -> HashSet<u32> {
        let mut out = HashSet::new();
        for &f in &self.vert_faces[u as usize] {
            if !self.face_alive[f as usize] {
                continue;
            }
            for &w in &self.faces[f as usize] {
                if w != u {
                    out.insert(w);
                }
            }
        }
        out
    }

    /// Alive faces that contain both `u` and `v` (the triangles on the edge).
    fn shared_faces(&self, u: u32, v: u32) -> Vec<u32> {
        self.vert_faces[u as usize]
            .iter()
            .copied()
            .filter(|&f| self.face_alive[f as usize] && self.faces[f as usize].contains(&v))
            .collect()
    }

    /// Edge-collapse link condition plus a normal-flip guard: the collapse keeps
    /// the mesh 2-manifold and introduces no folded triangle.
    fn is_legal(&self, u: u32, v: u32, target: Vec3) -> bool {
        let shared = self.shared_faces(u, v);
        // A real manifold edge is shared by one (boundary) or two faces.
        if shared.is_empty() || shared.len() > 2 {
            return false;
        }
        // Link condition: the common neighbours of u and v must be exactly the
        // vertices opposite the shared faces; any extra common neighbour would
        // pinch the surface into a non-manifold configuration.
        let opposite: HashSet<u32> = shared
            .iter()
            .flat_map(|&f| self.faces[f as usize])
            .filter(|&w| w != u && w != v)
            .collect();
        let nu = self.neighbors(u);
        let nv = self.neighbors(v);
        let common: HashSet<u32> = nu.intersection(&nv).copied().collect();
        if common != opposite {
            return false;
        }
        // Normal-flip guard over every face that survives the collapse.
        for &f in self.vert_faces[u as usize]
            .iter()
            .chain(&self.vert_faces[v as usize])
        {
            let fi = f as usize;
            if !self.face_alive[fi] {
                continue;
            }
            let tri = self.faces[fi];
            if tri.contains(&u) && tri.contains(&v) {
                continue; // removed by the collapse
            }
            let moved = |x: u32| {
                if x == u || x == v {
                    target
                } else {
                    self.pos[x as usize]
                }
            };
            let old = triangle_normal(
                self.pos[tri[0] as usize],
                self.pos[tri[1] as usize],
                self.pos[tri[2] as usize],
            );
            let new = triangle_normal(moved(tri[0]), moved(tri[1]), moved(tri[2]));
            match (old, new) {
                (Some(o), Some(ndir)) if o.dot(ndir) > 0.0 => {}
                _ => return false,
            }
        }
        true
    }

    /// Applies collapses cheapest-first until the triangle target or error
    /// ceiling stops progress.
    fn run(&mut self, target_tris: usize, max_error: f32) {
        while self.alive_faces > target_tris {
            let Some(entry) = self.heap.pop() else {
                break;
            };
            let (u, v) = (entry.u, entry.v);
            if !self.vert_alive[u as usize] || !self.vert_alive[v as usize] {
                continue;
            }
            if self.version[u as usize] != entry.ver_u || self.version[v as usize] != entry.ver_v {
                continue; // stale: re-scored since this entry was pushed
            }
            if entry.cost > max_error {
                break; // cheapest remaining collapse is already too costly
            }
            let (_, target) = self.evaluate(u, v);
            if !self.is_legal(u, v, target) {
                continue;
            }
            self.collapse(u, v, target);
        }
    }

    /// Merges `v` into `u`, moving `u` to `target` and re-scoring `u`'s edges.
    fn collapse(&mut self, u: u32, v: u32, target: Vec3) {
        let (uu, vv) = (u as usize, v as usize);
        self.pos[uu] = target;
        self.quad[uu] = self.quad[uu].add(&self.quad[vv]);

        // Faces touched by the collapse: u's and v's incident faces.
        let mut touched: Vec<u32> = self.vert_faces[uu].clone();
        for &f in &self.vert_faces[vv] {
            if !touched.contains(&f) {
                touched.push(f);
            }
        }

        for &f in &touched {
            let fi = f as usize;
            if !self.face_alive[fi] {
                continue;
            }
            let has_u = self.faces[fi].contains(&u);
            let has_v = self.faces[fi].contains(&v);
            if has_u && has_v {
                // Triangle on the collapsed edge: remove it and detach its
                // opposite vertex.
                self.face_alive[fi] = false;
                self.alive_faces -= 1;
                let opp: Vec<u32> = self.faces[fi]
                    .iter()
                    .copied()
                    .filter(|&w| w != u && w != v)
                    .collect();
                for w in opp {
                    self.vert_faces[w as usize].retain(|&x| x != f);
                }
            } else if has_v {
                for s in self.faces[fi].iter_mut() {
                    if *s == v {
                        *s = u;
                    }
                }
            }
        }

        // Rebuild u's adjacency from the touched faces that still contain it.
        let mut uf: Vec<u32> = touched
            .iter()
            .copied()
            .filter(|&f| self.face_alive[f as usize] && self.faces[f as usize].contains(&u))
            .collect();
        uf.sort_unstable();
        uf.dedup();
        self.vert_faces[uu] = uf;
        self.vert_faces[vv] = Vec::new();
        self.vert_alive[vv] = false;
        self.version[uu] += 1;

        let neighbors = self.neighbors(u);
        for w in neighbors {
            self.push_edge(u, w);
        }
    }

    /// Compacts surviving vertices and faces into a fresh [`DecimatedMesh`],
    /// dropping dead faces, degenerate triangles and duplicate faces.
    fn compact(&self) -> DecimatedMesh {
        let mut remap = vec![u32::MAX; self.pos.len()];
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut seen: HashSet<[u32; 3]> = HashSet::new();

        for (fi, f) in self.faces.iter().enumerate() {
            if !self.face_alive[fi] {
                continue;
            }
            let [a, b, c] = *f;
            if a == b || b == c || a == c {
                continue;
            }
            // Deduplicate identical faces regardless of rotation.
            let mut sorted = [a, b, c];
            sorted.sort_unstable();
            if !seen.insert(sorted) {
                continue;
            }
            let emit = |orig: u32, vertices: &mut Vec<Vec3>, remap: &mut Vec<u32>| -> u32 {
                let slot = &mut remap[orig as usize];
                if *slot == u32::MAX {
                    *slot = vertices.len() as u32;
                    vertices.push(self.pos[orig as usize]);
                }
                *slot
            };
            let na = emit(a, &mut vertices, &mut remap);
            let nb = emit(b, &mut vertices, &mut remap);
            let nc = emit(c, &mut vertices, &mut remap);
            indices.extend_from_slice(&[na, nb, nc]);
        }

        DecimatedMesh { vertices, indices }
    }
}

/// Adds a heavily weighted perpendicular constraint quadric to each endpoint of
/// every open boundary edge, freezing the silhouette.
fn add_boundary_constraints(
    vertices: &[Vec3],
    faces: &[[u32; 3]],
    face_alive: &[bool],
    quad: &mut [Quadric],
) {
    let mut edge_faces: HashMap<(u32, u32), u32> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        if !face_alive[fi] {
            continue;
        }
        for &(a, b) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            *edge_faces.entry(edge_key(a, b)).or_insert(0) += 1;
        }
    }

    for (fi, f) in faces.iter().enumerate() {
        if !face_alive[fi] {
            continue;
        }
        let Some(fn_dir) = triangle_normal(
            vertices[f[0] as usize],
            vertices[f[1] as usize],
            vertices[f[2] as usize],
        ) else {
            continue;
        };
        for &(a, b) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            if edge_faces.get(&edge_key(a, b)).copied() != Some(1) {
                continue; // interior edge
            }
            let pa = vertices[a as usize];
            let pb = vertices[b as usize];
            let edge = pb - pa;
            // Plane through the edge, perpendicular to the triangle.
            let normal = edge.cross(fn_dir);
            let len = normal.length();
            if len <= f32::MIN_POSITIVE {
                continue;
            }
            let n = normal / len;
            let constraint = Quadric::from_plane(n, -n.dot(pa)).scaled(BOUNDARY_WEIGHT);
            quad[a as usize] += constraint;
            quad[b as usize] += constraint;
        }
    }
}

/// Unit normal of triangle `abc`, or `None` when the triangle is degenerate.
fn triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Option<Vec3> {
    let n = (b - a).cross(c - a);
    let len = n.length();
    if len <= 1e-20 {
        None
    } else {
        Some(n / len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a closed, welded icosahedron subdivided `levels` times (midpoint
    /// subdivision, shared midpoints), i.e. a watertight 2-manifold sphere.
    fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<u32>) {
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
            let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
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
        let indices: Vec<u32> = faces.into_iter().flatten().collect();
        (verts, indices)
    }

    /// Builds a flat `n x n`-quad grid in the z = 0 plane (an open mesh with a
    /// square boundary).
    fn grid(n: usize) -> (Vec<Vec3>, Vec<u32>) {
        let mut verts = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                verts.push(Vec3::new(i as f32, j as f32, 0.0));
            }
        }
        let idx = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        let mut indices = Vec::new();
        for j in 0..n {
            for i in 0..n {
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        (verts, indices)
    }

    /// Asserts the mesh is a closed 2-manifold: every undirected edge is shared
    /// by exactly two triangles.
    fn assert_closed(indices: &[u32]) {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in indices.chunks_exact(3) {
            for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                *counts.entry(edge_key(a, b)).or_insert(0) += 1;
            }
        }
        for (_, c) in counts {
            assert_eq!(c, 2, "non-manifold edge with {c} incident faces");
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(decimate_mesh(&[], &[0, 1, 2], DecimateParams::to_ratio(0.5)).is_none());
        let v = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(decimate_mesh(&v, &[], DecimateParams::to_ratio(0.5)).is_none());
        assert!(decimate_mesh(&v, &[0, 1], DecimateParams::to_ratio(0.5)).is_none());
        assert!(decimate_mesh(&v, &[0, 1, 9], DecimateParams::to_ratio(0.5)).is_none());
    }

    #[test]
    fn halves_a_sphere_and_stays_closed() {
        let (v, i) = icosphere(2); // 320 triangles
        let input_tris = i.len() / 3;
        let out = decimate_mesh(&v, &i, DecimateParams::to_ratio(0.5)).expect("decimated");
        assert!(
            out.triangle_count() <= input_tris / 2 + input_tris / 10,
            "expected ~{} tris, got {}",
            input_tris / 2,
            out.triangle_count()
        );
        assert!(out.triangle_count() >= MIN_TRIANGLES);
        assert_closed(&out.indices);
    }

    #[test]
    fn aggressive_target_still_closed() {
        let (v, i) = icosphere(2);
        let out = decimate_mesh(&v, &i, DecimateParams::to_triangles(16)).expect("decimated");
        assert!(out.triangle_count() >= MIN_TRIANGLES);
        assert!(out.triangle_count() <= 40, "got {}", out.triangle_count());
        assert_closed(&out.indices);
    }

    #[test]
    fn decimated_sphere_stays_near_unit_radius() {
        let (v, i) = icosphere(3); // 1280 triangles
        let out = decimate_mesh(&v, &i, DecimateParams::to_ratio(0.25)).expect("decimated");
        // QEM keeps vertices close to the original surface (the unit sphere).
        for p in &out.vertices {
            assert!(
                (p.length() - 1.0).abs() < 0.15,
                "vertex drifted: r = {}",
                p.length()
            );
        }
    }

    #[test]
    fn target_above_input_returns_compacted_mesh() {
        let (v, i) = icosphere(1); // 80 triangles
        let out = decimate_mesh(&v, &i, DecimateParams::to_triangles(10_000)).expect("decimated");
        assert_eq!(out.triangle_count(), i.len() / 3);
        assert_closed(&out.indices);
    }

    #[test]
    fn grid_preserves_its_bounding_corners() {
        let (v, i) = grid(8); // 128 triangles, square boundary [0,8]^2
        let out = decimate_mesh(&v, &i, DecimateParams::to_ratio(0.3)).expect("decimated");
        assert!(out.triangle_count() < i.len() / 3);
        // Boundary weighting must keep every planar vertex on the z = 0 plane.
        for p in &out.vertices {
            assert!(p.z.abs() < 1e-5, "vertex left the plane: {p:?}");
        }
        // The four corners of the square must survive (they anchor the shape).
        for corner in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 8.0, 0.0),
            Vec3::new(8.0, 8.0, 0.0),
        ] {
            assert!(
                out.vertices.iter().any(|p| (*p - corner).length() < 1e-4),
                "missing corner {corner:?}"
            );
        }
    }

    #[test]
    fn decimation_is_deterministic() {
        let (v, i) = icosphere(2);
        let a = decimate_mesh(&v, &i, DecimateParams::to_ratio(0.4)).unwrap();
        let b = decimate_mesh(&v, &i, DecimateParams::to_ratio(0.4)).unwrap();
        assert_eq!(a, b);
    }
}
