//! Indexed triangle mesh primitive: a shared vertex pool plus an index buffer,
//! the `CPU` golden reference for the hardware ray-tracing built-in triangle
//! path.
//!
//! This differs from the two other triangle representations in `ray_scene`:
//! [`super::traversal`] intersects a loose *triangle soup* (three positions per
//! primitive, geometric normal only), and [`super::shaded_triangle`] stores
//! three positions **and** three shading normals *per primitive*. A real asset
//! instead shares one vertex buffer across many faces and references vertices
//! by index — exactly how a `GPU` bottom-level acceleration structure (`BLAS`)
//! consumes a triangle mesh. [`TriangleMesh`] models that layout: one
//! `positions` pool, optional matching `normals`/`uvs` pools, and an `indices`
//! buffer of vertex triples. Each face is intersected with Möller–Trumbore and
//! reports the barycentrically interpolated position, shading normal (or the
//! geometric normal when the mesh carries none), and texture coordinate.
//!
//! The vertex pools are stored **verbatim** (no re-normalization of normals in
//! construction) so a flat `GPU` layout decodes to a bit-identical mesh and the
//! packed traversal reproduces every hit's bits exactly; interpolated normals
//! are re-normalized only at the hit.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// Why [`TriangleMesh::new`] rejected its inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriangleMeshError {
    /// A triangle referenced a vertex index at or beyond the position pool.
    IndexOutOfRange {
        /// Zero-based triangle whose corner is out of range.
        triangle: usize,
        /// Which corner (`0`, `1`, or `2`) held the bad index.
        corner: usize,
        /// The offending vertex index.
        index: u32,
        /// Number of vertices actually present in the position pool.
        vertex_count: usize,
    },
    /// A non-empty `normals` pool did not match the position pool length.
    NormalCountMismatch {
        /// Length of the supplied normal pool.
        normals: usize,
        /// Length of the position pool it must match.
        positions: usize,
    },
    /// A non-empty `uvs` pool did not match the position pool length.
    UvCountMismatch {
        /// Length of the supplied `UV` pool.
        uvs: usize,
        /// Length of the position pool it must match.
        positions: usize,
    },
}

impl core::fmt::Display for TriangleMeshError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IndexOutOfRange {
                triangle,
                corner,
                index,
                vertex_count,
            } => write!(
                f,
                "triangle {triangle} corner {corner} references vertex {index} but the mesh has {vertex_count} vertices"
            ),
            Self::NormalCountMismatch { normals, positions } => write!(
                f,
                "normal pool has {normals} entries but the mesh has {positions} positions"
            ),
            Self::UvCountMismatch { uvs, positions } => write!(
                f,
                "UV pool has {uvs} entries but the mesh has {positions} positions"
            ),
        }
    }
}

impl std::error::Error for TriangleMeshError {}

/// An indexed triangle mesh: a shared vertex pool referenced by an index
/// buffer of vertex triples.
#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMesh {
    /// World-space vertex positions; `indices` reference entries here.
    positions: Vec<[f32; 3]>,
    /// Per-vertex shading normals, stored verbatim. Either empty (geometric
    /// shading) or exactly as long as `positions`.
    normals: Vec<[f32; 3]>,
    /// Per-vertex texture coordinates. Either empty (barycentric fallback) or
    /// exactly as long as `positions`.
    uvs: Vec<[f32; 2]>,
    /// Vertex-index triples, one per triangle (counter-clockwise front face).
    indices: Vec<[u32; 3]>,
}

impl TriangleMesh {
    /// Builds a mesh from a `positions` pool, optional `normals`/`uvs` pools
    /// (empty to omit), and an `indices` buffer of vertex triples.
    ///
    /// # Errors
    ///
    /// Returns [`TriangleMeshError`] when any index is out of range, or when a
    /// non-empty normal or `UV` pool length does not equal the position count.
    /// Pools are stored verbatim; normals need not be unit length because the
    /// interpolated normal is re-normalized at each hit.
    pub fn new(
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<[u32; 3]>,
    ) -> Result<Self, TriangleMeshError> {
        let vertex_count = positions.len();
        if !normals.is_empty() && normals.len() != vertex_count {
            return Err(TriangleMeshError::NormalCountMismatch {
                normals: normals.len(),
                positions: vertex_count,
            });
        }
        if !uvs.is_empty() && uvs.len() != vertex_count {
            return Err(TriangleMeshError::UvCountMismatch {
                uvs: uvs.len(),
                positions: vertex_count,
            });
        }
        for (triangle, tri) in indices.iter().enumerate() {
            for (corner, &index) in tri.iter().enumerate() {
                if index as usize >= vertex_count {
                    return Err(TriangleMeshError::IndexOutOfRange {
                        triangle,
                        corner,
                        index,
                        vertex_count,
                    });
                }
            }
        }
        Ok(Self {
            positions,
            normals,
            uvs,
            indices,
        })
    }

    /// The shared vertex positions.
    #[must_use]
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// The per-vertex shading normals (empty when the mesh shades geometric).
    #[must_use]
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }

    /// The per-vertex texture coordinates (empty when none were supplied).
    #[must_use]
    pub fn uvs(&self) -> &[[f32; 2]] {
        &self.uvs
    }

    /// The vertex-index triples, one per triangle.
    #[must_use]
    pub fn indices(&self) -> &[[u32; 3]] {
        &self.indices
    }

    /// True when the mesh carries per-vertex shading normals.
    #[must_use]
    pub fn has_normals(&self) -> bool {
        !self.normals.is_empty()
    }

    /// True when the mesh carries per-vertex texture coordinates.
    #[must_use]
    pub fn has_uvs(&self) -> bool {
        !self.uvs.is_empty()
    }

    /// Number of vertices in the shared pool.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles (index triples).
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// The three world-space positions of triangle `tri`.
    #[must_use]
    pub fn triangle_positions(&self, tri: usize) -> [[f32; 3]; 3] {
        let [i0, i1, i2] = self.indices[tri];
        [
            self.positions[i0 as usize],
            self.positions[i1 as usize],
            self.positions[i2 as usize],
        ]
    }

    /// Axis-aligned bounds of triangle `tri`.
    #[must_use]
    pub fn triangle_aabb(&self, tri: usize) -> Aabb {
        let [p0, p1, p2] = self.triangle_positions(tri);
        let mut lo = p0;
        let mut hi = p0;
        for v in [p1, p2] {
            for k in 0..3 {
                if v[k] < lo[k] {
                    lo[k] = v[k];
                }
                if v[k] > hi[k] {
                    hi[k] = v[k];
                }
            }
        }
        Aabb::new(lo, hi)
    }

    /// Axis-aligned bounds of the whole mesh (union of every position), or the
    /// empty box for a mesh with no vertices.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut iter = self.positions.iter();
        let Some(&first) = iter.next() else {
            return Aabb::empty();
        };
        let mut lo = first;
        let mut hi = first;
        for v in iter {
            for k in 0..3 {
                if v[k] < lo[k] {
                    lo[k] = v[k];
                }
                if v[k] > hi[k] {
                    hi[k] = v[k];
                }
            }
        }
        Aabb::new(lo, hi)
    }

    /// Nearest intersection of `ray` with triangle `tri`, or `None` on a miss.
    ///
    /// Uses the two-sided Möller–Trumbore test for `(t, u, v)`; the hit carries
    /// the barycentric position `w0·p0 + u·p1 + v·p2` (`w0 = 1 - u - v`), the
    /// shading normal (interpolated vertex normals when present, otherwise the
    /// geometric normal), and the texture coordinate (interpolated `UV`s when
    /// present, otherwise the barycentric `(u, v)` fallback). Both normals are
    /// oriented into the incoming ray's hemisphere, and the shading normal is
    /// forced into the geometric normal's hemisphere (pbrt's convention).
    #[must_use]
    pub fn intersect_triangle(&self, tri: usize, ray: &Ray) -> Option<MeshHit> {
        const EPS: f32 = 1e-8;
        let [i0, i1, i2] = self.indices[tri];
        let (i0, i1, i2) = (i0 as usize, i1 as usize, i2 as usize);
        let p0 = self.positions[i0];
        let p1 = self.positions[i1];
        let p2 = self.positions[i2];
        let e1 = sub(p1, p0);
        let e2 = sub(p2, p0);
        let dir = ray.direction();
        let p = cross(dir, e2);
        let det = dot(e1, p);
        if det.abs() < EPS {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = sub(ray.origin(), p0);
        let u = dot(tvec, p) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = cross(tvec, e1);
        let v = dot(dir, q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = dot(e2, q) * inv_det;
        if t < ray.t_min() || t > ray.t_max() {
            return None;
        }

        let w0 = 1.0 - u - v;
        let position = [
            w0 * p0[0] + u * p1[0] + v * p2[0],
            w0 * p0[1] + u * p1[1] + v * p2[1],
            w0 * p0[2] + u * p1[2] + v * p2[2],
        ];

        // Geometric normal from the face, oriented against the ray.
        let ng_raw = cross(e1, e2);
        let front_face = dot(dir, ng_raw) < 0.0;
        let geo = if front_face { ng_raw } else { negate(ng_raw) };
        let geometric_normal = normalize_or(geo, geo);

        let normal = if self.has_normals() {
            let n0 = self.normals[i0];
            let n1 = self.normals[i1];
            let n2 = self.normals[i2];
            let ns_raw = [
                w0 * n0[0] + u * n1[0] + v * n2[0],
                w0 * n0[1] + u * n1[1] + v * n2[1],
                w0 * n0[2] + u * n1[2] + v * n2[2],
            ];
            let ns_oriented = if dot(ns_raw, geometric_normal) < 0.0 {
                negate(ns_raw)
            } else {
                ns_raw
            };
            normalize_or(ns_oriented, geometric_normal)
        } else {
            geometric_normal
        };

        let uv = if self.has_uvs() {
            let t0 = self.uvs[i0];
            let t1 = self.uvs[i1];
            let t2 = self.uvs[i2];
            [
                w0 * t0[0] + u * t1[0] + v * t2[0],
                w0 * t0[1] + u * t1[1] + v * t2[1],
            ]
        } else {
            [u, v]
        };

        Some(MeshHit {
            t,
            u,
            v,
            triangle: tri as u32,
            position,
            normal,
            uv,
            front_face,
        })
    }

    /// Nearest intersection of `ray` with any triangle via a linear scan.
    ///
    /// This is the brute-force reference for [`TriangleMeshBvh::closest_hit`];
    /// production traversal should go through the `BVH`.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<MeshHit> {
        let mut ray = *ray;
        let mut best: Option<MeshHit> = None;
        for tri in 0..self.triangle_count() {
            if let Some(hit) = self.intersect_triangle(tri, &ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }
}

/// A ray/[`TriangleMesh`] intersection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Barycentric `u` (weight of the triangle's second vertex).
    pub u: f32,
    /// Barycentric `v` (weight of the third vertex); the first vertex's weight
    /// is `1 - u - v`.
    pub v: f32,
    /// Original (pre-`BVH`-reorder) triangle index that was hit.
    pub triangle: u32,
    /// World-space hit position, reconstructed from the barycentric weights.
    pub position: [f32; 3],
    /// Unit shading normal in the geometric hemisphere (interpolated vertex
    /// normals when present, otherwise the geometric normal).
    pub normal: [f32; 3],
    /// Interpolated texture coordinate, or the barycentric `(u, v)` fallback
    /// when the mesh carries no `UV`s.
    pub uv: [f32; 2],
    /// `true` when the ray struck the front (counter-clockwise) face.
    pub front_face: bool,
}

/// A single-level `BVH` over the triangles of one [`TriangleMesh`].
#[derive(Clone, Debug)]
pub struct TriangleMeshBvh {
    /// The owned mesh, kept intact (so a `GPU` layout decodes bit-identically).
    mesh: TriangleMesh,
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Maps each `BVH` primitive slot to its original triangle index; leaf
    /// slices index into this, so hits report the original triangle id.
    order: Vec<u32>,
}

impl TriangleMeshBvh {
    /// Builds a `BVH` over `mesh`'s triangles with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(mesh: TriangleMesh) -> Self {
        Self::build_with(mesh, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `mesh`'s triangles with the given binned-`SAH`
    /// `config`.
    ///
    /// Each triangle's [`TriangleMesh::triangle_aabb`] feeds the builder; the
    /// returned primitive order is kept as [`TriangleMeshBvh::order`] so leaf
    /// slices map back to original triangle indices without disturbing the
    /// shared vertex/index buffers.
    #[must_use]
    pub fn build_with(mesh: TriangleMesh, config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = (0..mesh.triangle_count())
            .map(|tri| mesh.triangle_aabb(tri))
            .collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        Self { mesh, nodes, order }
    }

    /// The owned mesh.
    #[must_use]
    pub fn mesh(&self) -> &TriangleMesh {
        &self.mesh
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of triangles in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.order.len()
    }

    /// True when the hierarchy holds no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or the empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |node| node.bounds)
    }

    /// The flattened node array.
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// The `BVH`-slot → original-triangle-index map (leaf slices index this).
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<MeshHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<MeshHit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for &slot in &self.order[start..end] {
                        if let Some(hit) = self.mesh.intersect_triangle(slot as usize, &ray) {
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                            best = Some(hit);
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = node.second_child;
                    let neg = ray.direction()[node.axis as usize] < 0.0;
                    let (near, far) = if neg {
                        (second_child, first_child)
                    } else {
                        (first_child, second_child)
                    };
                    if sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node_index = near;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// True when *any* triangle intersects `ray` inside its interval.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray
                .aabb_interval(&node.bounds, ray.t_min(), ray.t_max())
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for &slot in &self.order[start..end] {
                        if self.mesh.intersect_triangle(slot as usize, ray).is_some() {
                            return true;
                        }
                    }
                    match stack_pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    if sp < stack.len() {
                        stack[sp] = node.second_child;
                        sp += 1;
                    }
                    node_index = first_child;
                }
            } else {
                match stack_pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

/// Component-wise negation of a 3-vector.
fn negate(a: [f32; 3]) -> [f32; 3] {
    [-a[0], -a[1], -a[2]]
}

/// `a - b` component-wise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalizes `v`, or returns `fallback` when `v` is too short to normalize
/// reliably (keeps the hit's normal finite for degenerate inputs).
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = dot(v, v);
    if len_sq < 1e-20 {
        return fallback;
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Pops the top node index off the traversal stack, or `None` when empty.
fn stack_pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        None
    } else {
        *sp -= 1;
        Some(stack[*sp])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites so tests never depend on an external crate.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
        fn point(&mut self, lo: f32, hi: f32) -> [f32; 3] {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Independent (intersect-free) vector helpers for the oracles.
    fn o_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn o_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn o_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn o_norm(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / o_dot(v, v).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    /// A random non-degenerate single-triangle mesh plus independent copies of
    /// its three positions / normals / uvs, for direct `intersect_triangle`
    /// oracles.
    struct TriCase {
        /// Single-triangle mesh under test.
        mesh: TriangleMesh,
        /// Independent copy of the three vertex positions.
        pos: [[f32; 3]; 3],
        /// Independent copy of the three vertex normals.
        nrm: [[f32; 3]; 3],
        /// Independent copy of the three vertex uvs.
        uv: [[f32; 2]; 3],
    }

    fn random_triangle(rng: &mut Rng) -> Option<TriCase> {
        let p0 = rng.point(-3.0, 3.0);
        let p1 = rng.point(-3.0, 3.0);
        let p2 = rng.point(-3.0, 3.0);
        let area2 = o_cross(o_sub(p1, p0), o_sub(p2, p0));
        if o_dot(area2, area2).sqrt() < 0.5 {
            return None; // reject near-degenerate triangles
        }
        let n0 = o_norm(rng.point(-1.0, 1.0));
        let n1 = o_norm(rng.point(-1.0, 1.0));
        let n2 = o_norm(rng.point(-1.0, 1.0));
        let t0 = [rng.range(0.0, 1.0), rng.range(0.0, 1.0)];
        let t1 = [rng.range(0.0, 1.0), rng.range(0.0, 1.0)];
        let t2 = [rng.range(0.0, 1.0), rng.range(0.0, 1.0)];
        let mesh = TriangleMesh::new(
            vec![p0, p1, p2],
            vec![n0, n1, n2],
            vec![t0, t1, t2],
            vec![[0, 1, 2]],
        )
        .expect("valid single-triangle mesh");
        Some(TriCase {
            mesh,
            pos: [p0, p1, p2],
            nrm: [n0, n1, n2],
            uv: [t0, t1, t2],
        })
    }

    /// Aims a ray at the point with barycentric weights `(w0, u, v)` on the
    /// triangle, from a random origin off the plane; returns the ray and the
    /// aimed-at target, or `None` when the geometry grazes.
    fn ray_at_bary(
        rng: &mut Rng,
        pos: [[f32; 3]; 3],
        w0: f32,
        u: f32,
        v: f32,
    ) -> Option<(Ray, [f32; 3])> {
        let [p0, p1, p2] = pos;
        let target = [
            w0 * p0[0] + u * p1[0] + v * p2[0],
            w0 * p0[1] + u * p1[1] + v * p2[1],
            w0 * p0[2] + u * p1[2] + v * p2[2],
        ];
        let origin = o_sub(target, rng.point(-4.0, 4.0));
        let d = o_sub(target, origin);
        if o_dot(d, d).sqrt() < 0.5 {
            return None;
        }
        let dir = o_norm(d);
        let ng = o_norm(o_cross(o_sub(p1, p0), o_sub(p2, p0)));
        if o_dot(dir, ng).abs() < 0.2 {
            return None; // grazing: reject for numerical stability
        }
        Some((Ray::new(origin, dir, 1e-3, 1e6), target))
    }

    #[test]
    fn new_rejects_out_of_range_index() {
        let err = TriangleMesh::new(
            vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![],
            vec![],
            vec![[0, 1, 3]],
        )
        .unwrap_err();
        assert_eq!(
            err,
            TriangleMeshError::IndexOutOfRange {
                triangle: 0,
                corner: 2,
                index: 3,
                vertex_count: 3,
            }
        );
    }

    #[test]
    fn new_rejects_pool_length_mismatch() {
        let positions = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normal_err = TriangleMesh::new(
            positions.clone(),
            vec![[0.0, 0.0, 1.0]],
            vec![],
            vec![[0, 1, 2]],
        )
        .unwrap_err();
        assert_eq!(
            normal_err,
            TriangleMeshError::NormalCountMismatch {
                normals: 1,
                positions: 3,
            }
        );
        let uv_err =
            TriangleMesh::new(positions, vec![], vec![[0.0, 0.0]], vec![[0, 1, 2]]).unwrap_err();
        assert_eq!(
            uv_err,
            TriangleMeshError::UvCountMismatch {
                uvs: 1,
                positions: 3,
            }
        );
    }

    #[test]
    fn accessors_and_aabb() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, -1.0]],
            vec![],
            vec![],
            vec![[0, 1, 2]],
        )
        .unwrap();
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.triangle_count(), 1);
        assert!(!mesh.has_normals());
        assert!(!mesh.has_uvs());
        assert_eq!(
            mesh.triangle_positions(0),
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, -1.0]]
        );
        let bb = mesh.aabb();
        assert_eq!(bb.min, [0.0, 0.0, -1.0]);
        assert_eq!(bb.max, [2.0, 3.0, 0.0]);
    }

    #[test]
    fn hit_point_lies_on_triangle_plane() {
        // Residual oracle: the reported position must satisfy the plane
        // equation and reconstruct the aimed-at barycentric target. The plane
        // normal is recomputed independently of `intersect_triangle`.
        let mut rng = Rng::new(0x1357_9BDF);
        let mut hits = 0usize;
        for _ in 0..60_000 {
            let Some(TriCase { mesh, pos, .. }) = random_triangle(&mut rng) else {
                continue;
            };
            let b1 = rng.range(0.05, 0.9);
            let b2 = rng.range(0.05, 0.9);
            if b1 + b2 > 0.95 {
                continue;
            }
            let w0 = 1.0 - b1 - b2;
            let Some((ray, target)) = ray_at_bary(&mut rng, pos, w0, b1, b2) else {
                continue;
            };
            let hit = mesh.intersect_triangle(0, &ray).expect("aimed ray must hit");
            let [p0, p1, p2] = pos;
            let ng = o_norm(o_cross(o_sub(p1, p0), o_sub(p2, p0)));
            // Plane residual: hit point distance to the triangle's plane.
            let residual = o_dot(o_sub(hit.position, p0), ng);
            assert!(approx(residual, 0.0, 2e-3), "plane residual {residual}");
            // Barycentric reconstruction must land on the aimed target.
            for (k, (got, want)) in hit.position.iter().zip(target.iter()).enumerate() {
                assert!(
                    approx(*got, *want, 3e-3),
                    "position[{k}] {got} vs target {want}"
                );
            }
            // Recovered weights must match what we aimed with.
            assert!(approx(hit.u, b1, 3e-3) && approx(hit.v, b2, 3e-3));
            assert!(hit.t > 0.0);
            hits += 1;
        }
        assert!(hits > 2000, "too few validated hits: {hits}");
    }

    #[test]
    fn interpolated_normal_matches_barycentric_oracle() {
        let mut rng = Rng::new(0x2468_ACE0);
        let mut hits = 0usize;
        for _ in 0..60_000 {
            let Some(TriCase {
                mesh, pos, nrm, ..
            }) = random_triangle(&mut rng)
            else {
                continue;
            };
            let b1 = rng.range(0.05, 0.9);
            let b2 = rng.range(0.05, 0.9);
            if b1 + b2 > 0.95 {
                continue;
            }
            let w0 = 1.0 - b1 - b2;
            let Some((ray, _)) = ray_at_bary(&mut rng, pos, w0, b1, b2) else {
                continue;
            };
            let hit = mesh.intersect_triangle(0, &ray).expect("aimed ray must hit");
            // Oracle: blend the vertex normals, orient into the geometric
            // hemisphere, re-normalize — independently of the primitive.
            let [p0, p1, p2] = pos;
            let [n0, n1, n2] = nrm;
            let ng_raw = o_cross(o_sub(p1, p0), o_sub(p2, p0));
            let geo = if o_dot(ray.direction(), ng_raw) < 0.0 {
                ng_raw
            } else {
                [-ng_raw[0], -ng_raw[1], -ng_raw[2]]
            };
            let geo = o_norm(geo);
            let ns = [
                w0 * n0[0] + b1 * n1[0] + b2 * n2[0],
                w0 * n0[1] + b1 * n1[1] + b2 * n2[1],
                w0 * n0[2] + b1 * n1[2] + b2 * n2[2],
            ];
            let ns = if o_dot(ns, geo) < 0.0 {
                [-ns[0], -ns[1], -ns[2]]
            } else {
                ns
            };
            let expected = o_norm(ns);
            for (k, (got, want)) in hit.normal.iter().zip(expected.iter()).enumerate() {
                assert!(approx(*got, *want, 2e-3), "normal[{k}] {got} vs {want}");
            }
            // Shading normal must sit in the geometric hemisphere.
            assert!(o_dot(hit.normal, geo) >= -1e-4);
            hits += 1;
        }
        assert!(hits > 2000, "too few validated hits: {hits}");
    }

    #[test]
    fn uv_interpolation_matches_oracle() {
        let mut rng = Rng::new(0x0F0F_1234);
        let mut hits = 0usize;
        for _ in 0..40_000 {
            let Some(TriCase { mesh, pos, uv, .. }) = random_triangle(&mut rng) else {
                continue;
            };
            let b1 = rng.range(0.05, 0.9);
            let b2 = rng.range(0.05, 0.9);
            if b1 + b2 > 0.95 {
                continue;
            }
            let w0 = 1.0 - b1 - b2;
            let Some((ray, _)) = ray_at_bary(&mut rng, pos, w0, b1, b2) else {
                continue;
            };
            let hit = mesh.intersect_triangle(0, &ray).expect("aimed ray must hit");
            let [t0, t1, t2] = uv;
            let expected = [
                w0 * t0[0] + b1 * t1[0] + b2 * t2[0],
                w0 * t0[1] + b1 * t1[1] + b2 * t2[1],
            ];
            assert!(approx(hit.uv[0], expected[0], 3e-3));
            assert!(approx(hit.uv[1], expected[1], 3e-3));
            hits += 1;
        }
        assert!(hits > 2000, "too few validated hits: {hits}");
    }

    #[test]
    fn uv_fallback_is_barycentric_when_absent() {
        let mesh = TriangleMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![],
            vec![],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let ray = Ray::new([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], 1e-3, 1e6);
        let hit = mesh.intersect_triangle(0, &ray).unwrap();
        assert!(approx(hit.uv[0], hit.u, 1e-6) && approx(hit.uv[1], hit.v, 1e-6));
    }

    /// Builds a random closed-ish mesh: a cloud of vertices with random
    /// triangles indexing into it, plus matching normals and uvs.
    fn random_mesh(rng: &mut Rng, verts: usize, tris: usize) -> TriangleMesh {
        let positions: Vec<[f32; 3]> = (0..verts).map(|_| rng.point(-5.0, 5.0)).collect();
        let normals: Vec<[f32; 3]> = (0..verts).map(|_| o_norm(rng.point(-1.0, 1.0))).collect();
        let uvs: Vec<[f32; 2]> = (0..verts)
            .map(|_| [rng.range(0.0, 1.0), rng.range(0.0, 1.0)])
            .collect();
        let mut indices = Vec::with_capacity(tris);
        while indices.len() < tris {
            let a = rng.next_u32() as usize % verts;
            let b = rng.next_u32() as usize % verts;
            let c = rng.next_u32() as usize % verts;
            if a == b || b == c || a == c {
                continue;
            }
            indices.push([a as u32, b as u32, c as u32]);
        }
        TriangleMesh::new(positions, normals, uvs, indices).expect("valid mesh")
    }

    #[test]
    fn bvh_matches_brute_force() {
        let mut rng = Rng::new(0x5EED_F00D);
        let mesh = random_mesh(&mut rng, 48, 120);
        let bvh = TriangleMeshBvh::build(mesh.clone());
        assert_eq!(bvh.primitive_count(), mesh.triangle_count());
        let mut tested = 0usize;
        for _ in 0..5000 {
            let origin = rng.point(-8.0, 8.0);
            let dir = o_norm(rng.point(-1.0, 1.0));
            let ray = Ray::new(origin, dir, 1e-3, 1e6);
            let brute = mesh.intersect(&ray);
            let fast = bvh.closest_hit(&ray);
            match (brute, fast) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    // The nearest distance must agree. The *which-triangle*
                    // choice is only well-defined when there is no near-tie
                    // (coincident / crossing faces hit at the same `t` are a
                    // legitimate tie whose winner depends on visitation order).
                    assert!(approx(a.t, b.t, 1e-3), "t {} vs {}", a.t, b.t);
                    if a.triangle == b.triangle {
                        assert_eq!(a.front_face, b.front_face);
                        for (pa, pb) in a.position.iter().zip(b.position.iter()) {
                            assert!(approx(*pa, *pb, 2e-3));
                        }
                    }
                    tested += 1;
                }
                (a, b) => panic!("brute/bvh disagree: {a:?} vs {b:?}"),
            }
            assert_eq!(mesh.intersect(&ray).is_some(), bvh.any_hit(&ray));
        }
        assert!(tested > 100, "too few shared hits: {tested}");
    }

    #[test]
    fn empty_bvh_never_hits() {
        let mesh = TriangleMesh::new(vec![], vec![], vec![], vec![]).unwrap();
        let bvh = TriangleMeshBvh::build(mesh);
        assert!(bvh.is_empty());
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.bounds(), Aabb::empty());
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 1e6);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }
}
