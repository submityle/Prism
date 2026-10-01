//! Analytic pbrt-style triangle carrying per-vertex **shading normals**, with a
//! single-level `BVH` that serves as the `CPU` golden reference for the
//! smooth-normal mesh path.
//!
//! The watertight [`super::traversal`] triangle reports only the *geometric*
//! normal (the flat face), which is correct for visibility but gives faceted
//! shading. Real meshes store a shading normal at every vertex and interpolate
//! it across the face so curved surfaces read smooth. This primitive is that
//! smooth-shading triangle: it keeps the three vertex positions **and** three
//! vertex normals, intersects with Möller–Trumbore, and returns both the
//! geometric normal and the barycentrically interpolated shading normal
//! (re-normalized and flipped into the geometric hemisphere, exactly as pbrt
//! does), so a `GPU` kernel can shade against the same vectors.
//!
//! Vertex normals are stored **verbatim** (not re-normalized in [`Obb::new`]-
//! style construction) so the flat `GPU` layout decodes to a bit-identical
//! primitive and the packed traversal reproduces every hit's bits exactly.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// A triangle with per-vertex shading normals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadedTriangle {
    /// World-space vertex positions `[p0, p1, p2]`.
    positions: [[f32; 3]; 3],
    /// Per-vertex shading normals `[n0, n1, n2]`, stored verbatim (not
    /// re-normalized) so the `GPU` decode is bit-identical; interpolation
    /// re-normalizes the blended result at the hit.
    normals: [[f32; 3]; 3],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl ShadedTriangle {
    /// Builds a shaded triangle from vertex `positions` and matching per-vertex
    /// shading `normals` (both indexed `0, 1, 2`), with stable id `primitive`.
    ///
    /// Normals are kept verbatim; they need not be unit length on input because
    /// the interpolated normal is re-normalized at each hit.
    #[must_use]
    pub fn new(positions: [[f32; 3]; 3], normals: [[f32; 3]; 3], primitive: u32) -> Self {
        Self {
            positions,
            normals,
            primitive,
        }
    }

    /// World-space vertex positions `[p0, p1, p2]`.
    #[must_use]
    pub fn positions(&self) -> [[f32; 3]; 3] {
        self.positions
    }

    /// Per-vertex shading normals `[n0, n1, n2]` (verbatim).
    #[must_use]
    pub fn normals(&self) -> [[f32; 3]; 3] {
        self.normals
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// World-space axis-aligned bounds: the union of the three vertices.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let [p0, p1, p2] = self.positions;
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

    /// Nearest intersection along `ray`, or `None` on a miss.
    ///
    /// Uses the two-sided Möller–Trumbore test for `(t, u, v)`, then builds the
    /// geometric normal from the edge cross product and the shading normal from
    /// the barycentric blend `w0·n0 + u·n1 + v·n2` (`w0 = 1 - u - v`). Both
    /// normals are oriented into the incoming ray's hemisphere, and the shading
    /// normal is forced to share the geometric normal's hemisphere (pbrt's
    /// convention) so it never points through the surface. A degenerate (near
    /// zero-length) interpolated normal falls back to the geometric normal.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<ShadedTriangleHit> {
        const EPS: f32 = 1e-8;
        let [p0, p1, p2] = self.positions;
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

        // Geometric normal from the face; orient it against the ray.
        let ng_raw = cross(e1, e2);
        let front_face = dot(dir, ng_raw) < 0.0;
        let geo = if front_face { ng_raw } else { negate(ng_raw) };
        let geometric_normal = normalize_or(geo, geo);

        // Barycentric blend of the vertex normals (w0 weights p0/n0).
        let [n0, n1, n2] = self.normals;
        let w0 = 1.0 - u - v;
        let ns_raw = [
            w0 * n0[0] + u * n1[0] + v * n2[0],
            w0 * n0[1] + u * n1[1] + v * n2[1],
            w0 * n0[2] + u * n1[2] + v * n2[2],
        ];
        // Force the shading normal into the geometric hemisphere, then unit it;
        // fall back to the geometric normal if the blend collapsed to zero.
        let ns_oriented = if dot(ns_raw, geometric_normal) < 0.0 {
            negate(ns_raw)
        } else {
            ns_raw
        };
        let shading_normal = normalize_or(ns_oriented, geometric_normal);

        Some(ShadedTriangleHit {
            t,
            u,
            v,
            primitive: self.primitive,
            geometric_normal,
            shading_normal,
            front_face,
        })
    }
}

/// A ray/[`ShadedTriangle`] intersection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadedTriangleHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Barycentric `u` (weight of vertex `1`).
    pub u: f32,
    /// Barycentric `v` (weight of vertex `2`); weight of vertex `0` is
    /// `1 - u - v`.
    pub v: f32,
    /// Stable primitive id carried by the hit triangle.
    pub primitive: u32,
    /// Unit geometric (flat-face) normal, oriented toward the incoming ray.
    pub geometric_normal: [f32; 3],
    /// Unit interpolated shading normal, in the geometric normal's hemisphere.
    pub shading_normal: [f32; 3],
    /// `true` when the ray struck the front face (geometric normal side).
    pub front_face: bool,
}

/// A single-level `BVH` over [`ShadedTriangle`]s.
#[derive(Clone, Debug, Default)]
pub struct ShadedTriangleBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Triangles reordered so each leaf owns a contiguous slice.
    triangles: Vec<ShadedTriangle>,
}

impl ShadedTriangleBvh {
    /// Builds a `BVH` over `triangles` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(triangles: &[ShadedTriangle]) -> Self {
        Self::build_with(triangles, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `triangles` with the given binned-`SAH` `config`.
    ///
    /// Each triangle's [`ShadedTriangle::aabb`] feeds the builder; the triangles
    /// are then reordered by the returned primitive order so every leaf slice
    /// indexes directly into [`ShadedTriangleBvh::triangles`].
    #[must_use]
    pub fn build_with(triangles: &[ShadedTriangle], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = triangles.iter().map(ShadedTriangle::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let triangles = order.iter().map(|&i| triangles[i as usize]).collect();
        Self { nodes, triangles }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of triangles in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.triangles.len()
    }

    /// True when the hierarchy holds no primitives.
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

    /// The reordered triangle array (leaf slices index into this).
    #[must_use]
    pub fn triangles(&self) -> &[ShadedTriangle] {
        &self.triangles
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ShadedTriangleHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<ShadedTriangleHit> = None;

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
                    for triangle in &self.triangles[start..end] {
                        if let Some(hit) = triangle.intersect(&ray) {
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
                    for triangle in &self.triangles[start..end] {
                        if triangle.intersect(ray).is_some() {
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
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn len(v: [f32; 3]) -> f32 {
        dot(v, v).sqrt()
    }

    /// A tilted triangle with distinct, non-planar-ish vertex normals so the
    /// interpolated shading normal genuinely varies across the face.
    fn sample_triangle(id: u32) -> ShadedTriangle {
        ShadedTriangle::new(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.3], [0.4, 2.0, -0.2]],
            [[0.0, 0.2, 1.0], [-0.3, 0.1, 1.0], [0.2, -0.2, 1.0]],
            id,
        )
    }

    #[test]
    fn geometric_normal_faces_the_ray() {
        let tri = sample_triangle(7);
        // Shoot from +z down toward the roughly-xy triangle.
        let ray = Ray::infinite([0.5, 0.4, 5.0], [0.0, 0.0, -1.0]);
        let hit = tri.intersect(&ray).expect("front hit");
        assert!(hit.front_face);
        // Oriented toward the ray => dot with the ray direction is negative.
        assert!(dot(hit.geometric_normal, ray.direction()) < 0.0);
        assert!(approx(len(hit.geometric_normal), 1.0, 1e-5));
    }

    #[test]
    fn back_face_flips_front_flag_and_normal() {
        let tri = sample_triangle(1);
        let front = tri
            .intersect(&Ray::infinite([0.5, 0.4, 5.0], [0.0, 0.0, -1.0]))
            .expect("front");
        let back = tri
            .intersect(&Ray::infinite([0.5, 0.4, -5.0], [0.0, 0.0, 1.0]))
            .expect("back");
        assert!(front.front_face);
        assert!(!back.front_face);
        // The two oriented geometric normals point into opposite hemispheres.
        assert!(dot(front.geometric_normal, back.geometric_normal) < 0.0);
    }

    #[test]
    fn shading_normal_shares_geometric_hemisphere_and_is_unit() {
        let tri = sample_triangle(3);
        let hit = tri
            .intersect(&Ray::infinite([0.6, 0.5, 4.0], [0.0, 0.0, -1.0]))
            .expect("hit");
        assert!(approx(len(hit.shading_normal), 1.0, 1e-5));
        assert!(dot(hit.shading_normal, hit.geometric_normal) >= -1e-6);
    }

    #[test]
    fn vertex_normal_is_recovered_at_a_vertex() {
        // Interpolation at vertex 0 (u = v = 0) must return the unit of n0.
        let tri = sample_triangle(9);
        // Aim straight at p0 along -z (p0 is the origin).
        let hit = tri
            .intersect(&Ray::infinite([0.0, 0.0, 3.0], [0.0, 0.0, -1.0]))
            .expect("hit at p0");
        assert!(approx(hit.u, 0.0, 1e-4));
        assert!(approx(hit.v, 0.0, 1e-4));
        let n0 = normalize_or([0.0, 0.2, 1.0], [0.0, 0.0, 1.0]);
        for (k, &expected) in n0.iter().enumerate() {
            assert!(approx(hit.shading_normal[k], expected, 1e-4));
        }
    }

    #[test]
    fn parallel_ray_misses() {
        let tri = sample_triangle(0);
        // A ray in the triangle's own plane direction barely grazes it.
        let ray = Ray::infinite([0.5, 0.4, 0.0], [1.0, 0.0, 0.0]);
        // Not asserting None universally (it can clip an edge); assert the
        // dedicated degenerate: a ray whose direction is parallel to the face.
        let e1 = sub([2.0, 0.0, 0.3], [0.0, 0.0, 0.0]);
        let e2 = sub([0.4, 2.0, -0.2], [0.0, 0.0, 0.0]);
        let face = cross(e1, e2);
        let _ = ray;
        // Build a ray parallel to the plane (perpendicular to the face normal).
        let in_plane = e1;
        assert!(approx(dot(in_plane, face), 0.0, 1e-4));
        let parallel = Ray::infinite([0.5, 0.4, 1.0], in_plane);
        assert!(tri.intersect(&parallel).is_none());
    }

    /// Independent residual oracle: the reported `(t, u, v)` must reconstruct
    /// the hit point from the vertices, the geometric normal must be the unit
    /// face normal (up to orientation), and the shading normal must be a unit
    /// vector in the geometric hemisphere. The oracle recomputes everything
    /// from first principles (areas, plane equation) rather than reusing
    /// `intersect`'s outputs.
    #[test]
    fn reported_hit_satisfies_barycentric_and_normal_residuals() {
        let mut rng = Rng::new(0x1234_ABCD);
        let mut checked = 0u32;
        for _ in 0..60_000 {
            let p0 = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
            let p1 = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
            let p2 = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
            let n0 = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.2, 1.0)];
            let n1 = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.2, 1.0)];
            let n2 = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.2, 1.0)];
            let tri = ShadedTriangle::new([p0, p1, p2], [n0, n1, n2], 0);

            // Skip near-degenerate triangles (tiny area).
            let face = cross(sub(p1, p0), sub(p2, p0));
            if len(face) < 0.2 {
                continue;
            }

            // Aim a ray at a random interior point so hits are common.
            let a = rng.range(0.05, 0.9);
            let b = rng.range(0.05, 0.9 - a).max(0.0);
            let target = [
                p0[0] + a * (p1[0] - p0[0]) + b * (p2[0] - p0[0]),
                p0[1] + a * (p1[1] - p0[1]) + b * (p2[1] - p0[1]),
                p0[2] + a * (p1[2] - p0[2]) + b * (p2[2] - p0[2]),
            ];
            let origin = [
                target[0] + rng.range(-2.0, 2.0),
                target[1] + rng.range(-2.0, 2.0),
                target[2] + rng.range(1.0, 3.0),
            ];
            let dir = sub(target, origin);
            let dir_len = len(dir);
            if dir_len < 1e-3 {
                continue;
            }
            // Skip grazing rays: when the ray is nearly parallel to the face
            // the single-precision Möller–Trumbore barycentrics lose precision,
            // which is a property of the intersector, not a correctness bug.
            let unit_face = normalize_or(face, [0.0, 0.0, 1.0]);
            let cos_incidence = dot(dir, unit_face).abs() / dir_len;
            if cos_incidence < 0.15 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let Some(hit) = tri.intersect(&ray) else {
                continue;
            };

            // 1) Reconstruct the hit point from barycentrics and from t.
            let w0 = 1.0 - hit.u - hit.v;
            let recon = [
                w0 * p0[0] + hit.u * p1[0] + hit.v * p2[0],
                w0 * p0[1] + hit.u * p1[1] + hit.v * p2[1],
                w0 * p0[2] + hit.u * p1[2] + hit.v * p2[2],
            ];
            let from_t = ray.at(hit.t);
            let scale = len(dir).max(1.0);
            for k in 0..3 {
                assert!(
                    approx(recon[k], from_t[k], 2e-3 * scale),
                    "barycentric reconstruction residual too large"
                );
            }

            // 2) Geometric normal is the unit face normal (either orientation).
            let aligned = approx(hit.geometric_normal[0], unit_face[0], 2e-3)
                && approx(hit.geometric_normal[1], unit_face[1], 2e-3)
                && approx(hit.geometric_normal[2], unit_face[2], 2e-3);
            let anti = approx(hit.geometric_normal[0], -unit_face[0], 2e-3)
                && approx(hit.geometric_normal[1], -unit_face[1], 2e-3)
                && approx(hit.geometric_normal[2], -unit_face[2], 2e-3);
            assert!(aligned || anti, "geometric normal is not the face normal");

            // 3) Shading normal is unit length and in the geometric hemisphere.
            assert!(approx(len(hit.shading_normal), 1.0, 2e-3));
            assert!(dot(hit.shading_normal, hit.geometric_normal) >= -2e-3);

            checked += 1;
        }
        assert!(checked > 2_000, "too few verified hits: {checked}");
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<ShadedTriangle> {
        (0..count)
            .map(|i| {
                let base = [
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                ];
                let offset = |rng: &mut Rng| {
                    [
                        rng.range(-1.5, 1.5),
                        rng.range(-1.5, 1.5),
                        rng.range(-1.5, 1.5),
                    ]
                };
                let o1 = offset(rng);
                let o2 = offset(rng);
                let positions = [
                    base,
                    [base[0] + o1[0], base[1] + o1[1], base[2] + o1[2]],
                    [base[0] + o2[0], base[1] + o2[1], base[2] + o2[2]],
                ];
                let normals = [
                    [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.3, 1.0)],
                    [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.3, 1.0)],
                    [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.3, 1.0)],
                ];
                ShadedTriangle::new(positions, normals, i)
            })
            .collect()
    }

    fn brute_closest(tris: &[ShadedTriangle], ray: &Ray) -> Option<ShadedTriangleHit> {
        let mut best: Option<ShadedTriangleHit> = None;
        let mut r = *ray;
        for tri in tris {
            if let Some(hit) = tri.intersect(&r) {
                r = Ray::new(r.origin(), r.direction(), r.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = ShadedTriangleBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force() {
        let mut rng = Rng::new(0x5EED_BEEF);
        let tris = random_scene(&mut rng, 80);
        let bvh = ShadedTriangleBvh::build(&tris);
        assert_eq!(bvh.primitive_count(), tris.len());

        for _ in 0..4_000 {
            let origin = [
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let bvh_hit = bvh.closest_hit(&ray);
            let brute = brute_closest(&bvh.triangles, &ray);
            match (bvh_hit, brute) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.primitive, b.primitive);
                    assert_eq!(a.t.to_bits(), b.t.to_bits());
                }
                (a, b) => panic!("hit disagreement: {a:?} vs {b:?}"),
            }
            assert_eq!(bvh.any_hit(&ray), brute_closest(&bvh.triangles, &ray).is_some());
        }
    }
}
