//! Two-level acceleration: a top-level `BVH` (`TLAS`) over affine-transformed
//! instances of bottom-level `BVH`s (`BLAS`).
//!
//! This is the `CPU`-verifiable golden reference for the two-level traversal a
//! `GPU` kernel mirrors. A [`Tlas`] holds its own flattened [`LinearBvhNode`]
//! array (built by the *same* [`build_linear_bvh`] `SAH` builder the `BLAS`
//! uses, over each instance's world-space bounds) plus a reordered instance
//! table. It does **not** own the [`Bvh`] geometry: instances reference a
//! shared `BLAS` pool by index, so a scene keeps one `Bvh` per unique mesh and
//! instances it many times with different transforms — the standard `AAA`
//! layout that keeps the acceleration memory bounded by unique geometry, not
//! by instance count.
//!
//! Traversal transforms the world ray into each instance's object space with
//! the cached inverse ([`Affine3::inverse`]). Because the direction is carried
//! through the *linear* part without renormalizing, the ray parameter `t` is
//! identical in both spaces, so object-space hit distances compare directly in
//! world space and the running `t_max` prunes across instances exactly like a
//! single-level walk.

use super::bvh::{build_linear_bvh, Aabb, Bvh, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An affine transform: a 3×3 linear map (stored column-major) plus a
/// translation applied after it.
///
/// `cols[c]` is the image of basis vector `c`, so a point `p` maps to
/// `cols[0]*p.x + cols[1]*p.y + cols[2]*p.z + translation`. Column-major
/// storage matches the `GPU`-side `mat3` convention the kernel consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine3 {
    cols: [[f32; 3]; 3],
    translation: [f32; 3],
}

impl Affine3 {
    /// Builds a transform from explicit columns and a translation.
    #[must_use]
    pub const fn from_cols(cols: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        Self { cols, translation }
    }

    /// The identity transform.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// A pure translation.
    #[must_use]
    pub const fn from_translation(t: [f32; 3]) -> Self {
        Self {
            cols: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: t,
        }
    }

    /// A pure (possibly non-uniform) scale about the origin.
    #[must_use]
    pub const fn from_scale(s: [f32; 3]) -> Self {
        Self {
            cols: [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, s[2]]],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// A rotation from a quaternion `[w, x, y, z]`.
    ///
    /// The quaternion is normalized defensively (a zero quaternion yields the
    /// identity), then converted with the standard arithmetic form — no
    /// trigonometry, keeping this module's `libm`-free determinism so a `GPU`
    /// kernel reproduces the matrix bit-for-bit.
    #[must_use]
    pub fn from_quaternion(q: [f32; 4]) -> Self {
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if len <= f32::MIN_POSITIVE {
            return Self::identity();
        }
        let (w, x, y, z) = (q[0] / len, q[1] / len, q[2] / len, q[3] / len);
        // Column-major: cols[c][r] = R[r][c] of the quaternion rotation matrix.
        Self {
            cols: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + w * z),
                    2.0 * (x * z - w * y),
                ],
                [
                    2.0 * (x * y - w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z + w * x),
                ],
                [
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Read-only columns of the linear part.
    #[must_use]
    pub const fn columns(&self) -> [[f32; 3]; 3] {
        self.cols
    }

    /// Read-only translation.
    #[must_use]
    pub const fn translation(&self) -> [f32; 3] {
        self.translation
    }

    /// Applies the linear part to a direction/vector (ignores translation).
    #[must_use]
    pub fn transform_vector(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.cols[0][0] * v[0] + self.cols[1][0] * v[1] + self.cols[2][0] * v[2],
            self.cols[0][1] * v[0] + self.cols[1][1] * v[1] + self.cols[2][1] * v[2],
            self.cols[0][2] * v[0] + self.cols[1][2] * v[1] + self.cols[2][2] * v[2],
        ]
    }

    /// Applies the full affine map to a point.
    #[must_use]
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        let v = self.transform_vector(p);
        [
            v[0] + self.translation[0],
            v[1] + self.translation[1],
            v[2] + self.translation[2],
        ]
    }

    /// Composition `self ∘ inner`: the transform that applies `inner` first,
    /// then `self`.
    #[must_use]
    pub fn compose(&self, inner: &Affine3) -> Affine3 {
        Affine3 {
            cols: [
                self.transform_vector(inner.cols[0]),
                self.transform_vector(inner.cols[1]),
                self.transform_vector(inner.cols[2]),
            ],
            translation: self.transform_point(inner.translation),
        }
    }

    /// Inverse transform, or `None` when the linear part is singular.
    ///
    /// Uses the closed-form 3×3 adjugate over the determinant; the inverse
    /// translation is `-L⁻¹ · t` so that `inv.transform_point(self.transform_point(p)) == p`.
    #[must_use]
    pub fn inverse(&self) -> Option<Affine3> {
        // Row-form entries m[r][c] = cols[c][r].
        let a = self.cols[0][0];
        let d = self.cols[0][1];
        let g = self.cols[0][2];
        let b = self.cols[1][0];
        let e = self.cols[1][1];
        let h = self.cols[1][2];
        let c = self.cols[2][0];
        let f = self.cols[2][1];
        let i = self.cols[2][2];

        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if det.abs() <= f32::MIN_POSITIVE {
            return None;
        }
        let inv_det = 1.0 / det;

        // Inverse in row form.
        let inv00 = (e * i - f * h) * inv_det;
        let inv01 = (c * h - b * i) * inv_det;
        let inv02 = (b * f - c * e) * inv_det;
        let inv10 = (f * g - d * i) * inv_det;
        let inv11 = (a * i - c * g) * inv_det;
        let inv12 = (c * d - a * f) * inv_det;
        let inv20 = (d * h - e * g) * inv_det;
        let inv21 = (b * g - a * h) * inv_det;
        let inv22 = (a * e - b * d) * inv_det;

        // Back to column-major: inv_cols[c][r] = inv[r][c].
        let inv_cols = [
            [inv00, inv10, inv20],
            [inv01, inv11, inv21],
            [inv02, inv12, inv22],
        ];
        let t = self.translation;
        let inv_translation = [
            -(inv00 * t[0] + inv01 * t[1] + inv02 * t[2]),
            -(inv10 * t[0] + inv11 * t[1] + inv12 * t[2]),
            -(inv20 * t[0] + inv21 * t[1] + inv22 * t[2]),
        ];
        Some(Affine3 {
            cols: inv_cols,
            translation: inv_translation,
        })
    }
}

/// One placement of a `BLAS` into the scene.
///
/// Holds both the object→world transform and its cached inverse (world→object,
/// computed once at construction) plus the `BLAS` pool index the instance
/// references and a stable user-facing `instance_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    object_to_world: Affine3,
    world_to_object: Affine3,
    blas: usize,
    instance_id: u32,
}

impl Instance {
    /// Builds an instance of `BLAS` pool entry `blas` placed by `object_to_world`.
    ///
    /// Returns `None` when the transform is non-invertible (degenerate scale),
    /// since such an instance has no well-defined object space to trace in.
    #[must_use]
    pub fn new(object_to_world: Affine3, blas: usize, instance_id: u32) -> Option<Self> {
        let world_to_object = object_to_world.inverse()?;
        Some(Self {
            object_to_world,
            world_to_object,
            blas,
            instance_id,
        })
    }

    /// Object→world transform.
    #[must_use]
    pub const fn object_to_world(&self) -> Affine3 {
        self.object_to_world
    }

    /// Cached world→object transform.
    #[must_use]
    pub const fn world_to_object(&self) -> Affine3 {
        self.world_to_object
    }

    /// `BLAS` pool index this instance references.
    #[must_use]
    pub const fn blas(&self) -> usize {
        self.blas
    }

    /// Stable user-facing instance id, echoed on every [`TlasHit`].
    #[must_use]
    pub const fn instance_id(&self) -> u32 {
        self.instance_id
    }
}

/// A ray/instance intersection reported by the `TLAS`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TlasHit {
    /// Ray parameter at the hit (identical in world and object space).
    pub t: f32,
    /// Barycentric `u` (weight of the hit triangle's `v1`).
    pub u: f32,
    /// Barycentric `v` (weight of the hit triangle's `v2`).
    pub v: f32,
    /// Stable primitive id from the hit `BLAS` triangle.
    pub primitive: u32,
    /// Stable id of the instance that was hit.
    pub instance_id: u32,
    /// Index of the hit instance in [`Tlas::instances`] (reordered order).
    pub instance_index: u32,
}

/// A built top-level acceleration structure over a set of [`Instance`]s.
///
/// Empty input yields an empty structure; traversal never reports a hit.
#[derive(Clone, Debug, PartialEq)]
pub struct Tlas {
    nodes: Vec<LinearBvhNode>,
    instances: Vec<Instance>,
}

/// World-space bounds of `inst`: the `BLAS` root box's eight corners mapped
/// through `object_to_world` and unioned. An empty `BLAS` yields empty bounds.
fn instance_world_bounds(inst: &Instance, blases: &[Bvh]) -> Aabb {
    let local = blases[inst.blas].bounds();
    if local.is_empty() {
        return Aabb::empty();
    }
    let mut out = Aabb::empty();
    for &cx in &[local.min[0], local.max[0]] {
        for &cy in &[local.min[1], local.max[1]] {
            for &cz in &[local.min[2], local.max[2]] {
                out = out.enclose(inst.object_to_world.transform_point([cx, cy, cz]));
            }
        }
    }
    out
}

impl Tlas {
    /// Builds a `TLAS` over `instances` (referencing `blases`) with the default
    /// [`BvhBuildConfig`].
    #[must_use]
    pub fn build(instances: &[Instance], blases: &[Bvh]) -> Self {
        Self::build_with(instances, blases, BvhBuildConfig::default())
    }

    /// Builds a `TLAS` with an explicit builder configuration.
    #[must_use]
    pub fn build_with(instances: &[Instance], blases: &[Bvh], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = instances
            .iter()
            .map(|inst| instance_world_bounds(inst, blases))
            .collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let instances = order.iter().map(|&i| instances[i as usize]).collect();
        Self { nodes, instances }
    }

    /// Number of nodes in the flattened top-level hierarchy.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// True when the `TLAS` holds no instances.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Read-only view of the flattened nodes (depth-first order).
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// Read-only view of the reordered instance table; [`TlasHit::instance_index`]
    /// indexes into this slice.
    #[must_use]
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// Root world-space bounds, or [`Aabb::empty`] when the `TLAS` is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |n| n.bounds)
    }

    /// Nearest intersection along the world-space `ray`, or `None`.
    ///
    /// `blases` must be the same pool passed to [`Tlas::build`]. Walks the
    /// top-level nodes with an explicit stack, and in each leaf transforms the
    /// ray into every instance's object space to query its `BLAS`, keeping the
    /// globally nearest hit. The running `t_max` shrinks across instances so
    /// far candidates are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray, blases: &[Bvh]) -> Option<TlasHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<TlasHit> = None;
        let mut best_t = ray.t_max();
        let t_min = ray.t_min();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray.aabb_interval(&node.bounds, t_min, best_t).is_some() {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for (offset, inst) in self.instances[start..end].iter().enumerate() {
                        let obj_origin = inst.world_to_object.transform_point(ray.origin());
                        let obj_dir = inst.world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, best_t);
                        if let Some(hit) = blases[inst.blas].closest_hit(&obj_ray)
                            && hit.t < best_t
                        {
                            best_t = hit.t;
                            best = Some(TlasHit {
                                t: hit.t,
                                u: hit.u,
                                v: hit.v,
                                primitive: hit.primitive,
                                instance_id: inst.instance_id,
                                instance_index: (start + offset) as u32,
                            });
                        }
                    }
                    match pop(&mut stack, &mut sp) {
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
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// True when *any* instance intersects the world-space `ray` inside its
    /// interval. Returns on the first hit; the cheap shadow/AO query.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray, blases: &[Bvh]) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let t_min = ray.t_min();
        let t_max = ray.t_max();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if ray.aabb_interval(&node.bounds, t_min, t_max).is_some() {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for inst in &self.instances[start..end] {
                        let obj_origin = inst.world_to_object.transform_point(ray.origin());
                        let obj_dir = inst.world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, t_max);
                        if blases[inst.blas].any_hit(&obj_ray) {
                            return true;
                        }
                    }
                    match pop(&mut stack, &mut sp) {
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
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

#[inline]
fn pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        return None;
    }
    *sp -= 1;
    Some(stack[*sp])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::Triangle;

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
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    fn approx_pt(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        approx(a[0], b[0], eps) && approx(a[1], b[1], eps) && approx(a[2], b[2], eps)
    }

    /// A small non-trivial BLAS: two axis-facing quads (4 triangles) offset so
    /// rays along -z and +x both find geometry.
    fn sample_blas() -> Bvh {
        let tris = vec![
            // Quad in the z = 0 plane spanning x,y in [-1,1] (primitives 0,1).
            Triangle::new([-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 0),
            Triangle::new([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], 1),
            // Quad in the x = 0.5 plane spanning y,z in [-1,1] (primitives 2,3).
            Triangle::new([0.5, -1.0, -1.0], [0.5, 1.0, -1.0], [0.5, 1.0, 1.0], 2),
            Triangle::new([0.5, -1.0, -1.0], [0.5, 1.0, 1.0], [0.5, -1.0, 1.0], 3),
        ];
        Bvh::build(&tris)
    }

    fn random_affine(rng: &mut Rng) -> Affine3 {
        let quat = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let rot = Affine3::from_quaternion(quat);
        // Keep scales comfortably away from zero so the transform is invertible.
        let scale = Affine3::from_scale([
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
        ]);
        let trans = Affine3::from_translation([
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ]);
        // Apply scale, then rotation, then translation.
        trans.compose(&rot.compose(&scale))
    }

    /// Brute-force reference: query every instance's BLAS in object space and
    /// keep the globally nearest hit. Compared to `Tlas::closest_hit` by the
    /// stable `(instance_id, primitive, t)` tuple.
    fn brute_closest(
        instances: &[Instance],
        blases: &[Bvh],
        ray: &Ray,
    ) -> Option<(u32, u32, f32)> {
        let mut best: Option<(u32, u32, f32)> = None;
        let mut best_t = ray.t_max();
        for inst in instances {
            let obj_origin = inst.world_to_object().transform_point(ray.origin());
            let obj_dir = inst.world_to_object().transform_vector(ray.direction());
            let obj_ray = Ray::new(obj_origin, obj_dir, ray.t_min(), ray.t_max());
            if let Some(hit) = blases[inst.blas()].closest_hit(&obj_ray)
                && hit.t < best_t
            {
                best_t = hit.t;
                best = Some((inst.instance_id(), hit.primitive, hit.t));
            }
        }
        best
    }

    #[test]
    fn affine_inverse_roundtrips() {
        let mut rng = Rng::new(0x1234_5678);
        for _ in 0..2000 {
            let m = random_affine(&mut rng);
            let inv = m.inverse().expect("random affine is invertible");
            let p = [
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
            ];
            let round = inv.transform_point(m.transform_point(p));
            assert!(approx_pt(round, p, 1e-4), "roundtrip {round:?} != {p:?}");
            // Direction roundtrip (linear part only).
            let v = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            let vr = inv.transform_vector(m.transform_vector(v));
            assert!(approx_pt(vr, v, 1e-4), "vec roundtrip {vr:?} != {v:?}");
        }
    }

    #[test]
    fn singular_transform_has_no_inverse_and_no_instance() {
        let flat = Affine3::from_scale([1.0, 0.0, 1.0]);
        assert!(flat.inverse().is_none());
        assert!(Instance::new(flat, 0, 0).is_none());
    }

    #[test]
    fn identity_instance_matches_blas_directly() {
        let blas = sample_blas();
        let blases = vec![blas.clone()];
        let inst = Instance::new(Affine3::identity(), 0, 7).unwrap();
        let tlas = Tlas::build(&[inst], &blases);

        let mut rng = Rng::new(99);
        for _ in 0..3000 {
            let origin = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(2.0, 6.0)];
            let target = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let ray = Ray::infinite(origin, dir);
            let direct = blas.closest_hit(&ray);
            let via = tlas.closest_hit(&ray, &blases);
            match (direct, via) {
                (None, None) => {}
                (Some(d), Some(v)) => {
                    assert_eq!(d.primitive, v.primitive);
                    assert_eq!(v.instance_id, 7);
                    assert!(approx(d.t, v.t, 1e-5), "t {} != {}", d.t, v.t);
                    assert!(approx(d.u, v.u, 1e-4));
                    assert!(approx(d.v, v.v, 1e-4));
                }
                (d, v) => panic!("existence mismatch: {d:?} vs {v:?}"),
            }
        }
    }

    #[test]
    fn transformed_hit_point_is_consistent_in_world_space() {
        // For any hit, world_point == object_to_world(object_point). Since t is
        // preserved, ray.at(t) in world must equal the transformed object hit.
        let blas = sample_blas();
        let blases = vec![blas];
        let mut rng = Rng::new(0xDEAD_BEEF);
        for _ in 0..200 {
            let m = random_affine(&mut rng);
            let inst = Instance::new(m, 0, 42).unwrap();
            let tlas = Tlas::build(&[inst], &blases);
            for _ in 0..40 {
                let origin = [
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                ];
                let dir = [
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                ];
                let ray = Ray::infinite(origin, dir);
                if let Some(hit) = tlas.closest_hit(&ray, &blases) {
                    let world_pt = ray.at(hit.t);
                    // Reconstruct object-space hit point and map it forward.
                    let obj_origin = inst.world_to_object().transform_point(origin);
                    let obj_dir = inst.world_to_object().transform_vector(dir);
                    let obj_pt = [
                        obj_origin[0] + hit.t * obj_dir[0],
                        obj_origin[1] + hit.t * obj_dir[1],
                        obj_origin[2] + hit.t * obj_dir[2],
                    ];
                    let mapped = inst.object_to_world().transform_point(obj_pt);
                    assert!(
                        approx_pt(world_pt, mapped, 2e-3),
                        "world {world_pt:?} != mapped {mapped:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn multi_instance_returns_global_nearest() {
        let blas = sample_blas();
        let blases = vec![blas];
        // Two z=0 quads translated to z=2 and z=5; a ray down -z from z=10 hits
        // the nearer (z=5) instance first.
        let near = Instance::new(Affine3::from_translation([0.0, 0.0, 5.0]), 0, 100).unwrap();
        let far = Instance::new(Affine3::from_translation([0.0, 0.0, 2.0]), 0, 200).unwrap();
        let tlas = Tlas::build(&[far, near], &blases);
        let ray = Ray::infinite([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let hit = tlas.closest_hit(&ray, &blases).expect("hits nearer quad");
        assert_eq!(hit.instance_id, 100);
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
    }

    #[test]
    fn random_scene_matches_brute_force() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0xA5A5_1234);
        // Build a handful of random scenes, each cross-checked over many rays.
        for _ in 0..40 {
            let n = 1 + (rng.next_u32() % 12) as usize;
            let mut instances = Vec::with_capacity(n);
            for id in 0..n {
                let m = random_affine(&mut rng);
                instances.push(Instance::new(m, 0, id as u32).unwrap());
            }
            let tlas = Tlas::build(&instances, &blases);
            for _ in 0..300 {
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
                let ray = Ray::infinite(origin, dir);
                let brute = brute_closest(&instances, &blases, &ray);
                let via = tlas.closest_hit(&ray, &blases);
                match (brute, via) {
                    (None, None) => {}
                    (Some((bid, bprim, bt)), Some(v)) => {
                        assert_eq!(bid, v.instance_id, "instance mismatch");
                        assert_eq!(bprim, v.primitive, "primitive mismatch");
                        assert!(approx(bt, v.t, 1e-4), "t {bt} != {}", v.t);
                        // instance_index must resolve back to the same id.
                        assert_eq!(
                            tlas.instances()[v.instance_index as usize].instance_id(),
                            v.instance_id
                        );
                    }
                    (b, v) => panic!("existence mismatch: {b:?} vs {v:?}"),
                }
            }
        }
    }

    #[test]
    fn any_hit_agrees_with_closest_hit() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x0BADF00D);
        for _ in 0..20 {
            let n = 1 + (rng.next_u32() % 8) as usize;
            let instances: Vec<Instance> = (0..n)
                .map(|id| Instance::new(random_affine(&mut rng), 0, id as u32).unwrap())
                .collect();
            let tlas = Tlas::build(&instances, &blases);
            for _ in 0..400 {
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
                let ray = Ray::infinite(origin, dir);
                assert_eq!(
                    tlas.any_hit(&ray, &blases),
                    tlas.closest_hit(&ray, &blases).is_some()
                );
            }
        }
    }

    #[test]
    fn empty_tlas_and_empty_blas_never_hit() {
        let blases: Vec<Bvh> = vec![Bvh::build(&[])];
        let empty = Tlas::build(&[], &blases);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(empty.is_empty());
        assert!(empty.closest_hit(&ray, &blases).is_none());
        assert!(!empty.any_hit(&ray, &blases));

        // Instance of an empty BLAS: valid transform, but nothing to hit.
        let inst = Instance::new(Affine3::identity(), 0, 0).unwrap();
        let tlas = Tlas::build(&[inst], &blases);
        assert!(tlas.closest_hit(&ray, &blases).is_none());
        assert!(!tlas.any_hit(&ray, &blases));
    }

    #[test]
    fn scale_changes_hit_distance_predictably() {
        // A 2× uniform scale doubles the world-space distance to the same quad.
        let blases = vec![sample_blas()];
        let unit = Instance::new(Affine3::identity(), 0, 0).unwrap();
        let scaled = Instance::new(Affine3::from_scale([2.0, 2.0, 2.0]), 0, 0).unwrap();
        let tlas_unit = Tlas::build(&[unit], &blases);
        let tlas_scaled = Tlas::build(&[scaled], &blases);
        let ray = Ray::infinite([0.0, 0.0, 4.0], [0.0, 0.0, -1.0]);
        let h_unit = tlas_unit.closest_hit(&ray, &blases).unwrap();
        let h_scaled = tlas_scaled.closest_hit(&ray, &blases).unwrap();
        // Unit quad at z=0 -> t=4; scaled quad still at z=0 (scale about origin) -> t=4.
        assert!(approx(h_unit.t, 4.0, 1e-4));
        assert!(approx(h_scaled.t, 4.0, 1e-4));
        // Offsetting the scaled instance moves the plane to z=0 as well, but a
        // ray starting closer confirms t tracks world distance.
        let ray2 = Ray::infinite([0.0, 0.0, 4.0], [0.0, 0.0, -2.0]);
        let h2 = tlas_unit.closest_hit(&ray2, &blases).unwrap();
        assert!(approx(h2.t, 2.0, 1e-4), "t = {}", h2.t);
    }
}
