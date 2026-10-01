//! Analytic *bilinear patch* primitive and its single-level `BVH`.
//!
//! A bilinear patch is the ruled surface spanned by four corner points
//! `p00`, `p10`, `p11`, `p01`:
//!
//! ```text
//! P(u, v) = (1 − u)(1 − v)·p00 + u(1 − v)·p10 + u·v·p11 + (1 − u)·v·p01
//! ```
//!
//! with `u, v ∈ [0, 1]`. It is the lowest-order non-planar quad: four coplanar
//! corners give a flat quadrilateral, four non-coplanar corners give a saddle.
//! Bilinear patches are the native primitive for subdivision cages, cloth and
//! hair cards, displaced quads, and curved-quad tessellation, so a direct
//! analytic intersector avoids splitting every quad into two triangles (which
//! loses the curvature and introduces a diagonal seam).
//!
//! The intersection follows Reshetov's *"Cool Patches"* (Ray Tracing Gems,
//! 2019): for a fixed `u` the patch is a straight segment in `v`, so the ray is
//! first reduced to a quadratic in `u` whose (at most two) roots are each
//! back-substituted to recover `v` and the ray parameter `t`. Every step is
//! add/sub/mul/div with a single `sqrt` and a `copysign` for the numerically
//! stable quadratic, so the test is bit-reproducible on the `GPU` and free of
//! any transcendental call. The ray direction is never assumed unit: `t` comes
//! out directly in `direction`-length units (the whole solve scales linearly in
//! the direction), matching the convention [`Ray::at`] uses.
//!
//! Like the other `ray_scene` primitives this is a procedural primitive for the
//! `DXR`/Vulkan `AABB` path: the `BLAS` stores the corner `AABB` and the
//! intersection shader runs the analytic solve.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic bilinear patch defined by its four corners.
///
/// Corners follow the `(u, v)` convention `p00 = P(0, 0)`, `p10 = P(1, 0)`,
/// `p11 = P(1, 1)`, `p01 = P(0, 1)`, so `p00→p10` is the `v = 0` edge and
/// `p00→p01` is the `u = 0` edge. `primitive` is the caller's stable id
/// (mirroring [`super::bvh::Triangle`] and [`super::round_cone::RoundCone`]):
/// the [`BilinearPatchBvh`] builder reorders primitives internally but always
/// reports hits by this id. Corners are stored verbatim, so a packed patch
/// decodes back through [`BilinearPatch::new`] bit-for-bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilinearPatch {
    /// Corner at `(u, v) = (0, 0)`.
    p00: [f32; 3],
    /// Corner at `(u, v) = (1, 0)`.
    p10: [f32; 3],
    /// Corner at `(u, v) = (1, 1)`.
    p11: [f32; 3],
    /// Corner at `(u, v) = (0, 1)`.
    p01: [f32; 3],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl BilinearPatch {
    /// Builds a bilinear patch from its four corners (stored verbatim), tagged
    /// with stable id `primitive`.
    ///
    /// Corners use the `(u, v)` convention `p00 = P(0, 0)`, `p10 = P(1, 0)`,
    /// `p11 = P(1, 1)`, `p01 = P(0, 1)`.
    #[must_use]
    pub fn new(
        p00: [f32; 3],
        p10: [f32; 3],
        p11: [f32; 3],
        p01: [f32; 3],
        primitive: u32,
    ) -> Self {
        Self {
            p00,
            p10,
            p11,
            p01,
            primitive,
        }
    }

    /// Corner at `(u, v) = (0, 0)`.
    #[must_use]
    pub fn p00(&self) -> [f32; 3] {
        self.p00
    }

    /// Corner at `(u, v) = (1, 0)`.
    #[must_use]
    pub fn p10(&self) -> [f32; 3] {
        self.p10
    }

    /// Corner at `(u, v) = (1, 1)`.
    #[must_use]
    pub fn p11(&self) -> [f32; 3] {
        self.p11
    }

    /// Corner at `(u, v) = (0, 1)`.
    #[must_use]
    pub fn p01(&self) -> [f32; 3] {
        self.p01
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Surface point `P(u, v)` for parameters `u, v ∈ [0, 1]`.
    #[must_use]
    pub fn point(&self, u: f32, v: f32) -> [f32; 3] {
        let bottom = mix(self.p00, self.p10, u);
        let top = mix(self.p01, self.p11, u);
        mix(bottom, top, v)
    }

    /// Axis-aligned bounds of the four corners.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. A
    /// bilinear patch is contained in the convex hull of its corners, so the
    /// corner `AABB` is a tight, correct bound.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut min = self.p00;
        let mut max = self.p00;
        for c in [self.p10, self.p11, self.p01] {
            for axis in 0..3 {
                min[axis] = min[axis].min(c[axis]);
                max[axis] = max[axis].max(c[axis]);
            }
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/patch intersection inside the ray interval, or `None`.
    ///
    /// Reduces the ray to a quadratic in `u` (Reshetov 2019), back-substitutes
    /// each root in `[0, 1]` to recover `v` and `t`, and keeps the nearest valid
    /// hit. [`BilinearPatchHit::normal`] is the unit surface normal oriented
    /// *against* the incident ray and [`BilinearPatchHit::front_face`] is `true`
    /// when the ray struck the outward-facing side (the side the geometric
    /// normal `∂P/∂u × ∂P/∂v` points toward).
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<BilinearPatchHit> {
        let ro = ray.origin();
        let rd = ray.direction();

        // Translation-invariant edge vectors (used for the quadratic and for the
        // analytic surface normal).
        let e10 = sub(self.p10, self.p00); // ∂/∂u along the v = 0 edge.
        let e11 = sub(self.p11, self.p10); // v = 1 - side u edge direction term.
        let e00 = sub(self.p01, self.p00); // ∂/∂v along the u = 0 edge.
        let qn = cross(e10, sub(self.p01, self.p11));

        // Corners relative to the ray origin.
        let q00 = sub(self.p00, ro);
        let q10 = sub(self.p10, ro);

        // Quadratic a·u² + b·u + c = 0 (Reshetov's formulation).
        let a = dot(cross(q00, rd), e00);
        let c = dot(qn, rd);
        let b = dot(cross(q10, rd), e11) - a - c;

        let (u1, u2) = {
            let det = b * b - 4.0 * a * c;
            if det < 0.0 {
                return None;
            }
            let sq = det.sqrt();
            if c == 0.0 {
                // Degenerate (planar in u): linear equation b·u + a = 0.
                if b == 0.0 {
                    return None;
                }
                (-a / b, -1.0)
            } else {
                // Numerically stable roots: the large root is formed with a
                // same-sign addition, the small one via the product a/c.
                let big = (-b - sq.copysign(b)) * 0.5;
                (big / c, a / big)
            }
        };

        let mut best: Option<BilinearPatchHit> = None;
        let mut t_hi = ray.t_max();
        for u in [u1, u2] {
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            if let Some((t, v)) = self.solve_v(ray, q00, q10, e00, e11, u, t_hi) {
                let normal = self.oriented_normal(e10, e00, e11, u, v, rd);
                if let Some((normal, front_face)) = normal {
                    t_hi = t;
                    best = Some(BilinearPatchHit {
                        t,
                        primitive: self.primitive,
                        normal,
                        front_face,
                        u,
                        v,
                    });
                }
            }
        }
        best
    }

    /// Back-substitutes a `u` root to recover `(t, v)` for the vertical line at
    /// that `u`, returning `None` when the ray misses that line, the hit is
    /// behind the current nearest `t_hi`, or `v` leaves `[0, 1]`.
    fn solve_v(
        &self,
        ray: &Ray,
        q00: [f32; 3],
        q10: [f32; 3],
        e00: [f32; 3],
        e11: [f32; 3],
        u: f32,
        t_hi: f32,
    ) -> Option<(f32, f32)> {
        let rd = ray.direction();
        // Bottom point (relative to origin) and the vertical edge direction at u.
        let pa = mix(q00, q10, u);
        let pb = mix(e00, e11, u);
        let n0 = cross(rd, pb);
        let det = dot(n0, n0);
        if det <= 0.0 {
            return None;
        }
        let m = cross(n0, pa);
        let t = dot(m, pb) / det;
        let v = dot(m, rd) / det;
        if !(0.0..=1.0).contains(&v) {
            return None;
        }
        if t < ray.t_min() || t > t_hi {
            return None;
        }
        Some((t, v))
    }

    /// Unit surface normal at `(u, v)`, oriented against the incident ray.
    ///
    /// The geometric normal is `∂P/∂u × ∂P/∂v`; `front_face` records whether the
    /// ray approached that outward side before the normal is flipped to face the
    /// ray. Returns `None` for a degenerate (zero-area) tangent frame.
    fn oriented_normal(
        &self,
        e10: [f32; 3],
        e00: [f32; 3],
        e11: [f32; 3],
        u: f32,
        v: f32,
        rd: [f32; 3],
    ) -> Option<([f32; 3], bool)> {
        let f = sub(self.p11, self.p01);
        let dpdu = add(scale(e10, 1.0 - v), scale(f, v));
        let dpdv = add(scale(e00, 1.0 - u), scale(e11, u));
        let g = cross(dpdu, dpdv);
        let len2 = dot(g, g);
        if len2 <= 0.0 {
            return None;
        }
        let inv_len = 1.0 / len2.sqrt();
        let outward = scale(g, inv_len);
        let front_face = dot(rd, outward) < 0.0;
        let normal = if front_face {
            outward
        } else {
            [-outward[0], -outward[1], -outward[2]]
        };
        Some((normal, front_face))
    }
}

/// A ray/bilinear-patch intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilinearPatchHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the bilinear patch that was hit.
    pub primitive: u32,
    /// Unit surface normal, oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the ray struck the outward-facing side.
    pub front_face: bool,
    /// Patch `u` parameter of the hit, in `[0, 1]`.
    pub u: f32,
    /// Patch `v` parameter of the hit, in `[0, 1]`.
    pub v: f32,
}

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Adds `a` and `b` componentwise.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by scalar `s`.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a × b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Linear interpolation `a + t·(b − a)`, componentwise.
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// A single-level `BVH` over analytic [`BilinearPatch`] primitives.
///
/// Empty input yields an empty hierarchy ([`BilinearPatchBvh::is_empty`]);
/// traversal of an empty hierarchy simply never reports a hit. The layout and
/// ordered slab walk mirror the triangle [`super::bvh::Bvh`] and
/// [`super::round_cone::RoundConeBvh`] so every primitive kind shares one
/// acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct BilinearPatchBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Patches reordered so each leaf owns a contiguous slice.
    patches: Vec<BilinearPatch>,
}

impl BilinearPatchBvh {
    /// Builds a `BVH` over `patches` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(patches: &[BilinearPatch]) -> Self {
        Self::build_with(patches, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `patches` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each patch's [`BilinearPatch::aabb`] and reorders
    /// the primitives by the returned order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`BilinearPatchBvh::patches`].
    #[must_use]
    pub fn build_with(patches: &[BilinearPatch], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = patches.iter().map(BilinearPatch::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let patches = order.iter().map(|&i| patches[i as usize]).collect();
        Self { nodes, patches }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of patches in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.patches.len()
    }

    /// True when the hierarchy holds no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or an empty box when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or_else(Aabb::empty, |n| n.bounds)
    }

    /// Flattened `BVH` nodes (root at index `0` when present).
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// Patches in leaf-contiguous order.
    #[must_use]
    pub fn patches(&self) -> &[BilinearPatch] {
        &self.patches
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<BilinearPatchHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<BilinearPatchHit> = None;

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
                    for patch in &self.patches[start..end] {
                        if let Some(hit) = patch.intersect(&ray) {
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

    /// True when *any* patch intersects `ray` inside its interval.
    ///
    /// Returns on the first hit without tracking the nearest, so it is the cheap
    /// query for shadow and ambient-occlusion rays.
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
                    for patch in &self.patches[start..end] {
                        if patch.intersect(ray).is_some() {
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

    fn random_patch(rng: &mut Rng, primitive: u32) -> BilinearPatch {
        let p = |rng: &mut Rng| {
            [
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
            ]
        };
        BilinearPatch::new(p(rng), p(rng), p(rng), p(rng), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<BilinearPatch> {
        (0..count).map(|i| random_patch(rng, i)).collect()
    }

    fn norm(a: [f32; 3]) -> f32 {
        dot(a, a).sqrt()
    }

    #[test]
    fn point_matches_corner_parameters() {
        let patch = BilinearPatch::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0,
        );
        assert_eq!(patch.point(0.0, 0.0), patch.p00());
        assert_eq!(patch.point(1.0, 0.0), patch.p10());
        assert_eq!(patch.point(1.0, 1.0), patch.p11());
        assert_eq!(patch.point(0.0, 1.0), patch.p01());
    }

    #[test]
    fn flat_quad_is_hit_through_its_center() {
        // Unit square in the z = 0 plane.
        let patch = BilinearPatch::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            7,
        );
        let ray = Ray::infinite([0.5, 0.5, 2.0], [0.0, 0.0, -1.0]);
        let hit = patch.intersect(&ray).expect("center hit");
        assert_eq!(hit.primitive, 7);
        assert!((hit.t - 2.0).abs() < 1e-4, "t = {}", hit.t);
        assert!((hit.u - 0.5).abs() < 1e-4, "u = {}", hit.u);
        assert!((hit.v - 0.5).abs() < 1e-4, "v = {}", hit.v);
        assert!(hit.front_face);
        // Normal faces the ray (+z) on the front side.
        assert!(hit.normal[2] > 0.0, "normal = {:?}", hit.normal);
    }

    #[test]
    fn flat_quad_misses_outside_its_bounds() {
        let patch = BilinearPatch::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0,
        );
        // Pierces z = 0 well outside the [0,1]² footprint.
        let ray = Ray::infinite([5.0, 5.0, 2.0], [0.0, 0.0, -1.0]);
        assert!(patch.intersect(&ray).is_none());
    }

    #[test]
    fn back_face_flips_the_normal() {
        let patch = BilinearPatch::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0,
        );
        // Approach from below (−z side): ray travels +z.
        let ray = Ray::infinite([0.5, 0.5, -2.0], [0.0, 0.0, 1.0]);
        let hit = patch.intersect(&ray).expect("back hit");
        assert!(!hit.front_face);
        // Normal is flipped to oppose the +z ray.
        assert!(hit.normal[2] < 0.0, "normal = {:?}", hit.normal);
    }

    #[test]
    fn t_max_excludes_a_far_hit() {
        let patch = BilinearPatch::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0,
        );
        // The plane is at t = 2 for a unit −z ray from z = 2; cap t_max below it.
        let ray = Ray::new([0.5, 0.5, 2.0], [0.0, 0.0, -1.0], 0.0, 1.0);
        assert!(patch.intersect(&ray).is_none());
    }

    #[test]
    fn saddle_patch_is_curved_not_planar() {
        // Classic saddle: opposite corners lifted along ±z.
        let patch = BilinearPatch::new(
            [0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            3,
        );
        // Center P(0.5,0.5) = average of corners = z = 0.
        let center = patch.point(0.5, 0.5);
        assert!(center[2].abs() < 1e-6, "center z = {}", center[2]);
        let ray = Ray::infinite([0.5, 0.5, 5.0], [0.0, 0.0, -1.0]);
        let hit = patch.intersect(&ray).expect("saddle center hit");
        assert!((hit.t - 5.0).abs() < 1e-3, "t = {}", hit.t);
        let p = patch.point(hit.u, hit.v);
        assert!(norm(sub(p, [0.5, 0.5, 0.0])) < 1e-3);
    }

    #[test]
    fn reported_hit_lies_on_the_parametric_surface() {
        // Independent residual check: for every hit the solver reports, the ray
        // point `origin + t·dir` must equal the parametric point `P(u, v)` and
        // the normal must be perpendicular to both tangents. `point` does not
        // reuse `intersect`, so this validates the analytic solve, not just the
        // traversal.
        let mut rng = Rng::new(0x0B1E_11A7);
        let mut hits = 0u32;
        for _ in 0..40_000 {
            let patch = random_patch(&mut rng, 0);
            // Aim rays at a jittered point near the patch so hits are frequent.
            let target = patch.point(rng.unit(), rng.unit());
            let origin = [
                target[0] + rng.range(-6.0, 6.0),
                target[1] + rng.range(-6.0, 6.0),
                target[2] + rng.range(-6.0, 6.0),
            ];
            let dir = sub(target, origin);
            if dot(dir, dir) < 1e-4 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let Some(hit) = patch.intersect(&ray) else {
                continue;
            };
            assert!((0.0..=1.0).contains(&hit.u));
            assert!((0.0..=1.0).contains(&hit.v));
            let ph_ray = [
                origin[0] + hit.t * dir[0],
                origin[1] + hit.t * dir[1],
                origin[2] + hit.t * dir[2],
            ];
            let ph_param = patch.point(hit.u, hit.v);
            let residual = norm(sub(ph_ray, ph_param));
            let scale = 1.0 + norm(ph_param);
            assert!(
                residual < 2e-3 * scale,
                "residual {residual} too large (u={}, v={})",
                hit.u,
                hit.v
            );
            // Normal perpendicular to both parametric tangents.
            let e10 = sub(patch.p10(), patch.p00());
            let f = sub(patch.p11(), patch.p01());
            let e00 = sub(patch.p01(), patch.p00());
            let e11 = sub(patch.p11(), patch.p10());
            let dpdu = add(scale3(e10, 1.0 - hit.v), scale3(f, hit.v));
            let dpdv = add(scale3(e00, 1.0 - hit.u), scale3(e11, hit.u));
            let du = norm(dpdu);
            let dv = norm(dpdv);
            if du > 1e-3 && dv > 1e-3 {
                let cu = dot(hit.normal, dpdu).abs() / du;
                let cv = dot(hit.normal, dpdv).abs() / dv;
                assert!(cu < 5e-3, "normal not ⟂ ∂P/∂u: {cu}");
                assert!(cv < 5e-3, "normal not ⟂ ∂P/∂v: {cv}");
            }
            hits += 1;
        }
        assert!(hits > 2_000, "too few residual samples: {hits}");
    }

    /// Scales a vector by a scalar (test-local helper mirroring `scale`).
    fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = BilinearPatchBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force() {
        let mut rng = Rng::new(0xB11A_7C04);
        let scene = random_scene(&mut rng, 48);
        let bvh = BilinearPatchBvh::build(&scene);

        for _ in 0..4_000 {
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
            if dot(dir, dir) < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            // Brute-force nearest over the reordered leaf primitives.
            let mut brute: Option<BilinearPatchHit> = None;
            let mut r = ray;
            for patch in bvh.patches() {
                if let Some(hit) = patch.intersect(&r) {
                    r = Ray::new(r.origin(), r.direction(), r.t_min(), hit.t);
                    brute = Some(hit);
                }
            }
            let walked = bvh.closest_hit(&ray);
            match (brute, walked) {
                (None, None) => {}
                (Some(c), Some(p)) => {
                    assert_eq!(c.primitive, p.primitive);
                    assert_eq!(c.t.to_bits(), p.t.to_bits(), "t bits differ");
                }
                (c, p) => panic!("hit disagreement: {c:?} vs {p:?}"),
            }
            assert_eq!(brute.is_some(), bvh.any_hit(&ray));
        }
    }
}
