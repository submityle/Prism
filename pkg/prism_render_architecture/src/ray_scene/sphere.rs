//! Analytic sphere primitive and its single-level `BVH`.
//!
//! Triangle soups cover most rasterizable geometry, but a real path tracer also
//! wants *procedural* primitives that are defined by a closed-form surface
//! rather than tessellated meshes: particles, area/portal spheres, curve caps,
//! and debug proxies. On hardware ray tracing these ride the DXR/Vulkan
//! *procedural-primitive* path, where the `BLAS` stores an axis-aligned bounding
//! box per primitive and an *intersection shader* refines the hit inside that
//! box. This module is the `CPU` golden reference for exactly that path: an
//! [`Aabb`]-bounded [`Sphere`] with an analytic ray test the intersection shader
//! mirrors, plus a [`SphereBvh`] that reuses the shared binned-`SAH`
//! [`build_linear_bvh`] over the per-sphere boxes and the same ordered slab walk
//! the triangle [`super::bvh::Bvh`] uses.
//!
//! The intersection is the numerically stable quadratic solve (Numerical
//! Recipes / `pbrt` form): it forms the reduced discriminant and picks the root
//! branch by the sign of the linear coefficient to avoid the catastrophic
//! cancellation a naive `(-b ± √disc)/2a` suffers when the ray grazes the
//! sphere. Every step is `sqrt`/`copysign`/add/sub/mul/div and comparisons, so
//! it is bit-reproducible on the `GPU` and free of any transcendental call.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic sphere primitive in world space.
///
/// `primitive` is the caller's stable id (mirroring [`super::bvh::Triangle`]):
/// the [`SphereBvh`] builder reorders spheres internally but always reports hits
/// by this id so downstream shading can look up material/attributes. The radius
/// is stored non-negative; a caller-supplied negative radius is folded to its
/// magnitude so the derived [`Aabb`] is always well formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// Center in world space (`x`, `y`, `z`).
    center: [f32; 3],
    /// Non-negative radius.
    radius: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Sphere {
    /// Builds a sphere at `center` with `radius` (folded to its magnitude) and
    /// stable id `primitive`.
    #[must_use]
    pub fn new(center: [f32; 3], radius: f32, primitive: u32) -> Self {
        Self {
            center,
            radius: radius.abs(),
            primitive,
        }
    }

    /// Center in world space.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Non-negative radius.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds `center ± radius`.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores per
    /// sphere; a zero-radius sphere degenerates to the single center point.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let r = self.radius;
        Aabb::new(
            [self.center[0] - r, self.center[1] - r, self.center[2] - r],
            [self.center[0] + r, self.center[1] + r, self.center[2] + r],
        )
    }

    /// Nearest ray/sphere intersection inside `ray`'s `[t_min, t_max]` interval,
    /// or `None` when the ray misses or only grazes outside the interval.
    ///
    /// The reported [`SphereHit::normal`] is the unit surface normal oriented
    /// *against* the incident ray, and [`SphereHit::front_face`] is `true` when
    /// the ray struck the outward-facing side (so a back-face hit — the ray
    /// starting inside the sphere — reports `front_face == false` with a flipped
    /// normal). A degenerate zero-radius sphere or a zero-length ray direction
    /// never reports a hit.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<SphereHit> {
        if self.radius <= 0.0 {
            return None;
        }
        let origin = ray.origin();
        let direction = ray.direction();
        // Vector from the sphere center to the ray origin.
        let oc = [
            origin[0] - self.center[0],
            origin[1] - self.center[1],
            origin[2] - self.center[2],
        ];
        let a = dot(direction, direction);
        if a <= 0.0 {
            // Zero-length direction: no meaningful parametric surface.
            return None;
        }
        // Reduced quadratic `a·t² + 2·half_b·t + c_term = 0`.
        let half_b = dot(oc, direction);
        let c_term = dot(oc, oc) - self.radius * self.radius;
        let disc = half_b * half_b - a * c_term;
        if disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        // Numerically stable branch: `copysign` keeps `q` away from the
        // cancellation that `(-half_b + sqrt_disc)` would suffer for a grazing
        // ray. `q` is zero only when `half_b == 0` *and* `disc == 0`, i.e. a
        // tangent through the origin, which we route to the double root below.
        let q = -(half_b + sqrt_disc.copysign(half_b));
        let (t_near, t_far) = if q != 0.0 {
            let r0 = q / a;
            let r1 = c_term / q;
            if r0 <= r1 {
                (r0, r1)
            } else {
                (r1, r0)
            }
        } else {
            let r = -half_b / a;
            (r, r)
        };

        let t_min = ray.t_min();
        let t_max = ray.t_max();
        let range = t_min..=t_max;
        let t = if range.contains(&t_near) {
            t_near
        } else if range.contains(&t_far) {
            t_far
        } else {
            return None;
        };

        let point = ray.at(t);
        // Analytic outward normal: exactly unit for a point on the surface, and
        // formed with only sub/mul/div so it matches the `GPU` intersection
        // shader bit-for-bit without a renormalization step.
        let inv_r = 1.0 / self.radius;
        let outward = [
            (point[0] - self.center[0]) * inv_r,
            (point[1] - self.center[1]) * inv_r,
            (point[2] - self.center[2]) * inv_r,
        ];
        let front_face = dot(direction, outward) < 0.0;
        let normal = if front_face {
            outward
        } else {
            [-outward[0], -outward[1], -outward[2]]
        };
        Some(SphereHit {
            t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/sphere intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the sphere that was hit.
    pub primitive: u32,
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the outward-facing side was struck; `false` for a back face
    /// (ray originating inside the sphere), whose `normal` is flipped inward.
    pub front_face: bool,
}

/// A single-level `BVH` over analytic [`Sphere`] primitives.
///
/// Empty input yields an empty hierarchy ([`SphereBvh::is_empty`]); traversal of
/// an empty hierarchy simply never reports a hit. The layout and ordered slab
/// walk mirror the triangle [`super::bvh::Bvh`] so the two primitive kinds share
/// one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Spheres reordered so each leaf owns a contiguous slice.
    spheres: Vec<Sphere>,
}

impl SphereBvh {
    /// Builds a `BVH` over `spheres` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(spheres: &[Sphere]) -> Self {
        Self::build_with(spheres, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `spheres` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each sphere's [`Sphere::aabb`] and then reorders
    /// the spheres by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`SphereBvh::spheres`].
    #[must_use]
    pub fn build_with(spheres: &[Sphere], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = spheres.iter().map(Sphere::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let spheres = order.iter().map(|&i| spheres[i as usize]).collect();
        Self { nodes, spheres }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of spheres in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.spheres.len()
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

    /// The reordered sphere array (leaf slices index into this).
    #[must_use]
    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<SphereHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<SphereHit> = None;

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
                    for sphere in &self.spheres[start..end] {
                        if let Some(hit) = sphere.intersect(&ray) {
                            // Tighten the interval so far subtrees are pruned.
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

    /// True when *any* sphere intersects `ray` inside its interval.
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
                    for sphere in &self.spheres[start..end] {
                        if sphere.intersect(ray).is_some() {
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

/// Dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    /// Brute-force nearest hit over the *original* (unordered) sphere list, used
    /// as the ground truth the `BVH` must reproduce.
    fn brute_closest(spheres: &[Sphere], ray: &Ray) -> Option<SphereHit> {
        let mut best: Option<SphereHit> = None;
        let mut ray = *ray;
        for sphere in spheres {
            if let Some(hit) = sphere.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn analytic_front_face_hit_has_exact_t_and_outward_normal() {
        let sphere = Sphere::new([0.0, 0.0, -5.0], 1.0, 7);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let hit = sphere.intersect(&ray).expect("ray must hit the sphere");
        assert!(approx(hit.t, 4.0, 1e-6), "t = {}", hit.t);
        assert_eq!(hit.primitive, 7);
        assert!(hit.front_face);
        assert!(approx(hit.normal[0], 0.0, 1e-6));
        assert!(approx(hit.normal[1], 0.0, 1e-6));
        assert!(approx(hit.normal[2], 1.0, 1e-6));
    }

    #[test]
    fn tangent_ray_reports_a_single_grazing_hit() {
        // Center offset by exactly the radius along y: the ray grazes at z=-5.
        let sphere = Sphere::new([0.0, 1.0, -5.0], 1.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let hit = sphere.intersect(&ray).expect("tangent ray must still hit");
        assert!(approx(hit.t, 5.0, 1e-5), "t = {}", hit.t);
        // Grazing contact point sits on the -y side of the sphere.
        assert!(approx(hit.normal[1].abs(), 1.0, 1e-4));
    }

    #[test]
    fn ray_from_inside_reports_back_face_with_inward_normal() {
        let sphere = Sphere::new([0.0, 0.0, 0.0], 2.0, 3);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let hit = sphere.intersect(&ray).expect("origin inside must exit-hit");
        assert!(approx(hit.t, 2.0, 1e-6), "t = {}", hit.t);
        assert!(!hit.front_face);
        // Exit point is at z=-2; outward normal is -z, flipped to +z against ray.
        assert!(approx(hit.normal[2], 1.0, 1e-6));
    }

    #[test]
    fn sphere_behind_the_origin_is_missed() {
        let sphere = Sphere::new([0.0, 0.0, 5.0], 1.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(sphere.intersect(&ray).is_none());
    }

    #[test]
    fn interval_bounds_reject_hits_outside_the_range() {
        let sphere = Sphere::new([0.0, 0.0, -5.0], 1.0, 0);
        // The near hit is at t=4; a window past it only sees the far root at t=6.
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 4.5, 100.0);
        let hit = sphere.intersect(&ray).expect("far root should be found");
        assert!(approx(hit.t, 6.0, 1e-6), "t = {}", hit.t);
        // A window before both roots misses entirely.
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0, 3.0);
        assert!(sphere.intersect(&ray).is_none());
    }

    #[test]
    fn aabb_is_center_plus_or_minus_radius() {
        let sphere = Sphere::new([1.0, -2.0, 3.0], 2.0, 0);
        let aabb = sphere.aabb();
        assert_eq!(aabb.min, [-1.0, -4.0, 1.0]);
        assert_eq!(aabb.max, [3.0, 0.0, 5.0]);
    }

    #[test]
    fn negative_radius_is_folded_to_magnitude() {
        let sphere = Sphere::new([0.0, 0.0, 0.0], -2.5, 0);
        assert!(approx(sphere.radius(), 2.5, 0.0));
    }

    #[test]
    fn unnormalized_direction_scales_t_by_direction_length() {
        let sphere = Sphere::new([0.0, 0.0, -5.0], 1.0, 0);
        // Direction length 2 halves the parametric distance to the same point.
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -2.0]);
        let hit = sphere.intersect(&ray).expect("must hit");
        assert!(approx(hit.t, 2.0, 1e-6), "t = {}", hit.t);
        // The normal is still unit despite the unnormalized direction.
        let len = dot(hit.normal, hit.normal).sqrt();
        assert!(approx(len, 1.0, 1e-5), "|n| = {len}");
    }

    #[test]
    fn zero_radius_sphere_is_never_hit() {
        let sphere = Sphere::new([0.0, 0.0, -5.0], 0.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(sphere.intersect(&ray).is_none());
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = SphereBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        assert!(bvh.bounds().is_empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_returns_nearest_of_several_spheres() {
        let spheres = vec![
            Sphere::new([0.0, 0.0, -9.0], 1.0, 20),
            Sphere::new([0.0, 0.0, -5.0], 1.0, 10),
            Sphere::new([0.0, 0.0, -7.0], 1.0, 30),
        ];
        let bvh = SphereBvh::build(&spheres);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let hit = bvh.closest_hit(&ray).expect("must hit the nearest sphere");
        assert_eq!(hit.primitive, 10);
        assert!(approx(hit.t, 4.0, 1e-6), "t = {}", hit.t);
    }

    #[test]
    fn bvh_matches_brute_force_over_a_random_scene() {
        let mut rng = Rng::new(0xC0FF_EE42);
        let spheres: Vec<Sphere> = (0..64)
            .map(|i| {
                let center = [
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                ];
                Sphere::new(center, rng.range(0.2, 1.2), i)
            })
            .collect();
        let bvh = SphereBvh::build(&spheres);

        for _ in 0..2_000 {
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
            if dot(dir, dir) < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            let expected = brute_closest(&spheres, &ray);
            let actual = bvh.closest_hit(&ray);
            match (expected, actual) {
                (None, None) => {}
                (Some(e), Some(a)) => {
                    assert_eq!(e.primitive, a.primitive, "primitive mismatch");
                    assert!(approx(e.t, a.t, 1e-4), "t {} vs {}", e.t, a.t);
                }
                (e, a) => panic!("hit disagreement: {e:?} vs {a:?}"),
            }
            assert_eq!(
                bvh.any_hit(&ray),
                expected.is_some(),
                "any_hit disagrees with closest_hit"
            );
        }
    }

    #[test]
    fn bvh_build_is_bit_for_bit_deterministic() {
        let mut rng = Rng::new(0x1234_5678);
        let spheres: Vec<Sphere> = (0..48)
            .map(|i| {
                let center = [
                    rng.range(-5.0, 5.0),
                    rng.range(-5.0, 5.0),
                    rng.range(-5.0, 5.0),
                ];
                Sphere::new(center, rng.range(0.3, 1.0), i)
            })
            .collect();
        let a = SphereBvh::build(&spheres);
        let b = SphereBvh::build(&spheres);
        assert_eq!(a, b);
    }
}
