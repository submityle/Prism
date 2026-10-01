//! Analytic axis-aligned ellipsoid primitive and its single-level `BVH`.
//!
//! A sphere is the special case of equal radii; a general ellipsoid lets one
//! procedural primitive stand in for squashed/stretched proxies — capsule caps,
//! blobby metaball shells, lens elements, and bounding proxies — without paying
//! for a tessellated mesh. Like the [`super::sphere::Sphere`] it rides the
//! `DXR`/`Vulkan` *procedural-primitive* path: the `BLAS` stores one
//! axis-aligned box per primitive and an intersection shader refines the hit
//! inside it. This module is the `CPU` golden reference for that shader: an
//! [`Aabb`]-bounded [`Ellipsoid`] with an analytic ray test, plus an
//! [`EllipsoidBvh`] that reuses the shared binned-`SAH` [`build_linear_bvh`]
//! over the per-ellipsoid boxes and the same ordered slab walk the triangle
//! [`super::bvh::Bvh`] uses.
//!
//! The intersection scales the ray into the ellipsoid's unit-sphere frame
//! (dividing origin offset and direction componentwise by the radii), then runs
//! the same numerically stable reduced-quadratic solve as the sphere: it forms
//! the reduced discriminant and picks the root branch by the sign of the linear
//! coefficient to dodge the catastrophic cancellation a naive `(-b ± √disc)/2a`
//! suffers on a grazing ray. Because the direction is scaled rather than assumed
//! unit, the `t²` coefficient carries the full `dot(scaled_dir, scaled_dir)` and
//! the solve stays correct for any (non-degenerate) ray length. The surface
//! normal is the implicit gradient `(P − center) / radii²`, which is *not* unit
//! for an ellipsoid, so it is explicitly normalized. Every step is
//! `sqrt`/`copysign`/add/sub/mul/div and comparisons, so it is bit-reproducible
//! on the `GPU` and free of any transcendental call.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic axis-aligned ellipsoid primitive in world space.
///
/// `primitive` is the caller's stable id (mirroring [`super::bvh::Triangle`]):
/// the [`EllipsoidBvh`] builder reorders ellipsoids internally but always
/// reports hits by this id so downstream shading can look up
/// material/attributes. Each radius is stored non-negative; a caller-supplied
/// negative radius is folded to its magnitude so the derived [`Aabb`] is always
/// well formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipsoid {
    /// Center in world space (`x`, `y`, `z`).
    center: [f32; 3],
    /// Non-negative semi-axis radii along `x`, `y`, `z`.
    radii: [f32; 3],
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Ellipsoid {
    /// Builds an ellipsoid at `center` with per-axis `radii` (each folded to its
    /// magnitude) and stable id `primitive`.
    #[must_use]
    pub fn new(center: [f32; 3], radii: [f32; 3], primitive: u32) -> Self {
        Self {
            center,
            radii: [radii[0].abs(), radii[1].abs(), radii[2].abs()],
            primitive,
        }
    }

    /// Center in world space.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Non-negative semi-axis radii along `x`, `y`, `z`.
    #[must_use]
    pub fn radii(&self) -> [f32; 3] {
        self.radii
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds `center ± radii`.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores per
    /// ellipsoid; a zero radius on any axis collapses that axis to the center.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        Aabb::new(
            [
                self.center[0] - self.radii[0],
                self.center[1] - self.radii[1],
                self.center[2] - self.radii[2],
            ],
            [
                self.center[0] + self.radii[0],
                self.center[1] + self.radii[1],
                self.center[2] + self.radii[2],
            ],
        )
    }

    /// Nearest ray/ellipsoid intersection inside `ray`'s `[t_min, t_max]`
    /// interval, or `None` when the ray misses or only grazes outside it.
    ///
    /// The reported [`EllipsoidHit::normal`] is the unit surface normal oriented
    /// *against* the incident ray, and [`EllipsoidHit::front_face`] is `true`
    /// when the ray struck the outward-facing side (so a back-face hit — the ray
    /// starting inside the ellipsoid — reports `front_face == false` with a
    /// flipped normal). A degenerate ellipsoid (any zero radius) or a
    /// zero-length ray direction never reports a hit.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<EllipsoidHit> {
        let r = self.radii;
        if r[0] <= 0.0 || r[1] <= 0.0 || r[2] <= 0.0 {
            return None;
        }
        let origin = ray.origin();
        let direction = ray.direction();
        // Scale the ray into the unit-sphere frame: `so` is the center-relative
        // origin and `sd` the direction, each divided componentwise by `radii`.
        let so = [
            (origin[0] - self.center[0]) / r[0],
            (origin[1] - self.center[1]) / r[1],
            (origin[2] - self.center[2]) / r[2],
        ];
        let sd = [direction[0] / r[0], direction[1] / r[1], direction[2] / r[2]];
        // Reduced quadratic `a·t² + 2·half_b·t + c_term = 0`; `a` carries the
        // full scaled-direction length so unit direction is never assumed.
        let a = dot(sd, sd);
        if a <= 0.0 {
            // Zero-length direction: no meaningful parametric surface.
            return None;
        }
        let half_b = dot(so, sd);
        let c_term = dot(so, so) - 1.0;
        let disc = half_b * half_b - a * c_term;
        if disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        // Numerically stable branch (identical to the sphere): `copysign` keeps
        // the numerator away from the cancellation `(-half_b + sqrt_disc)` would
        // suffer for a grazing ray. The product form recovers the far root.
        let k = -(half_b + sqrt_disc.copysign(half_b));
        let (t_near, t_far) = if k != 0.0 {
            let r0 = k / a;
            let r1 = c_term / k;
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
        // Implicit gradient `∇F = ((P − c) / radii²)`, normalized: an ellipsoid
        // normal is not unit before scaling, so divide by its length (which is
        // strictly positive for any point on the surface).
        let grad = [
            (point[0] - self.center[0]) / (r[0] * r[0]),
            (point[1] - self.center[1]) / (r[1] * r[1]),
            (point[2] - self.center[2]) / (r[2] * r[2]),
        ];
        let inv_len = 1.0 / dot(grad, grad).sqrt();
        let outward = [grad[0] * inv_len, grad[1] * inv_len, grad[2] * inv_len];
        let front_face = dot(direction, outward) < 0.0;
        let normal = if front_face {
            outward
        } else {
            [-outward[0], -outward[1], -outward[2]]
        };
        Some(EllipsoidHit {
            t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/ellipsoid intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the ellipsoid that was hit.
    pub primitive: u32,
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the outward-facing side was struck; `false` for a back face
    /// (ray originating inside the ellipsoid), whose `normal` is flipped inward.
    pub front_face: bool,
}

/// A single-level `BVH` over analytic [`Ellipsoid`] primitives.
///
/// Empty input yields an empty hierarchy ([`EllipsoidBvh::is_empty`]);
/// traversal of an empty hierarchy simply never reports a hit. The layout and
/// ordered slab walk mirror the triangle [`super::bvh::Bvh`] so the two
/// primitive kinds share one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct EllipsoidBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Ellipsoids reordered so each leaf owns a contiguous slice.
    ellipsoids: Vec<Ellipsoid>,
}

impl EllipsoidBvh {
    /// Builds a `BVH` over `ellipsoids` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(ellipsoids: &[Ellipsoid]) -> Self {
        Self::build_with(ellipsoids, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `ellipsoids` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each ellipsoid's [`Ellipsoid::aabb`] and then
    /// reorders the ellipsoids by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`EllipsoidBvh::ellipsoids`].
    #[must_use]
    pub fn build_with(ellipsoids: &[Ellipsoid], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = ellipsoids.iter().map(Ellipsoid::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let ellipsoids = order.iter().map(|&i| ellipsoids[i as usize]).collect();
        Self { nodes, ellipsoids }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of ellipsoids in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.ellipsoids.len()
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

    /// The reordered ellipsoid array (leaf slices index into this).
    #[must_use]
    pub fn ellipsoids(&self) -> &[Ellipsoid] {
        &self.ellipsoids
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<EllipsoidHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<EllipsoidHit> = None;

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
                    for ellipsoid in &self.ellipsoids[start..end] {
                        if let Some(hit) = ellipsoid.intersect(&ray) {
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

    /// True when *any* ellipsoid intersects `ray` inside its interval.
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
                    for ellipsoid in &self.ellipsoids[start..end] {
                        if ellipsoid.intersect(ray).is_some() {
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

    /// Brute-force nearest hit over the *original* (unordered) ellipsoid list,
    /// used as the ground truth the `BVH` must reproduce.
    fn brute_closest(ellipsoids: &[Ellipsoid], ray: &Ray) -> Option<EllipsoidHit> {
        let mut best: Option<EllipsoidHit> = None;
        let mut ray = *ray;
        for ellipsoid in ellipsoids {
            if let Some(hit) = ellipsoid.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn axis_aligned_hit_has_exact_t_and_outward_normal() {
        // Ellipsoid with radii (2, 1, 3) centered at origin; a +x ray from the
        // far −x side hits the near surface at x = −2 (t = 8). The outward
        // normal there points −x and faces the ray (front face).
        let e = Ellipsoid::new([0.0, 0.0, 0.0], [2.0, 1.0, 3.0], 7);
        let ray = Ray::infinite([-10.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let hit = e.intersect(&ray).expect("axis ray must hit");
        assert_eq!(hit.primitive, 7);
        assert!(approx(hit.t, 8.0, 1e-5), "t = {}", hit.t);
        assert!(hit.front_face);
        assert!(approx(hit.normal[0], -1.0, 1e-5));
        assert!(approx(hit.normal[1], 0.0, 1e-5));
        assert!(approx(hit.normal[2], 0.0, 1e-5));
    }

    #[test]
    fn ray_from_inside_reports_back_face_with_flipped_normal() {
        let e = Ellipsoid::new([0.0, 0.0, 0.0], [2.0, 1.0, 3.0], 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let hit = e.intersect(&ray).expect("interior ray must exit the surface");
        assert!(approx(hit.t, 2.0, 1e-5), "t = {}", hit.t);
        assert!(!hit.front_face);
        // Outward is +x at the exit point, so the ray-facing normal is −x.
        assert!(approx(hit.normal[0], -1.0, 1e-5));
    }

    #[test]
    fn degenerate_radius_and_zero_direction_never_hit() {
        let flat = Ellipsoid::new([0.0, 0.0, 0.0], [2.0, 0.0, 3.0], 0);
        let ray = Ray::infinite([0.0, 10.0, 0.0], [0.0, -1.0, 0.0]);
        assert!(flat.intersect(&ray).is_none());

        let e = Ellipsoid::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let zero = Ray::infinite([5.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(e.intersect(&zero).is_none());
    }

    #[test]
    fn t_max_excludes_a_far_hit() {
        let e = Ellipsoid::new([0.0, 0.0, 0.0], [2.0, 1.0, 3.0], 0);
        // Near hit at t = 8 is excluded by a t_max short of it.
        let ray = Ray::new([-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 7.0);
        assert!(e.intersect(&ray).is_none());
        let ray_ok = Ray::new([-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 9.0);
        assert!(e.intersect(&ray_ok).is_some());
    }

    #[test]
    fn reported_hit_lies_on_the_implicit_surface() {
        // Independent residual check: the implicit equation
        // Σ ((P_i − c_i) / r_i)² = 1 must hold at every reported hit, with a
        // ray-facing unit normal perpendicular to the local tangent plane. The
        // oracle is the implicit equation itself, never `intersect`.
        let mut rng = Rng::new(0x0E11_9501);
        let mut hits = 0u32;
        for _ in 0..6_000 {
            let center = [
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
                rng.range(-4.0, 4.0),
            ];
            let radii = [
                rng.range(0.3, 3.0),
                rng.range(0.3, 3.0),
                rng.range(0.3, 3.0),
            ];
            let e = Ellipsoid::new(center, radii, 0);

            // Aim a ray from a random exterior point toward a jittered point
            // near the center so a large fraction of rays strike the surface.
            let origin = [
                center[0] + rng.range(-12.0, 12.0),
                center[1] + rng.range(-12.0, 12.0),
                center[2] + rng.range(-12.0, 12.0),
            ];
            let target = [
                center[0] + rng.range(-1.0, 1.0) * radii[0],
                center[1] + rng.range(-1.0, 1.0) * radii[1],
                center[2] + rng.range(-1.0, 1.0) * radii[2],
            ];
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let Some(hit) = e.intersect(&ray) else {
                continue;
            };
            hits += 1;

            let p = ray.at(hit.t);
            let nx = (p[0] - center[0]) / radii[0];
            let ny = (p[1] - center[1]) / radii[1];
            let nz = (p[2] - center[2]) / radii[2];
            let f = nx * nx + ny * ny + nz * nz;
            assert!((f - 1.0).abs() < 2e-3, "residual {} off the surface", f - 1.0);

            // Unit normal.
            let nlen = (hit.normal[0] * hit.normal[0]
                + hit.normal[1] * hit.normal[1]
                + hit.normal[2] * hit.normal[2])
                .sqrt();
            assert!(approx(nlen, 1.0, 1e-3), "normal not unit: {nlen}");

            // Normal faces the ray.
            let facing = hit.normal[0] * dir[0] + hit.normal[1] * dir[1] + hit.normal[2] * dir[2];
            assert!(facing <= 1e-4, "normal not oriented against the ray: {facing}");
        }
        assert!(hits > 2_000, "too few surface hits accumulated: {hits}");
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = EllipsoidBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.bounds(), Aabb::empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force() {
        let mut rng = Rng::new(0xA17E_B00C);
        let ellipsoids: Vec<Ellipsoid> = (0..48)
            .map(|i| {
                let center = [
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                ];
                let radii = [
                    rng.range(0.3, 1.4),
                    rng.range(0.3, 1.4),
                    rng.range(0.3, 1.4),
                ];
                Ellipsoid::new(center, radii, i)
            })
            .collect();
        let bvh = EllipsoidBvh::build(&ellipsoids);
        assert_eq!(bvh.primitive_count(), ellipsoids.len());

        for _ in 0..3_000 {
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
            let expected = brute_closest(&ellipsoids, &ray);
            let actual = bvh.closest_hit(&ray);
            match (expected, actual) {
                (None, None) => {}
                (Some(e), Some(a)) => {
                    assert_eq!(e.primitive, a.primitive);
                    assert_eq!(e.t.to_bits(), a.t.to_bits());
                    assert_eq!(e.front_face, a.front_face);
                }
                (e, a) => panic!("hit disagreement: {e:?} vs {a:?}"),
            }
            assert_eq!(bvh.any_hit(&ray), brute_closest(&ellipsoids, &ray).is_some());
        }
    }
}
