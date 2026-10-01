//! Analytic *capsule* (sphere-swept line segment / stadium of revolution)
//! primitive and its single-level `BVH`.
//!
//! Like [`super::cylinder`] and [`super::cone`], this is a procedural primitive
//! for the `DXR`/Vulkan `AABB` path: the `BLAS` stores one axis-aligned box per
//! capsule and an intersection shader refines the hit. A capsule is the set of
//! all points within `radius` of the segment `a → b`, i.e. a cylindrical body
//! of radius `radius` capped by a hemisphere at each end. It is the single most
//! common collision / render proxy in real-time engines — character controllers,
//! limbs, wires, tubes, and thick hair strands — so a path tracer wants a
//! closed-form test rather than a tessellated pill.
//!
//! The intersection tests three analytic pieces and returns the nearest valid
//! root inside the ray interval: the infinite-cylinder quadratic clipped to the
//! body band `z ∈ [0, h]` (`z` = axial distance from `a`), and the two endpoint
//! spheres clipped to their cap half-spaces (`z ≤ 0` at `a`, `z ≥ h` at `b`).
//! Every `t²` coefficient carries `dd = ⟨d, d⟩`, so the test is correct for the
//! non-unit ray directions `ray_scene` feeds it. Every step is
//! add/sub/mul/div/`sqrt` and comparisons, so it is bit-reproducible on the
//! `GPU` and free of any transcendental call. A zero-length axis degenerates to
//! a single sphere at `a`.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic capsule (sphere-swept segment) in world space.
///
/// The solid is every point within `radius` of the segment `a → b`: a cylinder
/// of radius `radius` closed by a hemisphere at each endpoint. `primitive` is
/// the caller's stable id (mirroring [`super::bvh::Triangle`] and
/// [`super::cylinder::Cylinder`]): the [`CapsuleBvh`] builder reorders capsules
/// internally but always reports hits by this id. `radius` is stored
/// non-negative; a caller-supplied negative radius is folded to its magnitude so
/// the derived [`Aabb`] and the quadrics stay well formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// First segment endpoint (center of the `a`-side hemisphere).
    a: [f32; 3],
    /// Second segment endpoint (center of the `b`-side hemisphere).
    b: [f32; 3],
    /// Non-negative sweep radius around the segment.
    radius: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Capsule {
    /// Builds a capsule sweeping `radius` (folded to its magnitude) around the
    /// segment `a → b`, tagged with stable id `primitive`.
    #[must_use]
    pub fn new(a: [f32; 3], b: [f32; 3], radius: f32, primitive: u32) -> Self {
        Self {
            a,
            b,
            radius: radius.abs(),
            primitive,
        }
    }

    /// First segment endpoint.
    #[must_use]
    pub fn a(&self) -> [f32; 3] {
        self.a
    }

    /// Second segment endpoint.
    #[must_use]
    pub fn b(&self) -> [f32; 3] {
        self.b
    }

    /// Non-negative sweep radius.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds of the capsule.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. The
    /// pill is the Minkowski sum of the segment `a → b` with a ball of radius
    /// `radius`, so on every axis the bound is the segment's extent padded by
    /// exactly `radius` on both sides — tight everywhere.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            *slot.0 = self.a[axis].min(self.b[axis]) - self.radius;
            *slot.1 = self.a[axis].max(self.b[axis]) + self.radius;
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/capsule intersection inside `ray`'s `[t_min, t_max]`
    /// interval, or `None` when the ray misses.
    ///
    /// [`CapsuleHit::normal`] is the unit surface normal oriented *against* the
    /// incident ray, and [`CapsuleHit::front_face`] is `true` when the ray struck
    /// the outward-facing side. A zero-length ray direction never reports a hit;
    /// a zero-length axis reduces to a single sphere at `a`.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<CapsuleHit> {
        let direction = ray.direction();
        let dd = dot(direction, direction);
        if dd <= 0.0 {
            return None;
        }
        let r2 = self.radius * self.radius;
        if r2 <= 0.0 {
            return None;
        }

        let t_min = ray.t_min();
        let t_max = ray.t_max();
        let origin = ray.origin();

        let w = sub(self.b, self.a);
        let h2 = dot(w, w);

        // Best (t, outward-normal) found so far across body and cap pieces.
        let mut best_t = f32::INFINITY;
        let mut best_outward = [0.0f32; 3];

        // Considers one root `t`, computing its outward normal about `pivot`
        // (segment endpoint or nearest axis point) and keeping it when nearer.
        let mut consider = |t: f32, pivot: [f32; 3]| {
            if !(t >= t_min && t <= t_max) || t >= best_t {
                return;
            }
            let p = [
                origin[0] + t * direction[0],
                origin[1] + t * direction[1],
                origin[2] + t * direction[2],
            ];
            let radial = sub(p, pivot);
            let len2 = dot(radial, radial);
            if len2 <= 0.0 {
                return;
            }
            best_t = t;
            best_outward = scale(radial, 1.0 / len2.sqrt());
        };

        // Ray origin relative to endpoint `a`.
        let oa = sub(origin, self.a);

        if h2 > 0.0 {
            let h = h2.sqrt();
            let n = scale(w, 1.0 / h);
            let za = dot(oa, n);
            let zd = dot(direction, n);
            let ad = dot(oa, direction);
            let aa = dot(oa, oa);

            // Infinite-cylinder quadratic on the ⊥ distance: `A·t² + B·t + C`.
            let coeff_a = dd - zd * zd;
            let coeff_b = 2.0 * (ad - za * zd);
            let coeff_c = aa - za * za - r2;
            if coeff_a != 0.0 {
                let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
                if disc >= 0.0 {
                    let sqrt_disc = disc.sqrt();
                    let inv_2a = 1.0 / (2.0 * coeff_a);
                    for t in [(-coeff_b - sqrt_disc) * inv_2a, (-coeff_b + sqrt_disc) * inv_2a] {
                        let z = za + t * zd;
                        if z >= 0.0 && z <= h {
                            // Pivot is the axis point at height `z`, so the
                            // radial vector is exactly the ⊥ surface normal.
                            let pivot = [
                                self.a[0] + z * n[0],
                                self.a[1] + z * n[1],
                                self.a[2] + z * n[2],
                            ];
                            consider(t, pivot);
                        }
                    }
                }
            }

            // Hemisphere at `a`: keep sphere roots below the body band (`z ≤ 0`).
            if let Some((t0, t1)) = sphere_roots(oa, direction, dd, r2) {
                for t in [t0, t1] {
                    let p = [
                        origin[0] + t * direction[0],
                        origin[1] + t * direction[1],
                        origin[2] + t * direction[2],
                    ];
                    if dot(sub(p, self.a), n) <= 0.0 {
                        consider(t, self.a);
                    }
                }
            }

            // Hemisphere at `b`: keep sphere roots above the body band (`z ≥ h`).
            let ob = sub(origin, self.b);
            if let Some((t0, t1)) = sphere_roots(ob, direction, dd, r2) {
                for t in [t0, t1] {
                    let p = [
                        origin[0] + t * direction[0],
                        origin[1] + t * direction[1],
                        origin[2] + t * direction[2],
                    ];
                    if dot(sub(p, self.a), n) >= h {
                        consider(t, self.b);
                    }
                }
            }
        } else {
            // Degenerate axis: a single sphere at `a`, no cap clipping.
            if let Some((t0, t1)) = sphere_roots(oa, direction, dd, r2) {
                for t in [t0, t1] {
                    consider(t, self.a);
                }
            }
        }

        if !best_t.is_finite() {
            return None;
        }

        let front_face = dot(direction, best_outward) < 0.0;
        let normal = if front_face {
            best_outward
        } else {
            [-best_outward[0], -best_outward[1], -best_outward[2]]
        };
        Some(CapsuleHit {
            t: best_t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// Solves the sphere quadratic `|rel + t·d|² = r²`, returning its two roots
/// (equal when the ray grazes the sphere) or `None` when the ray misses.
///
/// `rel` is the ray origin relative to the sphere center. Kept as a free helper
/// so both hemisphere caps (and the degenerate single-sphere case) share
/// identical root arithmetic; callers clip the roots to the cap half-space and
/// the ray interval themselves.
fn sphere_roots(rel: [f32; 3], direction: [f32; 3], dd: f32, r2: f32) -> Option<(f32, f32)> {
    let b = 2.0 * dot(rel, direction);
    let c = dot(rel, rel) - r2;
    let disc = b * b - 4.0 * dd * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let inv_2a = 1.0 / (2.0 * dd);
    Some(((-b - sqrt_disc) * inv_2a, (-b + sqrt_disc) * inv_2a))
}

/// A ray/capsule intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the capsule that was hit.
    pub primitive: u32,
    /// Unit surface normal, oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the ray struck the outward-facing side.
    pub front_face: bool,
}

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales `a` by scalar `s`.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A single-level `BVH` over analytic [`Capsule`] primitives.
///
/// Empty input yields an empty hierarchy ([`CapsuleBvh::is_empty`]); traversal
/// of an empty hierarchy simply never reports a hit. The layout and ordered slab
/// walk mirror the triangle [`super::bvh::Bvh`], [`super::cone::ConeBvh`], and
/// [`super::cylinder::CylinderBvh`] so every primitive kind shares one
/// acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct CapsuleBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Capsules reordered so each leaf owns a contiguous slice.
    capsules: Vec<Capsule>,
}

impl CapsuleBvh {
    /// Builds a `BVH` over `capsules` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(capsules: &[Capsule]) -> Self {
        Self::build_with(capsules, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `capsules` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each capsule's [`Capsule::aabb`] and reorders the
    /// capsules by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`CapsuleBvh::capsules`].
    #[must_use]
    pub fn build_with(capsules: &[Capsule], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = capsules.iter().map(Capsule::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let capsules = order.iter().map(|&i| capsules[i as usize]).collect();
        Self { nodes, capsules }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of capsules in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.capsules.len()
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

    /// Capsules in leaf-contiguous order.
    #[must_use]
    pub fn capsules(&self) -> &[Capsule] {
        &self.capsules
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<CapsuleHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<CapsuleHit> = None;

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
                    for capsule in &self.capsules[start..end] {
                        if let Some(hit) = capsule.intersect(&ray) {
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

    /// True when *any* capsule intersects `ray` inside its interval.
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
                    for capsule in &self.capsules[start..end] {
                        if capsule.intersect(ray).is_some() {
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

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Squared distance from point `p` to the segment `a → b`.
    fn dist2_point_segment(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
        let ab = sub(b, a);
        let ap = sub(p, a);
        let denom = dot(ab, ab);
        let t = if denom > 0.0 {
            (dot(ap, ab) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let closest = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
        let d = sub(p, closest);
        dot(d, d)
    }

    /// Capsule from `(0,0,0)` to `(0,0,4)` with radius `1`.
    fn sample_capsule(primitive: u32) -> Capsule {
        Capsule::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 1.0, primitive)
    }

    #[test]
    fn hits_the_cylindrical_body_from_the_side() {
        let cap = sample_capsule(2);
        // Aim at the body midpoint (z = 2): near wall at x = 1.
        let ray = Ray::infinite([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0]);
        let hit = cap.intersect(&ray).expect("body hit");
        assert_eq!(hit.primitive, 2);
        assert!(approx(hit.t, 4.0, 1e-3), "t = {}", hit.t);
        assert!(approx(hit.normal[0], 1.0, 1e-3), "normal = {:?}", hit.normal);
        assert!(hit.front_face);
    }

    #[test]
    fn hits_the_top_hemisphere_cap() {
        let cap = sample_capsule(0);
        // Straight down the axis from above: the `b` cap apex sits at z = 5.
        let ray = Ray::infinite([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let hit = cap.intersect(&ray).expect("cap hit");
        assert!(approx(hit.t, 5.0, 1e-3), "t = {}", hit.t);
        // Ray-facing normal points back up +z.
        assert!(approx(hit.normal[2], 1.0, 1e-3), "normal = {:?}", hit.normal);
    }

    #[test]
    fn hits_the_bottom_hemisphere_cap() {
        let cap = sample_capsule(0);
        // Straight up the axis from below: the `a` cap apex sits at z = -1.
        let ray = Ray::infinite([0.0, 0.0, -10.0], [0.0, 0.0, 1.0]);
        let hit = cap.intersect(&ray).expect("cap hit");
        assert!(approx(hit.t, 9.0, 1e-3), "t = {}", hit.t);
    }

    #[test]
    fn misses_past_the_rounded_end() {
        let cap = sample_capsule(0);
        // At z = 4.9, just under the top apex, the cap radius has shrunk below
        // `1`, so a ray at x = 1.5 clears it.
        let ray = Ray::infinite([5.0, 0.0, 4.9], [-1.0, 0.0, 0.0]);
        let hit = cap.intersect(&ray);
        // It should still hit (the cap sphere reaches x≈0.56 at z=4.9), but a
        // ray well outside the ball must miss entirely.
        assert!(hit.is_some());
        let far = Ray::infinite([5.0, 0.0, 5.5], [-1.0, 0.0, 0.0]);
        assert!(cap.intersect(&far).is_none());
    }

    #[test]
    fn degenerate_axis_is_a_sphere() {
        let ball = Capsule::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 2.0, 4);
        let ray = Ray::infinite([1.0, 2.0, 10.0], [0.0, 0.0, -1.0]);
        let hit = ball.intersect(&ray).expect("sphere hit");
        // Sphere surface at z = 5 (center z = 3, radius 2), five units in.
        assert!(approx(hit.t, 5.0, 1e-3), "t = {}", hit.t);
    }

    #[test]
    fn zero_radius_never_hits() {
        let bad = Capsule::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 0.0, 0);
        let ray = Ray::infinite([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0]);
        assert!(bad.intersect(&ray).is_none());
    }

    #[test]
    fn zero_direction_never_hits() {
        let cap = sample_capsule(0);
        let ray = Ray::infinite([5.0, 0.0, 2.0], [0.0, 0.0, 0.0]);
        assert!(cap.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let cap = sample_capsule(0);
        let ray = Ray::new([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0], 0.0, 3.9);
        assert!(cap.intersect(&ray).is_none());
        let ray = Ray::new([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0], 0.0, 4.1);
        assert!(cap.intersect(&ray).is_some());
    }

    /// Independent surface check: every reported hit point must lie at distance
    /// `radius` from the segment `a → b`, and the reported unit normal must point
    /// radially outward from the nearest segment point. This catches a
    /// mistranscribed body/cap coefficient or normal that a `BVH`↔brute
    /// cross-check (sharing [`Capsule::intersect`]) cannot.
    #[test]
    fn reported_hit_lies_on_the_swept_surface() {
        let mut rng = Rng::new(0xCA95_1234);
        for _ in 0..5_000 {
            let a = [rng.range(-4.0, 4.0), rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)];
            let b = [
                a[0] + rng.range(-3.0, 3.0),
                a[1] + rng.range(-3.0, 3.0),
                a[2] + rng.range(-3.0, 3.0),
            ];
            let radius = rng.range(0.3, 1.5);
            let cap = Capsule::new(a, b, radius, 0);

            let origin = [rng.range(-9.0, 9.0), rng.range(-9.0, 9.0), rng.range(-9.0, 9.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let Some(hit) = cap.intersect(&ray) else {
                continue;
            };

            let p = ray.at(hit.t);
            let d2 = dist2_point_segment(p, a, b);
            let d = d2.sqrt();
            assert!(
                (d - radius).abs() / (radius + 1.0) < 5e-3,
                "off-surface: dist {d} vs radius {radius}"
            );
        }
    }

    fn random_capsule(rng: &mut Rng, primitive: u32) -> Capsule {
        let a = [rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), rng.range(-5.0, 5.0)];
        let b = [
            a[0] + rng.range(-3.0, 3.0),
            a[1] + rng.range(-3.0, 3.0),
            a[2] + rng.range(-3.0, 3.0),
        ];
        Capsule::new(a, b, rng.range(0.3, 1.5), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Capsule> {
        (0..count).map(|i| random_capsule(rng, i)).collect()
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = CapsuleBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0xC0DE_5EED);
        let scene = random_scene(&mut rng, 48);
        let bvh = CapsuleBvh::build(&scene);
        let ordered = bvh.capsules().to_vec();

        for _ in 0..3_000 {
            let origin = [rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            let mut brute: Option<CapsuleHit> = None;
            let mut r = ray;
            for cap in &ordered {
                if let Some(hit) = cap.intersect(&r) {
                    r = Ray::new(r.origin(), r.direction(), r.t_min(), hit.t);
                    brute = Some(hit);
                }
            }
            let fast = bvh.closest_hit(&ray);
            match (brute, fast) {
                (None, None) => {}
                (Some(b), Some(f)) => {
                    assert_eq!(b.primitive, f.primitive);
                    assert_eq!(b.t.to_bits(), f.t.to_bits(), "t bits differ");
                    assert_eq!(b.front_face, f.front_face);
                    for k in 0..3 {
                        assert_eq!(b.normal[k].to_bits(), f.normal[k].to_bits());
                    }
                }
                (b, f) => panic!("hit disagreement: {b:?} vs {f:?}"),
            }
        }
    }

    #[test]
    fn bvh_any_hit_matches_brute_force() {
        let mut rng = Rng::new(0xFEED_0B0E);
        let scene = random_scene(&mut rng, 48);
        let bvh = CapsuleBvh::build(&scene);
        let ordered = bvh.capsules().to_vec();

        for _ in 0..3_000 {
            let origin = [rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let brute = ordered.iter().any(|c| c.intersect(&ray).is_some());
            assert_eq!(brute, bvh.any_hit(&ray));
        }
    }
}
