//! Analytic finite *capped* cylinder primitive and its single-level `BVH`.
//!
//! Like [`super::sphere`] and [`super::aabb_primitive`], this is a procedural
//! primitive for the `DXR`/Vulkan `AABB` path: the `BLAS` stores one
//! axis-aligned box per cylinder and an intersection shader refines the hit.
//! Finite cylinders are the natural proxy for tube / capsule area lights, wires,
//! pipes, and hair strand cores, so a path tracer wants a closed-form test
//! rather than a tessellated tube.
//!
//! A [`Cylinder`] is the solid swept by a disk of `radius` moving along the
//! segment from `base` to `top`, closed by two disk end caps. The intersection
//! forms the lateral-surface quadratic in the classic axis-relative reduced form
//! (à la Inigo Quilez / `pbrt`), solves it with the same numerically stable
//! `copysign` branch [`super::sphere::Sphere::intersect`] uses, and separately
//! tests the two cap planes; it then returns the nearest of all valid roots
//! inside the ray interval. Every step is add/sub/mul/div/`sqrt`/`copysign` and
//! comparisons, so it is bit-reproducible on the `GPU` and free of any
//! transcendental call.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic finite capped cylinder in world space.
///
/// The solid is the disk of `radius` swept along the axis segment `base → top`,
/// closed by two disk caps. `primitive` is the caller's stable id (mirroring
/// [`super::bvh::Triangle`] and [`super::sphere::Sphere`]): the [`CylinderBvh`]
/// builder reorders cylinders internally but always reports hits by this id.
/// The radius is stored non-negative; a caller-supplied negative radius is
/// folded to its magnitude so the derived [`Aabb`] stays well formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
    /// Center of the base cap (axis start).
    base: [f32; 3],
    /// Center of the top cap (axis end).
    top: [f32; 3],
    /// Non-negative radius.
    radius: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Cylinder {
    /// Builds a cylinder spanning `base → top` with `radius` (folded to its
    /// magnitude) and stable id `primitive`.
    #[must_use]
    pub fn new(base: [f32; 3], top: [f32; 3], radius: f32, primitive: u32) -> Self {
        Self {
            base,
            top,
            radius: radius.abs(),
            primitive,
        }
    }

    /// Center of the base cap (axis start).
    #[must_use]
    pub fn base(&self) -> [f32; 3] {
        self.base
    }

    /// Center of the top cap (axis end).
    #[must_use]
    pub fn top(&self) -> [f32; 3] {
        self.top
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

    /// Tight axis-aligned bounds of the capped cylinder.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. The
    /// per-axis cap-disk half extent is `radius · √(1 − axisᵢ² / |axis|²)`, the
    /// exact projected radius of a disk whose normal is the (normalized) axis,
    /// so the box hugs the solid rather than using the loose `± radius` cube. A
    /// degenerate zero-length axis falls back to that loose cube.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let ba = sub(self.top, self.base);
        let baba = dot(ba, ba);
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            let lo = self.base[axis].min(self.top[axis]);
            let hi = self.base[axis].max(self.top[axis]);
            let e = if baba > 0.0 {
                // Projected radius of the cap disk onto world axis `axis`.
                let frac = 1.0 - (ba[axis] * ba[axis]) / baba;
                // Clamp guards tiny negative round-off before the sqrt.
                self.radius * frac.max(0.0).sqrt()
            } else {
                self.radius
            };
            *slot.0 = lo - e;
            *slot.1 = hi + e;
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/cylinder intersection inside `ray`'s `[t_min, t_max]`
    /// interval, or `None` when the ray misses.
    ///
    /// Considers both lateral-surface roots and both cap planes and returns the
    /// nearest valid hit. [`CylinderHit::normal`] is the unit surface normal
    /// oriented *against* the incident ray, and [`CylinderHit::front_face`] is
    /// `true` when the outward-facing side was struck (a ray originating inside
    /// the solid reports `front_face == false` with an inward-flipped normal). A
    /// zero-radius cylinder, a zero-length axis, or a zero-length ray direction
    /// never reports a hit.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<CylinderHit> {
        if self.radius <= 0.0 {
            return None;
        }
        let ba = sub(self.top, self.base);
        let baba = dot(ba, ba);
        if baba <= 0.0 {
            return None;
        }
        let direction = ray.direction();
        let dd = dot(direction, direction);
        if dd <= 0.0 {
            return None;
        }
        let oc = sub(ray.origin(), self.base);

        let bard = dot(ba, direction);
        let baoc = dot(ba, oc);
        // Reduced lateral quadratic `k2·t² + 2·k1·t + k0 = 0` (Quilez form):
        // the axis-parallel component is projected out so `k2` is the squared
        // ray speed perpendicular to the axis.
        let k2 = baba * dd - bard * bard;
        let k1 = baba * dot(oc, direction) - baoc * bard;
        let k0 = baba * dot(oc, oc) - baoc * baoc - self.radius * self.radius * baba;

        let t_min = ray.t_min();
        let t_max = ray.t_max();
        let mut best_t = f32::INFINITY;
        let mut best_outward = [0.0f32; 3];
        let inv_r = 1.0 / self.radius;

        // Lateral surface: skip when the ray runs parallel to the axis (`k2` is
        // zero) so only the caps can be hit.
        if k2 > 0.0 {
            let disc = k1 * k1 - k2 * k0;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                // Stable roots via `copysign`, matching `Sphere::intersect`.
                let q = -(k1 + sqrt_disc.copysign(k1));
                let (r0, r1) = if q != 0.0 {
                    (q / k2, k0 / q)
                } else {
                    let r = -k1 / k2;
                    (r, r)
                };
                for t in [r0.min(r1), r0.max(r1)] {
                    if t < t_min || t > t_max || t >= best_t {
                        continue;
                    }
                    // Axis coordinate of the hit, in `[0, baba]` on the body.
                    let y = baoc + t * bard;
                    if y < 0.0 || y > baba {
                        continue;
                    }
                    // Perpendicular offset = full offset minus axial component.
                    let s = y / baba;
                    let point_rel = [
                        oc[0] + t * direction[0] - ba[0] * s,
                        oc[1] + t * direction[1] - ba[1] * s,
                        oc[2] + t * direction[2] - ba[2] * s,
                    ];
                    best_t = t;
                    best_outward = scale(point_rel, inv_r);
                }
            }
        }

        // Caps: the plane `y = 0` (base, outward `-axiŝ`) and `y = baba`
        // (top, outward `+axiŝ`). `bard == 0` means the ray is parallel to
        // both cap planes and cannot cross them.
        if bard != 0.0 {
            let inv_bard = 1.0 / bard;
            let inv_sqrt_baba = 1.0 / baba.sqrt();
            for (y_cap, sign) in [(0.0f32, -1.0f32), (baba, 1.0f32)] {
                let t = (y_cap - baoc) * inv_bard;
                if t < t_min || t > t_max || t >= best_t {
                    continue;
                }
                let s = y_cap / baba;
                let point_rel = [
                    oc[0] + t * direction[0] - ba[0] * s,
                    oc[1] + t * direction[1] - ba[1] * s,
                    oc[2] + t * direction[2] - ba[2] * s,
                ];
                // Inside the cap disk when the perpendicular offset is within r.
                if dot(point_rel, point_rel) > self.radius * self.radius {
                    continue;
                }
                best_t = t;
                best_outward = scale(ba, sign * inv_sqrt_baba);
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
        Some(CylinderHit {
            t: best_t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/cylinder intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the cylinder that was hit.
    pub primitive: u32,
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the outward-facing side was struck; `false` for a back face
    /// (ray originating inside the solid), whose `normal` is flipped inward.
    pub front_face: bool,
}

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scales `a` by `s` componentwise.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A single-level `BVH` over analytic [`Cylinder`] primitives.
///
/// Empty input yields an empty hierarchy ([`CylinderBvh::is_empty`]); traversal
/// of an empty hierarchy simply never reports a hit. The layout and ordered slab
/// walk mirror the triangle [`super::bvh::Bvh`] and [`super::sphere::SphereBvh`]
/// so every primitive kind shares one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct CylinderBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Cylinders reordered so each leaf owns a contiguous slice.
    cylinders: Vec<Cylinder>,
}

impl CylinderBvh {
    /// Builds a `BVH` over `cylinders` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(cylinders: &[Cylinder]) -> Self {
        Self::build_with(cylinders, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `cylinders` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each cylinder's [`Cylinder::aabb`] and reorders the
    /// cylinders by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`CylinderBvh::cylinders`].
    #[must_use]
    pub fn build_with(cylinders: &[Cylinder], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = cylinders.iter().map(Cylinder::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let cylinders = order.iter().map(|&i| cylinders[i as usize]).collect();
        Self { nodes, cylinders }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of cylinders in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.cylinders.len()
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

    /// Cylinders in leaf-contiguous order.
    #[must_use]
    pub fn cylinders(&self) -> &[Cylinder] {
        &self.cylinders
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<CylinderHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<CylinderHit> = None;

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
                    for cylinder in &self.cylinders[start..end] {
                        if let Some(hit) = cylinder.intersect(&ray) {
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

    /// True when *any* cylinder intersects `ray` inside its interval.
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
                    for cylinder in &self.cylinders[start..end] {
                        if cylinder.intersect(ray).is_some() {
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

    /// Unit-radius cylinder along `+y` from the origin to `(0, 2, 0)`.
    fn unit_cylinder(primitive: u32) -> Cylinder {
        Cylinder::new([0.0, 0.0, 0.0], [0.0, 2.0, 0.0], 1.0, primitive)
    }

    #[test]
    fn lateral_hit_from_the_side() {
        let cyl = unit_cylinder(3);
        // Ray from +z toward -z, at mid-height, hits the lateral surface at z=1.
        let ray = Ray::infinite([0.0, 1.0, 5.0], [0.0, 0.0, -1.0]);
        let hit = cyl.intersect(&ray).expect("side hit");
        assert_eq!(hit.primitive, 3);
        assert!(approx(hit.t, 4.0, 1e-4), "t = {}", hit.t);
        assert!(hit.front_face);
        // Outward normal points toward +z where the ray entered.
        assert!(approx(hit.normal[2], 1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn ray_above_the_top_misses_the_body() {
        let cyl = unit_cylinder(0);
        // Aimed at the side but above y = 2, so the lateral root is out of range
        // and there is no cap to catch a horizontal ray.
        let ray = Ray::infinite([0.0, 3.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(cyl.intersect(&ray).is_none());
    }

    #[test]
    fn ray_outside_radius_misses() {
        let cyl = unit_cylinder(0);
        let ray = Ray::infinite([2.0, 1.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(cyl.intersect(&ray).is_none());
    }

    #[test]
    fn top_cap_hit_from_above() {
        let cyl = unit_cylinder(7);
        // Straight down the axis from above: first surface is the top cap y = 2.
        let ray = Ray::infinite([0.0, 5.0, 0.0], [0.0, -1.0, 0.0]);
        let hit = cyl.intersect(&ray).expect("cap hit");
        assert_eq!(hit.primitive, 7);
        assert!(approx(hit.t, 3.0, 1e-4), "t = {}", hit.t);
        assert!(hit.front_face);
        assert!(approx(hit.normal[1], 1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn ray_from_inside_reports_back_face() {
        let cyl = unit_cylinder(0);
        // Origin inside the solid, aimed out through the lateral surface.
        let ray = Ray::infinite([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let hit = cyl.intersect(&ray).expect("exit hit");
        assert!(approx(hit.t, 1.0, 1e-4), "t = {}", hit.t);
        assert!(!hit.front_face);
        // Inward-flipped normal opposes the outward +z surface normal.
        assert!(approx(hit.normal[2], -1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn behind_origin_is_missed() {
        let cyl = unit_cylinder(0);
        let ray = Ray::infinite([0.0, 1.0, 5.0], [0.0, 0.0, 1.0]);
        assert!(cyl.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let cyl = unit_cylinder(0);
        // Nearest side surface is at t = 4; a shorter interval must miss.
        let ray = Ray::new([0.0, 1.0, 5.0], [0.0, 0.0, -1.0], 0.0, 3.0);
        assert!(cyl.intersect(&ray).is_none());
        let ray = Ray::new([0.0, 1.0, 5.0], [0.0, 0.0, -1.0], 0.0, 5.0);
        assert!(cyl.intersect(&ray).is_some());
    }

    #[test]
    fn zero_radius_never_hits() {
        let cyl = Cylinder::new([0.0, 0.0, 0.0], [0.0, 2.0, 0.0], 0.0, 0);
        let ray = Ray::infinite([0.0, 1.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(cyl.intersect(&ray).is_none());
    }

    #[test]
    fn degenerate_axis_never_hits() {
        let cyl = Cylinder::new([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0, 0);
        let ray = Ray::infinite([5.0, 1.0, 1.0], [-1.0, 0.0, 0.0]);
        assert!(cyl.intersect(&ray).is_none());
    }

    fn random_cylinder(rng: &mut Rng, primitive: u32) -> Cylinder {
        let base = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        let top = [
            base[0] + rng.range(-3.0, 3.0),
            base[1] + rng.range(-3.0, 3.0),
            base[2] + rng.range(-3.0, 3.0),
        ];
        Cylinder::new(base, top, rng.range(0.2, 1.2), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Cylinder> {
        (0..count).map(|i| random_cylinder(rng, i)).collect()
    }

    fn brute_closest(cylinders: &[Cylinder], ray: &Ray) -> Option<CylinderHit> {
        let mut best: Option<CylinderHit> = None;
        let mut ray = *ray;
        for cylinder in cylinders {
            if let Some(hit) = cylinder.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = CylinderBvh::build(&[]);
        assert!(bvh.is_empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0x0c17_11de_9abc_1234u64);
        let cylinders = random_scene(&mut rng, 64);
        let bvh = CylinderBvh::build(&cylinders);

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

            let expected = brute_closest(&cylinders, &ray);
            let actual = bvh.closest_hit(&ray);
            match (expected, actual) {
                (None, None) => {}
                (Some(e), Some(a)) => {
                    assert_eq!(e.primitive, a.primitive);
                    assert_eq!(e.t.to_bits(), a.t.to_bits(), "t bits differ");
                    assert_eq!(e.front_face, a.front_face);
                    for k in 0..3 {
                        assert_eq!(
                            e.normal[k].to_bits(),
                            a.normal[k].to_bits(),
                            "normal[{k}] bits differ"
                        );
                    }
                }
                (e, a) => panic!("hit disagreement: {e:?} vs {a:?}"),
            }
        }
    }

    #[test]
    fn bvh_any_hit_matches_brute_force() {
        let mut rng = Rng::new(0x0c17_11de_5678_9abcu64);
        let cylinders = random_scene(&mut rng, 48);
        let bvh = CylinderBvh::build(&cylinders);

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
            assert_eq!(bvh.any_hit(&ray), brute_closest(&cylinders, &ray).is_some());
        }
    }
}
