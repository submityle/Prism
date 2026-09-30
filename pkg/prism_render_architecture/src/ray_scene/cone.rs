//! Analytic finite *capped cone* (conical frustum) primitive and its
//! single-level `BVH`.
//!
//! Like [`super::cylinder`] and [`super::disk`], this is a procedural primitive
//! for the `DXR`/Vulkan `AABB` path: the `BLAS` stores one axis-aligned box per
//! cone and an intersection shader refines the hit. A capped cone is the natural
//! proxy for a spot-light cone volume, a tapered tube / horn, a lamp shade, or a
//! projector frustum, so a path tracer wants a closed-form test rather than a
//! tessellated cone.
//!
//! A [`Cone`] is the solid swept between a base cap of radius `radius_base` at
//! `base` and a top cap of radius `radius_top` at `top`, closed by two disk
//! caps. Equal radii degenerate to a [`super::cylinder::Cylinder`]; a zero top
//! radius degenerates to a sharp-tipped cone. The intersection forms the
//! lateral-surface quadratic in the reduced Inigo-Quilez `iCappedCone` form and
//! separately tests the two cap planes exactly as [`super::cylinder`] does; it
//! then returns the nearest of all valid roots inside the ray interval. Every
//! step is add/sub/mul/div/`sqrt` and comparisons, so it is bit-reproducible on
//! the `GPU` and free of any transcendental call.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic finite capped cone (conical frustum) in world space.
///
/// The solid tapers from a disk of `radius_base` at `base` to a disk of
/// `radius_top` at `top`, closed by the two caps. `primitive` is the caller's
/// stable id (mirroring [`super::bvh::Triangle`] and
/// [`super::cylinder::Cylinder`]): the [`ConeBvh`] builder reorders cones
/// internally but always reports hits by this id. Both radii are stored
/// non-negative; a caller-supplied negative radius is folded to its magnitude
/// so the derived [`Aabb`] and the quadric stay well formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cone {
    /// Center of the base cap (axis start), carrying `radius_base`.
    base: [f32; 3],
    /// Center of the top cap (axis end), carrying `radius_top`.
    top: [f32; 3],
    /// Non-negative radius of the base cap.
    radius_base: f32,
    /// Non-negative radius of the top cap.
    radius_top: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Cone {
    /// Builds a cone spanning `base → top` tapering from `radius_base` to
    /// `radius_top` (each folded to its magnitude), tagged with stable id
    /// `primitive`.
    #[must_use]
    pub fn new(
        base: [f32; 3],
        top: [f32; 3],
        radius_base: f32,
        radius_top: f32,
        primitive: u32,
    ) -> Self {
        Self {
            base,
            top,
            radius_base: radius_base.abs(),
            radius_top: radius_top.abs(),
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

    /// Non-negative radius of the base cap.
    #[must_use]
    pub fn radius_base(&self) -> f32 {
        self.radius_base
    }

    /// Non-negative radius of the top cap.
    #[must_use]
    pub fn radius_top(&self) -> f32 {
        self.radius_top
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds of the capped cone.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. Each
    /// cap circle projects onto world axis `i` with half extent
    /// `radius · √(1 − axisᵢ²/|axis|²)` (the exact projected radius of the
    /// circle, as in [`super::disk::Disk::aabb`]); the box is the union of the
    /// two projected cap circles, so it hugs the frustum rather than using a
    /// loose cube. A zero-length axis falls back to a loose per-cap cube.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let ba = sub(self.top, self.base);
        let baba = dot(ba, ba);
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            let frac = if baba > 0.0 {
                // Projected-radius fraction of a cap circle onto world `axis`.
                (1.0 - (ba[axis] * ba[axis]) / baba).max(0.0).sqrt()
            } else {
                1.0
            };
            let eb = self.radius_base * frac;
            let et = self.radius_top * frac;
            let lo = (self.base[axis] - eb).min(self.top[axis] - et);
            let hi = (self.base[axis] + eb).max(self.top[axis] + et);
            *slot.0 = lo;
            *slot.1 = hi;
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/cone intersection inside `ray`'s `[t_min, t_max]` interval,
    /// or `None` when the ray misses.
    ///
    /// [`ConeHit::normal`] is the unit surface normal oriented *against* the
    /// incident ray, and [`ConeHit::front_face`] is `true` when the ray struck
    /// the outward-facing side (a ray arriving from inside reports
    /// `front_face == false` with a flipped normal). A zero-length axis or a
    /// zero-length ray direction never reports a hit.
    ///
    /// The lateral surface uses the reduced Inigo-Quilez `iCappedCone`
    /// quadratic `k2·t² + 2·k1·t + k0 = 0`; the two caps reuse the exact plane +
    /// radius test from [`super::cylinder::Cylinder::intersect`]. All candidate
    /// roots inside the interval are considered and the nearest is returned.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<ConeHit> {
        let ba = sub(self.top, self.base);
        let m0 = dot(ba, ba);
        if m0 <= 0.0 {
            return None;
        }
        let direction = ray.direction();
        let dd = dot(direction, direction);
        if dd <= 0.0 {
            return None;
        }
        let oa = sub(ray.origin(), self.base);
        let ra = self.radius_base;
        let rb = self.radius_top;

        let m1 = dot(oa, ba);
        let m2 = dot(direction, ba);
        let m3 = dot(direction, oa);
        let m5 = dot(oa, oa);

        let t_min = ray.t_min();
        let t_max = ray.t_max();
        let mut best_t = f32::INFINITY;
        let mut best_outward = [0.0f32; 3];

        // Lateral surface: reduced quadratic (Quilez `iCappedCone`).
        let rr = ra - rb;
        let hy = m0 + rr * rr;
        // The `t²` coefficient carries `dd = dot(dir, dir)`; Quilez's shader
        // assumes a unit ray direction (`dd == 1`), but callers here pass
        // arbitrary-length directions, so keep the general form.
        let k2 = m0 * m0 * dd - m2 * m2 * hy;
        let k1 = m0 * m0 * m3 - m1 * m2 * hy + m0 * ra * (rr * m2);
        let k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0 - m0 * ra);
        let h = k1 * k1 - k2 * k0;
        if h >= 0.0 {
            let sqrt_h = h.sqrt();
            let roots = if k2 != 0.0 {
                let r0 = (-k1 - sqrt_h) / k2;
                let r1 = (-k1 + sqrt_h) / k2;
                [r0.min(r1), r0.max(r1)]
            } else if k1 != 0.0 {
                // Degenerate to a linear equation `2·k1·t + k0 = 0`.
                let r = -k0 / (2.0 * k1);
                [r, r]
            } else {
                [f32::NAN, f32::NAN]
            };
            for t in roots {
                if !(t >= t_min && t <= t_max) || t >= best_t {
                    continue;
                }
                // Axis coordinate of the hit, valid on the body in `(0, m0)`.
                let y = m1 + t * m2;
                if y <= 0.0 || y >= m0 {
                    continue;
                }
                // Unnormalized outward normal (Quilez form):
                // `m0·(m0·oa + t·m0·rd − ba·rr·ra) − ba·hy·y`.
                let inner = [
                    m0 * oa[0] + t * m0 * direction[0] - ba[0] * rr * ra,
                    m0 * oa[1] + t * m0 * direction[1] - ba[1] * rr * ra,
                    m0 * oa[2] + t * m0 * direction[2] - ba[2] * rr * ra,
                ];
                let n = [
                    m0 * inner[0] - ba[0] * hy * y,
                    m0 * inner[1] - ba[1] * hy * y,
                    m0 * inner[2] - ba[2] * hy * y,
                ];
                let nn = dot(n, n);
                if nn <= 0.0 {
                    continue;
                }
                best_t = t;
                best_outward = scale(n, 1.0 / nn.sqrt());
            }
        }

        // Caps: base plane `y = 0` (radius `ra`, outward `−axiŝ`) and top plane
        // `y = m0` (radius `rb`, outward `+axiŝ`). `m2 == 0` means the ray runs
        // parallel to both cap planes and cannot cross them.
        if m2 != 0.0 {
            let inv_m2 = 1.0 / m2;
            let inv_sqrt_m0 = 1.0 / m0.sqrt();
            for (y_cap, r_cap, sign) in [(0.0f32, ra, -1.0f32), (m0, rb, 1.0f32)] {
                if r_cap <= 0.0 {
                    continue;
                }
                let t = (y_cap - m1) * inv_m2;
                if !(t >= t_min && t <= t_max) || t >= best_t {
                    continue;
                }
                let s = y_cap / m0;
                let point_rel = [
                    oa[0] + t * direction[0] - ba[0] * s,
                    oa[1] + t * direction[1] - ba[1] * s,
                    oa[2] + t * direction[2] - ba[2] * s,
                ];
                // Inside the cap disk when the perpendicular offset is within r.
                if dot(point_rel, point_rel) > r_cap * r_cap {
                    continue;
                }
                best_t = t;
                best_outward = scale(ba, sign * inv_sqrt_m0);
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
        Some(ConeHit {
            t: best_t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/cone intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConeHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the cone that was hit.
    pub primitive: u32,
    /// Unit surface normal oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the ray struck the outward-facing side; `false` when it
    /// arrived from inside, in which case `normal` is flipped to oppose it.
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

/// A single-level `BVH` over analytic [`Cone`] primitives.
///
/// Empty input yields an empty hierarchy ([`ConeBvh::is_empty`]); traversal of
/// an empty hierarchy simply never reports a hit. The layout and ordered slab
/// walk mirror the triangle [`super::bvh::Bvh`],
/// [`super::cylinder::CylinderBvh`], and [`super::disk::DiskBvh`] so every
/// primitive kind shares one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct ConeBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Cones reordered so each leaf owns a contiguous slice.
    cones: Vec<Cone>,
}

impl ConeBvh {
    /// Builds a `BVH` over `cones` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(cones: &[Cone]) -> Self {
        Self::build_with(cones, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `cones` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each cone's [`Cone::aabb`] and reorders the cones
    /// by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`ConeBvh::cones`].
    #[must_use]
    pub fn build_with(cones: &[Cone], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = cones.iter().map(Cone::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let cones = order.iter().map(|&i| cones[i as usize]).collect();
        Self { nodes, cones }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of cones in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.cones.len()
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

    /// Cones in leaf-contiguous order.
    #[must_use]
    pub fn cones(&self) -> &[Cone] {
        &self.cones
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ConeHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<ConeHit> = None;

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
                    for cone in &self.cones[start..end] {
                        if let Some(hit) = cone.intersect(&ray) {
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

    /// True when *any* cone intersects `ray` inside its interval.
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
                    for cone in &self.cones[start..end] {
                        if cone.intersect(ray).is_some() {
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

    /// Frustum from `z = 0` (radius 2) to `z = 2` (radius 1), axis along `+z`.
    fn sample_cone(primitive: u32) -> Cone {
        Cone::new([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 2.0, 1.0, primitive)
    }

    #[test]
    fn hits_base_cap_from_below() {
        let cone = sample_cone(3);
        // Straight down the axis from below the base cap.
        let ray = Ray::infinite([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        let hit = cone.intersect(&ray).expect("base cap hit");
        assert_eq!(hit.primitive, 3);
        assert!(approx(hit.t, 5.0, 1e-4), "t = {}", hit.t);
        assert!(hit.front_face);
        // Base cap outward normal is -z; oriented against the +z ray it stays -z.
        assert!(approx(hit.normal[2], -1.0, 1e-4), "normal = {:?}", hit.normal);
    }

    #[test]
    fn hits_lateral_surface_from_the_side() {
        let cone = sample_cone(0);
        // Aim inward at mid-height where the radius is 1.5.
        let ray = Ray::infinite([5.0, 0.0, 1.0], [-1.0, 0.0, 0.0]);
        let hit = cone.intersect(&ray).expect("lateral hit");
        // Enters the frustum surface at x ≈ 1.5 (t ≈ 3.5).
        assert!(hit.front_face);
        // Lateral normal has no radial-inward bias: it points outward (+x side).
        assert!(hit.normal[0] > 0.0, "normal = {:?}", hit.normal);
    }

    #[test]
    fn axis_ray_misses_when_outside_radius() {
        let cone = sample_cone(0);
        // Parallel to axis but offset beyond the widest radius (2).
        let ray = Ray::infinite([3.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        assert!(cone.intersect(&ray).is_none());
    }

    #[test]
    fn degenerate_axis_never_hits() {
        let cone = Cone::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 1.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(cone.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let cone = sample_cone(0);
        let ray = Ray::new([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 0.0, 4.0);
        assert!(cone.intersect(&ray).is_none());
        let ray = Ray::new([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 0.0, 6.0);
        assert!(cone.intersect(&ray).is_some());
    }

    #[test]
    fn sharp_tip_cone_hits_apex_side() {
        // Zero top radius: a true cone tapering to a point at z = 2.
        let cone = Cone::new([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 1.0, 0.0, 0);
        // Side ray near the base where the radius is close to 1.
        let ray = Ray::infinite([5.0, 0.0, 0.2], [-1.0, 0.0, 0.0]);
        assert!(cone.intersect(&ray).is_some());
    }

    fn random_cone(rng: &mut Rng, primitive: u32) -> Cone {
        let base = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        let top = [
            base[0] + rng.range(-3.0, 3.0),
            base[1] + rng.range(-3.0, 3.0),
            base[2] + rng.range(0.5, 3.0),
        ];
        Cone::new(base, top, rng.range(0.3, 1.5), rng.range(0.0, 1.2), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Cone> {
        (0..count).map(|i| random_cone(rng, i)).collect()
    }

    fn brute_closest(cones: &[Cone], ray: &Ray) -> Option<ConeHit> {
        let mut best: Option<ConeHit> = None;
        let mut ray = *ray;
        for cone in cones {
            if let Some(hit) = cone.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = ConeBvh::build(&[]);
        assert!(bvh.is_empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0x0d15_c000_9abc_1234u64);
        let cones = random_scene(&mut rng, 64);
        let bvh = ConeBvh::build(&cones);

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

            let expected = brute_closest(&cones, &ray);
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
        let mut rng = Rng::new(0x0d15_c000_5678_9abcu64);
        let cones = random_scene(&mut rng, 48);
        let bvh = ConeBvh::build(&cones);

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
            assert_eq!(bvh.any_hit(&ray), brute_closest(&cones, &ray).is_some());
        }
    }
}
