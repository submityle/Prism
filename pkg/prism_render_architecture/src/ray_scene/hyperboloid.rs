//! Analytic *hyperboloid of one sheet* (hourglass / cooling-tower wall)
//! primitive and its single-level `BVH`.
//!
//! Like [`super::cone`] and [`super::paraboloid`], this is a procedural
//! primitive for the `DXR`/Vulkan `AABB` path: the `BLAS` stores one
//! axis-aligned box per hyperboloid and an intersection shader refines the hit.
//! A hyperboloid of revolution is the natural closed-form proxy for a cooling
//! tower, an hourglass or waisted vase wall, a hyperbolic lamp shade, or a
//! reflector neck, so a path tracer wants an exact quadric test rather than a
//! tessellated mesh.
//!
//! A [`Hyperboloid`] is the surface of revolution whose radius grows away from
//! a circular *waist*: at signed axial distance `z` from the waist plane the
//! radius obeys `ρ(z)² = waist² + flare² · z²`. It is centered on the `center`
//! waist point, its axis and half-height come from `top` (the axis spans
//! `center ± (top − center)`, symmetric about the waist), `waist` is the throat
//! radius and `flare` the radial growth per axial unit. `flare = 0` degenerates
//! to a [`super::cylinder::Cylinder`] of radius `waist`; `waist = 0` degenerates
//! to a double cone through the throat.
//!
//! The intersection substitutes the ray into the implicit quadric
//! `F = |p − center|² − (1 + flare²)·(⟨p − center, n̂⟩)² − waist² = 0`, giving a
//! quadratic in `t` whose `t²` coefficient carries `dd = ⟨d, d⟩` (so the test is
//! correct for the non-unit ray directions `ray_scene` feeds it), clips both
//! roots to the finite axial band `z ∈ [−h, h]`, and returns the nearest valid
//! root. The normal is the analytic gradient `2·(p − center) − 2(1 + flare²)·z·n̂`
//! (normalized). Every step is add/sub/mul/div/`sqrt` and comparisons, so it is
//! bit-reproducible on the `GPU` and free of any transcendental call.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic hyperboloid of one sheet (surface of revolution) in world space.
///
/// The wall is swept by the profile `ρ(z)² = waist² + flare²·z²` around the axis
/// `center → top` and mirrored below the waist, spanning `center ± (top −
/// center)`. `primitive` is the caller's stable id (mirroring
/// [`super::bvh::Triangle`] and [`super::cone::Cone`]): the [`HyperboloidBvh`]
/// builder reorders hyperboloids internally but always reports hits by this id.
/// `waist` and `flare` are stored non-negative; caller-supplied negatives are
/// folded to their magnitude so the derived [`Aabb`] and the quadric stay well
/// formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyperboloid {
    /// Waist (throat) center, on the axis, midplane of the symmetric surface.
    center: [f32; 3],
    /// One rim end of the axis; the axis spans `center ± (top − center)`.
    top: [f32; 3],
    /// Non-negative throat radius at the waist plane.
    waist: f32,
    /// Non-negative radial growth per axial unit away from the waist.
    flare: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Hyperboloid {
    /// Builds a hyperboloid with waist at `center`, axis and half-height from
    /// `top`, throat radius `waist` and slope `flare` (each folded to its
    /// magnitude), tagged with stable id `primitive`.
    #[must_use]
    pub fn new(center: [f32; 3], top: [f32; 3], waist: f32, flare: f32, primitive: u32) -> Self {
        Self {
            center,
            top,
            waist: waist.abs(),
            flare: flare.abs(),
            primitive,
        }
    }

    /// Waist (throat) center of the surface.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        self.center
    }

    /// Rim end of the axis (half-height reference).
    #[must_use]
    pub fn top(&self) -> [f32; 3] {
        self.top
    }

    /// Non-negative throat radius at the waist plane.
    #[must_use]
    pub fn waist(&self) -> f32 {
        self.waist
    }

    /// Non-negative radial growth per axial unit away from the waist.
    #[must_use]
    pub fn flare(&self) -> f32 {
        self.flare
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds of the hyperboloid.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. The
    /// surface is contained in the cylinder of radius `ρ_max = √(waist² +
    /// flare²·h²)` (its widest, reached at either rim) around the axis segment
    /// `[center − (top − center), top]`, so the box is that cylinder's bound:
    /// each axis `i` picks the segment endpoints and pads by the exact projected
    /// radius of a rim circle onto that axis, `ρ_max · √(1 − axisᵢ²/|axis|²)` (as
    /// in [`super::disk::Disk::aabb`]). A zero-length axis falls back to a loose
    /// per-endpoint cube of radius `waist`.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let ba = sub(self.top, self.center);
        let baba = dot(ba, ba);
        let bottom = sub(self.center, ba);
        let rho_max = if baba > 0.0 {
            (self.waist * self.waist + self.flare * self.flare * baba)
                .max(0.0)
                .sqrt()
        } else {
            self.waist
        };
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            let frac = if baba > 0.0 {
                (1.0 - (ba[axis] * ba[axis]) / baba).max(0.0).sqrt()
            } else {
                1.0
            };
            let e = rho_max * frac;
            let lo = self.top[axis].min(bottom[axis]) - e;
            let hi = self.top[axis].max(bottom[axis]) + e;
            *slot.0 = lo;
            *slot.1 = hi;
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/hyperboloid intersection inside `ray`'s `[t_min, t_max]`
    /// interval, or `None` when the ray misses.
    ///
    /// [`HyperboloidHit::normal`] is the unit surface normal oriented *against*
    /// the incident ray, and [`HyperboloidHit::front_face`] is `true` when the
    /// ray struck the outward-facing (convex, away-from-axis) side. A zero-length
    /// axis or a zero-length ray direction never reports a hit.
    ///
    /// The ray is substituted into the implicit quadric
    /// `F = |p − center|² − (1 + flare²)·⟨p − center, n̂⟩² − waist² = 0`, giving a
    /// quadratic in `t`; both roots are considered and clipped to the finite band
    /// `z ∈ [−h, h]`, and the nearest valid root is returned. The normal is the
    /// analytic gradient `2·(p − center) − 2(1 + flare²)·z·n̂` (normalized).
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<HyperboloidHit> {
        let w = sub(self.top, self.center);
        let h2 = dot(w, w);
        if h2 <= 0.0 {
            return None;
        }
        let direction = ray.direction();
        let dd = dot(direction, direction);
        if dd <= 0.0 {
            return None;
        }

        let h = h2.sqrt();
        let inv_h = 1.0 / h;
        // Unit axis from the waist toward `top`.
        let n = scale(w, inv_h);
        // `1 + flare²` weights the axial term in the quadric.
        let g = 1.0 + self.flare * self.flare;

        // Ray origin relative to the waist center.
        let a = sub(ray.origin(), self.center);
        let za = dot(a, n);
        let zd = dot(direction, n);
        let ad = dot(a, direction);
        let aa = dot(a, a);

        // `A·t² + B·t + C = 0` for `F = |a + t·d|² − g·(za + t·zd)² − waist²`.
        let coeff_a = dd - g * zd * zd;
        let coeff_b = 2.0 * (ad - g * za * zd);
        let coeff_c = aa - g * za * za - self.waist * self.waist;

        let t_min = ray.t_min();
        let t_max = ray.t_max();

        // Collect the candidate roots (quadratic, or linear when `A == 0`, i.e.
        // the ray runs along a surface asymptote direction).
        let roots = if coeff_a != 0.0 {
            let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
            if disc < 0.0 {
                return None;
            }
            let sqrt_disc = disc.sqrt();
            let inv_2a = 1.0 / (2.0 * coeff_a);
            let r0 = (-coeff_b - sqrt_disc) * inv_2a;
            let r1 = (-coeff_b + sqrt_disc) * inv_2a;
            [r0.min(r1), r0.max(r1)]
        } else if coeff_b != 0.0 {
            let r = -coeff_c / coeff_b;
            [r, r]
        } else {
            return None;
        };

        let mut best_t = f32::INFINITY;
        let mut best_outward = [0.0f32; 3];
        for t in roots {
            if !(t >= t_min && t <= t_max) || t >= best_t {
                continue;
            }
            // Signed axial distance from the waist; valid on `[−h, h]`.
            let z = za + t * zd;
            if z < -h || z > h {
                continue;
            }
            // Point relative to the waist center.
            let p = [a[0] + t * direction[0], a[1] + t * direction[1], a[2] + t * direction[2]];
            // Outward gradient of `F`: `2·p − 2·g·z·n̂`.
            let gz = g * z;
            let grad = [
                2.0 * (p[0] - gz * n[0]),
                2.0 * (p[1] - gz * n[1]),
                2.0 * (p[2] - gz * n[2]),
            ];
            let nn = dot(grad, grad);
            if nn <= 0.0 {
                continue;
            }
            best_t = t;
            best_outward = scale(grad, 1.0 / nn.sqrt());
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
        Some(HyperboloidHit {
            t: best_t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/hyperboloid intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperboloidHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the hyperboloid that was hit.
    pub primitive: u32,
    /// Unit surface normal, oriented against the incident ray.
    pub normal: [f32; 3],
    /// `true` when the ray struck the outward-facing (convex) side.
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

/// A single-level `BVH` over analytic [`Hyperboloid`] primitives.
///
/// Empty input yields an empty hierarchy ([`HyperboloidBvh::is_empty`]);
/// traversal of an empty hierarchy simply never reports a hit. The layout and
/// ordered slab walk mirror the triangle [`super::bvh::Bvh`],
/// [`super::cone::ConeBvh`], and [`super::paraboloid::ParaboloidBvh`] so every
/// primitive kind shares one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct HyperboloidBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Hyperboloids reordered so each leaf owns a contiguous slice.
    hyperboloids: Vec<Hyperboloid>,
}

impl HyperboloidBvh {
    /// Builds a `BVH` over `hyperboloids` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(hyperboloids: &[Hyperboloid]) -> Self {
        Self::build_with(hyperboloids, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `hyperboloids` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each hyperboloid's [`Hyperboloid::aabb`] and
    /// reorders the hyperboloids by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`HyperboloidBvh::hyperboloids`].
    #[must_use]
    pub fn build_with(hyperboloids: &[Hyperboloid], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = hyperboloids.iter().map(Hyperboloid::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let hyperboloids = order.iter().map(|&i| hyperboloids[i as usize]).collect();
        Self { nodes, hyperboloids }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of hyperboloids in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.hyperboloids.len()
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

    /// Hyperboloids in leaf-contiguous order.
    #[must_use]
    pub fn hyperboloids(&self) -> &[Hyperboloid] {
        &self.hyperboloids
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<HyperboloidHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<HyperboloidHit> = None;

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
                    for hyperboloid in &self.hyperboloids[start..end] {
                        if let Some(hit) = hyperboloid.intersect(&ray) {
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

    /// True when *any* hyperboloid intersects `ray` inside its interval.
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
                    for hyperboloid in &self.hyperboloids[start..end] {
                        if hyperboloid.intersect(ray).is_some() {
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

    /// Cooling-tower-like wall: waist at the origin, axis along `+z` with
    /// half-height `4`, throat radius `1`, flaring at slope `0.5` (rim radius
    /// `√(1 + 0.25·16) = √5 ≈ 2.236`).
    fn sample_hyperboloid(primitive: u32) -> Hyperboloid {
        Hyperboloid::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 1.0, 0.5, primitive)
    }

    #[test]
    fn hits_the_waist_from_the_side() {
        let tower = sample_hyperboloid(2);
        // At the waist plane the radius is exactly `waist = 1`; aim inward.
        let ray = Ray::infinite([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        let hit = tower.intersect(&ray).expect("waist hit");
        assert_eq!(hit.primitive, 2);
        // Enters the near wall at x = 1, four units in from x = 5.
        assert!(approx(hit.t, 4.0, 1e-3), "t = {}", hit.t);
        // Outward normal at the waist points along +x (radially), opposed to the
        // −x ray, so it stays +x.
        assert!(approx(hit.normal[0], 1.0, 1e-3), "normal = {:?}", hit.normal);
        assert!(hit.front_face);
    }

    #[test]
    fn hits_the_flared_rim_wall() {
        let tower = sample_hyperboloid(0);
        // At z = 4 the radius is √5; aim inward along −x at that height.
        let ray = Ray::infinite([5.0, 0.0, 4.0], [-1.0, 0.0, 0.0]);
        let hit = tower.intersect(&ray).expect("rim wall hit");
        let rho = (5.0f32).sqrt();
        assert!(approx(hit.t, 5.0 - rho, 2e-3), "t = {}", hit.t);
    }

    #[test]
    fn misses_through_the_open_throat() {
        let tower = sample_hyperboloid(0);
        // Ray inside the throat, parallel to the axis: never crosses the wall.
        let ray = Ray::infinite([0.2, 0.0, -10.0], [0.0, 0.0, 1.0]);
        assert!(tower.intersect(&ray).is_none());
    }

    #[test]
    fn misses_beyond_the_axial_band() {
        let tower = sample_hyperboloid(0);
        // Radial ray above the rim (z = 6 > h = 4): clipped out of the band.
        let ray = Ray::infinite([5.0, 0.0, 6.0], [-1.0, 0.0, 0.0]);
        assert!(tower.intersect(&ray).is_none());
    }

    #[test]
    fn degenerate_axis_never_hits() {
        let bad = Hyperboloid::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0.5, 0);
        let ray = Ray::infinite([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(bad.intersect(&ray).is_none());
    }

    #[test]
    fn zero_direction_never_hits() {
        let tower = sample_hyperboloid(0);
        let ray = Ray::infinite([5.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(tower.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let tower = sample_hyperboloid(0);
        // The near wall is at t = 4; a t_max just short of it rejects the hit.
        let ray = Ray::new([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 3.9);
        assert!(tower.intersect(&ray).is_none());
        // Extending the interval past the wall admits it again.
        let ray = Ray::new([5.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 4.1);
        assert!(tower.intersect(&ray).is_some());
    }

    #[test]
    fn zero_flare_is_a_cylinder_wall() {
        // `flare = 0` collapses the profile to the constant radius `waist`.
        let tube = Hyperboloid::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 2.0, 0.0, 7);
        let ray = Ray::infinite([5.0, 0.0, 1.0], [-1.0, 0.0, 0.0]);
        let hit = tube.intersect(&ray).expect("cylinder wall hit");
        // Constant radius 2 → near wall at x = 2, three units in from x = 5.
        assert!(approx(hit.t, 3.0, 1e-3), "t = {}", hit.t);
    }

    /// Independent residual check: every reported hit point must satisfy the
    /// implicit quadric `F = |p−c|² − (1+flare²)·z² − waist² ≈ 0` and lie inside
    /// the axial band. This catches a mistranscribed coefficient or normal that
    /// a `BVH`↔brute cross-check (sharing [`Hyperboloid::intersect`]) cannot.
    #[test]
    fn reported_hit_lies_on_the_quadric() {
        let mut rng = Rng::new(0xB0DE_1234);
        for _ in 0..4_000 {
            let center = [rng.range(-4.0, 4.0), rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)];
            let top = [
                center[0] + rng.range(-3.0, 3.0),
                center[1] + rng.range(-3.0, 3.0),
                center[2] + rng.range(0.5, 3.0),
            ];
            let waist = rng.range(0.3, 1.5);
            let flare = rng.range(0.0, 1.2);
            let hyp = Hyperboloid::new(center, top, waist, flare, 0);

            let origin = [rng.range(-9.0, 9.0), rng.range(-9.0, 9.0), rng.range(-9.0, 9.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let Some(hit) = hyp.intersect(&ray) else {
                continue;
            };

            let p = ray.at(hit.t);
            let q = sub(p, center);
            let w = sub(top, center);
            let h2 = dot(w, w);
            let h = h2.sqrt();
            let n = scale(w, 1.0 / h);
            let z = dot(q, n);
            let g = 1.0 + flare * flare;
            let residual = dot(q, q) - g * z * z - waist * waist;
            let scale_ref = 1.0 + dot(q, q);
            assert!(
                residual.abs() / scale_ref < 5e-3,
                "off-surface residual {residual} (scaled {})",
                residual.abs() / scale_ref
            );
            assert!(z >= -h - 1e-3 && z <= h + 1e-3, "axial z = {z} out of band h = {h}");
        }
    }

    fn random_hyperboloid(rng: &mut Rng, primitive: u32) -> Hyperboloid {
        let center = [rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), rng.range(-5.0, 5.0)];
        let top = [
            center[0] + rng.range(-3.0, 3.0),
            center[1] + rng.range(-3.0, 3.0),
            center[2] + rng.range(0.5, 3.0),
        ];
        Hyperboloid::new(center, top, rng.range(0.3, 1.5), rng.range(0.0, 1.2), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Hyperboloid> {
        (0..count).map(|i| random_hyperboloid(rng, i)).collect()
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = HyperboloidBvh::build(&[]);
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
        let bvh = HyperboloidBvh::build(&scene);
        let ordered = bvh.hyperboloids().to_vec();

        for _ in 0..3_000 {
            let origin = [rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            let mut brute: Option<HyperboloidHit> = None;
            let mut r = ray;
            for hyp in &ordered {
                if let Some(hit) = hyp.intersect(&r) {
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
        let bvh = HyperboloidBvh::build(&scene);
        let ordered = bvh.hyperboloids().to_vec();

        for _ in 0..3_000 {
            let origin = [rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let brute = ordered.iter().any(|h| h.intersect(&ray).is_some());
            assert_eq!(brute, bvh.any_hit(&ray));
        }
    }
}
