//! Analytic *paraboloid* (parabolic dish / bowl) primitive and its single-level
//! `BVH`.
//!
//! Like [`super::cone`] and [`super::cylinder`], this is a procedural primitive
//! for the `DXR`/Vulkan `AABB` path: the `BLAS` stores one axis-aligned box per
//! paraboloid and an intersection shader refines the hit. A paraboloid of
//! revolution is the natural proxy for a reflector dish (headlamp / spot-light
//! reflector, satellite dish, parabolic microphone), a rounded goblet wall, or
//! any surface whose radius grows as the square root of the axial distance, so a
//! path tracer wants a closed-form quadric test rather than a tessellated bowl.
//!
//! A [`Paraboloid`] is the surface of revolution swept from an `apex` (the
//! vertex, radius `0`) to a `top` rim of radius `radius`, with the perpendicular
//! distance to the axis growing as `ρ(z) = radius · √(z / h)` for axial distance
//! `z ∈ [0, h]` (`h = |top − apex|`). Equivalently the surface satisfies
//! `k · ρ² = z` with `k = h / radius²`. The intersection substitutes the ray
//! into that implicit quadric, yielding a quadratic whose roots are clipped to
//! the finite `z ∈ [0, h]` band; the analytic gradient `2k · ρ⃗ − n̂` gives the
//! surface normal. Every step is add/sub/mul/div/`sqrt` and comparisons, so it
//! is bit-reproducible on the `GPU` and free of any transcendental call.
//!
//! The dish is open at the wide `top` rim (no cap disk), matching the classic
//! `pbrt` paraboloid quadric; the vertex end is closed naturally because the
//! radius collapses to a point at the `apex`.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic paraboloid of revolution in world space.
///
/// The surface is swept from the `apex` vertex (radius `0`) to a `top` rim of
/// radius `radius`, so the perpendicular distance to the axis grows as
/// `radius · √(z / h)` with `z` the axial distance from the `apex` and
/// `h = |top − apex|`. `primitive` is the caller's stable id (mirroring
/// [`super::bvh::Triangle`] and [`super::cone::Cone`]): the [`ParaboloidBvh`]
/// builder reorders paraboloids internally but always reports hits by this id.
/// `radius` is stored non-negative; a caller-supplied negative radius is folded
/// to its magnitude so the derived [`Aabb`] and the quadric stay well formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Paraboloid {
    /// Vertex of the dish (axis start, radius `0`).
    apex: [f32; 3],
    /// Center of the wide rim (axis end), carrying `radius`.
    top: [f32; 3],
    /// Non-negative radius of the rim at `top`.
    radius: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl Paraboloid {
    /// Builds a paraboloid spanning `apex → top` opening to `radius` at the rim
    /// (folded to its magnitude), tagged with stable id `primitive`.
    #[must_use]
    pub fn new(apex: [f32; 3], top: [f32; 3], radius: f32, primitive: u32) -> Self {
        Self {
            apex,
            top,
            radius: radius.abs(),
            primitive,
        }
    }

    /// Vertex of the dish (axis start).
    #[must_use]
    pub fn apex(&self) -> [f32; 3] {
        self.apex
    }

    /// Center of the wide rim (axis end).
    #[must_use]
    pub fn top(&self) -> [f32; 3] {
        self.top
    }

    /// Non-negative radius of the rim at `top`.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds of the paraboloid.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. The
    /// dish is contained in the cylinder of radius `radius` around the segment
    /// `apex → top` (because `ρ(z) = radius · √(z / h) ≤ radius`), so the box is
    /// that cylinder's bound: each axis `i` picks the segment endpoints and pads
    /// by the exact projected radius of a rim circle onto that axis,
    /// `radius · √(1 − axisᵢ²/|axis|²)` (as in [`super::disk::Disk::aabb`]). It
    /// is tight at the wide rim and only mildly loose toward the vertex. A
    /// zero-length axis falls back to a loose per-endpoint cube.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let ba = sub(self.top, self.apex);
        let baba = dot(ba, ba);
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            let frac = if baba > 0.0 {
                // Projected-radius fraction of the rim circle onto world `axis`.
                (1.0 - (ba[axis] * ba[axis]) / baba).max(0.0).sqrt()
            } else {
                1.0
            };
            let e = self.radius * frac;
            let lo = self.apex[axis].min(self.top[axis]) - e;
            let hi = self.apex[axis].max(self.top[axis]) + e;
            *slot.0 = lo;
            *slot.1 = hi;
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/paraboloid intersection inside `ray`'s `[t_min, t_max]`
    /// interval, or `None` when the ray misses.
    ///
    /// [`ParaboloidHit::normal`] is the unit surface normal oriented *against*
    /// the incident ray, and [`ParaboloidHit::front_face`] is `true` when the
    /// ray struck the outward-facing (convex) side. A zero-length axis, a
    /// zero-radius dish, or a zero-length ray direction never reports a hit.
    ///
    /// The ray is substituted into the implicit quadric `k · ρ² − z = 0`
    /// (`ρ` = perpendicular distance to the axis, `z` = axial distance from the
    /// `apex`, `k = h / radius²`), giving a quadratic in `t`; both roots are
    /// considered and clipped to the finite `z ∈ [0, h]` band, and the nearest
    /// valid root is returned. The normal is the analytic gradient
    /// `2k · ρ⃗ − n̂` (normalized).
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<ParaboloidHit> {
        let w = sub(self.top, self.apex);
        let h2 = dot(w, w);
        if h2 <= 0.0 {
            return None;
        }
        let r2 = self.radius * self.radius;
        if r2 <= 0.0 {
            return None;
        }
        let direction = ray.direction();
        let dd = dot(direction, direction);
        if dd <= 0.0 {
            return None;
        }

        let h = h2.sqrt();
        let inv_h = 1.0 / h;
        // Unit axis from apex toward the rim.
        let n = scale(w, inv_h);
        // Quadric coefficient: `z = k · ρ²` with `k = h / radius²`.
        let k = h / r2;

        // Ray origin relative to the apex.
        let a = sub(ray.origin(), self.apex);
        let za = dot(a, n);
        let zd = dot(direction, n);
        let ad = dot(a, direction);
        let aa = dot(a, a);

        // `A·t² + B·t + C = 0`, where the perpendicular pieces are
        // `dd − zd²` (⊥ speed²), `ad − za·zd` (⊥ dot), `aa − za²` (⊥ offset²).
        let coeff_a = k * (dd - zd * zd);
        let coeff_b = 2.0 * k * (ad - za * zd) - zd;
        let coeff_c = k * (aa - za * za) - za;

        let t_min = ray.t_min();
        let t_max = ray.t_max();

        // Collect the candidate roots (quadratic, or linear when `A == 0`, i.e.
        // the ray runs parallel to the axis).
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
            // Axial distance from the apex; valid on the dish in `[0, h]`.
            let z = za + t * zd;
            if z < 0.0 || z > h {
                continue;
            }
            // Point relative to the apex and its perpendicular (radial) part.
            let p = [a[0] + t * direction[0], a[1] + t * direction[1], a[2] + t * direction[2]];
            let perp = [p[0] - z * n[0], p[1] - z * n[1], p[2] - z * n[2]];
            // Outward gradient of `k · ρ² − z`: `2k · ρ⃗ − n̂`.
            let grad = [
                2.0 * k * perp[0] - n[0],
                2.0 * k * perp[1] - n[1],
                2.0 * k * perp[2] - n[2],
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
        Some(ParaboloidHit {
            t: best_t,
            primitive: self.primitive,
            normal,
            front_face,
        })
    }
}

/// A ray/paraboloid intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParaboloidHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the paraboloid that was hit.
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

/// A single-level `BVH` over analytic [`Paraboloid`] primitives.
///
/// Empty input yields an empty hierarchy ([`ParaboloidBvh::is_empty`]);
/// traversal of an empty hierarchy simply never reports a hit. The layout and
/// ordered slab walk mirror the triangle [`super::bvh::Bvh`],
/// [`super::cone::ConeBvh`], and [`super::cylinder::CylinderBvh`] so every
/// primitive kind shares one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct ParaboloidBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Paraboloids reordered so each leaf owns a contiguous slice.
    paraboloids: Vec<Paraboloid>,
}

impl ParaboloidBvh {
    /// Builds a `BVH` over `paraboloids` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(paraboloids: &[Paraboloid]) -> Self {
        Self::build_with(paraboloids, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `paraboloids` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each paraboloid's [`Paraboloid::aabb`] and reorders
    /// the paraboloids by the returned primitive order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`ParaboloidBvh::paraboloids`].
    #[must_use]
    pub fn build_with(paraboloids: &[Paraboloid], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = paraboloids.iter().map(Paraboloid::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let paraboloids = order.iter().map(|&i| paraboloids[i as usize]).collect();
        Self { nodes, paraboloids }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of paraboloids in the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.paraboloids.len()
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

    /// Paraboloids in leaf-contiguous order.
    #[must_use]
    pub fn paraboloids(&self) -> &[Paraboloid] {
        &self.paraboloids
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<ParaboloidHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<ParaboloidHit> = None;

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
                    for paraboloid in &self.paraboloids[start..end] {
                        if let Some(hit) = paraboloid.intersect(&ray) {
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

    /// True when *any* paraboloid intersects `ray` inside its interval.
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
                    for paraboloid in &self.paraboloids[start..end] {
                        if paraboloid.intersect(ray).is_some() {
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

    /// Dish opening along `+z` from the apex at the origin to the rim at
    /// `z = 4` with rim radius `2`.
    fn sample_paraboloid(primitive: u32) -> Paraboloid {
        Paraboloid::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 2.0, primitive)
    }

    #[test]
    fn hits_vertex_straight_down_the_axis() {
        let dish = sample_paraboloid(3);
        // Straight down the axis from above the rim toward the apex.
        let ray = Ray::infinite([0.0, 0.0, 6.0], [0.0, 0.0, -1.0]);
        let hit = dish.intersect(&ray).expect("apex hit");
        assert_eq!(hit.primitive, 3);
        // The apex sits at z = 0, six units below the origin.
        assert!(approx(hit.t, 6.0, 1e-3), "t = {}", hit.t);
        // At the apex the outward gradient of `k·ρ² − z` is the downward axis
        // (−z). The ray also travels −z, so it strikes the concave *interior*
        // (back) face: `front_face` is false and the ray-facing shading normal
        // flips to +z to oppose the incident ray.
        assert!(approx(hit.normal[2], 1.0, 1e-3), "normal = {:?}", hit.normal);
        assert!(!hit.front_face);
    }

    #[test]
    fn hits_the_dish_wall_from_the_side() {
        let dish = sample_paraboloid(0);
        // At z = 1 the radius is 2·√(1/4) = 1. Aim inward at that height.
        let ray = Ray::infinite([5.0, 0.0, 1.0], [-1.0, 0.0, 0.0]);
        let hit = dish.intersect(&ray).expect("wall hit");
        assert!(approx(hit.t, 4.0, 1e-3), "t = {}", hit.t);
        // Outward wall normal leans toward +x (radially outward).
        assert!(hit.normal[0] > 0.0, "normal = {:?}", hit.normal);
        assert!(hit.front_face);
    }

    #[test]
    fn misses_beyond_the_rim_radius() {
        let dish = sample_paraboloid(0);
        // Parallel to the axis but offset past the rim radius (2).
        let ray = Ray::infinite([3.0, 0.0, -5.0], [0.0, 0.0, 1.0]);
        assert!(dish.intersect(&ray).is_none());
    }

    #[test]
    fn misses_below_the_vertex() {
        let dish = sample_paraboloid(0);
        // On-axis ray heading away from the dish never reaches z ∈ [0, h].
        let ray = Ray::infinite([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        assert!(dish.intersect(&ray).is_none());
    }

    #[test]
    fn degenerate_axis_never_hits() {
        let dish = Paraboloid::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(dish.intersect(&ray).is_none());
    }

    #[test]
    fn zero_radius_never_hits() {
        let dish = Paraboloid::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 0.0, 0);
        let ray = Ray::infinite([0.0, 0.0, 6.0], [0.0, 0.0, -1.0]);
        assert!(dish.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let dish = sample_paraboloid(0);
        let ray = Ray::new([0.0, 0.0, 6.0], [0.0, 0.0, -1.0], 0.0, 5.0);
        assert!(dish.intersect(&ray).is_none());
        let ray = Ray::new([0.0, 0.0, 6.0], [0.0, 0.0, -1.0], 0.0, 7.0);
        assert!(dish.intersect(&ray).is_some());
    }

    fn random_paraboloid(rng: &mut Rng, primitive: u32) -> Paraboloid {
        let apex = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        let top = [
            apex[0] + rng.range(-3.0, 3.0),
            apex[1] + rng.range(-3.0, 3.0),
            apex[2] + rng.range(0.5, 3.0),
        ];
        Paraboloid::new(apex, top, rng.range(0.3, 1.5), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Paraboloid> {
        (0..count).map(|i| random_paraboloid(rng, i)).collect()
    }

    fn brute_closest(dishes: &[Paraboloid], ray: &Ray) -> Option<ParaboloidHit> {
        let mut best: Option<ParaboloidHit> = None;
        let mut ray = *ray;
        for dish in dishes {
            if let Some(hit) = dish.intersect(&ray) {
                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                best = Some(hit);
            }
        }
        best
    }

    #[test]
    fn reported_hit_lies_on_the_quadric() {
        // Independent ground truth: every reported hit must satisfy the
        // implicit quadric `k·ρ² = z` with `z ∈ [0, h]`, guarding against a
        // coefficient transcription bug (a spurious root off the surface).
        let mut rng = Rng::new(0xBADC_0FFE_E0DD_F00Du64);
        let dishes = random_scene(&mut rng, 32);
        let mut checked = 0u32;
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
            for dish in &dishes {
                if let Some(hit) = dish.intersect(&ray) {
                    let w = sub(dish.top(), dish.apex());
                    let h = dot(w, w).sqrt();
                    let n = scale(w, 1.0 / h);
                    let k = h / (dish.radius() * dish.radius());
                    let p = ray.at(hit.t);
                    let rel = sub(p, dish.apex());
                    let z = dot(rel, n);
                    let perp2 = dot(rel, rel) - z * z;
                    assert!(z >= -1e-3 && z <= h + 1e-3, "z = {z} h = {h}");
                    // Residual scaled by the surface extent to stay unitless.
                    let residual = (k * perp2 - z).abs() / (h + 1.0);
                    assert!(residual < 5e-3, "off-surface residual = {residual}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no hits exercised the on-surface check");
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = ParaboloidBvh::build(&[]);
        assert!(bvh.is_empty());
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0x0d15_c000_9abc_1234u64);
        let dishes = random_scene(&mut rng, 64);
        let bvh = ParaboloidBvh::build(&dishes);

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

            let expected = brute_closest(&dishes, &ray);
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
        let dishes = random_scene(&mut rng, 48);
        let bvh = ParaboloidBvh::build(&dishes);

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
            assert_eq!(bvh.any_hit(&ray), brute_closest(&dishes, &ray).is_some());
        }
    }
}
