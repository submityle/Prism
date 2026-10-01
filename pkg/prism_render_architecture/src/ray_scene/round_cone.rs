//! Analytic *round cone* (unequal-radius sphere-swept segment) primitive and
//! its single-level `BVH`.
//!
//! A round cone is the convex hull of two spheres: a sphere of radius `radius_a`
//! at endpoint `a` and a sphere of radius `radius_b` at endpoint `b`. Its
//! surface is a lateral cone band tangent to both spheres, closed by a spherical
//! cap at each end. When `radius_a == radius_b` it degenerates exactly to a
//! [`super::capsule::Capsule`]; it is the standard proxy for *tapered* shapes —
//! limbs, horns, tree branches, tapered cables and tusks — so a path tracer
//! wants a closed-form test rather than a tessellated taper.
//!
//! Like [`super::cylinder`], [`super::cone`] and [`super::capsule`], this is a
//! procedural primitive for the `DXR`/Vulkan `AABB` path: the `BLAS` stores one
//! axis-aligned box per round cone and an intersection shader refines the hit.
//!
//! The test intersects three analytic pieces and keeps the nearest valid root
//! inside the ray interval:
//! - the infinite cone that is externally tangent to both spheres (apex on the
//!   axis, half-angle `α` with `sin α = (radius_a − radius_b) / |b − a|`),
//!   clipped to the axial band between the two tangent circles, and
//! - the two endpoint spheres, each clipped to the cap region its tangent circle
//!   does not cover.
//!
//! Every `t²` coefficient carries `dd = ⟨d, d⟩`, so the test is correct for the
//! non-unit ray directions `ray_scene` feeds it, and every step is
//! add/sub/mul/div/`sqrt` and comparisons — bit-reproducible on the `GPU` and
//! free of any transcendental call. Degenerate inputs are handled explicitly: a
//! zero-length axis, or an axis shorter than the radius difference (one sphere
//! swallows the other), reduces to a single sphere at the larger endpoint.

use super::bvh::{build_linear_bvh, Aabb, BvhBuildConfig, LinearBvhNode};
use super::traversal::Ray;

/// An analytic round cone (sphere-swept segment with unequal end radii).
///
/// The solid is the convex hull of a `radius_a`-sphere at `a` and a
/// `radius_b`-sphere at `b`. `primitive` is the caller's stable id (mirroring
/// [`super::bvh::Triangle`] and [`super::capsule::Capsule`]): the
/// [`RoundConeBvh`] builder reorders primitives internally but always reports
/// hits by this id. Both radii are stored non-negative; caller-supplied negative
/// radii are folded to their magnitude so the derived [`Aabb`] and the quadrics
/// stay well formed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundCone {
    /// First segment endpoint (center of the `a`-side sphere).
    a: [f32; 3],
    /// Second segment endpoint (center of the `b`-side sphere).
    b: [f32; 3],
    /// Non-negative sphere radius at endpoint `a`.
    radius_a: f32,
    /// Non-negative sphere radius at endpoint `b`.
    radius_b: f32,
    /// Caller's stable primitive id, reported unchanged on every hit.
    primitive: u32,
}

impl RoundCone {
    /// Builds a round cone from the `radius_a`-sphere at `a` to the
    /// `radius_b`-sphere at `b` (both radii folded to their magnitude), tagged
    /// with stable id `primitive`.
    #[must_use]
    pub fn new(a: [f32; 3], b: [f32; 3], radius_a: f32, radius_b: f32, primitive: u32) -> Self {
        Self {
            a,
            b,
            radius_a: radius_a.abs(),
            radius_b: radius_b.abs(),
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

    /// Non-negative sphere radius at `a`.
    #[must_use]
    pub fn radius_a(&self) -> f32 {
        self.radius_a
    }

    /// Non-negative sphere radius at `b`.
    #[must_use]
    pub fn radius_b(&self) -> f32 {
        self.radius_b
    }

    /// Caller's stable primitive id.
    #[must_use]
    pub fn primitive(&self) -> u32 {
        self.primitive
    }

    /// Tight axis-aligned bounds of the round cone.
    ///
    /// This is the procedural-primitive `AABB` the hardware `BLAS` stores. The
    /// solid is the convex hull of the two end spheres, so on every axis the
    /// bound is the union of each endpoint's extent padded by that endpoint's
    /// radius — tight everywhere.
    #[must_use]
    pub fn aabb(&self) -> Aabb {
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for (axis, slot) in min.iter_mut().zip(max.iter_mut()).enumerate() {
            let lo_a = self.a[axis] - self.radius_a;
            let hi_a = self.a[axis] + self.radius_a;
            let lo_b = self.b[axis] - self.radius_b;
            let hi_b = self.b[axis] + self.radius_b;
            *slot.0 = lo_a.min(lo_b);
            *slot.1 = hi_a.max(hi_b);
        }
        Aabb::new(min, max)
    }

    /// Nearest ray/round-cone intersection inside `ray`'s `[t_min, t_max]`
    /// interval, or `None` when the ray misses.
    ///
    /// [`RoundConeHit::normal`] is the unit surface normal oriented *against*
    /// the incident ray, and [`RoundConeHit::front_face`] is `true` when the ray
    /// struck the outward-facing side. A zero-length ray direction never reports
    /// a hit; a zero-length axis (or one shorter than `|radius_a − radius_b|`)
    /// reduces to a single sphere at the larger endpoint.
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<RoundConeHit> {
        let direction = ray.direction();
        let dd = dot(direction, direction);
        if dd <= 0.0 {
            return None;
        }
        if self.radius_a <= 0.0 && self.radius_b <= 0.0 {
            return None;
        }

        let t_min = ray.t_min();
        let t_max = ray.t_max();
        let origin = ray.origin();

        let w = sub(self.b, self.a);
        let l2 = dot(w, w);
        let rr = self.radius_a - self.radius_b;

        // Best (t, outward-normal) found so far across every piece. `consider`
        // keeps the nearest in-interval root; it is the sole mutable borrow of
        // `best_t`/`best_outward`, so the free `sphere_roots`/`cone_roots`
        // helpers return their roots by value and never alias it.
        let mut best_t = f32::INFINITY;
        let mut best_outward = [0.0f32; 3];
        let mut consider = |t: f32, outward: [f32; 3]| {
            if !(t >= t_min && t <= t_max) || t >= best_t {
                return;
            }
            best_t = t;
            best_outward = outward;
        };

        if l2 <= rr * rr {
            // One sphere engulfs the other (or the axis is zero): the solid is
            // just the larger end sphere.
            let (center, radius) = if self.radius_a >= self.radius_b {
                (self.a, self.radius_a)
            } else {
                (self.b, self.radius_b)
            };
            emit_sphere(origin, direction, dd, center, radius, &mut consider);
            return finish(best_t, best_outward, direction, self.primitive);
        }

        let inv_l = 1.0 / l2.sqrt();
        let n = scale(w, inv_l);
        let l = l2 * inv_l;

        if rr == 0.0 {
            // Equal radii: the lateral band is a straight cylinder of radius
            // `radius_a`; this is exactly the capsule construction.
            self.intersect_cylinder(origin, direction, dd, n, l, &mut consider);
        } else {
            // Unequal radii: a true tangent cone band plus clipped end spheres.
            self.intersect_tapered(origin, direction, dd, n, l, rr, &mut consider);
        }

        finish(best_t, best_outward, direction, self.primitive)
    }

    /// Equal-radius lateral band: an infinite cylinder of radius `radius_a`
    /// clipped to `axial ∈ [0, l]`, plus hemispheres clipped at the band ends.
    fn intersect_cylinder(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        dd: f32,
        n: [f32; 3],
        l: f32,
        consider: &mut impl FnMut(f32, [f32; 3]),
    ) {
        let r = self.radius_a;
        let r2 = r * r;
        let oa = sub(origin, self.a);
        let za = dot(oa, n);
        let zd = dot(direction, n);
        let ad = dot(oa, direction);
        let aa = dot(oa, oa);

        let coeff_a = dd - zd * zd;
        let coeff_b = 2.0 * (ad - za * zd);
        let coeff_c = aa - za * za - r2;
        if coeff_a != 0.0 {
            let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let inv_2a = 1.0 / (2.0 * coeff_a);
                for t in [
                    (-coeff_b - sqrt_disc) * inv_2a,
                    (-coeff_b + sqrt_disc) * inv_2a,
                ] {
                    let z = za + t * zd;
                    if z >= 0.0 && z <= l {
                        let axis_point = [
                            self.a[0] + z * n[0],
                            self.a[1] + z * n[1],
                            self.a[2] + z * n[2],
                        ];
                        let p = point_at(origin, direction, t);
                        let radial = sub(p, axis_point);
                        let len2 = dot(radial, radial);
                        if len2 > 0.0 {
                            consider(t, scale(radial, 1.0 / len2.sqrt()));
                        }
                    }
                }
            }
        }

        // Hemisphere at `a` (band start, `axial ≤ 0`).
        if let Some((t0, t1)) = sphere_roots(oa, direction, dd, r2) {
            for t in [t0, t1] {
                let p = point_at(origin, direction, t);
                if dot(sub(p, self.a), n) <= 0.0 {
                    let radial = sub(p, self.a);
                    let len2 = dot(radial, radial);
                    if len2 > 0.0 {
                        consider(t, scale(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }

        // Hemisphere at `b` (band end, `axial ≥ l`).
        let ob = sub(origin, self.b);
        if let Some((t0, t1)) = sphere_roots(ob, direction, dd, r2) {
            for t in [t0, t1] {
                let p = point_at(origin, direction, t);
                if dot(sub(p, self.a), n) >= l {
                    let radial = sub(p, self.b);
                    let len2 = dot(radial, radial);
                    if len2 > 0.0 {
                        consider(t, scale(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }
    }

    /// Unequal-radius lateral band: the external tangent cone clipped to the
    /// axial band between the two tangent circles, plus the end spheres clipped
    /// to the caps those circles leave exposed.
    ///
    /// `sin α = rr / l` is the cone half-angle sine (`rr = radius_a − radius_b`,
    /// signed), guaranteed `|sin α| < 1` because the caller only reaches this
    /// branch when `l² > rr²`. The tangent circles sit at axial coordinates
    /// `radius_a · sin α` (from `a`) and `l + radius_b · sin α`; the lateral
    /// surface spans between them and the spheres own the rest.
    fn intersect_tapered(
        &self,
        origin: [f32; 3],
        direction: [f32; 3],
        dd: f32,
        n: [f32; 3],
        l: f32,
        rr: f32,
        consider: &mut impl FnMut(f32, [f32; 3]),
    ) {
        let sin_a = rr / l;
        let cos_a = (1.0 - sin_a * sin_a).sqrt();
        let band_lo = self.radius_a * sin_a;
        let band_hi = l + self.radius_b * sin_a;

        // Tangent-cone lateral band, solved **without** the apex. The apex sits
        // at axial `radius_a / sin α`, which blows up as `sin α → 0`; building
        // the quadratic from `origin − apex` then subtracts near-equal large
        // magnitudes and loses all precision for shallow tapers. Instead we use
        // the apex-free implicit form, with `q = P − a`, `axial = ⟨q, n⟩`, and
        // radial distance `rad` (so `rad² = ⟨q, q⟩ − axial²`):
        //
        //   rad · cos α = radius_a − axial · sin α   (valid where RHS ≥ 0)
        //   ⇒ cos²α · (⟨q, q⟩ − axial²) = (radius_a − axial · sin α)²
        //
        // Substituting `P = origin + t · direction` yields a quadratic whose
        // coefficients stay `O(radius, length)` for every taper angle. Each
        // `t²` term carries `dd = ⟨direction, direction⟩`, so the ray direction
        // need not be unit.
        let oa = sub(origin, self.a);
        let za = dot(oa, n);
        let zd = dot(direction, n);
        let ad = dot(oa, direction);
        let aa = dot(oa, oa);
        let k = cos_a * cos_a;
        // RHS linear form `radius_a − axial · sin α = rhs0 + t · rhs1`.
        let rhs0 = self.radius_a - sin_a * za;
        let rhs1 = -sin_a * zd;
        let coeff_a = k * (dd - zd * zd) - rhs1 * rhs1;
        let coeff_b = k * (ad - za * zd) - rhs0 * rhs1;
        let coeff_c = k * (aa - za * za) - rhs0 * rhs0;
        if coeff_a != 0.0 {
            // `coeff_b` here is the half-`b` form (`b/2`), so the discriminant
            // and roots drop the factor of two: `t = (−b/2 ± √disc) / a`.
            let disc = coeff_b * coeff_b - coeff_a * coeff_c;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let inv_a = 1.0 / coeff_a;
                for t in [(-coeff_b - sqrt_disc) * inv_a, (-coeff_b + sqrt_disc) * inv_a] {
                    let axial = za + t * zd;
                    if !(axial >= band_lo && axial <= band_hi) {
                        continue;
                    }
                    // Correct nappe: squaring admits the mirror cone where
                    // `rad · cos α = −(radius_a − axial · sin α)`. Keep only the
                    // physical sheet, where `radius_a − axial · sin α ≥ 0`.
                    if rhs0 + t * rhs1 < 0.0 {
                        continue;
                    }
                    let p = point_at(origin, direction, t);
                    let radial = [
                        p[0] - self.a[0] - axial * n[0],
                        p[1] - self.a[1] - axial * n[1],
                        p[2] - self.a[2] - axial * n[2],
                    ];
                    let len2 = dot(radial, radial);
                    if len2 <= 0.0 {
                        continue;
                    }
                    let radial_u = scale(radial, 1.0 / len2.sqrt());
                    // Outward normal: a cone surface's outward normal tilts off
                    // the radial toward the *narrow* end (the apex) by the cone
                    // half-angle α. With signed `sin α = rr / l` and axis `n`
                    // oriented `a → b`, that tilt is `+ sin α · n` for both
                    // taper senses (verified against the IQ round-cone SDF
                    // gradient). Unit by construction (`cos²α + sin²α = 1`).
                    let outward = [
                        cos_a * radial_u[0] + sin_a * n[0],
                        cos_a * radial_u[1] + sin_a * n[1],
                        cos_a * radial_u[2] + sin_a * n[2],
                    ];
                    consider(t, outward);
                }
            }
        }

        // Sphere at `a`, clipped to the cap below the tangent circle (`oa`
        // was computed above for the lateral band).
        if let Some((t0, t1)) = sphere_roots(oa, direction, dd, self.radius_a * self.radius_a) {
            for t in [t0, t1] {
                let p = point_at(origin, direction, t);
                if dot(sub(p, self.a), n) <= band_lo {
                    let radial = sub(p, self.a);
                    let len2 = dot(radial, radial);
                    if len2 > 0.0 {
                        consider(t, scale(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }

        // Sphere at `b`, clipped to the cap above the tangent circle.
        let ob = sub(origin, self.b);
        if let Some((t0, t1)) = sphere_roots(ob, direction, dd, self.radius_b * self.radius_b) {
            for t in [t0, t1] {
                let p = point_at(origin, direction, t);
                if dot(sub(p, self.b), n) >= self.radius_b * sin_a {
                    let radial = sub(p, self.b);
                    let len2 = dot(radial, radial);
                    if len2 > 0.0 {
                        consider(t, scale(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }
    }
}

/// Emits both roots of a sphere of radius `radius` centered at `center` to
/// `consider`, with the outward normal `normalize(p − center)`.
fn emit_sphere(
    origin: [f32; 3],
    direction: [f32; 3],
    dd: f32,
    center: [f32; 3],
    radius: f32,
    consider: &mut impl FnMut(f32, [f32; 3]),
) {
    let rel = sub(origin, center);
    if let Some((t0, t1)) = sphere_roots(rel, direction, dd, radius * radius) {
        for t in [t0, t1] {
            let p = point_at(origin, direction, t);
            let radial = sub(p, center);
            let len2 = dot(radial, radial);
            if len2 > 0.0 {
                consider(t, scale(radial, 1.0 / len2.sqrt()));
            }
        }
    }
}

/// Finalizes the best candidate into a [`RoundConeHit`], orienting the normal
/// against the incident ray, or `None` when nothing was hit.
fn finish(
    best_t: f32,
    best_outward: [f32; 3],
    direction: [f32; 3],
    primitive: u32,
) -> Option<RoundConeHit> {
    if !best_t.is_finite() {
        return None;
    }
    let front_face = dot(direction, best_outward) < 0.0;
    let normal = if front_face {
        best_outward
    } else {
        [-best_outward[0], -best_outward[1], -best_outward[2]]
    };
    Some(RoundConeHit {
        t: best_t,
        primitive,
        normal,
        front_face,
    })
}

/// Solves the sphere quadratic `|rel + t·d|² = r²`, returning its two roots
/// (equal when the ray grazes the sphere) or `None` when the ray misses.
///
/// `rel` is the ray origin relative to the sphere center. Kept as a free helper
/// so every cap and the degenerate single-sphere case share identical root
/// arithmetic; callers clip the roots to their cap region and the ray interval.
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

/// World hit position `origin + t·direction`.
fn point_at(origin: [f32; 3], direction: [f32; 3], t: f32) -> [f32; 3] {
    [
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    ]
}

/// A ray/round-cone intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundConeHit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Stable id of the round cone that was hit.
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

/// A single-level `BVH` over analytic [`RoundCone`] primitives.
///
/// Empty input yields an empty hierarchy ([`RoundConeBvh::is_empty`]); traversal
/// of an empty hierarchy simply never reports a hit. The layout and ordered slab
/// walk mirror the triangle [`super::bvh::Bvh`] and [`super::capsule::CapsuleBvh`]
/// so every primitive kind shares one acceleration-structure contract.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundConeBvh {
    /// Flattened `BVH` nodes; the root (when present) is index `0`.
    nodes: Vec<LinearBvhNode>,
    /// Round cones reordered so each leaf owns a contiguous slice.
    cones: Vec<RoundCone>,
}

impl RoundConeBvh {
    /// Builds a `BVH` over `cones` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(cones: &[RoundCone]) -> Self {
        Self::build_with(cones, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `cones` with the given binned-`SAH` `config`.
    ///
    /// The builder runs over each round cone's [`RoundCone::aabb`] and reorders
    /// the primitives by the returned order so every leaf's
    /// `[first_primitive, first_primitive + primitive_count)` slice indexes
    /// directly into [`RoundConeBvh::cones`].
    #[must_use]
    pub fn build_with(cones: &[RoundCone], config: BvhBuildConfig) -> Self {
        let bounds: Vec<Aabb> = cones.iter().map(RoundCone::aabb).collect();
        let (nodes, order) = build_linear_bvh(&bounds, config);
        let cones = order.iter().map(|&i| cones[i as usize]).collect();
        Self { nodes, cones }
    }

    /// Number of flattened `BVH` nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of round cones in the hierarchy.
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

    /// Round cones in leaf-contiguous order.
    #[must_use]
    pub fn cones(&self) -> &[RoundCone] {
        &self.cones
    }

    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<RoundConeHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<RoundConeHit> = None;

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

    /// True when *any* round cone intersects `ray` inside its interval.
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

    /// Branchy sign helper (transcendental-free) for the reference `SDF`.
    fn sgn(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// Reference round-cone signed distance (Inigo Quilez), used as an
    /// independent oracle: a reported hit point must satisfy `|sdf| ≈ 0`. It
    /// shares no arithmetic with [`RoundCone::intersect`], so it catches a
    /// mistranscribed cone/cap coefficient a `BVH`↔brute cross-check cannot.
    fn sd_round_cone(p: [f32; 3], a: [f32; 3], b: [f32; 3], r1: f32, r2: f32) -> f32 {
        let ba = sub(b, a);
        let l2 = dot(ba, ba);
        let rr = r1 - r2;
        let a2 = l2 - rr * rr;
        let il2 = 1.0 / l2;
        let pa = sub(p, a);
        let y = dot(pa, ba);
        let z = y - l2;
        let x_vec = [
            pa[0] * l2 - ba[0] * y,
            pa[1] * l2 - ba[1] * y,
            pa[2] * l2 - ba[2] * y,
        ];
        let x2 = dot(x_vec, x_vec);
        let y2 = y * y * l2;
        let z2 = z * z * l2;
        let k = sgn(rr) * rr * rr * x2;
        if sgn(z) * a2 * z2 > k {
            return (x2 + z2).sqrt() * il2 - r2;
        }
        if sgn(y) * a2 * y2 < k {
            return (x2 + y2).sqrt() * il2 - r1;
        }
        ((x2 * a2 * il2).sqrt() + y * rr) * il2 - r1
    }

    /// A capsule-like round cone from `(0,0,0)` to `(0,0,4)`, radius `1` at both
    /// ends.
    fn equal_sample(primitive: u32) -> RoundCone {
        RoundCone::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 1.0, 1.0, primitive)
    }

    #[test]
    fn equal_radii_match_a_capsule_body() {
        let cone = equal_sample(7);
        // Side-on ray at the body midpoint: near wall at x = 1.
        let ray = Ray::infinite([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0]);
        let hit = cone.intersect(&ray).expect("body hit");
        assert_eq!(hit.primitive, 7);
        assert!(approx(hit.t, 4.0, 1e-3), "t = {}", hit.t);
        assert!(approx(hit.normal[0], 1.0, 1e-3), "normal = {:?}", hit.normal);
        assert!(hit.front_face);
    }

    #[test]
    fn tapered_body_normal_tilts_toward_the_narrow_end() {
        // Wide at `a` (r = 2), narrow at `b` (r = 0.5), axis along +z, length 4.
        let cone = RoundCone::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 2.0, 0.5, 3);
        // Aim at the lateral band from +x near the middle.
        let ray = Ray::infinite([6.0, 0.0, 2.0], [-1.0, 0.0, 0.0]);
        let hit = cone.intersect(&ray).expect("band hit");
        // A cone wall's outward normal tilts toward the narrow end: here the
        // taper narrows toward `b` (+z), so sin α = (2 - 0.5)/4 = 0.375 gives a
        // +z axial component (matches the IQ round-cone SDF gradient). The ray
        // travels −x, so the ray-facing normal keeps the +x, +z outward sense.
        assert!(hit.normal[0] > 0.0, "normal = {:?}", hit.normal);
        assert!(hit.normal[2] > 0.0, "normal should tilt toward narrow end (+z)");
        let len = (hit.normal[0] * hit.normal[0]
            + hit.normal[1] * hit.normal[1]
            + hit.normal[2] * hit.normal[2])
            .sqrt();
        assert!(approx(len, 1.0, 1e-3), "normal not unit: {len}");
    }

    #[test]
    fn hits_the_wide_end_sphere_cap() {
        let cone = RoundCone::new([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], 2.0, 0.5, 0);
        // Straight down the axis: the `a` cap apex sits at z = -2.
        let ray = Ray::infinite([0.0, 0.0, -10.0], [0.0, 0.0, 1.0]);
        let hit = cone.intersect(&ray).expect("cap hit");
        assert!(approx(hit.t, 8.0, 1e-3), "t = {}", hit.t);
        assert!(approx(hit.normal[2], -1.0, 1e-3), "normal = {:?}", hit.normal);
    }

    #[test]
    fn engulfed_axis_is_a_single_sphere() {
        // Axis length 1 but radius difference 2 → the big sphere swallows the
        // small one; the solid is the r = 3 sphere at `a`.
        let cone = RoundCone::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 3.0, 1.0, 4);
        let ray = Ray::infinite([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let hit = cone.intersect(&ray).expect("sphere hit");
        assert!(approx(hit.t, 7.0, 1e-3), "t = {}", hit.t);
    }

    #[test]
    fn zero_direction_never_hits() {
        let cone = equal_sample(0);
        let ray = Ray::infinite([5.0, 0.0, 2.0], [0.0, 0.0, 0.0]);
        assert!(cone.intersect(&ray).is_none());
    }

    #[test]
    fn t_max_excludes_far_hit() {
        let cone = equal_sample(0);
        let near = Ray::new([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0], 0.0, 3.9);
        assert!(cone.intersect(&near).is_none());
        let far = Ray::new([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0], 0.0, 4.1);
        assert!(cone.intersect(&far).is_some());
    }

    /// Independent surface + normal check: every reported hit point must satisfy
    /// the reference `SDF` ≈ 0, and the analytic normal must be (anti)parallel to
    /// the `SDF` gradient (central difference).
    #[test]
    fn reported_hit_lies_on_the_swept_surface() {
        let mut rng = Rng::new(0xC0FF_EE42);
        let mut checked = 0u32;
        for _ in 0..8_000 {
            let a = [rng.range(-4.0, 4.0), rng.range(-4.0, 4.0), rng.range(-4.0, 4.0)];
            // Keep the axis comfortably longer than the radius gap so this stays
            // a proper (non-engulfed) round cone the SDF oracle also models.
            let b = [
                a[0] + rng.range(-3.0, 3.0),
                a[1] + rng.range(-3.0, 3.0),
                a[2] + rng.range(1.5, 3.0),
            ];
            let r1 = rng.range(0.3, 1.5);
            let r2 = rng.range(0.3, 1.5);
            let cone = RoundCone::new(a, b, r1, r2, 0);
            let ba = sub(b, a);
            let l = dot(ba, ba).sqrt();
            if l <= (r1 - r2).abs() + 0.3 {
                continue;
            }

            // Aim each ray at a jittered point around the segment so the sweep
            // is actually exercised (purely random directions almost always
            // miss a unit-scale primitive). The jitter radius comfortably
            // exceeds the fattest end, so rays sweep across the body, both
            // caps, and the empty space just outside — covering hits and misses
            // alike while keeping a high surface-hit yield.
            let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
            let jitter = r1.max(r2) + 0.5 * l + 0.6;
            let target = [
                mid[0] + rng.range(-jitter, jitter),
                mid[1] + rng.range(-jitter, jitter),
                mid[2] + rng.range(-jitter, jitter),
            ];
            let origin = [
                mid[0] + rng.range(-12.0, 12.0),
                mid[1] + rng.range(-12.0, 12.0),
                mid[2] + rng.range(-12.0, 12.0),
            ];
            let dir = sub(target, origin);
            if dot(dir, dir) < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let Some(hit) = cone.intersect(&ray) else {
                continue;
            };

            let p = ray.at(hit.t);
            let d = sd_round_cone(p, a, b, r1, r2);
            assert!(
                d.abs() / (r1.max(r2) + 1.0) < 6e-3,
                "off-surface: sdf {d} for r1 {r1} r2 {r2} l {l}"
            );

            // SDF gradient by central difference → outward normal.
            let e = 2e-3;
            let gx = sd_round_cone([p[0] + e, p[1], p[2]], a, b, r1, r2)
                - sd_round_cone([p[0] - e, p[1], p[2]], a, b, r1, r2);
            let gy = sd_round_cone([p[0], p[1] + e, p[2]], a, b, r1, r2)
                - sd_round_cone([p[0], p[1] - e, p[2]], a, b, r1, r2);
            let gz = sd_round_cone([p[0], p[1], p[2] + e], a, b, r1, r2)
                - sd_round_cone([p[0], p[1], p[2] - e], a, b, r1, r2);
            let grad = [gx, gy, gz];
            let gl = dot(grad, grad).sqrt();
            if gl < 1e-5 {
                continue;
            }
            let gu = scale(grad, 1.0 / gl);
            // Reported normal faces the ray; |dot| with the outward gradient ≈ 1.
            let align = dot(gu, hit.normal).abs();
            assert!(align > 0.95, "normal/gradient misaligned: {align}");
            checked += 1;
        }
        assert!(checked > 500, "too few samples exercised: {checked}");
    }

    fn random_cone(rng: &mut Rng, primitive: u32) -> RoundCone {
        let a = [rng.range(-5.0, 5.0), rng.range(-5.0, 5.0), rng.range(-5.0, 5.0)];
        let b = [
            a[0] + rng.range(-3.0, 3.0),
            a[1] + rng.range(-3.0, 3.0),
            a[2] + rng.range(-3.0, 3.0),
        ];
        RoundCone::new(a, b, rng.range(0.3, 1.5), rng.range(0.3, 1.5), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<RoundCone> {
        (0..count).map(|i| random_cone(rng, i)).collect()
    }

    #[test]
    fn empty_bvh_never_hits() {
        let bvh = RoundConeBvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn bvh_closest_hit_matches_brute_force_bit_for_bit() {
        let mut rng = Rng::new(0xBADC_0DE1);
        let scene = random_scene(&mut rng, 48);
        let bvh = RoundConeBvh::build(&scene);
        let ordered = bvh.cones().to_vec();

        for _ in 0..3_000 {
            let origin = [rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dot(dir, dir) < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            let mut brute: Option<RoundConeHit> = None;
            let mut r = ray;
            for cone in &ordered {
                if let Some(hit) = cone.intersect(&r) {
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
        let mut rng = Rng::new(0x0B0E_FEED);
        let scene = random_scene(&mut rng, 48);
        let bvh = RoundConeBvh::build(&scene);
        let ordered = bvh.cones().to_vec();

        for _ in 0..3_000 {
            let origin = [rng.range(-10.0, 10.0), rng.range(-10.0, 10.0), rng.range(-10.0, 10.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            if dot(dir, dir) < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let brute = ordered.iter().any(|c| c.intersect(&ray).is_some());
            assert_eq!(brute, bvh.any_hit(&ray));
        }
    }
}
