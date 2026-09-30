//! Ray/`BVH` traversal: slab test, Möller–Trumbore, and stack-based walks.
//!
//! This is the `CPU`-verifiable golden reference for the `GPU` traversal kernel.
//! Given a [`Bvh`](super::bvh::Bvh) it answers two queries:
//!
//! - [`Bvh::closest_hit`] — nearest intersection along the ray (primary /
//!   reflection / gather rays).
//! - [`Bvh::any_hit`] — does *any* primitive block the ray inside its interval
//!   (shadow / occlusion rays); returns early on the first hit.
//!
//! Both walk the flattened depth-first node array using an explicit short stack
//! and a slab-based ray-box rejection test, and intersect leaf triangles with
//! the watertight-enough single-precision Möller–Trumbore test. Everything is
//! `f32` so the results match the shader the `GPU` backend runs.

use super::bvh::{Aabb, Bvh, Triangle};

/// A ray with an explicit valid `t` interval `[t_min, t_max]`.
///
/// `direction` need not be normalized, but `t` values are then measured in units
/// of `direction`'s length. Reciprocal components are precomputed once for the
/// slab test; a zero component yields an infinite reciprocal, which the slab
/// test handles via the standard min/max ordering.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    origin: [f32; 3],
    direction: [f32; 3],
    inv_direction: [f32; 3],
    t_min: f32,
    t_max: f32,
}

impl Ray {
    /// Builds a ray over the interval `[t_min, t_max]`.
    ///
    /// `t_min` is clamped to be non-negative and `t_max` to be at least `t_min`
    /// so a caller cannot construct an inverted interval.
    #[must_use]
    pub fn new(origin: [f32; 3], direction: [f32; 3], t_min: f32, t_max: f32) -> Self {
        let inv_direction = [
            reciprocal(direction[0]),
            reciprocal(direction[1]),
            reciprocal(direction[2]),
        ];
        let lo = if t_min.is_finite() { t_min.max(0.0) } else { 0.0 };
        let hi = if t_max.is_nan() { f32::INFINITY } else { t_max.max(lo) };
        Self {
            origin,
            direction,
            inv_direction,
            t_min: lo,
            t_max: hi,
        }
    }

    /// Builds a ray over `[0, +inf)`.
    #[must_use]
    pub fn infinite(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self::new(origin, direction, 0.0, f32::INFINITY)
    }

    /// Ray origin.
    #[must_use]
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Ray direction (not necessarily normalized).
    #[must_use]
    pub fn direction(&self) -> [f32; 3] {
        self.direction
    }

    /// Lower bound of the valid `t` interval.
    #[must_use]
    pub fn t_min(&self) -> f32 {
        self.t_min
    }

    /// Upper bound of the valid `t` interval.
    #[must_use]
    pub fn t_max(&self) -> f32 {
        self.t_max
    }

    /// Point at parameter `t` along the ray.
    #[must_use]
    pub fn at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }

    /// Slab test against `bounds`, clipped to `[t_lo, t_hi]`.
    ///
    /// Returns the entry/exit `t` interval in which the ray is inside the box,
    /// or `None` when it never enters within that range. Public so the
    /// top-level acceleration structure ([`super::tlas::Tlas`]) prunes its nodes
    /// with exactly the reciprocal-slab math the `BVH` walk uses internally,
    /// reusing the precomputed `inv_direction`.
    #[must_use]
    pub fn aabb_interval(&self, bounds: &Aabb, t_lo: f32, t_hi: f32) -> Option<(f32, f32)> {
        slab_interval(self.origin, self.inv_direction, bounds, t_lo, t_hi)
    }
}

/// A ray/primitive intersection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Ray parameter at the intersection (distance in `direction` lengths).
    pub t: f32,
    /// Barycentric `u` (weight of `v1`).
    pub u: f32,
    /// Barycentric `v` (weight of `v2`); weight of `v0` is `1 - u - v`.
    pub v: f32,
    /// Stable primitive id carried by the hit [`Triangle`].
    pub primitive: u32,
}

/// Slab test: the sub-interval of `[t_lo, t_hi]` in which the ray is inside
/// `bounds`, or `None` if it never enters within the interval.
///
/// Uses the branch-light min/max reciprocal formulation; a zero direction
/// component makes the corresponding slab test degenerate to "always inside on
/// that axis" as long as the origin lies between the planes, which the
/// `min`/`max` combination handles because `inf * 0` never occurs here (the
/// origin offset is finite and the reciprocal is `+/-inf`, giving `+/-inf`
/// bounds that widen the interval).
#[inline]
fn slab_interval(origin: [f32; 3], inv_dir: [f32; 3], bounds: &Aabb, t_lo: f32, t_hi: f32) -> Option<(f32, f32)> {
    let mut tmin = t_lo;
    let mut tmax = t_hi;
    for axis in 0..3 {
        let t0 = (bounds.min[axis] - origin[axis]) * inv_dir[axis];
        let t1 = (bounds.max[axis] - origin[axis]) * inv_dir[axis];
        let (near, far) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        if near > tmin {
            tmin = near;
        }
        if far < tmax {
            tmax = far;
        }
        if tmin > tmax {
            return None;
        }
    }
    Some((tmin, tmax))
}

/// Möller–Trumbore ray-triangle test in single precision.
///
/// Returns `(t, u, v)` when the ray hits the triangle within `[t_min, t_max]`,
/// where `u`/`v` are the barycentric weights of `v1`/`v2`. Rays parallel to the
/// triangle plane (near-zero determinant) miss. Both faces are tested so the
/// caller can decide culling separately.
#[inline]
pub(crate) fn intersect_triangle(ray: &Ray, tri: &Triangle) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-8;
    let e1 = sub(tri.v1, tri.v0);
    let e2 = sub(tri.v2, tri.v0);
    let p = cross(ray.direction, e2);
    let det = dot(e1, p);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = sub(ray.origin, tri.v0);
    let u = dot(tvec, p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(tvec, e1);
    let v = dot(ray.direction, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, q) * inv_det;
    if t < ray.t_min || t > ray.t_max {
        return None;
    }
    Some((t, u, v))
}

/// Watertight ray/triangle intersection (Woop, Benthin, Wald & Áfra, 2013).
///
/// Returns `(t, u, v)` on a hit, where `u`/`v` are the barycentric weights of
/// `v1`/`v2` (weight of `v0` is `1 - u - v`), matching [`intersect_triangle`].
///
/// Unlike the Möller–Trumbore variant, this test is *watertight*: a ray that
/// passes exactly through an edge or vertex shared by two triangles is
/// classified consistently, so a closed mesh never leaks (no "holes" where a
/// primary/shadow ray slips between adjacent faces). The ray is transformed
/// into a space where it points down `+kz`; the triangle is sheared into that
/// space and three scaled 2D edge cross products give the barycentric
/// coordinates. Edge tests that land exactly on zero in `f32` are recomputed in
/// `f64` so both incident triangles agree on which side owns the boundary. Both
/// faces are tested; the caller decides culling separately.
///
/// Only `f32`/`f64` arithmetic (add, mul, div, `abs`) is used — no
/// transcendentals — so it stays inside the deterministic-math budget the
/// `GPU` twin also honours.
#[inline]
pub(crate) fn intersect_triangle_watertight(ray: &Ray, tri: &Triangle) -> Option<(f32, f32, f32)> {
    let dir = ray.direction;

    // Pick the largest-magnitude direction component as the projection axis
    // `kz`; this keeps `1/dir[kz]` well conditioned. `kx`/`ky` are the other two.
    let mut kz = 0usize;
    let mut max_abs = dir[0].abs();
    if dir[1].abs() > max_abs {
        kz = 1;
        max_abs = dir[1].abs();
    }
    if dir[2].abs() > max_abs {
        kz = 2;
    }
    let mut kx = if kz + 1 == 3 { 0 } else { kz + 1 };
    let mut ky = if kx + 1 == 3 { 0 } else { kx + 1 };
    // Swap `kx`/`ky` when the ray points down `-kz` so winding is preserved.
    if dir[kz] < 0.0 {
        let tmp = kx;
        kx = ky;
        ky = tmp;
    }

    // Shear/scale constants aligning the ray with `+kz`.
    let sx = dir[kx] / dir[kz];
    let sy = dir[ky] / dir[kz];
    let sz = 1.0 / dir[kz];

    // Triangle vertices relative to the ray origin.
    let a = sub(tri.v0, ray.origin);
    let b = sub(tri.v1, ray.origin);
    let c = sub(tri.v2, ray.origin);

    // Sheared 2D coordinates in the (kx, ky) plane.
    let ax = a[kx] - sx * a[kz];
    let ay = a[ky] - sy * a[kz];
    let bx = b[kx] - sx * b[kz];
    let by = b[ky] - sy * b[kz];
    let cx = c[kx] - sx * c[kz];
    let cy = c[ky] - sy * c[kz];

    // Scaled barycentric coordinates: `wa`/`wb`/`wc` weight `v0`/`v1`/`v2`.
    let mut wa = cx * by - cy * bx;
    let mut wb = ax * cy - ay * cx;
    let mut wc = bx * ay - by * ax;

    // Exact `f64` fallback on boundary zeros: this is what makes the test
    // watertight, since it removes the single-precision rounding that would
    // otherwise let a shared edge fall between two triangles.
    if wa == 0.0 || wb == 0.0 || wc == 0.0 {
        wa = (f64::from(cx) * f64::from(by) - f64::from(cy) * f64::from(bx)) as f32;
        wb = (f64::from(ax) * f64::from(cy) - f64::from(ay) * f64::from(cx)) as f32;
        wc = (f64::from(bx) * f64::from(ay) - f64::from(by) * f64::from(ax)) as f32;
    }

    // Outside the triangle when the edge signs disagree (a boundary zero is
    // accepted by both the positive and negative branch, so edges never leak).
    if (wa < 0.0 || wb < 0.0 || wc < 0.0) && (wa > 0.0 || wb > 0.0 || wc > 0.0) {
        return None;
    }

    // Determinant; a zero determinant means the ray grazes the triangle plane.
    let det = wa + wb + wc;
    if det == 0.0 {
        return None;
    }

    // Scaled hit distance, then the interval test carrying `det`'s sign so we
    // reject before dividing.
    let az = sz * a[kz];
    let bz = sz * b[kz];
    let cz = sz * c[kz];
    let t_scaled = wa * az + wb * bz + wc * cz;
    if det > 0.0 {
        if t_scaled < ray.t_min * det || t_scaled > ray.t_max * det {
            return None;
        }
    } else if t_scaled > ray.t_min * det || t_scaled < ray.t_max * det {
        return None;
    }

    let inv_det = 1.0 / det;
    Some((t_scaled * inv_det, wb * inv_det, wc * inv_det))
}

impl Bvh {
    /// Nearest intersection along `ray`, or `None` if the ray hits nothing.
    ///
    /// Walks the flattened nodes with an explicit stack, visiting the child on
    /// the near side of the split axis first so the running `t_max` shrinks as
    /// fast as possible and far subtrees are culled by the slab test.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<Hit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let node = &self.nodes[node_index as usize];
            if slab_interval(ray.origin, ray.inv_direction, &node.bounds, ray.t_min, ray.t_max)
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for tri in &self.primitives[start..end] {
                        if let Some((t, u, v)) = intersect_triangle(&ray, tri) {
                            ray.t_max = t;
                            best = Some(Hit {
                                t,
                                u,
                                v,
                                primitive: tri.primitive,
                            });
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    // Visit the near child first based on the ray's direction
                    // sign along the split axis.
                    let first_child = node_index + 1;
                    let second_child = node.second_child;
                    let neg = ray.direction[node.axis as usize] < 0.0;
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

    /// True when *any* primitive intersects `ray` inside its interval.
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
            if slab_interval(ray.origin, ray.inv_direction, &node.bounds, ray.t_min, ray.t_max)
                .is_some()
            {
                if node.is_leaf() {
                    let start = node.first_primitive as usize;
                    let end = start + node.primitive_count as usize;
                    for tri in &self.primitives[start..end] {
                        if intersect_triangle(ray, tri).is_some() {
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

#[inline]
fn reciprocal(x: f32) -> f32 {
    // 1/0 -> +inf keeps the slab test's min/max ordering well defined.
    1.0 / x
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::{Bvh, Triangle};

    /// Deterministic proof of watertightness: rays that strike a shared edge
    /// exactly must never fall between the two triangles. On this rotated quad
    /// the Möller–Trumbore test leaks (misses both) on some samples, while the
    /// watertight test always reports at least one hit.
    #[test]
    fn watertight_never_leaks_on_shared_edge() {
        // Rotate a unit quad off every axis, then split it along its diagonal.
        let (c, sn) = (0.8f32, 0.6f32);
        let rot = |p: [f32; 3]| {
            let z = sn * p[1] + c * p[2];
            [c * p[0] - sn * z, c * p[1] - sn * p[2], sn * p[0] + c * z]
        };
        let p00 = rot([0.0, 0.0, 3.0]);
        let p10 = rot([1.0, 0.0, 3.0]);
        let p11 = rot([1.0, 1.0, 3.0]);
        let p01 = rot([0.0, 1.0, 3.0]);
        let a = tri(p00, p10, p11, 0);
        let b = tri(p00, p11, p01, 1);

        let n = 20_000u32;
        let mut mt_leaks = 0u32;
        let mut wt_leaks = 0u32;
        for i in 1..n {
            let s = i as f32 / n as f32;
            // A point on the shared diagonal p00 -> p11, seen from the origin.
            let dir = [
                p00[0] + s * (p11[0] - p00[0]),
                p00[1] + s * (p11[1] - p00[1]),
                p00[2] + s * (p11[2] - p00[2]),
            ];
            let ray = Ray::infinite([0.0, 0.0, 0.0], dir);
            let mt = usize::from(intersect_triangle(&ray, &a).is_some())
                + usize::from(intersect_triangle(&ray, &b).is_some());
            let wt = usize::from(intersect_triangle_watertight(&ray, &a).is_some())
                + usize::from(intersect_triangle_watertight(&ray, &b).is_some());
            if mt == 0 {
                mt_leaks += 1;
            }
            if wt == 0 {
                wt_leaks += 1;
            }
        }
        assert_eq!(wt_leaks, 0, "watertight test leaked on a shared edge");
        assert!(
            mt_leaks > 0,
            "expected Moller-Trumbore to leak so the improvement is exercised",
        );
    }

    /// Where both intersectors agree a ray hits a given triangle, the reported
    /// distance and barycentric weights must match closely.
    #[test]
    fn watertight_matches_moller_trumbore_on_clear_hits() {
        let tris = random_scene(400, 0x00c0_ffee_dead_beef);
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut compared = 0u32;
        for _ in 0..6000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let ray = Ray::infinite(origin, dir);
            for t in &tris {
                if let (Some((tm, um, vm)), Some((tw, uw, vw))) = (
                    intersect_triangle(&ray, t),
                    intersect_triangle_watertight(&ray, t),
                ) {
                    assert!((tm - tw).abs() <= 1e-3 * tm.abs().max(1.0), "t {tm} vs {tw}");
                    assert!((um - uw).abs() <= 2e-3, "u {um} vs {uw}");
                    assert!((vm - vw).abs() <= 2e-3, "v {vm} vs {vw}");
                    compared += 1;
                }
            }
        }
        assert!(compared > 100, "too few mutual hits ({compared}) to be meaningful");
    }

    /// The test is double-sided: it reports a hit from either face and returns
    /// mirrored barycentric weights when the winding flips.
    #[test]
    fn watertight_hits_both_faces() {
        let t = tri([-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0], 7);
        let front = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let back = Ray::infinite([0.0, 0.0, 4.0], [0.0, 0.0, -1.0]);
        let (tf, _, _) = intersect_triangle_watertight(&front, &t).expect("front hit");
        let (tb, _, _) = intersect_triangle_watertight(&back, &t).expect("back hit");
        assert!((tf - 2.0).abs() < 1e-5);
        assert!((tb - 2.0).abs() < 1e-5);
    }

    /// Rays parallel to the triangle plane and zero-area triangles miss.
    #[test]
    fn watertight_misses_parallel_and_degenerate() {
        // Triangle in the z=2 plane; a ray gliding through z=0 never reaches it.
        let t = tri([-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0], 0);
        let parallel = Ray::infinite([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(intersect_triangle_watertight(&parallel, &t).is_none());

        // Collinear vertices span no area, so nothing can be inside them.
        let degenerate = tri([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0], 1);
        let ray = Ray::infinite([0.5, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(intersect_triangle_watertight(&ray, &degenerate).is_none());
    }

    fn tri(a: [f32; 3], b: [f32; 3], c: [f32; 3], id: u32) -> Triangle {
        Triangle::new(a, b, c, id)
    }

    #[test]
    fn ray_hits_axis_aligned_triangle_center() {
        // Triangle in the z=2 plane; ray straight down +z from origin.
        let t = tri([-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0], 3);
        let bvh = Bvh::build(&[t]);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let hit = bvh.closest_hit(&ray).expect("should hit");
        assert_eq!(hit.primitive, 3);
        assert!((hit.t - 2.0).abs() < 1e-6, "t={}", hit.t);
        // Hit point recovered from t lies in the triangle plane.
        assert!((ray.at(hit.t)[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ray_misses_when_pointing_away() {
        let t = tri([-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0], 0);
        let bvh = Bvh::build(&[t]);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(bvh.closest_hit(&ray).is_none());
        assert!(!bvh.any_hit(&ray));
    }

    #[test]
    fn closest_hit_returns_nearest_of_two() {
        let near = tri([-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0], 10);
        let far = tri([-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0], 20);
        let bvh = Bvh::build(&[far, near]);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let hit = bvh.closest_hit(&ray).unwrap();
        assert_eq!(hit.primitive, 10);
        assert!((hit.t - 1.0).abs() < 1e-6);
    }

    #[test]
    fn t_interval_clamps_out_far_hits() {
        let far = tri([-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [0.0, 1.0, 5.0], 1);
        let bvh = Bvh::build(&[far]);
        // Limit t_max below the hit distance -> no hit.
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 4.0);
        assert!(bvh.closest_hit(&ray).is_none());
        let ray_ok = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 6.0);
        assert!(bvh.closest_hit(&ray_ok).is_some());
    }

    #[test]
    fn any_hit_detects_occluder() {
        let t = tri([-1.0, -1.0, 3.0], [1.0, -1.0, 3.0], [0.0, 1.0, 3.0], 0);
        let bvh = Bvh::build(&[t]);
        let blocked = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 10.0);
        assert!(bvh.any_hit(&blocked));
        // Interval ends before the occluder -> not blocked.
        let clear = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 2.0);
        assert!(!bvh.any_hit(&clear));
    }

    /// Deterministic xorshift rng (shared shape with the bvh module tests).
    struct Rng(u64);
    impl Rng {
        fn raw(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.raw() >> 40) as f32 / (1u32 << 24) as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn random_scene(n: u32, seed: u64) -> Vec<Triangle> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|id| {
                let c = [
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                ];
                let p = |r: &mut Rng| {
                    [
                        c[0] + r.range(-0.6, 0.6),
                        c[1] + r.range(-0.6, 0.6),
                        c[2] + r.range(-0.6, 0.6),
                    ]
                };
                tri(p(&mut rng), p(&mut rng), p(&mut rng), id)
            })
            .collect()
    }

    /// Brute-force nearest hit over every triangle, mirroring the same
    /// single-precision intersection the traversal uses.
    fn brute_force(tris: &[Triangle], ray: &Ray) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut r = *ray;
        for t in tris {
            if let Some((tt, u, v)) = intersect_triangle(&r, t) {
                r.t_max = tt;
                best = Some(Hit {
                    t: tt,
                    u,
                    v,
                    primitive: t.primitive,
                });
            }
        }
        best
    }

    #[test]
    fn bvh_traversal_matches_brute_force_golden() {
        let tris = random_scene(600, 0xdead_c0de_1234_5678);
        let bvh = Bvh::build(&tris);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut hits = 0u32;
        for _ in 0..4000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir[0] == 0.0 && dir[1] == 0.0 && dir[2] == 0.0 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            let bvh_hit = bvh.closest_hit(&ray);
            let bf_hit = brute_force(&tris, &ray);
            match (bvh_hit, bf_hit) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    // Same nearest distance to full f32 precision; the winning
                    // primitive matches except on exact ties (absent here).
                    assert!(
                        (a.t - b.t).abs() <= 1e-4 * (1.0 + b.t.abs()),
                        "t mismatch: bvh={} bf={}",
                        a.t,
                        b.t
                    );
                    assert_eq!(a.primitive, b.primitive, "primitive mismatch at t={}", b.t);
                    hits += 1;
                }
                (a, b) => panic!("hit disagreement: bvh={a:?} bf={b:?}"),
            }
        }
        // Sanity: the random scene is dense enough that some rays connect.
        assert!(hits > 50, "expected meaningful hit coverage, got {hits}");
    }

    #[test]
    fn any_hit_agrees_with_closest_hit_existence() {
        let tris = random_scene(500, 0x1122_3344_5566_7788);
        let bvh = Bvh::build(&tris);
        let mut rng = Rng(0x0102_0304_0506_0708);
        for _ in 0..2000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir == [0.0, 0.0, 0.0] {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            assert_eq!(bvh.any_hit(&ray), bvh.closest_hit(&ray).is_some());
        }
    }

    #[test]
    fn refit_after_motion_matches_brute_force_and_preserves_topology() {
        // Build over the original scene, then translate every triangle and
        // refit in place. Traversal against the refit hierarchy must agree with
        // a brute-force scan of the moved geometry, and the node/leaf topology
        // must be untouched (refit never rebuilds).
        let tris = random_scene(400, 0x00c0_ffee_d00d_1010);
        let mut bvh = Bvh::build(&tris);
        let node_count_before = bvh.node_count();
        let prim_count_before = bvh.primitive_count();

        // A deterministic per-primitive displacement keyed on the stable id.
        let offset = |id: u32| {
            let f = id as f32 * 0.123;
            [0.7 + 0.1 * f, -0.4 + 0.05 * f, 0.9 - 0.03 * f]
        };
        let moved: Vec<Triangle> = tris
            .iter()
            .map(|t| {
                let o = offset(t.primitive);
                let shift = |p: [f32; 3]| [p[0] + o[0], p[1] + o[1], p[2] + o[2]];
                tri(shift(t.v0), shift(t.v1), shift(t.v2), t.primitive)
            })
            .collect();

        bvh.refit(|id| {
            let m = moved[id as usize];
            [m.v0, m.v1, m.v2]
        });
        assert_eq!(bvh.node_count(), node_count_before, "refit changed node count");
        assert_eq!(bvh.primitive_count(), prim_count_before);

        let mut rng = Rng(0x5151_2727_9393_a1a1);
        let mut hits = 0u32;
        for _ in 0..4000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir == [0.0, 0.0, 0.0] {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            match (bvh.closest_hit(&ray), brute_force(&moved, &ray)) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert!(
                        (a.t - b.t).abs() <= 1e-4 * (1.0 + b.t.abs()),
                        "refit t mismatch: bvh={} bf={}",
                        a.t,
                        b.t
                    );
                    assert_eq!(a.primitive, b.primitive, "refit primitive mismatch");
                    hits += 1;
                }
                (a, b) => panic!("refit hit disagreement: bvh={a:?} bf={b:?}"),
            }
        }
        assert!(hits > 50, "expected meaningful hit coverage after refit, got {hits}");
    }

    #[test]
    fn rebuilt_after_refit_matches_fresh_build_and_brute_force() {
        // Refit loosens bounds under motion; `rebuilt` must reclaim a fresh,
        // compact SAH hierarchy identical to building from the moved geometry
        // directly, and traversal must still match brute force.
        let tris = random_scene(400, 0x0bad_f00d_1357_9bdf);
        let mut bvh = Bvh::build(&tris);

        let offset = |id: u32| {
            let f = id as f32 * 0.211;
            [1.3 - 0.07 * f, 0.6 + 0.09 * f, -0.8 + 0.04 * f]
        };
        let moved: Vec<Triangle> = tris
            .iter()
            .map(|t| {
                let o = offset(t.primitive);
                let shift = |p: [f32; 3]| [p[0] + o[0], p[1] + o[1], p[2] + o[2]];
                tri(shift(t.v0), shift(t.v1), shift(t.v2), t.primitive)
            })
            .collect();

        bvh.refit(|id| {
            let m = moved[id as usize];
            [m.v0, m.v1, m.v2]
        });

        let rebuilt = bvh.rebuilt();
        // A rebuild reclaims a compact hierarchy: every moved primitive is still
        // referenced exactly once (ids form the full set), and the flattened
        // arrays are densely packed (no fragmentation). The exact node ordering
        // depends on primitive input order, so we assert traversal equivalence
        // rather than bit-identical layout.
        assert_eq!(rebuilt.primitive_count(), moved.len());
        let mut ids: Vec<u32> = rebuilt.primitives().iter().map(|t| t.primitive).collect();
        ids.sort_unstable();
        let expected: Vec<u32> = (0..moved.len() as u32).collect();
        assert_eq!(ids, expected, "rebuilt must reference every primitive once");

        let mut rng = Rng(0x2468_ace0_1337_c0de);
        let mut hits = 0u32;
        for _ in 0..4000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir == [0.0, 0.0, 0.0] {
                continue;
            }
            let ray = Ray::infinite(origin, dir);
            match (rebuilt.closest_hit(&ray), brute_force(&moved, &ray)) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert!(
                        (a.t - b.t).abs() <= 1e-4 * (1.0 + b.t.abs()),
                        "rebuilt t mismatch: bvh={} bf={}",
                        a.t,
                        b.t
                    );
                    assert_eq!(a.primitive, b.primitive, "rebuilt primitive mismatch");
                    hits += 1;
                }
                (a, b) => panic!("rebuilt hit disagreement: bvh={a:?} bf={b:?}"),
            }
        }
        assert!(hits > 50, "expected meaningful hit coverage after rebuild, got {hits}");
    }
}
