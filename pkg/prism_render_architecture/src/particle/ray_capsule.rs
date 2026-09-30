//! Analytic ray-capsule intersection for the particle subsystem's picking,
//! collision-probe, and analytic-primitive raytrace contracts (design §10,
//! §14).
//!
//! A **capsule** is the set of points within `radius` of a finite line segment
//! `a -> b`: a cylindrical side wall swept along the segment and closed off at
//! each end by a *hemispherical* cap rather than a flat disk. Given a [`Ray`]
//! and a [`Capsule`] this module reports the nearest forward surface hit with
//! its position and outward unit normal ([`Capsule::first_hit`]) or an unsigned
//! boolean predicate ([`Capsule::intersects`]), solved entirely in closed form
//! in the Inigo-Quilez real-time-raytrace style.
//!
//! # How it differs from its siblings
//! * [`crate::particle::ray_cylinder`] closes a finite cylinder with two **flat
//!   disk caps**; this module closes it with two **round hemisphere caps**, so
//!   the ends bulge out by `radius` and the surface is everywhere smooth.
//! * [`crate::particle::ray_sphere`] is the single-center round primitive; a
//!   capsule degenerates *to* a sphere exactly when its two endpoints coincide,
//!   and this solver handles that degenerate case without a `NaN`.
//! * [`crate::particle::capsule_sdf`] evaluates a *signed distance field* — a
//!   scalar distance to the surface used by a raymarcher or metaball kernel —
//!   whereas this module solves the ray-surface **intersection** in closed form
//!   and never marches.
//!
//! # The three surface pieces
//! The capsule surface is solved as the union of three analytic pieces, and the
//! smallest non-negative ray parameter across all of them wins:
//!
//! 1. the **side wall** — projecting the ray into the plane perpendicular to
//!    the unit axis reduces the infinite-cylinder problem to a scalar quadratic
//!    in the ray parameter `t`; a root is a wall hit only when its axial
//!    coordinate `m = (hit - a) · axis` lands inside the `[0, height]` segment,
//! 2. the **near hemisphere** — the sphere of `radius` centered at `a`, whose
//!    root counts only on the outer half `m <= 0`, and
//! 3. the **far hemisphere** — the sphere of `radius` centered at `b`, whose
//!    root counts only on the outer half `m >= height`.
//!
//! The reported outward normal is reconstructed uniformly from the winning
//! point as `(point - closest_point_on_segment(point)) / radius`, which is the
//! radial direction on the wall and the sphere-center direction on either cap,
//! and is always unit length and outward even when the ray origin is inside the
//! volume.
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled in
//! this file and every computation uses only `+ - * /`, `f32::sqrt`,
//! `f32::abs`, `f32::min`, `f32::max`, and `f32::clamp`. No transcendental
//! function is ever called and no exact `==` / `!=` is ever written on a
//! production `f32` (magnitudes are compared against an explicit epsilon and
//! divisors are floored away from zero), so the `CPU` reference here agrees bit
//! for bit with a future `GPU` (`WESL`) kernel that packs the same capsule
//! through the `std430` helpers in [`crate::particle::gpu_layout`].
//!
//! The degenerate cases the solver handles explicitly rather than turning into
//! a `NaN` are: a zero-length ray direction and a zero (or near-zero) radius are
//! rejected as non-intersecting, and coincident endpoints collapse the whole
//! primitive to a single sphere that the two hemispheres cover between them.

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Epsilon used to guard divisions, to classify the quadratic discriminant, and
/// to compare parameters against zero without ever writing an exact `==` / `!=`
/// on a production `f32`.
const CMP_EPS: f32 = 1.0e-6;

/// A hand-rolled three-component vector, kept local so the module stays a
/// zero-dependency contract and its vector math is auditable in one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// A vector with all three lanes set to `s`.
    #[must_use]
    pub const fn splat(s: f32) -> Self {
        Self { x: s, y: s, z: s }
    }

    /// Component-wise sum `self + rhs`.
    #[must_use]
    pub fn plus(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    pub fn minus(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self × rhs` (right-handed), provided for callers that
    /// build an orthonormal frame around a hit normal.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only a
    /// comparison is needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Unit vector in the same direction, or the zero vector when the length is
    /// below [`CMP_EPS`], so a degenerate input never produces a `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len < CMP_EPS {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// A parametric ray `origin + t * dir` with `t >= 0` denoting the forward
/// half-line.
///
/// The solver works for any non-degenerate `dir` because it carries the
/// `dir · dir` term explicitly; construct with [`Ray::new_normalized`] when the
/// ray parameter `t` should read out as a Euclidean distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// The point the ray emanates from.
    pub origin: Vec3,
    /// The ray direction (not required to be unit length).
    pub dir: Vec3,
}

impl Ray {
    /// Builds a ray from an origin and a (possibly non-unit) direction.
    #[must_use]
    pub const fn new(origin: Vec3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    /// Builds a ray whose direction is normalized to unit length, so the ray
    /// parameter `t` returned by the solver equals the Euclidean distance from
    /// the origin. A degenerate direction collapses to the zero vector, which
    /// the solver later rejects as non-intersecting.
    #[must_use]
    pub fn new_normalized(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir: dir.normalize_or_zero(),
        }
    }

    /// Returns a copy of this ray with its direction normalized to unit length.
    #[must_use]
    pub fn normalized(self) -> Self {
        Self::new_normalized(self.origin, self.dir)
    }

    /// Evaluates the ray position at parameter `t`.
    #[must_use]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin.plus(self.dir.scale(t))
    }
}

/// A resolved forward intersection: the ray parameter, the world-space hit
/// point, and the outward-facing unit surface normal at that point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayCapsuleHit {
    /// The ray parameter at the hit (`>= 0`). Equals the hit distance when the
    /// ray direction is unit length.
    pub t: f32,
    /// The world-space intersection point `ray.at(t)`.
    pub point: Vec3,
    /// The unit surface normal pointing *out* of the capsule at `point`. It is
    /// the radial direction from the axis segment on the side wall and the
    /// sphere-center direction on either hemisphere cap. Always outward, even
    /// when the ray origin is inside the volume.
    pub normal: Vec3,
}

/// `std430` byte size of one packed [`Capsule`]: two `vec4<f32>` slots holding
/// `[a.xyz, radius]` and `[b.xyz, 0]`.
pub const RAY_CAPSULE_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// An analytic capsule primitive: the set of points within `radius` of the
/// segment `a -> b`, closed by a hemisphere at each end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule {
    /// The first (near) segment endpoint and center of the near hemisphere.
    pub a: Vec3,
    /// The second (far) segment endpoint and center of the far hemisphere.
    pub b: Vec3,
    /// The capsule radius. Values at or below [`CMP_EPS`] are degenerate and
    /// never report a hit.
    pub radius: f32,
}

impl Capsule {
    /// Builds a capsule from its two segment endpoints and a radius.
    #[must_use]
    pub const fn new(a: Vec3, b: Vec3, radius: f32) -> Self {
        Self { a, b, radius }
    }

    /// The `a -> b` segment vector (not unit length).
    #[must_use]
    pub fn axis(&self) -> Vec3 {
        self.b.minus(self.a)
    }

    /// The unit axis direction `a -> b`, or the zero vector when the endpoints
    /// coincide within [`CMP_EPS`].
    #[must_use]
    pub fn unit_axis(&self) -> Vec3 {
        self.axis().normalize_or_zero()
    }

    /// The segment length `|b - a|`, i.e. the axial span between the two cap
    /// centers (the round caps then extend the solid by `radius` past each end).
    #[must_use]
    pub fn height(&self) -> f32 {
        self.axis().length()
    }

    /// Whether the capsule is degenerate — its radius is at or below the compare
    /// epsilon — in which case it encloses no volume and never intersects.
    /// Coincident endpoints are *not* degenerate: they simply make the capsule a
    /// single sphere.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.radius <= CMP_EPS
    }

    /// The point on the axis segment `a -> b` closest to `p`, clamped to the
    /// endpoints. Used to reconstruct the outward normal and to classify
    /// containment.
    #[must_use]
    pub fn closest_on_segment(&self, p: Vec3) -> Vec3 {
        let ba = self.axis();
        let baba = ba.length_squared();
        if baba < CMP_EPS {
            // Coincident endpoints: the segment is the single point `a`.
            return self.a;
        }
        let h = (p.minus(self.a).dot(ba) / baba).clamp(0.0, 1.0);
        self.a.plus(ba.scale(h))
    }

    /// Whether `p` lies strictly inside the solid capsule, i.e. its distance to
    /// the axis segment is less than the radius by more than [`CMP_EPS`]. A
    /// degenerate capsule contains nothing.
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        if self.is_degenerate() {
            return false;
        }
        let d2 = p.minus(self.closest_on_segment(p)).length_squared();
        d2 < self.radius * self.radius - CMP_EPS
    }

    /// Reconstructs the outward unit normal at a surface `point`: the direction
    /// from the closest axis-segment point to `point`. Falls back to the zero
    /// vector only if `point` sits exactly on the axis (never reached for a real
    /// surface hit, which is `radius` away from the axis).
    #[must_use]
    fn outward_normal(&self, point: Vec3) -> Vec3 {
        point
            .minus(self.closest_on_segment(point))
            .normalize_or_zero()
    }

    /// Returns the nearest forward hit (`t >= 0`) against the side wall or
    /// either hemisphere cap, or `None` when the capsule is missed, lies
    /// entirely behind the origin, or is degenerate.
    ///
    /// The side wall forms the quadratic `a t^2 + b t + c = 0` in the plane
    /// perpendicular to the unit axis `ca`: with `d⊥ = d - (d·ca) ca` and
    /// `o⊥ = (o - a) - ((o - a)·ca) ca`, the coefficients are `a = d⊥·d⊥`,
    /// `b = 2 d⊥·o⊥`, and `c = o⊥·o⊥ - r^2`, and a root counts only when its
    /// axial coordinate `m = (hit - a)·ca` satisfies `0 <= m <= height`. Each cap
    /// is a ray-sphere solve about the corresponding endpoint accepted only on
    /// its outer half (`m <= 0` for the near cap, `m >= height` for the far cap).
    /// The smallest non-negative candidate wins and its normal is reconstructed
    /// from the winning point.
    #[must_use]
    pub fn first_hit(&self, ray: Ray) -> Option<RayCapsuleHit> {
        if self.is_degenerate() {
            return None;
        }
        if ray.dir.length_squared() <= CMP_EPS {
            // A zero-length direction is not a ray.
            return None;
        }
        let ca = self.unit_axis();
        let height = self.height();
        let r2 = self.radius * self.radius;

        let mut best_t = f32::INFINITY;
        let mut best_point = Vec3::ZERO;
        let mut hit_any = false;
        let mut push = |t: f32, point: Vec3| {
            if t < best_t {
                best_t = t;
                best_point = point;
                hit_any = true;
            }
        };

        // --- Side wall: quadratic in the plane perpendicular to the axis. ---
        let oc = ray.origin.minus(self.a);
        let d_par = ray.dir.dot(ca);
        let oc_par = oc.dot(ca);
        let d_perp = ray.dir.minus(ca.scale(d_par));
        let oc_perp = oc.minus(ca.scale(oc_par));
        let a_wall = d_perp.dot(d_perp);
        if a_wall > CMP_EPS {
            let b = 2.0 * d_perp.dot(oc_perp);
            let c = oc_perp.dot(oc_perp) - r2;
            let disc = b * b - 4.0 * a_wall * c;
            if disc >= 0.0 {
                let sqrt_disc = disc.max(0.0).sqrt();
                let inv_2a = 1.0 / (2.0 * a_wall);
                let roots = [(-b - sqrt_disc) * inv_2a, (-b + sqrt_disc) * inv_2a];
                for raw_t in roots {
                    if raw_t >= -CMP_EPS {
                        let t = raw_t.max(0.0);
                        let point = ray.at(t);
                        let m = point.minus(self.a).dot(ca);
                        if m >= -CMP_EPS && m <= height + CMP_EPS {
                            push(t, point);
                        }
                    }
                }
            }
        }

        // --- Hemisphere caps: a ray-sphere solve about each endpoint,
        //     accepted only on the outer hemisphere. ---
        let a_sph = ray.dir.dot(ray.dir);
        let inv_2a_sph = 1.0 / (2.0 * a_sph);
        let caps = [(self.a, false), (self.b, true)];
        for (center, is_far) in caps {
            let oc_s = ray.origin.minus(center);
            let b = 2.0 * ray.dir.dot(oc_s);
            let c = oc_s.dot(oc_s) - r2;
            let disc = b * b - 4.0 * a_sph * c;
            if disc < 0.0 {
                continue;
            }
            let sqrt_disc = disc.max(0.0).sqrt();
            let roots = [(-b - sqrt_disc) * inv_2a_sph, (-b + sqrt_disc) * inv_2a_sph];
            for raw_t in roots {
                if raw_t >= -CMP_EPS {
                    let t = raw_t.max(0.0);
                    let point = ray.at(t);
                    let m = point.minus(self.a).dot(ca);
                    let on_cap = if is_far {
                        m >= height - CMP_EPS
                    } else {
                        m <= CMP_EPS
                    };
                    if on_cap {
                        push(t, point);
                    }
                }
            }
        }

        if hit_any {
            Some(RayCapsuleHit {
                t: best_t,
                point: best_point,
                normal: self.outward_normal(best_point),
            })
        } else {
            None
        }
    }

    /// Whether the ray's forward half-line strikes the capsule surface (side
    /// wall or either hemisphere cap).
    #[must_use]
    pub fn intersects(&self, ray: Ray) -> bool {
        self.first_hit(ray).is_some()
    }

    /// Serializes the capsule to its little-endian `std430` byte image: two
    /// `vec4<f32>` slots holding `[a.x, a.y, a.z, radius]` and
    /// `[b.x, b.y, b.z, 0]`, exactly the eight words a `WESL` kernel would read
    /// in two `vec4` loads.
    #[must_use]
    pub fn to_std430(&self) -> [u8; RAY_CAPSULE_STD430_SIZE] {
        let mut bytes = [0u8; RAY_CAPSULE_STD430_SIZE];
        let words = [
            self.a.x,
            self.a.y,
            self.a.z,
            self.radius,
            self.b.x,
            self.b.y,
            self.b.z,
            0.0,
        ];
        for (i, word) in words.iter().enumerate() {
            let start = i * U32_STRIDE;
            bytes[start..start + U32_STRIDE].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Total `std430` byte size for a storage buffer of `count` capsules, clamped up
/// to one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(RAY_CAPSULE_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Absolute tolerance for parameter / position comparisons that are not
    /// bit-exact (they involve a `sqrt`).
    const EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {b}, got {a}");
    }

    fn approx_vec(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    /// A canonical unit-radius capsule along +Y with cap centers at `y = 0` and
    /// `y = 4` (so the solid spans `y in [-1, 5]` once the round caps are added).
    fn unit_cap() -> Capsule {
        Capsule::new(Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0), 1.0)
    }

    #[test]
    fn vec3_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -1.0, 0.5);
        assert_eq!(a.plus(b), Vec3::new(5.0, 1.0, 3.5));
        assert_eq!(a.minus(b), Vec3::new(-3.0, 3.0, 2.5));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.dot(Vec3::new(1.0, 1.0, 1.0)), 6.0);
        assert_eq!(Vec3::splat(3.0), Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_length_and_normalize() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length_squared(), 25.0);
        approx(a.length(), 5.0);
        approx_vec(a.normalize_or_zero(), Vec3::new(0.6, 0.8, 0.0));
    }

    #[test]
    fn normalize_zero_vector_is_zero_not_nan() {
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
    }

    #[test]
    fn ray_at_and_normalized() {
        let r = Ray::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
        approx_vec(r.at(3.0), Vec3::new(1.0, 6.0, 0.0));
        let n = r.normalized();
        approx(n.dir.length(), 1.0);
        approx_vec(n.dir, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn capsule_axis_height_and_unit_axis() {
        let cap = unit_cap();
        approx_vec(cap.axis(), Vec3::new(0.0, 4.0, 0.0));
        approx(cap.height(), 4.0);
        approx_vec(cap.unit_axis(), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn coincident_endpoints_are_not_degenerate() {
        let cap = Capsule::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0), 2.0);
        assert!(!cap.is_degenerate());
        approx(cap.height(), 0.0);
        assert_eq!(cap.unit_axis(), Vec3::ZERO);
    }

    #[test]
    fn side_wall_hit_reports_radial_normal() {
        let cap = unit_cap();
        // Fire in -X at mid-height; the near wall is at x = 1.
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = cap.first_hit(r).expect("should hit the wall");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(1.0, 2.0, 0.0));
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn far_side_wall_hit_is_further() {
        let cap = unit_cap();
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        // The near wall is at t = 4; the exit wall at x = -1 is t = 6.
        let near = cap.first_hit(r).expect("hit");
        approx(near.t, 4.0);
        assert!(near.t < 6.0);
    }

    #[test]
    fn near_hemisphere_cap_hit_along_axis() {
        let cap = unit_cap();
        // Straight up the +Y axis from below: first surface is the round cap
        // apex at y = -1.
        let r = Ray::new_normalized(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = cap.first_hit(r).expect("should hit near cap");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(0.0, -1.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn far_hemisphere_cap_hit_along_axis() {
        let cap = unit_cap();
        // Straight down the -Y axis from above: first surface is the far cap
        // apex at y = 5.
        let r = Ray::new_normalized(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = cap.first_hit(r).expect("should hit far cap");
        approx(hit.t, 5.0);
        approx_vec(hit.point, Vec3::new(0.0, 5.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn cap_bulge_is_hit_where_a_flat_cap_would_miss() {
        let cap = unit_cap();
        // Aim below the near cap center but within the hemisphere: a flat-capped
        // cylinder ending at y = 0 would be missed here, but the round cap is
        // struck.
        let r = Ray::new_normalized(Vec3::new(5.0, -0.5, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = cap.first_hit(r).expect("round cap should be hit");
        // At y = -0.5 the near-cap sphere has radius sqrt(1 - 0.25) = 0.8660.
        approx(hit.point.x, 0.8660254);
        approx(hit.point.y, -0.5);
        // Normal points from cap center `a` outwards.
        approx_vec(hit.normal, Vec3::new(0.8660254, -0.5, 0.0));
    }

    #[test]
    fn ray_parallel_to_axis_offset_grazes_wall() {
        let cap = unit_cap();
        // Parallel to +Y, offset by exactly the radius in X, coming from below:
        // it grazes the surface tangentially at the cap equator.
        let r = Ray::new_normalized(Vec3::new(1.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = cap.first_hit(r).expect("tangent graze should register");
        // Tangent to the near cap sphere of radius 1 centered at origin: the
        // grazing point is at the equator y = 0, x = 1.
        approx(hit.point.x, 1.0);
        approx(hit.point.y, 0.0);
    }

    #[test]
    fn ray_parallel_to_axis_just_outside_misses() {
        let cap = unit_cap();
        // Parallel to the axis but offset just beyond the radius: a clean miss.
        let r = Ray::new_normalized(Vec3::new(1.001, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(cap.first_hit(r).is_none());
    }

    #[test]
    fn origin_inside_reports_far_exit_only() {
        let cap = unit_cap();
        // Start at the axis center, fire out +X; the only forward hit is the
        // exit wall at x = 1.
        let r = Ray::new_normalized(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cap.first_hit(r).expect("exit hit expected");
        approx(hit.t, 1.0);
        approx_vec(hit.point, Vec3::new(1.0, 2.0, 0.0));
        // The reported normal is still the outward wall normal.
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn origin_inside_cap_region_exits_through_cap() {
        let cap = unit_cap();
        // Inside the near hemisphere, fire down -Y; exit through the cap apex.
        let r = Ray::new_normalized(Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = cap.first_hit(r).expect("cap exit expected");
        approx_vec(hit.point, Vec3::new(0.0, -1.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn clean_miss_beside_the_capsule() {
        let cap = unit_cap();
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cap.first_hit(r).is_none());
    }

    #[test]
    fn miss_beyond_the_far_cap() {
        let cap = unit_cap();
        // Passes above the far cap apex (y = 5) so nothing is struck.
        let r = Ray::new_normalized(Vec3::new(5.0, 6.5, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cap.first_hit(r).is_none());
    }

    #[test]
    fn hit_entirely_behind_origin_is_none() {
        let cap = unit_cap();
        // Capsule is behind the origin along the ray direction.
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(cap.first_hit(r).is_none());
    }

    #[test]
    fn degenerate_zero_radius_is_rejected() {
        let cap = Capsule::new(Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0), 0.0);
        assert!(cap.is_degenerate());
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cap.first_hit(r).is_none());
    }

    #[test]
    fn degenerate_zero_direction_is_rejected() {
        let cap = unit_cap();
        let r = Ray::new(Vec3::new(5.0, 2.0, 0.0), Vec3::ZERO);
        assert!(cap.first_hit(r).is_none());
    }

    #[test]
    fn coincident_endpoints_behave_like_a_sphere() {
        // A capsule with a = b is a sphere of the given radius at that point.
        let cap = Capsule::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cap.first_hit(r).expect("sphere hit expected");
        approx(hit.point.x, 1.0);
        approx_vec(hit.normal, Vec3::new(-1.0, 0.0, 0.0));
        // The far side is the exit at x = 3.
        let inside = Ray::new_normalized(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let exit = cap.first_hit(inside).expect("exit expected");
        approx(exit.point.x, 3.0);
    }

    #[test]
    fn non_axis_aligned_capsule_wall_hit() {
        // Axis along +X, cap centers at x = 0 and x = 4.
        let cap = Capsule::new(Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::new(2.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = cap.first_hit(r).expect("should hit the wall");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(2.0, 1.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn diagonal_capsule_wall_normal_is_perpendicular_to_axis() {
        // A capsule along the diagonal (1,1,0); a wall normal must be orthogonal
        // to that axis.
        let cap = Capsule::new(Vec3::ZERO, Vec3::new(4.0, 4.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::new(2.0, 2.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = cap.first_hit(r).expect("wall hit expected");
        approx(hit.point.z, 1.0);
        // Normal is purely +Z here (perpendicular to the in-plane axis).
        approx_vec(hit.normal, Vec3::new(0.0, 0.0, 1.0));
        // And it is unit length.
        approx(hit.normal.length(), 1.0);
        // Orthogonal to the axis direction.
        approx(hit.normal.dot(cap.unit_axis()), 0.0);
    }

    #[test]
    fn reported_normal_is_always_unit_length() {
        let cap = unit_cap();
        let rays = [
            Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
            Ray::new_normalized(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            Ray::new_normalized(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
            Ray::new_normalized(Vec3::new(5.0, -0.5, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        ];
        for r in rays {
            let hit = cap.first_hit(r).expect("hit expected");
            approx(hit.normal.length(), 1.0);
        }
    }

    #[test]
    fn normal_points_outward_from_the_surface() {
        let cap = unit_cap();
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = cap.first_hit(r).expect("hit");
        // Stepping a hair along the outward normal must land outside the solid,
        // and stepping inward must land inside.
        let outside = hit.point.plus(hit.normal.scale(0.01));
        let inside = hit.point.minus(hit.normal.scale(0.01));
        assert!(!cap.contains(outside));
        assert!(cap.contains(inside));
    }

    #[test]
    fn t_is_monotone_for_a_closer_capsule() {
        let near = Capsule::new(Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 4.0, 0.0), 1.0);
        let far = Capsule::new(Vec3::new(8.0, 0.0, 0.0), Vec3::new(8.0, 4.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::new(-5.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let near_hit = near.first_hit(r).expect("near hit");
        let far_hit = far.first_hit(r).expect("far hit");
        assert!(near_hit.t < far_hit.t, "nearer capsule must be hit first");
    }

    #[test]
    fn wall_hit_is_closer_than_cap_when_both_would_hit() {
        let cap = unit_cap();
        // A ray that would strike both the wall and the far cap; the wall is
        // nearer and must win.
        let r = Ray::new_normalized(Vec3::new(5.0, 3.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = cap.first_hit(r).expect("hit");
        approx(hit.point.x, 1.0);
        approx(hit.point.y, 3.0);
        // Mid-wall normal is purely radial.
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn non_unit_direction_scales_the_root() {
        let cap = unit_cap();
        // Direction of length 2: the ray parameter is half the Euclidean
        // distance to the wall (which is 4).
        let r = Ray::new(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-2.0, 0.0, 0.0));
        let hit = cap.first_hit(r).expect("wall hit");
        approx(hit.t, 2.0);
        approx_vec(hit.point, Vec3::new(1.0, 2.0, 0.0));
    }

    #[test]
    fn intersects_agrees_with_first_hit() {
        let cap = unit_cap();
        let hit_ray = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let miss_ray = Ray::new_normalized(Vec3::new(5.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(cap.intersects(hit_ray), cap.first_hit(hit_ray).is_some());
        assert_eq!(cap.intersects(miss_ray), cap.first_hit(miss_ray).is_some());
        assert!(cap.intersects(hit_ray));
        assert!(!cap.intersects(miss_ray));
    }

    #[test]
    fn contains_classifies_inside_and_outside() {
        let cap = unit_cap();
        // On the axis, mid-segment: inside.
        assert!(cap.contains(Vec3::new(0.0, 2.0, 0.0)));
        // Within the round near cap (below y = 0 but within radius of `a`).
        assert!(cap.contains(Vec3::new(0.0, -0.5, 0.0)));
        // Just outside the radius on the side.
        assert!(!cap.contains(Vec3::new(1.5, 2.0, 0.0)));
        // Beyond the far cap apex.
        assert!(!cap.contains(Vec3::new(0.0, 6.0, 0.0)));
    }

    #[test]
    fn closest_on_segment_clamps_to_endpoints() {
        let cap = unit_cap();
        // Beyond `b`: clamps to `b`.
        approx_vec(
            cap.closest_on_segment(Vec3::new(0.0, 100.0, 0.0)),
            Vec3::new(0.0, 4.0, 0.0),
        );
        // Before `a`: clamps to `a`.
        approx_vec(
            cap.closest_on_segment(Vec3::new(0.0, -100.0, 0.0)),
            Vec3::ZERO,
        );
        // Mid: projects onto the axis.
        approx_vec(
            cap.closest_on_segment(Vec3::new(3.0, 2.0, 0.0)),
            Vec3::new(0.0, 2.0, 0.0),
        );
    }

    #[test]
    fn tangent_ray_touches_a_single_point() {
        let cap = unit_cap();
        // A ray tangent to the side wall at x = 1 (grazing at z = 0), mid-height.
        let r = Ray::new_normalized(Vec3::new(1.0, 2.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = cap.first_hit(r).expect("tangent hit");
        approx(hit.point.x, 1.0);
        approx(hit.point.z, 0.0);
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn std430_size_is_two_vec4_slots() {
        assert_eq!(RAY_CAPSULE_STD430_SIZE, 32);
        assert_eq!(RAY_CAPSULE_STD430_SIZE, 2 * VEC4_STRIDE);
        assert_eq!(RAY_CAPSULE_STD430_SIZE % 16, 0);
    }

    #[test]
    fn std430_roundtrip_decodes_all_fields() {
        let cap = Capsule::new(Vec3::new(1.5, -2.25, 3.75), Vec3::new(4.5, 5.5, 6.5), 8.25);
        let bytes = cap.to_std430();
        assert_eq!(bytes.len(), 32);
        let mut words = [0.0f32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            let start = i * U32_STRIDE;
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[start..start + U32_STRIDE]);
            *w = f32::from_le_bytes(buf);
        }
        assert_eq!(words, [1.5, -2.25, 3.75, 8.25, 4.5, 5.5, 6.5, 0.0]);
    }

    #[test]
    fn gpu_storage_bytes_is_multiple_of_sixteen() {
        for count in [0usize, 1, 2, 7, 64, 1000] {
            assert_eq!(
                gpu_storage_bytes(count) % 16,
                0,
                "count {count} not 16-aligned"
            );
        }
    }

    #[test]
    fn gpu_storage_bytes_empty_reserves_one_element() {
        assert_eq!(gpu_storage_bytes(0), RAY_CAPSULE_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(1), 32);
        assert_eq!(gpu_storage_bytes(4), 128);
        assert_eq!(gpu_storage_bytes(256), 8192);
    }

    #[test]
    fn packing_many_capsules_is_contiguous() {
        let caps = [
            Capsule::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 1.0),
            Capsule::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0), 0.5),
        ];
        let mut blob: Vec<u8> = Vec::new();
        for c in &caps {
            blob.extend_from_slice(&c.to_std430());
        }
        assert_eq!(blob.len(), caps.len() * RAY_CAPSULE_STD430_SIZE);
        assert_eq!(blob.len(), gpu_storage_bytes(caps.len()));
    }
}
