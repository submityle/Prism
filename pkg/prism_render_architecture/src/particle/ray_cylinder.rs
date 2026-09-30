//! Analytic ray-finite-cylinder intersection for the particle subsystem's
//! picking, collision-probe, and analytic-primitive raytrace contracts (design
//! §10, §14).
//!
//! This module owns the *closed-form* solution of the ray against a **finite,
//! capped cylinder**: a segment of axis `base -> base + axis * height` swept at
//! a radius `r`, closed off by two flat circular end caps. Given a [`Ray`] and
//! a [`Cylinder`] it reports the nearest forward surface hit with its position
//! and outward unit normal ([`Cylinder::first_hit`]) or an unsigned boolean
//! predicate ([`Cylinder::intersects`]).
//!
//! It is the analytic sibling of the closed-form [`crate::particle::ray_sphere`]
//! primitive and the marched [`crate::particle::capsule_sdf`] field, and it is
//! deliberately *not* a signed-distance evaluator: this file solves the two
//! geometric pieces of a finite cylinder in closed form and never marches. The
//! two pieces are
//!
//! 1. the **side wall** — projecting the ray into the plane perpendicular to
//!    the axis reduces the infinite-cylinder problem to a scalar quadratic in
//!    the ray parameter `t`; a root is a wall hit only when its axial
//!    coordinate lands inside the `[0, height]` segment, and
//! 2. the **end caps** — two disk-bounded planes at the base and top; a plane
//!    root is a cap hit only when the point falls within the radius.
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled in
//! this file and every computation uses only `+ - * /`, `f32::sqrt`,
//! `f32::abs`, `f32::min`, `f32::max`, and `f32::clamp`. No transcendental
//! function is ever called and no exact `==` / `!=` is ever written on a
//! production `f32` (magnitudes are compared against an explicit epsilon and
//! divisors are floored away from zero), so the `CPU` reference here agrees bit
//! for bit with a future `GPU` (`WESL`) kernel that packs the same cylinder
//! through the `std430` helpers in [`crate::particle::gpu_layout`].
//!
//! The degenerate cases the solver rejects rather than turning into a `NaN`
//! are: a zero-length ray direction, a zero (or near-zero) radius, a zero (or
//! near-zero) height, and a zero-length axis. A ray parallel to the axis simply
//! contributes no side-wall roots (its perpendicular projection collapses) and
//! is answered by the caps alone.

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

    /// Unit vector in the same direction, or the zero vector when the input is
    /// shorter than [`CMP_EPS`] (so a degenerate input can never yield a
    /// `NaN`).
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
pub struct RayCylinderHit {
    /// The ray parameter at the hit (`>= 0`). Equals the hit distance when the
    /// ray direction is unit length.
    pub t: f32,
    /// The world-space intersection point `ray.at(t)`.
    pub point: Vec3,
    /// The unit surface normal pointing *out* of the cylinder at `point`. On the
    /// side wall it is the radial direction from the axis to the point; on an
    /// end cap it is the axis direction (`-axis` at the base, `+axis` at the
    /// top). Always outward, even when the ray origin is inside the volume.
    pub normal: Vec3,
}

/// `std430` byte size of one packed [`Cylinder`]: two `vec4<f32>` slots holding
/// `[base.xyz, radius]` and `[axis.xyz, height]`.
pub const RAY_CYLINDER_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// An analytic finite, capped cylinder primitive.
///
/// The solid is the set of points within `radius` of the axis segment running
/// from `base` to `base + axis * height`, closed by a flat disk at each end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
    /// The center of the base end cap.
    pub base: Vec3,
    /// The axis direction. Need not be unit length; the solver normalizes it and
    /// rejects a near-zero axis as degenerate.
    pub axis: Vec3,
    /// The cylinder radius. Values at or below [`CMP_EPS`] are degenerate and
    /// never report a hit.
    pub radius: f32,
    /// The distance from the base cap to the top cap along the unit axis. Values
    /// at or below [`CMP_EPS`] are degenerate and never report a hit.
    pub height: f32,
}

impl Cylinder {
    /// Builds a cylinder from a base point, axis direction, radius, and height.
    #[must_use]
    pub const fn new(base: Vec3, axis: Vec3, radius: f32, height: f32) -> Self {
        Self {
            base,
            axis,
            radius,
            height,
        }
    }

    /// The unit axis direction, or the zero vector when the stored axis is
    /// shorter than [`CMP_EPS`].
    #[must_use]
    pub fn unit_axis(&self) -> Vec3 {
        self.axis.normalize_or_zero()
    }

    /// Whether the cylinder is degenerate — its radius or height is at or below
    /// the compare epsilon, or its axis is too short to define a direction — in
    /// which case it encloses no volume and never intersects.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.radius <= CMP_EPS || self.height <= CMP_EPS || self.axis.length_squared() < CMP_EPS
    }

    /// The world-space center of the top end cap (`base + unit_axis * height`).
    #[must_use]
    pub fn top(&self) -> Vec3 {
        self.base.plus(self.unit_axis().scale(self.height))
    }

    /// Whether `p` lies strictly inside the solid cylinder (axially within
    /// `(0, height)` and radially within the radius), both by more than
    /// [`CMP_EPS`]. A degenerate cylinder contains nothing.
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        if self.is_degenerate() {
            return false;
        }
        let ca = self.unit_axis();
        let rel = p.minus(self.base);
        let m = rel.dot(ca);
        if m <= CMP_EPS || m >= self.height - CMP_EPS {
            return false;
        }
        let perp = rel.minus(ca.scale(m));
        perp.length_squared() < self.radius * self.radius - CMP_EPS
    }

    /// Returns the nearest forward hit (`t >= 0`) against the side wall or an
    /// end cap, or `None` when the cylinder is missed, lies entirely behind the
    /// origin, or is degenerate.
    ///
    /// The side wall forms the quadratic `a t^2 + b t + c = 0` in the plane
    /// perpendicular to the unit axis `ca`: with `d⊥ = d - (d·ca) ca` and
    /// `o⊥ = (o - base) - ((o - base)·ca) ca`, the coefficients are `a = d⊥·d⊥`,
    /// `b = 2 d⊥·o⊥`, and `c = o⊥·o⊥ - r^2`. A root counts only when its axial
    /// coordinate `m = (hit - base)·ca` satisfies `0 <= m <= height`. Each cap
    /// is a ray-plane solve `t = ((center - o)·ca) / (d·ca)` accepted only when
    /// the hit is within `r` of the cap center. The smallest non-negative
    /// candidate wins; the reported normal always points out of the solid.
    #[must_use]
    pub fn first_hit(&self, ray: Ray) -> Option<RayCylinderHit> {
        if self.is_degenerate() {
            return None;
        }
        if ray.dir.length_squared() <= CMP_EPS {
            // A zero-length direction is not a ray.
            return None;
        }
        let ca = self.unit_axis();

        let mut best: Option<RayCylinderHit> = None;
        let mut push = |t: f32, point: Vec3, normal: Vec3| {
            let closer = match best {
                Some(h) => t < h.t,
                None => true,
            };
            if closer {
                best = Some(RayCylinderHit { t, point, normal });
            }
        };

        // --- Side wall: quadratic in the plane perpendicular to the axis. ---
        let oc = ray.origin.minus(self.base);
        let d_par = ray.dir.dot(ca);
        let oc_par = oc.dot(ca);
        let d_perp = ray.dir.minus(ca.scale(d_par));
        let oc_perp = oc.minus(ca.scale(oc_par));
        let a = d_perp.dot(d_perp);
        if a > CMP_EPS {
            let b = 2.0 * d_perp.dot(oc_perp);
            let c = oc_perp.dot(oc_perp) - self.radius * self.radius;
            let disc = b * b - 4.0 * a * c;
            if disc >= -CMP_EPS {
                let sqrt_disc = disc.max(0.0).sqrt();
                let inv_2a = 1.0 / (2.0 * a);
                let roots = [(-b - sqrt_disc) * inv_2a, (-b + sqrt_disc) * inv_2a];
                for raw_t in roots {
                    if raw_t >= -CMP_EPS {
                        let t = raw_t.max(0.0);
                        let point = ray.at(t);
                        let m = point.minus(self.base).dot(ca);
                        if m >= -CMP_EPS && m <= self.height + CMP_EPS {
                            let axis_point = self.base.plus(ca.scale(m));
                            let normal = point.minus(axis_point).normalize_or_zero();
                            push(t, point, normal);
                        }
                    }
                }
            }
        }

        // --- End caps: two disk-bounded planes at the base and the top. ---
        let caps = [
            (self.base, ca.scale(-1.0)),
            (self.base.plus(ca.scale(self.height)), ca),
        ];
        let denom = ray.dir.dot(ca);
        if denom.abs() > CMP_EPS {
            let inv_denom = 1.0 / denom;
            for (center, out_normal) in caps {
                let raw_t = center.minus(ray.origin).dot(ca) * inv_denom;
                if raw_t >= -CMP_EPS {
                    let t = raw_t.max(0.0);
                    let point = ray.at(t);
                    let rel = point.minus(center);
                    let axial = rel.dot(ca);
                    let perp = rel.minus(ca.scale(axial));
                    if perp.length_squared() <= self.radius * self.radius + CMP_EPS {
                        push(t, point, out_normal);
                    }
                }
            }
        }

        best
    }

    /// Whether the ray's forward half-line strikes the cylinder surface (side
    /// wall or either cap).
    #[must_use]
    pub fn intersects(&self, ray: Ray) -> bool {
        self.first_hit(ray).is_some()
    }

    /// Serializes the cylinder to its little-endian `std430` byte image: two
    /// `vec4<f32>` slots holding `[base.x, base.y, base.z, radius]` and
    /// `[axis.x, axis.y, axis.z, height]`, exactly the eight words a `WESL`
    /// kernel would read in two `vec4` loads.
    #[must_use]
    pub fn to_std430(&self) -> [u8; RAY_CYLINDER_STD430_SIZE] {
        let mut bytes = [0u8; RAY_CYLINDER_STD430_SIZE];
        let words = [
            self.base.x,
            self.base.y,
            self.base.z,
            self.radius,
            self.axis.x,
            self.axis.y,
            self.axis.z,
            self.height,
        ];
        for (i, word) in words.iter().enumerate() {
            let start = i * U32_STRIDE;
            bytes[start..start + U32_STRIDE].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Total `std430` byte size for a storage buffer of `count` cylinders, clamped
/// up to one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(RAY_CYLINDER_STD430_SIZE, count)
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

    /// A canonical unit-radius cylinder along +Y spanning `y in [0, 4]`.
    fn unit_cyl() -> Cylinder {
        Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 1.0, 4.0)
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
    fn front_side_hit_reports_t_and_outward_normal() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = cyl.first_hit(r).expect("should hit side wall");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(1.0, 2.0, 0.0));
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn grazing_ray_outside_radius_misses() {
        let cyl = unit_cyl();
        // Travels along -x at a constant perpendicular distance of 2 from the
        // axis, so it never reaches the radius-1 wall.
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 2.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cyl.first_hit(r).is_none());
        assert!(!cyl.intersects(r));
    }

    #[test]
    fn passing_through_both_walls_takes_near_t() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let hit = cyl.first_hit(r).expect("should hit near wall");
        // Near wall is at x = 1 (t = 4), far wall at x = -1 (t = 6).
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(1.0, 2.0, 0.0));
    }

    #[test]
    fn top_cap_hit_has_axis_aligned_normal() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(0.5, 10.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = cyl.first_hit(r).expect("should hit top cap");
        approx(hit.t, 6.0);
        approx_vec(hit.point, Vec3::new(0.5, 4.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn base_cap_hit_has_downward_normal() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(0.5, -10.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = cyl.first_hit(r).expect("should hit base cap");
        approx(hit.t, 10.0);
        approx_vec(hit.point, Vec3::new(0.5, 0.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn origin_inside_hits_far_wall_with_outward_normal() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = cyl.first_hit(r).expect("inside origin should hit far wall");
        approx(hit.t, 1.0);
        approx_vec(hit.point, Vec3::new(1.0, 2.0, 0.0));
        // Normal still points *out* of the volume even though we hit from inside.
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn parallel_to_axis_hits_cap() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = cyl
            .first_hit(r)
            .expect("axis-parallel ray should hit base cap");
        approx(hit.t, 5.0);
        approx_vec(hit.point, Vec3::ZERO);
        approx_vec(hit.normal, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn parallel_to_axis_outside_radius_misses() {
        let cyl = unit_cyl();
        // Perpendicular distance 2 > radius 1: the caps are missed and the side
        // quadratic degenerates (no perpendicular component).
        let r = Ray::new_normalized(Vec3::new(2.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(cyl.first_hit(r).is_none());
    }

    #[test]
    fn hit_beyond_height_segment_is_not_a_side_hit() {
        let cyl = unit_cyl();
        // Aimed at the infinite cylinder wall at y = 10, above the top cap
        // (height 4). Horizontal, so the caps are parallel and unhit too.
        let r = Ray::new_normalized(Vec3::new(5.0, 10.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cyl.first_hit(r).is_none());
    }

    #[test]
    fn backward_ray_reports_no_hit() {
        let cyl = unit_cyl();
        // Cylinder is entirely behind the origin along +x.
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(cyl.first_hit(r).is_none());
        assert!(!cyl.intersects(r));
    }

    #[test]
    fn side_normal_is_unit_length() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(3.0, 1.5, 3.0), Vec3::new(-1.0, 0.0, -1.0));
        let hit = cyl.first_hit(r).expect("diagonal ray should hit the wall");
        approx(hit.normal.length(), 1.0);
        // Radial normal has no axial component.
        approx(hit.normal.y, 0.0);
    }

    #[test]
    fn cap_normal_is_unit_length() {
        let cyl = unit_cyl();
        let r = Ray::new_normalized(Vec3::new(0.25, 9.0, 0.25), Vec3::new(0.0, -1.0, 0.0));
        let hit = cyl.first_hit(r).expect("should hit top cap");
        approx(hit.normal.length(), 1.0);
    }

    #[test]
    fn solver_is_deterministic() {
        let cyl = Cylinder::new(
            Vec3::new(1.0, -2.0, 0.5),
            Vec3::new(0.0, 0.0, 3.0),
            1.25,
            5.0,
        );
        let r = Ray::new_normalized(Vec3::new(6.0, -2.0, 2.0), Vec3::new(-1.0, 0.0, 0.1));
        let a = cyl.first_hit(r);
        let b = cyl.first_hit(r);
        assert_eq!(a, b);
        assert_eq!(cyl.intersects(r), cyl.intersects(r));
    }

    #[test]
    fn degenerate_zero_direction_is_rejected() {
        let cyl = unit_cyl();
        let r = Ray::new(Vec3::new(5.0, 2.0, 0.0), Vec3::ZERO);
        assert!(cyl.first_hit(r).is_none());
    }

    #[test]
    fn degenerate_zero_radius_is_rejected() {
        let cyl = Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.0, 4.0);
        assert!(cyl.is_degenerate());
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cyl.first_hit(r).is_none());
    }

    #[test]
    fn degenerate_zero_height_is_rejected() {
        let cyl = Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 1.0, 0.0);
        assert!(cyl.is_degenerate());
        let r = Ray::new_normalized(Vec3::new(5.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cyl.first_hit(r).is_none());
    }

    #[test]
    fn degenerate_zero_axis_is_rejected() {
        let cyl = Cylinder::new(Vec3::ZERO, Vec3::ZERO, 1.0, 4.0);
        assert!(cyl.is_degenerate());
        assert_eq!(cyl.unit_axis(), Vec3::ZERO);
        let r = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        assert!(cyl.first_hit(r).is_none());
    }

    #[test]
    fn non_unit_direction_scales_the_root() {
        let cyl = unit_cyl();
        // Direction of length 2: the ray parameter is half the distance.
        let r = Ray::new(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-2.0, 0.0, 0.0));
        let hit = cyl.first_hit(r).expect("should hit wall");
        // Distance to the wall is 4, so t = 4 / 2 = 2.
        approx(hit.t, 2.0);
        approx_vec(hit.point, Vec3::new(1.0, 2.0, 0.0));
    }

    #[test]
    fn contains_classifies_inside_and_outside() {
        let cyl = unit_cyl();
        assert!(cyl.contains(Vec3::new(0.0, 2.0, 0.0)));
        assert!(cyl.contains(Vec3::new(0.5, 0.5, 0.0)));
        // Outside the radius.
        assert!(!cyl.contains(Vec3::new(2.0, 2.0, 0.0)));
        // Beyond the top cap.
        assert!(!cyl.contains(Vec3::new(0.0, 5.0, 0.0)));
        // Below the base cap.
        assert!(!cyl.contains(Vec3::new(0.0, -1.0, 0.0)));
    }

    #[test]
    fn top_helper_matches_axis_endpoint() {
        let cyl = unit_cyl();
        approx_vec(cyl.top(), Vec3::new(0.0, 4.0, 0.0));
    }

    #[test]
    fn intersects_agrees_with_first_hit() {
        let cyl = unit_cyl();
        let hit_ray = Ray::new_normalized(Vec3::new(5.0, 2.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));
        let miss_ray = Ray::new_normalized(Vec3::new(5.0, 2.0, 2.0), Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(cyl.intersects(hit_ray), cyl.first_hit(hit_ray).is_some());
        assert_eq!(cyl.intersects(miss_ray), cyl.first_hit(miss_ray).is_some());
        assert!(cyl.intersects(hit_ray));
        assert!(!cyl.intersects(miss_ray));
    }

    #[test]
    fn non_axis_aligned_cylinder_side_hit() {
        // Axis along +x, base at origin, spanning x in [0, 4].
        let cyl = Cylinder::new(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0, 4.0);
        let r = Ray::new_normalized(Vec3::new(2.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = cyl.first_hit(r).expect("should hit the wall");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(2.0, 1.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn std430_size_is_two_vec4_slots() {
        assert_eq!(RAY_CYLINDER_STD430_SIZE, 32);
        assert_eq!(RAY_CYLINDER_STD430_SIZE, 2 * VEC4_STRIDE);
        assert_eq!(RAY_CYLINDER_STD430_SIZE % 16, 0);
    }

    #[test]
    fn std430_roundtrip_decodes_all_fields() {
        let cyl = Cylinder::new(
            Vec3::new(1.5, -2.25, 3.75),
            Vec3::new(0.0, 0.0, 1.0),
            4.5,
            8.25,
        );
        let bytes = cyl.to_std430();
        assert_eq!(bytes.len(), 32);
        let mut words = [0.0f32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            let start = i * U32_STRIDE;
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[start..start + U32_STRIDE]);
            *w = f32::from_le_bytes(buf);
        }
        assert_eq!(words, [1.5, -2.25, 3.75, 4.5, 0.0, 0.0, 1.0, 8.25]);
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
        assert_eq!(gpu_storage_bytes(0), RAY_CYLINDER_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(1), 32);
        assert_eq!(gpu_storage_bytes(4), 128);
        assert_eq!(gpu_storage_bytes(256), 8192);
    }

    #[test]
    fn packing_many_cylinders_is_contiguous() {
        let cyls = [
            Cylinder::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 1.0, 2.0),
            Cylinder::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 0.0, 0.0), 0.5, 4.0),
        ];
        let mut blob: Vec<u8> = Vec::new();
        for c in &cyls {
            blob.extend_from_slice(&c.to_std430());
        }
        assert_eq!(blob.len(), cyls.len() * RAY_CYLINDER_STD430_SIZE);
        assert_eq!(blob.len(), gpu_storage_bytes(cyls.len()));
    }
}
