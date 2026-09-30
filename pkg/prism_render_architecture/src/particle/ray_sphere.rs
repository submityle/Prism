//! Analytic ray-sphere intersection for the particle subsystem's picking,
//! collision-probe, and analytic-primitive raytrace contracts (design §10,
//! §14).
//!
//! This module owns the *closed-form* solution of the ray-sphere problem: given
//! a [`Ray`] and a [`Sphere`], it forms the scalar quadratic whose roots are the
//! ray parameters at the two surface crossings, solves it with the quadratic
//! formula, and reports either a boolean hit test ([`Sphere::intersects`]) or
//! the nearest forward hit with its position and outward unit normal
//! ([`Sphere::first_hit`]). It is the analytic sibling of the marched
//! [`crate::particle::sdf`] field and the closed-form
//! [`crate::particle::capsule_sdf`] primitives: no voxel grid, no texture, no
//! iteration — every answer is derived from the ray parameters and the sphere
//! parameters in one algebraic pass.
//!
//! Everything is a zero-dependency contract: the vector math is hand-rolled in
//! this file, and every computation uses only `+ - * /`, `f32::sqrt`,
//! `f32::abs`, `f32::min`, and `f32::max`. No transcendental function is ever
//! called and no exact `==` / `!=` is ever written on a production `f32`
//! (magnitudes are compared against an explicit epsilon), so the `CPU`
//! reference here agrees bit for bit with a future `GPU` (`WESL`) kernel that
//! packs the same sphere through the `std430` helpers in
//! [`crate::particle::gpu_layout`].
//!
//! The four degenerate cases the solver handles explicitly are: the ray origin
//! lying *inside* the sphere (only the far root is a forward hit), a *tangent*
//! ray (the discriminant collapses to a single double root), a *miss* (a
//! negative discriminant), and a hit that lies entirely *behind* the origin
//! (both roots negative, so [`Sphere::first_hit`] reports nothing even though
//! [`Sphere::intersects`] — which is unsigned — still sees the line crossing).
//! A degenerate near-zero radius or a near-zero direction is likewise rejected
//! rather than producing a `NaN`.

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

/// The two real roots of the ray-sphere quadratic, ordered `near <= far`.
///
/// For a tangent ray the two fields hold the same double root.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootPair {
    /// The smaller (nearer) ray parameter.
    pub t_near: f32,
    /// The larger (farther) ray parameter.
    pub t_far: f32,
}

impl RootPair {
    /// Whether the two roots coincide within [`CMP_EPS`] — i.e. the ray grazes
    /// the sphere tangentially.
    #[must_use]
    pub fn is_tangent(self) -> bool {
        (self.t_far - self.t_near).abs() <= CMP_EPS
    }
}

/// A resolved forward intersection: the ray parameter, the world-space hit
/// point, and the outward-facing unit surface normal at that point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaySphereHit {
    /// The ray parameter at the hit (`>= 0`). Equals the hit distance when the
    /// ray direction is unit length.
    pub t: f32,
    /// The world-space intersection point `ray.at(t)`.
    pub point: Vec3,
    /// The unit surface normal pointing *out* of the sphere at `point`. Always
    /// outward, even when the origin is inside the sphere and the hit is on the
    /// far wall.
    pub normal: Vec3,
}

/// `std430` byte size of one packed [`Sphere`]: a single `vec4` slot holding
/// `center.xyz` in the first three lanes and `radius` in the fourth.
pub const RAY_SPHERE_STD430_SIZE: usize = VEC4_STRIDE;

/// An analytic sphere primitive: a center and a radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// The sphere center.
    pub center: Vec3,
    /// The sphere radius. Values at or below [`CMP_EPS`] are treated as a
    /// degenerate point and never report a hit.
    pub radius: f32,
}

impl Sphere {
    /// Builds a sphere from a center and radius.
    #[must_use]
    pub const fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Whether the sphere is degenerate (its radius is at or below the compare
    /// epsilon), in which case it encloses no volume and never intersects.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        self.radius <= CMP_EPS
    }

    /// Whether `p` lies strictly inside the sphere (by more than [`CMP_EPS`]).
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        let r = self.radius;
        p.minus(self.center).length_squared() < r * r - CMP_EPS
    }

    /// Solves the ray-sphere quadratic and returns the ordered root pair, or
    /// `None` when the line misses the sphere, the direction is degenerate, or
    /// the sphere is degenerate.
    ///
    /// The quadratic in the ray parameter `t` is `a t^2 + b t + c = 0` with
    /// `a = d·d`, `b = 2 (o-c)·d`, and `c = (o-c)·(o-c) - r^2`. The discriminant
    /// `b^2 - 4 a c` classifies the three cases: negative → miss, near-zero →
    /// tangent double root, positive → two distinct crossings. Roots come
    /// straight out of the quadratic formula, using only `f32::sqrt`.
    #[must_use]
    pub fn solve(&self, ray: Ray) -> Option<RootPair> {
        if self.is_degenerate() {
            return None;
        }
        let a = ray.dir.dot(ray.dir);
        if a <= CMP_EPS {
            // A zero-length direction is not a ray; reject rather than divide by
            // a vanishing `a`.
            return None;
        }
        let oc = ray.origin.minus(self.center);
        let b = 2.0 * oc.dot(ray.dir);
        let c = oc.dot(oc) - self.radius * self.radius;
        let disc = b * b - 4.0 * a * c;
        if disc < -CMP_EPS {
            // Strictly negative discriminant: the line never reaches the sphere.
            return None;
        }
        let inv_2a = 1.0 / (2.0 * a);
        if disc <= CMP_EPS {
            // Discriminant clamped to zero: a single tangent double root.
            let t = -b * inv_2a;
            return Some(RootPair {
                t_near: t,
                t_far: t,
            });
        }
        let sqrt_disc = disc.sqrt();
        let r0 = (-b - sqrt_disc) * inv_2a;
        let r1 = (-b + sqrt_disc) * inv_2a;
        Some(RootPair {
            t_near: r0.min(r1),
            t_far: r0.max(r1),
        })
    }

    /// Whether the ray's forward half-line crosses the sphere surface.
    ///
    /// This is the boolean predicate: it is `true` for a tangent graze, for an
    /// origin inside the sphere, and for a two-point crossing, as long as at
    /// least one root is at or ahead of the origin (`t >= -CMP_EPS`).
    #[must_use]
    pub fn intersects(&self, ray: Ray) -> bool {
        match self.solve(ray) {
            Some(roots) => roots.t_far >= -CMP_EPS,
            None => false,
        }
    }

    /// Returns the nearest forward hit (`t >= 0`), or `None` when the sphere is
    /// missed or lies entirely behind the origin.
    ///
    /// When both roots are non-negative the near root is chosen. When the near
    /// root is negative but the far root is not, the origin is inside the
    /// sphere and the far wall is the first forward hit. When both roots are
    /// negative the whole intersection is behind the origin and this reports
    /// `None`. The returned normal always points *out* of the sphere, computed
    /// as the unit vector from the center to the hit point.
    #[must_use]
    pub fn first_hit(&self, ray: Ray) -> Option<RaySphereHit> {
        let roots = self.solve(ray)?;
        let t = if roots.t_near >= -CMP_EPS {
            roots.t_near
        } else if roots.t_far >= -CMP_EPS {
            roots.t_far
        } else {
            return None;
        };
        // Clamp a tiny negative epsilon root up to exactly zero so the reported
        // parameter is never negative.
        let t = t.max(0.0);
        let point = ray.at(t);
        let normal = point.minus(self.center).normalize_or_zero();
        Some(RaySphereHit { t, point, normal })
    }

    /// Serializes the sphere to its little-endian `std430` byte image: a single
    /// `vec4<f32>` holding `[center.x, center.y, center.z, radius]`, exactly the
    /// four words a `WESL` kernel would read in one `vec4` load.
    #[must_use]
    pub fn to_std430(&self) -> [u8; RAY_SPHERE_STD430_SIZE] {
        let mut bytes = [0u8; RAY_SPHERE_STD430_SIZE];
        let words = [self.center.x, self.center.y, self.center.z, self.radius];
        for (i, word) in words.iter().enumerate() {
            let start = i * U32_STRIDE;
            bytes[start..start + U32_STRIDE].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Total `std430` byte size for a storage buffer of `count` spheres, clamped up
/// to one element so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(RAY_SPHERE_STD430_SIZE, count)
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

    fn unit_x() -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
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
    fn vec3_length_matches_pythagoras() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length_squared(), 25.0);
        approx(v.length(), 5.0);
    }

    #[test]
    fn normalize_zero_is_zero_not_nan() {
        let n = Vec3::ZERO.normalize_or_zero();
        assert_eq!(n, Vec3::ZERO);
        assert!(!n.x.is_nan());
    }

    #[test]
    fn normalize_unit_length() {
        let n = Vec3::new(0.0, 0.0, 7.0).normalize_or_zero();
        approx_vec(n, Vec3::new(0.0, 0.0, 1.0));
        approx(n.length(), 1.0);
    }

    #[test]
    fn ray_new_normalized_has_unit_direction() {
        let r = Ray::new_normalized(Vec3::ZERO, Vec3::new(0.0, 5.0, 0.0));
        approx(r.dir.length(), 1.0);
        approx_vec(r.dir, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn ray_normalized_of_zero_dir_is_zero() {
        let r = Ray::new(Vec3::ZERO, Vec3::ZERO).normalized();
        assert_eq!(r.dir, Vec3::ZERO);
    }

    #[test]
    fn ray_at_walks_along_direction() {
        let r = Ray::new(Vec3::new(1.0, 0.0, 0.0), unit_x());
        approx_vec(r.at(3.0), Vec3::new(4.0, 0.0, 0.0));
    }

    #[test]
    fn hit_along_positive_x_axis() {
        let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        let hit = s.first_hit(r).expect("should hit");
        approx(hit.t, 4.0);
        approx_vec(hit.point, Vec3::new(4.0, 0.0, 0.0));
        approx_vec(hit.normal, Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn hit_along_positive_y_axis() {
        let s = Sphere::new(Vec3::new(0.0, 5.0, 0.0), 2.0);
        let r = Ray::new_normalized(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let hit = s.first_hit(r).expect("should hit");
        approx(hit.t, 3.0);
        approx_vec(hit.point, Vec3::new(0.0, 3.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, -1.0, 0.0));
    }

    #[test]
    fn hit_along_negative_z_axis() {
        let s = Sphere::new(Vec3::new(0.0, 0.0, -5.0), 1.5);
        let r = Ray::new_normalized(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let hit = s.first_hit(r).expect("should hit");
        approx(hit.t, 3.5);
        approx_vec(hit.point, Vec3::new(0.0, 0.0, -3.5));
        approx_vec(hit.normal, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn tangent_ray_is_single_double_root() {
        // Ray along x at height y = r just grazes the sphere at the top.
        let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 1.0, 0.0), unit_x());
        let roots = s.solve(r).expect("tangent should solve");
        assert!(roots.is_tangent(), "expected tangent double root");
        approx(roots.t_near, 5.0);
        approx(roots.t_far, 5.0);
    }

    #[test]
    fn tangent_hit_normal_points_outward() {
        let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 1.0, 0.0), unit_x());
        let hit = s.first_hit(r).expect("tangent should hit");
        approx(hit.t, 5.0);
        approx_vec(hit.point, Vec3::new(5.0, 1.0, 0.0));
        approx_vec(hit.normal, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn origin_inside_gives_single_positive_root() {
        let s = Sphere::new(Vec3::ZERO, 2.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        let roots = s.solve(r).expect("inside should solve");
        assert!(roots.t_near < 0.0, "near root behind origin");
        assert!(roots.t_far > 0.0, "far root ahead of origin");
        let hit = s.first_hit(r).expect("inside should hit");
        approx(hit.t, 2.0);
        approx_vec(hit.point, Vec3::new(2.0, 0.0, 0.0));
        // Outward normal on the far wall still points away from the center.
        approx_vec(hit.normal, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn origin_inside_first_hit_t_is_positive() {
        let s = Sphere::new(Vec3::new(1.0, 1.0, 1.0), 3.0);
        let r = Ray::new_normalized(Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 2.0, -2.0));
        let hit = s.first_hit(r).expect("inside should hit");
        assert!(hit.t > 0.0);
        approx(hit.t, 3.0);
    }

    #[test]
    fn clear_miss_returns_none() {
        let s = Sphere::new(Vec3::new(0.0, 10.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        assert!(s.solve(r).is_none());
        assert!(!s.intersects(r));
        assert!(s.first_hit(r).is_none());
    }

    #[test]
    fn hit_entirely_behind_origin_is_no_forward_hit() {
        // Sphere is behind the origin; the ray points away from it.
        let s = Sphere::new(Vec3::new(-5.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        let roots = s.solve(r).expect("line still crosses");
        assert!(roots.t_near < 0.0 && roots.t_far < 0.0);
        assert!(s.first_hit(r).is_none());
        // The unsigned boolean predicate sees no forward crossing either.
        assert!(!s.intersects(r));
    }

    #[test]
    fn intersects_true_for_forward_hit() {
        let s = Sphere::new(Vec3::new(3.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        assert!(s.intersects(r));
    }

    #[test]
    fn intersects_true_when_origin_inside() {
        let s = Sphere::new(Vec3::ZERO, 5.0);
        let r = Ray::new_normalized(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert!(s.intersects(r));
    }

    #[test]
    fn roots_are_ordered_near_le_far() {
        let s = Sphere::new(Vec3::new(6.0, 0.0, 0.0), 2.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        let roots = s.solve(r).expect("should hit");
        assert!(roots.t_near <= roots.t_far);
        approx(roots.t_near, 4.0);
        approx(roots.t_far, 8.0);
    }

    #[test]
    fn first_hit_picks_the_near_root() {
        let s = Sphere::new(Vec3::new(6.0, 0.0, 0.0), 2.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        let hit = s.first_hit(r).expect("should hit");
        approx(hit.t, 4.0);
    }

    #[test]
    fn normal_is_unit_length() {
        let s = Sphere::new(Vec3::new(2.0, 3.0, -1.0), 1.25);
        let r = Ray::new_normalized(Vec3::ZERO, Vec3::new(2.0, 3.0, -1.0));
        let hit = s.first_hit(r).expect("should hit");
        approx(hit.normal.length(), 1.0);
    }

    #[test]
    fn normal_points_outward_on_near_wall() {
        // Outward normal on the near wall opposes the ray direction.
        let s = Sphere::new(Vec3::new(0.0, 0.0, 4.0), 1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let r = Ray::new_normalized(Vec3::ZERO, dir);
        let hit = s.first_hit(r).expect("should hit");
        assert!(hit.normal.dot(dir) < 0.0, "near-wall normal faces the ray");
        approx_vec(hit.normal, Vec3::new(0.0, 0.0, -1.0));
    }

    #[test]
    fn hit_point_lies_on_the_sphere_surface() {
        let s = Sphere::new(Vec3::new(1.0, -2.0, 3.0), 2.5);
        let r = Ray::new_normalized(Vec3::new(-4.0, -2.0, 3.0), unit_x());
        let hit = s.first_hit(r).expect("should hit");
        let d = hit.point.minus(s.center).length();
        approx(d, 2.5);
    }

    #[test]
    fn hit_point_equals_ray_at_t() {
        let s = Sphere::new(Vec3::new(0.0, 0.0, 7.0), 2.0);
        let r = Ray::new_normalized(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = s.first_hit(r).expect("should hit");
        approx_vec(hit.point, r.at(hit.t));
    }

    #[test]
    fn degenerate_radius_never_hits() {
        let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 0.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        assert!(s.is_degenerate());
        assert!(s.solve(r).is_none());
        assert!(!s.intersects(r));
        assert!(s.first_hit(r).is_none());
    }

    #[test]
    fn degenerate_direction_never_hits() {
        let s = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let r = Ray::new(Vec3::ZERO, Vec3::ZERO);
        assert!(s.solve(r).is_none());
        assert!(!s.intersects(r));
        assert!(s.first_hit(r).is_none());
    }

    #[test]
    fn contains_classifies_inside_and_outside() {
        let s = Sphere::new(Vec3::ZERO, 2.0);
        assert!(s.contains(Vec3::new(1.0, 0.0, 0.0)));
        assert!(!s.contains(Vec3::new(3.0, 0.0, 0.0)));
        // A point on the surface is not strictly inside.
        assert!(!s.contains(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn non_unit_direction_scales_the_root() {
        // With a direction of length 2, the ray parameter is half the distance.
        let s = Sphere::new(Vec3::new(10.0, 0.0, 0.0), 1.0);
        let r = Ray::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0));
        let roots = s.solve(r).expect("should hit");
        // Distance to near wall is 9, so t = 9 / 2 = 4.5.
        approx(roots.t_near, 4.5);
        approx(roots.t_far, 5.5);
        // The evaluated hit point is still at the true surface distance.
        approx_vec(r.at(roots.t_near), Vec3::new(9.0, 0.0, 0.0));
    }

    #[test]
    fn solver_is_deterministic() {
        let s = Sphere::new(Vec3::new(4.0, 1.0, -2.0), 1.75);
        let r = Ray::new_normalized(Vec3::new(-1.0, 1.0, -2.0), Vec3::new(1.0, 0.0, 0.0));
        let a = s.first_hit(r);
        let b = s.first_hit(r);
        assert_eq!(a, b);
        let ra = s.solve(r);
        let rb = s.solve(r);
        assert_eq!(ra, rb);
    }

    #[test]
    fn epsilon_comparison_never_uses_exact_equality() {
        // Origin exactly on the surface: the near root sits at t ~= 0. The
        // epsilon-guarded classification must accept it as a non-negative
        // forward hit (clamped up to 0) instead of tripping an exact == on
        // zero and discarding it, and the far wall must still land at t ~= 2.
        let s = Sphere::new(Vec3::new(1.0, 0.0, 0.0), 1.0);
        let r = Ray::new_normalized(Vec3::ZERO, unit_x());
        let hit = s.first_hit(r).expect("origin on surface should hit");
        assert!(hit.t >= 0.0);
        approx(hit.t, 0.0);
        let roots = s.solve(r).expect("origin on surface should solve");
        approx(roots.t_near, 0.0);
        approx(roots.t_far, 2.0);
    }

    #[test]
    fn tangent_predicate_flags_only_grazes() {
        let graze = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let r_graze = Ray::new_normalized(Vec3::new(0.0, 1.0, 0.0), unit_x());
        assert!(graze.solve(r_graze).expect("solve").is_tangent());
        let through = Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0);
        let r_through = Ray::new_normalized(Vec3::ZERO, unit_x());
        assert!(!through.solve(r_through).expect("solve").is_tangent());
    }

    #[test]
    fn std430_size_is_one_vec4_slot() {
        assert_eq!(RAY_SPHERE_STD430_SIZE, 16);
        assert_eq!(RAY_SPHERE_STD430_SIZE, VEC4_STRIDE);
        assert_eq!(RAY_SPHERE_STD430_SIZE % 16, 0);
    }

    #[test]
    fn std430_roundtrip_decodes_center_and_radius() {
        let s = Sphere::new(Vec3::new(1.5, -2.25, 3.75), 4.5);
        let bytes = s.to_std430();
        assert_eq!(bytes.len(), 16);
        let mut words = [0.0f32; 4];
        for (i, w) in words.iter_mut().enumerate() {
            let start = i * U32_STRIDE;
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[start..start + U32_STRIDE]);
            *w = f32::from_le_bytes(buf);
        }
        assert_eq!(words, [1.5, -2.25, 3.75, 4.5]);
    }

    #[test]
    fn gpu_storage_bytes_is_multiple_of_sixteen() {
        for count in [0usize, 1, 2, 7, 64, 1000] {
            let bytes = gpu_storage_bytes(count);
            assert_eq!(bytes % 16, 0, "count {count} not 16-aligned");
        }
    }

    #[test]
    fn gpu_storage_bytes_empty_reserves_one_element() {
        assert_eq!(gpu_storage_bytes(0), RAY_SPHERE_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(1), 16);
        assert_eq!(gpu_storage_bytes(4), 64);
        assert_eq!(gpu_storage_bytes(256), 4096);
    }

    #[test]
    fn packing_many_spheres_is_contiguous() {
        let spheres = [
            Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            Sphere::new(Vec3::new(1.0, 2.0, 3.0), 0.5),
            Sphere::new(Vec3::new(-4.0, 5.0, -6.0), 2.0),
        ];
        let mut blob: Vec<u8> = Vec::new();
        for s in &spheres {
            blob.extend_from_slice(&s.to_std430());
        }
        assert_eq!(blob.len(), spheres.len() * RAY_SPHERE_STD430_SIZE);
        assert_eq!(blob.len(), gpu_storage_bytes(spheres.len()));
    }

    #[test]
    fn offset_start_hit_distance_is_correct() {
        let s = Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0);
        let start = Vec3::new(-3.0, 0.0, 0.0);
        let r = Ray::new_normalized(start, unit_x());
        let hit = s.first_hit(r).expect("should hit");
        // Near wall at x = -1, so distance from x = -3 is 2.
        approx(hit.t, 2.0);
        approx_vec(hit.normal, Vec3::new(-1.0, 0.0, 0.0));
    }
}
