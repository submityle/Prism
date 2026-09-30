//! Closed-form `ray`-triangle and `ray`-plane intersection with `barycentric`
//! coordinates (design §14, §22).
//!
//! Particle collision against static geometry, decal projection, and beam /
//! trace effects all reduce to the same analytic question: where does a `ray`
//! pierce a triangle or a plane, and what are the surface `barycentric`
//! coordinates there? This module owns the `CPU`-verifiable *closed-form*
//! answer to that question — the branch-free `Moller-Trumbore` triangle test,
//! the one-division `ray`-plane test, and the `barycentric` reconstruction of a
//! point — so the acceleration and query layers can build on an exact
//! reference.
//!
//! # Strict scope
//! This module is *only* the analytic single-`ray`-single-primitive kernel:
//! * [`bvh`](crate::particle::bvh) owns building the acceleration structure
//!   that decides *which* triangles a `ray` is tested against.
//! * [`raytrace`](crate::particle::raytrace) owns the per-particle collision
//!   *query* that walks that structure and applies the collision response.
//!
//! Because it is a leaf contract it hand-rolls its own [`Vec3`] rather than
//! importing another particle module's vector type, and its helpers are
//! private — nothing here is a shared primitive.
//!
//! # No transcendental math
//! Every routine is polynomial plus at most one `sqrt` and one reciprocal:
//! lengths use `sqrt`, the `Moller-Trumbore` and `ray`-plane tests use a single
//! guarded division, and there is no `sin`/`cos`/`pow` anywhere. Degenerate
//! inputs never produce a `NaN`: a `ray` parallel to the triangle or plane, a
//! zero-area triangle, and a zero-length vector all fall back to a miss or to
//! [`Vec3::ZERO`] instead of dividing by a near-zero quantity.
//!
//! # Facing convention
//! Triangles are wound counter-clockwise: the geometric normal follows
//! `(v1 - v0) x (v2 - v0)`, and the *front* face is the side that normal points
//! toward. A front-facing hit is a `ray` travelling against the normal, which
//! yields a positive `Moller-Trumbore` determinant; `cull_backface` keeps only
//! those.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Magnitude below which a determinant, plane denominator, or vector length is
/// treated as degenerate, so the guarded division falls back to a miss / zero
/// instead of amplifying round-off into a `NaN` or a spurious far hit.
const EPS: f32 = 1.0e-7;

/// A three-component vector in the same right-handed space the triangle
/// vertices are expressed in.
///
/// Arithmetic methods are named `plus` / `minus` / `scale` (not the operator
/// names) to keep the closed-form algebra explicit and to avoid implying a
/// component-wise `Mul` on the cross/dot products.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// First (x) component.
    pub x: f32,
    /// Second (y) component.
    pub y: f32,
    /// Third (z) component.
    pub z: f32,
}

impl Vec3 {
    /// The additive-identity vector `(0, 0, 0)`, used as the degenerate
    /// fallback for normalization.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its three components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum `self + other`.
    #[must_use]
    pub fn plus(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Component-wise difference `self - other`.
    #[must_use]
    pub fn minus(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Uniform scale `self * scalar`.
    #[must_use]
    pub fn scale(&self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    /// The Euclidean dot product `self · other`.
    #[must_use]
    pub fn dot(&self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// The right-handed cross product `self × other`.
    #[must_use]
    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// The squared length `self · self`, avoiding the `sqrt` when only relative
    /// magnitudes matter.
    #[must_use]
    pub fn length_squared(&self) -> f32 {
        self.dot(*self)
    }

    /// The Euclidean length `sqrt(self · self)`.
    #[must_use]
    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    /// The unit vector in the same direction, or [`Vec3::ZERO`] when the length
    /// is below [`EPS`] so a near-zero vector never divides to a `NaN`.
    #[must_use]
    pub fn normalized(&self) -> Vec3 {
        let len = self.length();
        if len < EPS {
            return Vec3::ZERO;
        }
        self.scale(1.0 / len)
    }
}

/// A half-line with an `origin` point and a `dir` direction. The direction is
/// used as supplied (it need not be unit length); the returned `t` is measured
/// in units of `dir`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// The point the `ray` starts from.
    pub origin: Vec3,
    /// The direction the `ray` travels (not required to be normalized).
    pub dir: Vec3,
}

impl Ray {
    /// Builds a `ray` from its origin and direction.
    #[must_use]
    pub const fn new(origin: Vec3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    /// The point `origin + t * dir` at parameter `t` along the `ray`.
    #[must_use]
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin.plus(self.dir.scale(t))
    }
}

/// A successful triangle intersection: the `ray` parameter `t` and the two
/// free `barycentric` coordinates `u`, `v`. The third weight is `w = 1 - u - v`
/// and multiplies the first vertex `v0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Distance along the `ray` (in units of the `ray` direction) to the hit.
    pub t: f32,
    /// `barycentric` weight of the second vertex `v1`.
    pub u: f32,
    /// `barycentric` weight of the third vertex `v2`.
    pub v: f32,
}

/// Intersects a `ray` with the triangle `v0`, `v1`, `v2` using the
/// `Moller-Trumbore` algorithm.
///
/// Returns the [`Hit`] (parameter and `barycentric` coordinates) when the `ray`
/// crosses the triangle strictly in front of its origin (`t > EPS`), or `None`
/// on a miss. The determinant is guarded against zero, so a `ray` lying in the
/// triangle's plane is a miss rather than a division by a near-zero value. When
/// `cull_backface` is `true`, only front faces (positive determinant) are
/// considered; when `false`, both facings hit.
#[must_use]
pub fn intersect_moller_trumbore(
    ray: &Ray,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    cull_backface: bool,
) -> Option<Hit> {
    let edge1 = v1.minus(v0);
    let edge2 = v2.minus(v0);
    let pvec = ray.dir.cross(edge2);
    let det = edge1.dot(pvec);

    if cull_backface {
        if det < EPS {
            return None;
        }
    } else if det.abs() < EPS {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = ray.origin.minus(v0);

    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let qvec = tvec.cross(edge1);
    let v = ray.dir.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = edge2.dot(qvec) * inv_det;
    if t > EPS {
        Some(Hit { t, u, v })
    } else {
        None
    }
}

/// Intersects a `ray` with the infinite plane through `plane_point` with normal
/// `plane_normal`.
///
/// Returns the `ray` parameter `t` such that `origin + t * dir` lies on the
/// plane, or `None` when the `ray` is parallel to the plane (the denominator is
/// below [`EPS`]). A negative `t` means the intersection is behind the `ray`
/// origin; the caller decides whether that counts. The normal need not be unit
/// length. The division is guarded so a parallel `ray` never yields a `NaN`.
#[must_use]
pub fn intersect_plane(ray: &Ray, plane_point: Vec3, plane_normal: Vec3) -> Option<f32> {
    let denom = ray.dir.dot(plane_normal);
    if denom.abs() < EPS {
        return None;
    }
    let t = plane_point.minus(ray.origin).dot(plane_normal) / denom;
    Some(t)
}

/// Reconstructs the surface point from `barycentric` coordinates `(u, v)` over
/// the triangle `v0`, `v1`, `v2`, using the implied weight `w = 1 - u - v` on
/// `v0`. This is the exact inverse of the coordinates [`Hit`] reports.
#[must_use]
pub fn barycentric_to_point(v0: Vec3, v1: Vec3, v2: Vec3, u: f32, v: f32) -> Vec3 {
    let w = 1.0 - u - v;
    v0.scale(w).plus(v1.scale(u)).plus(v2.scale(v))
}

/// The unit surface normal of the triangle `v0`, `v1`, `v2`, following the
/// right-handed winding `(v1 - v0) × (v2 - v0)`.
///
/// A degenerate (zero-area or collinear) triangle has no defined normal, so
/// [`Vec3::ZERO`] is returned instead of a `NaN` from normalizing a near-zero
/// vector.
#[must_use]
pub fn triangle_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    let edge1 = v1.minus(v0);
    let edge2 = v2.minus(v0);
    edge1.cross(edge2).normalized()
}

/// Total `std430` byte size of a `GPU` storage buffer holding `count`
/// `vec4`-aligned triangle-hit records, reusing the shared
/// [`storage_bytes`](crate::particle::gpu_layout::storage_bytes) rule (an empty
/// batch still reserves one element).
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(VEC4_STRIDE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for comparing computed floats that pass through a `sqrt` or a
    /// division; exact integer-valued results are compared with `assert_eq!`.
    const CMP_EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    #[test]
    fn vec3_new_stores_components() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(a.z, 3.0);
    }

    #[test]
    fn vec3_plus_and_minus_are_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.plus(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.minus(a), Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn vec3_scale_is_exact() {
        let a = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(a.scale(2.0), Vec3::new(2.0, -4.0, 6.0));
        assert_eq!(a.scale(0.0), Vec3::ZERO);
    }

    #[test]
    fn vec3_dot_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_cross_anti_commutes() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-4.0, 5.0, 6.0);
        assert_eq!(a.cross(b), b.cross(a).scale(-1.0));
    }

    #[test]
    fn vec3_length_squared_is_exact() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length_squared(), 25.0);
    }

    #[test]
    fn vec3_length_uses_sqrt() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(a.length(), 5.0));
    }

    #[test]
    fn vec3_normalized_is_unit_length() {
        let n = Vec3::new(0.0, 5.0, 0.0).normalized();
        assert!(vec_approx(n, Vec3::new(0.0, 1.0, 0.0)));
        assert!(approx(n.length(), 1.0));
    }

    #[test]
    fn vec3_normalized_zero_is_zero_not_nan() {
        let n = Vec3::ZERO.normalized();
        assert_eq!(n, Vec3::ZERO);
        assert!(!n.x.is_nan());
    }

    #[test]
    fn ray_new_and_point_at() {
        let r = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(r.origin, Vec3::ZERO);
        assert_eq!(r.point_at(3.0), Vec3::new(0.0, 0.0, 3.0));
    }

    /// A counter-clockwise triangle in the `z = 0` plane whose geometric normal
    /// points toward `+z` (so its front face is the `+z` side).
    fn unit_triangle() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn mt_hits_front_facing_triangle() {
        let (v0, v1, v2) = unit_triangle();
        // Front face is the +z side; hit it travelling toward -z.
        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = intersect_moller_trumbore(&ray, v0, v1, v2, false).expect("hit");
        assert!(approx(hit.t, 1.0));
        assert!(approx(hit.u, 0.25));
        assert!(approx(hit.v, 0.25));
    }

    #[test]
    fn mt_center_barycentric_is_one_third() {
        let (v0, v1, v2) = unit_triangle();
        let third = 1.0 / 3.0;
        let ray = Ray::new(Vec3::new(third, third, -2.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_moller_trumbore(&ray, v0, v1, v2, false).expect("hit");
        assert!(approx(hit.u, third));
        assert!(approx(hit.v, third));
    }

    #[test]
    fn mt_hits_at_second_vertex() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_moller_trumbore(&ray, v0, v1, v2, false).expect("hit");
        assert!(approx(hit.u, 1.0));
        assert!(approx(hit.v, 0.0));
    }

    #[test]
    fn mt_misses_outside_triangle() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.9, 0.9, -1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_moller_trumbore(&ray, v0, v1, v2, false).is_none());
    }

    #[test]
    fn mt_negative_barycentric_misses() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(-0.1, 0.5, -1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_moller_trumbore(&ray, v0, v1, v2, false).is_none());
    }

    #[test]
    fn mt_parallel_ray_misses() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.25, 0.25, -1.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(intersect_moller_trumbore(&ray, v0, v1, v2, false).is_none());
    }

    #[test]
    fn mt_behind_origin_misses() {
        let (v0, v1, v2) = unit_triangle();
        // Triangle sits behind the origin along the travel direction.
        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_moller_trumbore(&ray, v0, v1, v2, false).is_none());
    }

    #[test]
    fn mt_backface_is_culled_when_requested() {
        let (v0, v1, v2) = unit_triangle();
        // Shooting from -z toward +z hits the triangle's back face.
        let ray = Ray::new(Vec3::new(0.25, 0.25, -1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_moller_trumbore(&ray, v0, v1, v2, true).is_none());
    }

    #[test]
    fn mt_backface_hits_when_not_culled() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.25, 0.25, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_moller_trumbore(&ray, v0, v1, v2, false).expect("hit");
        assert!(approx(hit.t, 1.0));
    }

    #[test]
    fn mt_front_face_survives_culling() {
        let (v0, v1, v2) = unit_triangle();
        // Front face is the +z side; travel toward -z to hit it.
        let ray = Ray::new(Vec3::new(0.25, 0.25, 1.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(intersect_moller_trumbore(&ray, v0, v1, v2, true).is_some());
    }

    #[test]
    fn mt_barycentric_weights_sum_to_one() {
        let (v0, v1, v2) = unit_triangle();
        let ray = Ray::new(Vec3::new(0.2, 0.3, -1.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_moller_trumbore(&ray, v0, v1, v2, false).expect("hit");
        let w = 1.0 - hit.u - hit.v;
        assert!(approx(hit.u + hit.v + w, 1.0));
    }

    #[test]
    fn mt_degenerate_triangle_misses() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(2.0, 0.0, 0.0);
        let ray = Ray::new(Vec3::new(0.5, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_moller_trumbore(&ray, v0, v1, v2, false).is_none());
    }

    #[test]
    fn mt_hit_point_matches_barycentric_reconstruction() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(2.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 3.0, 0.0);
        let ray = Ray::new(Vec3::new(0.5, 0.5, -4.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_moller_trumbore(&ray, v0, v1, v2, false).expect("hit");
        let from_bary = barycentric_to_point(v0, v1, v2, hit.u, hit.v);
        let from_ray = ray.point_at(hit.t);
        assert!(vec_approx(from_bary, from_ray));
        assert!(vec_approx(from_bary, Vec3::new(0.5, 0.5, 0.0)));
    }

    #[test]
    fn plane_intersects_along_axis() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let t =
            intersect_plane(&ray, Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 1.0)).expect("hit");
        assert!(approx(t, 7.0));
        assert!(vec_approx(ray.point_at(t), Vec3::new(0.0, 0.0, 2.0)));
    }

    #[test]
    fn plane_parallel_ray_misses() {
        let ray = Ray::new(Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(intersect_plane(&ray, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)).is_none());
    }

    #[test]
    fn plane_behind_origin_yields_negative_t() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, 1.0));
        let t = intersect_plane(&ray, Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0)).expect("hit");
        assert!(t < 0.0);
        assert!(approx(t, -5.0));
    }

    #[test]
    fn barycentric_to_point_recovers_vertices() {
        let v0 = Vec3::new(1.0, 0.0, 0.0);
        let v1 = Vec3::new(0.0, 1.0, 0.0);
        let v2 = Vec3::new(0.0, 0.0, 1.0);
        assert!(vec_approx(barycentric_to_point(v0, v1, v2, 0.0, 0.0), v0));
        assert!(vec_approx(barycentric_to_point(v0, v1, v2, 1.0, 0.0), v1));
        assert!(vec_approx(barycentric_to_point(v0, v1, v2, 0.0, 1.0), v2));
    }

    #[test]
    fn barycentric_to_point_centroid() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(3.0, 0.0, 0.0);
        let v2 = Vec3::new(0.0, 3.0, 0.0);
        let third = 1.0 / 3.0;
        let c = barycentric_to_point(v0, v1, v2, third, third);
        assert!(vec_approx(c, Vec3::new(1.0, 1.0, 0.0)));
    }

    #[test]
    fn triangle_normal_is_unit_and_faces_positive_z() {
        let (v0, v1, v2) = unit_triangle();
        let n = triangle_normal(v0, v1, v2);
        assert!(vec_approx(n, Vec3::new(0.0, 0.0, 1.0)));
        assert!(approx(n.length(), 1.0));
    }

    #[test]
    fn triangle_normal_degenerate_is_zero() {
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 1.0, 1.0);
        let v2 = Vec3::new(2.0, 2.0, 2.0);
        assert_eq!(triangle_normal(v0, v1, v2), Vec3::ZERO);
    }

    #[test]
    fn gpu_storage_bytes_reuses_shared_layout() {
        assert_eq!(gpu_storage_bytes(0), VEC4_STRIDE);
        assert_eq!(gpu_storage_bytes(4), VEC4_STRIDE * 4);
    }
}
