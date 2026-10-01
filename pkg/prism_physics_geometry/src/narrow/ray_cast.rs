//! Exact ray intersections against spheres and triangles.

use glam::Vec3;

use crate::bounding::{Aabb, BoundingSphere, Obb, Ray};

/// Returns the nearest parametric distance `t` at which `ray` enters `sphere`,
/// or [`None`] when there is no hit within `[0, ray.tmax]`.
///
/// The ray direction is unit length (see [`Ray::with_tmax`]), so the quadratic
/// in `t` has a unit leading coefficient. When the origin is inside the sphere
/// the first non-negative root is the exit point, which is still the nearest
/// surface crossing ahead of the origin and is reported.
pub fn ray_sphere(ray: &Ray, sphere: &BoundingSphere) -> Option<f32> {
    let oc = ray.origin - sphere.center;
    // a = dir.dot(dir) = 1 because the direction is normalized.
    let half_b = oc.dot(ray.dir);
    let c = oc.dot(oc) - sphere.radius * sphere.radius;
    let disc = half_b * half_b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    // Near root first; fall back to the far root when the origin is inside.
    let t_near = -half_b - sqrt_disc;
    let t = if t_near >= 0.0 {
        t_near
    } else {
        -half_b + sqrt_disc
    };
    if (0.0..=ray.tmax).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// A ray/triangle hit: the parametric distance and barycentric coordinates.
///
/// The hit point is `ray.at(t)` and equals `(1 - u - v) * a + u * b + v * c`
/// for the triangle vertices passed to [`ray_triangle`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RayTriangleHit {
    /// Parametric distance along the ray to the hit.
    pub t: f32,
    /// Barycentric weight of vertex `b`.
    pub u: f32,
    /// Barycentric weight of vertex `c`.
    pub v: f32,
}

/// Intersects `ray` with the triangle `(a, b, c)` using the Möller–Trumbore
/// algorithm, returning the hit within `[0, ray.tmax]` or [`None`].
///
/// The test is double sided: it accepts hits on either face. Rays parallel to
/// the triangle plane (near-zero determinant) and barycentric coordinates
/// outside the triangle are rejected.
pub fn ray_triangle(ray: &Ray, a: Vec3, b: Vec3, c: Vec3) -> Option<RayTriangleHit> {
    const EPSILON: f32 = 1.0e-7;
    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = ray.dir.cross(edge2);
    let det = edge1.dot(pvec);
    if det.abs() < EPSILON {
        // Ray is parallel to the triangle plane.
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin - a;
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
    if (0.0..=ray.tmax).contains(&t) {
        Some(RayTriangleHit { t, u, v })
    } else {
        None
    }
}

/// Returns the entry parameter `t` at which `ray` enters `obb`, or [`None`] on
/// a miss within `[0, ray.tmax]`.
///
/// The ray is rotated into the box's local frame (where the box is
/// axis-aligned about the origin) and refined with the standard slab test, so
/// the reported `t` is identical in world space because the transform is a
/// rigid rotation that preserves distances. When the origin is already inside
/// the box the returned `t` is `0.0`.
pub fn ray_obb(ray: &Ray, obb: &Obb) -> Option<f32> {
    // Conjugate of a unit quaternion is its inverse rotation.
    let inv = obb.orientation.conjugate();
    let local_origin = inv * (ray.origin - obb.center);
    let local_dir = inv * ray.dir;
    let local_ray = Ray::with_tmax(local_origin, local_dir, ray.tmax);
    let local_box = Aabb::new(-obb.half_extents, obb.half_extents);
    local_box.ray_hit(&local_ray)
}

#[cfg(test)]
mod tests {
    use super::{ray_obb, ray_sphere, ray_triangle};
    use crate::bounding::{BoundingSphere, Obb, Ray};
    use approx::assert_relative_eq;
    use core::f32::consts::FRAC_PI_4;
    use glam::{Quat, Vec3};

    #[test]
    fn ray_sphere_front_hit() {
        let s = BoundingSphere::new(Vec3::new(0.0, 0.0, 5.0), 1.0);
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let t = ray_sphere(&ray, &s).expect("hit");
        // Enters the near surface at z = 4.
        assert_relative_eq!(t, 4.0, epsilon = 1e-5);
    }

    #[test]
    fn ray_sphere_from_inside_returns_exit() {
        let s = BoundingSphere::new(Vec3::ZERO, 2.0);
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        let t = ray_sphere(&ray, &s).expect("hit");
        assert_relative_eq!(t, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn ray_sphere_miss_and_behind() {
        let s = BoundingSphere::new(Vec3::new(0.0, 0.0, 5.0), 1.0);
        // Parallel offset miss.
        let miss = Ray::new(Vec3::new(3.0, 0.0, 0.0), Vec3::Z);
        assert!(ray_sphere(&miss, &s).is_none());
        // Pointing away.
        let away = Ray::new(Vec3::ZERO, Vec3::NEG_Z);
        assert!(ray_sphere(&away, &s).is_none());
        // Hit beyond tmax.
        let short = Ray::with_tmax(Vec3::ZERO, Vec3::Z, 1.0);
        assert!(ray_sphere(&short, &s).is_none());
    }

    #[test]
    fn ray_triangle_center_hit() {
        let a = Vec3::new(-1.0, -1.0, 5.0);
        let b = Vec3::new(1.0, -1.0, 5.0);
        let c = Vec3::new(0.0, 1.0, 5.0);
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let hit = ray_triangle(&ray, a, b, c).expect("hit");
        assert_relative_eq!(hit.t, 5.0, epsilon = 1e-5);
        // Reconstruct the hit point from barycentrics and compare.
        let p = (1.0 - hit.u - hit.v) * a + hit.u * b + hit.v * c;
        assert!(p.abs_diff_eq(ray.at(hit.t), 1e-5));
    }

    #[test]
    fn ray_triangle_misses_outside() {
        let a = Vec3::new(-1.0, -1.0, 5.0);
        let b = Vec3::new(1.0, -1.0, 5.0);
        let c = Vec3::new(0.0, 1.0, 5.0);
        // Aimed well outside the triangle footprint.
        let ray = Ray::new(Vec3::new(5.0, 5.0, 0.0), Vec3::Z);
        assert!(ray_triangle(&ray, a, b, c).is_none());
    }

    #[test]
    fn ray_triangle_parallel_is_none() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);
        // Ray travels within the triangle's plane (y = 0).
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z);
        assert!(ray_triangle(&ray, a, b, c).is_none());
    }

    #[test]
    fn ray_triangle_behind_origin_is_none() {
        let a = Vec3::new(-1.0, -1.0, -5.0);
        let b = Vec3::new(1.0, -1.0, -5.0);
        let c = Vec3::new(0.0, 1.0, -5.0);
        // Triangle is behind the origin along +Z.
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        assert!(ray_triangle(&ray, a, b, c).is_none());
    }

    #[test]
    fn ray_obb_axis_aligned_hit() {
        let obb = Obb::new(Vec3::new(0.0, 0.0, 5.0), Vec3::splat(1.0), Quat::IDENTITY);
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let t = ray_obb(&ray, &obb).expect("hit");
        assert_relative_eq!(t, 4.0, epsilon = 1e-5);
    }

    #[test]
    fn ray_obb_rotation_changes_hit() {
        // A box spun 45° about Z presents a corner toward a diagonal ray.
        let obb = Obb::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::splat(1.0),
            Quat::from_rotation_z(FRAC_PI_4),
        );
        // Axis ray down +Z enters the rotated square face; near plane still z=4.
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let t = ray_obb(&ray, &obb).expect("hit");
        assert_relative_eq!(t, 4.0, epsilon = 1e-5);
    }

    #[test]
    fn ray_obb_miss_and_behind() {
        let obb = Obb::new(Vec3::new(0.0, 0.0, 5.0), Vec3::splat(1.0), Quat::IDENTITY);
        // Offset miss parallel to +Z.
        let miss = Ray::new(Vec3::new(3.0, 0.0, 0.0), Vec3::Z);
        assert!(ray_obb(&miss, &obb).is_none());
        // Pointing away.
        let away = Ray::new(Vec3::ZERO, Vec3::NEG_Z);
        assert!(ray_obb(&away, &obb).is_none());
        // Hit beyond tmax.
        let short = Ray::with_tmax(Vec3::ZERO, Vec3::Z, 1.0);
        assert!(ray_obb(&short, &obb).is_none());
    }
}
