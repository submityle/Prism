//! Intersection and overlap queries between [`crate::geom`] primitives.
//!
//! Ray queries return structured [`RayHit`] data (parameter, point, and
//! surface normal). Overlap tests return booleans, and frustum-culling queries
//! return a [`Containment`] classification. Every routine is validated in the
//! crate tests against closed-form analytic references ("相交金标准对拍").

use crate::float::f32 as mf;
use crate::geom::aabb::Aabb3;
use crate::geom::frustum::Frustum;
use crate::geom::plane::Plane;
use crate::geom::ray::Ray3;
use crate::geom::sphere::BoundingSphere;
use crate::vec::Vec3;

/// Structured result of a ray intersection.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RayHit {
    /// Ray parameter `t` of the hit, so the point is `ray.at(t)`.
    pub t: f32,
    /// World-space hit point.
    pub point: Vec3,
    /// Unit surface normal at the hit, oriented against the ray direction.
    pub normal: Vec3,
}

/// How a volume relates to a frustum (or another bounding volume).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Containment {
    /// Fully outside: can be culled.
    Outside,
    /// Partially overlapping the boundary.
    Intersecting,
    /// Fully inside.
    Inside,
}

impl Containment {
    /// True when the volume is at least partially visible (not [`Containment::Outside`]).
    #[inline]
    pub fn is_visible(self) -> bool {
        self != Containment::Outside
    }
}

/// Intersect a ray with a sphere, returning the nearest non-negative hit.
///
/// Works for any (non-zero) ray direction length; the returned normal points
/// outward from the sphere, flipped if the ray starts inside.
pub fn ray_sphere(ray: Ray3, sphere: BoundingSphere) -> Option<RayHit> {
    let oc = ray.origin - sphere.center;
    let a = ray.direction.dot(ray.direction);
    if a <= 0.0 {
        return None;
    }
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - sphere.radius * sphere.radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = mf::sqrt(disc);
    let inv2a = 1.0 / (2.0 * a);
    let t0 = (-b - sqrt_disc) * inv2a;
    let t1 = (-b + sqrt_disc) * inv2a;
    // Prefer the nearest hit in front of the origin.
    let (t, inside) = if t0 >= 0.0 {
        (t0, false)
    } else if t1 >= 0.0 {
        (t1, true)
    } else {
        return None;
    };
    let point = ray.at(t);
    let mut normal = (point - sphere.center) * (1.0 / sphere.radius);
    if inside {
        normal = -normal;
    }
    Some(RayHit { t, point, normal })
}

/// Intersect a ray with an axis-aligned box using the slab method.
///
/// Returns the entry hit (or the exit hit when the origin is inside). The
/// normal is axis-aligned and points against the ray.
pub fn ray_aabb(ray: Ray3, aabb: Aabb3) -> Option<RayHit> {
    let origin = [ray.origin.x, ray.origin.y, ray.origin.z];
    let dir = [ray.direction.x, ray.direction.y, ray.direction.z];
    let lo = [aabb.min.x, aabb.min.y, aabb.min.z];
    let hi = [aabb.max.x, aabb.max.y, aabb.max.z];

    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    let mut enter_axis = 0usize;
    let mut enter_sign = 1.0f32;

    for axis in 0..3 {
        if mf::abs(dir[axis]) <= 1.0e-20 {
            // Ray is parallel to this slab: miss if the origin is outside it.
            if origin[axis] < lo[axis] || origin[axis] > hi[axis] {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dir[axis];
        let mut t1 = (lo[axis] - origin[axis]) * inv;
        let mut t2 = (hi[axis] - origin[axis]) * inv;
        let mut sign = -1.0;
        if t1 > t2 {
            core::mem::swap(&mut t1, &mut t2);
            sign = 1.0;
        }
        if t1 > t_enter {
            t_enter = t1;
            enter_axis = axis;
            enter_sign = sign;
        }
        if t2 < t_exit {
            t_exit = t2;
        }
        if t_enter > t_exit {
            return None;
        }
    }

    if t_exit < 0.0 {
        return None;
    }
    let (t, axis, sign) = if t_enter >= 0.0 {
        (t_enter, enter_axis, enter_sign)
    } else {
        // Origin inside the box: report the exit face.
        let mut exit_axis = 0usize;
        let mut exit_t = f32::INFINITY;
        let mut exit_sign = 1.0f32;
        for axis in 0..3 {
            if mf::abs(dir[axis]) <= 1.0e-20 {
                continue;
            }
            let inv = 1.0 / dir[axis];
            let t1 = (lo[axis] - origin[axis]) * inv;
            let t2 = (hi[axis] - origin[axis]) * inv;
            let (far, s) = if t1 > t2 { (t1, -1.0) } else { (t2, 1.0) };
            if far < exit_t {
                exit_t = far;
                exit_axis = axis;
                exit_sign = s;
            }
        }
        (exit_t, exit_axis, exit_sign)
    };

    let mut normal = Vec3::ZERO;
    match axis {
        0 => normal.x = sign,
        1 => normal.y = sign,
        _ => normal.z = sign,
    }
    Some(RayHit { t, point: ray.at(t), normal })
}

/// Intersect a ray with a plane, returning the hit at non-negative `t`.
///
/// Returns [`None`] when the ray is parallel to the plane. The normal is
/// oriented against the ray direction.
pub fn ray_plane(ray: Ray3, plane: Plane) -> Option<RayHit> {
    let denom = plane.normal.dot(ray.direction);
    if mf::abs(denom) <= 1.0e-20 {
        return None;
    }
    let t = -plane.signed_distance(ray.origin) / denom;
    if t < 0.0 {
        return None;
    }
    let normal = if denom > 0.0 { -plane.normal } else { plane.normal };
    Some(RayHit { t, point: ray.at(t), normal })
}

/// Intersect a ray with a triangle via the Möller-Trumbore algorithm,
/// returning the barycentric coordinates `(t, u, v)`.
///
/// `u` and `v` are the weights of `b` and `c` respectively; the weight of `a`
/// is `1 - u - v`. Returns [`None`] on a miss, a back-parallel ray, or a
/// degenerate triangle. Both faces are considered hittable.
pub fn ray_triangle_bary(ray: Ray3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1.0e-8;
    let edge1 = b - a;
    let edge2 = c - a;
    let pvec = ray.direction.cross(edge2);
    let det = edge1.dot(pvec);
    if mf::abs(det) < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = ray.origin - a;
    let u = tvec.dot(pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = tvec.cross(edge1);
    let v = ray.direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(qvec) * inv_det;
    if t < 0.0 {
        return None;
    }
    Some((t, u, v))
}

/// Intersect a ray with a triangle, returning a [`RayHit`] with a geometric
/// normal oriented against the ray.
pub fn ray_triangle(ray: Ray3, a: Vec3, b: Vec3, c: Vec3) -> Option<RayHit> {
    let (t, _u, _v) = ray_triangle_bary(ray, a, b, c)?;
    let mut normal = (b - a).cross(c - a).normalize();
    if normal.dot(ray.direction) > 0.0 {
        normal = -normal;
    }
    Some(RayHit { t, point: ray.at(t), normal })
}

/// True if two axis-aligned boxes overlap (touching counts as overlap).
#[inline]
pub fn aabb_aabb(a: Aabb3, b: Aabb3) -> bool {
    a.min.x <= b.max.x
        && a.max.x >= b.min.x
        && a.min.y <= b.max.y
        && a.max.y >= b.min.y
        && a.min.z <= b.max.z
        && a.max.z >= b.min.z
}

/// True if two spheres overlap (touching counts as overlap).
#[inline]
pub fn sphere_sphere(a: BoundingSphere, b: BoundingSphere) -> bool {
    let r = a.radius + b.radius;
    (a.center - b.center).length_squared() <= r * r
}

/// True if a sphere overlaps an axis-aligned box.
#[inline]
pub fn sphere_aabb(sphere: BoundingSphere, aabb: Aabb3) -> bool {
    aabb.distance_squared(sphere.center) <= sphere.radius * sphere.radius
}

/// Classify a sphere against a frustum for culling.
pub fn frustum_sphere(frustum: &Frustum, sphere: BoundingSphere) -> Containment {
    let mut result = Containment::Inside;
    for plane in &frustum.planes {
        let dist = plane.signed_distance(sphere.center);
        if dist < -sphere.radius {
            return Containment::Outside;
        }
        if dist < sphere.radius {
            result = Containment::Intersecting;
        }
    }
    result
}

/// Classify an axis-aligned box against a frustum for culling using the
/// positive/negative vertex ("p-vertex / n-vertex") test.
pub fn frustum_aabb(frustum: &Frustum, aabb: Aabb3) -> Containment {
    let center = aabb.center();
    let extent = aabb.half_extents();
    let mut result = Containment::Inside;
    for plane in &frustum.planes {
        let n = plane.normal;
        // Signed distance of the box center, plus the projection radius of the
        // box onto the plane normal.
        let r = extent.x * mf::abs(n.x) + extent.y * mf::abs(n.y) + extent.z * mf::abs(n.z);
        let s = plane.signed_distance(center);
        if s < -r {
            return Containment::Outside;
        }
        if s < r {
            result = Containment::Intersecting;
        }
    }
    result
}

/// Convenience: true if `aabb` is at least partially inside `frustum`.
#[inline]
pub fn frustum_intersects_aabb(frustum: &Frustum, aabb: Aabb3) -> bool {
    frustum_aabb(frustum, aabb).is_visible()
}

/// Convenience: true if `sphere` is at least partially inside `frustum`.
#[inline]
pub fn frustum_intersects_sphere(frustum: &Frustum, sphere: BoundingSphere) -> bool {
    frustum_sphere(frustum, sphere).is_visible()
}
