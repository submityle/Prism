//! Ray-versus-shape intersection primitives for spatial queries.
//!
//! Each analytic [`ColliderShape`] is intersected against a
//! [`Ray`] in the shape's own local frame: the ray is transformed by the
//! inverse body pose, the closed-form intersection is solved locally, and the
//! resulting time-of-impact, contact point, and outward surface normal are
//! mapped back into world space. Planes are handled directly in world space
//! because they are unbounded half-spaces.
//!
//! The routines return the *first* forward intersection (smallest non-negative
//! parametric distance) within the ray's [`Ray::tmax`]. A ray that starts
//! inside a finite shape reports a time-of-impact of zero.
//!
//! # Provenance
//!
//! The ray/sphere, slab-based ray/box, and ray/capsule (cylinder plus spherical
//! caps) intersection tests are standard, publicly documented
//! computational-geometry techniques (see Ericson, *Real-Time Collision
//! Detection*, and the classic analytic capsule intersection). This file
//! contains no Unreal Engine source or derived code.

use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use glam::Vec3;
use prism_physics_geometry::Ray;

/// Numerical tolerance for near-parallel and degenerate-direction guards.
const RAY_EPS: f32 = 1e-6;

/// The local result of a ray/shape intersection: parametric distance and the
/// outward surface normal at the hit, both in the frame the ray was expressed
/// in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RayShapeHit {
    /// Parametric distance along the ray direction to the first intersection.
    pub time_of_impact: f32,
    /// World-space intersection point.
    pub point: Vec3,
    /// Outward unit surface normal at the intersection.
    pub normal: Vec3,
}

/// Intersects `ray` (world space) against `shape` posed by `pose`.
///
/// Returns the nearest forward hit within `ray.tmax`, or [`None`] on a miss.
pub(crate) fn raycast_shape(
    shape: &ColliderShape,
    pose: &Isometry,
    ray: &Ray,
) -> Option<RayShapeHit> {
    match *shape {
        ColliderShape::Plane { normal, offset } => ray_plane_world(ray, pose, normal, offset),
        ColliderShape::Sphere { radius } => {
            let (o, d) = to_local(ray, pose);
            let (t, n_local) = ray_sphere_at(o, d, Vec3::ZERO, radius, ray.tmax)?;
            Some(finish(ray, pose, t, n_local))
        }
        ColliderShape::Cuboid { half_extents } => {
            let (o, d) = to_local(ray, pose);
            let (t, n_local) = ray_cuboid_local(o, d, half_extents, ray.tmax)?;
            Some(finish(ray, pose, t, n_local))
        }
        ColliderShape::Capsule {
            half_height,
            radius,
        } => {
            let (o, d) = to_local(ray, pose);
            let a = Vec3::new(0.0, -half_height, 0.0);
            let b = Vec3::new(0.0, half_height, 0.0);
            let (t, n_local) = ray_capsule(o, d, a, b, radius, ray.tmax)?;
            Some(finish(ray, pose, t, n_local))
        }
        // Mesh colliders store only an arena handle, so they cannot be solved
        // from the `ColliderShape` alone. The registry-aware
        // [`dispatch`](crate::query::dispatch) layer resolves the handle and
        // routes them to [`crate::query::mesh`]; this analytic leaf is never
        // called for them. The arm exists only to keep the match exhaustive.
        ColliderShape::ConvexHull { .. } | ColliderShape::TriangleMesh { .. } => None,
    }
}

/// Transforms a world-space ray into the shape-local frame of `pose`.
///
/// The returned direction is still unit length because `pose` is a rigid
/// transform (rotation only affects directions).
fn to_local(ray: &Ray, pose: &Isometry) -> (Vec3, Vec3) {
    let inv = pose.inverse();
    let o = inv.transform_point(ray.origin);
    let d = inv.transform_vector(ray.dir);
    (o, d)
}

/// Builds the world-space [`RayShapeHit`] from a local time-of-impact and a
/// local outward normal.
fn finish(ray: &Ray, pose: &Isometry, t: f32, normal_local: Vec3) -> RayShapeHit {
    RayShapeHit {
        time_of_impact: t,
        point: ray.at(t),
        normal: pose.transform_vector(normal_local).normalize_or_zero(),
    }
}

/// Intersects a ray with a sphere of radius `radius` centered at `center`,
/// all in one frame.
///
/// Returns the parametric distance and the outward normal at the hit. A ray
/// origin inside the sphere yields a time-of-impact of zero.
pub(crate) fn ray_sphere_at(
    origin: Vec3,
    dir: Vec3,
    center: Vec3,
    radius: f32,
    tmax: f32,
) -> Option<(f32, Vec3)> {
    let m = origin - center;
    let b = m.dot(dir);
    let c = m.dot(m) - radius * radius;
    // Origin outside the sphere and ray pointing away: no hit.
    if c > 0.0 && b > 0.0 {
        return None;
    }
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let mut t = -b - sqrt_disc;
    if t < 0.0 {
        // Origin is inside the sphere: the surface is at distance zero.
        t = 0.0;
    }
    if t > tmax {
        return None;
    }
    let point = origin + dir * t;
    let normal = (point - center).normalize_or(-dir);
    Some((t, normal))
}

/// Intersects a ray with an axis-aligned box of half-extents `half` centered at
/// the local origin, using the slab method.
///
/// Returns the parametric distance and the outward face normal. A ray starting
/// inside the box reports a time-of-impact of zero with a normal opposing the
/// ray direction.
pub(crate) fn ray_cuboid_local(
    origin: Vec3,
    dir: Vec3,
    half: Vec3,
    tmax: f32,
) -> Option<(f32, Vec3)> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    let mut near_axis = 0usize;
    let mut near_sign = 0.0f32;

    let o = [origin.x, origin.y, origin.z];
    let d = [dir.x, dir.y, dir.z];
    let h = [half.x, half.y, half.z];

    for axis in 0..3 {
        if d[axis].abs() < RAY_EPS {
            // Ray parallel to this slab: miss if the origin is outside it.
            if o[axis] < -h[axis] || o[axis] > h[axis] {
                return None;
            }
        } else {
            let inv = 1.0 / d[axis];
            let mut t1 = (-h[axis] - o[axis]) * inv;
            let mut t2 = (h[axis] - o[axis]) * inv;
            let mut sign = -1.0;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
                sign = 1.0;
            }
            if t1 > t_near {
                t_near = t1;
                near_axis = axis;
                near_sign = sign;
            }
            if t2 < t_far {
                t_far = t2;
            }
            if t_near > t_far {
                return None;
            }
        }
    }

    if t_far < 0.0 {
        return None;
    }
    if t_near < 0.0 {
        // Origin inside the box: report the surface at distance zero.
        return Some((0.0, -dir.normalize_or(Vec3::Y)));
    }
    if t_near > tmax {
        return None;
    }
    let mut normal = Vec3::ZERO;
    match near_axis {
        0 => normal.x = near_sign,
        1 => normal.y = near_sign,
        _ => normal.z = near_sign,
    }
    Some((t_near, normal))
}

/// Intersects a ray with a capsule defined by the segment `a` to `b` and radius
/// `radius`, all in one frame.
///
/// The capsule is the union of a cylinder around the segment and two spherical
/// end caps. Returns the parametric distance and the outward normal at the
/// first forward hit.
pub(crate) fn ray_capsule(
    origin: Vec3,
    dir: Vec3,
    a: Vec3,
    b: Vec3,
    radius: f32,
    tmax: f32,
) -> Option<(f32, Vec3)> {
    let ba = b - a;
    let oa = origin - a;
    let baba = ba.dot(ba);
    let bard = ba.dot(dir);
    let baoa = ba.dot(oa);
    let rdoa = dir.dot(oa);
    let oaoa = oa.dot(oa);

    let mut best: Option<f32> = None;

    // Cylinder body: solve the quadratic for the infinite cylinder around the
    // segment, then accept only hits whose axial coordinate lies on the segment.
    let aa = baba - bard * bard;
    if aa.abs() > RAY_EPS {
        let bb = baba * rdoa - baoa * bard;
        let cc = baba * oaoa - baoa * baoa - radius * radius * baba;
        let disc = bb * bb - aa * cc;
        if disc >= 0.0 {
            let t = (-bb - disc.sqrt()) / aa;
            let y = baoa + t * bard;
            if t >= 0.0 && t <= tmax && y >= 0.0 && y <= baba {
                best = Some(t);
            }
        }
    }

    // End caps: intersect the two bounding spheres and keep the nearer valid one.
    if let Some((t, _)) = ray_sphere_at(origin, dir, a, radius, tmax) {
        best = Some(best.map_or(t, |cur| cur.min(t)));
    }
    if let Some((t, _)) = ray_sphere_at(origin, dir, b, radius, tmax) {
        best = Some(best.map_or(t, |cur| cur.min(t)));
    }

    let t = best?;
    let point = origin + dir * t;
    let normal = capsule_normal(point, a, ba, baba);
    Some((t, normal))
}

/// Computes the outward capsule surface normal at `point` for the segment
/// starting at `a` with direction `ba` (whose squared length is `baba`).
fn capsule_normal(point: Vec3, a: Vec3, ba: Vec3, baba: f32) -> Vec3 {
    let h = if baba > RAY_EPS {
        ((point - a).dot(ba) / baba).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let closest = a + ba * h;
    (point - closest).normalize_or(Vec3::Y)
}

/// Intersects a ray with a world-space plane `dot(normal, x) = offset` posed by
/// `pose`.
///
/// The returned normal faces the incoming ray so that queries always see the
/// side they approach from.
fn ray_plane_world(ray: &Ray, pose: &Isometry, normal: Vec3, offset: f32) -> Option<RayShapeHit> {
    let n = pose.transform_vector(normal).normalize_or_zero();
    if n == Vec3::ZERO {
        return None;
    }
    let point_on_plane = pose.transform_point(normal * offset);
    let plane_d = n.dot(point_on_plane);
    let denom = ray.dir.dot(n);
    if denom.abs() < RAY_EPS {
        return None;
    }
    let t = (plane_d - ray.origin.dot(n)) / denom;
    if t < 0.0 || t > ray.tmax {
        return None;
    }
    let facing = if denom > 0.0 { -n } else { n };
    Some(RayShapeHit {
        time_of_impact: t,
        point: ray.at(t),
        normal: facing,
    })
}
