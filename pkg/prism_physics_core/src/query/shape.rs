//! Sphere-sweep (spherecast) queries against analytic shapes.
//!
//! A spherecast moves a sphere of a given radius along a ray and reports the
//! first shape it touches, together with the world-space contact point on the
//! shape surface and the outward contact normal. This is the query character
//! controllers and cameras use to move without tunnelling through geometry.
//!
//! Each shape reduces the swept sphere to a ray test against an *inflated*
//! shape (a Minkowski sum with the sphere):
//!
//! - sphere: a ray against a sphere grown by the sweep radius;
//! - capsule: a ray against a capsule whose radius is grown by the sweep radius;
//! - plane: a ray against a plane offset outward by the sweep radius;
//! - box: a ray against a *rounded* box, resolved by a slab test against the
//!   grown box followed by edge (capsule) and corner (sphere) refinement.
//!
//! # Provenance
//!
//! The inflated-shape reduction and the rounded-box sweep (slab test plus
//! edge/corner Voronoi-region refinement) are standard, publicly documented
//! swept-collision techniques (see Ericson, *Real-Time Collision Detection*).
//! This file contains no Unreal Engine source or derived code.

use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use crate::query::ray::{ray_capsule, ray_sphere_at, RayShapeHit};
use glam::Vec3;
use prism_physics_geometry::Ray;

/// Numerical tolerance for near-parallel and degenerate guards.
const SWEEP_EPS: f32 = 1e-6;

/// Sweeps a sphere of radius `radius` along `ray` against `shape` posed by
/// `pose`.
///
/// Returns the first contact within `ray.tmax`: the distance travelled by the
/// sphere center, the world-space contact point on the shape surface, and the
/// outward contact normal.
pub(crate) fn spherecast_shape(
    shape: &ColliderShape,
    pose: &Isometry,
    ray: &Ray,
    radius: f32,
) -> Option<RayShapeHit> {
    match *shape {
        ColliderShape::Plane { normal, offset } => {
            spherecast_plane_world(ray, pose, normal, offset, radius)
        }
        ColliderShape::Sphere { radius: r } => {
            let (o, d) = to_local(ray, pose);
            let (t, normal) = ray_sphere_at(o, d, Vec3::ZERO, r + radius, ray.tmax)?;
            let contact_local = normal * r;
            Some(finish(pose, t, contact_local, normal))
        }
        ColliderShape::Capsule {
            half_height,
            radius: r,
        } => {
            let (o, d) = to_local(ray, pose);
            let a = Vec3::new(0.0, -half_height, 0.0);
            let b = Vec3::new(0.0, half_height, 0.0);
            let (t, normal) = ray_capsule(o, d, a, b, r + radius, ray.tmax)?;
            let center_at_toi = o + d * t;
            let contact_local = center_at_toi - normal * radius;
            Some(finish(pose, t, contact_local, normal))
        }
        ColliderShape::Cuboid { half_extents } => {
            let (o, d) = to_local(ray, pose);
            let (t, normal, contact_local) =
                spherecast_cuboid_local(o, d, half_extents, radius, ray.tmax)?;
            Some(finish(pose, t, contact_local, normal))
        }
    }
}

/// Transforms a world-space ray into the shape-local frame of `pose`.
fn to_local(ray: &Ray, pose: &Isometry) -> (Vec3, Vec3) {
    let inv = pose.inverse();
    (
        inv.transform_point(ray.origin),
        inv.transform_vector(ray.dir),
    )
}

/// Builds the world-space [`RayShapeHit`] from a local contact point and normal.
fn finish(pose: &Isometry, t: f32, contact_local: Vec3, normal_local: Vec3) -> RayShapeHit {
    RayShapeHit {
        time_of_impact: t,
        point: pose.transform_point(contact_local),
        normal: pose.transform_vector(normal_local).normalize_or_zero(),
    }
}

/// Sweeps a sphere of radius `radius` against a world-space plane.
fn spherecast_plane_world(
    ray: &Ray,
    pose: &Isometry,
    normal: Vec3,
    offset: f32,
    radius: f32,
) -> Option<RayShapeHit> {
    let n = pose.transform_vector(normal).normalize_or_zero();
    if n == Vec3::ZERO {
        return None;
    }
    let point_on_plane = pose.transform_point(normal * offset);
    let plane_d = n.dot(point_on_plane);
    let start = ray.origin.dot(n) - plane_d;
    let denom = ray.dir.dot(n);
    // The sphere center must reach a signed distance of +/-radius from the plane
    // on the side it starts. Choose the facing side by the sign of the start.
    let target = radius * start.signum();
    if denom.abs() < SWEEP_EPS {
        return None;
    }
    let t = (target - start) / denom;
    if t < 0.0 || t > ray.tmax {
        return None;
    }
    let facing = if start >= 0.0 { n } else { -n };
    let center_at_toi = ray.at(t);
    Some(RayShapeHit {
        time_of_impact: t,
        point: center_at_toi - facing * radius,
        normal: facing,
    })
}

/// Sweeps a sphere of radius `radius` against a local axis-aligned box of
/// half-extents `half`.
///
/// Returns the sweep distance, the outward contact normal, and the contact
/// point on the box surface, all in the box-local frame.
fn spherecast_cuboid_local(
    origin: Vec3,
    dir: Vec3,
    half: Vec3,
    radius: f32,
    tmax: f32,
) -> Option<(f32, Vec3, Vec3)> {
    let grown = half + Vec3::splat(radius);
    let (t_near, axis, sign) = slab_enter(origin, dir, grown, tmax)?;

    // The sphere center starts already within the grown box: treat as an
    // immediate contact and derive the normal from the closest box face.
    if t_near <= 0.0 {
        let clamped = origin.clamp(-half, half);
        let normal = (origin - clamped).normalize_or(Vec3::Y);
        return Some((0.0, normal, clamped));
    }

    let center = origin + dir * t_near;
    // Count how many axes lie outside the original box: 1 => face, 2 => edge,
    // 3 => corner.
    let outside = [
        center.x.abs() > half.x,
        center.y.abs() > half.y,
        center.z.abs() > half.z,
    ];
    let count = outside.iter().filter(|&&o| o).count();

    match count {
        0 | 1 => {
            let contact = center.clamp(-half, half);
            let mut normal = Vec3::ZERO;
            match axis {
                0 => normal.x = sign,
                1 => normal.y = sign,
                _ => normal.z = sign,
            }
            Some((t_near, normal, contact))
        }
        2 => {
            let (a, b) = box_edge(center, half, &outside);
            let (t, normal) = ray_capsule(origin, dir, a, b, radius, tmax)?;
            let contact = closest_on_segment(origin + dir * t, a, b);
            Some((t, normal, contact))
        }
        _ => {
            let corner = Vec3::new(
                sign_extent(center.x, half.x),
                sign_extent(center.y, half.y),
                sign_extent(center.z, half.z),
            );
            let (t, normal) = ray_sphere_at(origin, dir, corner, radius, tmax)?;
            Some((t, normal, corner))
        }
    }
}

/// Runs a slab test against a box of half-extents `half` centered at the origin
/// and returns the entry distance with the entering face axis and its sign.
fn slab_enter(origin: Vec3, dir: Vec3, half: Vec3, tmax: f32) -> Option<(f32, usize, f32)> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    let mut near_axis = 0usize;
    let mut near_sign = 0.0f32;

    let o = [origin.x, origin.y, origin.z];
    let d = [dir.x, dir.y, dir.z];
    let h = [half.x, half.y, half.z];

    for axis in 0..3 {
        if d[axis].abs() < SWEEP_EPS {
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
    if t_far < 0.0 || t_near > tmax {
        return None;
    }
    Some((t_near, near_axis, near_sign))
}

/// Builds the box edge (as a segment) touched when the swept sphere center is
/// outside the box on exactly the two axes flagged in `outside`.
fn box_edge(center: Vec3, half: Vec3, outside: &[bool; 3]) -> (Vec3, Vec3) {
    let mut a = Vec3::ZERO;
    let mut b = Vec3::ZERO;
    let c = [center.x, center.y, center.z];
    let h = [half.x, half.y, half.z];
    let mut lo = [0.0f32; 3];
    let mut hi = [0.0f32; 3];
    for axis in 0..3 {
        if outside[axis] {
            let s = if c[axis] < 0.0 { -h[axis] } else { h[axis] };
            lo[axis] = s;
            hi[axis] = s;
        } else {
            // The free axis spans the full box edge.
            lo[axis] = -h[axis];
            hi[axis] = h[axis];
        }
    }
    a.x = lo[0];
    a.y = lo[1];
    a.z = lo[2];
    b.x = hi[0];
    b.y = hi[1];
    b.z = hi[2];
    (a, b)
}

/// Returns the box-face coordinate (`+half` or `-half`) on the side of `value`.
fn sign_extent(value: f32, half: f32) -> f32 {
    if value < 0.0 {
        -half
    } else {
        half
    }
}

/// Returns the closest point on the segment `a` to `b` from `point`.
fn closest_on_segment(point: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 < SWEEP_EPS {
        return a;
    }
    let t = ((point - a).dot(ab) / len2).clamp(0.0, 1.0);
    a + ab * t
}
