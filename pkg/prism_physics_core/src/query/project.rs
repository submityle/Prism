//! Closest-point projection of a query point onto analytic shapes.
//!
//! [`project_point_shape`] maps a world-space point into a shape's local frame,
//! finds the nearest point on that shape's surface, and reports whether the
//! query point lies inside the shape. Results are transformed back into world
//! space. This underpins nearest-surface queries used by gameplay code such as
//! snapping, proximity checks, and distance fields.
//!
//! # Provenance
//!
//! Point-to-primitive closest-point formulas (sphere radial projection, box
//! clamp with interior face selection, segment projection for capsules, and
//! plane projection) are standard computational-geometry results and contain no
//! Unreal Engine source or derived code.

use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use glam::Vec3;

/// Numerical tolerance for degenerate-direction guards.
const PROJ_EPS: f32 = 1e-6;

/// The local outcome of projecting a point onto a shape: the nearest surface
/// point, the outward normal there, and whether the query point was inside.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShapeProjection {
    /// World-space nearest point on the shape surface.
    pub point: Vec3,
    /// Outward unit surface normal at [`ShapeProjection::point`].
    pub normal: Vec3,
    /// Unsigned distance from the query point to the surface.
    pub distance: f32,
    /// Whether the query point lies inside (or on) the shape.
    pub is_inside: bool,
}

/// Projects the world-space `point` onto `shape` posed by `pose`.
pub(crate) fn project_point_shape(
    shape: &ColliderShape,
    pose: &Isometry,
    point: Vec3,
) -> ShapeProjection {
    let inv = pose.inverse();
    let local = inv.transform_point(point);
    let (surface_local, normal_local, is_inside) = match *shape {
        ColliderShape::Sphere { radius } => project_sphere(local, radius),
        ColliderShape::Cuboid { half_extents } => project_cuboid(local, half_extents),
        ColliderShape::Capsule {
            half_height,
            radius,
        } => project_capsule(local, half_height, radius),
        ColliderShape::Plane { normal, offset } => project_plane(local, normal, offset),
    };
    let surface = pose.transform_point(surface_local);
    let normal = pose.transform_vector(normal_local).normalize_or_zero();
    ShapeProjection {
        point: surface,
        normal,
        distance: (point - surface).length(),
        is_inside,
    }
}

/// Projects a local point onto a sphere of radius `radius` at the origin.
fn project_sphere(local: Vec3, radius: f32) -> (Vec3, Vec3, bool) {
    let len = local.length();
    let dir = if len > PROJ_EPS { local / len } else { Vec3::X };
    (dir * radius, dir, len <= radius)
}

/// Projects a local point onto an axis-aligned box of half-extents `half`.
///
/// When the point is inside the box it is pushed out through the nearest face;
/// otherwise it is clamped to the box surface.
fn project_cuboid(local: Vec3, half: Vec3) -> (Vec3, Vec3, bool) {
    let clamped = local.clamp(-half, half);
    let inside = clamped == local;
    if inside {
        // Interior point: exit through the face with the least penetration.
        let dist = half - local.abs();
        let (surface, normal) = if dist.x <= dist.y && dist.x <= dist.z {
            let sign = sign_of(local.x);
            (
                Vec3::new(sign * half.x, local.y, local.z),
                Vec3::new(sign, 0.0, 0.0),
            )
        } else if dist.y <= dist.z {
            let sign = sign_of(local.y);
            (
                Vec3::new(local.x, sign * half.y, local.z),
                Vec3::new(0.0, sign, 0.0),
            )
        } else {
            let sign = sign_of(local.z);
            (
                Vec3::new(local.x, local.y, sign * half.z),
                Vec3::new(0.0, 0.0, sign),
            )
        };
        (surface, normal, true)
    } else {
        let normal = (local - clamped).normalize_or(Vec3::Y);
        (clamped, normal, false)
    }
}

/// Projects a local point onto a capsule aligned with the local Y axis.
fn project_capsule(local: Vec3, half_height: f32, radius: f32) -> (Vec3, Vec3, bool) {
    let axis_y = local.y.clamp(-half_height, half_height);
    let axis_point = Vec3::new(0.0, axis_y, 0.0);
    let delta = local - axis_point;
    let len = delta.length();
    let normal = if len > PROJ_EPS { delta / len } else { Vec3::X };
    (axis_point + normal * radius, normal, len <= radius)
}

/// Projects a local point onto a plane `dot(normal, x) = offset`.
fn project_plane(local: Vec3, normal: Vec3, offset: f32) -> (Vec3, Vec3, bool) {
    let n = normal.normalize_or(Vec3::Y);
    let signed = local.dot(n) - offset;
    let surface = local - n * signed;
    // The plane bounds a half-space; a point below it (negative side) is inside.
    (surface, n, signed <= 0.0)
}

/// Returns `1.0` for non-negative inputs and `-1.0` otherwise.
fn sign_of(value: f32) -> f32 {
    if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}
