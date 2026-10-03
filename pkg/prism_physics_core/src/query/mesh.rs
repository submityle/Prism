//! Registry-resolved scene queries against the two mesh collider families:
//! bounded [`ConvexHull`](crate::collider::ColliderShape::ConvexHull) polytopes
//! and concave [`TriangleMesh`](crate::collider::ColliderShape::TriangleMesh)
//! surfaces.
//!
//! The analytic leaves ([`raycast_shape`](crate::query::ray::raycast_shape),
//! [`project_point_shape`](crate::query::project::project_point_shape), ...) match
//! on a [`ColliderShape`](crate::collider::ColliderShape) alone, but a mesh
//! collider only stores an arena *handle*; its vertices live in the owning
//! [`ShapeRegistry`](crate::collider::ShapeRegistry). The registry-aware
//! [`dispatch`](crate::query::dispatch) layer resolves the handle to the backing
//! [`ConvexMeshData`] / [`TriangleMesh`] and calls the leaves here, which take
//! the already-resolved geometry plus the collider's world pose.
//!
//! Every function works in the shape-local frame -- the world-space query is
//! pulled back through `pose.inverse()`, solved locally by the geometry crate,
//! and the witness pushed back out to world space. Because `pose` is a rigid
//! isometry (rotation + translation, no scale) the parametric ray distance and
//! the point-to-surface distance are preserved, so only points and directions
//! need transforming.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It is a
//! thin pose adapter over this workspace's own convex-polytope and
//! triangle-mesh geometry queries.

use crate::collider::convex_mesh::ConvexMeshData;
use crate::math::transform::Isometry;
use crate::query::project::ShapeProjection;
use crate::query::ray::RayShapeHit;
use glam::Vec3;
use prism_physics_geometry::{Ray, TriangleMesh};

/// Transforms a world-space ray into the shape-local frame of `pose`.
///
/// `pose` is rigid, so the direction stays unit length and parametric distances
/// along the ray are identical in both frames.
fn ray_to_local(ray: &Ray, pose: &Isometry) -> Ray {
    let inv = pose.inverse();
    let origin = inv.transform_point(ray.origin);
    let dir = inv.transform_vector(ray.dir);
    let mut local = Ray::new(origin, dir);
    local.tmax = ray.tmax;
    local
}

/// Casts `ray` against the convex hull `mesh` posed by `pose`.
///
/// Returns the nearest forward hit within `ray.tmax`, or [`None`] on a miss.
/// The reported normal is the hull's outward surface normal in world space.
#[must_use]
pub(crate) fn raycast_convex(
    mesh: &ConvexMeshData,
    pose: &Isometry,
    ray: &Ray,
) -> Option<RayShapeHit> {
    let local = ray_to_local(ray, pose);
    let hit = mesh.ray_cast(local.origin, local.dir, local.tmax)?;
    Some(RayShapeHit {
        time_of_impact: hit.time,
        point: pose.transform_point(hit.point),
        normal: pose.transform_vector(hit.normal).normalize_or_zero(),
    })
}

/// Casts `ray` against the triangle mesh `mesh` posed by `pose`.
///
/// Returns the nearest forward triangle hit within `ray.tmax`, or [`None`] on a
/// miss. The reported normal is the triangle's geometric face normal in world
/// space.
#[must_use]
pub(crate) fn raycast_trimesh(
    mesh: &TriangleMesh,
    pose: &Isometry,
    ray: &Ray,
) -> Option<RayShapeHit> {
    let local = ray_to_local(ray, pose);
    let hit = mesh.ray_cast(&local)?;
    Some(RayShapeHit {
        time_of_impact: hit.t,
        point: pose.transform_point(hit.point),
        normal: pose.transform_vector(hit.normal).normalize_or_zero(),
    })
}

/// Projects the world-space `point` onto the convex hull `mesh` posed by
/// `pose`.
#[must_use]
pub(crate) fn project_convex(
    mesh: &ConvexMeshData,
    pose: &Isometry,
    point: Vec3,
) -> ShapeProjection {
    let inv = pose.inverse();
    let local_point = inv.transform_point(point);
    let proj = mesh.project_point(local_point);
    ShapeProjection {
        point: pose.transform_point(proj.point),
        normal: pose.transform_vector(proj.normal).normalize_or_zero(),
        distance: proj.distance,
        is_inside: proj.is_inside,
    }
}

/// Projects the world-space `point` onto the triangle mesh `mesh` posed by
/// `pose`, returning [`None`] only for an empty mesh.
///
/// A triangle mesh is a surface, not a solid, so the projection never reports
/// `is_inside`; the outward normal is the direction from the surface toward the
/// query point (its face normal when the point lies off to one side).
#[must_use]
pub(crate) fn project_trimesh(
    mesh: &TriangleMesh,
    pose: &Isometry,
    point: Vec3,
) -> Option<ShapeProjection> {
    let inv = pose.inverse();
    let local_point = inv.transform_point(point);
    let closest = mesh.closest_point(local_point)?;
    let surface = pose.transform_point(closest.point);
    let to_point = point - surface;
    let normal = if to_point.length_squared() > 1.0e-12 {
        to_point.normalize()
    } else {
        // The query point lies on the surface: fall back to the triangle's own
        // face normal so the caller still gets a meaningful outward direction.
        mesh.triangle(closest.triangle as usize)
            .map(|[a, b, c]| {
                pose.transform_vector((b - a).cross(c - a))
                    .normalize_or_zero()
            })
            .unwrap_or(Vec3::Y)
    };
    Some(ShapeProjection {
        point: surface,
        normal,
        distance: closest.distance,
        is_inside: false,
    })
}

/// Sweeps a sphere of `radius` along `ray` against the triangle mesh `mesh`
/// posed by `pose`.
///
/// Returns the first triangle contact within `ray.tmax`, or [`None`] on a miss.
/// The reported normal is the triangle surface normal pointing back toward the
/// sphere centre, matching the analytic spherecast convention.
#[must_use]
pub(crate) fn spherecast_trimesh(
    mesh: &TriangleMesh,
    pose: &Isometry,
    ray: &Ray,
    radius: f32,
) -> Option<RayShapeHit> {
    let local = ray_to_local(ray, pose);
    let hit = mesh.sphere_cast(&local, radius)?;
    Some(RayShapeHit {
        time_of_impact: hit.t,
        point: pose.transform_point(hit.point),
        normal: pose.transform_vector(hit.normal).normalize_or_zero(),
    })
}
