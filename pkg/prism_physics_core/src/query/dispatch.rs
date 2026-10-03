//! Registry-aware scene-query dispatch.
//!
//! The analytic query leaves ([`raycast_shape`](crate::query::ray::raycast_shape),
//! [`spherecast_shape`](crate::query::shape::spherecast_shape),
//! [`project_point_shape`](crate::query::project::project_point_shape),
//! [`shapecast_shape`](crate::query::convex_sweep::shapecast_shape)) match on a
//! [`ColliderShape`] alone. That is sufficient for the analytic primitives, but
//! the two mesh collider families
//! ([`ConvexHull`](crate::collider::ColliderShape::ConvexHull) and
//! [`TriangleMesh`](crate::collider::ColliderShape::TriangleMesh)) store only an
//! arena *handle*; their vertices live in the owning
//! [`ShapeRegistry`](crate::collider::ShapeRegistry).
//!
//! This module is the single seam where a query becomes registry-aware. Each
//! `*_in` entry point resolves mesh handles against the registry and routes
//! them to the pose-adapter leaves in [`crate::query::mesh`] (or, for a convex
//! sphere sweep, the shared conservative-advancement reduction), while analytic
//! shapes fall through to the original leaves unchanged. Keeping the resolution
//! here means [`PhysicsWorld`](crate::world::PhysicsWorld) query methods never
//! branch on shape kind themselves.
//!
//! # Provenance
//!
//! This module only composes existing leaves and the workspace's own geometry
//! solver; it contains no Unreal Engine source or derived code.

use crate::ccd::CcdSupport;
use crate::collider::{ColliderShape, ShapeRegistry};
use crate::math::transform::Isometry;
use crate::query::mesh;
use crate::query::project::ShapeProjection;
use crate::query::ray::RayShapeHit;
use glam::Vec3;
use prism_physics_geometry::{conservative_advancement, BoundingSphere, Ray};

/// Separation (world units) at which a swept query reports contact. Mirrors the
/// tolerance used by the convex-sweep and CCD paths so results stay consistent.
const SWEEP_CONTACT_TOLERANCE: f32 = 1.0e-4;

/// Casts `ray` against `shape` posed by `pose`, resolving mesh handles against
/// `shapes`.
///
/// Convex hulls and triangle meshes are resolved to their backing geometry and
/// routed to [`crate::query::mesh`]; analytic shapes defer to
/// [`raycast_shape`](crate::query::ray::raycast_shape).
pub(crate) fn raycast_shape_in(
    shape: &ColliderShape,
    pose: &Isometry,
    ray: &Ray,
    shapes: &ShapeRegistry,
) -> Option<RayShapeHit> {
    match *shape {
        ColliderShape::ConvexHull { mesh: handle, .. } => {
            let data = shapes.convex_mesh(handle)?;
            mesh::raycast_convex(data, pose, ray)
        }
        ColliderShape::TriangleMesh { mesh: handle, .. } => {
            let data = shapes.tri_mesh(handle)?;
            mesh::raycast_trimesh(data.mesh(), pose, ray)
        }
        _ => crate::query::ray::raycast_shape(shape, pose, ray),
    }
}

/// Sweeps a sphere of `radius` along `ray` against `shape` posed by `pose`.
///
/// Triangle meshes route to the dedicated triangle sphere-cast; convex hulls are
/// solved by conservatively advancing a [`BoundingSphere`] mover against the
/// hull support map; analytic shapes defer to
/// [`spherecast_shape`](crate::query::shape::spherecast_shape).
pub(crate) fn spherecast_shape_in(
    shape: &ColliderShape,
    pose: &Isometry,
    ray: &Ray,
    radius: f32,
    shapes: &ShapeRegistry,
) -> Option<RayShapeHit> {
    match *shape {
        ColliderShape::TriangleMesh { mesh: handle, .. } => {
            let data = shapes.tri_mesh(handle)?;
            mesh::spherecast_trimesh(data.mesh(), pose, ray, radius)
        }
        ColliderShape::ConvexHull { .. } => {
            // Model the swept sphere as a `BoundingSphere` mover translated along
            // the ray and conservatively advanced against the hull.
            let target = CcdSupport::from_shape_in(shape, pose.translation, pose.rotation, shapes)?;
            let mover = BoundingSphere::new(ray.origin, radius);
            let motion = ray.dir * ray.tmax;
            let toi = conservative_advancement(&mover, &target, motion, SWEEP_CONTACT_TOLERANCE)?;
            Some(RayShapeHit {
                // `toi.toi` is the fraction of `motion`; the ray direction is
                // unit length, so the parametric distance is `toi.toi * tmax`.
                time_of_impact: toi.toi * ray.tmax,
                point: toi.point,
                // Report the hull's outward normal (toward the sphere centre).
                normal: -toi.normal,
            })
        }
        _ => crate::query::shape::spherecast_shape(shape, pose, ray, radius),
    }
}

/// Sweeps `mover` (posed at `mover_pose`) along `motion` against `target`
/// (posed at `target_pose`), resolving mesh handles against `shapes`.
///
/// This is a thin forward to [`shapecast_shape`](crate::query::convex_sweep::shapecast_shape),
/// which already dispatches convex, plane, and triangle-mesh targets internally.
pub(crate) fn shapecast_shape_in(
    mover: &ColliderShape,
    mover_pose: &Isometry,
    target: &ColliderShape,
    target_pose: &Isometry,
    motion: Vec3,
    shapes: &ShapeRegistry,
) -> Option<RayShapeHit> {
    crate::query::convex_sweep::shapecast_shape(
        mover,
        mover_pose,
        target,
        target_pose,
        motion,
        shapes,
    )
}

/// Projects the world-space `point` onto `shape` posed by `pose`, resolving mesh
/// handles against `shapes`.
///
/// Convex hulls and triangle meshes route to [`crate::query::mesh`]; analytic
/// shapes defer to [`project_point_shape`](crate::query::project::project_point_shape).
/// A missing or empty mesh falls back to a degenerate witness at the query
/// point so callers always receive a projection.
pub(crate) fn project_point_shape_in(
    shape: &ColliderShape,
    pose: &Isometry,
    point: Vec3,
    shapes: &ShapeRegistry,
) -> ShapeProjection {
    match *shape {
        ColliderShape::ConvexHull { mesh: handle, .. } => shapes
            .convex_mesh(handle)
            .map(|data| mesh::project_convex(data, pose, point))
            .unwrap_or_else(|| degenerate_projection(point)),
        ColliderShape::TriangleMesh { mesh: handle, .. } => shapes
            .tri_mesh(handle)
            .and_then(|data| mesh::project_trimesh(data.mesh(), pose, point))
            .unwrap_or_else(|| degenerate_projection(point)),
        _ => crate::query::project::project_point_shape(shape, pose, point),
    }
}

/// Builds a degenerate projection whose witness is the query point itself,
/// returned only when a mesh handle cannot be resolved or the mesh is empty.
fn degenerate_projection(point: Vec3) -> ShapeProjection {
    ShapeProjection {
        point,
        normal: Vec3::Y,
        distance: 0.0,
        is_inside: false,
    }
}
