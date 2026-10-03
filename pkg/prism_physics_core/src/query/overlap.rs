//! Overlap predicates for spatial queries.
//!
//! Overlap tests answer "does this region touch this shape?" without producing
//! a full contact manifold for the caller. Arbitrary shape-versus-shape overlap
//! reuses the narrow-phase [`generate_contact_in`] dispatch so that the query path
//! and the simulation path agree exactly on what "touching" means. Axis-aligned
//! box overlap is a fast conservative test used for region culling.
//!
//! # Provenance
//!
//! These predicates compose the existing narrow-phase routines and standard
//! world-space bounding-box construction; they contain no Unreal Engine source
//! or derived code.

use crate::collide::generate_contact_in;
use crate::collider::{ColliderShape, ShapeRegistry};
use crate::math::transform::Isometry;
use glam::Vec3;
use prism_physics_geometry::Aabb;

/// Registry-aware overlap test for any shape pair, including the mesh collider
/// families whose geometry lives in `shapes`.
///
/// Delegates to the registry-aware narrow phase
/// ([`generate_contact_in`]) so query-time overlap and solver-time contact
/// generation agree exactly, even when one or both shapes are a
/// [`ColliderShape::ConvexHull`] or [`ColliderShape::TriangleMesh`].
pub(crate) fn shapes_overlap_in(
    shape_a: &ColliderShape,
    pose_a: &Isometry,
    shape_b: &ColliderShape,
    pose_b: &Isometry,
    shapes: &ShapeRegistry,
) -> bool {
    generate_contact_in(shape_a, pose_a, shape_b, pose_b, shapes).is_some()
}

/// Computes the world-space axis-aligned bounding box of `shape` posed by
/// `pose`.
///
/// A [`ColliderShape::Plane`] is unbounded and reports a box spanning the full
/// finite float range so that region tests treat it as covering everything.
pub(crate) fn world_shape_aabb(shape: &ColliderShape, pose: &Isometry) -> Aabb {
    if matches!(shape, ColliderShape::Plane { .. }) {
        let big = Vec3::splat(f32::MAX);
        return Aabb::new(-big, big);
    }
    let (min, max) = shape.local_aabb();
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let world: [Vec3; 8] = core::array::from_fn(|i| pose.transform_point(corners[i]));
    Aabb::from_points(&world).unwrap_or_else(|| Aabb::new(pose.translation, pose.translation))
}
