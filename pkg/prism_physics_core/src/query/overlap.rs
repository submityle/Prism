//! Overlap predicates for spatial queries.
//!
//! Overlap tests answer "does this region touch this shape?" without producing
//! a full contact manifold for the caller. Arbitrary shape-versus-shape overlap
//! reuses the narrow-phase [`generate_contact`] dispatch so that the query path
//! and the simulation path agree exactly on what "touching" means. Axis-aligned
//! box overlap is a fast conservative test used for region culling.
//!
//! # Provenance
//!
//! These predicates compose the existing narrow-phase routines and standard
//! world-space bounding-box construction; they contain no Unreal Engine source
//! or derived code.

use crate::collide::generate_contact;
use crate::collider::ColliderShape;
use crate::math::transform::Isometry;
use glam::Vec3;
use prism_physics_geometry::Aabb;

/// Returns `true` when the two posed shapes overlap or exactly touch.
///
/// This delegates to the narrow-phase contact generator, so the notion of
/// "overlap" is identical to the one the solver resolves.
pub(crate) fn shapes_overlap(
    shape_a: &ColliderShape,
    pose_a: &Isometry,
    shape_b: &ColliderShape,
    pose_b: &Isometry,
) -> bool {
    generate_contact(shape_a, pose_a, shape_b, pose_b).is_some()
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
