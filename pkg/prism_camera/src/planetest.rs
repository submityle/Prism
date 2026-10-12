//! Inside / outside / intersecting classification of bounding volumes against
//! the six planes of a camera frustum.
//!
//! The classification is the visibility decision a renderer makes per
//! instance: [`Containment::Outside`] is safe to cull, while
//! [`Containment::Inside`] and [`Containment::Intersecting`] are visible (the
//! latter conservatively, since the volume crosses a boundary).

use prism_math::geom::{Aabb3, BoundingSphere, Frustum as GeomFrustum, Plane};
use prism_math::intersect::{frustum_aabb, frustum_sphere};
use prism_math::Vec3;

pub use prism_math::intersect::Containment;

/// Classify a sphere against six inward-facing planes.
///
/// The planes must already be normalized so signed distances are Euclidean.
#[inline]
#[must_use]
pub fn classify_sphere(planes: &[Plane; 6], center: Vec3, radius: f32) -> Containment {
    frustum_sphere(
        &GeomFrustum::from_planes(*planes),
        BoundingSphere::new(center, radius),
    )
}

/// Classify an axis-aligned box, given by its `min`/`max` corners, against six
/// inward-facing planes.
///
/// The planes must already be normalized so signed distances are Euclidean.
/// The corners are sorted, so they may be supplied in any order.
#[inline]
#[must_use]
pub fn classify_aabb(planes: &[Plane; 6], min: Vec3, max: Vec3) -> Containment {
    frustum_aabb(&GeomFrustum::from_planes(*planes), Aabb3::new(min, max))
}
