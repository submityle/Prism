//! Narrow-phase ray/primitive intersection tests.
//!
//! These are exact analytic intersections against individual primitives,
//! intended to refine the candidate leaves reported by the broad-phase BVH.
//! They are engine-agnostic implementations of publicly documented algorithms
//! and contain no Unreal Engine source or derived code.

mod ccd;
mod closest_point;
mod distance;
mod epa;
mod gjk;
mod manifold;
mod minkowski;
mod ray_cast;
mod support;
mod triangle_box;

pub use ccd::{conservative_advancement, TimeOfImpact};
pub use closest_point::{
    closest_point_on_aabb, closest_point_on_segment, closest_point_on_triangle,
    closest_points_segment_segment, SegmentClosest,
};
pub use distance::{gjk_closest_points, ClosestPoints};
pub use epa::{gjk_contact, Contact};
pub use gjk::gjk_intersect;
pub use manifold::{
    contact_manifold, ClipShape, ContactManifold, FacePolygon, ManifoldPoint,
};
pub use ray_cast::{ray_capsule, ray_obb, ray_sphere, ray_triangle, RayTriangleHit};
pub use support::{Inflated, SupportMap, Translated};
pub use triangle_box::triangle_aabb_overlap;
