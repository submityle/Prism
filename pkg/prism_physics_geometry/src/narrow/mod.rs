//! Narrow-phase ray/primitive intersection tests.
//!
//! These are exact analytic intersections against individual primitives,
//! intended to refine the candidate leaves reported by the broad-phase BVH.
//! They are engine-agnostic implementations of publicly documented algorithms
//! and contain no Unreal Engine source or derived code.

mod closest_point;
mod epa;
mod gjk;
mod ray_cast;
mod support;

pub use closest_point::{
    closest_point_on_aabb, closest_point_on_segment, closest_point_on_triangle,
    closest_points_segment_segment, SegmentClosest,
};
pub use epa::{gjk_contact, Contact};
pub use gjk::gjk_intersect;
pub use ray_cast::{ray_obb, ray_sphere, ray_triangle, RayTriangleHit};
pub use support::SupportMap;
