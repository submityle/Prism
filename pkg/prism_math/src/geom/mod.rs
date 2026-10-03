//! Geometric primitives: rays, boxes, spheres, planes, segments, and frustums.
//!
//! These types are the inputs to the [`crate::intersect`] routines and provide
//! their own constructors, containment queries, and merge/expand helpers where
//! they make sense. All operations are `f32` and `no_std`-friendly.

pub mod aabb;
pub mod frustum;
pub mod plane;
pub mod ray;
pub mod segment;
pub mod sphere;

pub use aabb::Aabb3;
pub use frustum::Frustum;
pub use plane::Plane;
pub use ray::Ray3;
pub use segment::Segment3;
pub use sphere::BoundingSphere;
