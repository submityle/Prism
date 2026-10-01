//! Bounding volumes: axis-aligned boxes, spheres, rays, planes, and frusta.
//!
//! These are small, `Copy` value types built on [`glam::Vec3`] and form the
//! geometric vocabulary used by the broad-phase.

mod aabb;
mod frustum;
mod plane;
mod ray;
mod sphere;

pub use aabb::Aabb;
pub use frustum::Frustum;
pub use plane::Plane;
pub use ray::Ray;
pub use sphere::BoundingSphere;
