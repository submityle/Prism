//! Bounding volumes: axis-aligned boxes, spheres, rays, planes, and frusta.
//!
//! These are small, `Copy` value types built on [`glam::Vec3`] and form the
//! geometric vocabulary used by the broad-phase.

mod aabb;
mod capsule;
mod frustum;
mod obb;
mod plane;
mod ray;
mod sphere;

pub use aabb::Aabb;
pub use capsule::Capsule;
pub use frustum::Frustum;
pub use obb::Obb;
pub use plane::Plane;
pub use ray::Ray;
pub use sphere::BoundingSphere;
