//! Bounding volumes: axis-aligned boxes, spheres, and rays.
//!
//! These are small, `Copy` value types built on [`glam::Vec3`] and form the
//! geometric vocabulary used by the broad-phase.

mod aabb;
mod ray;
mod sphere;

pub use aabb::Aabb;
pub use ray::Ray;
pub use sphere::BoundingSphere;
