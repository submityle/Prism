//! Narrow-phase ray/primitive intersection tests.
//!
//! These are exact analytic intersections against individual primitives,
//! intended to refine the candidate leaves reported by the broad-phase BVH.
//! They are engine-agnostic implementations of publicly documented algorithms
//! and contain no Unreal Engine source or derived code.

mod ray_cast;

pub use ray_cast::{ray_sphere, ray_triangle, RayTriangleHit};
