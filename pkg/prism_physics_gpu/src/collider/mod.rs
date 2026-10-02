//! Collider aggregation layer: shape colliders built from the shared broad- and
//! narrow-phase primitives.
//!
//! A `collider` is one level up from a single geometric test. Where
//! [`crate::narrowphase`] answers "does this sphere touch this one triangle",
//! a collider answers "where does this sphere touch this whole mesh", stitching
//! the `LBVH` broad phase, the per-triangle narrow phase, and a manifold
//! reduction into one deterministic query. The pieces it composes are each
//! already GPU-twinned and parity-tested, so the collider inherits their
//! bit-for-bit device agreement.
//!
//! * [`Trimesh`] — a static indexed triangle-mesh collider and its per-triangle
//!   bounding boxes.
//! * [`cpu_sphere_trimesh_collide`] — the sphere-versus-mesh CPU golden, one
//!   deepest contact per sphere.

mod sphere_trimesh;
mod trimesh;

pub use sphere_trimesh::cpu_sphere_trimesh_collide;
pub use trimesh::Trimesh;
