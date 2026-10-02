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
//! * [`GpuSphereTrimeshCollider`] — the device twin of that golden.
//! * [`cpu_capsule_trimesh_collide`] — the capsule-versus-mesh CPU golden,
//!   one deepest contact per capsule.
//! * [`GpuCapsuleTrimeshCollider`] — the device twin of the capsule golden.
//! * [`cpu_obb_trimesh_collide`] — the oriented-box-versus-mesh CPU golden,
//!   one deepest contact per box.
//!
//! The deterministic deepest-contact reduction every mesh collider shares lives
//! in the private `reduce` module, so the tie-break rule stays identical across
//! shapes and across CPU and GPU.

mod capsule_trimesh;
mod capsule_trimesh_gpu;
mod obb_trimesh;
mod reduce;
mod sphere_trimesh;
mod sphere_trimesh_gpu;
mod trimesh;

pub use capsule_trimesh::cpu_capsule_trimesh_collide;
pub use capsule_trimesh_gpu::GpuCapsuleTrimeshCollider;
pub use obb_trimesh::cpu_obb_trimesh_collide;
pub use sphere_trimesh::cpu_sphere_trimesh_collide;
pub use sphere_trimesh_gpu::GpuSphereTrimeshCollider;
pub use trimesh::Trimesh;
