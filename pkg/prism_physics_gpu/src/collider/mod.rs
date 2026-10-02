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
//! * [`GpuObbTrimeshCollider`] — the device twin of the OBB golden.
//! * [`cpu_obb_trimesh_manifold_collide`] — the oriented-box-versus-mesh
//!   manifold golden, one merged multi-point contact per box.
//!
//! The deterministic deepest-contact reduction every mesh collider shares lives
//! in the private `reduce` module, so the tie-break rule stays identical across
//! shapes and across CPU and GPU.

mod capsule_trimesh;
mod capsule_trimesh_gpu;
mod obb_trimesh;
mod obb_trimesh_gpu;
mod obb_trimesh_manifold;
mod reduce;
mod sphere_trimesh;
mod sphere_trimesh_gpu;
mod trimesh;
mod trimesh_capsule_sweep;
mod trimesh_capsule_sweep_gpu;
mod trimesh_closest_point;
mod trimesh_closest_point_gpu;
mod trimesh_raycast;
mod trimesh_raycast_gpu;
mod trimesh_sphere_sweep;
mod trimesh_sphere_sweep_gpu;

pub use capsule_trimesh::cpu_capsule_trimesh_collide;
pub use capsule_trimesh_gpu::GpuCapsuleTrimeshCollider;
pub use obb_trimesh::cpu_obb_trimesh_collide;
pub use obb_trimesh_gpu::GpuObbTrimeshCollider;
pub use obb_trimesh_manifold::cpu_obb_trimesh_manifold_collide;
pub use sphere_trimesh::cpu_sphere_trimesh_collide;
pub use sphere_trimesh_gpu::GpuSphereTrimeshCollider;
pub use trimesh::Trimesh;
pub use trimesh_capsule_sweep::{
    cpu_trimesh_capsule_sweep, cpu_trimesh_capsule_sweep_built, cpu_trimesh_capsule_sweep_bvh,
    CapsuleSweep, CapsuleSweepHit,
};
pub use trimesh_capsule_sweep_gpu::GpuTrimeshCapsuleSweep;
pub use trimesh_closest_point::{
    cpu_trimesh_closest_point, cpu_trimesh_closest_point_built, cpu_trimesh_closest_point_bvh,
    TrimeshClosestHit,
};
pub use trimesh_closest_point_gpu::GpuTrimeshClosestPoint;
pub use trimesh_raycast::{
    cpu_trimesh_raycast, cpu_trimesh_raycast_built, cpu_trimesh_raycast_bvh, MeshRay, TrimeshRayHit,
};
pub use trimesh_raycast_gpu::GpuTrimeshRayCast;
pub use trimesh_sphere_sweep::{
    cpu_trimesh_sphere_sweep, cpu_trimesh_sphere_sweep_built, cpu_trimesh_sphere_sweep_bvh,
    SphereSweep, TrimeshSweepHit,
};
pub use trimesh_sphere_sweep_gpu::GpuTrimeshSphereSweep;
