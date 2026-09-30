//! Hardware and software ray-scene contracts.
//!
//! This module owns the `CPU`-verifiable policy layer for ray tracing: how
//! acceleration structures are updated, which trace backend serves a workload,
//! and how a ray cone's footprint selects a texture mip level, plus the
//! software `BVH` build and traversal that serve as the `CPU` golden reference
//! for the `GPU` kernels. The physical `GPU` `BVH` build/compaction and the
//! traversal kernel are pending the GPU backend; everything here is
//! deterministic arithmetic the backend mirrors bit-for-bit.
//!
//! Submodules:
//! - [`acceleration`] — `BLAS`/`TLAS` update-strategy decisions and rebuild
//!   budgeting (`Reuse`/`Refit`/`Rebuild`/`BuildAndCompact`).
//! - [`scheduler`] — per-structure [`AccelerationScheduler`] tying the update
//!   policy, refit-quality feedback, and per-frame rebuild budget into one
//!   cross-frame lifecycle (mandatory correctness work vs. deferrable rebuilds).
//! - [`backend`] — capability-driven [`TraceBackend`] fallback selection.
//! - [`footprint`] — ray-cone [`RayFootprint`] and texture-`LOD` (mip) math.
//! - [`bvh`] — software `BVH`: primitive bounds, binned-`SAH` build, and the
//!   flattened [`LinearBvhNode`] layout the `GPU` builder mirrors.
//! - [`tlas`] — two-level acceleration: a top-level `BVH` over affine
//!   [`tlas::Instance`]s of a shared `BLAS` pool, with object-space ray
//!   transform and cross-instance nearest-hit pruning.
//! - [`motion`] — two-level *matrix motion blur* [`motion::MotionTlas`]: DXR
//!   matrix-motion instances with two key poses blended per ray at a
//!   normalized shutter `time`, over a `BVH` built once on conservative swept
//!   bounds; shares the top-level walk, inclusion masks, and [`tlas::TlasHit`].
//! - [`motion_gpu_layout`] — flat, `GPU`-uploadable [`MotionTlas`] buffer
//!   layout ([`motion_gpu_layout::MOTION_INSTANCE_WORDS`] stride packing both
//!   key poses) plus a packed matrix-motion walk that blends and inverts the
//!   pose per ray and reproduces the in-memory motion walk bit-for-bit.
//! - [`sphere`] — analytic [`sphere::Sphere`] primitive (`DXR`/Vulkan
//!   procedural-primitive `AABB` path) with a numerically stable ray test
//!   and a single-level [`sphere::SphereBvh`] reusing the shared
//!   binned-`SAH` build and ordered slab walk.
//! - [`traversal`] — [`Ray`]/`BVH` slab + Möller–Trumbore intersection with
//!   closest-hit and any-hit walks (the `GPU` traversal kernel's golden ref).
//! - [`ray_offset`] — Wächter-Binder watertight secondary-ray origin offset
//!   (adaptive integer-`ULP` push) that keeps shadow/reflection/`GI` rays from
//!   self-intersecting the surface they leave, at any scene scale.
//! - [`gpu_layout`] — flat, `GPU`-uploadable `BVH`/`TLAS` buffer layout (the
//!   authoritative `WESL` kernel `ABI`) plus a packed traversal that reproduces
//!   the in-memory walk bit-for-bit as the `CPU`↔`GPU` parity reference.

pub mod acceleration;
pub mod backend;
pub mod bvh;
pub mod footprint;
pub mod gpu_layout;
pub mod motion;
pub mod motion_gpu_layout;
pub mod ray_offset;
pub mod scheduler;
pub mod sphere;
pub mod tlas;
pub mod traversal;

pub use acceleration::{
    update_scratch_bytes, AccelerationUpdate, AccelerationUpdatePolicy, GeometryChange,
    RebuildLedger,
};
pub use backend::{
    select_backend, BackendCapabilities, BackendRejection, BackendSelection, TraceBackend,
    TraceRequirements,
};
pub use footprint::{log2_linear, RayFootprint};
pub use bvh::{Aabb, Axis, Bvh, BvhBuildConfig, LinearBvhNode, Triangle};
pub use tlas::{Affine3, Instance, Tlas, TlasHit};
pub use motion::{MotionInstance, MotionTlas};
pub use motion_gpu_layout::{GpuMotionTlasBuffers, MOTION_INSTANCE_WORDS};
pub use ray_offset::offset_ray_origin;
pub use scheduler::{AccelerationScheduler, ScheduledUpdate};
pub use sphere::{Sphere, SphereBvh, SphereHit};
pub use traversal::{Hit, Ray};
pub use gpu_layout::{
    GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers, TlasPackedHit, BLAS_OFFSET_WORDS, INSTANCE_WORDS,
    NODE_WORDS, TRIANGLE_WORDS,
};
