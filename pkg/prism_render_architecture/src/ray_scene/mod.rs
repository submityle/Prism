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
//! - [`backend`] — capability-driven [`TraceBackend`] fallback selection.
//! - [`footprint`] — ray-cone [`RayFootprint`] and texture-`LOD` (mip) math.
//! - [`bvh`] — software `BVH`: primitive bounds, binned-`SAH` build, and the
//!   flattened [`LinearBvhNode`] layout the `GPU` builder mirrors.
//! - [`tlas`] — two-level acceleration: a top-level `BVH` over affine
//!   [`tlas::Instance`]s of a shared `BLAS` pool, with object-space ray
//!   transform and cross-instance nearest-hit pruning.
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
pub mod ray_offset;
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
pub use ray_offset::offset_ray_origin;
pub use traversal::{Hit, Ray};
pub use gpu_layout::{
    GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers, TlasPackedHit, BLAS_OFFSET_WORDS, INSTANCE_WORDS,
    NODE_WORDS, TRIANGLE_WORDS,
};
