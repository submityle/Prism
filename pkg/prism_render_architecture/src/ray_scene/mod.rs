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
//! - [`bvh_wide`] — compressed *wide* (`BVH8`) acceleration structure: the
//!   binary [`bvh::Bvh`] collapsed (Ylitie et al.) into nodes with up to
//!   [`bvh_wide::WIDE_BRANCHING`] children whose bounds are *quantized* to a
//!   per-node byte lattice (floored low / ceiled high corners), so a
//!   [`bvh_wide::WideBvh`] fetches many boxes per cache line yet stays a
//!   conservative superset and reproduces [`bvh::Bvh::closest_hit`]
//!   bit-for-bit on rays with a unique nearest hit.
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
//! - [`aabb_primitive`] — analytic axis-aligned box [`aabb_primitive::AabbPrimitive`]
//!   procedural primitive (`DXR`/Vulkan `AABB` path) with a slab intersection
//!   that reports the face normal and front/back flag, plus a single-level
//!   [`aabb_primitive::AabbBvh`] reusing the shared binned-`SAH` build and
//!   ordered slab walk.
//! - [`sphere`] — analytic [`sphere::Sphere`] primitive (`DXR`/Vulkan
//!   procedural-primitive `AABB` path) with a numerically stable ray test
//!   and a single-level [`sphere::SphereBvh`] reusing the shared
//!   binned-`SAH` build and ordered slab walk.
//! - [`sphere_gpu_layout`] — flat, `GPU`-uploadable [`sphere::SphereBvh`]
//!   buffer layout ([`sphere_gpu_layout::SPHERE_WORDS`] stride, shared
//!   [`gpu_layout::NODE_WORDS`] nodes) plus a packed procedural-primitive
//!   walk that reproduces the in-memory sphere walk bit-for-bit.
//! - [`aabb_primitive_gpu_layout`] — flat, `GPU`-uploadable [`aabb_primitive::AabbBvh`]
//!   buffer layout ([`aabb_primitive_gpu_layout::AABB_PRIMITIVE_WORDS`] stride,
//!   shared [`gpu_layout::NODE_WORDS`] nodes) plus a packed procedural-primitive
//!   walk that reproduces the in-memory box walk bit-for-bit.
//! - [`traversal`] — [`Ray`]/`BVH` slab + Möller–Trumbore intersection with
//!   closest-hit and any-hit walks (the `GPU` traversal kernel's golden ref).
//! - [`traversal_stackless`] — stackless (threaded / escape-index) `BVH` walk:
//!   a [`traversal_stackless::BvhEscapeTable`] precomputes one skip index per
//!   node so a ray descends with a single cursor and no per-thread stack
//!   (`GPU`-friendly), reproducing [`Bvh::closest_hit`]/[`Bvh::any_hit`]
//!   (and watertight variants) bit-for-bit.
//! - [`traversal_stackless_gpu_layout`] — flat, `GPU`-uploadable stackless
//!   `BVH`: the packed [`gpu_layout::GpuBvhBuffers`] geometry plus a parallel
//!   escape-index `array<u32>` ([`traversal_stackless_gpu_layout::GpuStacklessBvh`]),
//!   with a packed single-cursor walk that reproduces the in-memory stackless
//!   walk (and the stack walk) bit-for-bit.
//! - [`ray_offset`] — Wächter-Binder watertight secondary-ray origin offset
//!   (adaptive integer-`ULP` push) that keeps shadow/reflection/`GI` rays from
//!   self-intersecting the surface they leave, at any scene scale.
//! - [`gpu_layout`] — flat, `GPU`-uploadable `BVH`/`TLAS` buffer layout (the
//!   authoritative `WESL` kernel `ABI`) plus a packed traversal that reproduces
//!   the in-memory walk bit-for-bit as the `CPU`↔`GPU` parity reference.

pub mod acceleration;
pub mod backend;
pub mod bvh;
pub mod bvh_wide;
pub mod footprint;
pub mod gpu_layout;
pub mod motion;
pub mod motion_gpu_layout;
pub mod ray_offset;
pub mod scheduler;
pub mod aabb_primitive;
pub mod sphere;
pub mod aabb_primitive_gpu_layout;
pub mod sphere_gpu_layout;
pub mod tlas;
pub mod traversal;
pub mod traversal_stackless;
pub mod traversal_stackless_gpu_layout;

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
pub use aabb_primitive::{AabbBvh, AabbHit, AabbPrimitive};
pub use sphere::{Sphere, SphereBvh, SphereHit};
pub use aabb_primitive_gpu_layout::{GpuAabbBvhBuffers, AABB_PRIMITIVE_WORDS};
pub use sphere_gpu_layout::{GpuSphereBvhBuffers, SPHERE_WORDS};
pub use traversal::{Hit, Ray};
pub use traversal_stackless::{BvhEscapeTable, ESCAPE_SENTINEL};
pub use traversal_stackless_gpu_layout::GpuStacklessBvh;
pub use gpu_layout::{
    GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers, TlasPackedHit, BLAS_OFFSET_WORDS, INSTANCE_WORDS,
    NODE_WORDS, TRIANGLE_WORDS,
};
