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
//! - [`bvh_wide_gpu_layout`] — flat, `GPU`-uploadable [`bvh_wide::WideBvh`]:
//!   the compressed wide nodes packed into [`bvh_wide_gpu_layout::WIDE_NODE_WORDS`]
//!   words each (origin, per-axis power-of-two dequant exponents, and every
//!   child's quantized corners/tag/payload) plus a packed wide walk that
//!   reproduces the in-memory [`bvh_wide::WideBvh`] traversal bit-for-bit.
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
//! - [`curve`] — cubic-Bézier round-curve primitive (hair/grass) with a
//!   pbrt-style recursive ray-curve subdivision test: a per-ray orthonormal
//!   frame projects the swept spine so each level culls a ray-frame
//!   [`Aabb`] and, at refinement depth, a swept-segment cap/chord test
//!   reports the [`curve::CurveHit`]; a single-level [`curve::CurveBvh`]
//!   reuses the shared binned-`SAH` build and ordered stack walk.
//! - [`curve_gpu_layout`] — flat, `GPU`-uploadable [`curve::CurveBvh`]
//!   buffer layout ([`curve_gpu_layout::CURVE_WORDS`] stride packing the
//!   four control points plus start/end widths, shared
//!   [`gpu_layout::NODE_WORDS`] nodes) plus a packed curve walk that
//!   reproduces the in-memory curve walk bit-for-bit.
//! - [`cylinder`] — analytic finite *capped* cylinder [`cylinder::Cylinder`]
//!   procedural primitive (`DXR`/Vulkan `AABB` path, tube/capsule area
//!   lights) with a lateral-surface quadratic plus cap-plane tests that
//!   report the nearest of all valid roots, and a single-level
//!   [`cylinder::CylinderBvh`] reusing the shared binned-`SAH` build and
//!   ordered slab walk.
//! - [`cylinder_gpu_layout`] — flat, `GPU`-uploadable [`cylinder::CylinderBvh`]
//!   buffer layout ([`cylinder_gpu_layout::CYLINDER_WORDS`] stride packing
//!   `base`/`top` endpoints plus radius, shared [`gpu_layout::NODE_WORDS`]
//!   nodes) plus a packed cylinder walk that reproduces the in-memory
//!   cylinder walk bit-for-bit.
//! - [`disk`] — analytic oriented disk [`disk::Disk`] procedural primitive
//!   (`DXR`/Vulkan `AABB` path, round area lights / emitter faces) with a
//!   single ray/plane solve plus a radius test that reports the
//!   [`disk::DiskHit`], and a single-level [`disk::DiskBvh`] reusing the
//!   shared binned-`SAH` build and ordered slab walk.
//! - [`disk_gpu_layout`] — flat, `GPU`-uploadable [`disk::DiskBvh`] buffer
//!   layout ([`disk_gpu_layout::DISK_WORDS`] stride packing `center`,
//!   `normal`, and radius, shared [`gpu_layout::NODE_WORDS`] nodes) plus a
//!   packed disk walk that reproduces the in-memory disk walk bit-for-bit.
//! - [`sphere_gpu_layout`] — flat, `GPU`-uploadable [`sphere::SphereBvh`]
//!   buffer layout ([`sphere_gpu_layout::SPHERE_WORDS`] stride, shared
//!   [`gpu_layout::NODE_WORDS`] nodes) plus a packed procedural-primitive
//!   walk that reproduces the in-memory sphere walk bit-for-bit.
//! - [`aabb_primitive_gpu_layout`] — flat, `GPU`-uploadable [`aabb_primitive::AabbBvh`]
//!   buffer layout ([`aabb_primitive_gpu_layout::AABB_PRIMITIVE_WORDS`] stride,
//!   shared [`gpu_layout::NODE_WORDS`] nodes) plus a packed procedural-primitive
//!   walk that reproduces the in-memory box walk bit-for-bit.
//! - [`rectangle`] — analytic oriented rectangle/parallelogram
//!   [`rectangle::Rectangle`] procedural primitive (`DXR`/Vulkan `AABB`
//!   path, rectangular area lights / quad emitters) defined by a center and
//!   two half-edge vectors, with a single ray/plane solve plus a
//!   reciprocal-basis containment test (exact for non-orthogonal edges) that
//!   reports the [`rectangle::RectangleHit`], and a single-level
//!   [`rectangle::RectangleBvh`] reusing the shared binned-`SAH` build and
//!   ordered slab walk.
//! - [`rectangle_gpu_layout`] — flat, `GPU`-uploadable
//!   [`rectangle::RectangleBvh`] buffer layout
//!   ([`rectangle_gpu_layout::RECTANGLE_WORDS`] stride packing `center` plus
//!   the two half-edge vectors, shared [`gpu_layout::NODE_WORDS`] nodes) plus
//!   a packed rectangle walk that reproduces the in-memory rectangle walk
//!   bit-for-bit.
//! - [`cone`] — analytic finite *capped* cone / cone-frustum
//!   [`cone::Cone`] procedural primitive (`DXR`/Vulkan `AABB` intersection
//!   shader analogue): a base + top point with a base and top radius, the
//!   lateral surface solved by the reduced Inigo-Quilez `iCappedCone`
//!   quadratic and the two caps by the exact plane + radius test, reports the
//!   [`cone::ConeHit`], and a single-level [`cone::ConeBvh`] reusing the shared
//!   binned-`SAH` build and slab traversal.
//! - [`cone_gpu_layout`] — flat, `GPU`-uploadable [`cone::ConeBvh`] buffer
//!   layout ([`cone_gpu_layout::CONE_WORDS`] stride packing `base`/`top` plus
//!   the two radii) with node records reusing the shared [`NODE_WORDS`] and a
//!   packed cone walk that reproduces the in-memory cone walk bit-for-bit.
//! - [`paraboloid`] — analytic paraboloid / parabolic dish
//!   [`paraboloid::Paraboloid`] procedural primitive (reflector dishes,
//!   spotlight cups, satellite antennas): an `apex` + rim `top` + rim
//!   `radius`, the wall solved from the implicit quadric `k · ρ² − z = 0`
//!   (`dd`-carrying `t²` coefficient, no unit-direction assumption), reporting
//!   the [`paraboloid::ParaboloidHit`], with a single-level
//!   [`paraboloid::ParaboloidBvh`] reusing the shared binned-`SAH` build and
//!   slab traversal.
//! - [`paraboloid_gpu_layout`] — flat, `GPU`-uploadable
//!   [`paraboloid::ParaboloidBvh`] buffer layout
//!   ([`paraboloid_gpu_layout::PARABOLOID_WORDS`] stride packing `apex`/`top`
//!   plus the rim radius) with node records reusing the shared [`NODE_WORDS`]
//!   and a packed paraboloid walk that reproduces the in-memory walk
//!   bit-for-bit.
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
pub mod bvh_wide_gpu_layout;
pub mod curve;
pub mod curve_gpu_layout;
pub mod cylinder;
pub mod cylinder_gpu_layout;
pub mod disk;
pub mod disk_gpu_layout;
pub mod rectangle;
pub mod rectangle_gpu_layout;
pub mod cone;
pub mod cone_gpu_layout;
pub mod paraboloid;
pub mod paraboloid_gpu_layout;
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
pub use bvh_wide::{WideBvh, WideChild, WideNode, QUANT_STEPS, WIDE_BRANCHING};
pub use bvh_wide_gpu_layout::{
    GpuWideBvh, CHILD_EMPTY, CHILD_INTERIOR, CHILD_LEAF, WIDE_NODE_WORDS, WIDE_TRIANGLE_WORDS,
};
pub use tlas::{Affine3, Instance, Tlas, TlasHit};
pub use motion::{MotionInstance, MotionTlas};
pub use motion_gpu_layout::{GpuMotionTlasBuffers, MOTION_INSTANCE_WORDS};
pub use ray_offset::offset_ray_origin;
pub use scheduler::{AccelerationScheduler, ScheduledUpdate};
pub use aabb_primitive::{AabbBvh, AabbHit, AabbPrimitive};
pub use sphere::{Sphere, SphereBvh, SphereHit};
pub use curve::{Curve, CurveBvh, CurveHit};
pub use curve_gpu_layout::{GpuCurveBvhBuffers, CURVE_WORDS};
pub use cylinder::{Cylinder, CylinderBvh, CylinderHit};
pub use cylinder_gpu_layout::{GpuCylinderBvhBuffers, CYLINDER_WORDS};
pub use disk::{Disk, DiskBvh, DiskHit};
pub use disk_gpu_layout::{GpuDiskBvhBuffers, DISK_WORDS};
pub use rectangle::{Rectangle, RectangleBvh, RectangleHit};
pub use cone::{Cone, ConeBvh, ConeHit};
pub use cone_gpu_layout::{GpuConeBvhBuffers, CONE_WORDS};
pub use paraboloid::{Paraboloid, ParaboloidBvh, ParaboloidHit};
pub use paraboloid_gpu_layout::{GpuParaboloidBvhBuffers, PARABOLOID_WORDS};
pub use rectangle_gpu_layout::{GpuRectangleBvhBuffers, RECTANGLE_WORDS};
pub use aabb_primitive_gpu_layout::{GpuAabbBvhBuffers, AABB_PRIMITIVE_WORDS};
pub use sphere_gpu_layout::{GpuSphereBvhBuffers, SPHERE_WORDS};
pub use traversal::{Hit, Ray};
pub use traversal_stackless::{BvhEscapeTable, ESCAPE_SENTINEL};
pub use traversal_stackless_gpu_layout::GpuStacklessBvh;
pub use gpu_layout::{
    GpuBlasPool, GpuBvhBuffers, GpuTlasBuffers, TlasPackedHit, BLAS_OFFSET_WORDS, INSTANCE_WORDS,
    NODE_WORDS, TRIANGLE_WORDS,
};
