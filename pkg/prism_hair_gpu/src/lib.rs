//! Optional `wgpu` compute twins of Prism's strand-hair guide-solver kernels.
//!
//! Each kernel here is the on-device counterpart of a `CPU` golden standard in
//! [`prism_render_architecture::hair`], validated against that reference so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same values as the reference, not merely that its shader
//! compiles. The twins share one dispatch shape — one thread per query, a
//! uniform count plus a read-only query buffer plus a read-write value buffer —
//! so new kernels slot in beside the existing ones.
//!
//! # Scope
//!
//! * [`GpuColliderProjector`] evaluates
//!   [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)
//!   for a batch of point/collider pairs, covering both the sphere and capsule
//!   branches the analytic body-collision tier uses (see [`collision`]).
//! * [`GpuWindField`] evaluates
//!   [`wind_acceleration`](prism_render_architecture::hair::wind::wind_acceleration)
//!   for a batch of sample point/time/field triples, reproducing the steady,
//!   gust and turbulent terms of the wind coupling (see [`wind`]).
//! * [`GpuStrandFrames`] evaluates
//!   [`build_strand_frames`](prism_render_architecture::hair::frames::build_strand_frames),
//!   walking the double-reflection rotation-minimizing frame transport one
//!   thread per strand so every control point gets a coherent orthonormal
//!   tangent/normal/bitangent basis (see [`frames`]).
//! * [`GpuRibbon`] evaluates
//!   [`build_ribbon`](prism_render_architecture::hair::ribbon::build_ribbon),
//!   meshing each strand into its view-independent `Cards` LOD ribbon proxy one
//!   thread per strand — two edge vertices per control point offset `±radius`
//!   along the rotation-minimizing bitangent, with an arc-length `v` coordinate
//!   (see [`ribbon`]).
//!
//! * [`GpuSdfCollider`] evaluates
//!   [`push_out_of_field`](prism_render_architecture::hair::sdf_collision::push_out_of_field),
//!   pushing a batch of query points out of a union SDF (sphere, capsule,
//!   half-space, box) along the field gradient, one thread per point, for the
//!   tighter body-collision tier layered on the analytic proxies (see
//!   [`sdf_collision`]).
//!
//! * [`GpuSelfCollisionJacobi`] evaluates
//!   [`accumulate_jacobi_corrections`](prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections),
//!   accumulating each strand particle's parallel-safe (Jacobi) self-collision
//!   correction from a read-only snapshot, one thread per particle, walking a
//!   host-built per-particle neighbor slice so the reduction order matches the
//!   reference (see [`self_collision_jacobi`]).
//!
//! * [`GpuStrandMetrics`] evaluates
//!   [`strand_arc_length`](prism_render_architecture::hair::decimation::strand_arc_length)
//!   and
//!   [`strand_curvature`](prism_render_architecture::hair::decimation::strand_curvature),
//!   folding each render strand's control polyline into its arc length and
//!   transcendental-free turning one thread per strand over the strand-major
//!   fixed-stride point pool, the two cheap scalars the density/decimation LOD
//!   ranking consumes (see [`strand_metrics`]).
//! * [`GpuGuideSolver`] evaluates
//!   [`simulate_guides`](prism_render_architecture::hair::dynamics::simulate_guides),
//!   advancing the sparse guide strands one thread per strand through the
//!   full `XPBD` substep/iteration schedule (compliant edge-length, bending,
//!   goal-pose and long-range-attachment constraints plus analytic body
//!   push-out), the core strand-dynamics stage the render strands are
//!   interpolated from (see [`guide_solver`]).
//!
//! # Portability
//!
//! The projection uses only `sqrt`, `min`, `max`, `clamp`, `dot` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! The projection contains no transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form geometry. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per component. See [`collision`]
//! for the full rationale.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic sphere/capsule collider push-out plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
#![forbid(unsafe_code)]

pub mod collision;
pub mod context;
pub mod frames;
pub mod guide_solver;
pub mod ribbon;
pub mod sdf_collision;
pub mod self_collision_jacobi;
pub mod strand_metrics;
pub mod wind;

pub use collision::{query_for, CollisionQuery, GpuColliderProjector};
pub use context::{block_on, GpuContext};
pub use frames::{GpuStrandFrame, GpuStrandFrames};
pub use guide_solver::GpuGuideSolver;
pub use ribbon::{GpuRibbon, GpuRibbonMesh, RibbonStrandInput};
pub use sdf_collision::GpuSdfCollider;
pub use self_collision_jacobi::GpuSelfCollisionJacobi;
pub use strand_metrics::{GpuStrandMetric, GpuStrandMetrics};
pub use wind::{query_for_wind, GpuWindField, WindQuery};
