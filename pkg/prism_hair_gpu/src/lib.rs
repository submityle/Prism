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
pub mod wind;

pub use collision::{query_for, CollisionQuery, GpuColliderProjector};
pub use context::{block_on, GpuContext};
pub use wind::{query_for_wind, GpuWindField, WindQuery};
