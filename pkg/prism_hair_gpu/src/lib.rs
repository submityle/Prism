//! Optional `wgpu` compute twin of Prism's strand-hair body-collider
//! projection.
//!
//! Strand grooms keep hair off the body by projecting each free particle out of
//! a small set of analytic collider proxies (spheres and capsules fitted to the
//! head, neck and shoulders) after every constraint sweep. The `CPU` golden
//! standard for that projection lives in
//! [`prism_render_architecture::hair::collision`]; this crate is the `GPU`
//! twin, validated against that reference so a passing real-device parity test
//! is direct evidence the ported kernel computes the same projected positions
//! as the reference, not merely that its shader compiles.
//!
//! # Scope
//!
//! [`GpuColliderProjector`] evaluates
//! [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)
//! for a batch of point/collider pairs, covering both the sphere and capsule
//! branches the analytic body-collision tier uses.
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

pub use collision::{query_for, CollisionQuery, GpuColliderProjector};
pub use context::{block_on, GpuContext};
