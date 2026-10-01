//! `GPU` body-proxy cloth collision and per-particle backstops.
//!
//! A garment is simulated against a cheap proxy of the animated body: a small
//! set of analytic [`prism_physics_core::BodyCollider`] primitives (sphere,
//! capsule, half-space) that approximate limbs and torso, plus optional
//! per-particle [`prism_physics_core::Backstop`] planes that stop cloth from
//! sinking into the skinned surface. Both tiers are position-level projections
//! run after the internal constraints, the faithful `GPU` twin of
//! [`prism_physics_core`]'s `resolve_body_collisions_with_friction` and
//! `resolve_backstops`.
//!
//! # Layout
//!
//! - [`collider`] — host-side packing of authored colliders/backstops into the
//!   fixed `#[repr(C)]` records the kernel reads.
//! - [`cpu`] — the [`cpu_cloth_body_collision`] / [`cpu_cloth_backstops`] golden
//!   twins (delegating to [`prism_physics_core`]).
//! - [`gpu`] — the real-device [`GpuClothBodyCollision`] pipeline pair.
//!
//! # Correctness model
//!
//! The pass is *per-particle independent*: one thread owns one particle and (for
//! the body pass) walks every collider in slice order, which is exactly the
//! sequential golden's per-particle write set. There is no colouring and no
//! atomic accumulation, so the only float divergence from the `CPU` is a few
//! `ULP` in `inverseSqrt`/division; parity is verified within a tight relative
//! tolerance rather than bit-for-bit, matching the rest of the solver.
//!
//! # Provenance
//!
//! Analytic body-proxy projections and the one-sided backstop are standard
//! position-based collision techniques; the tangential-friction projection is
//! Macklin et al. (2014). No Unreal Engine source or derived code.

pub mod collider;
pub mod cpu;
pub mod gpu;

pub use collider::{
    pack_backstops, pack_body_colliders, GpuBackstop, GpuBodyCollider, COLLIDER_CAPSULE,
    COLLIDER_HALF_SPACE, COLLIDER_SPHERE,
};
pub use cpu::{cpu_cloth_backstops, cpu_cloth_body_collision};
pub use gpu::GpuClothBodyCollision;
