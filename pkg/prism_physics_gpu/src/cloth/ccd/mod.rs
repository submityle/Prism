//! `GPU` continuous-collision (tunnelling) sweep for cloth particles against
//! rigid body proxies, the faithful twin of [`prism_physics_core`]'s
//! `resolve_ccd`.
//!
//! The substep [`XPBD`](crate::xpbd) solver only projects particles out of the
//! analytic [`BodyCollider`](prism_physics_core::BodyCollider) proxies at their
//! *end-of-step* position. A thin garment moving fast against a thin collider
//! can start in front of a wall and finish behind it inside a single substep,
//! tunnelling through without the discrete body-collision pass
//! ([`super::body`]) ever seeing an overlap. This module closes that gap by
//! sweeping each free particle's `prev -> curr` segment against every collider,
//! solving for the earliest valid time of impact (TOI), snapping the particle
//! onto the surface plus a skin offset, reflecting its normal velocity by
//! restitution, and damping its tangential slide with position-level Coulomb
//! friction.
//!
//! # Layout
//!
//! - [`cpu`] — the [`cpu_cloth_ccd`] golden twin (delegating to
//!   [`prism_physics_core::resolve_ccd`]).
//! - [`gpu`] — the real-device [`GpuClothCcd`] pipeline.
//!
//! The authored colliders are packed with the shared
//! [`super::body::collider`] records
//! ([`GpuBodyCollider`](super::body::collider::GpuBodyCollider)) that the
//! body-collision kernel already uses, so sphere/capsule/half-space packing
//! lives in exactly one place.
//!
//! # Correctness model
//!
//! The pass is *per-particle independent*: thread `i` reads a read-only
//! previous position, walks every collider in slice order (ties broken toward
//! the first, matching the golden's strict `t < best` update), and writes only
//! its own `positions[i]` / `velocities[i]`. This is exactly the sequential
//! golden's per-particle write set, so a single dispatch reproduces the whole
//! sweep. The only float divergence from the `CPU` is a few `ULP` in
//! `sqrt`/division inside the closed-form TOI and projection, so parity is
//! verified within a tight tolerance rather than bit-for-bit.
//!
//! # Provenance
//!
//! The closed-form swept-primitive TOI solvers are standard analytic
//! continuous-collision geometry, and the tangential-friction projection reuses
//! the shared primitive published by Macklin et al. (2014), "Unified Particle
//! Physics for Real-Time Applications". No Unreal Engine source or derived code.

pub mod cpu;
pub mod gpu;

pub use cpu::cpu_cloth_ccd;
pub use gpu::GpuClothCcd;
