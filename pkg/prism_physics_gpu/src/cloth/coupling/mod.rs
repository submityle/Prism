//! `GPU` two-way rigid coupling between cloth particles and rigid proxies.
//!
//! One-way body collision ([`super::body`]) only pushes particles *out* of a
//! rigid proxy; the body never notices. This module closes the loop, the
//! faithful `GPU` twin of [`prism_physics_core`]'s `resolve_two_way_coupling`:
//! a light prop resting on a hammock dents the cloth and the cloth pushes the
//! prop back up. The pass contributes only the contact half (a mass-weighted
//! split of each push-out between the particle and the body, plus the Newton
//! reaction impulse each body accrues); an authoritative rigid-body integrator
//! consumes the reaction impulse separately.
//!
//! # Layout
//!
//! - [`cpu`] — the [`cpu_cloth_coupling`] golden twin (delegating to
//!   [`prism_physics_core::couple_particle_against_body`] and
//!   `resolve_two_way_coupling`).
//! - [`gpu`] — the real-device [`GpuClothCoupling`] pipeline.
//!
//! The authored collider is packed with the shared [`super::body::collider`]
//! records ([`GpuBodyCollider`](super::body::collider::GpuBodyCollider)) the
//! body-collision kernel already uses, so sphere/capsule/half-space packing
//! lives in exactly one place.
//!
//! # Correctness model
//!
//! Bodies are processed in slice order (the host dispatches one kernel per
//! body, each seeing the particle positions after the previous body's
//! corrections), matching the sequential golden. Within a body the pass is
//! Jacobi and *per-particle independent*: thread `i` corrects `positions[i]`
//! and emits its `body_delta` / `impulse` contributions to per-particle slots,
//! which the host sums in index order (preserving the golden's accumulation
//! order) before translating the body once and accumulating its reaction. The
//! only float divergence from the `CPU` is a few `ULP` in
//! `inverseSqrt`/division, so parity is checked within a tight tolerance.
//!
//! # Provenance
//!
//! The inverse-mass-weighted contact split and the Newton reaction impulse are
//! textbook position-based-dynamics / rigid-body contact mechanics. No Unreal
//! Engine source or derived code.

pub mod cpu;
pub mod gpu;

pub use cpu::cpu_cloth_coupling;
pub use gpu::GpuClothCoupling;
