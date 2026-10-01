//! `GPU` continuous self-collision (self-CCD) sweep for cloth particles, the
//! faithful twin of [`prism_physics_core`]'s
//! [`resolve_self_ccd_jacobi`](prism_physics_core::resolve_self_ccd_jacobi).
//!
//! The discrete self-collision tier ([`super`]) only separates cloth samples
//! that overlap at their *end-of-step* position. A thin garment folding onto
//! itself fast can have two sheets start apart and finish crossed within a
//! single substep, tunnelling through each other without the discrete pass ever
//! seeing an overlap. This module closes that gap by sweeping every candidate
//! particle pair's `prev -> curr` motion, solving for the earliest valid
//! swept-pair time of impact (TOI), separating the pair to the fabric
//! thickness, and recovering a normal restitution impulse — all in one
//! parallel-safe (Jacobi) iteration.
//!
//! # Layout
//!
//! - [`prep`] — the host-side deterministic broad phase (uniform spatial hash
//!   over swept boxes, candidate-pair enumeration, and the per-particle `CSR`
//!   incidence list) laid out in the golden's exact reduction order.
//! - [`cpu`] — the [`cpu_cloth_self_ccd`] golden twin, delegating to
//!   [`prism_physics_core::resolve_self_ccd_jacobi`].
//! - [`gpu`] — the real-device [`GpuClothSelfCcd`] pipeline pair.
//!
//! # Correctness model
//!
//! The candidate set and the per-particle incidence list are built on the host
//! (integer-exact), so the only floating-point work on the `GPU` is the
//! swept-pair resolution itself. The two own-slot passes each read only a
//! frozen snapshot, so the result is independent of invocation order (a Jacobi
//! iteration, never Gauss-Seidel) and bit-matches the golden's own
//! parallel-safe pass up to a few `ULP` of `sqrt`/division rounding, verified by
//! the parity suite within a tight tolerance.
//!
//! # Provenance
//!
//! The closed-form swept-pair TOI resolution is standard analytic
//! continuous-collision geometry; the Jacobi own-slot accumulate/apply split is
//! standard parallel position-based dynamics; uniform spatial hashing is the
//! classical Teschner et al. 2003 scheme. No Unreal Engine source or derived
//! code.

pub mod cpu;
pub mod gpu;
pub mod prep;

pub use cpu::cpu_cloth_self_ccd;
pub use gpu::GpuClothSelfCcd;
pub use prep::{build as build_cloth_self_ccd_prep, ClothSelfCcdPrep};
