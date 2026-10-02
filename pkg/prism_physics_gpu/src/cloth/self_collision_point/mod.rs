//! `GPU` point (vertex-vertex) self-collision for cloth particles, the faithful
//! twin of [`prism_physics_core`]'s
//! [`resolve_self_collision_jacobi`](prism_physics_core::resolve_self_collision_jacobi)
//! and its friction variant
//! [`resolve_self_collision_with_friction_jacobi`](prism_physics_core::resolve_self_collision_with_friction_jacobi).
//!
//! The point self-collision tier separates every pair of cloth samples that end
//! a step closer than the fabric thickness. Unlike the virtual-particle tier
//! ([`super`]) that also seeds barycentric face samples, this tier resolves the
//! real vertex-vertex pairs directly, optionally layering position-level
//! Coulomb friction on the tangential slide. The Gauss-Seidel core cannot map
//! to a compute kernel because every invocation must read the *same* frozen
//! snapshot, so this module runs the Jacobi reformulation the hardware
//! actually executes.
//!
//! # Layout
//!
//! - [`prep`] — the host-side deterministic broad phase (uniform spatial hash,
//!   27-cell candidate-pair enumeration, and the per-particle `CSR` incidence
//!   list) laid out in the golden's exact reduction order.
//! - [`cpu`] — the [`cpu_cloth_self_collision_point`] golden twin, delegating to
//!   [`prism_physics_core`].
//! - [`gpu`] — the real-device [`GpuClothSelfCollisionPoint`] pipeline pair.
//!
//! # Correctness model
//!
//! The candidate set and the per-particle incidence list are built on the host
//! (integer-exact), so the only floating-point work on the `GPU` is the
//! separating-push arithmetic. The two own-slot passes each read only a frozen
//! snapshot, so the result is independent of invocation order (a Jacobi
//! iteration, never Gauss-Seidel) and matches the golden's own parallel-safe
//! pass up to a few `ULP` of `sqrt`/division rounding, verified by the parity
//! suite within a tight tolerance.
//!
//! # Provenance
//!
//! The inverse-mass-weighted separation is standard position-based dynamics;
//! the tangential-friction projection is the one published by Macklin et al.
//! (2014), "Unified Particle Physics for Real-Time Applications"; the Jacobi
//! own-slot accumulate/apply split is standard parallel position-based
//! dynamics; uniform spatial hashing is the classical Teschner et al. 2003
//! scheme. No Unreal Engine source or derived code.

pub mod cpu;
pub mod gpu;
pub mod prep;

pub use cpu::cpu_cloth_self_collision_point;
pub use gpu::GpuClothSelfCollisionPoint;
pub use prep::{build as build_cloth_self_collision_point_prep, ClothSelfCollisionPointPrep};
