//! One-sided non-penetration contact solving, `GPU` and `CPU`.
//!
//! This module closes the collision-response loop that the narrow phase opens:
//! given the [`Contact`](crate::narrowphase::Contact) manifold for a set of
//! overlapping bounding spheres, it pushes the penetrating pairs apart with a
//! compliant, one-sided `XPBD` constraint until they no longer interpenetrate.
//!
//! # Pipeline
//!
//! 1. [`contact_constraints`] turns the dense narrow-phase contact list into a
//!    compact [`ContactConstraint`] list, deriving each rest separation from the
//!    two particle radii.
//! 2. [`cpu_resolve_contacts`] (the golden twin) or [`GpuContactSolver::resolve`]
//!    (the device kernel) advances the particle state one frame step under those
//!    constraints, using the same colour-ordered substep `XPBD` schedule as the
//!    distance solver but with the [one-sided projection](cpu) that only ever
//!    separates a pair and never attracts it.
//!
//! # Relationship to the distance solver
//!
//! The contact solver is intentionally *separate* from the `XPBD` distance
//! solver in [`crate::xpbd`]: it reuses the proven [`Colouring`](crate::xpbd::Colouring)
//! partitioner (through the shared [`ColouredEdge`](crate::xpbd::ColouredEdge)
//! trait) and mirrors its substep structure, but keeps its own kernel so the
//! already-verified distance solver stays untouched. Co-solving both constraint
//! families in a single interleaved sweep is deliberately left to a later stage.
//!
//! # Correctness model
//!
//! As with every kernel in this crate, [`cpu_resolve_contacts`] performs the
//! identical `f32` arithmetic, in the identical order, as
//! `shaders/contacts_resolve.wgsl`, and the real-device parity test bounds the
//! remaining floating-point-reassociation difference with a tight tolerance.
//!
//! # Provenance
//!
//! Substep `XPBD` with the canonical one-sided contact constraint (Müller et
//! al.). No Unreal Engine source or derived code.

mod build;
mod constraint;
mod cpu;
mod gpu;
mod layout;

pub use build::contact_constraints;
pub use constraint::ContactConstraint;
pub use cpu::cpu_resolve_contacts;
pub use gpu::GpuContactSolver;
