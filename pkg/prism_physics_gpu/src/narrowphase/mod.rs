//! Sphere-sphere narrow phase: candidate pairs to contact manifolds.
//!
//! The broad phase ([`crate::broadphase`]) prunes the `O(n^2)` collision test to
//! a set of candidate pairs whose bounding spheres *might* touch. The narrow
//! phase closes the remaining gap: it takes each candidate pair and produces the
//! precise contact a solver needs — a unit normal, a penetration depth, and a
//! world contact point — or rejects the pair when the spheres do not actually
//! penetrate. This is the missing link between the broad phase and the `XPBD`
//! constraint solver ([`crate::xpbd`]).
//!
//! Both paths run the identical [`sphere::sphere_sphere_contact`] geometry: the
//! [`cpu_narrowphase`] golden twin and the [`GpuNarrowphase`] device kernel emit
//! one contact slot per input pair, so a passing real-device parity test is
//! direct evidence the ported kernel builds the same manifolds as the reference.
//!
//! # Dense output
//!
//! Emitting a slot per pair (rather than compacting on the fly) keeps the
//! contact index aligned with the pair index, which the parity test relies on
//! and which lets a downstream [`crate::scan`] compaction stream the survivors
//! without a second pass over the pairs. It is a deliberate design choice, not a
//! stub: the `None` / invalid slots carry a real "no penetration" decision.
//!
//! Provenance: textbook sphere-sphere manifold construction; no Unreal Engine
//! source or derived code.

mod contact;
mod cpu;
mod gpu;
mod layout;
mod sphere;

pub use contact::Contact;
pub use cpu::cpu_narrowphase;
pub use gpu::GpuNarrowphase;
