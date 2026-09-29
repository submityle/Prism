//! Optional `wgpu` compute backend for Prism's next-generation physics engine.
//!
//! This crate is milestone `M5`: it moves the mass-parallel stages of the
//! simulation (broad-phase neighbour finding and the position-based constraint
//! solver) onto the `GPU` through `wgpu` compute pipelines, targeting the
//! hundred-thousand-to-million particle regime that the pure `CPU` reference in
//! [`prism_physics_core`] cannot reach in real time.
//!
//! # Correctness model
//!
//! Every kernel here is paired with a `CPU` golden twin. The twin lives next to
//! the kernel (for example [`broadphase::cpu_broadphase`] and
//! [`xpbd::cpu_solve`]) and runs the identical arithmetic, so a passing
//! real-device parity test is direct evidence that the ported kernel computes
//! the same result as the reference, not merely that its `WGSL` compiles. Each
//! `CPU` twin is in turn anchored against an independent brute-force reference
//! in its own unit tests, closing the loop from first principles.
//!
//! The strength of the parity claim depends on the kernel. The broad phase is
//! integer-exact: its candidate-pair set matches the twin bit-for-bit. The
//! `XPBD` solver is floating-point: `GPU` reassociation (fused multiply-add,
//! differing division and square-root rounding) perturbs the low bits, so its
//! parity is verified within a tight tolerance rather than byte-for-byte.
//!
//! # Provenance
//!
//! All algorithms are standard, openly published techniques (Teschner et al.
//! 2003 spatial hashing for the broad phase; extended position-based dynamics,
//! Müller et al., for the constraint solver, with textbook greedy first-fit
//! graph colouring). This crate contains no Unreal Engine source or derived
//! code.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
#![forbid(unsafe_code)]

pub mod broadphase;
pub mod buffer;
pub mod context;
pub mod xpbd;

pub use broadphase::{cpu_broadphase, BroadphaseConfig, BroadphaseError, CandidatePair, Particle};
pub use context::GpuContext;
pub use xpbd::{
    cpu_solve, Colouring, DistanceConstraint, GpuXpbdSolver, ParticleState, XpbdConfig, XpbdError,
};
