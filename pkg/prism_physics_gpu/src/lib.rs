//! Optional `wgpu` compute backend for Prism's next-generation physics engine.
//!
//! This crate is milestone `M5`: it moves the mass-parallel stages of the
//! simulation (broad-phase neighbour finding first, position projection and
//! particle solvers to follow) onto the `GPU` through `wgpu` compute pipelines,
//! targeting the hundred-thousand-to-million particle regime that the pure
//! `CPU` reference in [`prism_physics_core`] cannot reach in real time.
//!
//! # Correctness model
//!
//! Every kernel here is paired with a byte-for-byte `CPU` golden twin. The
//! twin lives next to the kernel (for example [`broadphase::cpu_broadphase`])
//! and runs the identical integer/float arithmetic, so a passing real-device
//! parity test is direct evidence that the ported kernel computes the same
//! result as the reference, not merely that its `WGSL` compiles. Each `CPU`
//! twin is in turn anchored against an independent brute-force reference in its
//! own unit tests, closing the loop from first principles.
//!
//! # Provenance
//!
//! All algorithms are standard, openly published techniques (Teschner et al.
//! 2003 spatial hashing for the broad phase; extended-position-based-dynamics
//! projection to follow). This crate contains no Unreal Engine source or
//! derived code.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
#![forbid(unsafe_code)]

pub mod broadphase;
pub mod buffer;
pub mod context;

pub use broadphase::{cpu_broadphase, BroadphaseConfig, BroadphaseError, CandidatePair, Particle};
pub use context::GpuContext;
