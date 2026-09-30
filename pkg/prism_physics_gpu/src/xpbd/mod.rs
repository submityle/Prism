//! Colour-ordered substep `XPBD` distance-constraint solver, `GPU` and `CPU`.
//!
//! This module ports the extended-position-based-dynamics stretch solver of
//! [`prism_physics_core`](prism_physics_core::soft) onto the `GPU`. A frame step
//! is split into equal substeps; each substep predicts positions under gravity,
//! resets the per-constraint Lagrange multipliers, projects the distance
//! constraints in *colour order*, then recovers velocities from the net motion.
//!
//! The constraint graph is partitioned by [`Colouring`] so that constraints of
//! the same colour share no particle. Projecting one colour per compute
//! dispatch — with the implicit barrier between passes — makes the massively
//! parallel device sweep reproduce the sequential Gauss-Seidel order of the
//! [`cpu_solve`] golden twin, which is what the real-device parity test checks.
//!
//! # Correctness model
//!
//! [`cpu_solve`] performs the identical `f32` arithmetic, in the identical
//! order, as `shaders/xpbd.wgsl`. It is not byte-for-byte identical to the
//! device result: `GPU` floating-point reassociation (fused multiply-add,
//! differing division and square-root rounding) perturbs the low bits, so the
//! parity test bounds the difference with a tight tolerance rather than exact
//! equality. The `CPU` twin is in turn anchored against an independent
//! sequential reference and a closed-form free-fall check in its own tests.
//!
//! # Provenance
//!
//! Substep `XPBD` with the canonical stretch constraint (Müller et al.); the
//! constraint graph colouring is textbook greedy first-fit. No Unreal Engine
//! source or derived code.

mod coloring;
mod config;
mod constraint;
mod cpu;
mod gpu;
mod state;

pub use coloring::{ColouredEdge, Colouring, MAX_COLOURS};
pub use config::{XpbdConfig, XpbdError};
pub use constraint::DistanceConstraint;
pub use cpu::cpu_solve;
pub use gpu::GpuXpbdSolver;
pub use state::ParticleState;
