//! `GPU` Vertex Block Descent (VBD) soft-body solver, the faithful twin of
//! [`prism_physics_core`]'s coloured VBD stepper
//! ([`VbdSolver::step_colored`](prism_physics_core::VbdSolver::step_colored)).
//!
//! VBD (Chen et al., SIGGRAPH 2024) advances an implicit-Euler step by
//! per-vertex block coordinate descent: each vertex solves a local `3x3`
//! system assembled from its inertial term and incident elastic Hessians. Graph
//! colouring lets a whole colour relax in parallel while the across-colour
//! schedule stays Gauss-Seidel, which is exactly what maps onto a `GPU`:
//! one dispatch per colour. The method is unconditionally stable and holds up
//! under very stiff springs where low-iteration XPBD drifts.
//!
//! # Layout
//!
//! - [`prep`] — the host-side deterministic flattening (padded state arrays, the
//!   incident-spring `CSR`, and the colour-major sweep order) laid out to match
//!   the golden's `Adjacency` and [`VbdColoring`] exactly.
//! - [`cpu`] — the [`cpu_vbd`] golden twin, delegating to `step_colored`.
//! - [`gpu`] — the real-device [`GpuVbd`] three-kernel pipeline.
//!
//! # Correctness model
//!
//! The topology is built on the host (integer-exact), so the only
//! floating-point work on the `GPU` is the per-vertex `3x3` solve. The
//! colour-major dispatch reproduces the golden's Gauss-Seidel-across-colours /
//! Jacobi-within-colour sweep, so the two agree up to a few `ULP` of
//! division/sqrt rounding, verified by the parity suite within a tight
//! tolerance.
//!
//! # Provenance
//!
//! VBD follows Chen et al., "Vertex Block Descent" (SIGGRAPH 2024); greedy graph
//! colouring is a standard, publicly documented technique. No Unreal Engine
//! source or derived code.

pub mod cpu;
pub mod gpu;
pub mod prep;

pub use cpu::cpu_vbd;
pub use gpu::GpuVbd;
pub use prep::{build as build_vbd_prep, GpuSpring, VbdPrep};
