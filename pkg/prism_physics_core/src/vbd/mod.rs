//! Vertex Block Descent (VBD) solver (milestone M7).
//!
//! VBD (Chen et al., SIGGRAPH 2024) minimises the implicit-Euler variational
//! energy by block coordinate descent, one vertex at a time. For each vertex a
//! local 3x3 system built from the incident elastic-energy Hessians is solved
//! and the vertex position is updated. The method is unconditionally stable,
//! handles very stiff materials and large deformation without the drift XPBD
//! shows at low iteration counts, and parallelises by graph colouring.
//!
//! The kernel is split into small, single-concept modules:
//!
//! - [`config`] — the [`VbdConfig`] tunables (gravity, substeps, iterations,
//!   damping).
//! - [`element`] — the [`SpringElement`] energy and its force / PSD Hessian,
//!   plus the [`SpringSet`] topology container.
//! - [`coloring`] — the [`VbdColoring`] greedy vertex graph colouring that lets
//!   the parallel (GPU) sweep reproduce a valid Gauss-Seidel pass.
//! - [`system`] — the per-vertex [`VertexSystem`] `3x3` local solve.
//! - [`solver`] — the stateless [`VbdSolver`] stepping loop.
//! - [`body`] — the [`VbdBody`] convenience bundle of particles, springs, and
//!   configuration.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The VBD
//! energy formulation, per-vertex Gauss-Seidel block descent, and the spring
//! and stable-Neo-Hookean tetrahedral energies are implemented from standard,
//! publicly documented literature (Chen et al. 2024; Smith et al. 2018).

pub mod body;
pub mod coloring;
pub mod config;
pub mod element;
pub mod solver;
pub mod system;

pub use body::VbdBody;
pub use coloring::{color_springs, VbdColoring};
pub use config::VbdConfig;
pub use element::{outer, SpringContribution, SpringElement, SpringSet};
pub use solver::VbdSolver;
pub use system::VertexSystem;
