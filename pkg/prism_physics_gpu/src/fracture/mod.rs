//! `GPU`-accelerated fragment classification for destruction.
//!
//! The convex Voronoi *carving* in [`prism_physics_core::fracture`] decides the
//! shape of each fragment on the `CPU`; this module supplies the mass-parallel
//! companion that bins a large point cloud (debris particles, surface samples,
//! or voxel centres) into those fragments on the `GPU`. Given a set of seed
//! sites and a batch of query points, [`GpuVoronoiAssign`] reports the owning
//! Voronoi cell and the distance to the nearest cell wall for every point, at
//! the hundred-thousand-to-million point scale a real-time destruction system
//! needs.
//!
//! # Layout
//!
//! - [`config`] — the [`VoronoiAssignConfig`] tunable and the [`NO_CELL`]
//!   sentinel.
//! - [`cpu`] — the [`cpu_assign_cells`] golden twin and its [`CellAssignment`]
//!   output.
//! - [`gpu`] — the real-device [`GpuVoronoiAssign`] classifier.
//!
//! # Correctness model
//!
//! The kernel is paired with the [`cpu_assign_cells`] twin running the
//! identical arithmetic. The owning cell index is integer-exact whenever the
//! nearest site is unambiguous (the argmin is taken over squared distances with
//! a strict comparison, so both sides keep the lowest index on a tie), and the
//! clearance carries a normalising square root and is therefore checked within
//! a tight tolerance.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It reuses
//! [`prism_physics_core::fracture::Plane`] for the bisector cell walls; nearest-
//! site Voronoi membership is a standard, publicly documented result.

pub mod config;
pub mod cpu;
pub mod gpu;

pub use config::{VoronoiAssignConfig, NO_CELL};
pub use cpu::{cpu_assign_cells, CellAssignment};
pub use gpu::GpuVoronoiAssign;
