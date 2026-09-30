//! `GPU` bounded uniform grid for particle neighbourhood queries.
//!
//! A uniform grid is the workhorse acceleration structure for particle systems:
//! sorting particles by cell makes every neighbourhood a contiguous, coherent
//! slice, which is exactly what fluid, `MPM`, and contact solvers gather over.
//! This module builds that structure over a *bounded, dense* lattice, where a
//! point maps to a single collision-free linear cell index (out-of-bounds
//! points clamp to the boundary), in contrast to the collision-prone spatial
//! hash used by the broad phase.
//!
//! The build has three device stages: a hash pass computes each particle's cell
//! index, the sibling [`GpuRadixSort`](crate::radix::GpuRadixSort) stably orders
//! the particle indices by cell, and a "find cell start" pass turns the sorted
//! keys into per-cell `[start, end)` ranges. All three run in one submission by
//! composing on [`GpuRadixSort::record_sort`], so intermediate data never leaves
//! the device. Each stage is mirrored by the [`cpu_grid_sort`] golden twin;
//! because the build is a pure integer permutation plus a range scan, a passing
//! real-device parity test is bit-for-bit evidence of a faithful port.
//!
//! # Provenance
//!
//! The bounded uniform grid built by hashing to a linear cell index, sorting by
//! cell, and finding cell starts is the classical technique of Green, "Particle
//! Simulation using CUDA" (NVIDIA 2008). This module contains no Unreal Engine
//! source or derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod layout;

pub use config::{GridConfig, GridError};
pub use cpu::{cpu_grid_sort, GridBuild, EMPTY};
pub use gpu::GpuUniformGrid;
