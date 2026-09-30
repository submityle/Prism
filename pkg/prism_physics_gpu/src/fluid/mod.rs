//! `GPU` `FLIP`/`APIC` free-surface fluid solver and its `CPU` golden twin.
//!
//! This module ports the particle-grid transfer of the `CPU` reference in
//! [`prism_physics_core`](prism_physics_core::fluid) onto the `GPU`. The fluid
//! state is a set of particles ([`FluidParticles`]) carrying velocity onto a
//! staggered marker-and-cell grid ([`GridDims`]): particle-to-grid scatter,
//! per-face normalisation, and grid-to-particle gather with the `PIC`/`FLIP`
//! blend.
//!
//! # Layout
//!
//! - [`config`] — solver tunables ([`FluidConfig`]) and errors ([`FluidError`]).
//! - [`grid`] — staggered-grid dimensions, face indexing, and cell tags.
//! - [`particle`] — the particle Structure-of-Arrays ([`FluidParticles`]).
//! - [`quantize`] — the fixed-point scale constants shared by the scatter.
//! - [`cpu`] — the golden twin ([`GoldenGrid`], [`particle_to_grid`],
//!   [`grid_to_particle`]) running the identical arithmetic as the kernels.
//! - [`gpu`] — the real-device [`GpuFluidSolver`].
//!
//! # Correctness model
//!
//! The integer momentum/weight accumulators the scatter builds are exact and
//! order-independent, so the `CPU` twin and the device agree on them
//! bit-for-bit; only the final per-face division and the trilinear gather are
//! floating point. A pure-`PIC` transfer (`blend = 0`) over the zero saved
//! field of this transfer-only stage is therefore an exact round-trip, checked
//! against the device within a tight tolerance by `tests/fluid_parity.rs`.
//!
//! # Provenance
//!
//! Trilinear `P2G`/`G2P` with the `PIC`/`FLIP` blend follows Zhu and Bridson
//! 2005 and Bridson; the fixed-point atomic scatter is a standard `GPU`
//! technique. No Unreal Engine source or derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod grid;
pub mod particle;
pub mod quantize;

pub use config::{FluidConfig, FluidError, TransferMode};
pub use cpu::{
    add_gravity, advect_rk2, enforce_solid_faces, extrapolate_axis, fluid_step, grid_to_particle,
    particle_to_grid, AxisDims, GoldenGrid, PressureConfig,
};
pub use gpu::{GpuAdvect, GpuExtrapolate, GpuFluidSolver, GpuGridOps, GpuPressureSolver};
pub use grid::{CellType, GridDims};
pub use particle::FluidParticles;
