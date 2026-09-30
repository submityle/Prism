//! Real-device `wgpu` compute backend for the `FLIP`/`APIC` fluid transfer.
//!
//! [`GpuFluidSolver`] compiles `shaders/fluid_transfer.wgsl` and drives the
//! particle-to-grid scatter, per-face normalisation, and grid-to-particle
//! gather as a chain of compute passes. The [`layout`] submodule holds the
//! shared bind-group descriptor helpers.
//!
//! # Provenance
//!
//! Trilinear `P2G`/`G2P` with the `PIC`/`FLIP` blend (Zhu and Bridson 2005;
//! Bridson) and fixed-point atomic scatter (standard `GPU` technique). No
//! Unreal Engine source or derived code.

mod advect;
mod apic_step;
mod extrapolate;
mod grid_ops;
mod layout;
mod pressure;
mod solver;
mod step;

pub use advect::GpuAdvect;
pub use apic_step::GpuFluidApicStep;
pub use extrapolate::GpuExtrapolate;
pub use grid_ops::GpuGridOps;
pub use pressure::GpuPressureSolver;
pub use solver::GpuFluidSolver;
pub use step::GpuFluidStep;
