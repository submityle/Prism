//! FLIP/APIC free-surface fluid solver on a MAC grid (milestone M5.5).
//!
//! This solver simulates incompressible liquid with marker particles advected
//! through a staggered Marker-And-Cell (MAC) velocity grid. Each step splats
//! particle velocities to the grid (P2G), applies body forces, enforces
//! incompressibility with a pressure projection (a discrete Poisson solve),
//! then interpolates the corrected velocity field back to the particles using
//! a PIC/FLIP blend (optionally the APIC affine transfer) for low numerical
//! dissipation and lively splashes.
//!
//! This module is populated by milestone M5.5.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The MAC
//! grid, PIC/FLIP/APIC transfers, and pressure projection are implemented from
//! standard, publicly documented computational-fluid-dynamics literature
//! (Bridson, "Fluid Simulation for Computer Graphics"; Zhu & Bridson 2005).

//! # Layout
//!
//! - [`config`] — solver parameters and the PIC/FLIP/APIC transfer mode.
//! - [`mac_grid`] — the staggered MAC velocity grid and cell classification.
//! - [`particle`] — the marker-particle Structure-of-Arrays store.
//! - [`transfer`] — P2G / G2P transfers and RK2 advection.
//! - [`pressure`] — the divergence-free pressure projection.
//! - [`solver`] — the single-step driver.

pub mod config;
pub mod mac_grid;
pub mod particle;
pub mod pressure;
pub mod solver;
pub mod transfer;

pub use config::{FluidConfig, TransferMode};
pub use mac_grid::{CellType, MacGrid};
pub use particle::MarkerParticles;
pub use pressure::{max_fluid_divergence, project};
pub use solver::FluidSolver;
pub use transfer::{advect, clamp_to_fluid_domain, grid_to_particle, particle_to_grid};
