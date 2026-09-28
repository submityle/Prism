//! Material Point Method (MLS-MPM) hybrid grid/particle solver (milestone M5.5).
//!
//! The Material Point Method represents a continuum as a cloud of *material
//! points* (particles carrying mass, momentum, and a deformation gradient)
//! coupled to a transient background *grid*. Each step transfers particle
//! state to the grid (P2G), updates grid momentum under gravity and boundary
//! conditions, then transfers back (G2P) while advecting particles and their
//! deformation gradients. MPM natively models sand, snow, mud, and other
//! elastoplastic continua that neither UE nor Unity support natively.
//!
//! Prism uses the MLS-MPM formulation (Hu et al. 2018) with quadratic B-spline
//! weights and the APIC affine transfer, which is compact, stable, and
//! angular-momentum conserving.
//!
//! # Layout
//!
//! - [`config`] — material, plasticity and simulation parameters.
//! - [`svd`] — robust `sqrt`-based 3x3 SVD / symmetric eigen-solve.
//! - [`weights`] — quadratic B-spline weights and gradients.
//! - [`grid`] — the transient background grid.
//! - [`particle`] — the Structure-of-Arrays material-point store.
//! - [`constitutive`] — fixed-corotated elasticity and snow plasticity.
//! - [`transfer`] — APIC P2G / G2P transfers.
//! - [`solver`] — the single-step solver and wall boundary conditions.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! MLS-MPM transfers, quadratic B-spline weights, fixed-corotated elasticity,
//! and snow plasticity are implemented from standard, publicly documented
//! computational-mechanics literature (Stomakhin et al. 2013; Hu et al. 2018).

pub mod config;
pub mod constitutive;
pub mod expf;
pub mod grid;
pub mod particle;
pub mod solver;
pub mod svd;
pub mod transfer;
pub mod weights;

pub use config::{BoundaryCondition, MpmConfig, MpmMaterial, SnowPlasticity};
pub use constitutive::{
    cofactor, corotated_pf, corotated_piola, hardening_factor, snow_return_mapping, PlasticUpdate,
};
pub use grid::Grid;
pub use particle::MaterialPoints;
pub use solver::{apply_grid_boundary, clamp_particles, MpmSolver};
pub use svd::{polar_rotation, svd3, symmetric_eigen, Svd3};
pub use transfer::{grid_to_particle, particle_to_grid};
pub use weights::QuadraticWeights;
