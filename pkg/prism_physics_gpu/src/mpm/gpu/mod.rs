//! Real-device `wgpu` compute pipelines for the MLS-MPM solver.
//!
//! Each submodule holds one concept:
//!
//! - [`layout`] — compute-visible bind-group layout/entry helpers (a local copy
//!   of the fluid module's helpers so the MPM pipelines do not couple to the
//!   fluid solver's internals).
//! - [`params`] — small host-side packing helpers that lay `glam` vectors and
//!   matrices out to match their WGSL `std430` uploads and read them back.
//! - [`constitutive`] — the constitutive probe pipeline
//!   [`constitutive::GpuMpmConstitutive`].
//! - [`p2g`] — the particle-to-grid affine scatter pipeline
//!   [`p2g::GpuMpmP2g`].
//! - [`grid_update`] — the node-wise finalise / gravity / boundary pipeline
//!   [`grid_update::GpuMpmGridUpdate`].
//! - [`g2p`] — the grid-to-particle affine gather pipeline
//!   [`g2p::GpuMpmG2p`].
//! - [`step`] — the fused full-step / multi-step advance orchestrator
//!   [`step::GpuMpmStep`].
//! - [`solver`] — the resident, GPU-driven multi-frame solver
//!   [`solver::GpuMpmResident`].
//!
//! # Provenance
//!
//! Plain `wgpu` pipeline construction plus the standard MLS-MPM constitutive
//! model (Stomakhin et al. 2013; Hu et al. 2018). No Unreal Engine source or
//! derived code.

pub mod constitutive;
pub mod g2p;
pub mod grid_update;
pub mod layout;
pub mod p2g;
pub mod params;
pub mod solver;
pub mod step;

pub use constitutive::{ConstitutiveOutput, GpuMpmConstitutive};
pub use g2p::{G2pParticles, GpuMpmG2p};
pub use grid_update::{BoundaryMode, GpuMpmGridUpdate};
pub use p2g::{GpuMpmP2g, P2gGrid};
pub use solver::GpuMpmResident;
pub use step::{GpuMpmStep, StepConfig, StepInputs, StepParticles};
