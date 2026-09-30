//! `GPU` `CFL` adaptive time-step reduction.
//!
//! An explicit simulation must cap its time step by the
//! Courant-Friedrichs-Lewy (`CFL`) condition, which bounds how far the fastest
//! particle may travel in one step. This module reduces a velocity field to its
//! largest speed on the `GPU` — the only global quantity the bound needs — and
//! turns it into a clamped, stable time step.
//!
//! # Layout
//!
//! - [`config`] — the [`CflConfig`] tunable and its [`CflConfig::suggest_dt`]
//!   time-step formula.
//! - [`cpu`] — the [`cpu_max_speed`] and [`cpu_cfl_dt`] golden twins.
//! - [`gpu`] — the real-device [`GpuCflReduce`] reducer.
//! - [`layout`] — the shared bind-group layout helpers.
//!
//! # Correctness model
//!
//! The kernel is paired with the [`cpu_max_speed`] twin running the identical
//! reduction: a maximum over squared speeds (order independent and exact on the
//! `IEEE` 754 bit patterns) followed by one square root. Only that closing root
//! carries floating-point rounding, so parity is verified within a tight
//! tolerance rather than bit-for-bit.
//!
//! # Provenance
//!
//! The `CFL` condition is a classical, openly published stability criterion and
//! shared-memory tree reduction is a standard `GPU` technique. No Unreal Engine
//! source or derived code.

pub mod config;
pub mod cpu;
pub mod gpu;
pub mod layout;

pub use config::CflConfig;
pub use cpu::{cpu_cfl_dt, cpu_max_speed};
pub use gpu::GpuCflReduce;
