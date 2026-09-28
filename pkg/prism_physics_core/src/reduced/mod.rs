//! Reduced-order (modal subspace) soft-body simulation (milestone M7).
//!
//! A deformable body's motion is projected onto a small linear subspace
//! spanned by its low-frequency vibration modes. Simulating the reduced
//! coordinates `q` (with world displacement `u = U q`) turns a system with
//! thousands of degrees of freedom into one with a handful, enabling huge soft
//! bodies to run in real time. The modal basis is precomputed once from the
//! rest-shape mass and stiffness matrices via a generalised symmetric
//! eigenproblem, then integrated with a decoupled implicit scheme.
//!
//! The pipeline is split into single-concept files:
//!
//! * [`modes`] — a trig-free symmetric (Jacobi) eigensolver.
//! * [`config`] — [`ReducedConfig`] tunables (mode count, damping, substeps).
//! * [`subspace`] — assembling the stiffness/mass matrices and extracting the
//!   truncated modal basis as a [`ReducedModel`].
//! * [`integrate`] — the decoupled backward-Euler [`ReducedState`] stepper.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Linear
//! modal analysis, the Jacobi symmetric eigensolver, and reduced-coordinate
//! implicit integration are implemented from standard, publicly documented
//! numerical-methods and computer-graphics literature (Pentland & Williams
//! 1989; Golub & Van Loan).

pub mod config;
pub mod integrate;
pub mod modes;
pub mod subspace;

pub use config::ReducedConfig;
pub use integrate::ReducedState;
pub use modes::{SymmetricEigen, SymmetricMatrix};
pub use subspace::{ReducedMode, ReducedModel};
