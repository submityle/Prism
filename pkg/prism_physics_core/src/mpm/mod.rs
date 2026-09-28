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
//! This module is populated by milestone M5.5.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! MLS-MPM transfers, quadratic B-spline weights, fixed-corotated elasticity,
//! and snow plasticity are implemented from standard, publicly documented
//! computational-mechanics literature (Stomakhin et al. 2013; Hu et al. 2018).
