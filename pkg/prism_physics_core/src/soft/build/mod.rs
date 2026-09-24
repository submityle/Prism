//! Authoring builders for the unified soft-body kernel.
//!
//! The three families of deformable object Prism ships are all just a
//! [`crate::soft::body::SoftBody`] populated with different particles and
//! constraints. These builders wire up that topology for the common cases:
//!
//! - [`cloth`] &mdash; a rectangular sheet ([`ClothGrid`] / [`Cloth`]) with
//!   structural, shear, and bending constraints.
//! - [`rope`] &mdash; a one-dimensional chain ([`RopeGrid`] / [`Rope`]) with
//!   link and optional bending constraints.
//! - [`softbody`] &mdash; a solid tetrahedral block ([`SoftBoxGrid`] /
//!   [`SoftBox`]) with edge and volume constraints.
//!
//! Each builder samples its rest state from the authored geometry, so callers
//! only choose dimensions, spacing, mass, and compliance.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The cloth,
//! rope, and tetrahedral-lattice topologies are standard, publicly documented
//! simulation constructions.

pub mod cloth;
pub mod rope;
pub mod softbody;

pub use cloth::{Cloth, ClothGrid};
pub use rope::{Rope, RopeGrid};
pub use softbody::{SoftBox, SoftBoxGrid};
