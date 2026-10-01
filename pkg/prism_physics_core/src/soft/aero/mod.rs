//! Wind coupling and per-triangle aerodynamics for cloth and soft bodies.
//!
//! A garment only reads as *cloth* when the surrounding air pushes on it: a
//! steady breeze billows a skirt, a gust snaps a flag taut, and the drag of the
//! air lets a cape settle instead of swinging like a rigid sheet. This module
//! models that coupling the way production cloth engines do: wind acts on the
//! mesh *per triangle*, decomposed into a drag component along the face normal
//! and a lift component in the face plane, scaled by area (and optionally by a
//! quadratic dynamic-pressure term). The resulting force is spread across the
//! face's three vertices as an external velocity increment before the XPBD
//! substep prediction runs.
//!
//! The module is split into cohesive pieces:
//!
//! * [`field`] — the [`WindField`] and [`AeroParams`] inputs, each with a
//!   finite, range-clamped [`sanitized`](WindField::sanitized) copy.
//! * [`triangle`] — the pure per-face force [`triangle_aero_force`] and the
//!   deterministic index-hashed [`turbulence_offset`].
//! * [`apply`] — the mesh-wide velocity pre-pass [`apply_aero_forces`] and the
//!   [`ParticleColumnsMut`](crate::soft::particle::storage::ParticleColumnsMut)
//!   convenience entry point [`apply_aero_to_columns`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! per-triangle drag/lift decomposition, the optional quadratic
//! dynamic-pressure term, and the integer-hash turbulence are standard,
//! publicly documented cloth-aerodynamics techniques.

pub mod apply;
pub mod field;
pub mod triangle;

mod sanitize;

pub use apply::{apply_aero_forces, apply_aero_to_columns};
pub use field::{AeroParams, WindField};
pub use triangle::{triangle_aero_force, turbulence_offset};
