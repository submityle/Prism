//! `GPU` compute pipelines for fragment classification.
//!
//! - [`layout`] — the shared bind-group layout helpers.
//! - [`assign`] — the [`GpuVoronoiAssign`] nearest-site classifier.
//!
//! # Provenance
//!
//! No Unreal Engine source or derived code.

pub mod assign;
pub mod layout;

pub use assign::GpuVoronoiAssign;
