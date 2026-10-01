//! Permanent-damage models acting on the distance-constraint graph.
//!
//! Both models in this module key off the tensile strain of a distance edge and
//! permanently alter the constraint graph:
//!
//! * [`tearing`] removes an edge once its tensile strain exceeds a break
//!   threshold, so an over-stretched garment rips instead of stretching without
//!   bound.
//! * [`plasticity`] creeps an edge's rest length toward its current length once
//!   it is stretched past a yield strain, capturing permanent wrinkles and sag.
//!
//! They act purely on `DistanceConstraint`s and the current particle positions,
//! are `O(edges)`, and are deterministic (edges visited in list order), so they
//! compose cleanly with the substep XPBD solve and with networked lock-step
//! replay.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Both the
//! strain-threshold tear and the rest-length plastic creep are standard,
//! publicly documented position-based-dynamics techniques.

pub mod plasticity;
pub mod tearing;

mod strain;

pub use plasticity::{apply_plasticity, plastic_rest_length, PlasticParams};
pub use tearing::{apply_tearing, tear_flag, tear_flags, tear_report, TearReport, TearingParams};
