//! Source clustering: fuse spatially near, timbrally similar voices into a few
//! representative virtual sources (design section 33).
//!
//! When a scene has far more voices than the render budget allows, many of them
//! are quiet, close together, and spectrally alike. Rendering each one wastes a
//! slot the listener cannot resolve. This module collapses such groups into a
//! handful of representative sources that preserve perceived position and
//! loudness, and smoothly fades members in and out so the collapse is
//! inaudible:
//!
//! - [`timbre`] -- the compact spectral fingerprint used to tell voices apart.
//! - [`cluster`] -- the representative-source model with a loudness-weighted
//!   centroid and an energy-conserving downmix.
//! - [`assignment`] -- the deterministic leader-clustering policy and the
//!   object-bed capacity merge.
//! - [`fade`] -- per-voice membership crossfades.
//!
//! All stages are deterministic functions of the per-voice energy, position,
//! and timbre, so a fixed scene yields a fixed clustering for golden testing.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements the source-clustering half of design section 33. Timbres derive
//! from the critical-band spectra of [`crate::masking`]; clustering feeds the
//! virtual-voice management of design section 25 and the object-bed budget of
//! the spatialiser.

pub mod assignment;
pub mod cluster;
pub mod fade;
pub mod timbre;
