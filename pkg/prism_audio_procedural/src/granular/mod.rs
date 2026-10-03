//! Granular synthesis: dense clouds of short grains for scatter textures.
//!
//! Many physical sounds are not single resonant events but a spray of tiny
//! ones: splashing liquid, scattering gravel, hissing sand, rustling foliage,
//! crackling fire. This module synthesises such textures granularly, from a
//! cloud of very short windowed-sinusoid [`grain`]s rather than from any sampled
//! source. A fixed-capacity [`pool`] bounds the number of simultaneous grains
//! with deterministic voice stealing, and the [`engine`] schedules new grains at
//! a controllable rate with physics- or soundscape-driven jitter on pitch, pan,
//! length, and amplitude. Every stochastic choice is drawn from the crate's
//! seeded [`crate::rng`], so an identical seed and parameter history reproduce a
//! texture bit-for-bit, and the whole render path is allocation-free.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The windowed-grain
//! cloud is a standard, publicly documented granular-synthesis technique.
//!
//! # Relationship
//! Implements design section 47.4 (granular synthesis); composes
//! [`grain::Grain`] voices in a [`pool::GrainPool`] scheduled by
//! [`engine::GranularEngine`], driven by [`crate::rng`].

pub mod engine;
pub mod grain;
pub mod pool;

pub use engine::{GrainParams, GranularEngine};
pub use grain::Grain;
pub use pool::GrainPool;
