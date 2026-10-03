//! Wave-acoustics baking and runtime perceptual parameter fields.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements design section 43 (wave acoustics and hybrid propagation): an
//! offline wave solve on a probe grid, perceptual parameter encoding, and a
//! runtime trilinear-interpolating lookup that feeds the same spatial
//! parameter bus as the geometric backend. Built on [`prism_audio_core`] and
//! aligned with [`prism_audio_spatial`].
//!
//! # Pipeline
//!
//! The crate is organised as an offline-to-runtime pipeline:
//!
//! 1. [`solver`] runs an ARD-style wave solve on a voxelised scene and samples
//!    time-domain impulse responses at probe positions.
//! 2. [`encoding`] distils each impulse response into a compact set of
//!    perceptual parameters (occlusion, decay, arrival direction, wet-dry).
//! 3. [`field`] quantises those parameters per probe into a storable
//!    parameter field; [`grid`] defines the probe lattice.
//! 4. [`bake`] orchestrates steps 1 to 3 into a finished field.
//! 5. [`lookup`] trilinearly interpolates the field at runtime (allocation
//!    free, lock free, panic free), [`openings`] blends dynamic door/window
//!    states, [`hybrid`] overlays the wave contribution with a geometric
//!    backend on the shared [`prism_audio_spatial::SpatialParams`] bus, and
//!    [`backend`] packages the whole thing as a pluggable backend.
//!
//! # Real-time contract
//!
//! Everything reachable from the runtime lookup is allocation free, lock free,
//! and panic free, so it is safe on a device callback thread. The wave solve
//! and baking allocate and must run on an offline or task thread.
//!
//! # Determinism
//!
//! All transcendental math routes through [`bevy_math::ops`] rather than `f32`
//! intrinsics, so bakes and lookups are bit-reproducible across targets.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod bake;
pub mod encoding;
pub mod field;
pub mod grid;
pub mod hybrid;
pub mod lookup;
pub mod openings;
pub mod solver;

pub use backend::{WaveBackend, WaveParameterSource};
pub use bake::{bake_field, BakeConfig, SourcePlacement};
pub use encoding::{
    encode_perceptual, DirectionalProbe, PerceptualParams, DEFAULT_DIRECT_WINDOW_MS,
};
pub use field::{BitDepth, GridTier, ParameterField, ParameterFieldBuilder, ProbeRanges};
pub use grid::{Aabb, ProbeGrid, TrilinearSample};
pub use hybrid::{blend_spatial, HybridWeights};
pub use lookup::{ParameterLookup, WaveParamSmoother};
pub use openings::MultiStateOpening;
pub use solver::{
    ImpulseResponse, Partition, SolveConfig, VoxelScene, WaveSolver, DEFAULT_SOUND_SPEED,
};
