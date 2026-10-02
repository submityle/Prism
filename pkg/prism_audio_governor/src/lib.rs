//! Performance-adaptive governance and psychoacoustic virtualisation for
//! Prism's next-generation audio engine.
//!
//! This crate turns a fixed CPU budget into audible quality decisions without
//! ever touching the render graph's topology. It has three pillars:
//!
//! - [`governor`] -- the CPU-budget closed loop (design section 32). It smooths
//!   per-block cost telemetry, applies a hysteretic raise/lower policy at block
//!   boundaries, and maps the resulting tier onto a concrete audio level-of-
//!   detail profile (oversampling, reverb quality, spatialisation mode,
//!   modulation rate, virtualisation threshold), capped by a platform power
//!   profile. It also scores per-voice effective importance.
//! - [`masking`] -- masking-aware virtualisation (design section 33). It
//!   partitions the spectrum into critical bands and decides which voices are
//!   buried by louder neighbours and may retire to virtual state.
//! - [`clustering`] -- source clustering (design section 33). It fuses nearby,
//!   similar voices into a few representative virtual sources with a loudness-
//!   weighted centroid and an energy-conserving downmix, crossfading members as
//!   they come and go.
//!
//! Everything is deterministic classical DSP: decisions are pure functions of
//! energy, position, and timing telemetry, so a fixed input stream produces a
//! fixed trajectory that can be checked against golden data. There is no
//! machine learning anywhere in the crate.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements design sections 32 (`QualityGovernor` + audio LOD) and 33
//! (masking-aware virtualisation + source clustering). Consumes telemetry from
//! design section 21, feeds the virtual-voice management of design section 25,
//! and reuses the click-free smoothing contract of design section 7.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod clustering;
pub mod governor;
pub mod masking;
