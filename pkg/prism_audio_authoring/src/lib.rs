//! Procedural content graph (Patch) and the full modulation system.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 11 (the Patch procedural content graph and its
//! deterministic compiler into a `prism_audio_core` runtime audio graph) and
//! design section 12 (the modulation system: control buses, envelopes, curves,
//! sources, and the modulation matrix). The runtime graph primitives,
//! parameter smoothing, and low-frequency oscillator all come from
//! `prism_audio_core`; this crate is the authoring layer on top of them.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod rng;

pub mod modulation;
pub mod patch;

pub use rng::Rng;
