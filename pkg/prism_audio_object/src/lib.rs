//! Object-based and immersive output (bed plus objects, metadata, downmix).
//!
//! This crate implements the object-audio layer of the engine: a classic
//! Atmos-style *bed plus objects* mix model, time-varying object metadata
//! streams, a hardware object-budget model with an energy-preserving
//! clustering fallback, downmix of objects into channel beds (VBAP-style
//! panning) or binaural direction parameters, and encoding of objects and beds
//! into an `AmbiX` (ACN + SN3D) Ambisonic transport. Everything here is
//! data-driven classic DSP and geometry; there is no AI/ML and no per-sample
//! audio processing in these control-rate reductions.
//!
//! # Module map
//!
//! * [`bed`] -- channel-bed layouts (2.0 / 5.1.4 / 7.1.4) and speaker geometry.
//! * [`object`] -- a single dynamic [`object::AudioObject`] and its id.
//! * [`metadata`] -- keyframed, sampleable object metadata streams.
//! * [`scene`] -- an [`scene::ObjectScene`] aggregating a bed and objects.
//! * [`budget`] -- the hardware renderable-object budget model.
//! * [`clustering`] -- energy-preserving object clustering fallback.
//! * [`pan`] -- VBAP panning onto a bed's speakers.
//! * [`fold`] -- object-to-bed downmix matrices and object-to-binaural params.
//! * [`ambisonics`] -- object/bed to `AmbiX` encoding (reuses `prism_audio_spatial`).
//! * [`render`] -- the unified scene render entry point.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 44.2 (object-based audio, Atmos-style beds plus
//! objects, MPEG-H-style metadata, object-budget clustering, downmix to beds or
//! binaural, and Ambisonics transport). Built on `prism_audio_core` and reuses
//! the `AmbiX` encoder of `prism_audio_spatial`.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod ambisonics;
pub mod bed;
pub mod budget;
pub mod clustering;
pub mod fold;
pub mod metadata;
pub mod object;
pub mod pan;
pub mod render;
pub mod scene;
