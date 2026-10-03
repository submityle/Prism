//! Six-degree-of-freedom immersive audio interchange and loudspeaker-array
//! rendering for Prism's next-generation audio engine.
//!
//! This crate adds the interoperability layer and array-output paths described
//! in design section 49: an MPEG-I-style Encoder Input Format (EIF) declarative
//! scene model that maps onto the engine's shared acoustic truth, ADM/BW64/
//! S-ADM production interchange for object/bed/scene audio, and a family of
//! loudspeaker-array panners (VBAP, `AllRAD`, WFS, beamforming) that render
//! objects and HOA scene audio to arbitrary physical speaker layouts. All
//! rendering is deterministic classic DSP: amplitude panning, matrix decoding,
//! and geometric delay/gain driving functions, with no AI/ML.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The crate implements only the public standard semantics and published
//! algorithms of MPEG-I Immersive Audio (ISO/IEC 23090-4), MPEG-H 3D Audio
//! (ISO/IEC 23008-3), ADM (ITU-R BS.2076), BW64 (ITU-R BS.2088), Serial ADM
//! (ITU-R BS.2125), VBAP (Pulkki 1997), `AllRAD` (Zotter and Frank 2012), and
//! WFS (Berkhout 1988).
//!
//! # Relationship
//!
//! Builds on `prism_audio_core` (buffers, parameters, deterministic math),
//! `prism_audio_spatial` (HOA encode/decode/rotation, panner geometry,
//! spatial parameters), and `prism_audio_object` (bed-plus-objects model,
//! metadata streams, VBAP downmix). This crate is the interoperability and
//! array-output superset: it imports/exports standard scenes and renders the
//! same source-position truth to physical arrays, rather than maintaining a
//! second acoustic truth.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod adm;
pub mod array;
pub mod eif;
