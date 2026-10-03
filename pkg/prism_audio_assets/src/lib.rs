//! Asset banks, streaming media, codecs, and dialogue/localization.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design sections 20 and 44.1 (banks, streaming, codec matrix) and
//! section 35 (dialogue, localization, captions).
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bank;
pub mod codec;
pub mod dialogue;
pub mod streaming;
