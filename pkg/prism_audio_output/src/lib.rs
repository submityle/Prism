//! Output rendering/delivery layer for Prism's next-generation audio engine.
//!
//! Scaffold crate; modules land incrementally. See
//! `docs/prism_audio_engine_design_zh.md` section 48 (output render chain:
//! downmix matrix, bass management, delivery profiles).
//!
//! # Provenance
//!
//! This crate contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is pure
//! classic DSP with no AI/ML.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
