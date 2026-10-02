//! Content/authoring model for Prism's next-generation audio engine.
//!
//! Scaffold crate; modules land incrementally. See
//! `docs/prism_audio_engine_design_zh.md` section 18 (Event / Container /
//! State / Switch / RTPC).
//!
//! # Provenance
//!
//! This crate contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original data-driven orchestration layer with no AI/ML.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
