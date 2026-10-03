//! Accessibility layer for the next-generation audio engine.
//!
//! This crate collects the data-driven accessibility accommodations that an
//! editor or runtime can toggle on behalf of a player. Every type here operates
//! at the control-rate boundary: it either annotates events with metadata that
//! the telemetry ring forwards to the UI, or it emits bus-level parameter
//! adjustments (downmix matrices, ducking boosts, compression settings) that
//! the main render graph applies through its normal smoothed-parameter path.
//! There is no AI/ML and no per-sample processing on the real-time thread here;
//! this layer only produces settings and reports.
//!
//! # Module map
//!
//! * [`caption`] -- caption and audio-description metadata carried on an event
//!   and the [`caption::CaptionReport`] forwarded through the telemetry ring.
//! * [`downmix`] -- one-touch [`downmix::MonoDownmix`] for single-sided hearing,
//!   producing a channel-fold matrix for the output stage.
//! * [`dialogue`] -- dialogue-priority [`dialogue::DialogueBoost`] that
//!   strengthens ducking of non-dialogue buses for intelligibility.
//! * [`visual_cue`] -- [`visual_cue::VisualCue`] descriptors (direction / type /
//!   intensity) emitted for key sound events so the UI can render an indicator.
//! * [`compression`] -- night / hard-of-hearing [`compression::CompressionProfile`]
//!   presets that narrow the dynamic-range window.
//! * [`profile`] -- the aggregate [`profile::AccessibilityProfile`] that composes
//!   the individual accommodations into one applied configuration.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 23 (accessibility), aligned with the `bevy_a11y`
//! model. Captions ride the event metadata contract and are surfaced through
//! the telemetry ring of design section 26; the dialogue boost reuses the
//! ducking / HDR window of design section 13; compression profiles narrow the
//! same HDR window; the downmix matrix is consumed by the output downmix stage
//! of design section 48. This crate produces settings only and reuses
//! `prism_audio_core` gain/loudness primitives rather than re-implementing them.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod caption;
pub mod compression;
pub mod dialogue;
pub mod downmix;
pub mod profile;
pub mod visual_cue;
