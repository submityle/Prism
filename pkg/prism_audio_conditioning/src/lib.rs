//! Offline asset conditioning and authoring pipeline for Prism's audio engine.
//!
//! The runtime chapters describe how a sound plays; this crate is the offline
//! dual that prepares assets *before* they enter the engine. Every stage is a
//! pure function from input bytes/samples plus parameters to a read-only,
//! content-addressed artifact, runs off the audio thread, and is deterministic
//! and golden-reproducible. Nothing here is a real-time facility.
//!
//! # Module map
//!
//! - [`bank`] authors bank layouts: dependency-graph de-duplication, streaming
//!   vs resident partitioning, and prefetch hints.
//! - [`codec_tier`] classifies an asset's content (bandwidth, dynamic range,
//!   channel correlation) into a recommended codec tier and records encoder
//!   pre-roll/padding.
//! - [`config`] holds the per-stage configuration records.
//! - [`content_hash`] computes a deterministic content hash and keys the
//!   conditioning cache.
//! - [`decode`] orchestrates the import/decode matrix over
//!   [`prism_audio_assets::codec`] into unified f32 PCM.
//! - [`hrtf_condition`] conditions a SOFA HRIR dataset: resample to the target
//!   rate, diffuse-field equalization, ITD extraction, and minimum-phasing.
//! - [`lipsync`] exports an offline viseme/energy-envelope timeline.
//! - [`loop_point`] detects seamless forward/ping-pong loop points via
//!   zero-crossing alignment and autocorrelation.
//! - [`loudness_offline`] computes BS.1770 integrated LUFS, true-peak, and
//!   loudness range.
//! - [`marker`] assembles a labeled transient/beat/bar marker timeline.
//! - [`pcm`] holds the [`pcm::ConditionedPcm`] canonical sample container.
//! - [`pipeline`] orchestrates the stages into a single conditioning job.
//! - [`resample_offline`] performs high-quality offline polyphase-sinc
//!   resampling to the project rate.
//! - [`tempo`] estimates tempo/beat grid by autocorrelating the onset
//!   envelope.
//! - [`transient`] detects transients/onsets via spectral flux.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. BS.1770
//! loudness, spectral-flux onset detection, autocorrelation tempo estimation,
//! polyphase-sinc resampling, and AES69/SOFA are public standards and classic
//! DSP/MIR techniques; only the ideas are borrowed.
//!
//! # Relationship
//! Implements design section 51 (asset conditioning and authoring pipeline),
//! the offline dual of sections 10, 13, 16, 19, 20, 34, 35, and 44.1. It reuses
//! `prism_audio_assets::codec` for decoding, `prism_audio_resample` for
//! high-quality resampling, `prism_audio_hrtf` for HRIR types, and the analysis
//! nodes of `prism_audio_core` for loudness and spectral analysis.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bank;
pub mod codec_tier;
pub mod config;
pub mod content_hash;
pub mod decode;
pub mod hrtf_condition;
pub mod lipsync;
pub mod loop_point;
pub mod loudness_offline;
pub mod marker;
pub mod pcm;
pub mod pipeline;
pub mod resample_offline;
pub mod tempo;
pub mod transient;
