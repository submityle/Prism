//! Cross-modal haptic and motion output for the next-generation audio engine.
//!
//! Haptics in this engine are *audio-sourced*: a haptic signal is derived from
//! the same audio material that drives the loudspeaker/headphone output, so the
//! felt vibration is sample-synchronous with the heard sound by construction.
//! Sources send into a parallel [`bus::HapticBus`] exactly as they send into an
//! aux reverb bus; the bus runs a band-split / rectify / envelope-follow /
//! downsample [`transcode`] chain and hands a low-rate [`waveform::HapticWaveform`]
//! to a pluggable [`backend::HapticBackend`]. All processing is classic
//! data-driven DSP and bookkeeping; there is no AI/ML and the haptic path is a
//! pure side-chain that never feeds audio back into the render graph.
//!
//! # Module map
//!
//! * [`waveform`] -- the low-sample-rate [`waveform::HapticWaveform`] buffer and
//!   its mono/stereo-actuator channel model.
//! * [`transcode`] -- the audio-to-haptic [`transcode::HapticTranscoder`]:
//!   crossover band split, rectification, envelope following, and decimation to
//!   the haptic rate.
//! * [`bus`] -- the [`bus::HapticBus`] that mixes per-source sends and drives the
//!   transcoder, mirroring the aux-send contract of the main graph.
//! * [`backend`] -- the [`backend::HapticBackend`] trait plus the wide-band,
//!   dual-motor rumble, and silent-fallback implementations.
//! * [`spatial`] -- directional [`spatial::SpatialWeighting`] that maps source
//!   direction/distance to left/right actuator intensity.
//! * [`latency`] -- [`latency::LatencyAligner`], playhead- and PDC-aware
//!   compensation of per-backend intrinsic latency.
//! * [`governor`] -- [`governor::HapticGovernor`], CPU-budget-driven quality
//!   scaling and low-tier bypass.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 36 (haptics and cross-modal output). The haptic
//! bus mirrors the aux-send model of design section 17 and shares the playhead
//! of design section 8 and the plugin-delay-compensation contract of design
//! section 29; spatial weighting consumes the distance/direction shaping of
//! design section 15; the governor obeys the budget authority of design
//! section 32. Transcoding reuses the `prism_audio_core` DSP primitives rather
//! than re-implementing filters and detectors.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod backend;
pub mod bus;
pub mod governor;
pub mod latency;
pub mod spatial;
pub mod transcode;
pub mod waveform;
