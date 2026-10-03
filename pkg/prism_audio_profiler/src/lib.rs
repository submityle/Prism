//! Read-only profiling, telemetry aggregation, and visual-debug data model.
//!
//! This crate is the analysis and visual-debug layer of the engine. It never
//! participates in the real-time (RT) audio callback: every type here consumes
//! already-produced telemetry, audio buffers, or compiled-graph descriptions
//! and reduces them to compact, inspectable snapshots for an editor or runtime
//! profiler panel. All reductions are classic data-driven DSP and bookkeeping;
//! there is no AI/ML and no per-sample processing on the RT thread.
//!
//! # Module map
//!
//! * [`frame`] -- the per-block [`frame::AudioTelemetry`] data contract (block
//!   index, playhead, voice counts, master peak/RMS, CPU load) mirrored from
//!   the engine's RT telemetry ring, plus derived helpers.
//! * [`session`] -- [`session::ProfilerSession`], a bounded ring that records
//!   telemetry frames and an [`session::EventTimeline`] of time-stamped events
//!   for recording and deterministic playback.
//! * [`meters`] -- [`meters::MeterSnapshot`] and [`meters::MeterProbe`], which
//!   compute peak / RMS / LUFS / true-peak / correlation readings off a bus
//!   buffer by reusing the `prism_audio_core` meter primitives.
//! * [`spectrum`] -- [`spectrum::SpectrumSnapshot`] band-energy reduction built
//!   on the `prism_audio_core` FFT analyzer, with octave / third-octave bands.
//! * [`voice_monitor`] -- [`voice_monitor::VoiceMonitor`] listing active and
//!   virtual voices with a "why is this silent" [`voice_monitor::SilenceReason`]
//!   diagnosis.
//! * [`graph_inspect`] -- [`graph_inspect::GraphInspection`], a read-only export
//!   of compiled-graph nodes, connections, and reported latency.
//! * [`golden_diff`] -- [`golden_diff::GoldenDiff`], a per-sample comparison of a
//!   rendered buffer against a reference for regression localization.
//! * [`insights`] -- [`insights::InsightsReport`], an Audio-Insights-style
//!   read-only rollup combining a session window, meter snapshots, and the
//!   voice monitor into one aggregate view.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 26 (profiling, telemetry, and visual debugging).
//! The [`frame::AudioTelemetry`] fields mirror the `prism_audio_rt` telemetry
//! ring contract (section 21) by value; this crate intentionally does not
//! depend on `prism_audio_rt` so it stays free of the RT transport's `std`-only
//! lock-free queue and remains a pure, `no_std`-capable analysis layer. The
//! engine's Bevy integration layer maps the RT telemetry frame into
//! [`frame::AudioTelemetry`] at the control-rate boundary. Meter and spectrum
//! reductions reuse the `prism_audio_core` analysis primitives rather than
//! re-implementing them.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod frame;
pub mod golden_diff;
pub mod graph_inspect;
pub mod insights;
pub mod meters;
pub mod session;
pub mod spectrum;
pub mod voice_monitor;

pub use frame::AudioTelemetry;
pub use golden_diff::{GoldenDiff, GoldenDiffConfig};
pub use graph_inspect::{GraphConnection, GraphInspection, GraphNodeInfo};
pub use insights::{InsightsReport, VoiceRollup};
pub use meters::{MeterProbe, MeterSnapshot};
pub use session::{EventKind, EventTimeline, ProfilerSession, TimelineEvent};
pub use spectrum::{BandScale, SpectrumSnapshot};
pub use voice_monitor::{SilenceReason, VoiceMonitor, VoiceStatus, VoiceState};
