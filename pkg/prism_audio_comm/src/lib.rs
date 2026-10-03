//! Real-time communication audio for Prism's next-generation audio engine.
//!
//! This crate implements the real-time communication layer of design section
//! 45 entirely with classic DSP and classic networking machinery; it contains
//! no AI/ML. It is organised one concept per file:
//!
//! - [`uplink`]: the microphone capture chain (high-pass, acoustic echo
//!   cancellation, noise suppression, automatic gain control, voice activity
//!   detection), section 45.2.
//! - [`transport`]: the codec insertion point with a self-contained PCM
//!   default, the host transport abstraction, the adaptive jitter buffer, and
//!   packet-loss concealment, section 45.3.
//! - [`positional`]: positional-voice parameters for the engine's spatial
//!   renderer, section 45.4.
//! - [`pipeline`]: the end-to-end [`pipeline::VoiceCommPipeline`] assembling
//!   both directions, section 45.1.
//! - [`rng`]: a deterministic integer RNG backing comfort noise.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 45 (real-time communication: uplink pre-processing,
//! codec/transport abstractions, jitter buffer, packet-loss concealment, and
//! positional voice) on top of `prism_audio_core`.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod pipeline;
pub mod positional;
pub mod rng;
pub mod transport;
pub mod uplink;

pub use pipeline::{
    CaptureStatus, DefaultVoiceCommPipeline, PipelineConfig, PlayoutKind, PlayoutStatus,
    VoiceCommPipeline,
};
pub use positional::{
    PositionalVoice, PositionalVoiceConfig, PositionalVoiceParams, VoiceSpatialMode,
};
pub use rng::CommRng;
pub use transport::{
    CodecError, JitterBuffer, JitterConfig, JitterResult, JitterStats, LinearPcmCodec,
    LoopbackTransport, PacketLossConcealer, PlcConfig, TransportError, VoiceCodec, VoicePacket,
    VoiceTransport,
};
pub use uplink::{UplinkChain, UplinkConfig, UplinkStatus};
