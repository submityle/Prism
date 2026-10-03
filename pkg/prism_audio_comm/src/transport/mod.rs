//! Codec, network transport, and packet-loss resilience for the downlink.
//!
//! This module groups the classic, non-DSP machinery of design section 45.3
//! into one concept-per-file layout:
//!
//! - [`packet`]: the on-the-wire [`VoicePacket`] and the host-provided
//!   [`VoiceTransport`] trait, plus an in-memory [`LoopbackTransport`].
//! - [`codec`]: the [`VoiceCodec`] insertion point and a self-contained
//!   [`LinearPcmCodec`] default so the crate works with no external codec.
//! - [`jitter_buffer`]: an adaptive, reordering [`JitterBuffer`] that converts
//!   bursty network arrivals into a steady playout stream.
//! - [`plc`]: the pitch-synchronous [`PacketLossConcealer`] that fills frames
//!   the jitter buffer reports as lost.
//!
//! None of this ever runs on the audio callback: the real-time thread only
//! pulls already-decoded PCM. The pieces are assembled into the downlink by
//! [`crate::pipeline`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 45.3. Built on `prism_audio_core`; feeds the
//! downlink path of [`crate::pipeline`] and interoperates with the uplink
//! encoder in [`crate::uplink`].

pub mod codec;
pub mod jitter_buffer;
pub mod packet;
pub mod plc;

pub use codec::{CodecError, LinearPcmCodec, VoiceCodec};
pub use jitter_buffer::{JitterBuffer, JitterConfig, JitterResult, JitterStats};
pub use packet::{LoopbackTransport, TransportError, VoicePacket, VoiceTransport};
pub use plc::{PacketLossConcealer, PlcConfig};
