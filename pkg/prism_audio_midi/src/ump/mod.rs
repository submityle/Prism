//! Universal MIDI Packet (UMP) parsing: the MIDI 2.0 core of this crate.
//!
//! The submodules build up from the raw word to a decoded message stream:
//!
//! * [`word`] - the 32-bit [`word::UmpWord`] and its [`word::MessageType`]
//!   classification plus the standard word-count table.
//! * [`scaling`] - MIDI 2.0 Min-Center-Max resolution conversion
//!   ([`scaling::scale_up`] / [`scaling::scale_down`]).
//! * [`message`] - the high-level [`message::MidiMessage`] tree and the
//!   per-type decoders that lift words into it.
//! * [`decoder`] - the incremental multi-word [`decoder::UmpDecoder`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The UMP layout is
//! taken from the publicly published MIDI 2.0 specification.
//!
//! # Relationship
//! Implements the MIDI 2.0 UMP decoding of design section 52; feeds decoded
//! messages to [`crate::expression`] and [`crate::mpe`].

pub mod decoder;
pub mod message;
pub mod scaling;
pub mod word;

pub use decoder::UmpDecoder;
pub use message::{
    ChannelVoice, MidiMessage, NoteAttribute, PITCH_BEND_CENTER_32, SystemMessage, UtilityMessage,
    decode_midi1, decode_midi2, decode_system, decode_utility,
};
pub use scaling::{scale_down, scale_up};
pub use word::{MessageType, UmpWord};
