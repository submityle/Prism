//! Expressive control and MIDI 2.0 / MPE for Prism's next-generation audio
//! engine.
//!
//! This crate turns an external expressive controller (an MPE keyboard, a wind
//! or string controller, or a MIDI 2.0 sequencer) into deterministic, sample
//! offset modulation writes that the engine's modulation layer can consume.
//! Nothing here synthesises audio: the crate decodes Universal MIDI Packets
//! (UMP), tracks per-note and per-channel expression state, allocates MPE
//! member channels, and folds expression into pure modulation data. Given an
//! identical sequence of input packets it produces an identical sequence of
//! modulation writes, so the whole path is golden-testable.
//!
//! The modules follow the design document's section 52 (expressive control /
//! MIDI 2.0 / MPE):
//!
//! * [`ump`] - Universal MIDI Packet word layout, high level message decoding,
//!   and the incremental multi-word [`ump::UmpDecoder`].
//! * [`expression`] - per-note expression state, high-resolution controllers,
//!   and per-channel state tracking.
//! * [`mpe`] - MPE lower/upper zone configuration and deterministic member
//!   channel allocation.
//! * [`mapping`] - mapping of expression dimensions onto engine modulation
//!   targets and the router that emits sample offset modulation writes.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. MIDI 2.0 (UMP,
//! MIDI-CI, Property Exchange) and MPE are publicly published MIDI Association
//! specifications; only their semantics are used.
//!
//! # Relationship
//! Implements design section 52 (expressive control / MIDI 2.0 / MPE /
//! per-note expression) and builds on [`prism_audio_core`] for the sample
//! scalar and deterministic math; the modulation writes it emits are intended
//! for the engine's section 12 modulation routing and section 8 sample
//! accurate event scheduler.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod expression;
pub mod mapping;
pub mod mpe;
pub mod ump;

pub use expression::{
    ChannelState, HighResController, PerNoteController, PerNoteExpression, PerNoteKey, PerNoteState,
};
pub use mapping::{
    Curve, ExpressionDimension, ExpressionRouter, ModulationTarget, ModulationWrite, TargetMapping,
    VoiceExpression,
};
pub use mpe::{MpeAllocator, MpeZone, ZoneKind};
pub use ump::{
    ChannelVoice, MessageType, MidiMessage, NoteAttribute, SystemMessage, UmpDecoder, UmpWord,
    UtilityMessage, scale_down, scale_up,
};
