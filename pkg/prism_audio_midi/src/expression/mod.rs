//! Per-note expression and high-resolution control state.
//!
//! This module holds the mutable state that turns a decoded message stream into
//! a readable snapshot of expression:
//!
//! * [`controller`] - named controllers, 14-bit CC pairing, and the RPN/NRPN
//!   state machine.
//! * [`channel_state`] - per-channel pitch bend, pressure, controllers, and
//!   pitch-bend range.
//! * [`per_note`] - the fixed-capacity table of per-note pitch bend, pressure,
//!   and controllers.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the expression-state tracking of design section 52; consumes
//! [`crate::ump`] messages and feeds [`crate::mapping`].

pub mod channel_state;
pub mod controller;
pub mod per_note;

pub use channel_state::{ChannelState, DEFAULT_PITCH_BEND_RANGE, normalized_bend};
pub use controller::{
    HighResCc, ParamKind, ParameterUpdate, PerNoteController, RpnNrpnParser,
};
pub use per_note::{
    MAX_ACTIVE_NOTES, MAX_PER_NOTE_CONTROLLERS, PerNoteExpression, PerNoteKey, PerNoteState,
};

/// Re-export so `crate::expression::HighResController` names the 14-bit pair
/// tracker; `HighResCc` is kept as the module-level name.
pub use controller::HighResCc as HighResController;
