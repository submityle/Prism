//! MIDI Polyphonic Expression (MPE) zones and member-channel allocation.
//!
//! MPE lets a single controller play notes that each bend, press, and change
//! timbre independently by spreading them across a block of member channels
//! governed by one manager channel. This module describes that layout and
//! assigns notes to channels deterministically:
//!
//! * [`zone`] - the lower / upper zone configuration ([`MpeZone`], [`ZoneKind`])
//!   that maps member ordinals onto channels.
//! * [`allocator`] - the round-robin [`MpeAllocator`] that hands a free (or
//!   least-loaded) member channel to each new note and recycles it on release.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The MPE channel
//! layout and allocation convention follow the publicly published MPE
//! specification.
//!
//! # Relationship
//! Implements the MPE part of design section 52. The channel an allocator
//! returns identifies which [`crate::expression`] state a note's messages
//! update and which voice [`crate::mapping`] attributes its writes to.

pub mod allocator;
pub mod zone;

pub use allocator::MpeAllocator;
pub use zone::{CHANNELS_PER_PORT, MpeZone, ZoneKind};
