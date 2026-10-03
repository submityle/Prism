//! Contact event bus: physics-truth collision facts turned into audio drive.
//!
//! The physics solver already computes the impulses, contact points, and
//! relative velocities needed to synthesise sound; this module is the typed,
//! deterministic conduit that carries those facts to the synthesis stages
//! without a second "acoustic collision" system. It is split into the event
//! value types ([`event`]), the pure physics-to-drive curves ([`energy_map`]),
//! the deterministic de-duplication/merge pass ([`merge`]), and the bounded
//! block collector ([`bus`]).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 47.1 (contact event bus) and the energy mapping
//! that drives sections 47.2-47.3.

pub mod bus;
pub mod energy_map;
pub mod event;
pub mod merge;

pub use bus::ContactEventBus;
pub use event::{
    ContactEvent, ContactId, ContactPoint, ImpactEvent, SeparationEvent, SustainEvent,
};
pub use merge::{cluster_to_group, merge_impacts, MergeConfig};
