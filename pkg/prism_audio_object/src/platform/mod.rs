//! Platform spatial backend bridge: capability, channel order, naming presets,
//! delivery negotiation, and the backend extension-point trait.
//!
//! This module maps the engine's control-rate object scene onto whatever a
//! platform spatial renderer (Windows Spatial Sound, Apple `CoreAudio`, Sony
//! Tempest 3D, a Dolby Atmos object renderer, headphone binaural, or a fixed
//! loudspeaker bed) can actually present. It describes each target's abilities
//! ([`capability`]), the channel orderings a bed delivery must follow
//! ([`channel_order`]), ready-made named presets ([`profile`]), the negotiation
//! that chooses an output format and gracefully degrades a scene that exceeds a
//! platform's object ceiling ([`delivery`]), and the pluggable backend trait a
//! platform/device integration implements ([`spatial_backend`]).
//!
//! Everything here is a pure control-rate reduction: no input/output and no
//! per-sample audio. The actual operating-system handoff lives in the
//! platform/device layer, which consumes the [`delivery::DeliveryPlan`] this
//! module produces.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 16 (platform spatial backend bridge) and the
//! platform-delivery half of section 44.2 (bed-plus-objects with hardware
//! object budgets). Built on [`crate::bed`], [`crate::budget`],
//! [`crate::render`], and [`crate::scene`].

pub mod capability;
pub mod channel_order;
pub mod delivery;
pub mod profile;
pub mod spatial_backend;

pub use capability::{ObjectRenderMode, PlatformCapability};
pub use channel_order::ChannelOrder;
pub use delivery::{negotiate_delivery, negotiate_scene, DeliveryPlan};
pub use profile::PlatformProfile;
pub use spatial_backend::{PlatformSpatialBackend, ProfileBackend};
