//! Positional voice parameters for the spatial renderer.
//!
//! Design section 45.4 renders decoded remote voice as an ordinary spatial
//! source. This module computes the per-speaker parameters that drive that
//! rendering without re-implementing any spatialisation: see
//! [`positional_voice`] for the azimuth/elevation/distance/gain math, the
//! proximity-chat fade, the 2D team-channel versus 3D world routing, and the
//! optional side-tone gain.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 45.4. Produces inputs for the engine's external
//! spatial/HRTF renderer; consumed by [`crate::pipeline`] on the downlink.

pub mod positional_voice;

pub use positional_voice::{
    PositionalVoice, PositionalVoiceConfig, PositionalVoiceParams, VoiceSpatialMode,
};
