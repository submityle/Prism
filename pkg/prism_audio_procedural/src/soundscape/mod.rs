//! Procedural environmental soundscape (section 37).
//!
//! Where the rest of the crate turns physics contacts into sound, this module
//! turns high-level environmental context into an ambience bed. A
//! [`SoundscapeState`] (biome, weather, time of day, indoor or outdoor) selects
//! a [`SoundscapePalette`] of [`SoundscapeElement`]s, and the
//! [`SoundscapeScheduler`] scatters those elements as discrete, seeded one-shots
//! around the listener - never a looping bed. A [`SpatialFilter`] vetoes or
//! attenuates placements for geometry, occlusion, and room ownership, and a
//! governable concurrency cap keeps the voice count inside budget. Because
//! every random choice is drawn from the crate's seeded generator, an identical
//! seed and state sequence reproduce the whole ambience bit-for-bit.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 37 (procedural soundscape); state, palette, and
//! element data feed the [`scheduler`], which draws from [`crate::rng`] and
//! emits one-shots the host spawns (optionally via [`crate::granular`]).

pub mod element;
pub mod palette;
pub mod scheduler;
pub mod state;

pub use element::SoundscapeElement;
pub use palette::SoundscapePalette;
pub use scheduler::{NullSpatialFilter, ScatteredOneShot, SoundscapeScheduler, SpatialFilter};
pub use state::SoundscapeState;
