//! Physics-coupled procedural audio and procedural soundscape.
//!
//! This crate turns physics-engine contact facts and high-level ambience state
//! into deterministic audio using only classical DSP driven by a seeded RNG.
//! There is no sample playback, no machine learning, and no third-party audio
//! middleware: struck bodies are rendered by modal synthesis, sustained contact
//! by filtered-noise friction and modal-excited rolling, scatter textures by
//! granular synthesis, and environmental ambience by a seeded one-shot
//! scheduler. Given the same seed and the same input sequence, every stage
//! produces bit-identical output, so the whole engine is golden-testable and
//! network-consistent.
//!
//! The modules follow the design document's section 47 (physics-coupled audio)
//! and section 37 (procedural soundscape):
//!
//! * [`contact`] - the contact event bus and physics-to-excitation curves
//!   (section 47.1).
//! * [`modal`] - modal synthesis of resonant bodies (section 47.2).
//! * [`continuous`] - friction, rolling, and sliding of sustained contact
//!   (section 47.3).
//! * [`granular`] - granular synthesis for splatter and scatter textures
//!   (section 47.4).
//! * [`material`] - acoustic material coupling (section 47.5).
//! * [`soundscape`] - procedural environmental ambience (section 37).
//! * [`dsp`] and [`rng`] - the shared DSP primitives and the deterministic
//!   seeded generator every stochastic stage draws from.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 47 (contact/modal/granular physics-coupled audio)
//! and section 37 (procedural soundscape/ambience); builds on
//! [`prism_audio_core`] for the sample type, parameter smoothing, and
//! deterministic math.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod contact;
pub mod continuous;
pub mod dsp;
pub mod granular;
pub mod material;
pub mod modal;
pub mod rng;
pub mod soundscape;

pub use contact::{
    ContactEvent, ContactEventBus, ContactId, ContactPoint, ImpactEvent, MergeConfig,
    SeparationEvent, SustainEvent,
};
pub use continuous::{ContactPhase, ContactVoice};
pub use granular::{Grain, GranularEngine, GrainParams, GrainPool};
pub use material::{
    MaterialCategory, MaterialId, MaterialLibrary, MaterialPairId, MaterialPairProfile,
};
pub use modal::{ModalBank, ModalSynth, Mode};
pub use rng::ProceduralRng;
pub use soundscape::{
    ScatteredOneShot, SoundscapeElement, SoundscapePalette, SoundscapeScheduler, SoundscapeState,
    SpatialFilter,
};
