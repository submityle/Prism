//! Authoring-time modulation system (design section 12).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Aggregates the modulation building blocks required by design section 12 and
//! builds on the control-rate oscillator primitive
//! `prism_audio_core::modulation::Lfo` rather than reimplementing it. The pieces
//! compose as follows:
//!
//! - [`curve`] maps a normalized input through a response shape.
//! - [`source`] defines the common [`Modulator`] abstraction.
//! - [`envelope`] and [`follower`] are gate-driven and signal-driven envelopes.
//! - [`lfo_source`], [`random`], and [`sample_hold`] are periodic and
//!   stochastic sources.
//! - [`control_bus`] aggregates named scalar signals.
//! - [`mixing`] defines how overlapping contributions combine.
//! - [`matrix`] routes sources onto control buses with depth, polarity,
//!   shaping, and mixing.

pub mod control_bus;
pub mod curve;
pub mod envelope;
pub mod follower;
pub mod lfo_source;
pub mod matrix;
pub mod mixing;
pub mod random;
pub mod sample_hold;
pub mod source;

pub use control_bus::{BusId, ControlBus, ControlBusBank};
pub use curve::{Curve, CurvePoint};
pub use envelope::{Envelope, EnvelopeConfig, EnvelopeStage};
pub use follower::EnvelopeFollower;
pub use lfo_source::LfoModulator;
pub use matrix::{ModMatrix, ModRoute, Polarity, RouteInput};
pub use mixing::ModMix;
pub use random::{RandomMode, RandomModulator};
pub use sample_hold::SampleAndHold;
pub use source::{ModContext, Modulator};
