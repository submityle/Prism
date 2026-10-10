//! Mapping expression onto engine modulation targets.
//!
//! This module bridges the expressive controller side and the engine side:
//!
//! * [`target`] - names the modulation destinations ([`ModulationTarget`]) and
//!   source dimensions ([`ExpressionDimension`]), and the [`TargetMapping`] that
//!   connects one to the other through a depth, offset, and [`Curve`].
//! * [`router`] - the [`ExpressionRouter`] that reads a [`VoiceExpression`]
//!   snapshot and emits sample-offset [`ModulationWrite`]s, folding channel and
//!   per-note pitch bend into a single semitone offset.
//!
//! Everything here is pure data: no module in this crate synthesises audio, so
//! the mapping layer only describes what the engine's modulation routing should
//! write and when.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the mapping and routing of design section 52 (expression to
//! modulation). Consumes [`crate::expression`] state and [`crate::mpe`] voice
//! attribution and produces writes for the engine's section 12 modulation
//! routing.

pub mod router;
pub mod target;

pub use router::{ExpressionRouter, ModulationWrite, VoiceExpression, MAX_VOICE_CONTROLLERS};
pub use target::{
    normalize_u16, normalize_u32, Curve, ExpressionDimension, ModulationTarget, TargetMapping,
};
