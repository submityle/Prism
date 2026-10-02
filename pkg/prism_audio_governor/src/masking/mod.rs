//! Psychoacoustic masking: critical bands plus the masking-aware virtualisation
//! decision of design section 33.
//!
//! Loud sources render quiet neighbours inaudible when the two overlap in the
//! same auditory region. This module models that effect with pure classical
//! DSP so the governor can retire masked, low-contribution voices to virtual
//! state before they cost a render slot:
//!
//! - [`critical_bands`] partitions the audible range into Bark/ERB-like bands,
//!   matching the ear's frequency resolution.
//! - [`masking_model`] spreads each masker's energy across neighbouring bands
//!   with an asymmetric triangular function and declares a probe masked when it
//!   sits below the combined masking profile by a margin.
//!
//! The margin is the single knob the [`crate::governor`] turns: a tight budget
//! shrinks it so more voices count as masked and virtualise, while a relaxed
//! budget widens it so only clearly-buried voices drop out. The verdicts are
//! deterministic functions of the per-band energies, so a fixed spectrum set
//! yields a fixed mask vector for golden testing.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Implements the masking-aware virtualisation half of design section 33. Its
//! verdicts feed the masking penalty in [`crate::governor::importance`] and the
//! virtual-voice promotion/demotion of design section 25.

pub mod critical_bands;
pub mod masking_model;
