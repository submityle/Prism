//! Declarative auto-mixing and category loudness governance (design section
//! 46.8).
//!
//! Game and interactive mixes are kept legible with a small set of declarative
//! rules rather than hand-wired side-chains: "when dialogue is active, duck
//! music by 12 dB", "when UI chimes play, duck ambience by 6 dB". This module
//! turns that intent into a deterministic, block-rate runtime:
//!
//! - [`category`] registers the coarse content classes (dialogue, UI, weapons,
//!   ambience, music, ...) addressed by every rule, each with a loudness
//!   priority.
//! - [`rule`] expresses one declarative duck as pure data: a trigger category,
//!   a target category, an attenuation in decibels, attack/release ballistics,
//!   and a response curve.
//! - [`ruleset`] aggregates categories plus rules into a single authoring
//!   artifact and validates it (no unknown categories, no self-ducks).
//! - [`compiler`] lowers a validated ruleset onto the design section 12
//!   modulation matrix: one reduction bus per category and, per rule, one
//!   activity input bus and one `Max`-folded unipolar route.
//! - [`activity`] provides the per-rule attack/release envelope that shapes a
//!   trigger's raw activity before it drives the matrix.
//! - [`runtime`] owns the compiled, tickable [`CompiledAutoMix`]: the host
//!   reports per-category activity each block, envelopes apply ballistics, the
//!   matrix resolves the strongest reduction per category, and the result reads
//!   back as a per-category gain multiplier.
//!
//! All smoothing is block-rate and determined only by the inputs, so the output
//! is golden-reproducible (design section 24).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! CRIWARE, or Google Resonance Audio source or derived code; no AI/ML.
//! Declarative category ducking and content-class loudness governance are
//! classical mixing idioms; this is an independent implementation expressed as
//! data compiled onto a generic modulation matrix.
//!
//! # Relationship
//! Implements design section 46.8 (declarative auto-mixing and category
//! loudness governance). The compile target is the design section 12 modulation
//! system in [`crate::modulation`]; the per-category gain multiplier this module
//! produces is applied by the host to the corresponding bus gains in the design
//! section 5 runtime graph.

pub mod activity;
pub mod category;
pub mod compiler;
pub mod rule;
pub mod ruleset;
pub mod runtime;

pub use activity::ActivityEnvelope;
pub use category::{Category, CategoryId, CategorySet};
pub use rule::{
    decibels_to_linear, DuckingRule, DEFAULT_ATTACK_SECONDS, DEFAULT_RELEASE_SECONDS,
};
pub use ruleset::{AutoMixError, AutoMixRuleset};
pub use runtime::CompiledAutoMix;
