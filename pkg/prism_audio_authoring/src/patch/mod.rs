//! Procedural content graph (Patch) authoring and compilation (section 11).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 11: a Patch is a small DSP graph (a data asset)
//! that is validated and compiled, via the same Kahn-topological compiler as
//! the runtime, into a `prism_audio_core::graph::AudioGraph` and then into a
//! single `AudioNode`. The pieces compose as follows:
//!
//! - [`description`] is the editable, serializable Patch data asset.
//! - [`builder`] assembles a [`description::PatchDescription`] incrementally.
//! - [`node_kind`] enumerates the primitive node library and nested patches.
//! - [`library`] holds the primitive `AudioNode` implementations.
//! - [`param`] and [`trigger`] carry lock-free host inputs into a running patch.
//! - [`port`] describes primitive port shapes.
//! - [`validate`] checks a description before compilation.
//! - [`compiler`] flattens nested patches and builds the runtime graph.
//! - [`compiled`] is the runnable product and its [`compiled::PatchNode`]
//!   wrapper.

pub mod builder;
pub mod compiled;
pub mod compiler;
pub mod description;
pub mod error;
pub mod library;
pub mod node_kind;
pub mod param;
pub mod port;
pub mod trigger;
pub mod validate;

pub use builder::PatchBuilder;
pub use compiled::{CompiledPatch, PatchNode};
pub use compiler::compile;
pub use description::{
    ExposedParam, ExposedTrigger, PatchConnection, PatchDescription, PatchOutput,
};
pub use error::PatchError;
pub use library::oscillator::OscWaveform;
pub use node_kind::NodeKind;
pub use param::{ParamCell, ParamHandle};
pub use port::{PortCount, SignalKind};
pub use trigger::{GateCell, TriggerHandle};
pub use validate::{validate, MAX_NESTING_DEPTH};

// `PatchNode` from `description` is the data-asset node; it is intentionally not
// re-exported here to avoid colliding with the runtime `compiled::PatchNode`.
pub use description::PatchNode as PatchNodeDescription;
