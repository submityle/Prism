//! Content/authoring model for Prism's next-generation audio engine.
//!
//! This crate is the **data-driven orchestration layer** that sits between the
//! game and the DSP runtime. It models the universal game-audio middleware
//! vocabulary — Events, Actions, Containers (random / sequence / blend / switch
//! / scatter), global States, per-object Switches, and RTPC game parameters —
//! as plain, serialisable, id-keyed data, and resolves a posted event into a
//! flat stream of concrete [`action::ResolvedAction`]s (which leaf sound to
//! play, at what gain, which parameter to write). It performs **no DSP**: the
//! lower runtime (`prism_audio_rt`) and voice/graph layers turn resolved
//! actions into audible voices.
//!
//! # Layout (one concept per module)
//!
//! - [`id`] — opaque `u32` identifier newtypes and the [`id::Playable`] ref.
//! - [`curve`] — breakpoint parameter-mapping curves.
//! - [`parameter`] — engine [`parameter::ParameterTarget`]s and settings.
//! - [`rng`] — deterministic xorshift64 generator for randomisation.
//! - [`container`] — the five container kinds and one-level resolution.
//! - [`state`] — global, mutually-exclusive state groups.
//! - [`switch`] — per-game-object discrete switches.
//! - [`rtpc`] — real-time parameter control definitions and bindings.
//! - [`action`] — authored [`action::Action`]s and flattened
//!   [`action::ResolvedAction`]s.
//! - [`event`] — named triggers carrying ordered actions.
//! - [`model`] — the immutable [`model::ContentModel`] registry.
//! - [`system`] — the live [`system::EventSystem`] runtime.
//!
//! # Provenance
//!
//! This crate contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original data-driven orchestration layer with no AI/ML.
//!
//! # Relationship
//!
//! See `docs/prism_audio_engine_design_zh.md` section 18 (Event / Container /
//! State / Switch / RTPC). The resolved-action stream produced here is consumed
//! by `prism_audio_rt`'s command ring and the voice/graph layers in
//! `prism_audio_core`.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod action;
pub mod container;
pub mod curve;
pub mod event;
pub mod id;
pub mod model;
pub mod parameter;
pub mod rng;
pub mod rtpc;
pub mod state;
pub mod switch;
pub mod system;

pub use action::{Action, ResolvedAction};
pub use container::{
    BlendLayer, Container, ContainerKind, ContainerPick, ContainerState, RandomMode, SequenceMode,
    SwitchBranch, WeightedChild,
};
pub use curve::{Breakpoint, Interpolation, ParameterCurve};
pub use event::Event;
pub use id::{
    BusId, ContainerId, EventId, GameObjectId, Playable, RtpcId, SoundId, StateGroupId, StateId,
    SwitchGroupId, SwitchId,
};
pub use model::ContentModel;
pub use parameter::{ParameterSetting, ParameterTarget};
pub use rng::Rng;
pub use rtpc::{RtpcBinding, RtpcDefinition, RtpcRegistry};
pub use state::{StateGroup, StateManager};
pub use switch::{SwitchGroup, SwitchManager};
pub use system::{EventSystem, MAX_CONTAINER_DEPTH};
