//! Interactive music system for Prism's next-generation audio engine.
//!
//! This crate is the **control-rate, data-driven music planner** that sits
//! between gameplay and the DSP runtime. It models the universal interactive
//! music vocabulary -- segments with entry/exit cues and pre/post-roll
//! margins, playlists that order/loop/randomise segments, beat/bar quantized
//! transitions, overlaid stingers, vertical layering driven by an intensity
//! value, horizontal re-sequencing by gameplay branch, and a Godot-style clip
//! graph of nodes (clips) and edges (conditional transitions) -- as plain,
//! serialisable, id-keyed data, and resolves a music request into a flat
//! stream of concrete [`action::MusicAction`]s (which segment/clip/layer to
//! play or stop, at what sample, at what gain). It performs **no DSP**: the
//! lower runtime and voice/graph layers in `prism_audio_core` turn resolved
//! actions into audible voices.
//!
//! # Layout (one concept per module)
//!
//! - [`id`] -- opaque `u32` identifier newtypes.
//! - [`rng`] -- deterministic xorshift64 generator for playlist randomisation.
//! - [`segment`] -- music [`segment::Segment`]s with cues/markers/margins.
//! - [`playlist`] -- [`playlist::Playlist`] ordering of segments.
//! - [`transition`] -- [`transition::TransitionType`] quantization and fades.
//! - [`stinger`] -- quantization-aligned overlay phrases.
//! - [`layer`] -- vertical [`layer::Layer`]ing by intensity with hysteresis.
//! - [`clip_graph`] -- the interactive [`clip_graph::ClipGraph`] flow.
//! - [`action`] -- resolved [`action::MusicAction`]s with sample timestamps.
//! - [`model`] -- the immutable [`model::MusicModel`] registry.
//! - [`system`] -- the live deterministic [`system::MusicSystem`] planner.
//!
//! # Provenance
//!
//! This crate contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. It is an
//! original, data-only interactive music planner built from classic tempo/beat
//! arithmetic and plain graph traversal, with no AI/ML.
//!
//! # Relationship
//!
//! See `docs/prism_audio_engine_design_zh.md` section 19 (interactive music
//! system). Quantization reuses `prism_audio_core`'s
//! [`prism_audio_core::scheduler::NamedClock`] and
//! [`prism_audio_core::scheduler::Grid`]; the resolved-action stream produced
//! here is consumed by the lower runtime and the voice/graph layers.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod action;
pub mod clip_graph;
pub mod id;
pub mod layer;
pub mod model;
pub mod playlist;
pub mod rng;
pub mod segment;
pub mod stinger;
pub mod system;
pub mod transition;

pub use action::MusicAction;
pub use clip_graph::{
    Clip, ClipEdge, ClipGraph, TransitionContext, TriggerCondition, MAX_GRAPH_DEPTH,
};
pub use id::{
    BranchId, ClipId, GraphId, LayerId, LayerSetId, MarkerId, PlaylistId, SegmentId, SoundId,
    StingerId,
};
pub use layer::{Layer, LayerSet};
pub use model::MusicModel;
pub use playlist::{Playlist, PlaylistCursor, PlaylistItem, PlaylistMode};
pub use rng::Rng;
pub use segment::{Marker, Segment};
pub use stinger::Stinger;
pub use system::MusicSystem;
pub use transition::{Fade, FadeCurve, Transition, TransitionType};
