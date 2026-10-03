//! Mixer-state snapshot and interpolated transition system for Prism's
//! next-generation audio engine.
//!
//! A snapshot is a named set of target values for mixer, bus, and effect
//! parameters (gains, send levels, filter cutoffs, HDR window settings). The
//! runtime does not jump between snapshots; it interpolates every affected
//! parameter from its current value toward the snapshot target over a timed
//! transition shaped by an interpolation curve. Several snapshots may be active
//! at once with normalized weights, producing a blended target. Snapshots are
//! driven by the global State model so that gameplay states (combat, stealth,
//! cinematic) can recolor the whole mix without per-source edits.
//!
//! # Module map
//!
//! - [`blend`] interpolates and weight-combines parameter values in the domain
//!   implied by their [`parameter::ParameterKind`].
//! - [`config`] holds [`config::SnapshotConfig`], the default transition
//!   duration and curve used when a caller does not specify one.
//! - [`mixer`] is the stateful [`mixer::SnapshotMixer`] that owns the registry,
//!   runs the active transition, and exposes the resolved parameter map.
//! - [`parameter`] defines [`parameter::ParameterId`] and
//!   [`parameter::ParameterKind`] (linear gain, decibel, hertz, ratio).
//! - [`registry`] holds the [`registry::SnapshotRegistry`] of named snapshots.
//! - [`resolved`] holds [`resolved::ResolvedParameters`], the flattened
//!   parameter-to-value map produced by resolving a snapshot or a blend.
//! - [`snapshot`] defines [`snapshot::SnapshotId`] and [`snapshot::Snapshot`],
//!   a snapshot's identity and its parameter targets.
//! - [`stack`] resolves a weighted set of snapshots into a single blended
//!   target map (normalized weights, deterministic ordering).
//! - [`state_binding`] maps State groups and states to snapshots so a
//!   `prism_audio_content::state::StateManager` can drive activation.
//! - [`target`] defines [`target::ParameterTarget`], one parameter's target
//!   value together with its kind.
//! - [`transition`] holds the [`transition::Transition`] that interpolates from
//!   a start map to a destination map over a duration and curve.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the Snapshot item of design section 13 (mixer-state snapshots
//! with interpolated transitions) driven by the States of design section 18.
//! Parameter-kind blending reuses `prism_audio_core` decibel/linear math, and
//! transition shaping reuses `prism_audio_content::curve::Interpolation`;
//! State-driven activation reads `prism_audio_content::state::StateManager`.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod blend;
pub mod config;
pub mod mixer;
pub mod parameter;
pub mod registry;
pub mod resolved;
pub mod snapshot;
pub mod stack;
pub mod state_binding;
pub mod target;
pub mod transition;
