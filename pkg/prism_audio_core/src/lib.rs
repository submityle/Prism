//! Backend-neutral real-time DSP kernel for Prism's next-generation audio
//! engine.
//!
//! This crate is the audio counterpart of [`prism_physics_core`]: a
//! device-agnostic, allocation-free-in-the-hot-path signal processing core that
//! higher layers (ECS front-end, spatializer, authoring/event runtime, device
//! backends) build upon.
//!
//! # Design overview
//!
//! - [`math`] holds the sample scalar, decibel/linear conversions, denormal
//!   flushing, and equal-power helpers.
//! - [`buffer`] holds the planar [`buffer::AudioBuffer`] block storage and the
//!   [`buffer::ChannelLayout`] descriptor.
//! - [`param`] holds sample-accurate parameter smoothing ([`param::Smoothed`])
//!   and the [`param::Ramp`] shapes used to avoid zipper noise.
//! - [`time`] holds the sample-accurate [`time::Transport`] and musical
//!   [`time::TimeSignature`] used by the scheduler.
//! - [`graph`] holds the unified render graph: the [`graph::AudioNode`] trait,
//!   the [`graph::AudioGraph`] container, deterministic topological
//!   compilation, and the allocation-free [`graph::AudioGraph::process`] block
//!   renderer.
//! - [`nodes`] holds concrete, fully implemented [`graph::AudioNode`]s (gain,
//!   biquad filters, ...). None of them are stubs.
//!
//! # Real-time contract
//!
//! Everything reachable from [`graph::AudioGraph::process`] is **allocation
//! free, lock free, and panic free** so it can run on a device callback thread.
//! Graph mutation ([`graph::AudioGraph::connect`], node insertion) and
//! compilation ([`graph::AudioGraph::compile`]) may allocate and must run off
//! the audio thread.
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, or FMOD source or derived code**. All DSP (biquad
//! coefficients, decibel math, equal-power panning, one-pole smoothing) is
//! implemented from standard, publicly documented signal-processing knowledge
//! (e.g. the RBJ Audio EQ Cookbook formulas).
//!
//! [`prism_physics_core`]: https://docs.rs/prism_physics_core
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod buffer;
pub mod graph;
pub mod math;
pub mod nodes;
pub mod param;
pub mod time;

pub use buffer::{AudioBuffer, ChannelLayout};
pub use graph::{AudioGraph, AudioNode, NodeId, PortRef, ProcessIo, RenderContext};
pub use math::{Sample, db_to_linear, linear_to_db};
pub use param::{Ramp, Smoothed};
pub use time::{TimeSignature, Transport};
